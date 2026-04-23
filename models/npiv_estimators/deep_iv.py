"""DeepIV (Hartford et al. 2017) — treatment-type-agnostic implementation.

Supports all four treatment types:

    scalar-continuous   a_dim=1,  treatment_family="continuous"
    scalar-binary       a_dim=1,  treatment_family="binary"
    vector-continuous   a_dim>1,  treatment_family="continuous"
    vector-binary       a_dim>1,  treatment_family="binary"

Stage 1 fits a conditional treatment density ``p(A | Z, W, L)``:

    continuous → diagonal-Gaussian mixture density network  (any a_dim)
    binary     → independent per-dimension Bernoulli         (any a_dim)

Stage 2 fits the structural response ``h(A, W, L)`` by minimising

    E_{Z,W,L} [ (Y − E_{A~p(A|Z,W,L)}[h(A, W, L)])² ]

via:

* **exact enumeration** over ``{0,1}^{a_dim}`` when ``treatment_family="binary"``
  and ``2^a_dim ≤ 256``  (zero Monte-Carlo variance);
* **Monte-Carlo integration** with ``num_samples`` treatment draws otherwise.

The design adapts the ZNet reference (``ZNet/models/treatment_effect_estimators/
deep_iv.py``) — which is binary-scalar-only — to cover all four treatment
types in a single class.
"""
from __future__ import annotations

import itertools
import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.npiv_estimators.helper import (
    MLP,
    DeepIVTreatmentOutput,
    NPIVOutcomeOutput,
    _ensure_2d,
    concat_context,
)
from models.npiv_estimators.parent_class import NPIVEstimator, TreatmentFamily


# ═══════════════════════════════════════════════════════════════════
#  Structural response net  h(A, W, L)
# ═══════════════════════════════════════════════════════════════════


class StructuralResponseNet(nn.Module):
    """Structural response surface ``h(A, W, L) → Y``.

    Accepts scalar or vector, continuous or binary ``A``. When
    ``y_dim == 1`` the forward returns shape ``[B]`` (squeezed) for
    backwards compatibility with scalar-outcome callers; otherwise it
    returns ``[B, y_dim]``.
    """

    def __init__(
        self,
        a_dim: int,
        w_dim: int,
        l_dim: int,
        y_dim: int = 1,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.y_dim = y_dim
        self.net = MLP(
            in_dim=a_dim + w_dim + l_dim,
            hidden_dims=hidden_dims,
            out_dim=y_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        a = _ensure_2d(a)
        out = self.net(torch.cat([a, w, l], dim=-1))
        return out.squeeze(-1) if self.y_dim == 1 else out


# ═══════════════════════════════════════════════════════════════════
#  Stage-1 treatment density models
# ═══════════════════════════════════════════════════════════════════


class GaussianMixtureTreatmentNet(nn.Module):
    """Mixture-density network for **continuous** treatment ``p(A | context)``.

    Each component is a diagonal Gaussian in ``a_dim`` dimensions, so it
    handles both scalar (``a_dim=1``) and vector (``a_dim>1``) continuous
    treatments.
    """

    def __init__(
        self,
        in_dim: int,
        a_dim: int,
        num_components: int = 5,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.a_dim = a_dim
        self.num_components = num_components
        self.min_scale = min_scale

        feat_dim = hidden_dims[-1] if hidden_dims else in_dim
        self.backbone = MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=feat_dim,
            activation=activation,
            dropout=dropout,
            activate_last=True,
        )
        self.logits_head = nn.Linear(feat_dim, num_components)
        self.mean_head = nn.Linear(feat_dim, num_components * a_dim)
        self.log_scale_head = nn.Linear(feat_dim, num_components * a_dim)

    def _params(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.backbone(x)
        logits = self.logits_head(h)
        means = self.mean_head(h).view(-1, self.num_components, self.a_dim)
        log_scales = self.log_scale_head(h).view(-1, self.num_components, self.a_dim)
        scales = F.softplus(log_scales) + self.min_scale
        return logits, means, scales

    def nll(self, x: Tensor, a: Tensor) -> DeepIVTreatmentOutput:
        a = _ensure_2d(a)
        logits, means, scales = self._params(x)
        a_expanded = a.unsqueeze(1)
        log_prob_per_dim = -0.5 * (
            ((a_expanded - means) / scales) ** 2
            + 2.0 * torch.log(scales)
            + math.log(2.0 * math.pi)
        )
        comp_log_prob = log_prob_per_dim.sum(dim=-1)
        log_mix = F.log_softmax(logits, dim=-1)
        log_prob = torch.logsumexp(log_mix + comp_log_prob, dim=-1)
        nll = -log_prob.mean()

        weights = torch.softmax(logits, dim=-1)
        mean = (weights.unsqueeze(-1) * means).sum(dim=1)
        return DeepIVTreatmentOutput(
            nll=nll, mean=mean, samples=None,
            extra={"mixture_logits": logits, "mixture_weights": weights,
                   "means": means, "scales": scales},
        )

    @torch.no_grad()
    def mean(self, x: Tensor) -> Tensor:
        logits, means, _ = self._params(x)
        weights = torch.softmax(logits, dim=-1)
        return (weights.unsqueeze(-1) * means).sum(dim=1)

    def sample(self, x: Tensor, num_samples: int = 1) -> Tensor:
        """Returns ``[B, num_samples, a_dim]`` treatment draws."""
        logits, means, scales = self._params(x)
        weights = torch.softmax(logits, dim=-1)
        bsz = x.shape[0]

        cat = torch.distributions.Categorical(probs=weights)
        comp_idx = cat.sample((num_samples,)).transpose(0, 1)

        means_exp = means.unsqueeze(1).expand(bsz, num_samples, self.num_components, self.a_dim)
        scales_exp = scales.unsqueeze(1).expand(bsz, num_samples, self.num_components, self.a_dim)
        gather_idx = comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.a_dim)

        chosen_means = means_exp.gather(2, gather_idx).squeeze(2)
        chosen_scales = scales_exp.gather(2, gather_idx).squeeze(2)
        eps = torch.randn_like(chosen_means)
        return chosen_means + chosen_scales * eps


class BernoulliTreatmentNet(nn.Module):
    """Independent-Bernoulli treatment model for **binary** ``A`` (any ``a_dim``)."""

    def __init__(
        self,
        in_dim: int,
        a_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.a_dim = a_dim
        self.net = MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=a_dim,
            activation=activation,
            dropout=dropout,
        )

    def logits(self, x: Tensor) -> Tensor:
        return self.net(x)

    def nll(self, x: Tensor, a: Tensor) -> DeepIVTreatmentOutput:
        a = _ensure_2d(a)
        logits = self.logits(x)
        nll = F.binary_cross_entropy_with_logits(logits, a.float())
        mean = torch.sigmoid(logits)
        return DeepIVTreatmentOutput(
            nll=nll, mean=mean, samples=None, extra={"logits": logits},
        )

    @torch.no_grad()
    def mean(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(x))

    def sample(self, x: Tensor, num_samples: int = 1) -> Tensor:
        """Returns ``[B, num_samples, a_dim]`` Bernoulli draws (float ∈ {0,1})."""
        probs = self.mean(x).unsqueeze(1).expand(-1, num_samples, -1)
        return torch.bernoulli(probs)


# ═══════════════════════════════════════════════════════════════════
#  DeepIV estimator
# ═══════════════════════════════════════════════════════════════════


class DeepIV(NPIVEstimator):
    """DeepIV (Hartford et al. 2017) with IV context ``(Z, W, L)``.

    The ``Z`` channel is DeepIV's instrument; ``W`` and ``L`` carry
    learned / observed covariates (confounder representation and
    observed covariates respectively).

    All four ``(a_dim, treatment_family)`` combinations are supported:

    +-----------------+------------+-------------+-------------------------------+
    | scalar/vector   | continuous | binary      | binary vector exact path      |
    +-----------------+------------+-------------+-------------------------------+
    | ``a_dim == 1``  | MDN        | Bernoulli   | enumerate ``{0,1}``           |
    | ``a_dim > 1``   | MDN        | Bernoulli   | enumerate up to ``a_dim≤8``   |
    +-----------------+------------+-------------+-------------------------------+

    Stage-1 is trained by minimising the treatment NLL; stage-2 by
    minimising ``E[(Y − E[h|Z,W,L])²]`` via :meth:`outcome_loss`.
    """

    def __init__(
        self,
        a_dim: int,
        z_dim: int,
        w_dim: int,
        l_dim: int,
        y_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        num_components: int = 5,
        treatment_hidden_dims: Sequence[int] = (256, 256),
        response_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.l_dim = l_dim
        self.y_dim = y_dim
        self.treatment_family = treatment_family

        in_dim = z_dim + w_dim + l_dim
        if treatment_family == "continuous":
            self.treatment_net: nn.Module = GaussianMixtureTreatmentNet(
                in_dim=in_dim, a_dim=a_dim, num_components=num_components,
                hidden_dims=treatment_hidden_dims,
                activation=activation, dropout=dropout, min_scale=min_scale,
            )
        elif treatment_family == "binary":
            self.treatment_net = BernoulliTreatmentNet(
                in_dim=in_dim, a_dim=a_dim,
                hidden_dims=treatment_hidden_dims,
                activation=activation, dropout=dropout,
            )
        else:
            raise ValueError("treatment_family must be 'continuous' or 'binary'.")

        self.response_net = StructuralResponseNet(
            a_dim=a_dim, w_dim=w_dim, l_dim=l_dim, y_dim=y_dim,
            hidden_dims=response_hidden_dims,
            activation=activation, dropout=dropout,
        )

    # ── Convenience accessors ───────────────────────────────────────

    def treatment_loss(
        self, a: Tensor, z: Tensor, w: Tensor, l: Tensor,
    ) -> DeepIVTreatmentOutput:
        return self.treatment_net.nll(concat_context(z, w, l), a)

    def sample_treatment(
        self, z: Tensor, w: Tensor, l: Tensor, num_samples: int = 1,
    ) -> Tensor:
        return self.treatment_net.sample(
            concat_context(z, w, l), num_samples=num_samples,
        )

    @torch.no_grad()
    def treatment_mean(self, z: Tensor, w: Tensor, l: Tensor) -> Tensor:
        return self.treatment_net.mean(concat_context(z, w, l))

    def response(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        return self.response_net(a, w, l)

    # ── Stage-2 outcome loss (exact for small binary a, MC otherwise) ──

    def outcome_loss(
        self,
        y: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
        num_samples: int = 8,
    ) -> NPIVOutcomeOutput:
        """Stage-2 MSE loss ``E[(Y − E[h|Z,W,L])²]``.

        Dispatches on treatment family / dimensionality:

        * ``binary`` with ``2^a_dim ≤ 256`` → exact enumeration.
        * otherwise → Monte-Carlo with ``num_samples`` draws.
        """
        if self.treatment_family == "binary" and 2 ** self.a_dim <= 256:
            return self._binary_exact_outcome_loss(y, z, w, l)
        return self._mc_outcome_loss(y, z, w, l, num_samples)

    def _binary_exact_outcome_loss(
        self, y: Tensor, z: Tensor, w: Tensor, l: Tensor,
    ) -> NPIVOutcomeOutput:
        y = _ensure_2d(y)
        bsz = z.shape[0]

        probs = self.treatment_net.mean(concat_context(z, w, l))    # [B, a_dim]
        combos = torch.tensor(
            list(itertools.product([0.0, 1.0], repeat=self.a_dim)),
            dtype=torch.float32, device=z.device,
        )                                                            # [C, a_dim]
        n_combos = combos.shape[0]

        log_p = torch.log(probs.clamp(min=1e-7))
        log_1mp = torch.log((1.0 - probs).clamp(min=1e-7))
        combos_e = combos.unsqueeze(0)                               # [1, C, a_dim]
        log_w = (
            combos_e * log_p.unsqueeze(1)
            + (1.0 - combos_e) * log_1mp.unsqueeze(1)
        ).sum(dim=-1)                                                # [B, C]
        weights = log_w.exp()

        combos_rep = combos.unsqueeze(0).expand(bsz, -1, -1).reshape(
            bsz * n_combos, self.a_dim,
        )
        w_rep = w.unsqueeze(1).expand(bsz, n_combos, -1).reshape(
            bsz * n_combos, w.shape[-1],
        )
        l_rep = l.unsqueeze(1).expand(bsz, n_combos, -1).reshape(
            bsz * n_combos, l.shape[-1],
        )
        raw_pred = _ensure_2d(self.response_net(combos_rep, w_rep, l_rep))
        pred_all = raw_pred.view(bsz, n_combos, self.y_dim)
        pred_mean = (weights.unsqueeze(-1) * pred_all).sum(dim=1)    # [B, y_dim]
        loss = F.mse_loss(pred_mean, y)

        return NPIVOutcomeOutput(
            loss=loss,
            metrics={
                "deepiv_outcome_loss": loss.detach(),
                "deepiv_pred_mean": pred_mean.mean().detach(),
                "deepiv_y_mean": y.mean().detach(),
            },
        )

    def _mc_outcome_loss(
        self,
        y: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
        num_samples: int,
    ) -> NPIVOutcomeOutput:
        """Unbiased MC stage-2 loss via Hartford's two-independent-sample trick.

        For ``f(z,w,l) := E_A[h(A,W,L) | Z,W,L]`` the target loss is
        ``E[(Y − f)²]``.  A single-batch MC plug-in
        ``MSE((1/S) Σ h(A_s), Y)`` is biased by ``Var_A[h] / S``, which —
        with the small defaults we ship (``S = 8``) — pushes the response
        net toward making ``h`` *constant in A* (lowest variance) and
        collapses the treatment signal.

        The two-sample estimator draws two independent MC batches and uses

            E[(Y − μ₁)(Y − μ₂)] = (Y − f)²

        where ``μ₁`` and ``μ₂`` are the two-batch sample means.  This is
        exactly unbiased for any ``S ≥ 1`` per side, and the reported
        ``loss`` value tracks the true ``(Y − f)²`` objective (not the
        attenuated variant).
        """
        y = _ensure_2d(y)
        # Draw two *independent* batches of S samples each (torch sampling
        # is iid, so taking two separate calls is enough — no shared noise).
        half = max(1, num_samples)
        a1 = self.sample_treatment(z, w, l, num_samples=half)
        a2 = self.sample_treatment(z, w, l, num_samples=half)

        def _mean_h(a_samples: Tensor) -> Tensor:
            bsz, s, a_dim = a_samples.shape
            w_exp = w.unsqueeze(1).expand(bsz, s, w.shape[-1])
            l_exp = l.unsqueeze(1).expand(bsz, s, l.shape[-1])
            raw = self.response_net(
                a_samples.reshape(bsz * s, a_dim),
                w_exp.reshape(bsz * s, w.shape[-1]),
                l_exp.reshape(bsz * s, l.shape[-1]),
            )
            raw = _ensure_2d(raw)
            return raw.view(bsz, s, self.y_dim).mean(dim=1)           # [B, y_dim]

        mu1 = _mean_h(a1)
        mu2 = _mean_h(a2)
        loss = ((y - mu1) * (y - mu2)).mean()

        return NPIVOutcomeOutput(
            loss=loss,
            metrics={
                "deepiv_outcome_loss": loss.detach(),
                "deepiv_pred_mean": (0.5 * (mu1 + mu2)).mean().detach(),
                "deepiv_y_mean": y.mean().detach(),
            },
        )

    # ── Prediction API ──────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        """Counterfactual ``ĥ(A, W, L)``; shape ``[B]`` when ``y_dim=1``."""
        return self.response(a, w, l)


__all__ = [
    "BernoulliTreatmentNet",
    "DeepIV",
    "GaussianMixtureTreatmentNet",
    "StructuralResponseNet",
]
