"""Deep Feature IV (DFIV; Xu et al. 2021).

Supports all four treatment types:

    scalar-continuous   a_dim=1,  treatment_family="continuous"
    scalar-binary       a_dim=1,  treatment_family="binary"
    vector-continuous   a_dim>1,  treatment_family="continuous"
    vector-binary       a_dim>1,  treatment_family="binary"

DFIV learns three feature networks (Xu et al. 2021, Algorithm 1):

* ``ψ(Z, W, L)`` — representation used to predict a treatment feature.
* ``φ(A)``       — treatment representation ("A features").
* ``ξ(W, L)``    — covariate representation used in the outcome stage.

Stage 1 (inner):  find a linear V minimising

    ||φ(A) − V · ψ(Z, W, L)||²  +  λ₁ · ||V||²    (closed form, per batch)

Stage 2 (outer):  find μ in the outer-product space ``(Vψ) ⊗ ξ``
such that

    Y ≈ μᵀ · ((V · ψ(Z,W,L)) ⊗ ξ(W,L))

by minimising a ridge-regularised squared loss (closed form per batch).

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from models.npiv_estimators.helper import (
    MLP,
    NPIVOutcomeOutput,
    _ensure_2d,
    concat_context,
    concat_wl,
)
from models.npiv_estimators.parent_class import NPIVEstimator, TreatmentFamily


# ═══════════════════════════════════════════════════════════════════
#  DFIV estimator
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DFIVStageOutputs:
    """Return value of :meth:`DFIV.stage_losses` (used by the trainer)."""

    loss_psi: Tensor
    loss_phi: Tensor
    loss_xi: Tensor


class DFIV(NPIVEstimator):
    """DFIV estimator with three learned feature networks (ψ, φ, ξ).

    Parameters
    ----------
    a_dim, z_dim, w_dim, l_dim:
        Input widths for ``A``, ``Z``, ``W``, ``L`` (``L`` can be 0).
    y_dim:
        Outcome width (only ``1`` is currently supported for the ridge
        closed form used in ``compute_mu``).
    treatment_family:
        ``"continuous"`` or ``"binary"`` — declared for parity with
        :class:`DeepIV`; DFIV's training is identical in either case
        because ``φ`` just sees ``A`` as a generic float vector.
    feature_dim:
        Common output width for ψ / φ / ξ.
    psi/phi/xi_hidden_dims:
        Hidden widths for each feature network.
    lambda1, lambda2:
        Ridge penalties for the stage-1 (V) and stage-2 (μ) closed-form
        solves respectively.

    Notes
    -----
    ``μ`` has dimension ``feature_dim * feature_dim`` and is held as a
    buffer on the module after training.  :meth:`compute_final_mu`
    must be called once on the full training data after ``train_dfiv``
    to fix ``μ`` for downstream prediction.
    """

    def __init__(
        self,
        a_dim: int,
        z_dim: int,
        w_dim: int,
        l_dim: int,
        y_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        feature_dim: int = 32,
        psi_hidden_dims: Sequence[int] = (256, 256),
        phi_hidden_dims: Sequence[int] = (256, 256),
        xi_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
    ) -> None:
        super().__init__()
        if y_dim != 1:
            raise NotImplementedError("DFIV currently supports scalar Y only.")
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.l_dim = l_dim
        self.y_dim = y_dim
        self.treatment_family = treatment_family
        self.feature_dim = feature_dim
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

        self.psi = MLP(
            in_dim=z_dim + w_dim + l_dim,
            hidden_dims=psi_hidden_dims,
            out_dim=feature_dim,
            activation=activation, dropout=dropout,
            activate_last=True,
        )
        self.phi = MLP(
            in_dim=a_dim,
            hidden_dims=phi_hidden_dims,
            out_dim=feature_dim,
            activation=activation, dropout=dropout,
            activate_last=True,
        )
        self.xi = MLP(
            in_dim=w_dim + l_dim,
            hidden_dims=xi_hidden_dims,
            out_dim=feature_dim,
            activation=activation, dropout=dropout,
            activate_last=True,
        )
        self.register_buffer("mu", torch.zeros(feature_dim * feature_dim))
        self._mu_set = False

    # ── Feature accessors ───────────────────────────────────────────

    def features_psi(self, z: Tensor, w: Tensor, l: Tensor) -> Tensor:
        return self.psi(concat_context(z, w, l))

    def features_phi(self, a: Tensor) -> Tensor:
        return self.phi(_ensure_2d(a))

    def features_xi(self, w: Tensor, l: Tensor) -> Tensor:
        return self.xi(concat_wl(w, l))

    # ── Linear algebra blocks ──────────────────────────────────────

    @staticmethod
    def batch_outer_vec(a: Tensor, b: Tensor) -> Tensor:
        """Row-wise outer product ``a ⊗ b`` flattened to ``[B, dim_a*dim_b]``."""
        batch_size, dim_a = a.shape
        dim_b = b.shape[1]
        outer = torch.bmm(a.unsqueeze(2), b.unsqueeze(1))  # [B, da, db]
        return outer.reshape(batch_size, dim_a * dim_b)

    def compute_V(self, psi: Tensor, phi: Tensor) -> Tensor:
        """Closed-form ridge ``V̂`` regressing ``φ`` on ``ψ``.

        ``V̂ = φᵀ · ψ · (ψᵀ · ψ + λ₁ · B · I)⁻¹``, shape ``[d_phi, d_psi]``.
        Uses ``torch.linalg.solve`` with a pinv fallback for robustness
        on ill-conditioned mini-batches.
        """
        batch_size, dim_psi = psi.shape
        gram = psi.transpose(0, 1) @ psi + self.lambda1 * batch_size * torch.eye(
            dim_psi, device=psi.device, dtype=psi.dtype,
        )
        rhs = phi.transpose(0, 1) @ psi                          # [d_phi, d_psi]
        try:
            V = torch.linalg.solve(gram.transpose(0, 1), rhs.transpose(0, 1))
            V = V.transpose(0, 1)
        except torch.linalg.LinAlgError:
            V = rhs @ torch.linalg.pinv(gram)
        return V

    def compute_mu(
        self, V: Tensor, psi: Tensor, xi: Tensor, y: Tensor,
    ) -> Tensor:
        """Closed-form ridge ``μ̂`` for the outcome stage.

        Models ``Y ≈ μᵀ · ((V · ψ) ⊗ ξ)`` by ridge least-squares:

            μ̂ = (Xᵀ X + λ₂ B I)⁻¹ Xᵀ y       with  X = (V·ψ) ⊗ ξ.

        Returns a flat vector of size ``d_V · d_xi``.
        """
        batch_size = psi.shape[0]
        V_psi = (V @ psi.transpose(0, 1)).transpose(0, 1)         # [B, d_V]
        X = DFIV.batch_outer_vec(V_psi, xi)                       # [B, d_V*d_xi]
        out_dim = X.shape[1]

        gram = X.transpose(0, 1) @ X + self.lambda2 * batch_size * torch.eye(
            out_dim, device=X.device, dtype=X.dtype,
        )
        rhs = X.transpose(0, 1) @ _ensure_2d(y)                    # [d, y_dim]
        try:
            mu = torch.linalg.solve(gram, rhs).squeeze(-1)
        except torch.linalg.LinAlgError:
            mu = (torch.linalg.pinv(gram) @ rhs).squeeze(-1)
        return mu

    # ── Loss functions ──────────────────────────────────────────────

    def loss_psi(self, V: Tensor, psi: Tensor, phi_detached: Tensor) -> Tensor:
        """Stage-1 objective updating ψ with φ held fixed.

        Minimises ``||φ − V·ψ||² / B + λ₁·||V||²``.  The caller must
        pass ``phi.detach()`` so gradients flow only through ψ.
        """
        batch_size = psi.shape[0]
        resid = phi_detached.transpose(0, 1) - V @ psi.transpose(0, 1)
        l = (resid ** 2).sum() / batch_size
        l = l + self.lambda1 * (V ** 2).sum()
        return l

    def loss_outcome(
        self,
        mu: Tensor,
        V: Tensor,
        psi: Tensor,
        xi: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Stage-2 outcome objective ``||Y − μᵀ((V·ψ) ⊗ ξ)||²  + λ₂ ||μ||²``."""
        V_psi = (V @ psi.transpose(0, 1)).transpose(0, 1)
        X = DFIV.batch_outer_vec(V_psi, xi)
        pred = X @ mu
        y1 = _ensure_2d(y).squeeze(-1) if y.ndim > 1 else y
        l = ((y1 - pred) ** 2).mean()
        l = l + self.lambda2 * (mu ** 2).sum()
        return l

    # ── Post-training μ computation + prediction ───────────────────

    @torch.no_grad()
    def compute_final_mu(
        self,
        a: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Recompute ``μ̂`` on the full training data and store it as a buffer.

        After feature training, the final predictor uses the ridge
        μ̂ fitted on all ``(A, Z, W, L, Y)`` jointly — no stochastic
        mini-batch residue.  Call this once before :meth:`predict`.
        """
        self.eval()
        psi = self.features_psi(z, w, l)
        phi = self.features_phi(a)
        xi = self.features_xi(w, l)
        V = self.compute_V(psi, phi)
        mu = self.compute_mu(V, psi, xi, y)
        self.mu = mu.detach()
        self._mu_set = True
        return self.mu

    @torch.no_grad()
    def predict(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        """Counterfactual response ``ĥ(A, W, L) = μᵀ (φ(A) ⊗ ξ([W, L]))``.

        Note this does **not** use ``Z`` (counterfactual prediction
        marginalises over the instrument channel implicitly via the
        stage-1 solve during training).  Returns shape ``[B]`` when
        ``y_dim == 1``.
        """
        if not self._mu_set:
            # _mu_set is a plain Python attribute and is not saved in state_dict.
            # After a checkpoint load, mu has been restored but _mu_set is still
            # False.  Recover by checking whether the mu buffer is non-zero.
            if float(self.mu.abs().max()) > 0:
                self._mu_set = True
            else:
                raise RuntimeError(
                    "DFIV.predict called before compute_final_mu; train first.",
                )
        phi = self.features_phi(a)           # [B, d_phi]
        xi = self.features_xi(w, l)          # [B, d_xi]
        X = DFIV.batch_outer_vec(phi, xi)     # [B, d_phi*d_xi]
        return X @ self.mu                    # [B]


__all__ = ["DFIV", "DFIVStageOutputs"]
