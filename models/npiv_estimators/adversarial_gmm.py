"""Adversarial GMM (Bennett et al.; Microsoft-style) for NPIV.

This module follows the core supervised-loss AGMM game used in
``microsoft/AdversarialGMM``:

    min_h max_f  E[(Y - h(A, W, L)) f(Z, W, L)] - E[f(Z, W, L)^2]

with alternating learner/adversary updates and an optional OLS term on
the learner objective.

Compared with the original research code, this implementation is
adapted for this repository's estimator interface and supports all four
treatment layouts through a single ``a_dim`` input:

    scalar-continuous   a_dim=1,  treatment_family="continuous"
    scalar-binary       a_dim=1,  treatment_family="binary"
    vector-continuous   a_dim>1,  treatment_family="continuous"
    vector-binary       a_dim>1,  treatment_family="binary"
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

from models.npiv_estimators.helper import (
    MLP,
    NPIVOutcomeOutput,
    _ensure_2d,
    concat_context,
    concat_wl,
)
from models.npiv_estimators.oadam import OAdam
from models.npiv_estimators.parent_class import NPIVEstimator, TreatmentFamily


def add_weight_decay(
    net: torch.nn.Module,
    l2_value: float,
    skip_list: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Match Microsoft's AGMM rule: skip bias and 1D parameters."""
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": float(l2_value)},
    ]


class AdversarialGMM(NPIVEstimator):
    """Neural AGMM estimator with alternating minimax updates.

    Parameters
    ----------
    a_dim, z_dim, w_dim, l_dim:
        Input widths of treatment/instrument/context channels.
    y_dim:
        Outcome width (the current scripts use scalar outcome ``1``).
    treatment_family:
        Declared treatment type (continuous/binary); AGMM accepts either
        because treatment enters as a generic float vector.
    learner_hidden_dims, adversary_hidden_dims:
        MLP widths for the structural learner ``h`` and test function
        adversary ``f``.
    """

    def __init__(
        self,
        a_dim: int,
        z_dim: int,
        w_dim: int,
        l_dim: int,
        y_dim: int = 1,
        adversary_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        learner_hidden_dims: Sequence[int] = (256, 256),
        adversary_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "leaky_relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.l_dim = l_dim
        self.y_dim = y_dim
        self.adversary_dim = adversary_dim
        self.treatment_family = treatment_family

        self.learner = MLP(
            in_dim=a_dim + w_dim + l_dim,
            hidden_dims=learner_hidden_dims,
            out_dim=y_dim,
            activation=activation,
            dropout=dropout,
        )
        self.adversary = MLP(
            in_dim=z_dim + w_dim + l_dim,
            hidden_dims=adversary_hidden_dims,
            out_dim=adversary_dim,
            activation=activation,
            dropout=dropout,
        )
        self._adversary_skip_list: tuple[str, ...] = ()

    def reset_parameters(self) -> None:
        """Reset both networks in the same way as the Microsoft reference."""
        self.learner.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None,
        )
        self.adversary.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None,
        )

    def make_optimizers(
        self,
        *,
        learner_lr: float,
        adversary_lr: float,
        learner_l2: float,
        adversary_l2: float,
        optimistic: bool = True,
    ) -> tuple[Optimizer, Optimizer]:
        """Build AGMM optimizers with official parameter-group handling."""
        opt_cls = OAdam if optimistic else torch.optim.Adam
        opt_kwargs = {"betas": (0.0, 0.01)} if optimistic else {}
        learner_opt = opt_cls(
            add_weight_decay(self.learner, learner_l2),
            lr=learner_lr,
            **opt_kwargs,
        )
        adversary_opt = opt_cls(
            add_weight_decay(
                self.adversary,
                adversary_l2,
                skip_list=self._adversary_skip_list,
            ),
            lr=adversary_lr,
            **opt_kwargs,
        )
        return learner_opt, adversary_opt

    # ── Networks ────────────────────────────────────────────────────

    def learner_forward(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        out = self.learner(torch.cat([_ensure_2d(a), concat_wl(w, l)], dim=-1))
        return out

    def adversary_forward(self, z: Tensor, w: Tensor, l: Tensor) -> Tensor:
        return self.adversary(concat_context(z, w, l))

    # ── Objectives (Microsoft supervised-loss AGMM form) ───────────

    def learner_loss(
        self,
        y: Tensor,
        a: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
        *,
        ols_weight: float = 0.0,
    ) -> NPIVOutcomeOutput:
        """Learner update: minimise ``E[(Y-h)f] + λ_ols E[(Y-h)^2]``."""
        y2 = _ensure_2d(y)
        pred = _ensure_2d(self.learner_forward(a, w, l))
        test = _ensure_2d(self.adversary_forward(z, w, l))
        resid = y2 - pred
        loss = torch.mean(resid * test) + float(ols_weight) * torch.mean(resid ** 2)
        return NPIVOutcomeOutput(
            loss=loss,
            metrics={
                "agmm_learner_loss": loss.detach(),
                "agmm_resid_mse": torch.mean(resid ** 2).detach(),
            },
        )

    def adversary_loss(
        self,
        y: Tensor,
        a: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
        *,
        adversary_norm_reg: float = 0.0,
    ) -> NPIVOutcomeOutput:
        """Adversary update: maximise ``E[(Y-h)f] - E[f^2]``.

        We return the minimisation form used in SGD:
        ``-E[(Y-h)f] + E[f^2] + λ_adv ||f||²``.
        """
        y2 = _ensure_2d(y)
        pred = _ensure_2d(self.learner_forward(a, w, l))
        test = _ensure_2d(self.adversary_forward(z, w, l))
        resid = y2 - pred
        test_sq = torch.mean(test ** 2)
        loss = -torch.mean(resid * test) + test_sq + float(adversary_norm_reg) * test_sq
        return NPIVOutcomeOutput(
            loss=loss,
            metrics={
                "agmm_adversary_loss": loss.detach(),
                "agmm_test_sq": test_sq.detach(),
            },
        )

    # ── Prediction API ──────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        out = self.learner_forward(a, w, l)
        return out.squeeze(-1) if self.y_dim == 1 else out


__all__ = ["AdversarialGMM"]
