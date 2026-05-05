"""DeepGMM with radial test functions for NPIV.

* ``modeler`` — structural MLP ``g(A, W, L) -> Y``
* ``critics`` — bank of Gaussian radial basis test functions over
  ``(Z, W, L)``, each with a learnable centre and bandwidth

The trainer in :mod:`trainer` optimises a squared-moment objective against
this critic bank and selects checkpoints by held-out moment violation.
"""
from __future__ import annotations

import math
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
from models.npiv_estimators.oadam import OAdam
from models.npiv_estimators.parent_class import NPIVEstimator, TreatmentFamily


def _xavier_init(net: MLP, activation: str) -> None:
    """Initialise an MLP with Xavier-normal weights."""
    name = activation.lower()
    try:
        if name == "leaky_relu":
            gain = torch.nn.init.calculate_gain("leaky_relu", 0.01)
        else:
            gain = torch.nn.init.calculate_gain(name)
    except ValueError:
        gain = 1.0

    layers = list(net.net.children())
    for layer in layers[:-1]:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
    final = layers[-1]
    if isinstance(final, nn.Linear):
        nn.init.xavier_normal_(final.weight.data, gain=1.0)
        nn.init.zeros_(final.bias.data)


class GaussianBasisCritic(nn.Module):
    """Scalar Gaussian basis function over the instrument/context channel."""

    def __init__(self, in_dim: int, init_sigma: float = 1.0) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.empty(in_dim))
        raw = math.log(math.expm1(max(float(init_sigma), 1e-3)))
        self._init_raw_log_sigma = raw
        self.raw_log_sigma = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.center, mean=0.0, std=1.0)
        with torch.no_grad():
            self.raw_log_sigma.fill_(self._init_raw_log_sigma)

    def sigma(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_log_sigma) + 1e-3

    def forward(self, x: Tensor) -> Tensor:
        sigma = self.sigma()
        diff = x - self.center
        dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-0.5 * dist_sq / (sigma * sigma))


class DeepGMM(NPIVEstimator):
    """DeepGMM with a modeler MLP and an RBF critic bank."""

    def __init__(
        self,
        a_dim: int,
        z_dim: int,
        w_dim: int,
        l_dim: int,
        y_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        learner_hidden_dims: Sequence[int] = (100, 100),
        activation: str = "leaky_relu",
        lambda_1: float = 0.25,
        num_critics: int = 40,
        critic_init_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.l_dim = l_dim
        self.y_dim = y_dim
        self.treatment_family = treatment_family
        self.activation = activation
        self.lambda_1 = float(lambda_1)
        self.num_critics = int(num_critics)
        self.critic_init_sigma = float(critic_init_sigma)

        self.modeler = MLP(
            in_dim=a_dim + w_dim + l_dim,
            hidden_dims=list(learner_hidden_dims),
            out_dim=y_dim,
            activation=activation,
            dropout=0.0,
        )
        ctx_dim = z_dim + w_dim + l_dim
        self.critics = nn.ModuleList(
            [
                GaussianBasisCritic(ctx_dim, init_sigma=critic_init_sigma)
                for _ in range(self.num_critics)
            ],
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _xavier_init(self.modeler, self.activation)
        for critic in self.critics:
            critic.reset_parameters()

    def g_forward(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        x = torch.cat([_ensure_2d(a), concat_wl(w, l)], dim=-1)
        return self.modeler(x)

    def critic_values(self, z: Tensor, w: Tensor, l: Tensor) -> Tensor:
        ctx = concat_context(z, w, l)
        vals = [critic(ctx) for critic in self.critics]
        return torch.cat(vals, dim=-1)

    def moment_terms(
        self,
        y: Tensor,
        a: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y2 = _ensure_2d(y)
        pred = _ensure_2d(self.g_forward(a, w, l))
        phi = self.critic_values(z, w, l)
        resid = y2 - pred
        moments = torch.mean(resid * phi, dim=0)
        gmm_norm_sq = torch.mean(moments * moments)
        resid_mse = torch.mean(resid * resid)
        return gmm_norm_sq, resid_mse, phi, pred

    def modeler_loss(
        self,
        y: Tensor,
        a: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
        *,
        ols_weight: float = 0.0,
    ) -> NPIVOutcomeOutput:
        gmm_norm_sq, resid_mse, _, _ = self.moment_terms(y, a, z, w, l)
        loss = gmm_norm_sq + float(ols_weight) * resid_mse
        return NPIVOutcomeOutput(
            loss=loss,
            metrics={
                "deepgmm_gmm_norm_sq": gmm_norm_sq.detach(),
                "deepgmm_resid_mse": resid_mse.detach(),
            },
        )

    def critic_loss(
        self,
        y: Tensor,
        a: Tensor,
        z: Tensor,
        w: Tensor,
        l: Tensor,
        *,
        reg_weight: float | None = None,
    ) -> NPIVOutcomeOutput:
        gmm_norm_sq, _, phi, _ = self.moment_terms(y, a, z, w, l)
        critic_reg = torch.mean(phi * phi)
        weight = self.lambda_1 if reg_weight is None else float(reg_weight)
        loss = -gmm_norm_sq + weight * critic_reg
        return NPIVOutcomeOutput(
            loss=loss,
            metrics={
                "deepgmm_gmm_norm_sq": gmm_norm_sq.detach(),
                "deepgmm_critic_reg": critic_reg.detach(),
            },
        )

    def make_optimizers(self, g_lr: float, f_lr: float) -> tuple[OAdam, OAdam]:
        g_opt = OAdam(self.modeler.parameters(), lr=g_lr, betas=(0.5, 0.9))
        f_opt = OAdam(self.critics.parameters(), lr=f_lr, betas=(0.5, 0.9))
        return g_opt, f_opt

    @torch.no_grad()
    def predict(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        out = self.g_forward(a, w, l)
        return out.squeeze(-1) if self.y_dim == 1 else out


__all__ = ["DeepGMM", "GaussianBasisCritic"]
