"""Ablation model for PIV: Proxy-Orth (orthogonalisation only).

**Sequential training.**

1. Fit :math:`q_{\\rho}^M \\circ e_{\\gamma}` to remove the proxy-predictable component of
   :math:`X` (MSE reconstruction :math:`X \\approx q_{\\rho}(e_{\\gamma}(M))`).

2. Freeze :math:`e_{\\gamma}, q_{\\rho}`. Define
   :math:`R_i^M = X_i - q_{\\rho}^M(e_{\\gamma}(M_i))`, and train :math:`h_{\\nu}^M` and
   :math:`t_{\\omega}^M` to predict treatment :math:`A` from :math:`Z_i = h_{\\nu}^M(R_i^M)`
   (continuous: MSE on logits; binary: BCEWithLogits).

At inference, :math:`\\widehat{V}_{\\mathrm{PO}} = h_{\\nu}^M(R^M)`; :math:`\\hat{X} =
q_{\\rho}(e_{\\gamma}(M))` is exported as the confounder branch ``W``.  ``L`` is unused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TreatmentFamily = Literal["continuous", "binary"]


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class _MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        act = _get_activation(activation)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act.__class__())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class AblationOutput:
    residual: Tensor
    z: Tensor
    loss: Tensor
    metrics: Dict[str, Tensor]


def _set_requires_grad(module: nn.Module | None, flag: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


class ProxyOrth(nn.Module):
    """Two-stage: fit :math:`q_{\\rho}\\circ e_{\\gamma}` on :math:`X`, then :math:`h_{\\nu},t_{\\omega}` on :math:`A|Z`."""

    def __init__(
        self,
        x_dim: int,
        m_dim: int,
        l_dim: int,
        z_dim: int,
        a_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        proxy_embed_dim: int = 16,
        proxy_hidden_dims: Sequence[int] = (128, 128),
        residual_hidden_dims: Sequence[int] = (256, 256),
        z_hidden_dims: Sequence[int] = (256, 256),
        treatment_hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        residual_weight: float = 1.0,
        treatment_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if treatment_family not in {"continuous", "binary"}:
            raise ValueError("treatment_family must be 'continuous' or 'binary'.")

        _ = (l_dim,)

        self.treatment_family = treatment_family  # informational only (exported checkpoints)
        self.a_dim = int(a_dim)
        self.residual_weight = float(residual_weight)
        self.treatment_weight = float(treatment_weight)

        self.proxy_encoder = _MLP(
            in_dim=m_dim,
            hidden_dims=proxy_hidden_dims,
            out_dim=proxy_embed_dim,
            activation=activation,
            dropout=dropout,
        )
        self.residualizer = _MLP(
            in_dim=proxy_embed_dim,
            hidden_dims=residual_hidden_dims,
            out_dim=x_dim,
            activation=activation,
            dropout=dropout,
        )
        # Trainable h_ν^M: R^M → Z (after stage 2; frozen during stage 1).
        self.h_nu = _MLP(
            in_dim=x_dim,
            hidden_dims=z_hidden_dims,
            out_dim=z_dim,
            activation=activation,
            dropout=dropout,
        )
        # Trainable t_ω^M: Z → A (after stage 2; frozen during stage 1).
        self.t_omega = _MLP(
            in_dim=z_dim,
            hidden_dims=treatment_hidden_dims,
            out_dim=a_dim,
            activation=activation,
            dropout=dropout,
        )

    def e_gamma(self, m: Tensor) -> Tensor:
        return self.proxy_encoder(m)

    def encode_proxy(self, m: Tensor) -> Tensor:
        return self.e_gamma(m)

    def set_stage1_trainable(self) -> None:
        """Train :math:`q_{\\rho}, e_{\\gamma}` only; freeze :math:`h_{\\nu}, t_{\\omega}`."""
        _set_requires_grad(self.proxy_encoder, True)
        _set_requires_grad(self.residualizer, True)
        _set_requires_grad(self.h_nu, False)
        _set_requires_grad(self.t_omega, False)

    def set_stage2_trainable(self) -> None:
        """Freeze :math:`q_{\\rho}, e_{\\gamma}`; train :math:`h_{\\nu}, t_{\\omega}`."""
        _set_requires_grad(self.proxy_encoder, False)
        _set_requires_grad(self.residualizer, False)
        _set_requires_grad(self.h_nu, True)
        _set_requires_grad(self.t_omega, True)

    def residualize(self, x: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        emb = self.e_gamma(m)
        x_hat = self.residualizer(emb)
        r_m = x - x_hat
        return r_m, x_hat

    def encode_z(self, x: Tensor, m: Tensor) -> Tensor:
        r_m, _ = self.residualize(x, m)
        return self.h_nu(r_m)

    def forward(
        self,
        x: Tensor,
        m: Tensor,
        l: Tensor,
        *,
        detach_q_path: bool = False,
    ) -> Dict[str, Tensor]:
        """If ``detach_q_path``, :math:`\\hat X, R^M` omit gradients through :math:`q_{\\rho},e_{\\gamma}` (stage 2)."""
        _ = l
        if detach_q_path:
            with torch.no_grad():
                emb = self.e_gamma(m)
                x_hat = self.residualizer(emb)
                r_m = x - x_hat
        else:
            emb = self.e_gamma(m)
            x_hat = self.residualizer(emb)
            r_m = x - x_hat
        z = self.h_nu(r_m)
        a_logits = self.t_omega(z)
        if self.treatment_family == "binary":
            a_hat = torch.sigmoid(a_logits)
        else:
            a_hat = a_logits
        return {
            "emb": emb,
            "a_logits": a_logits,
            "a_hat": a_hat,
            "x_hat": x_hat,
            "r_m": r_m,
            "residual": r_m,
            "z": z,
        }

    def compute_loss_stage1(self, x: Tensor, m: Tensor, l: Tensor) -> AblationOutput:
        """MSE reconstruction of :math:`X` only (train :math:`q_{\\rho}, e_{\\gamma}`)."""
        _ = l
        emb = self.e_gamma(m)
        x_hat = self.residualizer(emb)
        r_m = x - x_hat
        with torch.no_grad():
            z = self.h_nu(r_m.detach())
        recon_loss = F.mse_loss(x_hat, x)
        loss = self.residual_weight * recon_loss
        metrics = {
            "proxy_orth_loss": loss.detach(),
            "proxy_orth_recon_loss": recon_loss.detach(),
        }
        return AblationOutput(
            residual=r_m,
            z=z,
            loss=loss,
            metrics=metrics,
        )

    def compute_loss_stage2(self, x: Tensor, m: Tensor, l: Tensor, a: Tensor) -> AblationOutput:
        """Predict :math:`A` from :math:`Z = h_{\\nu}(R^M)` with frozen :math:`q_{\\rho},e_{\\gamma}`."""
        _ = l
        outputs = self.forward(x, m, l, detach_q_path=True)
        logits = outputs["a_logits"]
        if self.treatment_family == "binary":
            if logits.shape != a.shape:
                logits = logits.view_as(a)
            treatment_loss = F.binary_cross_entropy_with_logits(logits, a)
        else:
            treatment_loss = F.mse_loss(logits, a)
        loss = self.treatment_weight * treatment_loss
        metrics = {
            "proxy_orth_loss": loss.detach(),
            "proxy_orth_treatment_loss": treatment_loss.detach(),
        }
        return AblationOutput(
            residual=outputs["residual"],
            z=outputs["z"],
            loss=loss,
            metrics=metrics,
        )



def build_proxy_orth(
    x_dim: int,
    m_dim: int,
    l_dim: int,
    z_dim: int,
    a_dim: int = 1,
    treatment_family: TreatmentFamily = "continuous",
) -> ProxyOrth:
    return ProxyOrth(
        x_dim=x_dim,
        m_dim=m_dim,
        l_dim=l_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        treatment_family=treatment_family,
    )


__all__ = [
    "AblationOutput",
    "ProxyOrth",
    "TreatmentFamily",
    "build_proxy_orth",
]
