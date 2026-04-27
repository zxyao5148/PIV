"""Ablation models for Proxy IV (PIV).

This module implements two orthogonalisation-based baselines:

1) Proxy-Orth:
   - Residualise X against an embedding of proxy M.
   - Learn IV representation Z from residuals.
   - Predict treatment from (Z, L).

2) Conf-Orth:
   - Keep a confounding representation W = E1(X) (typically from PIV stage-1).
   - Residualise X against W.
   - Learn IV representation Z from residuals.
   - Predict treatment from (Z, L), i.e. without conditioning on W.
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
    prediction: Tensor
    logits: Tensor | None
    loss: Tensor
    metrics: Dict[str, Tensor]


class ProxyOrth(nn.Module):
    """Proxy-Orth baseline.

    q_rho^M:  X <- e_gamma(M)
    R^M:      X - q_rho^M(e_gamma(M))
    h_nu^M:   Z <- R^M
    t:        A <- (Z, L)
    """

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

        self.treatment_family = treatment_family
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
        self.z_encoder = _MLP(
            in_dim=x_dim,
            hidden_dims=z_hidden_dims,
            out_dim=z_dim,
            activation=activation,
            dropout=dropout,
        )
        self.treatment_head = _MLP(
            in_dim=z_dim + l_dim,
            hidden_dims=treatment_hidden_dims,
            out_dim=a_dim,
            activation=activation,
            dropout=dropout,
        )

    def encode_proxy(self, m: Tensor) -> Tensor:
        return self.proxy_encoder(m)

    def residualize(self, x: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        m_emb = self.encode_proxy(m)
        x_hat = self.residualizer(m_emb)
        residual = x - x_hat
        return residual, x_hat

    def encode_z(self, x: Tensor, m: Tensor) -> Tensor:
        residual, _ = self.residualize(x, m)
        return self.z_encoder(residual)

    def forward(self, x: Tensor, m: Tensor, l: Tensor) -> Dict[str, Tensor]:
        residual, x_hat = self.residualize(x, m)
        z = self.z_encoder(residual)
        logits_or_mean = self.treatment_head(torch.cat([z, l], dim=-1))
        out: Dict[str, Tensor] = {
            "x_hat": x_hat,
            "residual": residual,
            "z": z,
            "prediction": logits_or_mean,
        }
        if self.treatment_family == "binary":
            out["logits"] = logits_or_mean
            out["prediction"] = torch.sigmoid(logits_or_mean)
        return out

    def compute_loss(self, x: Tensor, m: Tensor, l: Tensor, a: Tensor) -> AblationOutput:
        outputs = self.forward(x, m, l)
        x_hat = outputs["x_hat"]
        prediction = outputs["prediction"]
        logits = outputs.get("logits")

        if a.ndim == 1:
            a = a.unsqueeze(-1)
        a = a.float()

        residual_loss = F.mse_loss(x_hat, x)
        if self.treatment_family == "continuous":
            treatment_loss = F.mse_loss(prediction, a)
        else:
            if logits is None:
                raise RuntimeError("Binary treatment requires logits.")
            treatment_loss = F.binary_cross_entropy_with_logits(logits, a)

        loss = self.residual_weight * residual_loss + self.treatment_weight * treatment_loss
        metrics = {
            "proxy_orth_loss": loss.detach(),
            "proxy_orth_residual_loss": residual_loss.detach(),
            "proxy_orth_treatment_loss": treatment_loss.detach(),
        }
        return AblationOutput(
            residual=outputs["residual"],
            z=outputs["z"],
            prediction=prediction,
            logits=logits,
            loss=loss,
            metrics=metrics,
        )


class ConfOrth(nn.Module):
    """Conf-Orth baseline.

    Keeps confounder representation W from stage-1 (E1(X)).
    q_rho^W:  X <- W
    R^W:      X - q_rho^W(W)
    h_nu^W:   Z <- R^W
    t:        A <- (Z, L), i.e. no treatment head input from W.
    """

    def __init__(
        self,
        x_dim: int,
        l_dim: int,
        w_dim: int,
        z_dim: int,
        a_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        w_hidden_dims: Sequence[int] = (256, 256),
        residual_hidden_dims: Sequence[int] = (256, 256),
        z_hidden_dims: Sequence[int] = (256, 256),
        treatment_hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        freeze_w_encoder: bool = True,
        residual_weight: float = 1.0,
        treatment_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if treatment_family not in {"continuous", "binary"}:
            raise ValueError("treatment_family must be 'continuous' or 'binary'.")

        self.treatment_family = treatment_family
        self.freeze_w_encoder = bool(freeze_w_encoder)
        self.residual_weight = float(residual_weight)
        self.treatment_weight = float(treatment_weight)

        self.w_encoder = _MLP(
            in_dim=x_dim,
            hidden_dims=w_hidden_dims,
            out_dim=w_dim,
            activation=activation,
            dropout=dropout,
        )
        self.residualizer = _MLP(
            in_dim=w_dim,
            hidden_dims=residual_hidden_dims,
            out_dim=x_dim,
            activation=activation,
            dropout=dropout,
        )
        self.z_encoder = _MLP(
            in_dim=x_dim,
            hidden_dims=z_hidden_dims,
            out_dim=z_dim,
            activation=activation,
            dropout=dropout,
        )
        self.treatment_head = _MLP(
            in_dim=z_dim + l_dim,
            hidden_dims=treatment_hidden_dims,
            out_dim=a_dim,
            activation=activation,
            dropout=dropout,
        )

    def load_stage1_encoder(self, stage1: nn.Module, freeze: bool = True) -> None:
        self.w_encoder.load_state_dict(stage1.encoder.state_dict())
        self.freeze_w_encoder = bool(freeze)
        for p in self.w_encoder.parameters():
            p.requires_grad_(not freeze)

    def encode_w(self, x: Tensor) -> Tensor:
        if self.freeze_w_encoder:
            with torch.no_grad():
                return self.w_encoder(x)
        return self.w_encoder(x)

    def residualize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        w = self.encode_w(x)
        x_hat = self.residualizer(w)
        residual = x - x_hat
        return w, residual, x_hat

    def encode_z(self, x: Tensor) -> Tensor:
        _, residual, _ = self.residualize(x)
        return self.z_encoder(residual)

    def forward(self, x: Tensor, l: Tensor) -> Dict[str, Tensor]:
        w, residual, x_hat = self.residualize(x)
        z = self.z_encoder(residual)
        logits_or_mean = self.treatment_head(torch.cat([z, l], dim=-1))
        out: Dict[str, Tensor] = {
            "w": w,
            "x_hat": x_hat,
            "residual": residual,
            "z": z,
            "prediction": logits_or_mean,
        }
        if self.treatment_family == "binary":
            out["logits"] = logits_or_mean
            out["prediction"] = torch.sigmoid(logits_or_mean)
        return out

    def compute_loss(self, x: Tensor, l: Tensor, a: Tensor) -> AblationOutput:
        outputs = self.forward(x, l)
        x_hat = outputs["x_hat"]
        prediction = outputs["prediction"]
        logits = outputs.get("logits")

        if a.ndim == 1:
            a = a.unsqueeze(-1)
        a = a.float()

        residual_loss = F.mse_loss(x_hat, x)
        if self.treatment_family == "continuous":
            treatment_loss = F.mse_loss(prediction, a)
        else:
            if logits is None:
                raise RuntimeError("Binary treatment requires logits.")
            treatment_loss = F.binary_cross_entropy_with_logits(logits, a)

        loss = self.residual_weight * residual_loss + self.treatment_weight * treatment_loss
        metrics = {
            "conf_orth_loss": loss.detach(),
            "conf_orth_residual_loss": residual_loss.detach(),
            "conf_orth_treatment_loss": treatment_loss.detach(),
        }
        return AblationOutput(
            residual=outputs["residual"],
            z=outputs["z"],
            prediction=prediction,
            logits=logits,
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


def build_conf_orth(
    x_dim: int,
    l_dim: int,
    w_dim: int,
    z_dim: int,
    a_dim: int = 1,
    treatment_family: TreatmentFamily = "continuous",
) -> ConfOrth:
    return ConfOrth(
        x_dim=x_dim,
        l_dim=l_dim,
        w_dim=w_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        treatment_family=treatment_family,
    )


__all__ = [
    "AblationOutput",
    "ConfOrth",
    "ProxyOrth",
    "TreatmentFamily",
    "build_conf_orth",
    "build_proxy_orth",
]
