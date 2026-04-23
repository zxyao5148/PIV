from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TreatmentFamily = Literal["continuous", "binary"]


@dataclass
class Stage1Output:
    w: Tensor
    pos_scores: Tensor
    neg_scores: Tensor
    loss: Tensor
    metrics: Dict[str, Tensor]


@dataclass
class Stage2Output:
    w: Tensor
    z: Tensor
    prediction: Tensor
    logits: Optional[Tensor]
    loss: Tensor
    metrics: Dict[str, Tensor]


def get_activation(name: str) -> nn.Module:
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


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        activate_last: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers: List[nn.Module] = []
        act = get_activation(activation)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or activate_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act.__class__())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ProxyScoreHead(nn.Module):
    """Stage-1 score head s1(g_C(x), m)."""

    def __init__(
        self,
        w_dim: int,
        m_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = MLP(
            in_dim=w_dim + m_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, w: Tensor, m: Tensor) -> Tensor:
        return self.net(torch.cat([w, m], dim=-1)).squeeze(-1)


class TreatmentHead(nn.Module):
    """Stage-2 treatment head t(z, w, l)."""

    def __init__(
        self,
        z_dim: int,
        w_dim: int,
        l_dim: int,
        a_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = MLP(
            in_dim=z_dim + w_dim + l_dim,
            hidden_dims=hidden_dims,
            out_dim=a_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, z: Tensor, w: Tensor, l: Tensor) -> Tensor:
        return self.net(torch.cat([z, w, l], dim=-1))


class Stage1Model(nn.Module):
    """
    Proxy-assisted stage-1 model.

    Learns a confounding representation W = E_theta1(X) and a score head s1(W, M)
    using the contrastive objective from the draft.
    """

    def __init__(
        self,
        x_dim: int,
        m_dim: int,
        w_dim: int,
        encoder_hidden_dims: Sequence[int] = (256, 256),
        score_hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.m_dim = m_dim
        self.w_dim = w_dim

        self.encoder = MLP(
            in_dim=x_dim,
            hidden_dims=encoder_hidden_dims,
            out_dim=w_dim,
            activation=activation,
            dropout=dropout,
        )
        self.score_head = ProxyScoreHead(
            w_dim=w_dim,
            m_dim=m_dim,
            hidden_dims=score_hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    @staticmethod
    def default_negative_sampler(m: Tensor) -> Tensor:
        if m.shape[0] < 2:
            raise ValueError("Need batch size >= 2 to create negative proxy pairs.")
        perm = torch.randperm(m.shape[0], device=m.device)
        if torch.all(perm == torch.arange(m.shape[0], device=m.device)):
            perm = perm.roll(1)
        return m[perm]

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def score(self, x: Tensor, m: Tensor) -> Tensor:
        w = self.encode(x)
        return self.score_head(w, m)

    def loss(
        self,
        x: Tensor,
        m_pos: Tensor,
        m_neg: Optional[Tensor] = None,
    ) -> Stage1Output:
        if m_neg is None:
            m_neg = self.default_negative_sampler(m_pos)

        w = self.encode(x)
        pos_scores = self.score_head(w, m_pos)
        neg_scores = self.score_head(w, m_neg)

        loss_pos = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        loss = loss_pos + loss_neg

        metrics = {
            "stage1_loss": loss.detach(),
            "stage1_pos_score_mean": pos_scores.mean().detach(),
            "stage1_neg_score_mean": neg_scores.mean().detach(),
            "stage1_pos_prob_mean": torch.sigmoid(pos_scores).mean().detach(),
            "stage1_neg_prob_mean": torch.sigmoid(neg_scores).mean().detach(),
        }
        return Stage1Output(
            w=w,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            loss=loss,
            metrics=metrics,
        )


class Stage2Model(nn.Module):
    """
    Stage-2 treatment prediction model.

    The same module supports:
    - continuous treatment: MSE
    - binary treatment: BCE with logits
    """

    def __init__(
        self,
        x_dim: int,
        l_dim: int,
        w_dim: int,
        z_dim: int,
        a_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        encoder_hidden_dims: Sequence[int] = (256, 256),
        treatment_hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        freeze_w_encoder: bool = True,
    ) -> None:
        super().__init__()
        if treatment_family not in {"continuous", "binary"}:
            raise ValueError("treatment_family must be 'continuous' or 'binary'.")

        self.treatment_family = treatment_family
        self.a_dim = a_dim
        self.freeze_w_encoder = freeze_w_encoder

        self.w_encoder = MLP(
            in_dim=x_dim,
            hidden_dims=encoder_hidden_dims,
            out_dim=w_dim,
            activation=activation,
            dropout=dropout,
        )
        self.z_encoder = MLP(
            in_dim=x_dim,
            hidden_dims=encoder_hidden_dims,
            out_dim=z_dim,
            activation=activation,
            dropout=dropout,
        )
        self.treatment_head = TreatmentHead(
            z_dim=z_dim,
            w_dim=w_dim,
            l_dim=l_dim,
            a_dim=a_dim,
            hidden_dims=treatment_hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def load_stage1_encoder(self, stage1: Stage1Model, freeze: bool = True) -> None:
        self.w_encoder.load_state_dict(stage1.encoder.state_dict())
        self.freeze_w_encoder = freeze
        if freeze:
            for p in self.w_encoder.parameters():
                p.requires_grad_(False)

    def encode_w(self, x: Tensor) -> Tensor:
        if self.freeze_w_encoder:
            with torch.no_grad():
                return self.w_encoder(x)
        return self.w_encoder(x)

    def encode_z(self, x: Tensor) -> Tensor:
        return self.z_encoder(x)

    def forward(
        self,
        x: Tensor,
        l: Tensor,
        w: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if w is None:
            w = self.encode_w(x)
        z = self.encode_z(x)
        logits_or_mean = self.treatment_head(z, w, l)

        out: Dict[str, Tensor] = {
            "w": w,
            "z": z,
            "prediction": logits_or_mean,
        }
        if self.treatment_family == "binary":
            out["logits"] = logits_or_mean
            out["prediction"] = torch.sigmoid(logits_or_mean)
        return out

    def loss(
        self,
        x: Tensor,
        l: Tensor,
        a: Tensor,
        w: Optional[Tensor] = None,
    ) -> Stage2Output:
        outputs = self.forward(x=x, l=l, w=w)
        prediction = outputs["prediction"]
        logits = outputs.get("logits")

        a = a.float()
        if a.ndim == 1:
            a = a.unsqueeze(-1)

        if self.treatment_family == "continuous":
            loss = F.mse_loss(prediction, a)
            metrics = {
                "stage2_loss": loss.detach(),
                "stage2_pred_loss": loss.detach(),
                "stage2_rmse": torch.sqrt(loss.detach().clamp_min(1e-12)),
            }
        else:
            if logits is None:
                raise RuntimeError("Binary treatment requires logits.")
            loss = F.binary_cross_entropy_with_logits(logits, a)
            metrics = {
                "stage2_loss": loss.detach(),
                "stage2_pred_loss": loss.detach(),
                "stage2_pred_mean": prediction.mean().detach(),
            }

        return Stage2Output(
            w=outputs["w"],
            z=outputs["z"],
            prediction=prediction,
            logits=logits,
            loss=loss,
            metrics=metrics,
        )


class ProxyIVModel(nn.Module):
    """
    Convenience wrapper containing both stages.

    Recommended workflow:
    1. Train stage1 with `stage1_loss`.
    2. Call `copy_stage1_into_stage2(freeze=True)`.
    3. Train stage2 with `stage2_loss`.
    4. Use `encode_representations` to export W and Z into downstream NPIV.
    """

    def __init__(self, stage1: Stage1Model, stage2: Stage2Model) -> None:
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def copy_stage1_into_stage2(self, freeze: bool = True) -> None:
        self.stage2.load_stage1_encoder(self.stage1, freeze=freeze)

    def stage1_loss(
        self,
        x: Tensor,
        m_pos: Tensor,
        m_neg: Optional[Tensor] = None,
    ) -> Stage1Output:
        return self.stage1.loss(x=x, m_pos=m_pos, m_neg=m_neg)

    def stage2_loss(
        self,
        x: Tensor,
        l: Tensor,
        a: Tensor,
        w: Optional[Tensor] = None,
    ) -> Stage2Output:
        return self.stage2.loss(x=x, l=l, a=a, w=w)

    @torch.no_grad()
    def encode_representations(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        w = self.stage2.encode_w(x)
        z = self.stage2.encode_z(x)
        return w, z

    @torch.no_grad()
    def forward_representations(self, x: Tensor, l: Tensor) -> Dict[str, Tensor]:
        return self.stage2.forward(x=x, l=l)


def build_proxyiv(
    x_dim: int,
    m_dim: int,
    l_dim: int,
    w_dim: int,
    z_dim: int,
    a_dim: int = 1,
    treatment_family: TreatmentFamily = "continuous",
    activation: str = "relu",
) -> ProxyIVModel:
    stage1 = Stage1Model(
        x_dim=x_dim,
        m_dim=m_dim,
        w_dim=w_dim,
        encoder_hidden_dims=(256, 256),
        score_hidden_dims=(128, 128),
        activation=activation,
    )
    stage2 = Stage2Model(
        x_dim=x_dim,
        l_dim=l_dim,
        w_dim=w_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        treatment_family=treatment_family,
        encoder_hidden_dims=(256, 256),
        treatment_hidden_dims=(128, 128),
        activation=activation,
        freeze_w_encoder=True,
    )
    return ProxyIVModel(stage1=stage1, stage2=stage2)


__all__ = [
    "MLP",
    "ProxyScoreHead",
    "Stage1Model",
    "Stage1Output",
    "Stage2Model",
    "Stage2Output",
    "ProxyIVModel",
    "TreatmentHead",
    "TreatmentFamily",
    "build_proxyiv",
]
