"""ZNet: Disentangled IV representation learning via multi-objective optimization.

Adapted from Fields et al. to PyTorch with support for all treatment types.
Learns C (confounder) and Z (instrument) from X using (A, Y) supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TreatmentFamily = Literal["continuous", "binary"]


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int,
                 activation: str = "relu") -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _pearson_abs(x: Tensor, y: Tensor) -> Tensor:
    """Mean |Pearson r| across all column pairs of x and y."""
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    x_n = x / (x.norm(dim=0, keepdim=True) + 1e-8)
    y_n = y / (y.norm(dim=0, keepdim=True) + 1e-8)
    corr = (x_n.T @ y_n) / x.shape[0]
    return corr.abs().mean()


@dataclass
class ZNetOutput:
    loss: Tensor
    metrics: Dict[str, Tensor]


class ZNet(nn.Module):
    """ZNet learns C and Z from X; Z predicts treatment, C predicts outcome.

    Training requires (X, A, Y).  Optionally pass proxy ``M`` so an MLP
    embedding is concatenated to the learned confounder for the outcome
    head and for :meth:`encode_c` (NPIV ``W``).
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        z_dim: int,
        a_dim: int,
        y_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        hidden_dim: int = 64,
        activation: str = "relu",
        t_hat_alpha: float = 1.0,
        c_y_corr_alpha: float = 1.0,
        z_resid_alpha: float = 1.0,
        cz_indep_alpha: float = 0.5,
        cy_mse_alpha: float = 1.0,
        use_proxy_embed: bool = False,
        m_dim: int = 1,
        proxy_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.treatment_family = treatment_family
        self.a_dim = a_dim
        self.t_hat_alpha = t_hat_alpha
        self.c_y_corr_alpha = c_y_corr_alpha
        self.z_resid_alpha = z_resid_alpha
        self.cz_indep_alpha = cz_indep_alpha
        self.cy_mse_alpha = cy_mse_alpha
        self.use_proxy_embed = use_proxy_embed
        self.m_dim = int(m_dim)
        self.proxy_embed_dim = int(proxy_embed_dim)
        self.c_dim = int(c_dim)

        self.c_encoder = _MLP(x_dim, [hidden_dim], c_dim, activation)
        self.z_encoder = _MLP(x_dim, [hidden_dim], z_dim, activation)
        self.t_head = _MLP(z_dim, [hidden_dim], a_dim, activation)
        cy_in = c_dim + (proxy_embed_dim if use_proxy_embed else 0) + a_dim
        self.cy_head = _MLP(cy_in, [hidden_dim], y_dim, activation)
        self.xty_head = _MLP(x_dim + a_dim, [hidden_dim], y_dim, activation)
        self.proxy_mlp: nn.Module | None
        if use_proxy_embed:
            self.proxy_mlp = _MLP(self.m_dim, [hidden_dim], proxy_embed_dim, activation)
        else:
            self.proxy_mlp = None

    # ── Public API ────────────────────────────────────────────────

    def encode_c(self, x: Tensor, m: Tensor | None = None) -> Tensor:
        c0 = self.c_encoder(x)
        if not self.use_proxy_embed or self.proxy_mlp is None:
            return c0
        if m is None:
            m = torch.zeros(x.shape[0], self.m_dim, device=x.device, dtype=x.dtype)
        return torch.cat([c0, self.proxy_mlp(m)], dim=-1)

    def encode_z(self, x: Tensor) -> Tensor:
        return self.z_encoder(x)

    def pretrain_xty(
        self, x: Tensor, a: Tensor, y: Tensor,
        epochs: int = 200, lr: float = 1e-3,
    ) -> None:
        """Pre-train the (X, A) → Y residual model, then freeze it."""
        opt = torch.optim.Adam(self.xty_head.parameters(), lr=lr)
        xa = torch.cat([x, a], dim=-1)
        self.xty_head.train()
        for _ in range(epochs):
            loss = F.mse_loss(self.xty_head(xa), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for p in self.xty_head.parameters():
            p.requires_grad_(False)

    def compute_loss(
        self, x: Tensor, a: Tensor, y: Tensor, m: Tensor | None = None,
    ) -> ZNetOutput:
        c0 = self.c_encoder(x)
        z = self.encode_z(x)
        if self.use_proxy_embed:
            assert m is not None and self.proxy_mlp is not None
            c = torch.cat([c0, self.proxy_mlp(m)], dim=-1)
        else:
            c = c0

        t_pred = self.t_head(z)
        if self.treatment_family == "continuous":
            t_loss = F.mse_loss(t_pred, a)
        else:
            t_loss = F.binary_cross_entropy_with_logits(t_pred, a.float())

        cy_pred = self.cy_head(torch.cat([c, a], dim=-1))
        cy_mse = F.mse_loss(cy_pred, y)

        with torch.no_grad():
            resid = y - self.xty_head(torch.cat([x, a], dim=-1))

        c_y_corr = 1.0 - _pearson_abs(c0, y) ** 2
        z_resid = _pearson_abs(z, resid) ** 2
        cz_indep = _pearson_abs(c0, z) ** 2

        loss = (self.t_hat_alpha * t_loss
                + self.c_y_corr_alpha * c_y_corr
                + self.z_resid_alpha * z_resid
                + self.cz_indep_alpha * cz_indep
                + self.cy_mse_alpha * cy_mse)

        return ZNetOutput(loss=loss, metrics={
            "znet_loss": loss.detach(),
            "znet_t_loss": t_loss.detach(),
        })


def build_znet(
    x_dim: int, c_dim: int, z_dim: int, a_dim: int,
    y_dim: int = 1,
    treatment_family: TreatmentFamily = "continuous",
    *,
    use_proxy_embed: bool = False,
    m_dim: int = 1,
    proxy_embed_dim: int = 4,
) -> ZNet:
    return ZNet(
        x_dim=x_dim, c_dim=c_dim, z_dim=z_dim, a_dim=a_dim,
        y_dim=y_dim, treatment_family=treatment_family,
        use_proxy_embed=use_proxy_embed,
        m_dim=m_dim,
        proxy_embed_dim=proxy_embed_dim,
    )


__all__ = ["ZNet", "ZNetOutput", "build_znet"]
