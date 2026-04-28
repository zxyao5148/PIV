"""VIV: Variational Instrumental Variable inference.

PyTorch adaptation of the structured-VAE approach from Li et al., supporting
all treatment types.  Learns latent Z (instrument) and C (confounder) from X
via reconstruction of (X, A, Y) with KL regularisation and an independence
penalty between Z and C.
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
                 activation: str = "elu") -> None:
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


class _GaussianEncoder(nn.Module):
    """Encoder that returns (mu, logvar) for a diagonal Gaussian."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], latent_dim: int,
                 activation: str = "elu") -> None:
        super().__init__()
        self.mu_net = _MLP(in_dim, hidden_dims, latent_dim, activation)
        self.logvar_net = _MLP(in_dim, hidden_dims, latent_dim, activation)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mu = self.mu_net(x)
        # Keep latent scale bounded to avoid exploding KL/reparameterized samples.
        logvar = self.logvar_net(x).clamp(min=-8.0, max=8.0)
        return mu, logvar

    @staticmethod
    def reparameterise(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp().clamp(max=10.0)
        return mu + torch.randn_like(mu) * std


def _pearson_abs(x: Tensor, y: Tensor) -> Tensor:
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    x_n = x / (x.norm(dim=0, keepdim=True) + 1e-8)
    y_n = y / (y.norm(dim=0, keepdim=True) + 1e-8)
    return ((x_n.T @ y_n) / x.shape[0]).abs().mean()


@dataclass
class VIVOutput:
    loss: Tensor
    metrics: Dict[str, Tensor]


class VIV(nn.Module):
    """Structured VAE that disentangles X into Z and C.

    Encoders:
      Z ~ q(z | x)     — instrument
      C ~ q(c | x)     — confounder
    Decoders:
      A ~ p(a | z, c)  — treatment reconstruction
      Y ~ p(y | c, a)  — outcome reconstruction
      X ~ p(x | z, c)  — observation reconstruction
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        z_dim: int,
        a_dim: int,
        y_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        hidden_dim: int = 128,
        activation: str = "elu",
        kl_weight: float = 0.1,
        indep_weight: float = 1.0,
        recon_a_weight: float = 1.0,
        recon_y_weight: float = 0.1,
        recon_x_weight: float = 0.1,
        use_proxy_embed: bool = False,
        m_dim: int = 1,
        proxy_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.treatment_family = treatment_family
        self.kl_w = kl_weight
        self.indep_w = indep_weight
        self.ra_w = recon_a_weight
        self.ry_w = recon_y_weight
        self.rx_w = recon_x_weight
        self.use_proxy_embed = use_proxy_embed
        self.m_dim = int(m_dim)
        self.proxy_embed_dim = int(proxy_embed_dim)
        self.c_dim = int(c_dim)

        h = [hidden_dim] * 2
        self.z_enc = _GaussianEncoder(x_dim, h, z_dim, activation)
        self.c_enc = _GaussianEncoder(x_dim, h, c_dim, activation)

        self.a_dec = _MLP(z_dim + c_dim, [hidden_dim], a_dim, activation)
        y_in = c_dim + (proxy_embed_dim if use_proxy_embed else 0) + a_dim
        self.y_dec = _MLP(y_in, [hidden_dim], y_dim, activation)
        self.x_dec = _MLP(z_dim + c_dim, [hidden_dim], x_dim, activation)
        self.proxy_mlp: nn.Module | None
        if use_proxy_embed:
            self.proxy_mlp = _MLP(self.m_dim, [hidden_dim], proxy_embed_dim, activation)
        else:
            self.proxy_mlp = None

    def encode_z(self, x: Tensor) -> Tensor:
        mu, _ = self.z_enc(x)
        return mu

    def encode_c(self, x: Tensor, m: Tensor | None = None) -> Tensor:
        mu, _ = self.c_enc(x)
        if not self.use_proxy_embed or self.proxy_mlp is None:
            return mu
        if m is None:
            m = torch.zeros(x.shape[0], self.m_dim, device=x.device, dtype=x.dtype)
        return torch.cat([mu, self.proxy_mlp(m)], dim=-1)

    def compute_loss(
        self, x: Tensor, a: Tensor, y: Tensor, m: Tensor | None = None,
    ) -> VIVOutput:
        z_mu, z_lv = self.z_enc(x)
        c_mu, c_lv = self.c_enc(x)
        z = _GaussianEncoder.reparameterise(z_mu, z_lv)
        c = _GaussianEncoder.reparameterise(c_mu, c_lv)

        a_pred = self.a_dec(torch.cat([z, c], -1))
        if self.use_proxy_embed:
            assert m is not None and self.proxy_mlp is not None
            c_y = torch.cat([c, self.proxy_mlp(m)], dim=-1)
        else:
            c_y = c
        y_pred = self.y_dec(torch.cat([c_y, a], -1))
        x_pred = self.x_dec(torch.cat([z, c], -1))

        if self.treatment_family == "continuous":
            a_recon = F.mse_loss(a_pred, a)
        else:
            a_recon = F.binary_cross_entropy_with_logits(a_pred, a.float())
        y_recon = F.mse_loss(y_pred, y)
        x_recon = F.mse_loss(x_pred, x)

        kl_z = -0.5 * (1 + z_lv - z_mu.pow(2) - z_lv.exp()).sum(-1).mean()
        kl_c = -0.5 * (1 + c_lv - c_mu.pow(2) - c_lv.exp()).sum(-1).mean()

        indep = _pearson_abs(z, c) ** 2

        loss = (self.ra_w * a_recon
                + self.ry_w * y_recon
                + self.rx_w * x_recon
                + self.kl_w * (kl_z + kl_c)
                + self.indep_w * indep)

        return VIVOutput(loss=loss, metrics={
            "viv_loss": loss.detach(),
            "viv_a_recon": a_recon.detach(),
        })


def build_viv(
    x_dim: int, c_dim: int, z_dim: int, a_dim: int,
    y_dim: int = 1,
    treatment_family: TreatmentFamily = "continuous",
    *,
    use_proxy_embed: bool = False,
    m_dim: int = 1,
    proxy_embed_dim: int = 4,
) -> VIV:
    return VIV(
        x_dim=x_dim, c_dim=c_dim, z_dim=z_dim, a_dim=a_dim,
        y_dim=y_dim, treatment_family=treatment_family,
        use_proxy_embed=use_proxy_embed,
        m_dim=m_dim,
        proxy_embed_dim=proxy_embed_dim,
    )


__all__ = ["VIV", "VIVOutput", "build_viv"]
