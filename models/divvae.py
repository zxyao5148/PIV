"""DIV.VAE: Disentangled IV via Variational AutoEncoder.

PyTorch implementation of Cheng et al. (2024) "Disentangled Representation
Learning for Causal Inference with Instruments" (arXiv:2412.04641).

Two encoders map X to IV representation Z and confounder representation C.
A decoder reconstructs X from (Z, C).  Two auxiliary predictors enforce
that (Z, C) jointly predict treatment A and that C alone (with A) predicts
outcome Y — thereby enforcing the exclusion restriction (Z → Y only
through A).  An orthogonality-promoting regulariser (OPR) via cosine
similarity encourages Z ⊥ C.

Adapted to support both continuous and binary treatments.
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
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], latent_dim: int,
                 activation: str = "elu") -> None:
        super().__init__()
        self.mu_net = _MLP(in_dim, hidden_dims, latent_dim, activation)
        self.logvar_net = _MLP(in_dim, hidden_dims, latent_dim, activation)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.mu_net(x), self.logvar_net(x)

    @staticmethod
    def reparameterise(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


def _cosine_similarity_mean(z: Tensor, c: Tensor) -> Tensor:
    """Mean absolute cosine similarity across the batch (OPR term)."""
    z_n = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    c_n = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
    return (z_n * c_n).sum(dim=-1).abs().mean()


@dataclass
class DIVVAEOutput:
    loss: Tensor
    metrics: Dict[str, Tensor]


class DIVVAE(nn.Module):
    """DIV.VAE: Disentangled IV via VAE.

    Encoders:
      Z ~ q(z | x)     -- IV representation
      C ~ q(c | x)     -- confounder representation
    Decoder:
      X ~ p(x | z, c)  -- observation reconstruction
    Auxiliary predictors:
      A ~ q(a | z, c)  -- treatment (uses BOTH Z and C)
      Y ~ q(y | a, c)  -- outcome  (uses C only, NOT Z → exclusion restriction)
    Regularisation:
      KL(q(z|x) || N(0,I)) + KL(q(c|x) || N(0,I))
      OPR: cosine similarity between Z and C samples
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
        alpha_a: float = 100.0,
        alpha_y: float = 100.0,
        opr_weight: float = 1.0,
        use_proxy_embed: bool = False,
        m_dim: int = 1,
        proxy_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.treatment_family = treatment_family
        self.kl_w = kl_weight
        self.alpha_a = alpha_a
        self.alpha_y = alpha_y
        self.opr_w = opr_weight
        self.use_proxy_embed = use_proxy_embed
        self.m_dim = int(m_dim)
        self.proxy_embed_dim = int(proxy_embed_dim)
        self.c_dim = int(c_dim)

        h = [hidden_dim] * 2

        self.z_enc = _GaussianEncoder(x_dim, h, z_dim, activation)
        self.c_enc = _GaussianEncoder(x_dim, h, c_dim, activation)

        self.x_dec = _MLP(z_dim + c_dim, h, x_dim, activation)

        self.a_pred = _MLP(z_dim + c_dim, [hidden_dim], a_dim, activation)
        y_in = c_dim + (proxy_embed_dim if use_proxy_embed else 0) + a_dim
        self.y_pred = _MLP(y_in, [hidden_dim], y_dim, activation)
        self.proxy_mlp: nn.Module | None
        if use_proxy_embed:
            self.proxy_mlp = _MLP(self.m_dim, [hidden_dim], proxy_embed_dim, activation)
        else:
            self.proxy_mlp = None

    # ── Public API (compatible with extract_cz) ──────────────────

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

    # ── Loss ─────────────────────────────────────────────────────

    def compute_loss(
        self, x: Tensor, a: Tensor, y: Tensor, m: Tensor | None = None,
    ) -> DIVVAEOutput:
        z_mu, z_lv = self.z_enc(x)
        c_mu, c_lv = self.c_enc(x)
        z = _GaussianEncoder.reparameterise(z_mu, z_lv)
        c = _GaussianEncoder.reparameterise(c_mu, c_lv)

        x_recon = F.mse_loss(self.x_dec(torch.cat([z, c], -1)), x)

        kl_z = -0.5 * (1 + z_lv - z_mu.pow(2) - z_lv.exp()).sum(-1).mean()
        kl_c = -0.5 * (1 + c_lv - c_mu.pow(2) - c_lv.exp()).sum(-1).mean()

        a_hat = self.a_pred(torch.cat([z, c], -1))
        if self.treatment_family == "continuous":
            a_loss = F.mse_loss(a_hat, a)
        else:
            a_loss = F.binary_cross_entropy_with_logits(a_hat, a.float())

        if self.use_proxy_embed:
            assert m is not None and self.proxy_mlp is not None
            c_y = torch.cat([c, self.proxy_mlp(m)], dim=-1)
        else:
            c_y = c
        y_hat = self.y_pred(torch.cat([c_y, a], -1))
        y_loss = F.mse_loss(y_hat, y)

        opr = _cosine_similarity_mean(z, c)

        loss = (x_recon
                + self.kl_w * (kl_z + kl_c)
                + self.alpha_a * a_loss
                + self.alpha_y * y_loss
                + self.opr_w * opr)

        return DIVVAEOutput(loss=loss, metrics={
            "divvae_loss": loss.detach(),
            "x_recon": x_recon.detach(),
            "a_loss": a_loss.detach(),
            "y_loss": y_loss.detach(),
            "opr": opr.detach(),
        })


def build_divvae(
    x_dim: int, c_dim: int, z_dim: int, a_dim: int,
    y_dim: int = 1,
    treatment_family: TreatmentFamily = "continuous",
    *,
    use_proxy_embed: bool = False,
    m_dim: int = 1,
    proxy_embed_dim: int = 4,
) -> DIVVAE:
    return DIVVAE(
        x_dim=x_dim, c_dim=c_dim, z_dim=z_dim, a_dim=a_dim,
        y_dim=y_dim, treatment_family=treatment_family,
        use_proxy_embed=use_proxy_embed,
        m_dim=m_dim,
        proxy_embed_dim=proxy_embed_dim,
    )


__all__ = ["DIVVAE", "DIVVAEOutput", "build_divvae"]
