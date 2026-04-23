"""GIV (Meta-EM): IV generation via iterative domain clustering.

PyTorch adaptation from the Meta-EM implementation, supporting all treatment
types.  Learns a representation from X using an EM loop that alternates
between GMM-based domain clustering on (rep, A) and network training with
per-domain treatment prediction heads.

Unlike ZNet/AutoIV/VIV, GIV does **not** use Y — it is entirely unsupervised
w.r.t. outcome.
"""
from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TreatmentFamily = Literal["continuous", "binary"]


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int,
                 activation: str = "tanh") -> None:
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


class GIV(nn.Module):
    """Meta-EM model: shared encoder + per-domain prediction heads.

    Architecture (following the original):
      mapping     : X → rep   (MLP, tanh activations)
      predictors  : rep → Â_k for each domain k   (linear heads)
      reconstruct : rep → X̂   (MLP, ReLU activations)

    The forward pass selects the prediction from the active domain via
    one-hot ``domain`` weights.
    """

    def __init__(
        self,
        x_dim: int,
        rep_dim: int,
        a_dim: int,
        n_domains: int = 3,
        treatment_family: TreatmentFamily = "continuous",
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.treatment_family = treatment_family
        self.a_dim = a_dim
        self.n_domains = n_domains
        self.x_dim = x_dim

        self.mapping = _MLP(x_dim, [hidden_dim], rep_dim, "tanh")
        self.predictors = nn.ModuleList(
            [nn.Linear(rep_dim, a_dim) for _ in range(n_domains)]
        )
        self.reconstruct = _MLP(rep_dim, [hidden_dim, hidden_dim], x_dim, "relu")

    # ── Public API (compatible with extract_cz) ───────────────────

    def encode_z(self, x: Tensor) -> Tensor:
        return self.mapping(x)

    def encode_c(self, x: Tensor) -> Tensor:
        return x.new_zeros(x.shape[0], 0)

    def forward(
        self, x: Tensor, domain_onehot: Tensor,
    ) -> tuple[Tensor, Tensor]:
        rep = self.mapping(x)
        recon = self.reconstruct(rep)
        preds = torch.stack([p(rep) for p in self.predictors], dim=1)
        pred = (preds * domain_onehot.unsqueeze(-1)).sum(1)
        return recon, pred

    def compute_step_loss(
        self, x: Tensor, a: Tensor, domain_onehot: Tensor,
    ) -> Tensor:
        recon, pred = self.forward(x, domain_onehot)
        if self.treatment_family == "continuous":
            pred_loss = F.mse_loss(pred, a)
        else:
            pred_loss = F.binary_cross_entropy_with_logits(pred, a.float())
        recon_loss = F.mse_loss(recon, x)
        return pred_loss + recon_loss * self.x_dim


def build_giv(
    x_dim: int, rep_dim: int, a_dim: int,
    n_domains: int = 3,
    treatment_family: TreatmentFamily = "continuous",
) -> GIV:
    return GIV(
        x_dim=x_dim, rep_dim=rep_dim, a_dim=a_dim,
        n_domains=n_domains, treatment_family=treatment_family,
    )


__all__ = ["GIV", "build_giv"]
