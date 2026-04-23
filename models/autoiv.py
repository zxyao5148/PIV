"""AutoIV: Automatic IV generation via variational mutual-information bounds.

PyTorch adaptation from the TensorFlow 1.x implementation, supporting all
treatment types.  Learns rep_C and rep_Z from X using MI-based objectives
with (A, Y) supervision.

Training uses three separate optimisers on the **full batch** (no mini-
batching), matching the original TF code:
  1. opt_lld   → MI estimator networks  (density fitting)
  2. opt_bound → representation encoders only  (MI bound objectives)
  3. opt_2stage → representation + prediction networks  (treatment / outcome MSE)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TreatmentFamily = Literal["continuous", "binary"]


def _small_init(module: nn.Module) -> None:
    """Match the original TF init: Normal(0, 0.1/sqrt(fan_out))."""
    if isinstance(module, nn.Linear):
        fan_out = module.weight.shape[0]
        std = 0.1 / math.sqrt(fan_out)
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


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


class _VarMI(nn.Module):
    """Variational MI estimator (matches the original TF ``mi_net``).

    Returns (neg-log-likelihood, signed contrastive bound).
    """

    def __init__(self, inp_dim: int, out_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mu_net = _MLP(inp_dim, [hidden_dim], out_dim, "elu")
        self.logvar_net = _MLP(inp_dim, [hidden_dim], out_dim, "elu")

    def forward(self, inp: Tensor, target: Tensor, sign: float = -1.0,
                weight: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mu = self.mu_net(inp)
        logvar = torch.tanh(self.logvar_net(inp))
        var = logvar.exp()

        lld = -(-0.5 * ((target - mu) ** 2 / var + logvar).sum(-1)).mean()

        perm = torch.randperm(target.shape[0], device=target.device)
        pos = -((mu - target) ** 2 / var).sum(-1)
        neg = -((mu - target[perm]) ** 2 / var).sum(-1)

        diff = pos - neg
        if weight is None:
            w = 1.0 / target.shape[0]
        else:
            w = weight
        bound = sign * (w * diff).sum()

        return lld, bound


@dataclass
class AutoIVOutput:
    loss_lld: Tensor
    loss_bound: Tensor
    loss_2stage: Tensor
    metrics: Dict[str, Tensor]


class AutoIV(nn.Module):
    """AutoIV learns rep_Z and rep_C from X via MI objectives.

    Three loss groups (matching the original TF implementation):
      loss_lld    – density-fitting for MI networks (detached representations)
      loss_bound  – MI bound objectives for representation encoders only
      loss_2stage – treatment + outcome prediction for rep + prediction heads
    """

    def __init__(
        self,
        x_dim: int,
        rep_dim: int,
        a_dim: int,
        y_dim: int = 1,
        treatment_family: TreatmentFamily = "continuous",
        hidden_dim: int = 128,
        activation: str = "elu",
        coef_lld_za: float = 1.0,
        coef_lld_zy: float = 1.0,
        coef_lld_ca: float = 1.0,
        coef_lld_cy: float = 1.0,
        coef_lld_zc: float = 1.0,
        coef_bound_za: float = 1.0,
        coef_bound_zy: float = 1.0,
        coef_bound_ca: float = 1.0,
        coef_bound_cy: float = 1.0,
        coef_bound_zc: float = 1.0,
        coef_pred_t: float = 1.0,
        coef_pred_y: float = 1.0,
        coef_reg: float = 0.001,
        sigma: float = 0.1,
        use_proxy_embed: bool = False,
        m_dim: int = 1,
        proxy_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.treatment_family = treatment_family
        self.a_dim = a_dim
        self.sigma = sigma
        self.coef_reg = coef_reg
        self.use_proxy_embed = use_proxy_embed
        self.m_dim = int(m_dim)
        self.proxy_embed_dim = int(proxy_embed_dim)

        self.coef_lld = dict(za=coef_lld_za, zy=coef_lld_zy, ca=coef_lld_ca,
                             cy=coef_lld_cy, zc=coef_lld_zc)
        self.coef_bnd = dict(za=coef_bound_za, zy=coef_bound_zy, ca=coef_bound_ca,
                             cy=coef_bound_cy, zc=coef_bound_zc)
        self.coef_pred_t = coef_pred_t
        self.coef_pred_y = coef_pred_y

        self.z_encoder = _MLP(x_dim, [hidden_dim] * 2, rep_dim, activation)
        self.c_encoder = _MLP(x_dim, [hidden_dim] * 2, rep_dim, activation)
        self.t_head = _MLP(rep_dim * 2, [hidden_dim], a_dim, activation)
        self.t_emb = _MLP(a_dim, [hidden_dim], rep_dim, activation)
        y_in = rep_dim * 2 + (proxy_embed_dim if use_proxy_embed else 0)
        self.y_head = _MLP(y_in, [hidden_dim], y_dim, activation)
        self.proxy_mlp: nn.Module | None
        if use_proxy_embed:
            self.proxy_mlp = _MLP(self.m_dim, [hidden_dim], proxy_embed_dim, activation)
        else:
            self.proxy_mlp = None

        mi_hidden = max(rep_dim // 2, 16)
        self.mi_za = _VarMI(rep_dim, a_dim, mi_hidden)
        self.mi_zy = _VarMI(rep_dim, y_dim, mi_hidden)
        self.mi_ca = _VarMI(rep_dim, a_dim, mi_hidden)
        self.mi_cy = _VarMI(rep_dim, y_dim, mi_hidden)
        self.mi_zc = _VarMI(rep_dim, rep_dim, mi_hidden)

        self.apply(_small_init)

    # ── Parameter groups (matching original var lists) ────────────

    def mi_parameters(self):
        """vars_lld: MI estimator networks."""
        mods = [self.mi_za, self.mi_zy, self.mi_ca, self.mi_cy, self.mi_zc]
        return [p for m in mods for p in m.parameters()]

    def rep_parameters(self):
        """vars_bound: representation encoders only."""
        return list(self.z_encoder.parameters()) + list(self.c_encoder.parameters())

    def stage_parameters(self):
        """vars_2stage: rep encoders + prediction heads."""
        mods = [self.z_encoder, self.c_encoder,
                self.t_head, self.t_emb, self.y_head]
        if self.proxy_mlp is not None:
            mods.append(self.proxy_mlp)
        return [p for m in mods for p in m.parameters()]

    # ── Public API ────────────────────────────────────────────────

    def encode_c(self, x: Tensor, m: Tensor | None = None) -> Tensor:
        rc = self.c_encoder(x)
        if not self.use_proxy_embed or self.proxy_mlp is None:
            return rc
        if m is None:
            m = torch.zeros(x.shape[0], self.m_dim, device=x.device, dtype=x.dtype)
        return torch.cat([rc, self.proxy_mlp(m)], dim=-1)

    def encode_z(self, x: Tensor) -> Tensor:
        return self.z_encoder(x)

    def _l2_reg(self, params) -> Tensor:
        return sum(p.pow(2).sum() for p in params) * 0.5

    def compute_loss(
        self, x: Tensor, a: Tensor, y: Tensor, m: Tensor | None = None,
    ) -> AutoIVOutput:
        rz = self.encode_z(x)
        rc = self.c_encoder(x)

        # ── 1) LLD loss (MI density fitting, representations detached) ──
        lld_za, _ = self.mi_za(rz.detach(), a, sign=-1.0)
        lld_zy, _ = self.mi_zy(rz.detach(), y, sign=1.0)
        lld_ca, _ = self.mi_ca(rc.detach(), a, sign=-1.0)
        lld_cy, _ = self.mi_cy(rc.detach(), y, sign=-1.0)
        lld_zc, _ = self.mi_zc(rz.detach(), rc.detach(), sign=1.0)

        loss_lld = (self.coef_lld["za"] * lld_za
                    + self.coef_lld["zy"] * lld_zy
                    + self.coef_lld["ca"] * lld_ca
                    + self.coef_lld["cy"] * lld_cy
                    + self.coef_lld["zc"] * lld_zc)

        # ── 2) Bound loss (MI objectives for rep encoders) ──────────
        # Conditional weighting for MI(Z, Y | A) using RBF kernel
        with torch.no_grad():
            perm = torch.randperm(a.shape[0], device=a.device)
            a_rand = a[perm]
            w = torch.exp(-((a - a_rand) ** 2).sum(-1) / (2 * self.sigma ** 2))
            w_zy = torch.softmax(w, dim=0)

        _, bnd_za = self.mi_za(rz, a, sign=-1.0)
        _, bnd_zy = self.mi_zy(rz, y, sign=1.0, weight=w_zy)
        _, bnd_ca = self.mi_ca(rc, a, sign=-1.0)
        _, bnd_cy = self.mi_cy(rc, y, sign=-1.0)
        _, bnd_zc = self.mi_zc(rz, rc, sign=1.0)

        reg_rep = self._l2_reg(self.rep_parameters())
        loss_bound = (self.coef_bnd["za"] * bnd_za
                      + self.coef_bnd["zy"] * bnd_zy
                      + self.coef_bnd["ca"] * bnd_ca
                      + self.coef_bnd["cy"] * bnd_cy
                      + self.coef_bnd["zc"] * bnd_zc
                      + self.coef_reg * reg_rep)

        # ── 3) 2-stage prediction loss ──────────────────────────────
        t_pred = self.t_head(torch.cat([rz, rc], -1))
        if self.treatment_family == "continuous":
            t_loss = F.mse_loss(t_pred, a)
        else:
            t_loss = F.binary_cross_entropy_with_logits(t_pred, a.float())

        t_emb = self.t_emb(t_pred.detach())
        if self.use_proxy_embed:
            assert m is not None and self.proxy_mlp is not None
            rc_y = torch.cat([rc, self.proxy_mlp(m)], dim=-1)
        else:
            rc_y = rc
        y_pred = self.y_head(torch.cat([rc_y, t_emb], -1))
        y_loss = F.mse_loss(y_pred, y)

        reg_stage = self._l2_reg(self.stage_parameters())
        loss_2stage = (self.coef_pred_t * t_loss
                       + self.coef_pred_y * y_loss
                       + self.coef_reg * reg_stage)

        return AutoIVOutput(
            loss_lld=loss_lld,
            loss_bound=loss_bound,
            loss_2stage=loss_2stage,
            metrics={
                "loss_2stage": loss_2stage.detach(),
                "loss_lld": loss_lld.detach(),
                "loss_bound": loss_bound.detach(),
            },
        )


def build_autoiv(
    x_dim: int, rep_dim: int, a_dim: int,
    y_dim: int = 1,
    treatment_family: TreatmentFamily = "continuous",
    *,
    use_proxy_embed: bool = False,
    m_dim: int = 1,
    proxy_embed_dim: int = 4,
) -> AutoIV:
    return AutoIV(
        x_dim=x_dim, rep_dim=rep_dim, a_dim=a_dim,
        y_dim=y_dim, treatment_family=treatment_family,
        use_proxy_embed=use_proxy_embed,
        m_dim=m_dim,
        proxy_embed_dim=proxy_embed_dim,
    )


__all__ = ["AutoIV", "AutoIVOutput", "build_autoiv"]
