"""Major training logic for IV-representation + NPIV experiments.

This module owns every function that actually runs an optimisation loop:

- Per-method IV-representation loops
    * :func:`train_stage1`, :func:`train_stage2`    — PIV two-stage ICA
    * :func:`pretrain_znet_xty`                     — ZNet (X, A)→Y residual head
    * :func:`train_baseline`                        — ZNet / VIV / DIV.VAE
    * :func:`train_autoiv`                          — AutoIV (three optimisers)
    * :func:`train_giv`                             — GIV Meta-EM

- NPIV loops
    * :func:`train_deepiv`                          — DeepIV (S1 density + S2 response)
    * :func:`train_dfiv`                            — Deep Feature IV (ψ / φ / ξ)

- Orchestration (checkpoint-aware dispatch over methods/estimators)
    * :func:`train_iv_representation`, :func:`load_iv_representation`
    * :func:`build_npiv_model`, :func:`fit_npiv`, :func:`load_npiv_checkpoint`

Small helpers (dimensions, loader wrappers, checkpoint fingerprints, kwargs
builders, normalisation stats, representation extraction) live in :mod:`utils`.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from synthetic_dataset import DataLoaderBundle
from models.ablation import build_proxy_orth
from models.autoiv import build_autoiv
from models.divvae import build_divvae
from models.npiv_estimators import DeepIV, DFIV
from models.proxyiv import build_proxyiv
from models.viv import build_viv
from models.znet import build_znet
from utils import (
    IVDims,
    collect_loader,
    empty_covariates,
    extract_cz,
    extract_wz,
    fingerprint_iv,
    fingerprint_npiv,
    load_checkpoint,
    load_npiv_norms,
    make_eval_loaders,
    make_iv_checkpoint,
    npiv_w_dim,
    proxy_embed_kwargs,
    representation_digest,
    sample_noneiv_noise,
    wl_norm_stats,
)

# ═══════════════════════════════════════════════════════════════════
#  IV-representation training loops (PIV, ZNet, baselines, AutoIV, GIV)
# ═══════════════════════════════════════════════════════════════════


def train_stage1(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    val_loader: DataLoader | None = None,
    patience: int = 0,
    log_every: int = 20,
) -> None:
    """PIV stage-1: proxy-contrastive objective on ``(X, M)``."""
    opt = torch.optim.Adam(model.stage1.parameters(), lr=lr)
    model.stage1.train()
    best_state: dict[str, Tensor] | None = None
    best_val = float("inf")
    bad_epochs = 0
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            x = batch["X"].to(device)
            m = batch["M"].to(device)
            out = model.stage1_loss(x, m)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            epoch_loss += out.loss.item() * x.shape[0]
        val_msg = ""
        if val_loader is not None:
            model.stage1.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["X"].to(device)
                    m = batch["M"].to(device)
                    out = model.stage1_loss(x, m)
                    val_loss += out.loss.item() * x.shape[0]
            val_loss = val_loss / len(val_loader.dataset)
            val_msg = f"  val_loss={val_loss:.4f}"
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.stage1.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
            model.stage1.train()
        if ep % log_every == 0 or ep == 1:
            print(
                f"  Stage-1  epoch {ep:3d}/{epochs}  "
                f"loss={epoch_loss / len(loader.dataset):.4f}{val_msg}",
            )
        if val_loader is not None and patience > 0 and bad_epochs >= patience:
            print(
                f"  Stage-1  early stop at epoch {ep} "
                f"(best_val={best_val:.4f}, patience={patience})",
            )
            break
    if best_state is not None:
        model.stage1.load_state_dict(best_state)


def train_stage2(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    val_loader: DataLoader | None = None,
    patience: int = 0,
    l_dim: int = 0,
    log_every: int = 20,
) -> None:
    """PIV stage-2: treatment prediction from ``(W, L)``."""
    trainable = [p for p in model.stage2.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    model.stage2.train()
    best_state: dict[str, Tensor] | None = None
    best_val = float("inf")
    bad_epochs = 0
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            x = batch["X"].to(device)
            a = batch["A"].to(device)
            l = batch["L"].to(device) if "L" in batch else empty_covariates(x.shape[0], device)
            out = model.stage2_loss(x, l, a)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            epoch_loss += out.loss.item() * x.shape[0]
        val_msg = ""
        if val_loader is not None:
            model.stage2.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["X"].to(device)
                    a = batch["A"].to(device)
                    l = (
                        batch["L"].to(device)
                        if "L" in batch
                        else empty_covariates(x.shape[0], device)
                    )
                    out = model.stage2_loss(x, l, a)
                    val_loss += out.loss.item() * x.shape[0]
            val_loss = val_loss / len(val_loader.dataset)
            val_msg = f"  val_loss={val_loss:.4f}"
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.stage2.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
            model.stage2.train()
        if ep % log_every == 0 or ep == 1:
            print(
                f"  Stage-2  epoch {ep:3d}/{epochs}  "
                f"loss={epoch_loss / len(loader.dataset):.4f}{val_msg}",
            )
        if val_loader is not None and patience > 0 and bad_epochs >= patience:
            print(
                f"  Stage-2  early stop at epoch {ep} "
                f"(best_val={best_val:.4f}, patience={patience})",
            )
            break
    if best_state is not None:
        model.stage2.load_state_dict(best_state)


def pretrain_znet_xty(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
) -> None:
    """Pre-train (and freeze) ZNet's ``(X, A) → Y`` residualiser head.

    ZNet's main loss uses ``y − xty_head(x, a)`` as a target for the instrument
    representation, which is only meaningful when ``xty_head`` is already a
    decent regressor; this routine fits it once before the main run.
    """
    xs, as_, ys = [], [], []
    for batch in loader:
        xs.append(batch["X"])
        as_.append(batch["A"])
        ys.append(batch["Y"])
    x_all = torch.cat(xs).to(device)
    a_all = torch.cat(as_).to(device)
    y_all = torch.cat(ys).to(device)
    model.pretrain_xty(x_all, a_all, y_all, epochs=epochs, lr=lr)


def train_baseline(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    log_every: int = 20,
    label: str = "Baseline",
    grad_clip: float = 0.0,
) -> None:
    """Generic training loop for models with a single ``compute_loss(x, a, y[, m])``.

    Covers ZNet / VIV / DIV.VAE — ``compute_loss`` returns an object with a
    single ``.loss`` attribute.  Picks up the proxy ``M`` automatically when
    ``model.use_proxy_embed`` is true.
    """
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )
    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            x = batch["X"].to(device)
            a = batch["A"].to(device)
            y = batch["Y"].to(device)
            if getattr(model, "use_proxy_embed", False):
                m = batch["M"].to(device)
                out = model.compute_loss(x, a, y, m)
            else:
                out = model.compute_loss(x, a, y)
            if not torch.isfinite(out.loss):
                raise RuntimeError(
                    f"{label} produced non-finite loss at epoch {ep}",
                )
            opt.zero_grad()
            out.loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )
            opt.step()
            epoch_loss += out.loss.item() * x.shape[0]
        if ep % log_every == 0 or ep == 1:
            print(
                f"  {label}  epoch {ep:3d}/{epochs}  "
                f"loss={epoch_loss / len(loader.dataset):.4f}",
            )


def train_proxy_orth(
    model: nn.Module,
    loader: DataLoader,
    s1_epochs: int,
    s2_epochs: int,
    lr: float,
    device: torch.device,
    *,
    val_loader: DataLoader | None = None,
    s1_patience: int = 0,
    s2_patience: int = 0,
    log_every: int = 20,
) -> None:
    """Two-stage training loop for Proxy-Orth."""

    def run_stage(
        *,
        stage: int,
        epochs: int,
        patience: int,
        label: str,
    ) -> None:
        if epochs <= 0:
            return
        if stage == 1:
            model.set_stage1_trainable()
        else:
            model.set_stage2_trainable()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        model.train()
        best_state: dict[str, Tensor] | None = None
        best_val = float("inf")
        bad_epochs = 0
        for ep in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                x = batch["X"].to(device)
                m = batch["M"].to(device)
                l = (
                    batch["L"].to(device)
                    if "L" in batch
                    else empty_covariates(x.shape[0], device)
                )
                if stage == 1:
                    out = model.compute_loss_stage1(x, m, l)
                else:
                    a = batch["A"].to(device)
                    out = model.compute_loss_stage2(x, m, l, a)
                opt.zero_grad()
                out.loss.backward()
                opt.step()
                epoch_loss += out.loss.item() * x.shape[0]
            val_msg = ""
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["X"].to(device)
                        m = batch["M"].to(device)
                        l = (
                            batch["L"].to(device)
                            if "L" in batch
                            else empty_covariates(x.shape[0], device)
                        )
                        if stage == 1:
                            out = model.compute_loss_stage1(x, m, l)
                        else:
                            a = batch["A"].to(device)
                            out = model.compute_loss_stage2(x, m, l, a)
                        val_loss += out.loss.item() * x.shape[0]
                val_loss = val_loss / len(val_loader.dataset)
                val_msg = f"  val_loss={val_loss:.4f}"
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                model.train()
            if ep % log_every == 0 or ep == 1:
                print(
                    f"  Proxy-Orth {label}  epoch {ep:3d}/{epochs}  "
                    f"loss={epoch_loss / len(loader.dataset):.4f}{val_msg}",
                )
            if val_loader is not None and patience > 0 and bad_epochs >= patience:
                print(
                    f"  Proxy-Orth {label}  early stop at epoch {ep} "
                    f"(best_val={best_val:.4f}, patience={patience})",
                )
                break
        if best_state is not None:
            model.load_state_dict(best_state)

    run_stage(stage=1, epochs=s1_epochs, patience=s1_patience, label="stage 1")
    run_stage(stage=2, epochs=s2_epochs, patience=s2_patience, label="stage 2")


@torch.no_grad()
def extract_proxy_orth_wz(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Extract ``(W, Z)`` from Proxy-Orth; uses residualized proxy block as W."""
    model.eval()
    ws, zs = [], []
    for batch in loader:
        x = batch["X"].to(device)
        m = batch["M"].to(device)
        l = (
            batch["L"].to(device)
            if "L" in batch
            else empty_covariates(x.shape[0], device)
        )
        out = model.forward(x, m, l)
        ws.append(out["x_hat"].cpu())
        zs.append(out["z"].cpu())
    return torch.cat(ws), torch.cat(zs)


def train_autoiv(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    log_every: int = 20,
    lr_decay_step: int = 100,
    lr_decay_rate: float = 0.95,
) -> None:
    """Full-batch, three-optimiser training for AutoIV (mirrors the original TF).

    The TF reference feeds the entire training set each step (no mini-batching);
    with three adversarial optimisers (MI density, MI bound, two-stage
    prediction) this is critical for stable convergence.
    """
    xs, as_, ys, ms = [], [], [], []
    for batch in loader:
        xs.append(batch["X"])
        as_.append(batch["A"])
        ys.append(batch["Y"])
        ms.append(batch["M"])
    x_all = torch.cat(xs).to(device)
    a_all = torch.cat(as_).to(device)
    y_all = torch.cat(ys).to(device)
    m_all = torch.cat(ms).to(device)

    opt_lld = torch.optim.Adam(model.mi_parameters(), lr=lr)
    opt_bound = torch.optim.Adam(model.rep_parameters(), lr=lr)
    opt_2stage = torch.optim.Adam(model.stage_parameters(), lr=lr)

    sched_lld = torch.optim.lr_scheduler.StepLR(opt_lld, lr_decay_step, lr_decay_rate)
    sched_bound = torch.optim.lr_scheduler.StepLR(opt_bound, lr_decay_step, lr_decay_rate)
    sched_2stage = torch.optim.lr_scheduler.StepLR(opt_2stage, lr_decay_step, lr_decay_rate)

    model.train()
    for ep in range(1, epochs + 1):
        # Step 1: MI density fitting
        out = (
            model.compute_loss(x_all, a_all, y_all, m_all)
            if getattr(model, "use_proxy_embed", False)
            else model.compute_loss(x_all, a_all, y_all)
        )
        opt_lld.zero_grad()
        out.loss_lld.backward()
        opt_lld.step()

        # Step 2: MI bound → rep encoders only
        out = (
            model.compute_loss(x_all, a_all, y_all, m_all)
            if getattr(model, "use_proxy_embed", False)
            else model.compute_loss(x_all, a_all, y_all)
        )
        opt_bound.zero_grad()
        out.loss_bound.backward()
        opt_bound.step()

        # Step 3: prediction → rep + prediction heads
        out = (
            model.compute_loss(x_all, a_all, y_all, m_all)
            if getattr(model, "use_proxy_embed", False)
            else model.compute_loss(x_all, a_all, y_all)
        )
        opt_2stage.zero_grad()
        out.loss_2stage.backward()
        opt_2stage.step()

        sched_lld.step()
        sched_bound.step()
        sched_2stage.step()

        if ep % log_every == 0 or ep == 1:
            print(
                f"  AutoIV  epoch {ep:3d}/{epochs}  "
                f"loss={out.metrics['loss_2stage'].item():.4f}",
            )


def train_giv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    em_iters: int = 20,
    inner_epochs: int = 200,
    batch_size: int = 400,
    lr: float = 1e-3,
    log_every_em: int = 5,
) -> None:
    """EM-loop training for GIV (Meta-EM).

    Alternates between GMM clustering on ``(rep, A)`` and network training
    with per-domain treatment prediction heads.  Does **not** use ``Y``.
    """
    from sklearn.mixture import GaussianMixture

    xs, as_ = [], []
    for batch in loader:
        xs.append(batch["X"])
        as_.append(batch["A"])
    x_all = torch.cat(xs)
    a_all = torch.cat(as_)
    n = x_all.shape[0]

    x_np = x_all.numpy()
    a_np = a_all.numpy()

    merge = np.concatenate([x_np, a_np], axis=1)
    gmm = GaussianMixture(
        n_components=model.n_domains, covariance_type="full", random_state=0,
    )
    clusters = gmm.fit_predict(merge).reshape(-1)

    x_dev = x_all.to(device)
    a_dev = a_all.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for em in range(1, em_iters + 1):
        domain_oh = torch.nn.functional.one_hot(
            torch.from_numpy(clusters.astype(np.int64)), model.n_domains,
        ).float().to(device)

        model.train()
        for _ in range(inner_epochs):
            idx = random.sample(range(n), min(batch_size, n))
            loss = model.compute_step_loss(x_dev[idx], a_dev[idx], domain_oh[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            rep = model.encode_z(x_dev).cpu().numpy()
        merge_rt = np.concatenate([rep, a_np], axis=1)
        gmm = GaussianMixture(
            n_components=model.n_domains, covariance_type="full", random_state=0,
        )
        clusters = gmm.fit_predict(merge_rt).reshape(-1)

        if em % log_every_em == 0 or em == 1:
            model.eval()
            with torch.no_grad():
                l = model.compute_step_loss(x_dev, a_dev, domain_oh).item()
            print(f"  GIV  EM iter {em:2d}/{em_iters}  loss={l:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  NPIV training loops (DeepIV / DFIV)
# ═══════════════════════════════════════════════════════════════════


def train_deepiv(
    deepiv: DeepIV,
    a: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    l: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
    *,
    a_val_raw: np.ndarray | None = None,
    z_val_raw: np.ndarray | None = None,
    w_val_raw: np.ndarray | None = None,
    l_val_raw: np.ndarray | None = None,
    y_val_raw: np.ndarray | None = None,
    batch_size: int = 256,
    log_every: int = 30,
    num_samples: int = 64,
    weight_decay: float = 1e-4,
    val_frac: float = 0.1,
    patience: int = 20,
    normalize_wl: bool = True,
    rng: np.random.Generator | None = None,
    grad_clip: float = 1.0,
) -> dict:
    """Train DeepIV with held-out early stopping in both stages.

    Stage-1 uses the held-out split to early-stop on treatment-density NLL.
    Stage-2 uses the same held-out split for early stopping with fixed
    Monte-Carlo seeds so the validation loss is a deterministic function of
    weights (noise-free early stopping without leaking training randomness).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    treatment_family = str(getattr(deepiv, "treatment_family", "continuous"))
    if treatment_family == "binary":
        a_arr_bin = np.asarray(a, dtype=np.float32)
        if a_arr_bin.ndim == 1:
            a_arr_bin = a_arr_bin.reshape(-1, 1)
        a_n = a_arr_bin
        if a_arr_bin.shape[1] > 1:
            a_mean: float | np.ndarray = np.zeros(a_arr_bin.shape[1], dtype=np.float64)
            a_std: float | np.ndarray = np.ones(a_arr_bin.shape[1], dtype=np.float64)
        else:
            a_mean = 0.0
            a_std = 1.0
    else:
        a_arr = np.asarray(a)
        if a_arr.ndim > 1 and a_arr.shape[1] > 1:
            a_mean = a_arr.mean(axis=0)        # (dA,)
            a_std = a_arr.std(axis=0) + 1e-8   # (dA,)
        else:
            a_mean = float(a_arr.mean())
            a_std = float(a_arr.std()) + 1e-8
        a_n = ((a_arr - a_mean) / a_std).astype(np.float32)
    y_mean = float(y.mean()); y_std = float(y.std()) + 1e-8
    w_mean, w_std, l_mean, l_std = wl_norm_stats(w, l, enabled=normalize_wl)
    z_mean = z.mean(axis=0); z_std = z.std(axis=0) + 1e-8

    y_n = ((y - y_mean) / y_std).astype(np.float32)
    w_n = ((w - w_mean) / w_std).astype(np.float32)
    l_n = ((l - l_mean) / l_std).astype(np.float32)
    z_n = ((z - z_mean) / z_std).astype(np.float32)

    has_external_val = all(
        arr is not None
        for arr in (a_val_raw, z_val_raw, w_val_raw, l_val_raw, y_val_raw)
    )
    n = a_n.shape[0]
    if has_external_val:
        n_val = int(np.asarray(a_val_raw).shape[0])
        tr_idx = np.arange(n)
        val_idx = np.empty((0,), dtype=np.int64)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

    def _ds(idx: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.as_tensor(a_n[idx]),
            torch.as_tensor(z_n[idx], dtype=torch.float32),
            torch.as_tensor(w_n[idx], dtype=torch.float32),
            torch.as_tensor(l_n[idx], dtype=torch.float32),
            torch.as_tensor(y_n[idx]),
        )

    train_ds = _ds(tr_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    if has_external_val:
        if treatment_family == "binary":
            a_val_arr = np.asarray(a_val_raw, dtype=np.float32)
            if a_val_arr.ndim == 1:
                a_val_arr = a_val_arr.reshape(-1, 1)
            a_val_n = a_val_arr
        else:
            a_val_n = ((np.asarray(a_val_raw) - a_mean) / a_std).astype(np.float32)
        z_val_n = ((np.asarray(z_val_raw) - z_mean) / z_std).astype(np.float32)
        w_val_n = ((np.asarray(w_val_raw) - w_mean) / w_std).astype(np.float32)
        l_val_n = ((np.asarray(l_val_raw) - l_mean) / l_std).astype(np.float32)
        y_val_n = ((np.asarray(y_val_raw).reshape(-1, 1) - y_mean) / y_std).astype(np.float32)
        val_ds = TensorDataset(
            torch.as_tensor(a_val_n),
            torch.as_tensor(z_val_n, dtype=torch.float32),
            torch.as_tensor(w_val_n, dtype=torch.float32),
            torch.as_tensor(l_val_n, dtype=torch.float32),
            torch.as_tensor(y_val_n),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        val_loader = (
            DataLoader(_ds(val_idx), batch_size=batch_size, shuffle=False)
            if n_val > 0 else None
        )

    # ── Stage 1: treatment density net ──
    treat_opt = torch.optim.Adam(deepiv.treatment_net.parameters(), lr=lr)
    sched_treat = torch.optim.lr_scheduler.CosineAnnealingLR(
        treat_opt, T_max=epochs, eta_min=lr * 1e-2,
    )

    @torch.no_grad()
    def _val_treatment_nll() -> float:
        if val_loader is None:
            return float("nan")
        deepiv.eval()
        total, count = 0.0, 0
        for a_b, z_b, w_b, l_b, _ in val_loader:
            a_b, z_b, w_b, l_b = (t.to(device) for t in (a_b, z_b, w_b, l_b))
            out = deepiv.treatment_loss(a_b, z_b, w_b, l_b)
            total += out.nll.item() * a_b.shape[0]
            count += a_b.shape[0]
        deepiv.train()
        return total / max(count, 1)

    deepiv.train()
    best_treat_val = float("inf")
    best_treat_state: dict[str, Tensor] | None = None
    treat_epochs_since_improve = 0
    for ep in range(1, epochs + 1):
        ep_nll = 0.0
        for a_b, z_b, w_b, l_b, _ in train_loader:
            a_b, z_b, w_b, l_b = (t.to(device) for t in (a_b, z_b, w_b, l_b))
            out = deepiv.treatment_loss(a_b, z_b, w_b, l_b)
            treat_opt.zero_grad()
            out.nll.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(deepiv.treatment_net.parameters(), grad_clip)
            treat_opt.step()
            ep_nll += out.nll.item() * a_b.shape[0]
        sched_treat.step()

        v_nll = _val_treatment_nll() if val_loader is not None else float("nan")
        if val_loader is not None:
            if v_nll < best_treat_val - 1e-6:
                best_treat_val = v_nll
                best_treat_state = {
                    k: vv.detach().cpu().clone()
                    for k, vv in deepiv.treatment_net.state_dict().items()
                }
                treat_epochs_since_improve = 0
            else:
                treat_epochs_since_improve += 1

        if ep % log_every == 0 or ep == 1:
            msg = f"    [DeepIV S1] epoch {ep:3d}/{epochs}  nll={ep_nll / len(train_ds):.4f}"
            if val_loader is not None:
                msg += f"  val_nll={v_nll:.4f}  best={best_treat_val:.4f}"
            print(msg)

        if val_loader is not None and patience > 0 and treat_epochs_since_improve >= patience:
            print(
                f"    [DeepIV S1] early stop at epoch {ep} "
                f"(no val improvement for {patience} epochs, best_val={best_treat_val:.4f})",
            )
            break

    if best_treat_state is not None:
        deepiv.treatment_net.load_state_dict(best_treat_state)

    for p in deepiv.treatment_net.parameters():
        p.requires_grad_(False)

    # ── Stage 2: response net with weight decay + early stopping ──
    resp_opt = torch.optim.Adam(
        deepiv.response_net.parameters(), lr=lr, weight_decay=weight_decay,
    )
    sched_resp = torch.optim.lr_scheduler.CosineAnnealingLR(
        resp_opt, T_max=epochs, eta_min=lr * 1e-2,
    )

    # Fixed seed used ONLY for the MC draws inside ``_val_loss`` so validation
    # is a deterministic function of response-net weights across epochs. The
    # surrounding torch RNG state is saved/restored; training sampling unaffected.
    val_mc_seed = int(rng.integers(0, 2**31 - 1))
    val_on_cuda = device.type == "cuda"

    def _val_loss() -> float:
        if val_loader is None:
            return float("nan")
        deepiv.eval()
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if val_on_cuda else None
        torch.manual_seed(val_mc_seed)
        if val_on_cuda:
            torch.cuda.manual_seed_all(val_mc_seed)
        try:
            total, count = 0.0, 0
            with torch.no_grad():
                for a_b, z_b, w_b, l_b, y_b in val_loader:
                    a_b, z_b, w_b, l_b, y_b = (
                        t.to(device) for t in (a_b, z_b, w_b, l_b, y_b)
                    )
                    out = deepiv.outcome_loss(
                        y_b, z_b, w_b, l_b, num_samples=num_samples,
                    )
                    total += out.loss.item() * a_b.shape[0]
                    count += a_b.shape[0]
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        deepiv.train()
        return total / max(count, 1)

    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_since_improve = 0

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for a_b, z_b, w_b, l_b, y_b in train_loader:
            a_b, z_b, w_b, l_b, y_b = (t.to(device) for t in (a_b, z_b, w_b, l_b, y_b))
            out = deepiv.outcome_loss(y_b, z_b, w_b, l_b, num_samples=num_samples)
            resp_opt.zero_grad()
            out.loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(deepiv.response_net.parameters(), grad_clip)
            resp_opt.step()
            ep_loss += out.loss.item() * a_b.shape[0]
        sched_resp.step()
        tr_mse = ep_loss / len(train_ds)
        v = _val_loss() if val_loader is not None else float("nan")

        if val_loader is not None:
            if v < best_val - 1e-6:
                best_val = v
                best_state = {
                    k: vv.detach().cpu().clone()
                    for k, vv in deepiv.response_net.state_dict().items()
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        if ep % log_every == 0 or ep == 1:
            msg = f"    [DeepIV S2] epoch {ep:3d}/{epochs}  tr_mse={tr_mse:.4f}"
            if val_loader is not None:
                msg += f"  val_mse={v:.4f}  best={best_val:.4f}"
            print(msg)

        if val_loader is not None and patience > 0 and epochs_since_improve >= patience:
            print(
                f"    [DeepIV S2] early stop at epoch {ep} "
                f"(no val improvement for {patience} epochs, best_val={best_val:.4f})",
            )
            break

    if best_state is not None:
        deepiv.response_net.load_state_dict(best_state)

    for p in deepiv.treatment_net.parameters():
        p.requires_grad_(True)

    @torch.no_grad()
    def _collect_deepiv_preds(
        a_eval: np.ndarray,
        w_eval: np.ndarray,
        l_eval: np.ndarray,
        batch_size_eval: int = 2048,
    ) -> np.ndarray:
        preds: list[np.ndarray] = []
        n_eval = int(np.asarray(a_eval).shape[0])
        for lo in range(0, n_eval, batch_size_eval):
            hi = min(lo + batch_size_eval, n_eval)
            a_slice = a_eval[lo:hi]
            if a_slice.ndim == 1:
                a_slice = a_slice.reshape(-1, 1)
            a_b = torch.as_tensor(a_slice, dtype=torch.float32, device=device)
            w_b = torch.as_tensor(w_eval[lo:hi], dtype=torch.float32, device=device)
            l_b = torch.as_tensor(l_eval[lo:hi], dtype=torch.float32, device=device)
            preds.append(deepiv.predict(a_b, w_b, l_b).detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)

    if has_external_val:
        a_cal = a_val_n
        w_cal = w_val_n
        l_cal = l_val_n
        y_cal = y_val_n.reshape(-1)
    elif val_idx.size > 0:
        a_cal = a_n[val_idx]
        w_cal = w_n[val_idx]
        l_cal = l_n[val_idx]
        y_cal = y_n[val_idx].reshape(-1)
    else:
        a_cal = a_n
        w_cal = w_n
        l_cal = l_n
        y_cal = y_n.reshape(-1)
    y_cal_pred = _collect_deepiv_preds(a_cal, w_cal, l_cal)
    y_cal_scale, y_cal_bias = _fit_affine_calibration(y_cal_pred, y_cal)

    return {
        "a_mean": a_mean, "a_std": a_std,
        "y_mean": y_mean, "y_std": y_std,
        "w_mean": w_mean.astype(np.float32), "w_std": w_std.astype(np.float32),
        "l_mean": l_mean.astype(np.float32), "l_std": l_std.astype(np.float32),
        "y_cal_scale": y_cal_scale, "y_cal_bias": y_cal_bias,
        "normalize_wl": bool(normalize_wl),
        "treatment_family": treatment_family,
    }


def train_dfiv(
    model: DFIV,
    a: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    l: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
    *,
    a_val_raw: np.ndarray | None = None,
    z_val_raw: np.ndarray | None = None,
    w_val_raw: np.ndarray | None = None,
    l_val_raw: np.ndarray | None = None,
    y_val_raw: np.ndarray | None = None,
    batch_size: int = 256,
    log_every: int = 30,
    lr_phi: float = 1e-4,
    weight_decay: float = 1e-4,
    weight_decay_psi: float | None = None,
    weight_decay_phi: float | None = None,
    val_frac: float = 0.1,
    patience: int = 20,
    normalize_wl: bool = True,
    rng: np.random.Generator | None = None,
    grad_clip: float = 1.0,
) -> dict:
    """Train :class:`DFIV` with alternating stage-1 (ψ) / stage-2 (ξ, φ) updates.

    Per-batch schedule (empirically robust across all four treatment types):

    1. Stage-1 on the first half of the training data: update ψ to minimise
       ``||φ(A) − V ψ(Z,W,L)||²`` with ``V`` via closed-form ridge and
       ``φ`` detached.
    2. Stage-2 on the second half: update ξ on the outcome objective
       ``||Y − μᵀ((V·ψ) ⊗ ξ)||²`` with ``V``/ ``φ``/``ψ`` detached and μ
       via closed-form ridge.
    3. Stage-2 on the second half: update φ on the **same** outcome
       objective at a much smaller learning rate (``lr_phi``).  A gentle
       φ step lets the treatment basis adapt to non-linear outcome
       shapes without destabilising the closed-form ``V``/``μ`` solves.
       Pass ``lr_phi = 0`` to freeze φ as a random ReLU basis.

    ``weight_decay`` (default ``1e-4``) is applied to all three
    feature-net optimisers.  DFIV's ℓ₂ penalty lives on the closed-form
    ``V`` / ``μ`` coefficients, so ``ψ`` / ``ξ`` / ``φ`` themselves are
    otherwise unregularised — and the stage-1 loss is homogeneous under
    ``ψ → c·ψ, V → V/c``, giving the optimiser a free direction to grow
    ``||ψ||`` unboundedly while ``V`` shrinks to compensate.  A tiny AdamW
    decay anchors ``ψ`` and is empirically crucial on the demands DGP:
    with ``wd = 0`` contrast MSE is ≈ 1800, with ``wd = 1e-4`` it drops
    to ≈ 190 (matching DeepIV).  ``run_npiv_simple.py`` is unaffected
    because its simpler DGPs do not stress the homogeneous direction.

    After training, ``μ̂`` is recomputed on the **full** training set via
    :meth:`DFIV.compute_final_mu` so prediction does not depend on any
    mini-batch μ estimate. The returned ``norms`` dict mirrors the one
    produced by :func:`train_deepiv` (same keys), letting
    :func:`evaluation.predict_hat_a` work unchanged for both downstreams.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    treatment_family = str(getattr(model, "treatment_family", "continuous"))
    if treatment_family == "binary":
        a_arr_bin = np.asarray(a, dtype=np.float32)
        if a_arr_bin.ndim == 1:
            a_arr_bin = a_arr_bin.reshape(-1, 1)
        a_n = a_arr_bin
        if a_arr_bin.shape[1] > 1:
            a_mean: float | np.ndarray = np.zeros(a_arr_bin.shape[1], dtype=np.float64)
            a_std: float | np.ndarray = np.ones(a_arr_bin.shape[1], dtype=np.float64)
        else:
            a_mean = 0.0
            a_std = 1.0
    else:
        a_arr = np.asarray(a)
        if a_arr.ndim > 1 and a_arr.shape[1] > 1:
            a_mean = a_arr.mean(axis=0)        # (dA,)
            a_std = a_arr.std(axis=0) + 1e-8   # (dA,)
        else:
            a_mean = float(a_arr.mean())
            a_std = float(a_arr.std()) + 1e-8
        a_n = ((a_arr - a_mean) / a_std).astype(np.float32)
    y_mean = float(y.mean()); y_std = float(y.std()) + 1e-8
    w_mean, w_std, l_mean, l_std = wl_norm_stats(w, l, enabled=normalize_wl)
    z_mean = z.mean(axis=0); z_std = z.std(axis=0) + 1e-8

    y_n = ((y - y_mean) / y_std).astype(np.float32)
    w_n = ((w - w_mean) / w_std).astype(np.float32)
    l_n = ((l - l_mean) / l_std).astype(np.float32)
    z_n = ((z - z_mean) / z_std).astype(np.float32)

    has_external_val = all(
        arr is not None
        for arr in (a_val_raw, z_val_raw, w_val_raw, l_val_raw, y_val_raw)
    )
    n = a_n.shape[0]
    if has_external_val:
        n_val = int(np.asarray(a_val_raw).shape[0])
        tr_idx = np.arange(n)
        val_idx = np.empty((0,), dtype=np.int64)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if tr_idx.size < 2:
            tr_idx = perm
            val_idx = perm[:1]

    use_full_batch = not (batch_size and int(batch_size) > 0)
    if use_full_batch:
        idx1 = tr_idx.copy()
        idx2 = tr_idx.copy()
    else:
        half = tr_idx.size // 2
        idx1, idx2 = tr_idx[:half], tr_idx[half:]
        if idx1.size == 0 or idx2.size == 0:
            idx1 = tr_idx[: max(1, tr_idx.size // 2)]
            idx2 = tr_idx[max(1, tr_idx.size // 2):]
        if idx2.size == 0:
            idx2 = idx1.copy()

    def _ds(idx: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.as_tensor(a_n[idx]),
            torch.as_tensor(z_n[idx], dtype=torch.float32),
            torch.as_tensor(w_n[idx], dtype=torch.float32),
            torch.as_tensor(l_n[idx], dtype=torch.float32),
            torch.as_tensor(y_n[idx]),
        )

    bs = int(batch_size) if batch_size and batch_size > 0 else max(int(tr_idx.size), 1)
    loader1 = DataLoader(_ds(idx1), batch_size=bs, shuffle=True)
    loader2 = DataLoader(_ds(idx2), batch_size=bs, shuffle=True)

    a_tr_full = torch.as_tensor(a_n[tr_idx], dtype=torch.float32, device=device)
    z_tr_full = torch.as_tensor(z_n[tr_idx], dtype=torch.float32, device=device)
    w_tr_full = torch.as_tensor(w_n[tr_idx], dtype=torch.float32, device=device)
    l_tr_full = torch.as_tensor(l_n[tr_idx], dtype=torch.float32, device=device)
    y_tr_full = torch.as_tensor(y_n[tr_idx], dtype=torch.float32, device=device)

    if has_external_val:
        if treatment_family == "binary":
            a_val_arr2 = np.asarray(a_val_raw, dtype=np.float32)
            if a_val_arr2.ndim == 1:
                a_val_arr2 = a_val_arr2.reshape(-1, 1)
            _a_val_np = a_val_arr2
        else:
            _a_val_np = ((np.asarray(a_val_raw) - a_mean) / a_std).astype(np.float32)
        a_val = torch.as_tensor(
            _a_val_np,
            dtype=torch.float32,
            device=device,
        )
        z_val = torch.as_tensor(
            ((np.asarray(z_val_raw) - z_mean) / z_std).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        w_val = torch.as_tensor(
            ((np.asarray(w_val_raw) - w_mean) / w_std).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        l_val = torch.as_tensor(
            ((np.asarray(l_val_raw) - l_mean) / l_std).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        y_val = torch.as_tensor(
            ((np.asarray(y_val_raw).reshape(-1, 1) - y_mean) / y_std).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
    else:
        a_val = (
            torch.as_tensor(a_n[val_idx], dtype=torch.float32, device=device)
            if n_val > 0 else None
        )
        z_val = (
            torch.as_tensor(z_n[val_idx], dtype=torch.float32, device=device)
            if n_val > 0 else None
        )
        w_val = (
            torch.as_tensor(w_n[val_idx], dtype=torch.float32, device=device)
            if n_val > 0 else None
        )
        l_val = (
            torch.as_tensor(l_n[val_idx], dtype=torch.float32, device=device)
            if n_val > 0 else None
        )
        y_val = (
            torch.as_tensor(y_n[val_idx], dtype=torch.float32, device=device)
            if n_val > 0 else None
        )

    wd = float(weight_decay)
    wd_psi = float(weight_decay_psi) if weight_decay_psi is not None else wd
    wd_phi = float(weight_decay_phi) if weight_decay_phi is not None else wd
    opt_psi = torch.optim.Adam(model.psi.parameters(), lr=lr, weight_decay=wd_psi)
    opt_xi = torch.optim.Adam(model.xi.parameters(), lr=lr, weight_decay=wd)
    opt_phi = (
        torch.optim.Adam(model.phi.parameters(), lr=float(lr_phi), weight_decay=wd_phi)
        if lr_phi and lr_phi > 0 else None
    )
    sched_psi = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_psi, T_max=epochs, eta_min=lr * 1e-2,
    )
    sched_xi = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_xi, T_max=epochs, eta_min=lr * 1e-2,
    )
    sched_phi = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_phi, T_max=epochs, eta_min=float(lr_phi) * 1e-2,
        ) if opt_phi is not None else None
    )

    model.to(device).train()
    last_stats: dict[str, float] = {}
    best_val = float("inf")
    best_state: dict[str, dict[str, Tensor]] | None = None
    epochs_since_improve = 0

    @torch.no_grad()
    def _val_loss() -> float:
        if (
            a_val is None or z_val is None or w_val is None
            or l_val is None or y_val is None
        ):
            return float("nan")
        # Mirror the training stage-2 loss: use V·ψ(Z_val) as features,
        # not φ(A_val). φ(A_val) would condition on the confounded treatment,
        # which biases early stopping toward the confounded fit.
        psi_tr = model.features_psi(z_tr_full, w_tr_full, l_tr_full)
        phi_tr = model.features_phi(a_tr_full)
        xi_tr = model.features_xi(w_tr_full, l_tr_full)
        V = model.compute_V(psi_tr, phi_tr)
        mu = model.compute_mu(V, psi_tr, xi_tr, y_tr_full)
        psi_v = model.features_psi(z_val, w_val, l_val)
        xi_v = model.features_xi(w_val, l_val)
        V_psi_v = (V @ psi_v.transpose(0, 1)).transpose(0, 1)
        X_v = model.batch_outer_vec(V_psi_v, xi_v)
        pred = X_v @ mu
        return float(torch.mean((y_val.reshape(-1) - pred) ** 2).item())

    for ep in range(1, epochs + 1):
        ep_psi = ep_xi = ep_phi = 0.0
        n_psi = n_xi = n_phi = 0
        for b1, b2 in zip(loader1, loader2):
            a1, z1, w1, l1, _ = (t.to(device) for t in b1)
            a2, z2, w2, l2, y2 = (t.to(device) for t in b2)

            # ── Stage 1: update ψ, φ detached ──
            psi1 = model.features_psi(z1, w1, l1)
            phi1 = model.features_phi(a1)
            V1 = model.compute_V(psi1, phi1)
            loss_psi = model.loss_psi(V1, psi1, phi1.detach())
            opt_psi.zero_grad(); loss_psi.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.psi.parameters(), grad_clip)
            opt_psi.step()
            ep_psi += loss_psi.item() * a1.shape[0]; n_psi += a1.shape[0]

            # ── Stage 2a: update ξ (ψ, φ, V detached) ──
            psi2 = model.features_psi(z2, w2, l2).detach()
            phi2 = model.features_phi(a2).detach()
            xi2 = model.features_xi(w2, l2)
            V2 = model.compute_V(psi2, phi2).detach()
            mu2 = model.compute_mu(V2, psi2, xi2, y2)
            loss_xi = model.loss_outcome(mu2, V2, psi2, xi2, y2)
            opt_xi.zero_grad(); loss_xi.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.xi.parameters(), grad_clip)
            opt_xi.step()
            ep_xi += loss_xi.item() * a2.shape[0]; n_xi += a2.shape[0]

            # ── Stage 2b (optional): gentle φ step ──
            if opt_phi is not None:
                psi2p = model.features_psi(z2, w2, l2).detach()
                phi2p = model.features_phi(a2)
                xi2p = model.features_xi(w2, l2).detach()
                V2p = model.compute_V(psi2p, phi2p)
                mu2p = model.compute_mu(V2p, psi2p, xi2p, y2)
                loss_phi = model.loss_outcome(mu2p, V2p, psi2p, xi2p, y2)
                opt_phi.zero_grad(); loss_phi.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.phi.parameters(), grad_clip)
                opt_phi.step()
                ep_phi += loss_phi.item() * a2.shape[0]; n_phi += a2.shape[0]

        sched_psi.step()
        sched_xi.step()
        if sched_phi is not None:
            sched_phi.step()

        last_stats = {
            "psi": ep_psi / max(n_psi, 1),
            "xi": ep_xi / max(n_xi, 1),
            "phi": ep_phi / max(n_phi, 1) if n_phi > 0 else float("nan"),
        }
        v = _val_loss() if n_val > 0 else float("nan")
        if n_val > 0:
            if v < best_val - 1e-6:
                best_val = v
                best_state = {
                    "psi": {
                        k: vv.detach().cpu().clone()
                        for k, vv in model.psi.state_dict().items()
                    },
                    "phi": {
                        k: vv.detach().cpu().clone()
                        for k, vv in model.phi.state_dict().items()
                    },
                    "xi": {
                        k: vv.detach().cpu().clone()
                        for k, vv in model.xi.state_dict().items()
                    },
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        if ep % log_every == 0 or ep == 1:
            msg = (
                f"    [DFIV] epoch {ep:3d}/{epochs}  "
                f"psi={last_stats['psi']:.4f}  xi={last_stats['xi']:.4f}"
            )
            if opt_phi is not None:
                msg += f"  phi={last_stats['phi']:.4f}"
            if n_val > 0:
                msg += f"  val_mse={v:.4f}  best={best_val:.4f}"
            print(msg)

        if n_val > 0 and patience > 0 and epochs_since_improve >= patience:
            print(
                f"    [DFIV] early stop at epoch {ep} "
                f"(no val improvement for {patience} epochs, best_val={best_val:.4f})",
            )
            break

    if best_state is not None:
        model.psi.load_state_dict(best_state["psi"])
        model.phi.load_state_dict(best_state["phi"])
        model.xi.load_state_dict(best_state["xi"])

    # ── Freeze μ on the full training data ──
    a_full = torch.as_tensor(a_n, device=device)
    z_full = torch.as_tensor(z_n, dtype=torch.float32, device=device)
    w_full = torch.as_tensor(w_n, dtype=torch.float32, device=device)
    l_full = torch.as_tensor(l_n, dtype=torch.float32, device=device)
    y_full = torch.as_tensor(y_n, device=device)
    model.compute_final_mu(a_full, z_full, w_full, l_full, y_full)

    @torch.no_grad()
    def _collect_dfiv_preds(
        a_eval: np.ndarray,
        w_eval: np.ndarray,
        l_eval: np.ndarray,
        batch_size_eval: int = 2048,
    ) -> np.ndarray:
        preds: list[np.ndarray] = []
        n_eval = int(np.asarray(a_eval).shape[0])
        for lo in range(0, n_eval, batch_size_eval):
            hi = min(lo + batch_size_eval, n_eval)
            a_slice = a_eval[lo:hi]
            if a_slice.ndim == 1:
                a_slice = a_slice.reshape(-1, 1)
            a_b = torch.as_tensor(a_slice, dtype=torch.float32, device=device)
            w_b = torch.as_tensor(w_eval[lo:hi], dtype=torch.float32, device=device)
            l_b = torch.as_tensor(l_eval[lo:hi], dtype=torch.float32, device=device)
            preds.append(model.predict(a_b, w_b, l_b).detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)

    if has_external_val:
        if treatment_family == "binary":
            a_cal_arr = np.asarray(a_val_raw, dtype=np.float32)
            if a_cal_arr.ndim == 1:
                a_cal_arr = a_cal_arr.reshape(-1, 1)
            a_cal = a_cal_arr
        else:
            a_cal = ((np.asarray(a_val_raw) - a_mean) / a_std).astype(np.float32)
        w_cal = ((np.asarray(w_val_raw) - w_mean) / w_std).astype(np.float32)
        l_cal = ((np.asarray(l_val_raw) - l_mean) / l_std).astype(np.float32)
        y_cal = ((np.asarray(y_val_raw).reshape(-1) - y_mean) / y_std).astype(np.float32)
    elif val_idx.size > 0:
        a_cal = a_n[val_idx]
        w_cal = w_n[val_idx]
        l_cal = l_n[val_idx]
        y_cal = y_n[val_idx].reshape(-1)
    else:
        a_cal = a_n
        w_cal = w_n
        l_cal = l_n
        y_cal = y_n.reshape(-1)
    y_cal_pred = _collect_dfiv_preds(a_cal, w_cal, l_cal)
    y_cal_scale, y_cal_bias = _fit_affine_calibration(y_cal_pred, y_cal)

    return {
        "a_mean": a_mean, "a_std": a_std,
        "y_mean": y_mean, "y_std": y_std,
        "w_mean": w_mean.astype(np.float32), "w_std": w_std.astype(np.float32),
        "l_mean": l_mean.astype(np.float32), "l_std": l_std.astype(np.float32),
        "y_cal_scale": y_cal_scale, "y_cal_bias": y_cal_bias,
        "normalize_wl": bool(normalize_wl),
        "treatment_family": treatment_family,
    }


# ═══════════════════════════════════════════════════════════════════
#  Orchestration: IV representation train / load
# ═══════════════════════════════════════════════════════════════════


def train_iv_representation(
    method: str,
    loaders: DataLoaderBundle,
    args: argparse.Namespace,
    dims: IVDims,
    device: torch.device,
    rng: np.random.Generator,
    seed: int,
    *,
    eval_loaders: DataLoaderBundle | None = None,
    fingerprint_extra: dict | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
    """Train an IV representation method and return train/val/test reps + ckpt.

    Training uses ``loaders.train`` (shuffled); the final ``(W, Z)`` extraction
    and any auxiliary collection go through ``eval_loaders`` (``shuffle=False``)
    so rows stay aligned with the canonical split ordering.

    Supported methods: ``PIV``, ``Proxy-Orth``, ``ZNet``, ``AutoIV``,
    ``VIV``, ``DIV.VAE``, ``TrueIV`` (uses simulator ``C, V``), and
    ``NoneIV`` (random noise).
    """
    eval_ld = (
        eval_loaders if eval_loaders is not None
        else make_eval_loaders(loaders)
    )

    if method == "PIV":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        piv = build_proxyiv(
            x_dim=dims.x_dim, m_dim=dims.m_dim, l_dim=dims.l_dim,
            w_dim=dims.w_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
            treatment_family=treatment_family,
        ).to(device)
        train_stage1(
            piv,
            loaders.train,
            args.s1_epochs,
            args.lr,
            device,
            val_loader=eval_ld.val,
            patience=int(getattr(args, "s1_patience", 0)),
        )
        piv.copy_stage1_into_stage2(freeze=True)
        train_stage2(
            piv,
            loaders.train,
            args.s2_epochs,
            args.lr,
            device,
            val_loader=eval_ld.val,
            patience=int(getattr(args, "s2_patience", 0)),
            l_dim=dims.l_dim,
        )
        w_tr, z_tr = extract_wz(piv, eval_ld.train, device)[:2]
        w_va, z_va = extract_wz(piv, eval_ld.val, device)[:2]
        w_te, z_te = extract_wz(piv, eval_ld.test, device)[:2]
        ckpt = make_iv_checkpoint(method, args, seed, {"state_dict": piv.state_dict()}, fingerprint_extra=fingerprint_extra)
        return w_tr.cpu(), z_tr.cpu(), w_va.cpu(), z_va.cpu(), w_te.cpu(), z_te.cpu(), ckpt

    if method == "Proxy-Orth":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        base_lr = float(getattr(args, "lr", 1e-3) or 1e-3)
        model = build_proxy_orth(
            x_dim=dims.x_dim,
            m_dim=dims.m_dim,
            l_dim=dims.l_dim,
            z_dim=dims.z_dim,
            a_dim=dims.a_dim,
            treatment_family=treatment_family,
        ).to(device)
        s1_epochs = int(
            getattr(args, "s1_epochs", None)
            or getattr(args, "epochs", None)
            or 200
        )
        s2_epochs = int(
            getattr(args, "s2_epochs", None)
            or getattr(args, "epochs", None)
            or 200
        )
        train_proxy_orth(
            model,
            loaders.train,
            s1_epochs,
            s2_epochs,
            base_lr,
            device,
            val_loader=eval_ld.val,
            s1_patience=int(getattr(args, "s1_patience", 0) or 0),
            s2_patience=int(getattr(args, "s2_patience", 0) or 0),
        )
        w_tr, z_tr = extract_proxy_orth_wz(model, eval_ld.train, device)
        w_va, z_va = extract_proxy_orth_wz(model, eval_ld.val, device)
        w_te, z_te = extract_proxy_orth_wz(model, eval_ld.test, device)
        ckpt = make_iv_checkpoint(
            method,
            args,
            seed,
            {"state_dict": model.state_dict()},
            fingerprint_extra=fingerprint_extra,
        )
        return (
            w_tr.cpu(),
            z_tr.cpu(),
            w_va.cpu(),
            z_va.cpu(),
            w_te.cpu(),
            z_te.cpu(),
            ckpt,
        )

    if method == "ZNet":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        znet = build_znet(
            x_dim=dims.x_dim, c_dim=dims.c_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
            y_dim=dims.y_dim, treatment_family=treatment_family,
            **proxy_embed_kwargs(args, dims.m_dim),
        ).to(device)
        pretrain_znet_xty(znet, loaders.train, device, epochs=200, lr=args.lr)
        train_baseline(
            znet, loaders.train, args.epochs, args.lr, device, label="ZNet",
        )
        c_tr, z_tr = extract_cz(znet, eval_ld.train, device)[:2]
        c_va, z_va = extract_cz(znet, eval_ld.val, device)[:2]
        c_te, z_te = extract_cz(znet, eval_ld.test, device)[:2]
        ckpt = make_iv_checkpoint(method, args, seed, {"state_dict": znet.state_dict()}, fingerprint_extra=fingerprint_extra)
        return c_tr.cpu(), z_tr.cpu(), c_va.cpu(), z_va.cpu(), c_te.cpu(), z_te.cpu(), ckpt

    if method == "AutoIV":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        autoiv = build_autoiv(
            x_dim=dims.x_dim, rep_dim=dims.z_dim, a_dim=dims.a_dim,
            y_dim=dims.y_dim, treatment_family=treatment_family,
            **proxy_embed_kwargs(args, dims.m_dim),
        ).to(device)
        train_autoiv(autoiv, loaders.train, args.epochs, args.lr, device)
        c_tr, z_tr = extract_cz(autoiv, eval_ld.train, device)[:2]
        c_va, z_va = extract_cz(autoiv, eval_ld.val, device)[:2]
        c_te, z_te = extract_cz(autoiv, eval_ld.test, device)[:2]
        ckpt = make_iv_checkpoint(method, args, seed, {"state_dict": autoiv.state_dict()}, fingerprint_extra=fingerprint_extra)
        return c_tr.cpu(), z_tr.cpu(), c_va.cpu(), z_va.cpu(), c_te.cpu(), z_te.cpu(), ckpt

    if method == "VIV":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        viv = build_viv(
            x_dim=dims.x_dim, c_dim=dims.c_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
            y_dim=dims.y_dim, treatment_family=treatment_family,
            **proxy_embed_kwargs(args, dims.m_dim),
        ).to(device)
        train_baseline(
            viv,
            loaders.train,
            args.epochs,
            args.lr,
            device,
            label="VIV",
            grad_clip=5.0,
        )
        c_tr, z_tr = extract_cz(viv, eval_ld.train, device)[:2]
        c_va, z_va = extract_cz(viv, eval_ld.val, device)[:2]
        c_te, z_te = extract_cz(viv, eval_ld.test, device)[:2]
        ckpt = make_iv_checkpoint(method, args, seed, {"state_dict": viv.state_dict()}, fingerprint_extra=fingerprint_extra)
        return c_tr.cpu(), z_tr.cpu(), c_va.cpu(), z_va.cpu(), c_te.cpu(), z_te.cpu(), ckpt

    if method == "DIV.VAE":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        divvae = build_divvae(
            x_dim=dims.x_dim, c_dim=dims.c_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
            y_dim=dims.y_dim, treatment_family=treatment_family,
            **proxy_embed_kwargs(args, dims.m_dim),
        ).to(device)
        train_baseline(
            divvae,
            loaders.train,
            args.epochs,
            min(float(args.lr), 1e-3),
            device,
            label="DIV.VAE",
            grad_clip=5.0,
        )
        c_tr, z_tr = extract_cz(divvae, eval_ld.train, device)[:2]
        c_va, z_va = extract_cz(divvae, eval_ld.val, device)[:2]
        c_te, z_te = extract_cz(divvae, eval_ld.test, device)[:2]
        if not all(torch.isfinite(t).all() for t in (c_tr, z_tr, c_va, z_va, c_te, z_te)):
            raise RuntimeError("DIV.VAE produced non-finite representations")
        ckpt = make_iv_checkpoint(method, args, seed, {"state_dict": divvae.state_dict()}, fingerprint_extra=fingerprint_extra)
        return c_tr.cpu(), z_tr.cpu(), c_va.cpu(), z_va.cpu(), c_te.cpu(), z_te.cpu(), ckpt

    # ── Oracles / baselines that read keys straight from the simulator ──
    tr = collect_loader(eval_ld.train, keys=["C", "V"])
    va = collect_loader(eval_ld.val, keys=["C", "V"])
    te = collect_loader(eval_ld.test, keys=["C", "V"])
    if method == "TrueIV":
        w_tr, z_tr = tr["C"], tr["V"]
        w_va, z_va = va["C"], va["V"]
        w_te, z_te = te["C"], te["V"]
        tens = {
            "w_tr": w_tr.cpu().numpy(), "z_tr": z_tr.cpu().numpy(),
            "w_va": w_va.cpu().numpy(), "z_va": z_va.cpu().numpy(),
            "w_te": w_te.cpu().numpy(), "z_te": z_te.cpu().numpy(),
        }
        ckpt = make_iv_checkpoint(method, args, seed, tens, fingerprint_extra=fingerprint_extra)
        return w_tr.cpu(), z_tr.cpu(), w_va.cpu(), z_va.cpu(), w_te.cpu(), z_te.cpu(), ckpt
    if method == "NoneIV":
        n_tr = loaders.train.dataset.n
        n_va = loaders.val.dataset.n
        n_te = loaders.test.dataset.n
        w_tr, z_tr = sample_noneiv_noise(n_tr, dims, rng)
        w_va, z_va = sample_noneiv_noise(n_va, dims, rng)
        w_te, z_te = sample_noneiv_noise(n_te, dims, rng)
        tens = {
            "w_tr": w_tr.numpy(), "z_tr": z_tr.numpy(),
            "w_va": w_va.numpy(), "z_va": z_va.numpy(),
            "w_te": w_te.numpy(), "z_te": z_te.numpy(),
        }
        ckpt = make_iv_checkpoint(method, args, seed, tens, fingerprint_extra=fingerprint_extra)
        return w_tr, z_tr, w_va, z_va, w_te, z_te, ckpt
    raise ValueError(f"Unknown method: {method}")


def load_iv_representation(
    path: Path,
    method: str,
    loaders: DataLoaderBundle,
    args: argparse.Namespace,
    dims: IVDims,
    device: torch.device,
    seed: int,
    rng: np.random.Generator,
    *,
    eval_loaders: DataLoaderBundle | None = None,
    fingerprint_extra: dict | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Restore train/val/test representations from disk; raise on mismatch.

    Extraction always runs through ``eval_loaders`` (``shuffle=False``) so the
    returned rows align with the canonical split order used by
    ``collect_loader`` on the same bundle.
    """
    ckpt = load_checkpoint(path)
    if ckpt.get("method") != method:
        raise ValueError(f"checkpoint method {ckpt.get('method')!r} != {method!r}")
    if ckpt.get("fingerprint") != fingerprint_iv(args, seed, extra=fingerprint_extra):
        raise ValueError("IV checkpoint fingerprint mismatch")

    eval_ld = (
        eval_loaders if eval_loaders is not None
        else make_eval_loaders(loaders)
    )

    if method == "PIV":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        piv = build_proxyiv(
            x_dim=dims.x_dim, m_dim=dims.m_dim, l_dim=dims.l_dim,
            w_dim=dims.w_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
            treatment_family=treatment_family,
        ).to(device)
        piv.load_state_dict(ckpt["payload"]["state_dict"])
        w_tr, z_tr = extract_wz(piv, eval_ld.train, device)[:2]
        w_va, z_va = extract_wz(piv, eval_ld.val, device)[:2]
        w_te, z_te = extract_wz(piv, eval_ld.test, device)[:2]
        return w_tr.cpu(), z_tr.cpu(), w_va.cpu(), z_va.cpu(), w_te.cpu(), z_te.cpu()

    if method == "Proxy-Orth":
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        model = build_proxy_orth(
            x_dim=dims.x_dim,
            m_dim=dims.m_dim,
            l_dim=dims.l_dim,
            z_dim=dims.z_dim,
            a_dim=dims.a_dim,
            treatment_family=treatment_family,
        ).to(device)
        model.load_state_dict(ckpt["payload"]["state_dict"])
        w_tr, z_tr = extract_proxy_orth_wz(model, eval_ld.train, device)
        w_va, z_va = extract_proxy_orth_wz(model, eval_ld.val, device)
        w_te, z_te = extract_proxy_orth_wz(model, eval_ld.test, device)
        return (
            w_tr.cpu(),
            z_tr.cpu(),
            w_va.cpu(),
            z_va.cpu(),
            w_te.cpu(),
            z_te.cpu(),
        )

    if method in ("ZNet", "AutoIV", "VIV", "DIV.VAE"):
        treatment_family = str(getattr(args, "treatment_type", "continuous"))
        if method == "ZNet":
            model = build_znet(
                x_dim=dims.x_dim, c_dim=dims.c_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
                y_dim=dims.y_dim, treatment_family=treatment_family,
                **proxy_embed_kwargs(args, dims.m_dim),
            ).to(device)
        elif method == "AutoIV":
            model = build_autoiv(
                x_dim=dims.x_dim, rep_dim=dims.z_dim, a_dim=dims.a_dim,
                y_dim=dims.y_dim, treatment_family=treatment_family,
                **proxy_embed_kwargs(args, dims.m_dim),
            ).to(device)
        elif method == "VIV":
            model = build_viv(
                x_dim=dims.x_dim, c_dim=dims.c_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
                y_dim=dims.y_dim, treatment_family=treatment_family,
                **proxy_embed_kwargs(args, dims.m_dim),
            ).to(device)
        else:
            model = build_divvae(
                x_dim=dims.x_dim, c_dim=dims.c_dim, z_dim=dims.z_dim, a_dim=dims.a_dim,
                y_dim=dims.y_dim, treatment_family=treatment_family,
                **proxy_embed_kwargs(args, dims.m_dim),
            ).to(device)
        model.load_state_dict(ckpt["payload"]["state_dict"])
        c_tr, z_tr = extract_cz(model, eval_ld.train, device)[:2]
        c_va, z_va = extract_cz(model, eval_ld.val, device)[:2]
        c_te, z_te = extract_cz(model, eval_ld.test, device)[:2]
        return c_tr.cpu(), z_tr.cpu(), c_va.cpu(), z_va.cpu(), c_te.cpu(), z_te.cpu()

    if method in ("TrueIV", "NoneIV"):
        p = ckpt["payload"]
        return (
            torch.from_numpy(np.asarray(p["w_tr"], dtype=np.float32)),
            torch.from_numpy(np.asarray(p["z_tr"], dtype=np.float32)),
            torch.from_numpy(np.asarray(p["w_va"], dtype=np.float32)),
            torch.from_numpy(np.asarray(p["z_va"], dtype=np.float32)),
            torch.from_numpy(np.asarray(p["w_te"], dtype=np.float32)),
            torch.from_numpy(np.asarray(p["z_te"], dtype=np.float32)),
        )

    raise ValueError(f"Unknown method: {method}")


# ═══════════════════════════════════════════════════════════════════
#  Orchestration: NPIV build / fit / load
# ═══════════════════════════════════════════════════════════════════


def build_npiv_model(
    downstream: str,
    args: argparse.Namespace,
    device: torch.device,
    dims: IVDims,
    *,
    w_dim: int,
) -> nn.Module:
    """Construct the NPIV model for the requested downstream estimator."""
    treatment_family = str(getattr(args, "treatment_type", "continuous"))
    if downstream == "deepiv":
        n_layers = max(1, int(args.deepiv_num_layers))
        treat_hidden = tuple([int(args.deepiv_treatment_hidden)] * n_layers)
        resp_hidden = tuple([int(args.deepiv_response_hidden)] * n_layers)
        return DeepIV(
            a_dim=dims.a_dim, z_dim=dims.z_dim, w_dim=w_dim, l_dim=dims.l_dim,
            treatment_family=treatment_family,
            num_components=int(args.deepiv_num_components),
            treatment_hidden_dims=treat_hidden,
            response_hidden_dims=resp_hidden,
            activation=str(args.deepiv_activation),
            dropout=float(args.deepiv_dropout),
            min_scale=float(getattr(args, "deepiv_min_scale", 1e-3)),
        ).to(device)
    if downstream == "dfiv":
        n_layers = max(1, int(args.dfiv_num_layers))
        psi_hidden = tuple([int(args.dfiv_psi_hidden)] * n_layers)
        phi_hidden = tuple([int(args.dfiv_phi_hidden)] * n_layers)
        xi_hidden = tuple([int(args.dfiv_xi_hidden)] * n_layers)
        return DFIV(
            a_dim=dims.a_dim, z_dim=dims.z_dim, w_dim=w_dim, l_dim=dims.l_dim,
            treatment_family=treatment_family,
            feature_dim=int(args.dfiv_feature_dim),
            psi_hidden_dims=psi_hidden,
            phi_hidden_dims=phi_hidden,
            xi_hidden_dims=xi_hidden,
            activation=str(args.dfiv_activation),
            dropout=float(args.dfiv_dropout),
            lambda1=float(args.dfiv_lambda1),
            lambda2=float(args.dfiv_lambda2),
        ).to(device)
    raise ValueError(downstream)


def load_npiv_checkpoint(
    path: Path,
    downstream: str,
    args: argparse.Namespace,
    device: torch.device,
    dims: IVDims,
    seed: int,
    iv_method: str,
    w_tr: Tensor,
    z_tr: Tensor,
    *,
    test: bool = False,
    fingerprint_extra: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Rebuild an NPIV model from disk and restore its normalisation stats.

    Verifies hyperparameter fingerprint; in training mode also verifies the
    representation digest.  Under ``test``, the digest check is skipped
    because ``(W, Z)`` are recomputed from a loaded IV net and small
    float/device nondeterminism can change the hash even when weights match.
    """
    ckpt = load_checkpoint(path)
    if ckpt.get("fingerprint") != fingerprint_npiv(args, seed, iv_method, downstream, extra=fingerprint_extra):
        raise ValueError("NPIV checkpoint fingerprint mismatch")
    dig = ckpt.get("representation_digest")
    if (
        not test
        and dig is not None
        and dig != representation_digest(w_tr, z_tr)
    ):
        raise ValueError("NPIV checkpoint does not match current IV representations")
    model = build_npiv_model(
        downstream, args, device, dims,
        w_dim=int(w_tr.shape[1]),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, load_npiv_norms(ckpt["norms"])


def _stable_npiv_seed(
    seed: int,
    iv_method: str,
    downstream: str,
    restart_idx: int = 0,
    init_seed_offset: int = 0,
) -> int:
    """Deterministic per-fit seed independent of outer method order."""
    payload = (
        f"{int(seed)}|{iv_method}|{downstream}|"
        f"{int(restart_idx)}|{int(init_seed_offset)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _seed_npiv_run(local_seed: int) -> None:
    """Seed Python / NumPy / PyTorch for one NPIV fit."""
    random.seed(int(local_seed))
    np.random.seed(int(local_seed) % (2**32))
    torch.manual_seed(int(local_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(local_seed))


def _fit_affine_calibration(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``target ≈ scale * pred + bias`` on 1D arrays."""
    pred_1 = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_1 = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred_1.size != target_1.size or pred_1.size == 0:
        return 1.0, 0.0
    X = np.stack([pred_1, np.ones_like(pred_1)], axis=1)
    coef, *_ = np.linalg.lstsq(X, target_1, rcond=None)
    scale = float(coef[0]) if np.isfinite(coef[0]) else 1.0
    bias = float(coef[1]) if np.isfinite(coef[1]) else 0.0
    return scale, bias


@torch.no_grad()
def _npiv_train_mse(
    model: nn.Module,
    a: np.ndarray,
    w: np.ndarray,
    l: np.ndarray,
    y: np.ndarray,
    norms: dict,
    device: torch.device,
    batch_size: int = 2048,
) -> float:
    """Original-scale in-sample response MSE used for restart selection."""
    model.eval()
    a_n = ((a - norms["a_mean"]) / norms["a_std"]).astype(np.float32)
    if a_n.ndim == 1:
        a_n = a_n.reshape(-1, 1)
    w_n = ((w - norms["w_mean"]) / norms["w_std"]).astype(np.float32)
    l_n = ((l - norms["l_mean"]) / norms["l_std"]).astype(np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    preds: list[np.ndarray] = []
    n = a_n.shape[0]
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        a_b = torch.as_tensor(a_n[lo:hi], dtype=torch.float32, device=device)
        w_b = torch.as_tensor(w_n[lo:hi], dtype=torch.float32, device=device)
        l_b = torch.as_tensor(l_n[lo:hi], dtype=torch.float32, device=device)
        pred_b = model.predict(a_b, w_b, l_b).detach().cpu().numpy().reshape(-1)
        preds.append(pred_b)

    pred = np.concatenate(preds, axis=0)
    pred = (
        pred * float(norms.get("y_cal_scale", 1.0))
        + float(norms.get("y_cal_bias", 0.0))
    )
    pred = pred * float(norms["y_std"]) + float(norms["y_mean"])
    return float(np.mean((pred - y) ** 2))


def fit_npiv(
    downstream: str,
    train_raw: dict[str, np.ndarray],
    w_tr: Tensor,
    z_tr: Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dims: IVDims,
    *,
    val_raw: dict[str, np.ndarray] | None = None,
    w_val: Tensor | None = None,
    z_val: Tensor | None = None,
    checkpoint_path: Path | None = None,
    iv_method: str = "",
    seed: int = 0,
    force_retrain: bool = False,
    test: bool = False,
    fingerprint_extra: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Fit (or load) an NPIV model. Returns ``(model, norms)``.

    ``train_raw`` must contain ``A``, ``Y`` (floats) and ``L`` (shape ``(n, l_dim)``).
    ``w_tr, z_tr`` are the IV train representations (CPU tensors).  If
    ``checkpoint_path`` points to a matching file and ``force_retrain`` is
    false, the saved weights are loaded; otherwise training runs and the model
    is saved.  ``test=True`` forbids training — missing/bad checkpoints
    raise ``SystemExit``.
    """
    z_np = z_tr.numpy()
    w_np = w_tr.numpy()
    a_np = train_raw["A"]
    l_np = train_raw["L"]
    y_np = train_raw["Y"].reshape(-1, 1)
    a_val_np = None if val_raw is None else val_raw["A"]
    l_val_np = None if val_raw is None else val_raw["L"]
    y_val_np = None if val_raw is None else val_raw["Y"].reshape(-1, 1)
    w_val_np = None if w_val is None else w_val.numpy()
    z_val_np = None if z_val is None else z_val.numpy()

    load_ok = (
        checkpoint_path is not None
        and checkpoint_path.exists()
        and not force_retrain
    )
    if load_ok:
        try:
            model, norms = load_npiv_checkpoint(
                checkpoint_path, downstream, args, device, dims, seed, iv_method,
                w_tr, z_tr, test=test, fingerprint_extra=fingerprint_extra,
            )
            print(f"    [{downstream}] loaded NPIV checkpoint {checkpoint_path.name}")
            return model, norms
        except (ValueError, KeyError, RuntimeError) as e:
            if test:
                raise SystemExit(
                    f"test: cannot load NPIV {checkpoint_path}: {e!r}",
                ) from e
            print(f"    [{downstream}] NPIV checkpoint unusable ({e!r}) — retraining.")
    elif test:
        raise SystemExit(
            f"plots-only: missing NPIV checkpoint (expected {checkpoint_path})",
        )

    wd = int(w_tr.shape[1])
    n_restarts = max(1, int(getattr(args, "npiv_restarts", 1)))
    init_seed_offset = int(getattr(args, "npiv_init_seed_offset", 0))
    best_model: nn.Module | None = None
    best_norms: dict | None = None
    best_score = float("inf")

    for restart_idx in range(n_restarts):
        local_seed = _stable_npiv_seed(
            seed, iv_method, downstream, restart_idx, init_seed_offset,
        )
        _seed_npiv_run(local_seed)
        split_rng = np.random.default_rng(local_seed ^ 0x5F3759DF)

        if downstream == "deepiv":
            model = build_npiv_model("deepiv", args, device, dims, w_dim=wd)
            deepiv_lr = (
                args.lr if float(getattr(args, "deepiv_lr", -1.0)) < 0
                else float(args.deepiv_lr)
            )
            deepiv_epochs = int(getattr(args, "deepiv_epochs", -1))
            if deepiv_epochs < 0:
                raise ValueError("DeepIV epochs are not specified")
            norms = train_deepiv(
                model, a_np, z_np, w_np, l_np, y_np,
                epochs=deepiv_epochs, lr=deepiv_lr, device=device,
                a_val_raw=a_val_np,
                z_val_raw=z_val_np,
                w_val_raw=w_val_np,
                l_val_raw=l_val_np,
                y_val_raw=y_val_np,
                batch_size=args.batch_size,
                log_every=max(1, deepiv_epochs // 10),
                num_samples=int(args.deepiv_num_samples),
                weight_decay=float(args.deepiv_weight_decay),
                val_frac=float(args.deepiv_val_frac),
                patience=int(args.deepiv_patience),
                normalize_wl=bool(args.npiv_normalize_wl),
                rng=split_rng,
            )
        elif downstream == "dfiv":
            model = build_npiv_model("dfiv", args, device, dims, w_dim=wd)
            dfiv_lr = args.lr if args.dfiv_lr < 0 else float(args.dfiv_lr)
            dfiv_epochs = int(getattr(args, "dfiv_epochs", -1))
            if dfiv_epochs < 0:
                raise ValueError("DFIV epochs are not specified")
            norms = train_dfiv(
                model, a_np, z_np, w_np, l_np, y_np,
                epochs=dfiv_epochs, lr=dfiv_lr, device=device,
                a_val_raw=a_val_np,
                z_val_raw=z_val_np,
                w_val_raw=w_val_np,
                l_val_raw=l_val_np,
                y_val_raw=y_val_np,
                batch_size=int(args.dfiv_batch_size),
                log_every=max(1, dfiv_epochs // 10),
                lr_phi=float(args.dfiv_lr_phi),
                weight_decay=float(args.dfiv_weight_decay),
                weight_decay_psi=(
                    float(args.dfiv_weight_decay_psi)
                    if getattr(args, "dfiv_weight_decay_psi", None) is not None
                    else None
                ),
                weight_decay_phi=(
                    float(args.dfiv_weight_decay_phi)
                    if getattr(args, "dfiv_weight_decay_phi", None) is not None
                    else None
                ),
                val_frac=float(getattr(args, "dfiv_val_frac", 0.1)),
                patience=int(getattr(args, "dfiv_patience", 20)),
                normalize_wl=bool(args.npiv_normalize_wl),
                rng=split_rng,
            )
        else:
            raise ValueError(downstream)

        score = _npiv_train_mse(model, a_np, w_np, l_np, y_np, norms, device)
        if n_restarts > 1:
            print(
                f"    [{downstream}] restart {restart_idx + 1}/{n_restarts}  "
                f"train_mse={score:.6f}",
            )
        if score < best_score:
            best_score = score
            best_model = model
            best_norms = norms

    if best_model is None or best_norms is None:
        raise RuntimeError(f"Failed to train downstream estimator: {downstream}")
    model = best_model
    norms = best_norms

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fingerprint": fingerprint_npiv(args, seed, iv_method, downstream, extra=fingerprint_extra),
                "downstream": downstream,
                "state_dict": model.state_dict(),
                "norms": norms,
                "representation_digest": representation_digest(w_tr, z_tr),
            },
            checkpoint_path,
        )
        print(f"    [{downstream}] saved NPIV checkpoint {checkpoint_path.name}")

    return model, norms


__all__ = [
    "build_npiv_model",
    "fit_npiv",
    "load_iv_representation",
    "load_npiv_checkpoint",
    "pretrain_znet_xty",
    "train_autoiv",
    "train_baseline",
    "train_deepiv",
    "train_dfiv",
    "train_giv",
    "train_iv_representation",
    "train_proxy_orth",
    "train_stage1",
    "train_stage2",
]
