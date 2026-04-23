"""Shared utilities for the ICA-IV codebase.

Small, reusable helpers consumed by :mod:`trainer`, :mod:`evaluation`, and the
scripts under ``scripts/``.  Nothing here runs an optimisation loop or owns a
top-level training/evaluation workflow; those live in :mod:`trainer` and
:mod:`evaluation`.

Sections:

* **Reproducibility** — :func:`seed_everything`, :func:`torch_generator`.
* **Dimensions & placeholders** — :class:`IVDims`, :func:`empty_covariates`.
* **Loader helpers** — :func:`make_eval_loaders`, :func:`collect_loader`,
  :func:`extract_wz`, :func:`extract_cz`.
* **IV-method kwargs / shapes** — :func:`proxy_embed_kwargs`,
  :func:`npiv_w_dim`, :func:`sample_noneiv_noise`.
* **NPIV normalisation stats** — :func:`wl_norm_stats`.
* **Checkpointing** — :func:`load_checkpoint`, :func:`method_tag`,
  :func:`representation_digest`, :func:`fingerprint_iv`,
  :func:`fingerprint_npiv`, :func:`make_iv_checkpoint`,
  :func:`load_npiv_norms`.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from synthetic_dataset import DataLoaderBundle


# ═══════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════


def seed_everything(seed: int = 42) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch RNGs; return a fresh NumPy ``Generator``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def torch_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded ``torch.Generator`` (e.g. for reproducible DataLoaders)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ═══════════════════════════════════════════════════════════════════
#  Dimensions & placeholders
# ═══════════════════════════════════════════════════════════════════


@dataclass
class IVDims:
    """Fixed input/output widths for an IV benchmark."""

    x_dim: int
    m_dim: int
    w_dim: int
    z_dim: int
    a_dim: int
    l_dim: int
    y_dim: int = 1
    c_dim: int | None = None  # defaults to ``w_dim`` (baselines share the budget)

    def __post_init__(self) -> None:
        if self.c_dim is None:
            self.c_dim = self.w_dim


def empty_covariates(n: int, device: torch.device) -> Tensor:
    """Zero-width ``L`` placeholder of shape ``(n, 0)`` for benchmarks with no covariates."""
    return torch.empty(n, 0, device=device)


# ═══════════════════════════════════════════════════════════════════
#  DataLoader helpers
# ═══════════════════════════════════════════════════════════════════


def make_eval_loaders(
    loaders: DataLoaderBundle, eval_batch_size: int = 1024,
) -> DataLoaderBundle:
    """Deterministic (``shuffle=False``) loaders over the same underlying datasets.

    Using these for every extraction / collection pass guarantees that
    :func:`collect_loader`, :func:`extract_wz`, and :func:`extract_cz` return
    rows in a single canonical order, so ``(A, Y, L)`` stay aligned with the
    extracted ``(W, Z)`` across passes.
    """
    return DataLoaderBundle(
        train=DataLoader(loaders.train.dataset, batch_size=eval_batch_size, shuffle=False),
        val=DataLoader(loaders.val.dataset, batch_size=eval_batch_size, shuffle=False),
        test=DataLoader(loaders.test.dataset, batch_size=eval_batch_size, shuffle=False),
    )


@torch.no_grad()
def collect_loader(loader: DataLoader, keys: Sequence[str]) -> dict[str, Tensor]:
    """Concatenate all batches from *loader* for the requested *keys*."""
    accum: dict[str, list[Tensor]] = {k: [] for k in keys}
    for batch in loader:
        for k in keys:
            accum[k].append(batch[k])
    return {k: torch.cat(v) for k, v in accum.items()}


# ═══════════════════════════════════════════════════════════════════
#  Representation extraction
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_wz(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_key: str | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Run *model* over *loader* and return ``(W, Z)`` (PIV / ProxyIVModel).

    If *target_key* is given (e.g. ``"V"``), the corresponding ground-truth
    tensor is collected in the same order and returned as a third element.
    """
    model.eval()
    ws, zs, extras = [], [], []
    for batch in loader:
        x = batch["X"].to(device)
        w, z = model.encode_representations(x)
        ws.append(w.cpu())
        zs.append(z.cpu())
        if target_key is not None:
            extras.append(batch[target_key])
    if target_key is not None:
        return torch.cat(ws), torch.cat(zs), torch.cat(extras)
    return torch.cat(ws), torch.cat(zs)


@torch.no_grad()
def extract_cz(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_key: str | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Run *model* over *loader* and return ``(C, Z)`` for baselines with
    ``encode_c`` / ``encode_z`` (optionally plus a ground-truth key)."""
    model.eval()
    cs, zs, extras = [], [], []
    for batch in loader:
        x = batch["X"].to(device)
        if getattr(model, "use_proxy_embed", False):
            m = batch["M"].to(device)
            cs.append(model.encode_c(x, m).cpu())
        else:
            cs.append(model.encode_c(x).cpu())
        zs.append(model.encode_z(x).cpu())
        if target_key is not None:
            extras.append(batch[target_key])
    if target_key is not None:
        return torch.cat(cs), torch.cat(zs), torch.cat(extras)
    return torch.cat(cs), torch.cat(zs)


# ═══════════════════════════════════════════════════════════════════
#  IV-method kwargs / shapes
# ═══════════════════════════════════════════════════════════════════


def proxy_embed_kwargs(args: argparse.Namespace, m_dim: int) -> dict:
    """Kwargs for ZNet / AutoIV / VIV / DIV.VAE controlling the proxy-``M`` embedding.

    ``args.proxy_embed_dim > 0`` concatenates an MLP embedding of the proxy
    ``M`` to the learned confounder ``C`` (so ``W = [C | MLP(M)]``);
    ``0`` disables it.
    """
    d = int(args.proxy_embed_dim)
    if d <= 0:
        return {"use_proxy_embed": False, "m_dim": m_dim, "proxy_embed_dim": 4}
    return {"use_proxy_embed": True, "m_dim": m_dim, "proxy_embed_dim": d}


def npiv_w_dim(method: str, args: argparse.Namespace, dims: IVDims) -> int:
    """Input width of the NPIV structural net's confounder branch for *method*.

    Accounts for the optional proxy-``M`` embedding for ZNet / AutoIV / VIV / DIV.VAE.
    """
    if method in ("ZNet", "AutoIV", "VIV", "DIV.VAE") and int(args.proxy_embed_dim) > 0:
        return int(dims.c_dim + int(args.proxy_embed_dim))
    return int(dims.c_dim)


def sample_noneiv_noise(
    n: int, dims: IVDims, rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """Independent Gaussian noise of the right ``(W, Z)`` shape for the NoneIV baseline."""
    w = torch.from_numpy(rng.standard_normal((n, dims.w_dim)).astype(np.float32))
    z = torch.from_numpy(rng.standard_normal((n, dims.z_dim)).astype(np.float32))
    return w, z


# ═══════════════════════════════════════════════════════════════════
#  NPIV normalisation stats
# ═══════════════════════════════════════════════════════════════════


def wl_norm_stats(
    w: np.ndarray, l: np.ndarray, enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-column z-score stats ``(w_mean, w_std, l_mean, l_std)`` for W and L.

    Returns identity stats (zeros / ones) when *enabled* is ``False`` so
    downstream code can always apply ``(x − mean) / std`` unconditionally.
    """
    if enabled:
        w_mean = w.mean(axis=0)
        w_std = w.std(axis=0) + 1e-8
        l_mean = l.mean(axis=0)
        l_std = l.std(axis=0) + 1e-8
    else:
        w_mean = np.zeros(w.shape[1], dtype=np.float64)
        w_std = np.ones(w.shape[1], dtype=np.float64)
        l_mean = np.zeros(l.shape[1], dtype=np.float64)
        l_std = np.ones(l.shape[1], dtype=np.float64)
    return w_mean, w_std, l_mean, l_std


# ═══════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════


def load_checkpoint(path: Path) -> object:
    """``torch.load`` onto CPU; compatible with both ``weights_only`` and legacy APIs."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def method_tag(method: str) -> str:
    """Filesystem-safe fragment for *method* (e.g. ``DIV.VAE`` → ``DIV_VAE``)."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", method).replace(".", "_")


def representation_digest(w_tr: Tensor, z_tr: Tensor) -> str:
    """SHA-256 digest tying an NPIV checkpoint to specific train-split ``(W, Z)``."""
    h = hashlib.sha256()
    h.update(np.asarray(w_tr.numpy(), dtype=np.float32).tobytes())
    h.update(np.asarray(z_tr.numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def fingerprint_iv(args: argparse.Namespace, seed: int) -> dict:
    """Hyperparameter fingerprint for IV-representation checkpoints."""
    return {
        # v6: PIV stage-2 removed the baseline branch entirely; the remaining
        # treatment head consumes (Z, W, L) directly. Older IV checkpoints are
        # architecturally incompatible and should not be reused.
        "version": 6,
        "kind": "iv_repr",
        "seed": int(seed),
        "n": int(args.n),
        "rho": float(args.rho),
        "coef_eps_a": float(args.coef_eps_a),
        "coef_u_outcome": float(args.coef_u_outcome),
        "s1_epochs": int(args.s1_epochs),
        "s2_epochs": int(args.s2_epochs),
        "baseline_epochs": int(args.baseline_epochs),
        "autoiv_epochs": int(args.autoiv_epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "proxy_embed_dim": int(args.proxy_embed_dim),
    }


def fingerprint_npiv(
    args: argparse.Namespace,
    seed: int,
    iv_method: str,
    downstream: str,
) -> dict:
    """Hyperparameter fingerprint for NPIV checkpoints.

    Extends :func:`fingerprint_iv` with NPIV-specific architecture and
    optimisation flags for the downstream estimators.
    """
    fp = dict(fingerprint_iv(args, seed))
    fp["kind"] = "npiv"
    fp["iv_method"] = iv_method
    fp["downstream"] = downstream
    fp["deepiv_epochs"] = int(getattr(args, "deepiv_epochs", -1))
    fp["dfiv_epochs"] = int(getattr(args, "dfiv_epochs", -1))
    fp["npiv_restarts"] = int(getattr(args, "npiv_restarts", 1))
    fp["npiv_init_seed_offset"] = int(getattr(args, "npiv_init_seed_offset", 0))
    # DeepIV architecture + optimisation
    fp["deepiv_num_components"] = int(args.deepiv_num_components)
    fp["deepiv_treatment_hidden"] = int(args.deepiv_treatment_hidden)
    fp["deepiv_response_hidden"] = int(args.deepiv_response_hidden)
    fp["deepiv_num_layers"] = int(args.deepiv_num_layers)
    fp["deepiv_dropout"] = float(args.deepiv_dropout)
    fp["deepiv_activation"] = str(args.deepiv_activation)
    fp["deepiv_num_samples"] = int(args.deepiv_num_samples)
    fp["deepiv_lr"] = float(getattr(args, "deepiv_lr", -1.0))
    fp["deepiv_weight_decay"] = float(args.deepiv_weight_decay)
    fp["deepiv_min_scale"] = float(getattr(args, "deepiv_min_scale", 1e-3))
    fp["deepiv_val_frac"] = float(args.deepiv_val_frac)
    fp["deepiv_patience"] = int(args.deepiv_patience)
    # DFIV architecture + optimisation
    fp["dfiv_feature_dim"] = int(args.dfiv_feature_dim)
    fp["dfiv_psi_hidden"] = int(args.dfiv_psi_hidden)
    fp["dfiv_phi_hidden"] = int(args.dfiv_phi_hidden)
    fp["dfiv_xi_hidden"] = int(args.dfiv_xi_hidden)
    fp["dfiv_num_layers"] = int(args.dfiv_num_layers)
    fp["dfiv_dropout"] = float(args.dfiv_dropout)
    fp["dfiv_activation"] = str(args.dfiv_activation)
    fp["dfiv_lambda1"] = float(args.dfiv_lambda1)
    fp["dfiv_lambda2"] = float(args.dfiv_lambda2)
    fp["dfiv_lr"] = float(args.dfiv_lr)
    fp["dfiv_lr_phi"] = float(args.dfiv_lr_phi)
    fp["dfiv_batch_size"] = int(args.dfiv_batch_size)
    fp["dfiv_weight_decay"] = float(args.dfiv_weight_decay)
    fp["dfiv_val_frac"] = float(getattr(args, "dfiv_val_frac", 0.1))
    fp["dfiv_patience"] = int(getattr(args, "dfiv_patience", 20))
    fp["npiv_normalize_wl"] = bool(args.npiv_normalize_wl)
    fp["npiv_normalize_z"] = True
    return fp


def make_iv_checkpoint(
    method: str, args: argparse.Namespace, seed: int, payload: dict,
) -> dict:
    """Build the canonical on-disk dict for an IV-representation checkpoint."""
    return {"method": method, "fingerprint": fingerprint_iv(args, seed), "payload": payload}


def load_npiv_norms(raw: object) -> dict:
    """Canonicalise on-disk NPIV normalisation stats into a dict of typed arrays."""
    if isinstance(raw, dict):
        out = dict(raw)
        for k in ("a_mean", "a_std", "y_mean", "y_std"):
            out[k] = float(out[k])
        for k in ("w_mean", "w_std", "l_mean", "l_std"):
            out[k] = np.asarray(out[k], dtype=np.float32)
        out["normalize_wl"] = bool(out.get("normalize_wl", True))
        return out
    raise ValueError(
        "NPIV checkpoint has legacy scalar norms without W/L stats; please retrain.",
    )


__all__ = [
    "IVDims",
    "collect_loader",
    "empty_covariates",
    "extract_cz",
    "extract_wz",
    "fingerprint_iv",
    "fingerprint_npiv",
    "load_checkpoint",
    "load_npiv_norms",
    "make_eval_loaders",
    "make_iv_checkpoint",
    "method_tag",
    "npiv_w_dim",
    "proxy_embed_kwargs",
    "representation_digest",
    "sample_noneiv_noise",
    "seed_everything",
    "torch_generator",
    "wl_norm_stats",
]
