#!/usr/bin/env python3
"""Shared utilities for the demands benchmark scripts."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sim_demands import (  # noqa: E402
    DemandsConfig,
    simulate_demands,
    structural_response_demands,
)
from synthetic_dataset import make_synthetic_proxyiv_dataloaders  # noqa: E402
from evaluation import build_counterfactual_eval, true_contrast_surface  # noqa: E402
from trainer import load_iv_representation, train_iv_representation  # noqa: E402
from utils import IVDims, make_eval_loaders, method_tag  # noqa: E402

DEMANDS_DIMS = IVDims(x_dim=3, m_dim=1, w_dim=1, z_dim=1, a_dim=1, l_dim=2, y_dim=1)
ALL_METHODS = [
    "PIV",
    "Proxy-Orth",
    "Conf-Orth",
    "ZNet",
    "AutoIV",
    "VIV",
    "DIV.VAE",
    "TrueIV",
    "NoneIV",
]


def demands_h0(a: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Demands structural response adapter: ``h0(a, L=[L1, L2])``."""
    return structural_response_demands(
        np.asarray(a, dtype=np.float32),
        np.asarray(l[:, 0], dtype=np.float32),
        np.asarray(l[:, 1], dtype=np.float32),
    ).astype(np.float64).ravel()


def rho_tag(rho: float) -> str:
    """Filesystem-friendly rho token."""
    return f"{rho:g}".replace("-", "m").replace(".", "p")


def build_eval_setup(
    args: argparse.Namespace,
    rho: float,
    *,
    include_surface: bool,
) -> dict[str, object]:
    """Build the shared evaluation geometry for one rho."""
    pop_cfg = DemandsConfig(
        n=int(args.pop_n),
        seed=int(args.pop_seed),
        rho=float(rho),
        coef_eps_a=float(args.coef_eps_a),
        coef_u_outcome=float(args.coef_u_outcome),
        coef_v_treatment=float(getattr(args, "coef_v_treatment", 1.0)),
        coef_c_treatment=float(getattr(args, "coef_c_treatment", 1.0)),
        coef_c_outcome=float(getattr(args, "coef_c_outcome", 1.0)),
        coef_d_outcome=float(getattr(args, "coef_d_outcome", 1.0)),
    )
    pop_raw = simulate_demands(pop_cfg)
    a_pop = np.asarray(pop_raw["A"], dtype=np.float64).reshape(-1)
    l_pop = np.asarray(pop_raw["L"], dtype=np.float64)

    override_str = str(args.contrast_ref_a).strip()
    a_ref_override = float(override_str) if override_str else None
    a_ref, a_grid, tau_pop_true = build_counterfactual_eval(
        a_pop,
        l_pop,
        demands_h0,
        a_ref_override=a_ref_override,
        cf_a_pctl_low=float(args.cf_a_pctl_low),
        cf_a_pctl_high=float(args.cf_a_pctl_high),
        cf_grid_n=int(args.cf_grid_n),
    )

    out: dict[str, object] = {
        "a_ref": float(a_ref),
        "a_grid": a_grid,
        "tau_pop_true": tau_pop_true,
    }
    if include_surface:
        surface_a_grid = np.linspace(
            float(a_grid[0]), float(a_grid[-1]),
            int(args.surface_na), dtype=np.float64,
        )
        surface_l1_grid = np.linspace(
            float(args.surface_l1_lo), float(args.surface_l1_hi),
            int(args.surface_nl1), dtype=np.float64,
        )
        contrast_surface_true = true_contrast_surface(
            surface_a_grid,
            float(a_ref),
            surface_l1_grid,
            vary_idx=0,
            l_units=l_pop,
            h0_fn=demands_h0,
        )
        out["contrast_surface_a_grid"] = surface_a_grid
        out["contrast_surface_l1_grid"] = surface_l1_grid
        out["contrast_surface_true"] = contrast_surface_true
    return out


def select_unit_plot_index(a_grid: np.ndarray, requested_index: int) -> int:
    """Resolve the unit-plot slice index from the grid."""
    if int(requested_index) >= 0:
        return min(int(requested_index), len(a_grid) - 1)
    return max(0, len(a_grid) // 2)


def build_demands_loaders(
    args: argparse.Namespace,
    *,
    seed: int,
    rho: float,
) -> tuple[dict[str, np.ndarray], object, object]:
    """Simulate one demands dataset and build train/test loaders."""
    cfg = DemandsConfig(
        n=int(args.n),
        seed=int(seed),
        rho=float(rho),
        coef_eps_a=float(args.coef_eps_a),
        coef_u_outcome=float(args.coef_u_outcome),
        coef_v_treatment=float(getattr(args, "coef_v_treatment", 1.0)),
        coef_c_treatment=float(getattr(args, "coef_c_treatment", 1.0)),
        coef_c_outcome=float(getattr(args, "coef_c_outcome", 1.0)),
        coef_d_outcome=float(getattr(args, "coef_d_outcome", 1.0)),
    )
    raw = simulate_demands(cfg)
    keys = ["X", "M", "A", "Y", "L", "C", "V", "D", "U"]
    loaders = make_synthetic_proxyiv_dataloaders(
        arrays=raw,
        keys=keys,
        required_keys=keys,
        batch_size=int(args.batch_size),
        train_frac=0.70,
        val_frac=0.15,
        seed=int(seed),
    )
    eval_loaders = make_eval_loaders(
        loaders,
        eval_batch_size=max(int(args.batch_size), 1024),
    )
    return raw, loaders, eval_loaders


def build_iv_checkpoint_path(
    ckpt_root: Path | None,
    *,
    seed: int,
    rho: float,
    method: str,
) -> Path | None:
    """Canonical IV checkpoint path for one `(seed, rho, method)`."""
    if ckpt_root is None:
        return None
    return ckpt_root / f"s{seed}_rho{rho_tag(rho)}_{method_tag(method)}_iv.pt"


def build_npiv_checkpoint_path(
    ckpt_root: Path | None,
    *,
    seed: int,
    rho: float,
    method: str,
    downstream: str,
    oracle_c: bool,
) -> Path | None:
    """Canonical NPIV checkpoint path for one `(seed, rho, method, downstream)`."""
    if ckpt_root is None:
        return None
    oracle_suffix = "_orc" if oracle_c and method != "PIV" else ""
    return (
        ckpt_root
        / f"s{seed}_rho{rho_tag(rho)}_{method_tag(method)}_{downstream}{oracle_suffix}_npiv.pt"
    )


def demands_fingerprint_extra(args: argparse.Namespace) -> dict:
    """Dataset-specific fingerprint fields for the demands benchmark."""
    return {
        "seed": int(args.seed),
        "rho": float(args.rho),
        "coef_eps_a": float(args.coef_eps_a),
        "coef_u_outcome": float(args.coef_u_outcome),
        "coef_v_treatment": float(getattr(args, "coef_v_treatment", 1.0)),
        "coef_c_treatment": float(getattr(args, "coef_c_treatment", 1.0)),
        "coef_c_outcome": float(getattr(args, "coef_c_outcome", 1.0)),
        "coef_d_outcome": float(getattr(args, "coef_d_outcome", 1.0)),
    }


def load_or_train_iv(
    method: str,
    iv_ckpt_path: Path | None,
    loaders,
    eval_loaders,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    rng: np.random.Generator,
    *,
    allow_train: bool,
    fingerprint_extra: dict | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load a matching IV checkpoint, optionally falling back to training."""
    if iv_ckpt_path is not None and iv_ckpt_path.exists() and not args.force_retrain:
        try:
            w_tr, z_tr, w_va, z_va, w_te, z_te = load_iv_representation(
                iv_ckpt_path,
                method,
                loaders,
                args,
                DEMANDS_DIMS,
                device,
                seed,
                rng,
                eval_loaders=eval_loaders,
                fingerprint_extra=fingerprint_extra,
            )
            print(f"\n  [{method}] loaded IV checkpoint {iv_ckpt_path.name}")
            return w_tr, z_tr, w_va, z_va, w_te, z_te
        except (ValueError, KeyError, RuntimeError) as exc:
            if not allow_train:
                raise SystemExit(
                    f"cannot load IV checkpoint {iv_ckpt_path}: {exc!r}",
                ) from exc
            print(f"\n  [{method}] IV checkpoint unusable ({exc!r}) — retraining.")

    if not allow_train:
        raise SystemExit(f"missing IV checkpoint for {method}: {iv_ckpt_path}")

    print(f"\n  [{method}] training representation …")
    w_tr, z_tr, w_va, z_va, w_te, z_te, iv_ckpt = train_iv_representation(
        method,
        loaders,
        args,
        DEMANDS_DIMS,
        device,
        rng,
        seed,
        eval_loaders=eval_loaders,
        fingerprint_extra=fingerprint_extra,
    )
    if iv_ckpt_path is not None:
        iv_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(iv_ckpt, iv_ckpt_path)
        print(f"  [{method}] saved IV checkpoint {iv_ckpt_path.name}")
    return w_tr, z_tr, w_va, z_va, w_te, z_te


def discover_available_seeds(
    ckpt_root: Path,
    *,
    rho: float,
) -> list[int]:
    """Discover seeds present in `ckpt_root` for one rho from IV checkpoint names."""
    pattern = re.compile(rf"^s(\d+)_rho{re.escape(rho_tag(rho))}_.+_iv\.pt$")
    seeds: set[int] = set()
    for path in ckpt_root.glob(f"s*_rho{rho_tag(rho)}_*_iv.pt"):
        match = pattern.match(path.name)
        if match:
            seeds.add(int(match.group(1)))
    return sorted(seeds)
