#!/usr/bin/env python3
"""Shared utilities for the synthetic NPIV benchmark scripts."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sim_npiv import (  # noqa: E402
    ALL_SURFACES,
    BINARY_SURFACES,
    CONTINUOUS_SURFACES,
    SimConfig,
    build_dgp_definition,
    simulate_dataset,
    structural_response,
)
from evaluation import build_counterfactual_eval  # noqa: E402
from synthetic_dataset import make_synthetic_proxyiv_dataloaders  # noqa: E402
from trainer import load_iv_representation, train_iv_representation  # noqa: E402
from utils import IVDims, make_eval_loaders, method_tag  # noqa: E402

ALL_METHODS = ["PIV", "ZNet", "AutoIV", "VIV", "DIV.VAE", "TrueIV", "NoneIV"]


def attach_npiv_fingerprint_compat(args: argparse.Namespace) -> None:
    """Provide shared checkpoint-fingerprint fields expected by ``fit_npiv``."""
    if not hasattr(args, "lr") or args.lr is None:
        args.lr = float(args.deepiv_lr)
    if not hasattr(args, "s1_epochs") or args.s1_epochs is None:
        args.s1_epochs = 0
    if not hasattr(args, "s2_epochs") or args.s2_epochs is None:
        args.s2_epochs = 0
    if not hasattr(args, "s1_patience") or args.s1_patience is None:
        args.s1_patience = 0
    if not hasattr(args, "s2_patience") or args.s2_patience is None:
        args.s2_patience = 0
    if not hasattr(args, "epochs") or args.epochs is None:
        args.epochs = 0
    if not hasattr(args, "proxy_embed_dim") or args.proxy_embed_dim is None:
        args.proxy_embed_dim = 0
    if not hasattr(args, "treatment_type") or args.treatment_type is None:
        args.treatment_type = "continuous"


def npiv_syn_fingerprint_extra(args: argparse.Namespace) -> dict:
    """Dataset-specific fingerprint fields for the synthetic NPIV benchmark."""
    return {
        "design_seed": int(args.design_seed),
        "data_seed": int(args.data_seed),
        "treatment_type": str(args.treatment_type),
        "dA": int(getattr(args, "dA", 1)),
    }


def sim_arrays(dataset) -> dict[str, np.ndarray]:
    """Convert the simulated dataclass payload to plain NumPy arrays."""
    raw = asdict(dataset)
    arrays: dict[str, np.ndarray] = {}
    scalar_keys = {
        "A",
        "Y",
        "U",
        "structural_response_obs",
        "confounder_nuisance_obs",
        "structural_noise_obs",
    }
    for key, value in raw.items():
        arr = np.asarray(value)
        if key in scalar_keys and arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays[key] = arr
    return arrays


def build_dims(raw: dict[str, np.ndarray]) -> IVDims:
    """Infer benchmark dimensions from one simulated dataset."""
    return IVDims(
        x_dim=int(raw["X"].shape[1]),
        m_dim=int(raw["M"].shape[1]),
        w_dim=int(raw["C"].shape[1]),
        z_dim=int(raw["V"].shape[1]),
        a_dim=int(raw["A"].shape[1]),
        l_dim=int(raw["L"].shape[1]),
        y_dim=1,
    )


def resolve_surface_l_grid(values: np.ndarray, override: str) -> np.ndarray:
    """Resolve the L-axis used for the saved conditional contrast surface."""
    if str(override).strip():
        toks = [tok.strip() for tok in str(override).split(",") if tok.strip()]
        if not toks:
            raise ValueError("--surface-l-values was provided but empty after parsing.")
        return np.asarray([float(tok) for tok in toks], dtype=np.float64)
    uniq = np.unique(np.asarray(values, dtype=np.float64))
    if uniq.size >= 2:
        return np.sort(uniq.astype(np.float64))
    return np.asarray([0.0, 1.0], dtype=np.float64)


def build_npiv_syn_bundle(args: argparse.Namespace) -> dict[str, object]:
    """Simulate one synthetic NPIV dataset and construct all loader objects."""
    cfg = SimConfig(
        n=int(args.n),
        dC=int(args.dC),
        dV=int(args.dV),
        dD=int(args.dD),
        dL=args.dL,
        dM=args.dM,
        dA=int(getattr(args, "dA", 1)),
        sigma_m=float(args.sigma_m),
        sigma_a=float(args.sigma_a),
        sigma_y=float(args.sigma_y),
    )
    dgp = build_dgp_definition(
        int(args.design_seed),
        str(args.surface),
        cfg,
        treatment_type=str(args.treatment_type),
    )
    dataset = simulate_dataset(int(args.data_seed), dgp, cfg)
    raw = sim_arrays(dataset)
    dims = build_dims(raw)
    keys = [
        "X", "M", "A", "Y", "L",
        "C", "V", "D", "U",
        "structural_response_obs", "confounder_nuisance_obs", "structural_noise_obs",
    ]
    loaders = make_synthetic_proxyiv_dataloaders(
        arrays=raw,
        keys=keys,
        required_keys=keys,
        batch_size=int(args.batch_size),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        seed=int(args.train_seed),
    )
    eval_loaders = make_eval_loaders(
        loaders,
        eval_batch_size=max(int(args.batch_size), 1024),
    )
    h0_fn = lambda a, l: structural_response(a, l, dgp)  # noqa: E731
    return {
        "cfg": cfg,
        "dgp": dgp,
        "dataset": dataset,
        "raw": raw,
        "dims": dims,
        "loaders": loaders,
        "eval_loaders": eval_loaders,
        "h0_fn": h0_fn,
        "treatment_type": str(dgp.treatment_type),
    }


def build_iv_checkpoint_path(
    ckpt_root: Path | None,
    *,
    surface: str,
    treatment_type: str,
    design_seed: int,
    data_seed: int,
    train_seed: int,
    method: str,
    dA: int = 1,
    run_tag: str = "",
) -> Path | None:
    """Canonical IV checkpoint path for one synthetic `(surface, seeds, method)`."""
    if ckpt_root is None:
        return None
    da_suffix = f"_dA{int(dA)}" if int(dA) > 1 else ""
    stem = (
        f"surf_{surface}"
        f"_{treatment_type}"
        f"_design{int(design_seed)}"
        f"_data{int(data_seed)}"
        f"_train{int(train_seed)}"
        f"_{method_tag(method)}{da_suffix}_iv.pt"
    )
    if str(run_tag).strip():
        stem = stem[:-3] + f"_{str(run_tag).strip()}.pt"
    return ckpt_root / stem


def build_npiv_checkpoint_path(
    ckpt_root: Path | None,
    *,
    surface: str,
    treatment_type: str,
    design_seed: int,
    data_seed: int,
    train_seed: int,
    method: str,
    downstream: str,
    oracle_c: bool = False,
    dA: int = 1,
    run_tag: str = "",
) -> Path | None:
    """Canonical NPIV checkpoint path for one synthetic `(surface, seeds, combo)`."""
    if ckpt_root is None:
        return None
    oracle_suffix = "_oracleC" if oracle_c else ""
    da_suffix = f"_dA{int(dA)}" if int(dA) > 1 else ""
    stem = (
        f"surf_{surface}"
        f"_{treatment_type}"
        f"_design{int(design_seed)}"
        f"_data{int(data_seed)}"
        f"_train{int(train_seed)}"
        f"_{method_tag(method)}_{downstream}{oracle_suffix}{da_suffix}_npiv.pt"
    )
    if str(run_tag).strip():
        stem = stem[:-3] + f"_{str(run_tag).strip()}.pt"
    return ckpt_root / stem


def artifact_stem(
    *,
    surface: str,
    treatment_type: str,
    design_seed: int,
    data_seed: int,
    train_seed: int,
    method: str,
    downstream: str,
    dA: int = 1,
    run_tag: str = "",
) -> str:
    """Filename stem shared by figures, summaries, and NPZ artifacts."""
    da_suffix = f"_dA{int(dA)}" if int(dA) > 1 else ""
    stem = (
        f"surf_{surface}"
        f"_{treatment_type}"
        f"_design{int(design_seed)}"
        f"_data{int(data_seed)}"
        f"_train{int(train_seed)}"
        f"_{method_tag(method)}_{downstream}{da_suffix}"
    )
    if str(run_tag).strip():
        stem = f"{stem}_{str(run_tag).strip()}"
    return stem


def load_or_train_iv(
    method: str,
    iv_ckpt_path: Path | None,
    loaders,
    eval_loaders,
    args: argparse.Namespace,
    dims: IVDims,
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
                dims,
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
        dims,
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


def select_unit_plot_index(a_grid: np.ndarray, requested_index: int) -> int:
    """Resolve the unit-plot slice index: midpoint by default, else clamped to grid."""
    if int(requested_index) >= 0:
        return min(int(requested_index), len(a_grid) - 1)
    return max(0, len(a_grid) // 2)


def build_npiv_syn_eval_setup(
    args: argparse.Namespace,
    *,
    surface: str,
    treatment_type: str,
) -> dict[str, object]:
    """Build stable evaluation geometry from a large fixed population draw.

    Uses ``args.pop_n`` / ``args.pop_seed`` with the same DGP structure
    (``args.design_seed``, ``surface``, ``treatment_type``) so that ``a_ref``
    and ``a_grid`` are identical across different ``train_seed`` replicates.
    """
    pop_cfg = SimConfig(
        n=int(args.pop_n),
        dC=int(args.dC),
        dV=int(args.dV),
        dD=int(args.dD),
        dL=args.dL,
        dM=args.dM,
        dA=int(getattr(args, "dA", 1)),
        sigma_m=float(args.sigma_m),
        sigma_a=float(args.sigma_a),
        sigma_y=float(args.sigma_y),
    )
    dgp = build_dgp_definition(
        int(args.design_seed),
        surface,
        pop_cfg,
        treatment_type=treatment_type,
    )
    dataset = simulate_dataset(int(args.pop_seed), dgp, pop_cfg)
    raw = sim_arrays(dataset)
    h0_fn = lambda a, l: structural_response(a, l, dgp)  # noqa: E731

    override = str(getattr(args, "contrast_ref_a", "")).strip()
    a_ref_override = float(override) if override else None
    a_ref, a_grid, tau_pop_true = build_counterfactual_eval(
        np.asarray(raw["A"], dtype=np.float64),
        np.asarray(raw["L"], dtype=np.float64),
        h0_fn,
        treatment_type=treatment_type,
        a_ref_override=a_ref_override,
        cf_a_pctl_low=float(getattr(args, "cf_a_pctl_low", 5.0)),
        cf_a_pctl_high=float(getattr(args, "cf_a_pctl_high", 95.0)),
        cf_grid_n=int(getattr(args, "cf_grid_n", 25)),
    )
    pop_l = np.asarray(raw["L"], dtype=np.float64)
    return {
        "a_ref": a_ref,  # float (scalar) or (dA,) array (vector continuous)
        "a_grid": a_grid,
        "tau_pop_true": tau_pop_true,
        "h0_fn": h0_fn,
        "pop_l": pop_l,
    }


def discover_available_train_seeds(
    ckpt_root: Path,
    *,
    surface: str,
    treatment_type: str,
    design_seed: int,
    data_seed: int,
    dA: int = 1,
) -> list[int]:
    """Find which train_seeds have at least one IV checkpoint for this scenario."""
    da_suffix = f"_dA{int(dA)}" if int(dA) > 1 else ""
    pattern = re.compile(
        rf"^surf_{re.escape(surface)}_{re.escape(treatment_type)}"
        rf"_design{int(design_seed)}_data{int(data_seed)}_train(\d+)_"
        rf".+{re.escape(da_suffix)}_iv\.pt$"
    )
    seeds: set[int] = set()
    prefix = (
        f"surf_{surface}_{treatment_type}"
        f"_design{int(design_seed)}_data{int(data_seed)}_train"
    )
    for path in ckpt_root.glob(f"{prefix}*_*_iv.pt"):
        m = pattern.match(path.name)
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds)
