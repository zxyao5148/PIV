#!/usr/bin/env python3
"""Shared utilities for the semi-synthetic IBM/LBIDD benchmark scripts."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sim_ibm import (  # noqa: E402
    DEFAULT_COVARIATE_PATH,
    DEFAULT_L_COLUMNS,
    SimConfig,
    build_dgp_definition,
    build_ibm_latent_blocks,
    load_ibm_covariates,
    sample_empirical_rows,
    simulate_ibm_dataset,
)
from data.sim_npiv import structural_response  # noqa: E402
from synthetic_dataset import make_synthetic_proxyiv_dataloaders  # noqa: E402
from trainer import load_iv_representation, train_iv_representation  # noqa: E402
from utils import IVDims, make_eval_loaders, method_tag  # noqa: E402

ALL_METHODS = [
    "PIV",
    "Proxy-Orth",
    "ZNet",
    "AutoIV",
    "VIV",
    "DIV.VAE",
    "TrueIV",
    "NoneIV",
]


def parse_column_list(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(DEFAULT_L_COLUMNS)
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def attach_ibm_fingerprint_compat(args: argparse.Namespace) -> None:
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


def ibm_fingerprint_extra(args: argparse.Namespace) -> dict:
    """Dataset-specific fingerprint fields for IBM/LBIDD checkpoints."""
    return {
        "design_seed": int(args.design_seed),
        "data_seed": int(args.data_seed),
        "autoenc_seed": int(args.autoenc_seed),
        "treatment_type": str(args.treatment_type),
        "dA": int(getattr(args, "dA", 1)),
        "l_columns": tuple(parse_column_list(args.l_columns)),
        "covariate_path": str(args.covariate_path),
        "vae_epochs": int(args.ae_epochs),
        "vae_beta": float(args.vae_beta),
        "vae_hidden_dim": int(args.ae_hidden_dim),
        "vae_depth": int(args.ae_depth),
        "vae_count_loss": str(args.ae_count_loss),
        "vae_count_skew_threshold": float(args.vae_count_skew_threshold),
        "resid_epochs": int(args.resid_epochs),
        "resid_hidden_dim": int(args.resid_hidden_dim),
        "resid_depth": int(args.resid_depth),
    }


def sim_arrays(dataset) -> dict[str, np.ndarray]:
    """Convert the simulated dataclass payload to plain NumPy arrays."""
    raw = asdict(dataset)
    keys = {
        "X", "M", "L", "A", "Y", "C", "V", "D", "U",
        "structural_response_obs", "confounder_nuisance_obs", "structural_noise_obs",
        "D_residualized", "sample_id",
    }
    scalar_keys = {
        "A",
        "Y",
        "U",
        "structural_response_obs",
        "confounder_nuisance_obs",
        "structural_noise_obs",
    }
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in raw:
            continue
        arr = np.asarray(raw[key])
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


def build_ibm_bundle(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Construct one IBM/LBIDD semi-synthetic dataset and loader bundle."""
    l_columns = parse_column_list(args.l_columns)
    cfg = SimConfig(
        n=int(args.n),
        dC=int(args.dC),
        dV=int(args.dV),
        dD=int(args.dD),
        dL=len(l_columns) if args.dL is None else args.dL,
        dM=args.dM,
        dA=int(getattr(args, "dA", 1)),
        sigma_m=float(args.sigma_m),
        sigma_a=float(args.sigma_a),
        sigma_y=float(args.sigma_y),
        max_treat_tries=int(args.max_treat_tries),
        min_continuous_confound_ratio=float(args.min_continuous_confound_ratio),
        binary_enable_joint_overlap_filter=not bool(args.binary_disable_joint_overlap_filter),
        binary_min_joint_cell_prob_p01=float(args.binary_min_joint_cell_prob_p01),
        binary_min_joint_cell_prob_p05=float(args.binary_min_joint_cell_prob_p05),
        binary_max_joint_cell_prob_p95=float(args.binary_max_joint_cell_prob_p95),
    )
    if cfg.dL != len(l_columns):
        raise ValueError(f"dL={cfg.dL} but --l-columns has {len(l_columns)} columns.")

    covariates = sample_empirical_rows(
        load_ibm_covariates(args.covariate_path),
        n=int(args.n),
        data_seed=int(args.data_seed),
    )
    L, C, V, D, metadata = build_ibm_latent_blocks(
        covariates,
        l_columns=l_columns,
        cfg=cfg,
        design_seed=int(args.design_seed),
        autoenc_seed=int(args.autoenc_seed),
        ae_epochs=int(args.ae_epochs),
        ae_batch_size=int(args.ae_batch_size),
        ae_lr=float(args.ae_lr),
        vae_beta=float(args.vae_beta),
        ae_hidden_dim=int(args.ae_hidden_dim),
        ae_depth=int(args.ae_depth),
        ae_max_categories=int(args.ae_max_categories),
        ae_count_max=int(args.ae_count_max),
        ae_count_skew_threshold=float(args.vae_count_skew_threshold),
        ae_count_loss=str(args.ae_count_loss),
        ae_poisson_weight=float(args.ae_poisson_weight),
        ae_nb_theta=float(args.ae_nb_theta),
        device=device,
    )
    dgp = build_dgp_definition(
        int(args.design_seed),
        str(args.surface),
        cfg,
        treatment_type=str(args.treatment_type),
    )
    dataset = simulate_ibm_dataset(
        int(args.data_seed),
        dgp,
        cfg,
        L=L,
        C=C,
        V=V,
        D=D,
        sample_id=covariates["sample_id"].to_numpy() if "sample_id" in covariates else np.arange(cfg.n),
        l_columns=l_columns,
        block_metadata=metadata,
        residualizer_seed=int(args.autoenc_seed) + 10_000,
        residualizer_epochs=int(args.resid_epochs),
        residualizer_batch_size=int(args.resid_batch_size),
        residualizer_lr=float(args.resid_lr),
        residualizer_hidden_dim=int(args.resid_hidden_dim),
        residualizer_depth=int(args.resid_depth),
        device=device,
    )
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
    h0_fn = lambda a, l: structural_response(a, l, dgp, cfg=cfg)  # noqa: E731
    return {
        "cfg": cfg,
        "dgp": dgp,
        "dataset": dataset,
        "metadata": metadata,
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
    autoenc_seed: int,
    train_seed: int,
    method: str,
    dA: int = 1,
) -> Path | None:
    """Canonical IV checkpoint path for one IBM semi-synthetic run."""
    if ckpt_root is None:
        return None
    da_suffix = f"_dA{int(dA)}" if int(dA) > 1 else ""
    return (
        ckpt_root
        / f"surf_{surface}_{treatment_type}_design{int(design_seed)}"
        f"_data{int(data_seed)}_ae{int(autoenc_seed)}_train{int(train_seed)}"
        f"_{method_tag(method)}{da_suffix}_iv.pt"
    )


def build_npiv_checkpoint_path(
    ckpt_root: Path | None,
    *,
    surface: str,
    treatment_type: str,
    design_seed: int,
    data_seed: int,
    autoenc_seed: int,
    train_seed: int,
    method: str,
    downstream: str,
    oracle_c: bool = False,
    dA: int = 1,
) -> Path | None:
    """Canonical NPIV checkpoint path for one IBM semi-synthetic run."""
    if ckpt_root is None:
        return None
    oracle_suffix = "_oracleC" if oracle_c and method not in {"PIV", "Proxy-Orth"} else ""
    da_suffix = f"_dA{int(dA)}" if int(dA) > 1 else ""
    return (
        ckpt_root
        / f"surf_{surface}_{treatment_type}_design{int(design_seed)}"
        f"_data{int(data_seed)}_ae{int(autoenc_seed)}_train{int(train_seed)}"
        f"_{method_tag(method)}_{downstream}{oracle_suffix}{da_suffix}_npiv.pt"
    )


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
            print(f"\n  [{method}] IV checkpoint unusable ({exc!r}) - retraining.")

    if not allow_train:
        raise SystemExit(f"missing IV checkpoint for {method}: {iv_ckpt_path}")

    print(f"\n  [{method}] training representation ...")
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
