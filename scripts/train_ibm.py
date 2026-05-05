#!/usr/bin/env python3
"""Train IV-representation models + NPIV estimators on semi-synthetic IBM/LBIDD data."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_configs import add_model_config_arguments, configured_model_args
from data.sim_ibm import (
    DEFAULT_COVARIATE_PATH,
    DEFAULT_L_COLUMNS,
    SimConfig,
    build_dgp_definition,
    build_ibm_latent_blocks,
    load_ibm_covariates,
    sample_empirical_rows,
    simulate_ibm_dataset,
)
from data.sim_npiv import structural_response
from evaluation import (  # noqa: E402
    build_counterfactual_eval,
    contrast_error,
    contrast_metric_label,
    hat_contrast_matrix,
    probe_w_to_target_r2,
    probe_wz_to_target_r2,
    probe_z_to_target_r2,
    true_contrast_matrix,
)
from synthetic_dataset import make_synthetic_proxyiv_dataloaders  # noqa: E402
from trainer import fit_npiv, load_iv_representation, train_iv_representation  # noqa: E402
from utils import IVDims, collect_loader, make_eval_loaders, method_tag, seed_everything  # noqa: E402

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
        "surface": str(args.surface),
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
    oracle_suffix = "_oracleC" if oracle_c and method not in {"PIV"} else ""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semi-synthetic IBM/LBIDD NPIV benchmark for one run.",
    )

    # Dataset / DGP
    parser.add_argument(
        "--surface",
        type=str,
        default="poly",
        choices=[
            "poly", "varying_slope", "soft_threshold", "sinusoidal",
            "binary_linear", "binary_threshold_modulated",
        ],
    )
    parser.add_argument(
        "--treatment-type",
        type=str,
        default="continuous",
        choices=["continuous", "binary"],
        dest="treatment_type",
    )
    parser.add_argument("--design-seed", type=int, default=1, dest="design_seed")
    parser.add_argument("--data-seed", type=int, default=1, dest="data_seed")
    parser.add_argument("--autoenc-seed", type=int, default=1, dest="autoenc_seed")
    parser.add_argument("--train-seed", type=int, default=1, dest="train_seed")

    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--dC", type=int, default=2)
    parser.add_argument("--dV", type=int, default=2)
    parser.add_argument("--dD", type=int, default=2)
    parser.add_argument("--dL", type=int, default=None)
    parser.add_argument("--dM", type=int, default=None)
    parser.add_argument("--dA", type=int, default=1)
    parser.add_argument("--sigma-m", type=float, default=0.50, dest="sigma_m")
    parser.add_argument("--sigma-a", type=float, default=0.50, dest="sigma_a")
    parser.add_argument("--sigma-y", type=float, default=0.50, dest="sigma_y")
    parser.add_argument(
        "--min-continuous-confound-ratio",
        type=float,
        default=0.50,
        dest="min_continuous_confound_ratio",
    )
    parser.add_argument("--max-treat-tries", type=int, default=2000, dest="max_treat_tries")
    parser.add_argument("--binary-disable-joint-overlap-filter", action="store_true")
    parser.add_argument("--binary-min-joint-cell-prob-p01", type=float, default=0.0025)
    parser.add_argument("--binary-min-joint-cell-prob-p05", type=float, default=0.0100)
    parser.add_argument("--binary-max-joint-cell-prob-p95", type=float, default=0.9900)

    # IBM covariate compression
    parser.add_argument(
        "--covariate-path",
        type=str,
        default=str(DEFAULT_COVARIATE_PATH.relative_to(PROJECT_ROOT)),
        dest="covariate_path",
    )
    parser.add_argument(
        "--l-columns",
        type=str,
        default=",".join(DEFAULT_L_COLUMNS),
        dest="l_columns",
    )
    parser.add_argument("--vae-epochs", "--ae-epochs", type=int, default=40, dest="ae_epochs")
    parser.add_argument("--vae-batch-size", "--ae-batch-size", type=int, default=512, dest="ae_batch_size")
    parser.add_argument("--vae-lr", "--ae-lr", type=float, default=1e-3, dest="ae_lr")
    parser.add_argument("--vae-beta", type=float, default=1.0, dest="vae_beta")
    parser.add_argument("--vae-hidden-dim", "--ae-hidden-dim", type=int, default=128, dest="ae_hidden_dim")
    parser.add_argument("--vae-depth", "--ae-depth", type=int, default=2, dest="ae_depth")
    parser.add_argument("--vae-max-categories", "--ae-max-categories", type=int, default=20, dest="ae_max_categories")
    parser.add_argument("--vae-count-max", "--ae-count-max", type=int, default=100, dest="ae_count_max")
    parser.add_argument("--vae-count-skew-threshold", type=float, default=1.0, dest="vae_count_skew_threshold")
    parser.add_argument(
        "--vae-count-loss",
        "--ae-count-loss",
        type=str,
        choices=["poisson", "negative_binomial"],
        default="poisson",
        dest="ae_count_loss",
    )
    parser.add_argument("--vae-count-weight", "--ae-poisson-weight", type=float, default=0.25, dest="ae_poisson_weight")
    parser.add_argument("--vae-nb-theta", "--ae-nb-theta", type=float, default=5.0, dest="ae_nb_theta")
    parser.add_argument("--resid-epochs", type=int, default=120, dest="resid_epochs")
    parser.add_argument("--resid-batch-size", type=int, default=512, dest="resid_batch_size")
    parser.add_argument("--resid-lr", type=float, default=1e-3, dest="resid_lr")
    parser.add_argument("--resid-hidden-dim", type=int, default=128, dest="resid_hidden_dim")
    parser.add_argument("--resid-depth", type=int, default=2, dest="resid_depth")

    parser.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    parser.add_argument("--train-frac", type=float, default=0.70, dest="train_frac")
    parser.add_argument("--val-frac", type=float, default=0.15, dest="val_frac")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Models
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["TrueIV"],
        choices=ALL_METHODS,
        metavar="METHOD",
    )
    parser.add_argument(
        "--downstream",
        nargs="+",
        default=["deepiv", "dfiv"],
        choices=["deepiv", "dfiv"],
    )
    parser.add_argument(
        "--skip-downstream",
        action="store_true",
        dest="skip_downstream",
        help="Train/evaluate IV representations and probes only.",
    )
    add_model_config_arguments(
        parser,
        include_iv_representation=True,
        include_npiv_estimators=True,
    )

    # Evaluation / Reporting
    parser.add_argument("--probe-epochs", type=int, default=240, dest="probe_epochs")
    parser.add_argument("--probe-lr", type=float, default=5e-4, dest="probe_lr")
    parser.add_argument("--contrast-ref-a", type=str, default="", dest="contrast_ref_a")
    parser.add_argument("--cf-grid-n", type=int, default=25, dest="cf_grid_n")
    parser.add_argument("--cf-a-pctl-low", type=float, default=5.0, dest="cf_a_pctl_low")
    parser.add_argument("--cf-a-pctl-high", type=float, default=95.0, dest="cf_a_pctl_high")
    parser.add_argument(
        "--mse-scale",
        type=str,
        default="var",
        choices=["raw", "var", "mean_sq", "range_sq"],
        dest="mse_scale",
        help="Scale for contrast error reporting; default is variance-normalized nMSE.",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "ibm"),
        dest="checkpoint_dir",
    )
    parser.add_argument(
        "--oracle-c",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="oracle_c",
        help="Replace learned W with oracle C except for PIV (default: on).",
    )
    parser.add_argument("--force-retrain", action="store_true", dest="force_retrain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rng = seed_everything(int(args.train_seed))

    ckpt_root = Path(args.checkpoint_dir).expanduser().resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    bundle = build_ibm_bundle(args, device)
    raw = bundle["raw"]
    dims = bundle["dims"]
    loaders = bundle["loaders"]
    eval_loaders = bundle["eval_loaders"]
    h0_fn = bundle["h0_fn"]
    treatment_type = str(bundle["treatment_type"])

    train_raw = collect_loader(eval_loaders.train, keys=["A", "Y", "L"])
    val_raw = collect_loader(eval_loaders.val, keys=["A", "Y", "L"])
    test_raw = collect_loader(eval_loaders.test, keys=["L"])
    probe_train = collect_loader(eval_loaders.train, keys=["C", "V", "D"])
    probe_test = collect_loader(eval_loaders.test, keys=["C", "V", "D"])

    if args.oracle_c:
        oracle_c_tr = collect_loader(eval_loaders.train, keys=["C"])["C"].float()
        oracle_c_va = collect_loader(eval_loaders.val, keys=["C"])["C"].float()
        oracle_c_te = collect_loader(eval_loaders.test, keys=["C"])["C"].float()
    else:
        oracle_c_tr = oracle_c_va = oracle_c_te = None

    l_te = test_raw["L"].float()
    a_ref_override = float(str(args.contrast_ref_a).strip()) if str(args.contrast_ref_a).strip() else None
    a_ref, a_grid, _ = build_counterfactual_eval(
        np.asarray(raw["A"], dtype=np.float64),
        np.asarray(raw["L"], dtype=np.float64),
        h0_fn,
        treatment_type=treatment_type,
        a_ref_override=a_ref_override,
        cf_a_pctl_low=float(args.cf_a_pctl_low),
        cf_a_pctl_high=float(args.cf_a_pctl_high),
        cf_grid_n=int(args.cf_grid_n),
    )
    true_tau = true_contrast_matrix(a_grid, a_ref, l_te.numpy(), h0_fn)

    dA = int(args.dA)
    a_ref_str = (
        f"{float(a_ref):.4f}" if np.ndim(a_ref) == 0 else
        np.array2string(np.asarray(a_ref), precision=4, separator=",")
    )
    a_grid_0 = a_grid[0] if a_grid.ndim == 1 else a_grid[0, 0]
    a_grid_last = a_grid[-1] if a_grid.ndim == 1 else a_grid[-1, 0]

    print(f"Checkpoints: {ckpt_root}")
    print(
        f"\n=== IBM run: treatment_type={treatment_type}  surface={args.surface}  "
        f"design_seed={args.design_seed}  data_seed={args.data_seed}  "
        f"autoenc_seed={args.autoenc_seed}  train_seed={args.train_seed}  dA={dA} ===",
    )
    print(
        f"Reference a0={a_ref_str}  "
        f"grid=[{a_grid_0:.3f}, {a_grid_last:.3f}] (n={len(a_grid)})",
    )

    mse_by_key: dict[tuple[str, str], float] = {}
    tau_hat_by_key: dict[tuple[str, str], np.ndarray] = {}
    metric_label = contrast_metric_label(args.mse_scale)
    probe_by_method: dict[tuple[str, str], float] = {}

    for method in list(dict.fromkeys(args.methods)):
        model_args = configured_model_args(
            args,
            dataset="ibm",
            iv_method=method,
            dA=int(args.dA),
        )
        attach_ibm_fingerprint_compat(model_args)
        fp_extra = ibm_fingerprint_extra(model_args)

        iv_ckpt_path = build_iv_checkpoint_path(
            ckpt_root,
            surface=str(args.surface),
            treatment_type=treatment_type,
            design_seed=int(args.design_seed),
            data_seed=int(args.data_seed),
            autoenc_seed=int(args.autoenc_seed),
            train_seed=int(args.train_seed),
            method=method,
            dA=dA,
        )
        w_tr, z_tr, w_va, z_va, w_te, z_te = load_or_train_iv(
            method,
            iv_ckpt_path,
            loaders,
            eval_loaders,
            model_args,
            dims,
            device,
            int(args.train_seed),
            rng,
            allow_train=True,
            fingerprint_extra=fp_extra,
        )

        if args.oracle_c and method not in {"PIV"}:
            assert oracle_c_tr is not None
            w_tr = oracle_c_tr
            w_va = oracle_c_va
            w_te = oracle_c_te

        w_tr_t = torch.as_tensor(w_tr, dtype=torch.float32)
        z_tr_t = torch.as_tensor(z_tr, dtype=torch.float32)
        w_te_t = torch.as_tensor(w_te, dtype=torch.float32)
        z_te_t = torch.as_tensor(z_te, dtype=torch.float32)
        probe_specs = [
            ("W->C", probe_w_to_target_r2, probe_train["C"], probe_test["C"]),
            ("W->V", probe_w_to_target_r2, probe_train["V"], probe_test["V"]),
            ("W->D", probe_w_to_target_r2, probe_train["D"], probe_test["D"]),
            ("Z->V", probe_z_to_target_r2, probe_train["V"], probe_test["V"]),
            ("Z->D", probe_z_to_target_r2, probe_train["D"], probe_test["D"]),
            ("(W,Z)->V", probe_wz_to_target_r2, probe_train["V"], probe_test["V"]),
            ("(W,Z)->D", probe_wz_to_target_r2, probe_train["D"], probe_test["D"]),
        ]
        for label, probe_fn, tgt_tr_raw, tgt_te_raw in probe_specs:
            tgt_tr = tgt_tr_raw.to(torch.float32)
            tgt_te = tgt_te_raw.to(torch.float32)
            if probe_fn is probe_w_to_target_r2:
                r2 = probe_fn(
                    w_tr_t,
                    tgt_tr,
                    w_te_t,
                    tgt_te,
                    epochs=int(model_args.probe_epochs),
                    lr=float(model_args.probe_lr),
                    device=device,
                )
            elif probe_fn is probe_z_to_target_r2:
                r2 = probe_fn(
                    z_tr_t,
                    tgt_tr,
                    z_te_t,
                    tgt_te,
                    epochs=int(model_args.probe_epochs),
                    lr=float(model_args.probe_lr),
                    device=device,
                )
            else:
                r2 = probe_fn(
                    w_tr_t,
                    z_tr_t,
                    tgt_tr,
                    w_te_t,
                    z_te_t,
                    tgt_te,
                    epochs=int(model_args.probe_epochs),
                    lr=float(model_args.probe_lr),
                    device=device,
                )
            probe_by_method[(method, label)] = float(r2)
            print(f"    [probe] {label}: R2 = {r2:.4f}")

        tr_np = {
            "A": train_raw["A"].numpy(),
            "Y": train_raw["Y"].numpy(),
            "L": train_raw["L"].numpy(),
        }
        val_np = {
            "A": val_raw["A"].numpy(),
            "Y": val_raw["Y"].numpy(),
            "L": val_raw["L"].numpy(),
        }

        if bool(args.skip_downstream):
            continue

        for downstream in list(dict.fromkeys(args.downstream)):
            print(f"    [{downstream}] NPIV ...")
            npiv_ckpt_path = build_npiv_checkpoint_path(
                ckpt_root,
                surface=str(args.surface),
                treatment_type=treatment_type,
                design_seed=int(args.design_seed),
                data_seed=int(args.data_seed),
                autoenc_seed=int(args.autoenc_seed),
                train_seed=int(args.train_seed),
                method=method,
                downstream=downstream,
                oracle_c=bool(args.oracle_c),
                dA=dA,
            )
            model, norms = fit_npiv(
                downstream,
                tr_np,
                w_tr,
                z_tr,
                model_args,
                device,
                dims,
                val_raw=val_np,
                w_val=w_va,
                z_val=z_va,
                checkpoint_path=npiv_ckpt_path,
                iv_method=method,
                seed=int(args.train_seed),
                force_retrain=bool(model_args.force_retrain),
                test=False,
                fingerprint_extra=fp_extra,
            )
            tau_hat = hat_contrast_matrix(
                model,
                a_grid,
                a_ref,
                w_te,
                l_te,
                device,
                norms,
            )
            mse = contrast_error(tau_hat, true_tau, scale=args.mse_scale)
            mse_by_key[(method, downstream)] = float(mse)
            tau_hat_by_key[(method, downstream)] = tau_hat
            print(f"       contrast {metric_label} = {mse:.6f}")

    print(f"\n{'=' * 72}")
    print(
        f"[IBM surface={args.surface}, design={args.design_seed}, "
        f"data={args.data_seed}, ae={args.autoenc_seed}, train={args.train_seed}] "
        f"Unit-level treatment contrast {metric_label}",
    )
    print(f"Reference a0 = {a_ref_str}")
    print(f"{'=' * 72}")
    for method in list(dict.fromkeys(args.methods)):
        print(
            f"  {method:10}  "
            f"W->C={probe_by_method[(method, 'W->C')]:.4f}  "
            f"W->V={probe_by_method[(method, 'W->V')]:.4f}  "
            f"W->D={probe_by_method[(method, 'W->D')]:.4f}  "
            f"Z->V={probe_by_method[(method, 'Z->V')]:.4f}  "
            f"Z->D={probe_by_method[(method, 'Z->D')]:.4f}  "
            f"(W,Z)->V={probe_by_method[(method, '(W,Z)->V')]:.4f}  "
            f"(W,Z)->D={probe_by_method[(method, '(W,Z)->D')]:.4f}",
        )
        if not bool(args.skip_downstream):
            for downstream in list(dict.fromkeys(args.downstream)):
                mse = mse_by_key[(method, downstream)]
                print(f"  {method:10}  {downstream:8}  {mse:.6f}")


if __name__ == "__main__":
    main()
