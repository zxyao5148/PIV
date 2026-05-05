#!/usr/bin/env python3
"""Train IV-representation models + NPIV estimators on semi-synthetic IBM/LBIDD data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_configs import add_model_config_arguments, configured_model_args
from data.sim_ibm import DEFAULT_COVARIATE_PATH, DEFAULT_L_COLUMNS
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
from trainer import fit_npiv  # noqa: E402
from utils import collect_loader, seed_everything  # noqa: E402

from scripts.ibm.common import (  # noqa: E402
    ALL_METHODS,
    attach_ibm_fingerprint_compat,
    build_ibm_bundle,
    build_iv_checkpoint_path,
    build_npiv_checkpoint_path,
    ibm_fingerprint_extra,
    load_or_train_iv,
)


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
