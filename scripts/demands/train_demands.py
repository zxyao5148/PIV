#!/usr/bin/env python3
"""Single-run demands benchmark.
"""
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
from evaluation import (
    contrast_error,
    contrast_metric_label,
    hat_contrast_matrix,
    probe_w_to_target_r2,
    probe_wz_to_target_r2,
    probe_z_to_target_r2,
    true_contrast_matrix,
)
from trainer import fit_npiv
from utils import collect_loader, seed_everything

from scripts.demands.common import (
    ALL_METHODS,
    DEMANDS_DIMS,
    PROJECT_ROOT,
    build_demands_loaders,
    build_eval_setup,
    build_iv_checkpoint_path,
    build_npiv_checkpoint_path,
    demands_fingerprint_extra,
    demands_h0,
    load_or_train_iv,
)


def parse_args() -> argparse.Namespace:
    """CLI for one demands run."""
    parser = argparse.ArgumentParser(
        description="Demands benchmark for one seed and one rho.",
    )

    # ── Run identity ────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.5)

    # ── DGP ─────────────────────────────────────────────────────────
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--pop-n", type=int, default=200_000, dest="pop_n")
    parser.add_argument("--pop-seed", type=int, default=20250101, dest="pop_seed")
    parser.add_argument("--coef-eps-a", type=float, default=1.0, dest="coef_eps_a")
    parser.add_argument("--coef-u-outcome", type=float, default=10.0, dest="coef_u_outcome")
    parser.add_argument("--coef-v-treatment", type=float, default=1.0, dest="coef_v_treatment")
    parser.add_argument("--coef-c-treatment", type=float, default=1.0, dest="coef_c_treatment")
    parser.add_argument("--coef-c-outcome", type=float, default=1.0, dest="coef_c_outcome")
    parser.add_argument("--coef-d-outcome", type=float, default=1.0, dest="coef_d_outcome")
    parser.add_argument("--contrast-ref-a", type=str, default="", dest="contrast_ref_a")
    parser.add_argument("--cf-grid-n", type=int, default=25, dest="cf_grid_n")
    parser.add_argument("--cf-a-pctl-low", type=float, default=5.0, dest="cf_a_pctl_low")
    parser.add_argument("--cf-a-pctl-high", type=float, default=95.0, dest="cf_a_pctl_high")

    # ── IV Representation Training ─────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--methods", nargs="+", default=["PIV", "TrueIV"],
        choices=ALL_METHODS, metavar="METHOD",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ── Shared NPIV Settings ────────────────────────────────────────
    parser.add_argument(
        "--downstream", nargs="+", default=["deepiv", "dfiv"],
        choices=["deepiv", "dfiv"],
    )
    parser.add_argument(
        "--oracle-c",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="oracle_c",
        help="Use oracle C as W for methods without a learned W branch (default: on).",
    )
    add_model_config_arguments(
        parser,
        include_iv_representation=True,
        include_npiv_estimators=True,
    )

    # ── Reporting ───────────────────────────────────────────────────
    parser.add_argument("--probe-epochs", type=int, default=200, dest="probe_epochs")
    parser.add_argument("--probe-lr", type=float, default=1e-3, dest="probe_lr")
    parser.add_argument(
        "--mse-scale",
        type=str,
        default="var",
        choices=["raw", "var", "mean_sq", "range_sq"],
        dest="mse_scale",
        help="Scale for contrast error reporting; default is variance-normalized nMSE.",
    )

    # ── Checkpointing ───────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "demands"),
        dest="checkpoint_dir",
    )
    parser.add_argument("--force-retrain", action="store_true", dest="force_retrain")

    return parser.parse_args()


def execute_single_run(
    args: argparse.Namespace,
    *,
    device: torch.device,
    ckpt_root,
) -> dict[tuple[str, str], float]:
    """Train/load one run and return contrast MSEs by `(method, downstream)`."""
    eval_setup = build_eval_setup(args, float(args.rho), include_surface=False)
    a_ref = float(eval_setup["a_ref"])
    a_grid = np.asarray(eval_setup["a_grid"], dtype=np.float64)

    _, loaders, eval_loaders = build_demands_loaders(
        args,
        seed=int(args.seed),
        rho=float(args.rho),
    )
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

    l_te = test_raw["L"]
    true_tau = true_contrast_matrix(a_grid, a_ref, l_te.numpy(), demands_h0)

    seed_everything(int(args.seed))
    rng = np.random.default_rng(int(args.seed) + 7)

    print(
        f"References: rho={args.rho:g}  a0={a_ref:.6f}  "
        f"grid=[{a_grid[0]:.3f}, {a_grid[-1]:.3f}] (n={len(a_grid)})",
    )
    print(
        f"\n=== Demands run: seed={args.seed}  rho={args.rho:g}  "
        f"a0={a_ref:.4f}  grid_n={len(a_grid)} ===",
    )

    mse_by_key: dict[tuple[str, str], float] = {}
    metric_label = contrast_metric_label(args.mse_scale)
    probe_by_method: dict[tuple[str, str], float] = {}
    for method in list(dict.fromkeys(args.methods)):
        model_args = configured_model_args(
            args,
            dataset="demands",
            iv_method=method,
        )
        fp_extra = demands_fingerprint_extra(model_args)
        iv_ckpt_path = build_iv_checkpoint_path(
            ckpt_root,
            seed=int(args.seed),
            rho=float(args.rho),
            method=method,
        )
        w_tr, z_tr, w_va, z_va, w_te, z_te = load_or_train_iv(
            method,
            iv_ckpt_path,
            loaders,
            eval_loaders,
            model_args,
            device,
            int(args.seed),
            rng,
            allow_train=True,
            fingerprint_extra=fp_extra,
        )

        if model_args.oracle_c and method not in {"PIV", "Proxy-Orth"}:
            assert (
                oracle_c_tr is not None
                and oracle_c_va is not None
                and oracle_c_te is not None
            )
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
        for downstream in list(dict.fromkeys(args.downstream)):
            print(f"    [{downstream}] NPIV …")
            npiv_ckpt_path = build_npiv_checkpoint_path(
                ckpt_root,
                seed=int(args.seed),
                rho=float(args.rho),
                method=method,
                downstream=downstream,
                oracle_c=bool(args.oracle_c),
            )
            model, norms = fit_npiv(
                downstream,
                tr_np,
                w_tr,
                z_tr,
                model_args,
                device,
                DEMANDS_DIMS,
                val_raw=val_np,
                w_val=w_va,
                z_val=z_va,
                checkpoint_path=npiv_ckpt_path,
                iv_method=method,
                seed=int(args.seed),
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
            print(f"       contrast {metric_label} = {mse:.6f}")

    print(f"\n{'=' * 72}")
    print(
        f"[rho={args.rho:g}, seed={args.seed}] "
        f"Unit-level treatment contrast {metric_label}",
    )
    print(f"Reference a0 = {a_ref:.6f}")
    print(f"{'=' * 72}")
    for method in list(dict.fromkeys(args.methods)):
        print(
            f"  {method:10}  "
            f"W->C={probe_by_method[(method, 'W->C')]:.4f}  "
            f"W->V={probe_by_method[(method, 'W->V')]:.4f}  "
            f"W->D={probe_by_method[(method, 'W->D')]:.4f}  "
            f"Z->V={probe_by_method[(method, 'Z->V')]:.4f}  "
            f"Z->D={probe_by_method[(method, 'Z->D')]:.4f}  "
            f"(W,Z)->V={probe_by_method[(method, '(W,Z)->V')]:.4f}",
        )
        for downstream in list(dict.fromkeys(args.downstream)):
            mse = mse_by_key[(method, downstream)]
            print(f"  {method:10}  {downstream:8}  {mse:.6f}")
    return mse_by_key


def main() -> None:
    args = parse_args()
    args.treatment_type = "continuous"
    device = torch.device(args.device)
    ckpt_root = (
        None
        if not str(args.checkpoint_dir).strip()
        else Path(args.checkpoint_dir).expanduser().resolve()
    )
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints: {ckpt_root}")
    execute_single_run(args, device=device, ckpt_root=ckpt_root)


if __name__ == "__main__":
    main()
