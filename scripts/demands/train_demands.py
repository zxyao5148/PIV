#!/usr/bin/env python3
"""Single-seed, single-rho demands benchmark.

This script trains or loads checkpoints for one `(seed, rho)` run and reports
only the unit-level treatment-contrast MSE for each `(method, downstream)`
combination. Richer evaluations such as population curves, contrast surfaces,
and probe metrics are intentionally left to `test_demands.py`, which rebuilds
them later from the saved checkpoints.
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

from evaluation import (
    contrast_mse,
    hat_contrast_matrix,
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
    parser.add_argument("--n", type=int, default=8_000)
    parser.add_argument("--pop-n", type=int, default=200_000, dest="pop_n")
    parser.add_argument("--pop-seed", type=int, default=20250101, dest="pop_seed")
    parser.add_argument("--coef-eps-a", type=float, default=1.0, dest="coef_eps_a")
    parser.add_argument("--coef-u-outcome", type=float, default=1.0, dest="coef_u_outcome")
    parser.add_argument("--contrast-ref-a", type=str, default="", dest="contrast_ref_a")
    parser.add_argument("--cf-grid-n", type=int, default=25, dest="cf_grid_n")
    parser.add_argument("--cf-a-pctl-low", type=float, default=5.0, dest="cf_a_pctl_low")
    parser.add_argument("--cf-a-pctl-high", type=float, default=95.0, dest="cf_a_pctl_high")

    # ── IV Representation Training ─────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--s1_epochs", type=int, default=200)
    parser.add_argument("--s2_epochs", type=int, default=200)
    parser.add_argument("--baseline_epochs", type=int, default=200)
    parser.add_argument("--autoiv_epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--proxy-embed-dim", type=int, default=4, dest="proxy_embed_dim")
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
    parser.add_argument("--npiv-restarts", type=int, default=1, dest="npiv_restarts")
    parser.add_argument(
        "--npiv-normalize-wl",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="npiv_normalize_wl",
    )
    parser.add_argument("--oracle-c", action="store_true", dest="oracle_c")

    # ── DeepIV ──────────────────────────────────────────────────────
    parser.add_argument("--deepiv-epochs", type=int, default=4000, dest="deepiv_epochs")
    parser.add_argument("--deepiv-num-components", type=int, default=5, dest="deepiv_num_components")
    parser.add_argument("--deepiv-treatment-hidden", type=int, default=256, dest="deepiv_treatment_hidden")
    parser.add_argument("--deepiv-response-hidden", type=int, default=256, dest="deepiv_response_hidden")
    parser.add_argument("--deepiv-num-layers", type=int, default=2, dest="deepiv_num_layers")
    parser.add_argument("--deepiv-activation", type=str, default="relu", dest="deepiv_activation")
    parser.add_argument("--deepiv-dropout", type=float, default=0.0, dest="deepiv_dropout")
    parser.add_argument("--deepiv-num-samples", type=int, default=8, dest="deepiv_num_samples")
    parser.add_argument("--deepiv-lr", type=float, default=1e-3, dest="deepiv_lr")
    parser.add_argument("--deepiv-weight-decay", type=float, default=1e-4, dest="deepiv_weight_decay")
    parser.add_argument("--deepiv-min-scale", type=float, default=1e-3, dest="deepiv_min_scale")
    parser.add_argument("--deepiv-val-frac", type=float, default=0.1, dest="deepiv_val_frac")
    parser.add_argument("--deepiv-patience", type=int, default=200, dest="deepiv_patience")

    # ── DFIV ────────────────────────────────────────────────────────
    parser.add_argument("--dfiv-epochs", type=int, default=2000, dest="dfiv_epochs")
    parser.add_argument("--dfiv-feature-dim", type=int, default=32, dest="dfiv_feature_dim")
    parser.add_argument("--dfiv-psi-hidden", type=int, default=256, dest="dfiv_psi_hidden")
    parser.add_argument("--dfiv-phi-hidden", type=int, default=256, dest="dfiv_phi_hidden")
    parser.add_argument("--dfiv-xi-hidden", type=int, default=256, dest="dfiv_xi_hidden")
    parser.add_argument("--dfiv-num-layers", type=int, default=2, dest="dfiv_num_layers")
    parser.add_argument("--dfiv-activation", type=str, default="relu", dest="dfiv_activation")
    parser.add_argument("--dfiv-dropout", type=float, default=0.0, dest="dfiv_dropout")
    parser.add_argument("--dfiv-lambda1", type=float, default=0.1, dest="dfiv_lambda1")
    parser.add_argument("--dfiv-lambda2", type=float, default=0.1, dest="dfiv_lambda2")
    parser.add_argument("--dfiv-lr", type=float, default=5e-4, dest="dfiv_lr")
    parser.add_argument("--dfiv-lr-phi", type=float, default=5e-5, dest="dfiv_lr_phi")
    parser.add_argument("--dfiv-batch-size", type=int, default=256, dest="dfiv_batch_size")
    parser.add_argument("--dfiv-weight-decay", type=float, default=1e-3, dest="dfiv_weight_decay")
    parser.add_argument("--dfiv-val-frac", type=float, default=0.1, dest="dfiv_val_frac")
    parser.add_argument("--dfiv-patience", type=int, default=200, dest="dfiv_patience")

    # ── Reporting ───────────────────────────────────────────────────
    parser.add_argument("--probe-epochs", type=int, default=200, dest="probe_epochs")
    parser.add_argument("--probe-lr", type=float, default=1e-3, dest="probe_lr")

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
    probe_train = collect_loader(eval_loaders.train, keys=["V", "D"])
    probe_test = collect_loader(eval_loaders.test, keys=["V", "D"])

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
    probe_by_method: dict[tuple[str, str], float] = {}
    for method in list(dict.fromkeys(args.methods)):
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
            args,
            device,
            int(args.seed),
            rng,
            allow_train=True,
        )

        if args.oracle_c and method != "PIV":
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
        for target in ("V", "D"):
            tgt_tr = probe_train[target].to(torch.float32)
            tgt_te = probe_test[target].to(torch.float32)
            if target == "V" and method != "PIV":
                r2 = probe_z_to_target_r2(
                    z_tr_t,
                    tgt_tr,
                    z_te_t,
                    tgt_te,
                    epochs=int(args.probe_epochs),
                    lr=float(args.probe_lr),
                    device=device,
                )
            else:
                r2 = probe_wz_to_target_r2(
                    w_tr_t,
                    z_tr_t,
                    tgt_tr,
                    w_te_t,
                    z_te_t,
                    tgt_te,
                    epochs=int(args.probe_epochs),
                    lr=float(args.probe_lr),
                    device=device,
                )
            probe_by_method[(method, target)] = float(r2)
            print(f"    [probe] ->{target}: R2 = {r2:.4f}")

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
                args,
                device,
                DEMANDS_DIMS,
                val_raw=val_np,
                w_val=w_va,
                z_val=z_va,
                checkpoint_path=npiv_ckpt_path,
                iv_method=method,
                seed=int(args.seed),
                force_retrain=bool(args.force_retrain),
                plots_only=False,
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
            mse = contrast_mse(tau_hat, true_tau)
            mse_by_key[(method, downstream)] = float(mse)
            print(f"       contrast MSE = {mse:.6f}")

    print(f"\n{'=' * 72}")
    print(
        f"[rho={args.rho:g}, seed={args.seed}] "
        "Unit-level treatment contrast MSE  E[(tau_hat-tau)^2]",
    )
    print(f"Reference a0 = {a_ref:.6f}")
    print(f"{'=' * 72}")
    for method in list(dict.fromkeys(args.methods)):
        print(
            f"  {method:10}  probe->V={probe_by_method[(method, 'V')]:.4f}  "
            f"probe->D={probe_by_method[(method, 'D')]:.4f}",
        )
        for downstream in list(dict.fromkeys(args.downstream)):
            mse = mse_by_key[(method, downstream)]
            print(f"  {method:10}  {downstream:8}  {mse:.6f}")
    return mse_by_key


def main() -> None:
    args = parse_args()
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
