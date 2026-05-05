#!/usr/bin/env python3
"""Train IV-representation models + NPIV estimators on demands benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_configs import add_model_config_arguments, configured_model_args
from data.sim_demands import (
    DemandsConfig,
    simulate_demands,
    structural_response_demands,
)
from evaluation import (
    build_counterfactual_eval,
    contrast_error,
    contrast_metric_label,
    hat_contrast_matrix,
    probe_w_to_target_r2,
    probe_wz_to_target_r2,
    probe_z_to_target_r2,
    true_contrast_matrix,
)
from synthetic_dataset import make_synthetic_proxyiv_dataloaders
from trainer import fit_npiv, load_iv_representation, train_iv_representation
from utils import IVDims, collect_loader, make_eval_loaders, method_tag, seed_everything

DEMANDS_DIMS = IVDims(x_dim=3, m_dim=1, w_dim=1, z_dim=1, a_dim=1, l_dim=2, y_dim=1)
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
    if include_surface:
        raise ValueError("surface evaluation is only available in the removed test script")

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
    return {
        "a_ref": float(a_ref),
        "a_grid": a_grid,
        "tau_pop_true": tau_pop_true,
    }


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
    oracle_suffix = "_orc" if oracle_c and method not in {"PIV"} else ""
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
            print(f"\n  [{method}] IV checkpoint unusable ({exc!r}) - retraining.")

    if not allow_train:
        raise SystemExit(f"missing IV checkpoint for {method}: {iv_ckpt_path}")

    print(f"\n  [{method}] training representation ...")
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
        help="Replace learned W with oracle C except for PIV (default: on).",
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

        if model_args.oracle_c and method not in {"PIV"}:
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
