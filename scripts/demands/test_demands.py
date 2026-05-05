#!/usr/bin/env python3
"""Aggregate demands results by loading saved checkpoints."""
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
    hat_contrast_surface,
    probe_wz_to_target_r2,
    probe_z_to_target_r2,
    true_contrast_matrix,
)
from trainer import fit_npiv
from utils import collect_loader, seed_everything
from vis.vis_demands import (
    save_contrast_mse_heatmap,
    save_population_contrast_figures,
    save_representation_probe_figure,
    save_treatment_effect_heatmap,
    save_unit_level_figures,
)

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
    discover_available_seeds,
    load_or_train_iv,
    rho_tag,
    select_unit_plot_index,
)


def parse_args() -> argparse.Namespace:
    """CLI for checkpoint-backed aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate demands checkpoints across seeds and rhos.",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default="",
        help="Output directory for aggregated figures.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Single rho used for the per-rho figure bundle.",
    )
    parser.add_argument(
        "--rho-list",
        "--rho_list",
        nargs="*",
        type=float,
        default=[],
        dest="rho_list",
        help="Rhos used as rows of the cross-rho MSE heatmap.",
    )
    parser.add_argument(
        "--rho-seeds",
        action="append",
        default=[],
        metavar="RHO:SEED1,SEED2",
        help=(
            "Optional per-rho seed list. Example: "
            "`--rho-seeds 0.1:1,2,3 --rho-seeds 0.5:4,5`."
        ),
    )

    # ── DGP And Evaluation Geometry ────────────────────────────────
    parser.add_argument("--n", type=int, default=8_000)
    parser.add_argument("--pop-n", type=int, default=200_000, dest="pop_n")
    parser.add_argument("--pop-seed", type=int, default=20250101, dest="pop_seed")
    parser.add_argument("--coef-eps-a", type=float, default=1.0, dest="coef_eps_a")
    parser.add_argument("--coef-u-outcome", type=float, default=1.0, dest="coef_u_outcome")
    parser.add_argument("--contrast-ref-a", type=str, default="", dest="contrast_ref_a")
    parser.add_argument("--cf-grid-n", type=int, default=25, dest="cf_grid_n")
    parser.add_argument("--cf-a-pctl-low", type=float, default=5.0, dest="cf_a_pctl_low")
    parser.add_argument("--cf-a-pctl-high", type=float, default=95.0, dest="cf_a_pctl_high")

    # ── IV Representation Loading ──────────────────────────────────
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

    # ── Shared NPIV Loading ────────────────────────────────────────
    parser.add_argument(
        "--downstream", nargs="+", default=["deepiv", "dfiv"],
        choices=["deepiv", "dfiv"],
    )
    parser.add_argument("--oracle-c", action="store_true", dest="oracle_c")
    add_model_config_arguments(
        parser,
        include_iv_representation=True,
        include_npiv_estimators=True,
    )

    # ── Aggregation And Figures ────────────────────────────────────
    parser.add_argument("--unit-plot-grid-index", type=int, default=-1, dest="unit_plot_grid_index")
    parser.add_argument("--unit-plot-bins", type=int, default=40, dest="unit_plot_bins")
    parser.add_argument("--surface-na", type=int, default=50, dest="surface_na")
    parser.add_argument("--surface-nl1", type=int, default=50, dest="surface_nl1")
    parser.add_argument("--surface-l1-lo", type=float, default=0.0, dest="surface_l1_lo")
    parser.add_argument("--surface-l1-hi", type=float, default=10.0, dest="surface_l1_hi")
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
    parser.add_argument(
        "--probe-targets", nargs="+", default=["V", "D"], dest="probe_targets",
        choices=["V", "D", "C", "U"],
    )

    # ── Checkpoint Loading ─────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "demands"),
        dest="checkpoint_dir",
    )
    parser.add_argument("--force-retrain", action="store_true", dest="force_retrain")

    return parser.parse_args()


def _parse_rho_seed_specs(specs: list[str]) -> dict[float, list[int]]:
    """Parse `rho:seed1,seed2` specs into a dictionary."""
    out: dict[float, list[int]] = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(
                f"bad --rho-seeds spec {spec!r}; expected format RHO:SEED1,SEED2",
            )
        rho_str, seeds_str = spec.split(":", 1)
        seed_tokens = [token.strip() for token in seeds_str.split(",") if token.strip()]
        if not seed_tokens:
            raise SystemExit(f"bad --rho-seeds spec {spec!r}; no seeds listed")
        out[float(rho_str)] = [int(token) for token in seed_tokens]
    return out


def determine_requested_rhos(args: argparse.Namespace) -> tuple[float | None, list[float]]:
    """Resolve the single-rho and cross-rho selections from the CLI."""
    single_rho = float(args.rho) if args.rho is not None else None
    heatmap_rhos = list(dict.fromkeys(float(rho) for rho in args.rho_list))
    if not heatmap_rhos:
        if single_rho is not None:
            heatmap_rhos = [single_rho]
        else:
            rho_seed_map = _parse_rho_seed_specs(list(args.rho_seeds))
            heatmap_rhos = list(rho_seed_map.keys())
    if single_rho is None and len(heatmap_rhos) == 1:
        single_rho = heatmap_rhos[0]
    if single_rho is None and not heatmap_rhos:
        raise SystemExit(
            "specify --rho, --rho-list, or --rho-seeds so the aggregator knows which rhos to load",
        )
    return single_rho, heatmap_rhos


def determine_seed_list_for_rho(
    ckpt_root: Path,
    *,
    rho: float,
    rho_seed_map: dict[float, list[int]],
) -> list[int]:
    """Choose the seeds used for one rho."""
    if rho in rho_seed_map:
        return list(dict.fromkeys(rho_seed_map[rho]))
    seeds = discover_available_seeds(ckpt_root, rho=rho)
    if not seeds:
        raise SystemExit(
            f"no IV checkpoints found for rho={rho:g} in {ckpt_root}; "
            "specify --rho-seeds or run the corresponding training jobs first",
        )
    return seeds


def create_args_for_rho(args: argparse.Namespace, rho: float) -> argparse.Namespace:
    """Clone `args` and pin the load-time rho to the checkpoint rho."""
    out = argparse.Namespace(**vars(args))
    out.rho = float(rho)
    return out


def evaluate_seed_checkpoint_bundle(
    args: argparse.Namespace,
    *,
    rho: float,
    seed: int,
    device: torch.device,
    ckpt_root: Path,
    eval_setup: dict[str, object],
    methods: list[str],
    downstreams: list[str],
) -> dict[str, object]:
    """Load checkpoints for one `(seed, rho)` pair and compute evaluation objects."""
    base_args = create_args_for_rho(args, rho)
    base_args.seed = int(seed)
    _, loaders, eval_loaders = build_demands_loaders(base_args, seed=seed, rho=rho)
    train_raw = collect_loader(eval_loaders.train, keys=["A", "Y", "L"])
    val_raw = collect_loader(eval_loaders.val, keys=["A", "Y", "L"])
    test_raw = collect_loader(eval_loaders.test, keys=["A", "Y", "L"])
    probe_train = collect_loader(eval_loaders.train, keys=list(args.probe_targets))
    probe_test = collect_loader(eval_loaders.test, keys=list(args.probe_targets))

    if args.oracle_c:
        oracle_c_tr = collect_loader(eval_loaders.train, keys=["C"])["C"].float()
        oracle_c_va = collect_loader(eval_loaders.val, keys=["C"])["C"].float()
        oracle_c_te = collect_loader(eval_loaders.test, keys=["C"])["C"].float()
    else:
        oracle_c_tr = oracle_c_va = oracle_c_te = None

    a_ref = float(eval_setup["a_ref"])
    a_grid = np.asarray(eval_setup["a_grid"], dtype=np.float64)
    l_te = test_raw["L"]
    true_tau = true_contrast_matrix(a_grid, a_ref, l_te.numpy(), demands_h0)
    unit_plot_index = select_unit_plot_index(a_grid, int(args.unit_plot_grid_index))
    unit_true = np.asarray(true_tau[:, unit_plot_index], dtype=np.float64)

    seed_everything(int(seed))
    rng = np.random.default_rng(int(seed) + 7)

    probe_r2: dict[tuple[str, str], float] = {}
    mse_map: dict[tuple[str, str], float] = {}
    pop_hat_store: dict[tuple[int, str, str], list[np.ndarray]] = {}
    unit_hat: dict[tuple[int, str, str], np.ndarray] = {}
    contrast_surface_store: dict[tuple[str, str], list[np.ndarray]] = {}

    for method in methods:
        load_args = configured_model_args(
            base_args,
            dataset="demands",
            iv_method=method,
        )
        fp_extra = demands_fingerprint_extra(load_args)
        iv_ckpt_path = build_iv_checkpoint_path(
            ckpt_root,
            seed=seed,
            rho=rho,
            method=method,
        )
        w_tr, z_tr, w_va, z_va, w_te, z_te = load_or_train_iv(
            method,
            iv_ckpt_path,
            loaders,
            eval_loaders,
            load_args,
            device,
            seed,
            rng,
            allow_train=False,
            fingerprint_extra=fp_extra,
        )
        if args.oracle_c and method not in {"PIV"}:
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
        for target in args.probe_targets:
            tgt_tr = probe_train[target].to(torch.float32)
            tgt_te = probe_test[target].to(torch.float32)
            if target == "D" or (target == "V" and method not in {"PIV", "Proxy-Orth"}):
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
            probe_r2[(method, target)] = float(r2)

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
        for downstream in downstreams:
            npiv_ckpt_path = build_npiv_checkpoint_path(
                ckpt_root,
                seed=seed,
                rho=rho,
                method=method,
                downstream=downstream,
                oracle_c=bool(args.oracle_c),
            )
            model, norms = fit_npiv(
                downstream,
                tr_np,
                w_tr,
                z_tr,
                load_args,
                device,
                DEMANDS_DIMS,
                val_raw=val_np,
                w_val=w_va,
                z_val=z_va,
                checkpoint_path=npiv_ckpt_path,
                iv_method=method,
                seed=seed,
                force_retrain=False,
                test=True,
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
            mse_map[(method, downstream)] = contrast_error(
                tau_hat,
                true_tau,
                scale=args.mse_scale,
            )
            pop_hat_store[(seed, method, downstream)] = [tau_hat.mean(axis=0)]
            unit_hat[(seed, method, downstream)] = np.asarray(
                tau_hat[:, unit_plot_index],
                dtype=np.float64,
            )
            contrast_surface = hat_contrast_surface(
                model,
                np.asarray(eval_setup["contrast_surface_a_grid"], dtype=np.float64),
                a_ref,
                np.asarray(eval_setup["contrast_surface_l1_grid"], dtype=np.float64),
                vary_idx=0,
                l_units=l_te,
                w=w_te,
                device=device,
                norms=norms,
            )
            contrast_surface_store.setdefault((method, downstream), []).append(
                contrast_surface,
            )

    return {
        "seed": seed,
        "unit_true": unit_true,
        "probe_r2": probe_r2,
        "mse_map": mse_map,
        "pop_hat_store": pop_hat_store,
        "unit_hat": unit_hat,
        "contrast_surface_store": contrast_surface_store,
    }


def aggregate_seed_evaluations_for_rho(
    args: argparse.Namespace,
    *,
    rho: float,
    seeds: list[int],
    device: torch.device,
    ckpt_root: Path,
    methods: list[str],
    downstreams: list[str],
) -> dict[str, object]:
    """Load all selected checkpoints for one rho and aggregate them."""
    eval_setup = build_eval_setup(args, rho, include_surface=True)
    aggregated = {
        "seeds": seeds,
        "a_ref": float(eval_setup["a_ref"]),
        "a_grid": np.asarray(eval_setup["a_grid"], dtype=np.float64),
        "tau_pop_true": np.asarray(eval_setup["tau_pop_true"], dtype=np.float64),
        "unit_plot_index": select_unit_plot_index(
            np.asarray(eval_setup["a_grid"], dtype=np.float64),
            int(args.unit_plot_grid_index),
        ),
        "unit_true_by_seed": {},
        "unit_hat": {},
        "probe_r2_store": {},
        "contrast_surface_a_grid": np.asarray(
            eval_setup["contrast_surface_a_grid"],
            dtype=np.float64,
        ),
        "contrast_surface_l1_grid": np.asarray(
            eval_setup["contrast_surface_l1_grid"],
            dtype=np.float64,
        ),
        "contrast_surface_true": np.asarray(
            eval_setup["contrast_surface_true"],
            dtype=np.float64,
        ),
        "contrast_surface_store": {},
        "pop_hat_store": {},
        "mse_store": {},
    }

    for seed in seeds:
        per_seed = evaluate_seed_checkpoint_bundle(
            args,
            rho=rho,
            seed=seed,
            device=device,
            ckpt_root=ckpt_root,
            eval_setup=eval_setup,
            methods=methods,
            downstreams=downstreams,
        )
        aggregated["unit_true_by_seed"][seed] = per_seed["unit_true"]
        aggregated["unit_hat"].update(per_seed["unit_hat"])
        aggregated["pop_hat_store"].update(per_seed["pop_hat_store"])
        for key, value in per_seed["probe_r2"].items():
            aggregated["probe_r2_store"].setdefault(key, []).append(value)
        for key, value in per_seed["mse_map"].items():
            aggregated["mse_store"].setdefault(key, []).append(value)
        for key, surfaces in per_seed["contrast_surface_store"].items():
            aggregated["contrast_surface_store"].setdefault(key, []).extend(surfaces)
    return aggregated


def report_rho_metrics(
    rho: float,
    aggregated: dict[str, object],
    *,
    methods: list[str],
    downstreams: list[str],
    probe_targets: list[str],
    mse_scale: str,
) -> None:
    """Print per-rho MSE and probe summaries."""
    metric_label = contrast_metric_label(mse_scale)
    seeds = aggregated["seeds"]
    print(f"\n{'=' * 72}")
    print(
        f"[rho={rho:g}] Unit-level treatment contrast {metric_label}  "
        f"(mean +- sd over seeds={seeds})",
    )
    print(f"Reference a0 = {aggregated['a_ref']:.6f}")
    print(f"{'=' * 72}")
    for method in methods:
        for downstream in downstreams:
            vals = aggregated["mse_store"].get((method, downstream), [])
            if not vals:
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {method:10}  {downstream:8}  {mean:.6f}  +- {sd:.6f}  (n={len(vals)})")

    print(f"\n{'=' * 72}")
    print(f"[rho={rho:g}] Representation-probe test R2  (mean +- sd over seeds)")
    print(f"{'=' * 72}")
    for method in methods:
        for target in probe_targets:
            vals = aggregated["probe_r2_store"].get((method, target), [])
            if not vals:
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {method:10}  ->{target:<2}  {mean:.4f}  +- {sd:.4f}  (n={len(vals)})")


def render_rho_figure_set(
    out_dir: Path,
    aggregated: dict[str, object],
    *,
    methods: list[str],
    downstreams: list[str],
    probe_targets: list[str],
    unit_plot_bins: int,
) -> None:
    """Save all per-rho figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_population_contrast_figures(
        out_dir,
        aggregated["a_grid"],
        aggregated["tau_pop_true"],
        aggregated["seeds"],
        methods,
        downstreams,
        aggregated["pop_hat_store"],
    )
    save_representation_probe_figure(
        out_dir,
        methods,
        probe_targets,
        aggregated["probe_r2_store"],
        aggregated["seeds"],
    )
    save_unit_level_figures(
        out_dir,
        aggregated["seeds"],
        methods,
        downstreams,
        aggregated["unit_true_by_seed"],
        aggregated["unit_hat"],
        aggregated["a_grid"],
        int(aggregated["unit_plot_index"]),
        unit_plot_bins=unit_plot_bins,
    )
    for downstream in downstreams:
        tau_hat_by_method: dict[str, np.ndarray] = {}
        n_seeds_eff = 0
        for method in methods:
            surfaces = aggregated["contrast_surface_store"].get((method, downstream), [])
            if not surfaces:
                continue
            tau_hat_by_method[method] = np.mean(np.stack(surfaces, axis=0), axis=0)
            n_seeds_eff = max(n_seeds_eff, len(surfaces))
        if not tau_hat_by_method:
            continue
        save_treatment_effect_heatmap(
            out_dir / f"demands_heatmap_{downstream}.png",
            a_grid=aggregated["contrast_surface_a_grid"],
            l1_grid=aggregated["contrast_surface_l1_grid"],
            a_ref=float(aggregated["a_ref"]),
            tau_true=aggregated["contrast_surface_true"],
            tau_hat_by_method=tau_hat_by_method,
            downstream=downstream,
            n_seeds=n_seeds_eff,
        )


def main() -> None:
    args = parse_args()
    args.treatment_type = "continuous"
    ckpt_root = Path(args.checkpoint_dir).expanduser().resolve()
    if not ckpt_root.exists():
        raise SystemExit(
            f"checkpoint directory does not exist: {ckpt_root}. "
            "Run scripts/demands/train_demands.py first.",
        )
    out_root = (
        Path(args.out_fig).expanduser().resolve()
        if str(args.out_fig).strip()
        else ckpt_root.parent / "figs" / "demands"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    single_rho, heatmap_rhos = determine_requested_rhos(args)
    rho_seed_map = _parse_rho_seed_specs(list(args.rho_seeds))
    methods = list(dict.fromkeys(args.methods))
    downstreams = list(dict.fromkeys(args.downstream))
    probe_targets = list(args.probe_targets)

    print(f"Checkpoints: {ckpt_root}")
    print(f"Methods: {methods}")
    print(f"Downstream: {downstreams}")
    print(f"Probe targets: {probe_targets}")
    print(f"single-rho figures: {single_rho if single_rho is not None else 'all selected rhos'}")
    print(f"cross-rho heatmap rows: {heatmap_rhos}")

    mse_by_rho: dict[tuple[float, str, str], list[float]] = {}
    max_seed_count = 0
    for rho in heatmap_rhos:
        seeds = determine_seed_list_for_rho(ckpt_root, rho=rho, rho_seed_map=rho_seed_map)
        aggregated = aggregate_seed_evaluations_for_rho(
            args,
            rho=rho,
            seeds=seeds,
            device=device,
            ckpt_root=ckpt_root,
            methods=methods,
            downstreams=downstreams,
        )
        report_rho_metrics(
            rho,
            aggregated,
            methods=methods,
            downstreams=downstreams,
            probe_targets=probe_targets,
            mse_scale=str(args.mse_scale),
        )
        max_seed_count = max(max_seed_count, len(seeds))
        for (method, downstream), vals in aggregated["mse_store"].items():
            mse_by_rho[(rho, method, downstream)] = list(vals)
        if single_rho is None or np.isclose(rho, single_rho, atol=1e-12, rtol=0.0):
            render_rho_figure_set(
                out_root / f"rho_{rho_tag(rho)}",
                aggregated,
                methods=methods,
                downstreams=downstreams,
                probe_targets=probe_targets,
                unit_plot_bins=int(args.unit_plot_bins),
            )

    print(f"\n{'=' * 72}")
    metric_label = contrast_metric_label(args.mse_scale)
    print(f"Cross-rho unit-level contrast {metric_label}  (mean +- sd over selected seeds)")
    print(f"{'=' * 72}")
    for rho in heatmap_rhos:
        for method in methods:
            for downstream in downstreams:
                vals = mse_by_rho.get((rho, method, downstream), [])
                if not vals:
                    continue
                mean = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                print(
                    f"  rho={rho:<5g}  {method:10}  {downstream:8}  "
                    f"{mean:.6f}  +- {sd:.6f}  (n={len(vals)})",
                )

    save_contrast_mse_heatmap(
        out_root,
        rhos=heatmap_rhos,
        methods=methods,
        downstreams=downstreams,
        mse_store=mse_by_rho,
        n_seeds=max_seed_count,
    )


if __name__ == "__main__":
    main()
