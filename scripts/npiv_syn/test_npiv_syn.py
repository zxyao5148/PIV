#!/usr/bin/env python3
"""Aggregate synthetic NPIV results by loading saved checkpoints across train_seeds."""
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
    probe_w_to_target_r2,
    probe_wz_to_target_r2,
    probe_z_to_target_r2,
    true_contrast_matrix,
    true_contrast_surface,
)
from trainer import fit_npiv
from utils import collect_loader, seed_everything

from scripts.npiv_syn.common import (
    ALL_METHODS,
    BINARY_SURFACES,
    CONTINUOUS_SURFACES,
    attach_npiv_fingerprint_compat,
    build_iv_checkpoint_path,
    build_npiv_checkpoint_path,
    build_npiv_syn_bundle,
    build_npiv_syn_eval_setup,
    discover_available_train_seeds,
    load_or_train_iv,
    npiv_syn_fingerprint_extra,
    select_unit_plot_index,
)
from vis.vis_npiv_syn import (
    save_binary_ate_bars,
    save_binary_cate_curves,
    save_mse_heatmap,
    save_population_contrast_figures,
    save_probe_figure,
    save_surface_heatmap,
    save_unit_level_figures,
)

_SURFACE_TO_TREATMENT = {
    **{s: "continuous" for s in CONTINUOUS_SURFACES},
    **{s: "binary" for s in BINARY_SURFACES},
}

_ALL_PROBE_LABELS = ["W->C", "W->V", "W->D", "Z->V", "Z->D", "(W,Z)->V"]
_DEFAULT_PROBE_LABELS = ["(W,Z)->V", "Z->D"]
_PROBE_TARGET = {
    "W->C": "C", "W->V": "V", "W->D": "D",
    "Z->V": "V", "Z->D": "D", "(W,Z)->V": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate synthetic NPIV checkpoints across train_seeds.",
    )

    # ── Scenario selection ─────────────────────────────────────────
    parser.add_argument(
        "--treatment-types", nargs="+",
        default=["continuous", "binary"],
        choices=["continuous", "binary"],
        dest="treatment_types",
    )
    parser.add_argument(
        "--surfaces", nargs="+", default=None,
        choices=list(_SURFACE_TO_TREATMENT.keys()),
        help="Surfaces to evaluate. Default: all surfaces for the selected treatment types.",
    )
    parser.add_argument(
        "--train-seeds", nargs="*", type=int, default=[],
        dest="train_seeds",
        help="Train seeds to aggregate. Empty = auto-discover from checkpoint dir.",
    )
    parser.add_argument("--design-seed", type=int, default=1, dest="design_seed")
    parser.add_argument("--data-seed", type=int, default=1, dest="data_seed")

    # ── Dataset / DGP ──────────────────────────────────────────────
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--dC", type=int, default=2)
    parser.add_argument("--dV", type=int, default=2)
    parser.add_argument("--dD", type=int, default=2)
    parser.add_argument("--dL", type=int, default=None)
    parser.add_argument("--dM", type=int, default=None)
    parser.add_argument("--dA", type=int, default=1)
    parser.add_argument("--sigma-m", type=float, default=0.50, dest="sigma_m")
    parser.add_argument("--sigma-a", type=float, default=1.00, dest="sigma_a")
    parser.add_argument("--sigma-y", type=float, default=1.00, dest="sigma_y")
    parser.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    parser.add_argument("--train-frac", type=float, default=0.70, dest="train_frac")
    parser.add_argument("--val-frac", type=float, default=0.15, dest="val_frac")

    # ── Population draw ─────────────────────────────────────────────
    parser.add_argument("--pop-n", type=int, default=50_000, dest="pop_n")
    parser.add_argument("--pop-seed", type=int, default=20250101, dest="pop_seed")

    # ── Models ─────────────────────────────────────────────────────
    parser.add_argument(
        "--methods", nargs="+", default=["PIV", "TrueIV"],
        choices=ALL_METHODS, metavar="METHOD",
    )
    parser.add_argument(
        "--downstream", nargs="+", default=["deepiv", "dfiv"],
        choices=["deepiv", "dfiv"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    add_model_config_arguments(
        parser, include_iv_representation=True, include_npiv_estimators=True,
    )

    # ── Evaluation ─────────────────────────────────────────────────
    parser.add_argument("--probe-epochs", type=int, default=240, dest="probe_epochs")
    parser.add_argument("--probe-lr", type=float, default=5e-4, dest="probe_lr")
    parser.add_argument(
        "--probes", nargs="+",
        default=list(_DEFAULT_PROBE_LABELS),
        choices=_ALL_PROBE_LABELS,
        dest="probes",
        metavar="PROBE",
        help="Representation probes to run. Default: (W,Z)->V Z->D",
    )
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
    parser.add_argument("--unit-plot-grid-index", type=int, default=-1, dest="unit_plot_grid_index")
    parser.add_argument("--unit-plot-bins", type=int, default=40, dest="unit_plot_bins")

    # ── Surface figure ──────────────────────────────────────────────
    parser.add_argument("--surface-na", type=int, default=40, dest="surface_na")
    parser.add_argument("--surface-nl", type=int, default=40, dest="surface_nl")
    parser.add_argument("--surface-l-lo", type=float, default=None, dest="surface_l_lo")
    parser.add_argument("--surface-l-hi", type=float, default=None, dest="surface_l_hi")

    # ── Checkpointing ───────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "npiv_syn"),
        dest="checkpoint_dir",
    )
    parser.add_argument("--out-fig", type=str, default="", dest="out_fig")
    parser.add_argument("--oracle-c", action="store_true", dest="oracle_c")
    parser.add_argument("--force-retrain", action="store_true", dest="force_retrain")

    return parser.parse_args()


def _resolve_scenarios(args: argparse.Namespace) -> list[tuple[str, str]]:
    requested_tt = list(dict.fromkeys(args.treatment_types))
    requested_surfaces = (
        list(dict.fromkeys(args.surfaces))
        if args.surfaces is not None
        else list(_SURFACE_TO_TREATMENT.keys())
    )
    scenarios: list[tuple[str, str]] = []
    for tt in requested_tt:
        for surf in requested_surfaces:
            if _SURFACE_TO_TREATMENT.get(surf) == tt:
                scenarios.append((tt, surf))
    return scenarios


def _resolve_train_seeds(
    args: argparse.Namespace,
    ckpt_root: Path,
    treatment_type: str,
    surface: str,
) -> list[int]:
    if args.train_seeds:
        return list(dict.fromkeys(args.train_seeds))
    seeds = discover_available_train_seeds(
        ckpt_root,
        surface=surface,
        treatment_type=treatment_type,
        design_seed=int(args.design_seed),
        data_seed=int(args.data_seed),
        dA=int(getattr(args, "dA", 1)),
    )
    if not seeds:
        raise SystemExit(
            f"No IV checkpoints found for surface={surface} treatment={treatment_type} "
            f"design={args.design_seed} data={args.data_seed} in {ckpt_root}.\n"
            "Run run_npiv_syn_batch.sh first.",
        )
    return seeds


def _make_bundle_args(
    args: argparse.Namespace,
    *,
    surface: str,
    treatment_type: str,
    train_seed: int,
) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.surface = surface
    out.treatment_type = treatment_type
    out.train_seed = int(train_seed)
    return out


def _build_l_grid(args: argparse.Namespace, pop_l: np.ndarray) -> np.ndarray:
    lo = float(args.surface_l_lo) if args.surface_l_lo is not None else float(pop_l[:, 0].min())
    hi = float(args.surface_l_hi) if args.surface_l_hi is not None else float(pop_l[:, 0].max())
    return np.linspace(lo, hi, int(args.surface_nl), dtype=np.float64)


def _run_probes(
    w_tr_t: torch.Tensor,
    z_tr_t: torch.Tensor,
    w_te_t: torch.Tensor,
    z_te_t: torch.Tensor,
    probe_train: dict,
    probe_test: dict,
    model_args: argparse.Namespace,
    device: torch.device,
    *,
    method: str,
    probes: list[str],
) -> dict[str, float]:
    r2s: dict[str, float] = {}
    kw = {"epochs": int(model_args.probe_epochs), "lr": float(model_args.probe_lr), "device": device}
    for label in probes:
        target_key = _PROBE_TARGET[label]
        tgt_tr = probe_train[target_key].to(torch.float32)
        tgt_te = probe_test[target_key].to(torch.float32)
        if label in ("W->C", "W->V", "W->D"):
            r2 = probe_w_to_target_r2(w_tr_t, tgt_tr, w_te_t, tgt_te, **kw)
        elif label in ("Z->V", "Z->D"):
            r2 = probe_z_to_target_r2(z_tr_t, tgt_tr, z_te_t, tgt_te, **kw)
        else:  # "(W,Z)->V": use W+Z for methods with a learned W branch.
            if method in {"PIV", "Proxy-Orth"}:
                r2 = probe_wz_to_target_r2(w_tr_t, z_tr_t, tgt_tr, w_te_t, z_te_t, tgt_te, **kw)
            else:
                r2 = probe_z_to_target_r2(z_tr_t, tgt_tr, z_te_t, tgt_te, **kw)
        r2s[label] = float(r2)
        print(f"    [probe] {method}  {label}: R2 = {r2:.4f}")
    return r2s


def evaluate_train_seed_bundle(
    args: argparse.Namespace,
    *,
    surface: str,
    treatment_type: str,
    train_seed: int,
    device: torch.device,
    ckpt_root: Path,
    eval_setup: dict[str, object],
    methods: list[str],
    downstreams: list[str],
    unit_plot_index: int,
    l_grid: np.ndarray,
) -> dict[str, object]:
    bundle_args = _make_bundle_args(
        args, surface=surface, treatment_type=treatment_type, train_seed=train_seed,
    )
    bundle = build_npiv_syn_bundle(bundle_args)
    loaders = bundle["loaders"]
    eval_loaders = bundle["eval_loaders"]
    dims = bundle["dims"]

    train_raw = collect_loader(eval_loaders.train, keys=["A", "Y", "L"])
    val_raw = collect_loader(eval_loaders.val, keys=["A", "Y", "L"])
    test_raw = collect_loader(eval_loaders.test, keys=["L"])
    needed_probe_keys = list(dict.fromkeys(_PROBE_TARGET[p] for p in args.probes))
    probe_train = collect_loader(eval_loaders.train, keys=needed_probe_keys)
    probe_test = collect_loader(eval_loaders.test, keys=needed_probe_keys)

    if args.oracle_c:
        oracle_c_tr = collect_loader(eval_loaders.train, keys=["C"])["C"].float()
        oracle_c_va = collect_loader(eval_loaders.val, keys=["C"])["C"].float()
        oracle_c_te = collect_loader(eval_loaders.test, keys=["C"])["C"].float()
    else:
        oracle_c_tr = oracle_c_va = oracle_c_te = None

    dA = int(getattr(args, "dA", 1))
    l_te = test_raw["L"].float()
    a_ref = eval_setup["a_ref"]  # float (scalar) or (dA,) array (vector)
    a_grid = np.asarray(eval_setup["a_grid"], dtype=np.float64)
    h0_fn = eval_setup["h0_fn"]

    true_tau = true_contrast_matrix(a_grid, a_ref, l_te.numpy(), h0_fn)
    unit_true = np.asarray(true_tau[:, unit_plot_index], dtype=np.float64)

    seed_everything(int(train_seed))
    rng = np.random.default_rng(int(train_seed) + 7)

    tr_np = {k: train_raw[k].numpy() for k in ("A", "Y", "L")}
    val_np = {k: val_raw[k].numpy() for k in ("A", "Y", "L")}

    probe_r2: dict[tuple[str, str], float] = {}
    mse_map: dict[tuple[str, str], float] = {}
    pop_hat_store: dict[tuple[int, str, str], list[np.ndarray]] = {}
    unit_hat: dict[tuple[int, str, str], np.ndarray] = {}
    contrast_surface_store: dict[tuple[str, str], list[np.ndarray]] = {}

    for method in methods:
        model_args = configured_model_args(bundle_args, dataset="npiv_syn", iv_method=method, dA=int(getattr(bundle_args, "dA", 1)))
        attach_npiv_fingerprint_compat(model_args)
        fp_extra = npiv_syn_fingerprint_extra(model_args)

        iv_ckpt_path = build_iv_checkpoint_path(
            ckpt_root,
            surface=surface, treatment_type=treatment_type,
            design_seed=int(args.design_seed), data_seed=int(args.data_seed),
            train_seed=int(train_seed), method=method, dA=dA,
        )
        w_tr, z_tr, w_va, z_va, w_te, z_te = load_or_train_iv(
            method, iv_ckpt_path, loaders, eval_loaders, model_args, dims,
            device, int(train_seed), rng,
            allow_train=False, fingerprint_extra=fp_extra,
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

        r2s = _run_probes(w_tr_t, z_tr_t, w_te_t, z_te_t,
                          probe_train, probe_test, model_args, device,
                          method=method, probes=args.probes)
        probe_r2.update({(method, label): r2 for label, r2 in r2s.items()})

        for downstream in downstreams:
            npiv_ckpt_path = build_npiv_checkpoint_path(
                ckpt_root,
                surface=surface, treatment_type=treatment_type,
                design_seed=int(args.design_seed), data_seed=int(args.data_seed),
                train_seed=int(train_seed), method=method, downstream=downstream,
                oracle_c=bool(args.oracle_c), dA=dA,
            )
            model, norms = fit_npiv(
                downstream, tr_np, w_tr, z_tr, model_args, device, dims,
                val_raw=val_np, w_val=w_va, z_val=z_va,
                checkpoint_path=npiv_ckpt_path,
                iv_method=method, seed=int(train_seed),
                force_retrain=False, test=True,
                fingerprint_extra=fp_extra,
            )
            tau_hat = hat_contrast_matrix(model, a_grid, a_ref, w_te, l_te, device, norms)
            mse = contrast_error(tau_hat, true_tau, scale=args.mse_scale)
            mse_map[(method, downstream)] = mse
            pop_hat_store[(train_seed, method, downstream)] = [tau_hat.mean(axis=0)]
            unit_hat[(train_seed, method, downstream)] = np.asarray(
                tau_hat[:, unit_plot_index], dtype=np.float64,
            )
            if dA == 1:
                surf_arr = hat_contrast_surface(
                    model, a_grid, a_ref, l_grid, 0, l_te, w_te, device, norms,
                )
                contrast_surface_store.setdefault((method, downstream), []).append(surf_arr)

    return {
        "train_seed": train_seed,
        "unit_true": unit_true,
        "probe_r2": probe_r2,
        "mse_map": mse_map,
        "pop_hat_store": pop_hat_store,
        "unit_hat": unit_hat,
        "contrast_surface_store": contrast_surface_store,
    }


def aggregate_evaluations_for_scenario(
    args: argparse.Namespace,
    *,
    surface: str,
    treatment_type: str,
    train_seeds: list[int],
    device: torch.device,
    ckpt_root: Path,
    methods: list[str],
    downstreams: list[str],
) -> dict[str, object]:
    dA = int(getattr(args, "dA", 1))
    eval_setup = build_npiv_syn_eval_setup(
        args, surface=surface, treatment_type=treatment_type,
    )
    a_grid = np.asarray(eval_setup["a_grid"], dtype=np.float64)
    l_grid = _build_l_grid(args, eval_setup["pop_l"])
    unit_plot_index = select_unit_plot_index(a_grid, int(args.unit_plot_grid_index))

    if dA == 1:
        true_surf = true_contrast_surface(
            a_grid, eval_setup["a_ref"], l_grid, 0,
            eval_setup["pop_l"], eval_setup["h0_fn"],
        )
    else:
        true_surf = None

    aggregated: dict[str, object] = {
        "train_seeds": train_seeds,
        "a_ref": eval_setup["a_ref"],
        "a_grid": a_grid,
        "tau_pop_true": np.asarray(eval_setup["tau_pop_true"], dtype=np.float64),
        "l_grid": l_grid,
        "true_surf": true_surf,
        "unit_plot_index": unit_plot_index,
        "unit_true_by_seed": {},
        "unit_hat": {},
        "probe_r2_store": {},
        "mse_store": {},
        "pop_hat_store": {},
        "contrast_surface_store": {},
        "dA": dA,
    }

    for train_seed in train_seeds:
        print(f"\n  --- train_seed={train_seed} ---")
        per_seed = evaluate_train_seed_bundle(
            args,
            surface=surface, treatment_type=treatment_type, train_seed=train_seed,
            device=device, ckpt_root=ckpt_root, eval_setup=eval_setup,
            methods=methods, downstreams=downstreams,
            unit_plot_index=unit_plot_index, l_grid=l_grid,
        )
        aggregated["unit_true_by_seed"][train_seed] = per_seed["unit_true"]
        aggregated["unit_hat"].update(per_seed["unit_hat"])
        aggregated["pop_hat_store"].update(per_seed["pop_hat_store"])
        for key, val in per_seed["probe_r2"].items():
            aggregated["probe_r2_store"].setdefault(key, []).append(val)
        for key, val in per_seed["mse_map"].items():
            aggregated["mse_store"].setdefault(key, []).append(val)
        for key, surfs in per_seed["contrast_surface_store"].items():
            aggregated["contrast_surface_store"].setdefault(key, []).extend(surfs)

    return aggregated


def report_scenario_metrics(
    surface: str,
    treatment_type: str,
    aggregated: dict[str, object],
    *,
    methods: list[str],
    downstreams: list[str],
    probes: list[str],
    mse_label: str,
) -> None:
    seeds = aggregated["train_seeds"]
    a_ref = aggregated["a_ref"]
    a_ref_str = (
        f"{float(a_ref):.6f}" if np.ndim(a_ref) == 0
        else np.array2string(np.asarray(a_ref), precision=6, separator=",")
    )
    print(f"\n{'=' * 72}")
    print(
        f"[surface={surface}  treatment={treatment_type}] "
        f"Unit-level contrast {mse_label}  (mean ± sd over seeds)",
    )
    print(f"Reference a0 = {a_ref_str}")
    print(f"{'=' * 72}")
    for method in methods:
        probe_parts = []
        for label in probes:
            vals = aggregated["probe_r2_store"].get((method, label), [])
            if vals:
                probe_parts.append(f"{label}={float(np.mean(vals)):.4f}")
        print(f"  {method:10}  " + "  ".join(probe_parts))
        for ds in downstreams:
            vals = aggregated["mse_store"].get((method, ds), [])
            if not vals:
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {method:10}  {ds:8}  {mean:.6f}  ± {sd:.6f}  (n={len(vals)})")

    if treatment_type == "binary":
        _print_binary_contrast_table_test(aggregated, methods, downstreams)


def _print_binary_contrast_table_test(
    aggregated: dict[str, object],
    methods: list[str],
    downstreams: list[str],
) -> None:
    """Print ground truth vs estimated population-mean contrasts for binary treatment."""
    a_grid = np.asarray(aggregated["a_grid"], dtype=np.float64)
    a_ref = aggregated["a_ref"]
    tau_pop_true = np.asarray(aggregated["tau_pop_true"], dtype=np.float64)
    train_seeds = aggregated["train_seeds"]
    pop_hat_store = aggregated["pop_hat_store"]

    dA = a_grid.shape[1] if a_grid.ndim > 1 else 1
    n_grid = len(a_grid)
    a_ref_arr = np.asarray(a_ref, dtype=np.float64).ravel()

    non_ref_idx = [
        j for j in range(n_grid)
        if not np.allclose(
            np.asarray(a_grid[j], dtype=np.float64).ravel(), a_ref_arr, atol=1e-6,
        )
    ]
    if not non_ref_idx:
        return

    keys = [(m, ds) for m in methods for ds in downstreams]
    header_parts = ["true"] + [f"{m}/{ds}" for m, ds in keys]
    print(f"\n  Binary contrast values (pop mean E_L[tau(a,L)], mean ± sd over seeds):")
    print(f"  {'a':12s}  " + "  ".join(f"{h:>20s}" for h in header_parts))
    for j in non_ref_idx:
        a_j = np.asarray(a_grid[j], dtype=np.float64).ravel()
        a_str = "[" + ",".join(f"{v:.0f}" for v in a_j) + "]" if dA > 1 else f"{float(a_j[0]):.1f}"
        true_val = float(tau_pop_true[j])
        row = [f"{true_val:+.4f}"]
        for m, ds in keys:
            seed_vals = [
                float(pop_hat_store[(ts, m, ds)][0][j])
                for ts in train_seeds
                if (ts, m, ds) in pop_hat_store
            ]
            if seed_vals:
                mean = float(np.mean(seed_vals))
                sd = float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0
                row.append(f"{mean:+.4f}±{sd:.4f}")
            else:
                row.append("n/a")
        print(f"  {a_str:12s}  " + "  ".join(f"{v:>20s}" for v in row))


def render_scenario_figures(
    out_dir: Path,
    aggregated: dict[str, object],
    *,
    surface: str,
    treatment_type: str,
    methods: list[str],
    downstreams: list[str],
    probes: list[str],
) -> None:
    dA = int(aggregated.get("dA", 1))
    if dA > 1:
        return  # visualization not supported for vector treatment (continuous or binary)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = aggregated["train_seeds"]
    a_grid = np.asarray(aggregated["a_grid"], dtype=np.float64)
    l_grid = np.asarray(aggregated["l_grid"], dtype=np.float64)
    n_seeds = len(train_seeds)

    save_population_contrast_figures(
        out_dir, a_grid, aggregated["tau_pop_true"],
        train_seeds, methods, downstreams, aggregated["pop_hat_store"],
        surface=surface, treatment_type=treatment_type,
    )
    save_probe_figure(
        out_dir, methods, probes, aggregated["probe_r2_store"], train_seeds,
        surface=surface, treatment_type=treatment_type,
    )
    save_unit_level_figures(
        out_dir, train_seeds, methods, downstreams,
        aggregated["unit_true_by_seed"], aggregated["unit_hat"],
        a_grid, int(aggregated["unit_plot_index"]),
        surface=surface, treatment_type=treatment_type,
        unit_plot_bins=40,
    )

    true_surf = np.asarray(aggregated["true_surf"], dtype=np.float64)
    a_ref_scalar = float(np.asarray(aggregated["a_ref"]).flat[0])

    for downstream in downstreams:
        tau_hat_by_method: dict[str, np.ndarray] = {}
        n_seeds_eff = 0
        for method in methods:
            surfs = aggregated["contrast_surface_store"].get((method, downstream), [])
            if not surfs:
                continue
            tau_hat_by_method[method] = np.mean(np.stack(surfs, axis=0), axis=0)
            n_seeds_eff = max(n_seeds_eff, len(surfs))
        if not tau_hat_by_method:
            continue

        if treatment_type == "continuous":
            save_surface_heatmap(
                out_dir / f"npiv_syn_surface_{downstream}.png",
                a_grid=a_grid, l_grid=l_grid, a_ref=a_ref_scalar,
                tau_true=true_surf, tau_hat_by_method=tau_hat_by_method,
                downstream=downstream, surface=surface, treatment_type=treatment_type,
                n_seeds=n_seeds_eff,
            )
        else:
            # Binary: shape (n_l, 2); column 1 = CATE at a=1 vs a_ref=0
            cate_true = true_surf[:, 1]
            cate_hat_by_method = {m: v[:, 1] for m, v in tau_hat_by_method.items()}
            save_binary_cate_curves(
                out_dir / f"npiv_syn_cate_{downstream}.png",
                l_grid=l_grid, cate_true=cate_true,
                cate_hat_by_method=cate_hat_by_method,
                downstream=downstream, surface=surface, n_seeds=n_seeds_eff,
            )

    if treatment_type == "binary":
        true_ate = float(np.asarray(aggregated["tau_pop_true"], dtype=np.float64)[1])
        ate_by_method_ds: dict[tuple[str, str], list[float]] = {}
        for (ts, method, ds), curves in aggregated["pop_hat_store"].items():
            for curve in curves:
                ate_by_method_ds.setdefault((method, ds), []).append(float(curve[1]))
        save_binary_ate_bars(
            out_dir / "npiv_syn_binary_ate.png",
            true_ate=true_ate, ate_by_method_ds=ate_by_method_ds,
            methods=methods, downstreams=downstreams, surface=surface, n_seeds=n_seeds,
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mse_label = contrast_metric_label(args.mse_scale)

    ckpt_root = Path(args.checkpoint_dir).expanduser().resolve()
    if not ckpt_root.exists():
        raise SystemExit(
            f"Checkpoint directory does not exist: {ckpt_root}. "
            "Run run_npiv_syn_batch.sh first.",
        )

    out_root = (
        Path(args.out_fig).expanduser().resolve()
        if str(args.out_fig).strip()
        else ckpt_root.parent / "figs" / "npiv_syn"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    scenarios = _resolve_scenarios(args)
    if not scenarios:
        raise SystemExit("No valid scenarios to evaluate. Check --treatment-types and --surfaces.")

    methods = list(dict.fromkeys(args.methods))
    downstreams = list(dict.fromkeys(args.downstream))

    print(f"Checkpoints: {ckpt_root}")
    print(f"Methods: {methods}  Downstream: {downstreams}")
    print(f"Scenarios: {scenarios}")

    mse_by_tt: dict[str, dict[tuple[str, str, str], list[float]]] = {}
    max_seed_count_by_tt: dict[str, int] = {}

    for treatment_type, surface in scenarios:
        train_seeds = _resolve_train_seeds(args, ckpt_root, treatment_type, surface)
        print(
            f"\n{'=' * 72}\n"
            f"=== Scenario: treatment={treatment_type}  surface={surface}  "
            f"seeds={train_seeds} ===",
        )
        aggregated = aggregate_evaluations_for_scenario(
            args,
            surface=surface, treatment_type=treatment_type, train_seeds=train_seeds,
            device=device, ckpt_root=ckpt_root, methods=methods, downstreams=downstreams,
        )
        report_scenario_metrics(
            surface, treatment_type, aggregated,
            methods=methods, downstreams=downstreams, probes=args.probes,
            mse_label=mse_label,
        )
        max_seed_count_by_tt[treatment_type] = max(
            max_seed_count_by_tt.get(treatment_type, 0), len(train_seeds),
        )
        for (method, ds), vals in aggregated["mse_store"].items():
            mse_by_tt.setdefault(treatment_type, {}).setdefault(
                (surface, method, ds), [],
            ).extend(vals)

        render_scenario_figures(
            out_root / treatment_type / surface,
            aggregated,
            surface=surface, treatment_type=treatment_type,
            methods=methods, downstreams=downstreams, probes=args.probes,
        )

    for treatment_type, mse_keyed in mse_by_tt.items():
        surfaces_for_tt = [s for (tt, s) in scenarios if tt == treatment_type]
        save_mse_heatmap(
            out_root / treatment_type,
            surfaces=surfaces_for_tt, methods=methods, downstreams=downstreams,
            mse_store=mse_keyed,
            n_seeds=max_seed_count_by_tt.get(treatment_type, 1),
            treatment_type=treatment_type,
            metric_label=mse_label,
        )


if __name__ == "__main__":
    main()
