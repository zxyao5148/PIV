"""Visualisation helpers for the synthetic NPIV benchmark."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MPL_CACHE = _PROJECT_ROOT / "outputs" / ".mplcache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


# ═══════════════════════════════════════════════════════════════════
#  Single-run figures (used by train_npiv_syn)
# ═══════════════════════════════════════════════════════════════════


def save_npiv_syn_contrast_surface(
    out_path: Path,
    *,
    a_grid: np.ndarray,
    l_grid: np.ndarray,
    a_ref: float,
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    method: str,
    downstream: str,
    surface: str,
    vary_idx: int,
    design_seed: int,
    data_seed: int,
    train_seed: int,
    contrast_mse: float | None = None,
) -> None:
    """Three-panel contrast-surface figure: truth, estimate, and error."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    l_grid = np.asarray(l_grid, dtype=np.float64)
    tau_true = np.asarray(tau_true, dtype=np.float64)
    tau_hat = np.asarray(tau_hat, dtype=np.float64)
    tau_err = tau_hat - tau_true

    a_lo, a_hi = float(a_grid[0]), float(a_grid[-1])
    l_lo, l_hi = float(l_grid[0]), float(l_grid[-1])
    if np.isclose(l_lo, l_hi):
        l_lo -= 0.5
        l_hi += 0.5
    extent = [a_lo, a_hi, l_lo, l_hi]

    main_vmax = float(max(np.abs(tau_true).max(), np.abs(tau_hat).max(), 1e-8))
    err_vmax = float(max(np.abs(tau_err).max(), 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    for ax, data, title, vmin, vmax in [
        (axes[0], tau_true, rf"Truth  $\tau(A, L_{{{vary_idx}}})$", -main_vmax, main_vmax),
        (axes[1], tau_hat, rf"Estimate  $\hat\tau(A, L_{{{vary_idx}}})$", -main_vmax, main_vmax),
        (axes[2], tau_err, r"$\hat\tau - \tau$", -err_vmax, err_vmax),
    ]:
        im = ax.imshow(
            data, origin="lower", aspect="auto", extent=extent,
            cmap="coolwarm", vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel("Intervention A")
        ax.set_ylabel(f"L[{vary_idx}]")

    if l_grid.size <= 8:
        for ax in axes:
            ax.set_yticks(l_grid)

    mse_note = "" if contrast_mse is None else f"  contrast MSE={contrast_mse:.4f}"
    fig.suptitle(
        f"Synthetic NPIV contrast surface — {method} + {downstream}\n"
        f"surface={surface}  a_ref={a_ref:.3f}  "
        f"design={design_seed}  data={data_seed}  train={train_seed}{mse_note}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_npiv_syn_marginalized_response(
    out_path: Path,
    *,
    a_grid: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    method: str,
    downstream: str,
    surface: str,
    design_seed: int,
    data_seed: int,
    train_seed: int,
    level_mse: float | None = None,
) -> None:
    """Sample-average structural curve μ(a) ≈ E_L[h(a, L)] on the test set (scalar a)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    h_true = np.asarray(h_true, dtype=np.float64)
    h_hat = np.asarray(h_hat, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.plot(a_grid, h_true, color="black", lw=2.4, label=r"Truth $\bar h(a)$")
    ax.plot(a_grid, h_hat, color="#d55e00", lw=2.0, label=rf"Estimate $\hat{{\bar h}}(a)$")
    mse_note = "" if level_mse is None else f"  level MSE={level_mse:.4f}"
    ax.set_xlabel("Treatment A")
    ax.set_ylabel(r"Marginal response $\bar h(a) = \mathbb{E}_i[h(a, L_i)]$")
    ax.set_title(
        f"Synthetic NPIV marginalized response — {method} + {downstream}\n"
        f"surface={surface}  "
        f"design={design_seed}  data={data_seed}  train={train_seed}{mse_note}",
    )
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_npiv_syn_overlay_marginalized_response(
    out_path: Path,
    *,
    a_grid: np.ndarray,
    h_true: np.ndarray,
    h_hat_by_label: dict[str, np.ndarray],
    downstream: str,
    surface: str,
    design_seed: int,
    data_seed: int,
    train_seed: int,
) -> None:
    """Overlay μ̂(a) for several IV methods (same downstream estimator); one truth curve."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    h_true = np.asarray(h_true, dtype=np.float64)
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(h_hat_by_label))))

    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.plot(a_grid, h_true, color="black", lw=2.6, label=r"Truth $\bar h(a)$")
    for idx, (label, h_hat) in enumerate(h_hat_by_label.items()):
        arr = np.asarray(h_hat, dtype=np.float64)
        ax.plot(a_grid, arr, lw=2.0, color=palette[idx], label=label)
    ax.set_xlabel("Treatment A")
    ax.set_ylabel(r"$\bar h(a)$")
    ax.set_title(
        f"Marginalized response overlay — {downstream}  surface={surface}\n"
        f"design={design_seed}  data={data_seed}  train={train_seed}",
    )
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_npiv_syn_population_contrast(
    out_path: Path,
    *,
    a_grid: np.ndarray,
    a_ref: float,
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    method: str,
    downstream: str,
    surface: str,
    design_seed: int,
    data_seed: int,
    train_seed: int,
    contrast_mse: float | None = None,
) -> None:
    """L-marginalized contrast curve for a single run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    tau_true = np.asarray(tau_true, dtype=np.float64)
    tau_hat = np.asarray(tau_hat, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.plot(a_grid, tau_true, color="black", lw=2.4, label=r"Truth $\bar\tau(a)$")
    ax.plot(a_grid, tau_hat, color="#d55e00", lw=2.0, label=rf"Estimate $\hat{{\bar\tau}}(a)$")
    ax.axhline(0.0, color="#777777", lw=0.8, alpha=0.6)
    ax.axvline(float(a_ref), color="#777777", lw=0.8, alpha=0.6, linestyle="--")
    ax.set_xlabel("Intervention A")
    ax.set_ylabel(r"L-marginalized contrast $\bar\tau(a)$")
    mse_note = "" if contrast_mse is None else f"  contrast MSE={contrast_mse:.4f}"
    ax.set_title(
        f"Synthetic NPIV marginalized contrast — {method} + {downstream}\n"
        f"surface={surface}  a_ref={a_ref:.3f}  "
        f"design={design_seed}  data={data_seed}  train={train_seed}{mse_note}",
    )
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Aggregated figures (used by test_npiv_syn)
# ═══════════════════════════════════════════════════════════════════


def _percentile_bin_mean(
    x_frac: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(int(n_bins), np.nan, dtype=np.float64)
    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (x_frac >= lo) & (x_frac <= hi if b == int(n_bins) - 1 else x_frac < hi)
        if np.any(mask):
            means[b] = float(np.mean(y[mask]))
    return centers.astype(np.float64), means


def save_population_contrast_figures(
    out_dir: Path,
    a_grid: np.ndarray,
    tau_pop_true: np.ndarray,
    train_seeds: list[int],
    methods: list[str],
    downstreams: list[str],
    pop_hat_store: dict[tuple[int, str, str], list[np.ndarray]],
    *,
    surface: str,
    treatment_type: str,
) -> None:
    """Mean L-marginalized population contrast curve per method, one figure per downstream."""
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(methods))))

    for ds in downstreams:
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        ax.plot(a_grid, tau_pop_true, "k-", lw=2.5, label=r"Truth $\bar\tau(a)$")
        for idx, method in enumerate(methods):
            curves: list[np.ndarray] = []
            for seed in train_seeds:
                curves.extend(pop_hat_store.get((seed, method, ds), []))
            if not curves:
                continue
            arr = np.stack(curves, axis=0)
            label = f"{method}  (mean n={arr.shape[0]})"
            ax.plot(a_grid, arr.mean(axis=0), lw=1.8, color=palette[idx], label=label)

        ax.set_xlabel(r"Intervention $a$")
        ax.set_ylabel(r"Population contrast $\bar\tau(a)$")
        ax.set_title(
            f"NPIV-syn: population contrast — surface={surface}  treatment={treatment_type}\n"
            f"(NPIV: {ds}; mean across n_seeds={len(train_seeds)})",
        )
        ax.legend(fontsize=8, loc="best")
        out_path = out_dir / f"npiv_syn_population_{ds}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"\nSaved population contrast figure ({ds}) to {out_path}")


def save_surface_heatmap(
    out_path: Path,
    *,
    a_grid: np.ndarray,
    l_grid: np.ndarray,
    a_ref: float,
    tau_true: np.ndarray,
    tau_hat_by_method: dict[str, np.ndarray],
    downstream: str,
    surface: str,
    treatment_type: str,
    n_seeds: int = 1,
) -> None:
    """Truth + all methods in one row — averaged contrast surface (continuous treatment).

    ``tau_true`` and each value in ``tau_hat_by_method`` have shape
    ``(len(l_grid), len(a_grid))``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tau_true = np.asarray(tau_true, dtype=np.float64)
    methods = list(tau_hat_by_method.keys())

    a_lo, a_hi = float(a_grid[0]), float(a_grid[-1])
    l_lo, l_hi = float(l_grid[0]), float(l_grid[-1])
    if np.isclose(l_lo, l_hi):
        l_lo -= 0.5
        l_hi += 0.5
    extent = [a_lo, a_hi, l_lo, l_hi]

    all_arrs = [tau_true] + [np.asarray(v, dtype=np.float64) for v in tau_hat_by_method.values()]
    vmax = float(max(np.abs(a).max() for a in all_arrs) or 1e-8)

    n_panels = 1 + len(methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.1 * n_panels, 4.2), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    im0 = axes[0].imshow(
        tau_true, origin="lower", aspect="auto", extent=extent,
        cmap="coolwarm", vmin=-vmax, vmax=vmax,
    )
    axes[0].set_title(rf"Truth  $\tau(A, L_0)$   ($a_0={a_ref:.3f}$)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    for ax, method in zip(axes[1:], methods):
        arr = np.asarray(tau_hat_by_method[method], dtype=np.float64)
        im = ax.imshow(
            arr, origin="lower", aspect="auto", extent=extent,
            cmap="coolwarm", vmin=-vmax, vmax=vmax,
        )
        ax.set_title(rf"$\hat\tau(A, L_0)$  —  {method}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes:
        ax.set_xlabel(r"Intervention $A$")
        ax.set_ylabel(r"$L[0]$")

    seeds_note = f"mean over n_seeds={n_seeds}" if n_seeds > 1 else "single seed"
    fig.suptitle(
        f"NPIV-syn surface — {downstream}  surface={surface}  ({seeds_note})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved surface heatmap ({downstream}) to {out_path}")


def save_binary_cate_curves(
    out_path: Path,
    *,
    l_grid: np.ndarray,
    cate_true: np.ndarray,
    cate_hat_by_method: dict[str, np.ndarray],
    downstream: str,
    surface: str,
    n_seeds: int = 1,
) -> None:
    """CATE-vs-L[0] curves for binary treatment, one line per method.

    ``cate_true`` and each value in ``cate_hat_by_method`` are 1-D arrays of
    length ``len(l_grid)`` representing ``E[h0(1,L)-h0(0,L) | L[0]=l]``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(cate_hat_by_method))))
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(l_grid, cate_true, "k-", lw=2.5, label="Truth CATE")
    for idx, (method, arr) in enumerate(cate_hat_by_method.items()):
        ax.plot(l_grid, arr, lw=1.8, color=palette[idx],
                label=f"{method}  (mean n={n_seeds})")
    ax.axhline(0.0, color="#777777", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$L[0]$")
    ax.set_ylabel(r"CATE  $E[h_0(1,L) - h_0(0,L) \mid L[0]=l]$")
    ax.set_title(
        f"NPIV-syn binary CATE curve — {downstream}  surface={surface}\n"
        f"(mean over n_seeds={n_seeds})",
    )
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved binary CATE curve ({downstream}) to {out_path}")


def save_binary_ate_bars(
    out_path: Path,
    *,
    true_ate: float,
    ate_by_method_ds: dict[tuple[str, str], list[float]],
    methods: list[str],
    downstreams: list[str],
    surface: str,
    n_seeds: int = 1,
) -> None:
    """Bar chart of estimated ATE per method vs truth, one subplot per downstream.

    ``ate_by_method_ds[(method, downstream)]`` is a list of ATE values over seeds.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(methods))))

    n_panels = max(len(downstreams), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for ax, ds in zip(axes, downstreams):
        x = np.arange(len(methods))
        means, sds = [], []
        for method in methods:
            vals = ate_by_method_ds.get((method, ds), [])
            means.append(float(np.mean(vals)) if vals else float("nan"))
            sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        bars = ax.bar(x, means, yerr=sds, capsize=4, color=palette[:len(methods)], alpha=0.7)
        ax.axhline(true_ate, color="black", lw=2.0, linestyle="--", label=f"Truth ATE={true_ate:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel("ATE estimate")
        ax.set_title(f"Binary ATE — {ds}")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"NPIV-syn binary ATE — surface={surface}  (mean ± sd over n_seeds={n_seeds})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved binary ATE bar chart to {out_path}")


def save_unit_level_figures(
    out_dir: Path,
    train_seeds: list[int],
    methods: list[str],
    downstreams: list[str],
    unit_true_by_seed: dict[int, np.ndarray],
    unit_hat: dict[tuple[int, str, str], np.ndarray],
    a_grid: np.ndarray,
    j_unit: int,
    *,
    surface: str,
    treatment_type: str,
    unit_plot_bins: int = 40,
) -> None:
    """Unit-level contrast vs percentile rank, averaged across seeds. One figure per downstream."""
    out_dir.mkdir(parents=True, exist_ok=True)
    a_sel = float(a_grid[j_unit])
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(methods))))
    use_bins = int(unit_plot_bins) > 0
    n_bins = max(2, int(unit_plot_bins)) if use_bins else 256

    for ds in downstreams:
        y_gt_rows: list[np.ndarray] = []
        hat_rows: dict[str, list[np.ndarray]] = {m: [] for m in methods}
        x_centers_ref: np.ndarray | None = None

        for seed in train_seeds:
            true_vec = unit_true_by_seed.get(seed)
            if true_vec is None or true_vec.shape[0] < 2:
                continue
            n = int(true_vec.shape[0])
            order = np.argsort(true_vec)
            x_frac = (np.arange(n) + 0.5) / n
            y_gt = true_vec[order]

            if use_bins:
                xc, yg = _percentile_bin_mean(x_frac, y_gt, n_bins)
                if x_centers_ref is None:
                    x_centers_ref = xc
                y_gt_rows.append(yg)
                for method in methods:
                    hat = unit_hat.get((seed, method, ds))
                    if hat is None:
                        hat_rows[method].append(np.full(n_bins, np.nan))
                    else:
                        _, ym = _percentile_bin_mean(x_frac, hat[order], n_bins)
                        hat_rows[method].append(ym)
            else:
                if x_centers_ref is None:
                    x_centers_ref = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
                q = x_centers_ref
                y_gt_rows.append(np.interp(q, x_frac, y_gt))
                for method in methods:
                    hat = unit_hat.get((seed, method, ds))
                    if hat is None:
                        hat_rows[method].append(np.full_like(q, np.nan))
                    else:
                        hat_rows[method].append(np.interp(q, x_frac, hat[order]))

        if x_centers_ref is None or not y_gt_rows:
            continue

        y_gt_mean = np.nanmean(np.stack(y_gt_rows, axis=0), axis=0)
        fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
        gt_mask = np.isfinite(y_gt_mean)
        ax.plot(x_centers_ref[gt_mask], y_gt_mean[gt_mask], color="#5c4033", lw=2.0, label="GT")

        for idx, method in enumerate(methods):
            rows = hat_rows.get(method, [])
            if not rows:
                continue
            ym_mean = np.nanmean(np.stack(rows, axis=0), axis=0)
            mask = np.isfinite(ym_mean)
            if not np.any(mask):
                continue
            ax.plot(x_centers_ref[mask], ym_mean[mask],
                    color=palette[idx], lw=1.8, alpha=1.0, label=method)

        mode_note = f"binned ({n_bins} bins)" if use_bins else "interpolated"
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel(r"order(true unit-level $\tau$) (percentile rank)")
        ax.set_ylabel(r"Unit-level contrast $\tau$")
        ax.set_title(
            f"NPIV-syn: unit-level contrast — {mode_note}\n"
            f"[surface={surface}  treatment={treatment_type}  NPIV: {ds}  "
            f"seeds={train_seeds}  $a$={a_sel:.3f}]",
        )
        ax.legend(fontsize=7, loc="best", ncol=2)
        out_path = out_dir / f"npiv_syn_unit_{ds}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"\nSaved unit-level figure ({ds}) to {out_path}")


def save_probe_figure(
    out_dir: Path,
    methods: list[str],
    probe_labels: list[str],
    probe_r2_store: dict[tuple[str, str], list[float]],
    train_seeds: list[int],
    *,
    surface: str,
    treatment_type: str,
    filename: str = "npiv_syn_probe_r2.png",
    exclude_methods: Sequence[str] = ("TrueIV", "NoneIV"),
) -> None:
    """Box plots of representation-probe R² across train_seeds, one panel per probe label."""
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = frozenset(exclude_methods)
    plot_methods = [m for m in methods if m not in skip]
    n_panels = len(probe_labels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(4.0, 2.5 * len(plot_methods)) * n_panels / max(n_panels, 1), 4.5),
        sharey=False, constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, label in zip(axes, probe_labels):
        data: list[list[float]] = []
        labels: list[str] = []
        for method in plot_methods:
            vals = list(probe_r2_store.get((method, label), []))
            if not vals:
                continue
            data.append(vals)
            labels.append(method)
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, labels=labels, showmeans=True, meanline=True,
                        patch_artist=True, widths=0.6)
        palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(data))))
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(1.5)
        for x, vals in enumerate(data, start=1):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, color="k", alpha=0.7, zorder=3)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        y_flat = [float(v) for row in data for v in row]
        if y_flat:
            y_min, y_max = min(y_flat), max(y_flat)
            span = max(y_max - y_min, 0.08)
            pad = 0.06 * span
            lo = max(y_min - pad, -0.35)
            hi = min(y_max + pad, 1.05)
            if hi - lo < 0.18:
                mid = 0.5 * (lo + hi)
                lo, hi = max(mid - 0.09, -0.35), min(mid + 0.09, 1.05)
            ax.set_ylim(lo, hi)
        ax.set_ylabel(rf"Test $R^2$")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"NPIV-syn: representation-probe quality — surface={surface}  treatment={treatment_type}\n"
        f"(test R² across n_seeds={len(train_seeds)})",
        fontsize=11,
    )
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved probe figure to {out_path}")


def save_mse_heatmap(
    out_dir: Path,
    *,
    surfaces: Sequence[str],
    methods: Sequence[str],
    downstreams: Sequence[str],
    mse_store: dict[tuple[str, str, str], list[float]],
    n_seeds: int,
    treatment_type: str,
    metric_label: str = "MSE",
    filename_template: str = "npiv_syn_mse_heatmap_{ds}.png",
) -> None:
    """Heatmap of unit-level contrast MSE: rows=surfaces, cols=methods, one fig per downstream."""
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces = list(surfaces)
    methods = list(methods)
    downstreams = list(downstreams)
    if not surfaces or not methods or not downstreams:
        return

    for ds in downstreams:
        mat = np.full((len(surfaces), len(methods)), np.nan, dtype=np.float64)
        for i, surf in enumerate(surfaces):
            for j, method in enumerate(methods):
                vals = mse_store.get((surf, method, ds), [])
                if vals:
                    mat[i, j] = float(np.mean(vals))
        if not np.any(np.isfinite(mat)):
            continue

        fig_w = max(5.5, 1.05 * len(methods) + 2.5)
        fig_h = max(2.6, 0.75 * len(surfaces) + 1.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        vmin = float(np.nanmin(mat))
        vmax = float(np.nanmax(mat))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_yticks(range(len(surfaces)))
        ax.set_yticklabels(surfaces)
        ax.set_xlabel("IV method")
        ax.set_ylabel("Surface")
        ax.set_title(
            f"NPIV-syn {metric_label} ({treatment_type}) — {ds}  (mean n_seeds={n_seeds})",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, label=metric_label)

        denom = max(vmax - vmin, 1e-12)
        for i in range(len(surfaces)):
            for j in range(len(methods)):
                v = mat[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="0.6")
                else:
                    rel = (v - vmin) / denom
                    ax.text(j, i, f"{v:.3g}", ha="center", va="center",
                            fontsize=9, color="white" if rel < 0.5 else "black")

        out_path = out_dir / filename_template.format(ds=ds)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"\nSaved MSE heatmap ({ds}) to {out_path}")


__all__ = [
    "save_binary_ate_bars",
    "save_binary_cate_curves",
    "save_mse_heatmap",
    "save_npiv_syn_contrast_surface",
    "save_npiv_syn_marginalized_response",
    "save_npiv_syn_overlay_marginalized_response",
    "save_npiv_syn_population_contrast",
    "save_population_contrast_figures",
    "save_probe_figure",
    "save_surface_heatmap",
    "save_unit_level_figures",
]
