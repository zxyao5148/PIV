#!/usr/bin/env python3
"""Demands benchmark figures (population curves + unit-level contrast plots).

Used by ``train_demands.py`` / ``test_demands.py``. Matplotlib is configured for headless ``Agg`` and a
workspace-local font cache (see ``MPLCONFIGDIR`` setup below).
"""
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


def percentile_bin_mean_fixed_grid(
    x_frac: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``n_bins`` equal-width bins on ``x_frac`` in ``[0, 1]`` with a fixed x-center grid.

    Each bin centre is ``0.5*(edges[b] + edges[b+1])``; empty bins return NaN so that
    multiple replicates share one abscissa and can be stacked and ``nanmean``'d.
    """
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    x_centers = 0.5 * (edges[:-1] + edges[1:])
    y_means = np.full(int(n_bins), np.nan, dtype=np.float64)
    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == int(n_bins) - 1:
            mask = (x_frac >= lo) & (x_frac <= hi)
        else:
            mask = (x_frac >= lo) & (x_frac < hi)
        if np.any(mask):
            y_means[b] = float(np.mean(y[mask]))
    return x_centers.astype(np.float64), y_means


def save_unit_level_figures(
    out_dir: Path,
    seeds: list[int],
    methods: list[str],
    downstreams: list[str],
    unit_true_by_seed: dict[int, np.ndarray],
    unit_hat: dict[tuple[int, str, str], np.ndarray],
    a_grid: np.ndarray,
    j_unit: int,
    *,
    unit_plot_bins: int = 40,
) -> None:
    """One figure per downstream, **averaged across replicate seeds**.

    Test units are ranked by each replicate's own true ``τ`` at ``a_grid[j_unit]``, so the
    x-axis is the percentile rank of the ordered unit.  Curves from different seeds share a
    common abscissa (``n_bins`` equal-width percentile bins when ``unit_plot_bins > 0``; a
    fixed 256-point interpolation grid when ``unit_plot_bins == 0``) and are then averaged
    with ``nanmean``.  A single seed degenerates to that seed's curve.

    Output paths: ``{out_dir}/demands_unit_{ds}.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    a_sel = float(a_grid[j_unit])
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(methods))))
    use_bins = int(unit_plot_bins) > 0
    n_bins = max(2, int(unit_plot_bins)) if use_bins else 0

    def _plot_one_unit_fig(
        ds: str,
        *,
        out_path: Path,
        x_plot_gt: np.ndarray,
        y_plot_gt: np.ndarray,
        series: list[tuple[str, np.ndarray, np.ndarray]],
    ) -> None:
        fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
        gt_mask = np.isfinite(y_plot_gt)
        ax.plot(
            x_plot_gt[gt_mask], y_plot_gt[gt_mask],
            color="#5c4033", lw=2.0 if use_bins else 1.2, label="GT",
        )
        for method, xm, ym in series:
            m = np.isfinite(ym)
            if not np.any(m):
                continue
            midx = methods.index(method) if method in methods else 0
            ax.plot(
                xm[m], ym[m],
                color=palette[midx],
                lw=1.8 if use_bins else 0.9,
                alpha=1.0 if use_bins else 0.9,
                label=method,
            )
        mode_note = (
            f"binned means ({n_bins} percentile bins)" if use_bins
            else "raw (interpolated onto 256-point percentile grid)"
        )

        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel(r"order(true unit-level $\tau$) (percentile rank)")
        ax.set_ylabel(r"Unit-level contrast $\tau$")
        ax.set_title(
            f"Demands: unit-level contrast — {mode_note}\n"
            f"[NPIV: {ds}, mean over seeds={seeds}, $a$={a_sel:.3f}]",
        )
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"\nSaved unit-level figure ({ds}) to {out_path}")

    for ds in downstreams:
        y_gt_rows: list[np.ndarray] = []
        hat_rows: dict[str, list[np.ndarray]] = {m: [] for m in methods}
        x_centers_ref: np.ndarray | None = None

        for seed in seeds:
            true_vec = unit_true_by_seed.get(seed)
            if true_vec is None:
                continue
            n = int(true_vec.shape[0])
            if n < 2:
                continue
            order = np.argsort(true_vec)
            x_frac = (np.arange(n) + 0.5) / n
            y_gt = true_vec[order]
            if use_bins:
                xc, yg = percentile_bin_mean_fixed_grid(x_frac, y_gt, n_bins)
                if x_centers_ref is None:
                    x_centers_ref = xc
                y_gt_rows.append(yg)
                for method in methods:
                    hat = unit_hat.get((seed, method, ds))
                    if hat is None:
                        hat_rows[method].append(
                            np.full(n_bins, np.nan, dtype=np.float64),
                        )
                        continue
                    _, ym = percentile_bin_mean_fixed_grid(
                        x_frac, hat[order], n_bins,
                    )
                    hat_rows[method].append(ym)
            else:
                if x_centers_ref is None:
                    raw_avg_pts = 256
                    x_centers_ref = (
                        (np.arange(raw_avg_pts, dtype=np.float64) + 0.5)
                        / raw_avg_pts
                    )
                q_grid = x_centers_ref
                y_gt_rows.append(np.interp(q_grid, x_frac, y_gt))
                for method in methods:
                    hat = unit_hat.get((seed, method, ds))
                    if hat is None:
                        hat_rows[method].append(
                            np.full_like(q_grid, np.nan, dtype=np.float64),
                        )
                        continue
                    hat_rows[method].append(np.interp(q_grid, x_frac, hat[order]))

        if x_centers_ref is None or not y_gt_rows:
            continue

        y_gt_mean = np.nanmean(np.stack(y_gt_rows, axis=0), axis=0)
        series: list[tuple[str, np.ndarray, np.ndarray]] = []
        for method in methods:
            rows = hat_rows.get(method, [])
            if not rows:
                continue
            ym_mean = np.nanmean(np.stack(rows, axis=0), axis=0)
            if not np.any(np.isfinite(ym_mean)):
                continue
            series.append((method, x_centers_ref, ym_mean))

        out_path = out_dir / f"demands_unit_{ds}.png"
        _plot_one_unit_fig(
            ds,
            out_path=out_path,
            x_plot_gt=x_centers_ref,
            y_plot_gt=y_gt_mean,
            series=series,
        )


def save_treatment_effect_heatmap(
    out_path: Path,
    *,
    a_grid: np.ndarray,
    l1_grid: np.ndarray,
    a_ref: float,
    tau_true: np.ndarray,
    tau_hat_by_method: dict[str, np.ndarray],
    downstream: str,
    n_seeds: int = 1,
) -> None:
    r"""Save a single-row treatment-effect contrast heatmap for one downstream.

    Panels share one colour scale and are arranged as:

    1. Truth    : ``τ(a, L_1) = E_{L_2}[h_0(a, L_1, L_2) − h_0(a_0, L_1, L_2)]``
    2. Method 1 : ``τ̂(a, L_1)``
    3. Method 2 : ``τ̂(a, L_1)``
    4. ...

    No difference heatmap and no 1-D marginal slice are shown.
    ``tau_true`` and every entry in ``tau_hat_by_method`` have shape
    ``(len(l1_grid), len(a_grid))``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tau_true = np.asarray(tau_true, dtype=np.float64)
    methods = list(tau_hat_by_method.keys())
    tau_hat_arrs = {
        method: np.asarray(tau_hat_by_method[method], dtype=np.float64)
        for method in methods
    }

    a_lo, a_hi = float(a_grid[0]), float(a_grid[-1])
    l1_lo, l1_hi = float(l1_grid[0]), float(l1_grid[-1])
    extent = [a_lo, a_hi, l1_lo, l1_hi]
    vmax_sym = float(max(
        [np.abs(tau_true).max(), *[np.abs(arr).max() for arr in tau_hat_arrs.values()], 1e-8],
    ))

    n_panels = 1 + len(methods)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.1 * n_panels, 4.2),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    im0 = axes[0].imshow(
        tau_true, origin="lower", aspect="auto", extent=extent,
        cmap="coolwarm", vmin=-vmax_sym, vmax=vmax_sym,
    )
    axes[0].set_title(rf"Truth  $\tau(A, L_1)$   ($a_0={a_ref:.3f}$)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    for ax, method in zip(axes[1:], methods):
        im = ax.imshow(
            tau_hat_arrs[method], origin="lower", aspect="auto", extent=extent,
            cmap="coolwarm", vmin=-vmax_sym, vmax=vmax_sym,
        )
        ax.set_title(rf"$\hat\tau(A, L_1)$  —  {method}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes:
        ax.set_xlabel(r"Intervention $A$")
        ax.set_ylabel(r"$L_1$  (time-of-year)")

    seeds_note = f"mean over n_seeds={n_seeds}" if n_seeds > 1 else "single seed"
    fig.suptitle(
        f"Demands treatment-effect contrast heatmap — {downstream}  "
        f"($L_2$ marginalised; {seeds_note})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved heatmap figure ({downstream}) to {out_path}")


def save_population_contrast_figures(
    out_dir: Path,
    a_grid: np.ndarray,
    tau_pop_true: np.ndarray,
    seeds: list[int],
    methods: list[str],
    downstreams: list[str],
    pop_hat_store: dict[tuple[int, str, str], list[np.ndarray]],
) -> None:
    """Mean population curve per method across replicate seeds (no SD band).

    Output paths: ``{out_dir}/demands_population_{ds}.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rep = len(seeds)
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(methods))))

    for ds in downstreams:
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        ax.plot(a_grid, tau_pop_true, "k-", lw=2.5, label=r"Truth $\bar\tau(a)$")

        for idx, method in enumerate(methods):
            curves: list[np.ndarray] = []
            for seed in seeds:
                curves.extend(pop_hat_store.get((seed, method, ds), []))
            if not curves:
                continue
            arr = np.stack(curves, axis=0)
            mean_c = arr.mean(axis=0)
            n_curves = arr.shape[0]
            label = f"{method}  (mean over n={n_curves})"
            ax.plot(a_grid, mean_c, lw=1.8, color=palette[idx], label=label)

        ax.set_xlabel(r"Intervention $a$")
        ax.set_ylabel(r"Population contrast $\bar\tau(a)$")
        ax.set_title(
            "Demands: population treatment-effect curve  "
            f"(NPIV: {ds}; mean across n_seeds={n_rep})",
        )
        ax.legend(fontsize=8, loc="best")
        out_path = out_dir / f"demands_population_{ds}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"\nSaved population curve figure ({ds}) to {out_path}")


def save_representation_probe_figure(
    out_dir: Path,
    methods: list[str],
    targets: list[str],
    probe_r2_store: dict[tuple[str, str], list[float]],
    seeds: list[int],
    *,
    filename: str = "demands_probe_r2.png",
    exclude_methods: Sequence[str] = ("TrueIV", "NoneIV"),
) -> None:
    """Box plots of representation-probe :math:`R^2` across replicate seeds.

    Two side-by-side panels (one per entry in ``targets``, typically ``["V", "D"]``)
    over seeds, one box per IV method.  For ``V``, training uses ``(W,Z)`` for PIV
    and ``Z`` only for other methods (see ``run_demands``); for ``D``, ``(W,Z)``
    is used for all methods.  The vertical axis is tightened to the plotted data
    (with a small margin, capped to a sensible band for :math:`R^2`).

    Parameters
    ----------
    out_dir:
        Directory to write the figure into.
    methods:
        IV methods (x-tick labels). The display order is preserved.
    targets:
        Probe targets (one subplot per entry), e.g. ``["V", "D"]``.
    probe_r2_store:
        Mapping ``(method, target) -> list of test R² across seeds``.
    seeds:
        Replicate seeds used (only its length is shown in the title).
    filename:
        Output filename placed under ``out_dir``.
    exclude_methods:
        Methods omitted from this figure only (default: ``TrueIV``, ``NoneIV``).
        Probe metrics in the log are unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rep = len(seeds)
    n_panels = len(targets)
    skip = frozenset(exclude_methods)
    plot_methods = [m for m in methods if m not in skip]
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(5.0, 3.0 * len(plot_methods)) * n_panels / max(n_panels, 1), 4.5),
        sharey=False,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        data: list[list[float]] = []
        labels: list[str] = []
        medians: list[float] = []
        for method in plot_methods:
            vals = list(probe_r2_store.get((method, target), []))
            if not vals:
                continue
            data.append(vals)
            labels.append(method)
            medians.append(float(np.median(vals)))
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            widths=0.6,
        )
        palette = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(data))))
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(1.5)
        for x, vals in enumerate(data, start=1):
            jitter = (np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals)))
            ax.scatter(
                np.full(len(vals), x) + jitter, vals,
                s=18, color="k", alpha=0.7, zorder=3,
            )
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
                lo, hi = mid - 0.09, mid + 0.09
                lo = max(lo, -0.35)
                hi = min(hi, 1.05)
            ax.set_ylim(lo, hi)

        if target == "V":
            ax.set_ylabel(r"Test $R^2$  (PIV: $(W,Z)\!\to\!V$; others: $Z\!\to\!V$)")
            ax.set_title(r"Probe $\to V$  (PIV uses $(W,Z)$; others $Z$ only)")
        else:
            ax.set_ylabel(rf"Test $R^2$  $(W, Z) \rightarrow {target}$")
            ax.set_title(rf"Probe $(W, Z) \rightarrow {target}$")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Demands: representation-probe quality  "
        f"(test $R^2$ across n_seeds={n_rep})",
        fontsize=12,
    )
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved representation probe figure to {out_path}")


def save_contrast_mse_heatmap(
    out_dir: Path,
    *,
    rhos: Sequence[float],
    methods: Sequence[str],
    downstreams: Sequence[str],
    mse_store: dict[tuple[float, str, str], list[float]],
    n_seeds: int,
    filename_template: str = "demands_mse_heatmap_{ds}.png",
) -> None:
    r"""Heatmap of the unit-level contrast MSE as a function of ``rho`` and IV method.

    One figure per downstream estimator. Rows are the supplied values of
    :math:`\rho` (in the order given), columns are IV methods, and each cell
    is the mean of ``mse_store[(rho, method, ds)]`` across replicate seeds.
    MSE values are annotated inside each cell.

    With a single ``rho`` the figure is a one-row heatmap.

    Parameters
    ----------
    out_dir:
        Directory to write the figure(s) into (created if missing).
    rhos:
        Values of :math:`\rho` to use as rows (display order preserved).
    methods:
        IV methods to use as columns (display order preserved).
    downstreams:
        Downstream estimators to produce one figure each for.
    mse_store:
        Mapping ``(rho, method, downstream) -> list of MSE across seeds``.
    n_seeds:
        Replicate-seed count, shown in the title.
    filename_template:
        Output filename template, formatted with ``ds``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rhos = list(rhos)
    methods = list(methods)
    downstreams = list(downstreams)
    if not rhos or not methods or not downstreams:
        return

    for ds in downstreams:
        mat = np.full((len(rhos), len(methods)), np.nan, dtype=np.float64)
        for i, rho in enumerate(rhos):
            for j, method in enumerate(methods):
                vals = mse_store.get((float(rho), method, ds), [])
                if vals:
                    mat[i, j] = float(np.mean(vals))
        if not np.any(np.isfinite(mat)):
            continue

        fig_w = max(5.5, 1.05 * len(methods) + 2.5)
        fig_h = max(2.6, 0.75 * len(rhos) + 1.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        vmin = float(np.nanmin(mat))
        vmax = float(np.nanmax(mat))
        im = ax.imshow(
            mat, aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_yticks(range(len(rhos)))
        ax.set_yticklabels([rf"$\rho={rho:g}$" for rho in rhos])
        ax.set_xlabel("IV method")
        ax.set_ylabel(r"Endogeneity $\rho$")
        ax.set_title(
            f"Demands unit-level contrast MSE — {ds}  "
            f"(mean across n_seeds={n_seeds})",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, label="MSE")

        denom = max(vmax - vmin, 1e-12)
        for i in range(len(rhos)):
            for j in range(len(methods)):
                v = mat[i, j]
                if not np.isfinite(v):
                    ax.text(
                        j, i, "—",
                        ha="center", va="center",
                        fontsize=9, color="0.6",
                    )
                    continue
                rel = (v - vmin) / denom
                color = "white" if rel < 0.5 else "black"
                ax.text(
                    j, i, f"{v:.3g}",
                    ha="center", va="center",
                    fontsize=9, color=color,
                )

        out_path = out_dir / filename_template.format(ds=ds)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"\nSaved MSE heatmap ({ds}) to {out_path}")
