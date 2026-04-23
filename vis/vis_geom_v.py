"""Visualisation helpers for the geom_v benchmark."""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from scipy.stats import gaussian_kde


# ── Primitives ────────────────────────────────────────────────────

def _kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    grid_size: int = 128,
    pad_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xx, yy, density) on a regular grid via Gaussian KDE."""
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    x_lo, x_hi = x.min(), x.max()
    y_lo, y_hi = y.min(), y.max()
    px = pad_frac * (x_hi - x_lo + 1e-8)
    py = pad_frac * (y_hi - y_lo + 1e-8)

    xx, yy = np.mgrid[
        x_lo - px : x_hi + px : complex(0, grid_size),
        y_lo - py : y_hi + py : complex(0, grid_size),
    ]
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, density


def plot_2d_density(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid_size: int = 128,
    show_xaxis: bool = True,
    show_yaxis: bool = True,
) -> None:
    """2-D KDE density heatmap."""
    xx, yy, density = _kde_2d(x, y, grid_size=grid_size)
    ax.pcolormesh(xx, yy, density, cmap="viridis", shading="auto")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
    if xlabel and show_xaxis:
        ax.set_xlabel(xlabel, fontsize=13)
    if ylabel and show_yaxis:
        ax.set_ylabel(ylabel, fontsize=13)
    if show_xaxis:
        ax.tick_params(axis="x", labelsize=10)
    else:
        ax.tick_params(axis="x", labelbottom=False, length=0)
    if show_yaxis:
        ax.tick_params(axis="y", labelsize=10)
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_aspect("auto")


# ── Combined figure builder ───────────────────────────────────────

def make_geom_v_figure(
    rows: Sequence[str],
    data: Mapping[str, dict],
    methods: Sequence[str] = ("PIV", "ZNet", "AutoIV", "VIV", "GIV"),
    cell_w: float = 3.2,
    cell_h: float = 2.8,
) -> Figure:
    """Build the combined geom_v diagnostic figure (V heatmaps only).

    Predictions are clipped to the ground-truth data range before plotting.
    """
    n_rows = len(rows)
    n_cols = 1 + len(methods)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
    )

    col_headers = ["Ground truth V"] + [f"Learned V by {m}" for m in methods]

    for ri, geom in enumerate(rows):
        d = data[geom]
        is_first_row = ri == 0
        is_last_row = ri == n_rows - 1
        is_first_col = True  # reset per cell below

        v_true = d["V_true"]
        v1_min, v1_max = float(v_true[:, 0].min()), float(v_true[:, 0].max())
        v2_min, v2_max = float(v_true[:, 1].min()), float(v_true[:, 1].max())

        # Col 0 – Ground truth V
        ax0 = axes[ri, 0]
        plot_2d_density(
            v_true[:, 0], v_true[:, 1], ax0,
            title=col_headers[0] if is_first_row else "",
            xlabel="V1", ylabel=f"V2",
            show_xaxis=is_last_row, show_yaxis=True,
        )

        # Cols 1..N – Learned V by each method
        for mi, method in enumerate(methods):
            ci = mi + 1
            ax = axes[ri, ci]
            v_hat = d[f"V_{method}"].copy()
            v_hat[:, 0] = np.clip(v_hat[:, 0], v1_min, v1_max)
            v_hat[:, 1] = np.clip(v_hat[:, 1], v2_min, v2_max)
            plot_2d_density(
                v_hat[:, 0], v_hat[:, 1], ax,
                title=col_headers[ci] if is_first_row else "",
                xlabel="V1", ylabel="V2",
                show_xaxis=is_last_row, show_yaxis=False,
            )

    fig.tight_layout()
    return fig


__all__ = [
    "make_geom_v_figure",
    "plot_2d_density",
]
