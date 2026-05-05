"""Shared evaluation utilities.

Two complementary groups live here:

1. **NPIV counterfactual contrasts.** Given a trained NPIV model ``ĥ`` and its
   normalisation stats ``norms`` (as produced by ``trainer.train_deepiv`` /
   ``trainer.train_dfiv``), evaluate the per-unit treatment contrast

       τ_i(a) = ĥ(a, W_i, L_i) − ĥ(a₀, W_i, L_i)

   on a counterfactual treatment grid ``a_grid`` and a reference ``a_ref``, plus
   its ``(A, L_vary)`` surface variant with the remaining nuisances marginalised.
   The ground-truth counterparts (:func:`true_contrast_matrix`,
   :func:`true_contrast_surface`) are pure DGP concerns: the caller supplies a
   structural-response callable.

2. **Representation probes.** Fit a small MLP probe from a representation
   ``(W, Z)`` or ``Z`` alone to the latent confounder ``V`` and report the
   coefficient of determination on held-out data. Used by ``run_geom_v.py``.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

# ═══════════════════════════════════════════════════════════════════
#  Normalisation / prediction
# ═══════════════════════════════════════════════════════════════════


def _norm_wl_tensor(
    x: Tensor, mean: np.ndarray, std: np.ndarray, device: torch.device,
) -> Tensor:
    """Per-column ``(x − mean) / std`` on ``device`` (float32)."""
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    return (x.to(device) - mean_t) / std_t


def _make_a_input(a_val: float | np.ndarray, n: int) -> np.ndarray:
    """Broadcast a scalar or vector treatment value to shape ``(n,)`` or ``(n, dA)``."""
    a_arr = np.asarray(a_val, dtype=np.float64)
    if a_arr.ndim == 0:
        return np.full(n, float(a_arr), dtype=np.float64)
    return np.broadcast_to(a_arr[None, :], (n, a_arr.shape[0])).copy()


@torch.no_grad()
def predict_hat_a(
    model: nn.Module,
    a_val: float | np.ndarray,
    w: Tensor,
    l: Tensor,
    device: torch.device,
    norms: dict,
) -> np.ndarray:
    """Return ``ĥ(a, W_i, L_i)`` on the original Y scale for all rows of ``w``.

    ``a_val`` may be a scalar (dA=1) or a 1-D array of shape ``(dA,)`` for
    vector continuous treatment.  ``norms`` must provide ``a_mean``, ``a_std``,
    ``y_mean``, ``y_std`` and per-column ``w_mean/w_std``, ``l_mean/l_std``.
    """
    model.eval()
    n = w.shape[0]
    treatment_family = str(norms.get("treatment_family", "continuous"))
    a_mean = np.asarray(norms["a_mean"], dtype=np.float64).ravel()

    if treatment_family == "binary":
        a_val_arr = np.asarray(a_val, dtype=np.float64).ravel()
        if a_val_arr.size > 1:
            a_tensor = torch.as_tensor(
                np.broadcast_to(a_val_arr[None, :], (n, a_val_arr.size)).copy(),
                dtype=torch.float32,
            ).to(device)
        else:
            a_tensor = torch.full((n, 1), float(a_val_arr[0]), dtype=torch.float32, device=device)
    elif a_mean.size > 1:
        # vector continuous treatment
        a_std = np.asarray(norms["a_std"], dtype=np.float64).ravel()
        a_val_np = np.asarray(a_val, dtype=np.float64).ravel()
        a_norm = (a_val_np - a_mean) / a_std  # (dA,)
        a_tensor = torch.as_tensor(
            np.broadcast_to(a_norm[None, :], (n, a_norm.shape[0])).copy(),
            dtype=torch.float32,
        ).to(device)
    else:
        # scalar continuous treatment
        a_norm = (float(np.asarray(a_val).flat[0]) - float(a_mean[0])) / float(
            np.asarray(norms["a_std"]).ravel()[0]
        )
        a_tensor = torch.full((n, 1), float(a_norm), dtype=torch.float32, device=device)

    w_n = _norm_wl_tensor(w, norms["w_mean"], norms["w_std"], device)
    l_n = _norm_wl_tensor(l, norms["l_mean"], norms["l_std"], device)
    pred_norm = model.predict(a_tensor, w_n, l_n)
    pred_norm = (
        pred_norm * float(norms.get("y_cal_scale", 1.0))
        + float(norms.get("y_cal_bias", 0.0))
    )
    return (
        pred_norm.cpu().numpy().ravel() * norms["y_std"] + norms["y_mean"]
    ).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════
#  Contrast matrices
# ═══════════════════════════════════════════════════════════════════


def hat_contrast_matrix(
    model: nn.Module,
    a_grid: np.ndarray,
    a_ref: float | np.ndarray,
    w: Tensor,
    l: Tensor,
    device: torch.device,
    norms: dict,
) -> np.ndarray:
    """Per-unit fitted contrasts ``τ̂[i, g] = ĥ(a_g, W_i, L_i) − ĥ(a_ref, W_i, L_i)``.

    ``a_ref`` may be a scalar (dA=1) or a 1-D array of shape ``(dA,)`` for
    vector continuous treatment.  ``a_grid`` rows must be consistent (scalars
    or ``(dA,)`` arrays).  Shape ``(n_units, len(a_grid))``.
    """
    n = w.shape[0]
    g = len(a_grid)
    hat = np.zeros((n, g), dtype=np.float64)
    ref = predict_hat_a(model, a_ref, w, l, device, norms)
    for j, av in enumerate(a_grid):
        hat[:, j] = predict_hat_a(model, av, w, l, device, norms) - ref
    return hat


def true_contrast_matrix(
    a_grid: np.ndarray,
    a_ref: float | np.ndarray,
    l_all: np.ndarray,
    h0_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Per-unit truth contrasts ``τ[i, g] = h₀(a_g, L_i) − h₀(a_ref, L_i)``.

    ``h0_fn(a_arr, l_rows)`` must accept an ``(n,)`` or ``(n, dA)`` treatment
    array and the ``(n, l_dim)`` covariate block, returning the length-``n``
    structural mean.  ``a_ref`` / each row of ``a_grid`` may be scalar or a
    ``(dA,)`` vector for vector continuous treatment.
    Shape ``(n_units, len(a_grid))``.
    """
    l_all = np.asarray(l_all, dtype=np.float64)
    n = l_all.shape[0]
    g = len(a_grid)
    out = np.zeros((n, g), dtype=np.float64)
    h_ref = np.asarray(
        h0_fn(_make_a_input(a_ref, n), l_all), dtype=np.float64,
    ).ravel()
    for j, av in enumerate(a_grid):
        h_a = np.asarray(
            h0_fn(_make_a_input(av, n), l_all), dtype=np.float64,
        ).ravel()
        out[:, j] = h_a - h_ref
    return out


def contrast_mse(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Mean squared error over units × grid points."""
    return float(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2))


def contrast_scale_denom(
    tau_true: np.ndarray,
    *,
    scale: str = "var",
    eps: float = 1e-12,
) -> float:
    """Return denominator used to normalize contrast MSE.

    Supported ``scale`` values:
    - ``"raw"``: no normalization (denominator 1.0)
    - ``"var"``: variance of ``tau_true``
    - ``"mean_sq"``: mean squared magnitude ``E[tau_true^2]``
    - ``"range_sq"``: squared range ``(max-min)^2``
    """
    tau = np.asarray(tau_true, dtype=np.float64).ravel()
    scale_key = str(scale).lower()
    if scale_key == "raw":
        return 1.0
    if scale_key == "var":
        denom = float(np.var(tau))
    elif scale_key == "mean_sq":
        denom = float(np.mean(tau ** 2))
    elif scale_key == "range_sq":
        denom = float((np.max(tau) - np.min(tau)) ** 2)
    else:
        raise ValueError(f"Unknown contrast MSE normalization scale: {scale!r}")
    return max(denom, float(eps))


def contrast_metric_label(scale: str = "var") -> str:
    """Human-readable label for the configured contrast error metric."""
    scale_key = str(scale).lower()
    if scale_key == "raw":
        return "MSE(raw)"
    return f"nMSE({scale_key})"


def contrast_error(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    *,
    scale: str = "var",
    eps: float = 1e-12,
) -> float:
    """Contrast prediction error on either raw or normalized scale."""
    raw = contrast_mse(tau_hat, tau_true)
    denom = contrast_scale_denom(tau_true, scale=scale, eps=eps)
    return float(raw / denom)


# ═══════════════════════════════════════════════════════════════════
#  Contrast-surface evaluation (τ on an (A, L_vary) mesh, other L + W marginalised)
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def hat_contrast_surface(
    model: nn.Module,
    a_grid: np.ndarray,
    a_ref: float,
    l_grid: np.ndarray,
    vary_idx: int,
    l_units: Tensor,
    w: Tensor,
    device: torch.device,
    norms: dict,
) -> np.ndarray:
    r"""Fitted contrast surface ``τ̂(a, L_vary)`` on a ``(len(l_grid), len(a_grid))`` mesh.

    For each mesh point ``(a_grid[g], l_grid[i])``:

    .. math::
        \hat\tau[i, g] \;=\; \operatorname*{mean}_{j} \Big[
            \hat h\!\big(a_g,\ W_j,\ L_j^{(i)}\big)
            \;-\; \hat h\!\big(a_{\text{ref}},\ W_j,\ L_j^{(i)}\big)\Big],

    where ``L_j^{(i)}`` is unit ``j``'s covariate row with column ``vary_idx``
    replaced by ``l_grid[i]``.  The per-unit pairs ``(W_j, L_{j,\setminus vary})``
    supply empirical Monte-Carlo samples of the nuisance marginalisation; the
    reference ``ĥ(a_ref, …)`` uses the same samples within each row so the MC
    variance of the difference is small.  Output on the original ``Y`` scale;
    shape ``(len(l_grid), len(a_grid))``.
    """
    l_grid = np.asarray(l_grid, dtype=np.float64)
    n_l = l_grid.shape[0]
    n_a = len(a_grid)
    out = np.zeros((n_l, n_a), dtype=np.float64)
    l_base = l_units.detach().to(torch.float32).cpu()
    for i in range(n_l):
        l_i = l_base.clone()
        l_i[:, int(vary_idx)] = float(l_grid[i])
        h_ref = predict_hat_a(model, float(a_ref), w, l_i, device, norms)
        for g in range(n_a):
            h_a = predict_hat_a(model, float(a_grid[g]), w, l_i, device, norms)
            out[i, g] = float(np.mean(h_a - h_ref))
    return out


def true_contrast_surface(
    a_grid: np.ndarray,
    a_ref: float,
    l_grid: np.ndarray,
    vary_idx: int,
    l_units: np.ndarray,
    h0_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    r"""Ground-truth contrast surface ``τ(a, L_vary)`` on the same mesh as
    :func:`hat_contrast_surface`.

    ``l_units`` supplies empirical Monte-Carlo samples for the marginalised
    (non-varying) covariate columns; ``h0_fn(a_vec, l_rows)`` must return the
    length-``n`` structural mean (same signature as in
    :func:`true_contrast_matrix`).
    """
    l_grid = np.asarray(l_grid, dtype=np.float64)
    l_units = np.asarray(l_units, dtype=np.float64)
    n_l = l_grid.shape[0]
    n_a = len(a_grid)
    n = l_units.shape[0]
    out = np.zeros((n_l, n_a), dtype=np.float64)
    for i in range(n_l):
        l_i = l_units.copy()
        l_i[:, int(vary_idx)] = float(l_grid[i])
        h_ref = np.asarray(
            h0_fn(np.full(n, float(a_ref), dtype=np.float64), l_i), dtype=np.float64,
        ).ravel()
        for g in range(n_a):
            h_a = np.asarray(
                h0_fn(np.full(n, float(a_grid[g]), dtype=np.float64), l_i),
                dtype=np.float64,
            ).ravel()
            out[i, g] = float(np.mean(h_a - h_ref))
    return out


# ═══════════════════════════════════════════════════════════════════
#  Representation probes  (Z or (W, Z) → V)
# ═══════════════════════════════════════════════════════════════════


def r2_score(pred: Tensor, target: Tensor) -> float:
    """Coefficient of determination between ``pred`` and ``target`` tensors."""
    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def train_probe_r2(
    input_train: Tensor,
    target_train: Tensor,
    input_test: Tensor,
    target_test: Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dims: Sequence[int] = (256, 256),
    batch_size: int = 256,
    device: torch.device | None = None,
    val_frac: float = 0.15,
    patience: int = 20,
    weight_decay: float = 1e-5,
    min_delta: float = 1e-6,
) -> tuple[float, Tensor]:
    """Fit a small MLP probe on train data and return ``(test_r2, test_pred)``.

    Works for any ``(input, target)`` pair — e.g. ``(W, Z) → V`` or ``Z → V``.
    """
    if device is None:
        device = torch.device("cpu")

    # Some learned reps (notably AutoIV) can vary wildly in scale across seeds.
    # Z-scoring the probe inputs on the train split keeps the probe comparison
    # focused on recoverable information rather than optimizer sensitivity.
    input_train = input_train.to(torch.float32)
    input_test = input_test.to(torch.float32)
    target_train = target_train.to(torch.float32)
    target_test = target_test.to(torch.float32)

    in_mean = input_train.mean(dim=0, keepdim=True)
    in_std = input_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    input_train = (input_train - in_mean) / in_std
    input_test = (input_test - in_mean) / in_std

    in_dim = input_train.shape[1]
    out_dim = target_train.shape[1]

    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    probe = nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = int(input_train.shape[0])
    if n_train < 2:
        raise ValueError("train_probe_r2 requires at least 2 training rows.")
    n_val = max(1, int(round(val_frac * n_train)))
    if n_val >= n_train:
        n_val = max(1, n_train // 5)
    n_subtrain = n_train - n_val
    if n_subtrain < 1:
        n_subtrain = n_train - 1
        n_val = 1

    perm = torch.randperm(n_train)
    tr_idx = perm[:n_subtrain]
    va_idx = perm[n_subtrain:]
    inp_subtrain = input_train[tr_idx]
    tgt_subtrain = target_train[tr_idx]
    inp_val = input_train[va_idx].to(device)
    tgt_val = target_train[va_idx].to(device)

    loader = DataLoader(
        TensorDataset(inp_subtrain, tgt_subtrain),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    bad_epochs = 0

    for _ in range(epochs):
        probe.train()
        for inp_b, tgt_b in loader:
            inp_b, tgt_b = inp_b.to(device), tgt_b.to(device)
            loss = F.mse_loss(probe(inp_b), tgt_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(probe(inp_val), tgt_val).item()
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        pred = probe(input_test.to(device)).cpu()
    return r2_score(pred, target_test), pred


def run_probe(
    c_tr: Tensor, z_tr: Tensor, v_tr: Tensor,
    c_te: Tensor, z_te: Tensor, v_te: Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> Tensor:
    """Concatenate ``(C, Z)``, train an MLP probe → V, return test predictions."""
    cz_tr = torch.cat([c_tr, z_tr], dim=-1)
    cz_te = torch.cat([c_te, z_te], dim=-1)
    _, v_recon = train_probe_r2(
        cz_tr, v_tr, cz_te, v_te,
        epochs=epochs, lr=lr, device=device,
    )
    return v_recon


def run_probe_z(
    z_tr: Tensor, v_tr: Tensor,
    z_te: Tensor, v_te: Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> Tensor:
    """Train an MLP probe ``Z → V`` and return test predictions."""
    _, v_recon = train_probe_r2(
        z_tr, v_tr, z_te, v_te,
        epochs=epochs, lr=lr, device=device,
    )
    return v_recon


def build_counterfactual_eval(
    a_pop: np.ndarray,
    l_pop: np.ndarray,
    h0_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    treatment_type: str = "continuous",
    a_ref_override: float | None = None,
    cf_a_pctl_low: float = 5.0,
    cf_a_pctl_high: float = 95.0,
    cf_grid_n: int = 25,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Build the shared counterfactual-evaluation setup ``(a₀, a_grid, τ̄_true)``.

    This utility is dataset-agnostic: callers supply a **large** population sample
    ``(a_pop, l_pop)`` from the DGP and a structural-response function
    :math:`h_0(a, L)`. Every replicate seed in a benchmark should share the same
    reference ``a₀`` and intervention grid ``a_grid`` so unit-level MSEs and
    population curves are comparable across replicates and methods.

    Rules used:

    - ``a_ref = a_ref_override`` if provided, else ``mean(a_pop)``.
    - Scalar treatment (``dA=1``): ``a_grid = linspace(p_lo, p_hi, cf_grid_n)``
      with ``p_lo`` / ``p_hi`` from the corresponding percentiles of ``a_pop``.
    - Vector continuous treatment (``dA>1``): construct a full Cartesian grid
      using per-dimension percentile bounds; points per dimension are chosen so
      total grid size is roughly ``cf_grid_n``.
      Falls back to per-dimension ``[min, max]`` if percentile bounds degenerate.
    - ``τ̄_true(a) = mean_i [ h_0(a, L_i) − h_0(a_ref, L_i) ]`` over ``l_pop``.

    Parameters
    ----------
    a_pop:
        1-D array of treatment values ``A`` from a large one-off population draw.
    l_pop:
        Matrix of covariates ``L`` from the same draw (``(n_pop, l_dim)``). Used
        to compute the ground-truth population contrast curve.
    h0_fn:
        Structural-response callable ``h0_fn(a, L) -> (n,)`` consumed by
        :func:`true_contrast_matrix`.
    a_ref_override:
        If not ``None``, pins ``a_ref`` to this exact value (tagged in the
        returned source string). Use for reproducibility / paper figures.
    cf_a_pctl_low, cf_a_pctl_high:
        Lower / upper percentiles of ``a_pop`` for the grid endpoints (default
        5th / 95th).
    cf_grid_n:
        Grid-size control. For scalar treatment, this is the exact number of
        points. For vector continuous treatment, the evaluator chooses roughly
        this many total points via a Cartesian grid.

    Returns
    -------
    ``(a_ref, a_grid, tau_pop_true)``
        * ``a_ref`` — float (scalar A) or ``(dA,)`` array (vector A) reference treatment.
        * ``a_grid`` — ``(cf_grid_n,)`` float64 array for scalar A, or
          ``(n_grid, dA)`` array for vector A (Cartesian full grid).
        * ``tau_pop_true`` — ``(cf_grid_n,)`` population mean contrast curve.
    """
    a_pop_arr = np.asarray(a_pop, dtype=np.float64)
    if a_pop_arr.ndim == 1:
        a_pop_arr = a_pop_arr.reshape(-1, 1)
    if a_pop_arr.shape[0] == 0:
        raise ValueError("build_counterfactual_eval: a_pop is empty.")
    dA = a_pop_arr.shape[1]

    if str(treatment_type) == "binary":
        if dA == 1:
            a_ref: float | np.ndarray = float(a_ref_override) if a_ref_override is not None else 0.0
            a_grid = np.asarray([0.0, 1.0], dtype=np.float64)
        else:
            import itertools as _itertools
            a_ref = np.zeros(dA, dtype=np.float64)
            a_grid = np.array(
                list(_itertools.product([0.0, 1.0], repeat=dA)), dtype=np.float64,
            )  # (2^dA, dA)
    elif dA > 1:
        # vector continuous treatment: evaluate on a full Cartesian grid over all
        # treatment dimensions (no PCA projection / slicing).
        a_ref_vec = a_pop_arr.mean(axis=0)  # (dA,)
        lo_q = np.percentile(a_pop_arr, cf_a_pctl_low, axis=0).astype(np.float64)   # (dA,)
        hi_q = np.percentile(a_pop_arr, cf_a_pctl_high, axis=0).astype(np.float64)   # (dA,)
        valid = lo_q < hi_q
        lo = np.where(valid, lo_q, np.min(a_pop_arr, axis=0))
        hi = np.where(valid, hi_q, np.max(a_pop_arr, axis=0))

        # Keep total grid size roughly near cf_grid_n while covering all dims.
        points_per_dim = max(2, int(round(float(cf_grid_n) ** (1.0 / float(dA)))))
        axes = [
            np.linspace(float(lo[j]), float(hi[j]), points_per_dim, dtype=np.float64)
            for j in range(dA)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        a_grid = np.stack([m.reshape(-1) for m in mesh], axis=1)  # (points_per_dim**dA, dA)
        a_ref = a_ref_vec  # (dA,)
    else:
        # scalar continuous treatment
        a_pop_1d = a_pop_arr.reshape(-1)
        a_ref = (
            float(a_ref_override)
            if a_ref_override is not None
            else float(a_pop_1d.mean())
        )
        lo = float(np.percentile(a_pop_1d, cf_a_pctl_low))
        hi = float(np.percentile(a_pop_1d, cf_a_pctl_high))
        if lo >= hi:
            lo, hi = float(a_pop_1d.min()), float(a_pop_1d.max())
        a_grid = np.linspace(lo, hi, int(cf_grid_n), dtype=np.float64)

    tau_pop_true = true_contrast_matrix(
        a_grid, a_ref, np.asarray(l_pop, dtype=np.float64), h0_fn,
    ).mean(axis=0)
    return a_ref, a_grid, tau_pop_true


def probe_wz_to_target_r2(
    w_tr: Tensor,
    z_tr: Tensor,
    target_tr: Tensor,
    w_te: Tensor,
    z_te: Tensor,
    target_te: Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dims: Sequence[int] = (256, 256),
    batch_size: int = 256,
    device: torch.device | None = None,
) -> float:
    """MLP probe ``(W, Z) → target``; returns test :math:`R^2`.

    Concatenates ``W`` and ``Z`` along the last dim before fitting a small
    supervised MLP on the training split. The target is typically a ground-truth
    latent (e.g. ``V`` or ``D`` in the demands benchmark) used to quantify how
    much structural information each method's ``(W, Z)`` representation retains.
    """
    wz_tr = torch.cat([w_tr, z_tr], dim=-1)
    wz_te = torch.cat([w_te, z_te], dim=-1)
    r2, _ = train_probe_r2(
        wz_tr, target_tr, wz_te, target_te,
        epochs=epochs,
        lr=lr,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        device=device,
    )
    return float(r2)


def probe_w_to_target_r2(
    w_tr: Tensor,
    target_tr: Tensor,
    w_te: Tensor,
    target_te: Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dims: Sequence[int] = (256, 256),
    batch_size: int = 256,
    device: torch.device | None = None,
) -> float:
    """MLP probe ``W → target``; returns test :math:`R^2`."""
    r2, _ = train_probe_r2(
        w_tr, target_tr, w_te, target_te,
        epochs=epochs,
        lr=lr,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        device=device,
    )
    return float(r2)


def probe_z_to_target_r2(
    z_tr: Tensor,
    target_tr: Tensor,
    z_te: Tensor,
    target_te: Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dims: Sequence[int] = (256, 256),
    batch_size: int = 256,
    device: torch.device | None = None,
) -> float:
    """MLP probe ``Z → target``; returns test :math:`R^2` (no ``W`` in the input)."""
    r2, _ = train_probe_r2(
        z_tr, target_tr, z_te, target_te,
        epochs=epochs,
        lr=lr,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        device=device,
    )
    return float(r2)


__all__ = [
    "build_counterfactual_eval",
    "contrast_error",
    "contrast_metric_label",
    "contrast_mse",
    "contrast_scale_denom",
    "hat_contrast_matrix",
    "hat_contrast_surface",
    "predict_hat_a",
    "probe_w_to_target_r2",
    "probe_wz_to_target_r2",
    "probe_z_to_target_r2",
    "r2_score",
    "run_probe",
    "run_probe_z",
    "train_probe_r2",
    "true_contrast_matrix",
    "true_contrast_surface",
]
