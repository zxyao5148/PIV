"""
Synthetic NPIV benchmark simulator for scalar/vector continuous treatment and
scalar/vector binary treatment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
from numpy.random import Generator, SeedSequence


TreatmentType = Literal["continuous", "binary"]
ContinuousSurfaceType = Literal[
    "poly",
    "varying_slope",
    "soft_threshold",
    "sinusoidal",
]
BinarySurfaceType = Literal["binary_linear", "binary_threshold_modulated"]
SurfaceType = Literal[
    "poly",
    "varying_slope",
    "soft_threshold",
    "sinusoidal",
    "binary_linear",
    "binary_threshold_modulated",
]

CONTINUOUS_SURFACES = [
    "poly",
    "varying_slope",
    "soft_threshold",
    "sinusoidal",
]
BINARY_SURFACES = ["binary_linear", "binary_threshold_modulated"]
ALL_SURFACES = CONTINUOUS_SURFACES + BINARY_SURFACES

SURFACE_CODE: Dict[str, int] = {
    "poly": 0,
    # Keep the previous code for varying_slope to preserve design-seed behavior.
    "varying_slope": 3,
    "binary_linear": 4,
    "binary_threshold_modulated": 5,
    "soft_threshold": 7,
    "sinusoidal": 9,
}


@dataclass
class SimConfig:
    """Configuration for the synthetic NPIV DGP."""

    n: int = 5000
    dC: int = 3
    dV: int = 3
    dD: int = 2
    dL: Optional[int] = None
    dM: Optional[int] = None
    dA: int = 1

    sigma_m: float = 0.50
    sigma_a: float = 1.00
    sigma_y: float = 1.00

    max_treat_tries: int = 2000
    min_sbeta_singular_value: float = 0.35
    min_beta_norm_anchor: float = 0.35

    min_B_singular_value: float = 0.35
    min_B_row_norm: float = 0.35

    max_proxy_tries: int = 2000
    min_lambda_singular_value: float = 0.35

    binary_pilot_n: int = 2048
    binary_min_signal_ratio: float = 0.50
    binary_min_global_treated_rate: float = 0.10
    binary_max_global_treated_rate: float = 0.90
    binary_min_anchor_treated_rate: float = 0.15
    binary_max_anchor_treated_rate: float = 0.85
    binary_min_distinct_patterns: int = 4
    binary_max_pattern_share: float = 0.85

    # Vector-binary joint-cell positivity / overlap checks.
    # These checks are active only for treatment_type="binary" and dA > 1.
    # They use oracle P(A cell | V,C,L), integrating out U and eps.
    binary_enable_joint_overlap_filter: bool = True
    binary_min_joint_cell_prob_p01: float = 0.0025
    binary_min_joint_cell_prob_p05: float = 0.0100
    binary_max_joint_cell_prob_p95: float = 0.9900

    alpha0_range: tuple[float, float] = (-0.5, 0.5)
    alpha_range: tuple[float, float] = (-1.0, 1.0)
    gamma_u_range: tuple[float, float] = (1.0, 2.0)
    # Simulator invariance knob: keep confounder contribution scale stable as dC changes.
    # When enabled, each row of alphaC is rescaled to the expected L2 norm for
    # ``alphaC_scale_ref_dim`` under ``alpha_range``.
    alphaC_enforce_invariant_scale: bool = True
    alphaC_scale_ref_dim: int = 2

    # Minimum marginal confounding correlation rho = gamma_u / sqrt(IV_var + gamma_u² + sigma_a²)
    # for continuous treatment.  rho < threshold → treatment theta rejected.
    # Set to 0.0 to disable.  Default 0.30 ensures U explains ≥ 9% of Var(A|C,L).
    min_continuous_confound_ratio: float = 0.30

    delta_range: tuple[float, float] = (-1.0, 1.0)
    delta_u_range: tuple[float, float] = (0.5, 1.5)

    mu0_range: tuple[float, float] = (-1.0, 1.0)
    muL_range: tuple[float, float] = (-1.0, 1.0)

    poly_beta_range: tuple[float, float] = (-1.0, 1.0)
    poly_quad_range: tuple[float, float] = (0.2, 1.0)

    vslope_tau0_range: tuple[float, float] = (0.5, 2.0)
    vslope_tauL_range: tuple[float, float] = (-1.0, 1.0)
    vslope_lambda_range: tuple[float, float] = (-1.0, 1.0)
    soft_threshold_tau0_range: tuple[float, float] = (0.8, 2.2)
    soft_threshold_tauL_range: tuple[float, float] = (-1.0, 1.0)
    soft_threshold_k_range: tuple[float, float] = (1.5, 4.0)
    soft_threshold_c0_range: tuple[float, float] = (-0.4, 0.4)
    soft_threshold_cL_range: tuple[float, float] = (-0.8, 0.8)
    sinus_tau0_range: tuple[float, float] = (0.6, 1.4)
    sinus_tauL_range: tuple[float, float] = (-0.5, 0.5)
    sinus_omega_range: tuple[float, float] = (0.6, 1.2)
    sinus_phase0_range: tuple[float, float] = (-0.5, 0.5)
    sinus_phaseL_range: tuple[float, float] = (-0.4, 0.4)
    sinus_lin0_range: tuple[float, float] = (0.8, 1.6)
    sinus_linL_range: tuple[float, float] = (-0.3, 0.3)

    binary_tau0_range: tuple[float, float] = (0.5, 2.0)
    binary_tauL_range: tuple[float, float] = (-1.0, 1.0)
    binary_mod_tau0_range: tuple[float, float] = (0.25, 1.5)
    binary_mod_tauL_range: tuple[float, float] = (-1.0, 1.0)
    binary_mod_gamma_range: tuple[float, float] = (-1.0, 1.0)

    proxy_loading_range: tuple[float, float] = (-1.0, 1.0)
    mix_linear_range: tuple[float, float] = (-0.25, 0.25)
    mix_quadratic_range: tuple[float, float] = (-0.06, 0.06)
    mix_cubic_range: tuple[float, float] = (-0.02, 0.02)

    def __post_init__(self) -> None:
        if self.dL is None:
            self.dL = self.dV
        if self.dM is None:
            self.dM = self.dC
        if self.dM < self.dC:
            raise ValueError("dM must be at least dC so that Lambda_C can have full column rank.")
        if self.dA < 1:
            raise ValueError("dA must be positive.")
        if self.max_treat_tries < 1:
            raise ValueError("max_treat_tries must be positive.")
        if self.max_proxy_tries < 1:
            raise ValueError("max_proxy_tries must be positive.")
        if self.binary_pilot_n < 1:
            raise ValueError("binary_pilot_n must be positive.")
        if not (0.0 <= self.binary_min_joint_cell_prob_p01 <= 1.0):
            raise ValueError("binary_min_joint_cell_prob_p01 must be in [0, 1].")
        if not (0.0 <= self.binary_min_joint_cell_prob_p05 <= 1.0):
            raise ValueError("binary_min_joint_cell_prob_p05 must be in [0, 1].")
        if not (0.0 <= self.binary_max_joint_cell_prob_p95 <= 1.0):
            raise ValueError("binary_max_joint_cell_prob_p95 must be in [0, 1].")
        if self.min_continuous_confound_ratio < 0.0:
            raise ValueError("min_continuous_confound_ratio must be nonnegative.")
        if self.alphaC_scale_ref_dim < 1:
            raise ValueError("alphaC_scale_ref_dim must be positive.")


@dataclass
class MixingCoeffs:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass
class TreatmentTheta:
    alpha0: np.ndarray
    alphaC: np.ndarray
    alphaL: np.ndarray
    gamma_u: np.ndarray
    accepted_attempt: int
    b0: Optional[np.ndarray] = None
    S_beta: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    binary_accept_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeTheta:
    surface: SurfaceType
    mu0: float
    muL: np.ndarray
    deltaC: np.ndarray
    deltaD: np.ndarray
    deltaU: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DGPDefinition:
    surface: SurfaceType
    treatment_type: TreatmentType
    design_seed: int
    proxy_loading: np.ndarray
    proxy_loading_attempt: int
    mixing: MixingCoeffs
    treatment: TreatmentTheta
    outcome: OutcomeTheta


@dataclass
class SimulatedDataset:
    X: np.ndarray
    M: np.ndarray
    L: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    C: np.ndarray
    V: np.ndarray
    D: np.ndarray
    U: np.ndarray
    structural_response_obs: np.ndarray
    confounder_nuisance_obs: np.ndarray
    structural_noise_obs: np.ndarray


def _uniform_vec(rng: Generator, shape: tuple[int, ...], low: float, high: float) -> np.ndarray:
    return rng.uniform(low, high, size=shape)


def _scalar_uniform(rng: Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _normal_cdf_np(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using only stdlib math + NumPy."""
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def _softplus_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, x)


def _binary_joint_cell_probs_from_marginals_np(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert independent Bernoulli marginal probabilities into joint cell probabilities.

    Parameters
    ----------
    p:
        Array [n, dA], where p[i, j] = P(A_j = 1 | context_i).

    Returns
    -------
    combos:
        Array [2**dA, dA] in itertools.product order: 00, 01, 10, 11 for dA=2.
    cell_probs:
        Array [n, 2**dA], conditional joint probabilities.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError("p must be a 2D array with shape [n, dA].")
    n, dA = p.shape
    combos = np.asarray(list(itertools.product([0.0, 1.0], repeat=dA)), dtype=float)
    cell_probs = np.ones((n, combos.shape[0]), dtype=float)
    for k, a_cell in enumerate(combos):
        prob_by_coord = np.where(a_cell[None, :] > 0.5, p, 1.0 - p)
        cell_probs[:, k] = np.prod(prob_by_coord, axis=1)
    return combos, cell_probs


def _summarize_joint_cell_overlap(cell_probs: np.ndarray, combos: np.ndarray) -> Dict[str, Any]:
    """Summarize conditional joint-cell probabilities."""
    cell_probs = np.asarray(cell_probs, dtype=float)
    combos = np.asarray(combos, dtype=float)
    flat = cell_probs.reshape(-1)

    per_cell_mean: Dict[str, float] = {}
    per_cell_p01: Dict[str, float] = {}
    per_cell_p05: Dict[str, float] = {}
    per_cell_min: Dict[str, float] = {}

    for k, a_cell in enumerate(combos):
        name = "".join(str(int(v)) for v in a_cell)
        vals = cell_probs[:, k]
        per_cell_mean[name] = float(np.mean(vals))
        per_cell_p01[name] = float(np.quantile(vals, 0.01))
        per_cell_p05[name] = float(np.quantile(vals, 0.05))
        per_cell_min[name] = float(np.min(vals))

    return {
        "joint_cell_prob_min": float(np.min(flat)),
        "joint_cell_prob_p01": float(np.quantile(flat, 0.01)),
        "joint_cell_prob_p05": float(np.quantile(flat, 0.05)),
        "joint_cell_prob_p10": float(np.quantile(flat, 0.10)),
        "joint_cell_prob_p50": float(np.quantile(flat, 0.50)),
        "joint_cell_prob_p95": float(np.quantile(flat, 0.95)),
        "joint_cell_prob_max": float(np.max(flat)),
        "joint_bad_mass_lt_001": float(np.mean(flat < 0.001)),
        "joint_bad_mass_lt_005": float(np.mean(flat < 0.005)),
        "joint_bad_mass_lt_010": float(np.mean(flat < 0.010)),
        "joint_cell_prob_mean_by_cell": per_cell_mean,
        "joint_cell_prob_p01_by_cell": per_cell_p01,
        "joint_cell_prob_p05_by_cell": per_cell_p05,
        "joint_cell_prob_min_by_cell": per_cell_min,
    }


def _anchor_L_values(dL: int, dV: int) -> np.ndarray:
    anchors = [np.zeros(dL, dtype=float)]
    for j in range(dV):
        ej = np.zeros(dL, dtype=float)
        ej[j] = 1.0
        anchors.append(ej)
    anchors.append(np.ones(dL, dtype=float))
    return np.stack(anchors, axis=0)


def _common_design_seedseq(design_seed: int) -> SeedSequence:
    return SeedSequence([int(design_seed)])


def _treatment_design_seedseq(design_seed: int, treatment_type: TreatmentType) -> SeedSequence:
    """Seed for treatment theta — surface-independent so A is shared across surfaces."""
    treatment_code = 0 if treatment_type == "continuous" else 1
    return SeedSequence([int(design_seed), treatment_code])


def _surface_design_seedseq(design_seed: int, surface: SurfaceType, treatment_type: TreatmentType) -> SeedSequence:
    treatment_code = 0 if treatment_type == "continuous" else 1
    return SeedSequence([int(design_seed), treatment_code, int(SURFACE_CODE[surface])])


def _rng_from_data_seed(data_seed: int) -> Generator:
    return np.random.default_rng(int(data_seed))


def _validate_surface_for_treatment(surface: str, treatment_type: TreatmentType) -> None:
    if treatment_type == "continuous" and surface not in CONTINUOUS_SURFACES:
        raise ValueError(f"Continuous treatment only supports surfaces {CONTINUOUS_SURFACES}, got {surface!r}.")
    if treatment_type == "binary" and surface not in BINARY_SURFACES:
        raise ValueError(f"Binary treatment only supports surfaces {BINARY_SURFACES}, got {surface!r}.")


def _validate_dims_for_treatment(cfg: SimConfig, treatment_type: TreatmentType) -> None:
    if treatment_type == "binary":
        if cfg.dA == 1:
            if cfg.dL < cfg.dV:
                raise ValueError("Scalar binary treatment requires dL >= dV for loading anchors.")
        else:
            if cfg.dA < cfg.dV:
                raise ValueError("Vector binary treatment requires dA >= dV so B can have full column rank.")
    else:
        if cfg.dA == 1:
            if cfg.dL < cfg.dV:
                raise ValueError("Scalar continuous treatment requires dL >= dV for loading anchors.")
        else:
            if cfg.dA < cfg.dV:
                raise ValueError("Vector continuous treatment requires dA >= dV so B can have full column rank.")


def propose_proxy_loading(rng: Generator, cfg: SimConfig) -> np.ndarray:
    return _uniform_vec(rng, (cfg.dM, cfg.dC), *cfg.proxy_loading_range)


def accept_proxy_loading(lam: np.ndarray, cfg: SimConfig) -> bool:
    svals = np.linalg.svd(lam, compute_uv=False)
    return float(svals[-1]) >= cfg.min_lambda_singular_value


def sample_proxy_loading(root_ss: SeedSequence, cfg: SimConfig) -> tuple[np.ndarray, int]:
    for attempt, ss in enumerate(root_ss.spawn(cfg.max_proxy_tries)):
        rng = np.random.default_rng(ss)
        lam = propose_proxy_loading(rng, cfg)
        if accept_proxy_loading(lam, cfg):
            return lam, attempt
    raise RuntimeError("Failed to find an acceptable proxy loading draw.")


def sample_mixing_coeffs(rng: Generator, cfg: SimConfig) -> MixingCoeffs:
    q = cfg.dC + cfg.dV + cfg.dD
    a = np.zeros((q, q), dtype=float)
    b = np.zeros((q, q), dtype=float)
    c = np.zeros((q, q), dtype=float)
    for j in range(1, q):
        for k in range(j):
            a[j, k] = _scalar_uniform(rng, *cfg.mix_linear_range)
            b[j, k] = _scalar_uniform(rng, *cfg.mix_quadratic_range)
            c[j, k] = _scalar_uniform(rng, *cfg.mix_cubic_range)
    return MixingCoeffs(a=a, b=b, c=c)


def mix_source(C: np.ndarray, V: np.ndarray, D: np.ndarray, mixing: MixingCoeffs) -> np.ndarray:
    S = np.concatenate([C, V, D], axis=1)
    n, q = S.shape
    X = np.zeros((n, q), dtype=float)
    X[:, 0] = S[:, 0]
    for j in range(1, q):
        xj = S[:, j].copy()
        for k in range(j):
            sk = S[:, k]
            xj += mixing.a[j, k] * sk + mixing.b[j, k] * (sk ** 2) + mixing.c[j, k] * (sk ** 3)
        X[:, j] = xj
    return X


def sample_proxy_measurements(C: np.ndarray, rng: Generator, dgp: DGPDefinition, cfg: SimConfig) -> np.ndarray:
    eps_m = rng.normal(loc=0.0, scale=cfg.sigma_m, size=(C.shape[0], cfg.dM))
    return C @ dgp.proxy_loading.T + eps_m


def _expected_uniform_second_moment(low: float, high: float) -> float:
    """Return E[X^2] for X ~ Uniform(low, high)."""
    lo = float(low)
    hi = float(high)
    return float((lo * lo + lo * hi + hi * hi) / 3.0)


def _rescale_rows_to_l2_norm(mat: np.ndarray, target_norm: float, eps: float = 1e-12) -> np.ndarray:
    """Rescale each row of `mat` to have approximately `target_norm` L2 norm."""
    arr = np.asarray(mat, dtype=float).copy()
    row_norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.maximum(row_norms, float(eps))
    arr *= float(target_norm) / safe
    return arr


def propose_treatment_theta(rng: Generator, cfg: SimConfig) -> TreatmentTheta:
    alphaC = _uniform_vec(rng, (cfg.dA if cfg.dA > 1 else 1, cfg.dC), *cfg.alpha_range)
    if bool(cfg.alphaC_enforce_invariant_scale):
        e_x2 = _expected_uniform_second_moment(float(cfg.alpha_range[0]), float(cfg.alpha_range[1]))
        target = math.sqrt(float(cfg.alphaC_scale_ref_dim) * max(e_x2, 0.0))
        alphaC = _rescale_rows_to_l2_norm(alphaC, target_norm=target)

    if cfg.dA > 1:
        return TreatmentTheta(
            alpha0=_uniform_vec(rng, (cfg.dA,), *cfg.alpha0_range),
            alphaC=alphaC,
            alphaL=_uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.alpha_range),
            gamma_u=_uniform_vec(rng, (cfg.dA,), *cfg.gamma_u_range),
            B=_uniform_vec(rng, (cfg.dA, cfg.dV), *cfg.alpha_range),
            accepted_attempt=-1,
        )
    return TreatmentTheta(
        alpha0=np.asarray([_scalar_uniform(rng, *cfg.alpha0_range)], dtype=float),
        alphaC=alphaC,
        alphaL=_uniform_vec(rng, (1, cfg.dL), *cfg.alpha_range),
        gamma_u=np.asarray([_scalar_uniform(rng, *cfg.gamma_u_range)], dtype=float),
        b0=_uniform_vec(rng, (cfg.dV,), *cfg.alpha_range),
        S_beta=_uniform_vec(rng, (cfg.dV, cfg.dL), *cfg.alpha_range),
        accepted_attempt=-1,
    )


def treatment_loading(L: np.ndarray, treatment_theta: TreatmentTheta) -> np.ndarray:
    if treatment_theta.b0 is None or treatment_theta.S_beta is None:
        raise ValueError("Scalar-treatment loading requested but this treatment theta uses matrix B.")
    return treatment_theta.b0[None, :] + L @ treatment_theta.S_beta.T


def _binary_scalar_signal_ratio_and_positivity_stats(
    treatment_theta: TreatmentTheta,
    cfg: SimConfig,
    pilot_rng: Generator,
) -> Dict[str, Any]:
    n = int(cfg.binary_pilot_n)
    C = pilot_rng.normal(size=(n, cfg.dC))
    V = pilot_rng.normal(size=(n, cfg.dV))
    U = pilot_rng.normal(size=(n, 1))
    eps = pilot_rng.normal(size=(n, 1))
    L = pilot_rng.binomial(n=1, p=0.5, size=(n, cfg.dL)).astype(float)

    betaL = treatment_loading(L, treatment_theta)
    iv_signal = np.sum(betaL * V, axis=1)
    nuis_signal = (
        C @ treatment_theta.alphaC[0][:, None]
        + L @ treatment_theta.alphaL[0][:, None]
        + treatment_theta.gamma_u[0] * U
        + eps
    ).reshape(-1)

    iv_sd = float(np.std(iv_signal))
    nuis_sd = float(np.std(nuis_signal))
    signal_ratio = iv_sd / max(nuis_sd, 1e-8)
    eta = treatment_theta.alpha0[0] + iv_signal + nuis_signal
    treated_rate_global = float(np.mean(eta >= 0.0))

    anchor_rates: Dict[str, float] = {}
    anchors = _anchor_L_values(cfg.dL, cfg.dV)
    for idx, l in enumerate(anchors):
        l_arr = np.repeat(l[None, :], n, axis=0)
        beta_l = treatment_loading(l_arr, treatment_theta)
        iv_l = np.sum(beta_l * V, axis=1)
        nuis_l = (
            C @ treatment_theta.alphaC[0][:, None]
            + l_arr @ treatment_theta.alphaL[0][:, None]
            + treatment_theta.gamma_u[0] * U
            + eps
        ).reshape(-1)
        eta_l = treatment_theta.alpha0[0] + iv_l + nuis_l
        anchor_rates[f"anchor_{idx}"] = float(np.mean(eta_l >= 0.0))

    return {
        "signal_ratio": signal_ratio,
        "iv_signal_sd": iv_sd,
        "nuisance_signal_sd": nuis_sd,
        "treated_rate_global": treated_rate_global,
        "anchor_treated_rates": anchor_rates,
    }


def _binary_vector_signal_ratio_and_positivity_stats(
    treatment_theta: TreatmentTheta,
    cfg: SimConfig,
    pilot_rng: Generator,
) -> Dict[str, Any]:
    n = int(cfg.binary_pilot_n)
    C = pilot_rng.normal(size=(n, cfg.dC))
    V = pilot_rng.normal(size=(n, cfg.dV))
    U = pilot_rng.normal(size=(n, cfg.dA))
    eps = pilot_rng.normal(size=(n, cfg.dA))
    L = pilot_rng.binomial(n=1, p=0.5, size=(n, cfg.dL)).astype(float)

    if treatment_theta.B is None:
        raise ValueError("Vector binary pilot stats require B.")

    iv_signal = V @ treatment_theta.B.T
    nuis_signal = C @ treatment_theta.alphaC.T + L @ treatment_theta.alphaL.T + U * treatment_theta.gamma_u[None, :] + eps
    eta = treatment_theta.alpha0[None, :] + iv_signal + nuis_signal
    A = (eta >= 0.0).astype(float)

    iv_sd_by_coord = np.std(iv_signal, axis=0)
    nuis_sd_by_coord = np.std(nuis_signal, axis=0)
    signal_ratio_by_coord = iv_sd_by_coord / np.maximum(nuis_sd_by_coord, 1e-8)
    treated_rates_by_coord = A.mean(axis=0)

    codes = (A.astype(np.int64) * (1 << np.arange(cfg.dA, dtype=np.int64))[None, :]).sum(axis=1)
    uniq, counts = np.unique(codes, return_counts=True)
    distinct_patterns = int(uniq.size)
    max_pattern_share = float(np.max(counts) / max(n, 1))

    # Joint-cell positivity / overlap diagnostic.
    # Observable first-stage probability P(A_j=1 | V,C,L), integrating out U_j and eps_j.
    observable_index = (
        treatment_theta.alpha0[None, :]
        + C @ treatment_theta.alphaC.T
        + L @ treatment_theta.alphaL.T
        + V @ treatment_theta.B.T
    )
    observable_noise_sd = np.sqrt(1.0 + treatment_theta.gamma_u[None, :] ** 2)
    p_obs = _normal_cdf_np(observable_index / observable_noise_sd)
    combos, joint_cell_probs = _binary_joint_cell_probs_from_marginals_np(p_obs)
    joint_overlap_stats = _summarize_joint_cell_overlap(joint_cell_probs, combos)

    out = {
        "signal_ratio_by_coord": signal_ratio_by_coord.astype(float).tolist(),
        "iv_signal_sd_by_coord": iv_sd_by_coord.astype(float).tolist(),
        "nuisance_signal_sd_by_coord": nuis_sd_by_coord.astype(float).tolist(),
        "treated_rates_by_coord": treated_rates_by_coord.astype(float).tolist(),
        "min_signal_ratio": float(np.min(signal_ratio_by_coord)),
        "distinct_patterns": distinct_patterns,
        "max_pattern_share": max_pattern_share,
    }
    out.update(joint_overlap_stats)
    return out


def _continuous_confound_ratio(
    treatment_theta: TreatmentTheta,
    cfg: SimConfig,
    pilot_rng: Generator,
) -> float:
    """Marginal confounding correlation rho = gamma_u / sqrt(E[IV_var(L)] + gamma_u² + sigma_a²).

    This is the correlation of U with A given (C, L) but NOT V — the endogeneity
    that a noise-instrument (NoneIV) estimator cannot remove.  When the IV signal
    E[||beta(L)||²] is large, rho shrinks even if gamma_u is moderate, which is
    why the confounding filter is needed.

    For vector treatment (dA > 1) each coordinate is checked and the minimum rho
    is returned (no pilot needed: IV variance ||B[j,:]||² is analytic for V~N(0,I)).
    For scalar treatment a pilot draw of L is used to estimate E[||beta(L)||²].
    """
    sigma_a2 = cfg.sigma_a ** 2
    if cfg.dA > 1:
        if treatment_theta.B is None:
            return 0.0
        iv_vars = np.sum(treatment_theta.B ** 2, axis=1)  # ||B[j,:]||² per treatment dim
        gamma_u = treatment_theta.gamma_u  # shape (dA,)
        rhos = gamma_u / np.sqrt(iv_vars + gamma_u ** 2 + sigma_a2)
        return float(np.min(rhos))
    else:
        n = cfg.binary_pilot_n
        L_pilot = pilot_rng.binomial(n=1, p=0.5, size=(n, cfg.dL)).astype(float)
        betaL = treatment_loading(L_pilot, treatment_theta)  # (n, dV)
        mean_iv_var = float(np.mean(np.sum(betaL ** 2, axis=1)))  # E_L[||beta(L)||²]
        gamma_u_val = float(treatment_theta.gamma_u[0])
        return gamma_u_val / float(np.sqrt(mean_iv_var + gamma_u_val ** 2 + sigma_a2))


def accept_treatment_theta(
    treatment_theta: TreatmentTheta,
    cfg: SimConfig,
    treatment_type: TreatmentType,
    pilot_rng: Optional[Generator] = None,
) -> bool:
    if cfg.dA > 1:
        if treatment_theta.B is None:
            return False
        svals = np.linalg.svd(treatment_theta.B, compute_uv=False)
        if float(svals[-1]) < cfg.min_B_singular_value:
            return False
        row_norms = np.linalg.norm(treatment_theta.B, axis=1)
        if float(row_norms.min()) < cfg.min_B_row_norm:
            return False
        if treatment_type == "continuous":
            if pilot_rng is not None and cfg.min_continuous_confound_ratio > 0.0:
                if _continuous_confound_ratio(treatment_theta, cfg, pilot_rng) < cfg.min_continuous_confound_ratio:
                    return False
            treatment_theta.binary_accept_stats = {}
            return True
        if pilot_rng is None:
            raise ValueError("pilot_rng is required for binary-treatment acceptance checks.")
        stats = _binary_vector_signal_ratio_and_positivity_stats(treatment_theta, cfg, pilot_rng)
        if float(stats["min_signal_ratio"]) < cfg.binary_min_signal_ratio:
            return False
        for rate in stats["treated_rates_by_coord"]:
            if not (cfg.binary_min_global_treated_rate <= float(rate) <= cfg.binary_max_global_treated_rate):
                return False
        if int(stats["distinct_patterns"]) < min(cfg.binary_min_distinct_patterns, 1 << cfg.dA):
            return False
        if float(stats["max_pattern_share"]) > cfg.binary_max_pattern_share:
            return False

        if cfg.binary_enable_joint_overlap_filter:
            if float(stats["joint_cell_prob_p01"]) < cfg.binary_min_joint_cell_prob_p01:
                return False
            if float(stats["joint_cell_prob_p05"]) < cfg.binary_min_joint_cell_prob_p05:
                return False
            if float(stats["joint_cell_prob_p95"]) > cfg.binary_max_joint_cell_prob_p95:
                return False

        treatment_theta.binary_accept_stats = stats
        return True

    if treatment_theta.S_beta is None:
        return False
    svals = np.linalg.svd(treatment_theta.S_beta, compute_uv=False)
    if float(svals[-1]) < cfg.min_sbeta_singular_value:
        return False
    L_anchor = _anchor_L_values(cfg.dL, cfg.dV)
    beta_anchor = treatment_loading(L_anchor, treatment_theta)
    beta_norms = np.linalg.norm(beta_anchor, axis=1)
    if float(beta_norms.min()) < cfg.min_beta_norm_anchor:
        return False
    if treatment_type == "continuous":
        if pilot_rng is not None and cfg.min_continuous_confound_ratio > 0.0:
            if _continuous_confound_ratio(treatment_theta, cfg, pilot_rng) < cfg.min_continuous_confound_ratio:
                return False
        treatment_theta.binary_accept_stats = {}
        return True
    if pilot_rng is None:
        raise ValueError("pilot_rng is required for binary-treatment acceptance checks.")
    stats = _binary_scalar_signal_ratio_and_positivity_stats(treatment_theta, cfg, pilot_rng)
    if float(stats["signal_ratio"]) < cfg.binary_min_signal_ratio:
        return False
    global_rate = float(stats["treated_rate_global"])
    if not (cfg.binary_min_global_treated_rate <= global_rate <= cfg.binary_max_global_treated_rate):
        return False
    for rate in stats["anchor_treated_rates"].values():
        if not (cfg.binary_min_anchor_treated_rate <= float(rate) <= cfg.binary_max_anchor_treated_rate):
            return False
    treatment_theta.binary_accept_stats = stats
    return True


def sample_treatment_theta_with_accept_reject(root_ss: SeedSequence, cfg: SimConfig, treatment_type: TreatmentType) -> TreatmentTheta:
    for attempt, ss in enumerate(root_ss.spawn(cfg.max_treat_tries)):
        proposal_ss, pilot_ss = ss.spawn(2)
        rng = np.random.default_rng(proposal_ss)
        theta = propose_treatment_theta(rng, cfg)
        pilot_rng = np.random.default_rng(pilot_ss)
        if accept_treatment_theta(theta, cfg, treatment_type, pilot_rng=pilot_rng):
            theta.accepted_attempt = attempt
            return theta
    raise RuntimeError(
        "Failed to find an acceptable treatment coefficient draw. "
        "If this happens after enabling joint overlap filtering, relax "
        "binary_min_joint_cell_prob_p01 / binary_min_joint_cell_prob_p05, "
        "increase max_treat_tries, or disable the filter for stress tests."
    )


def sample_outcome_theta(rng: Generator, surface: SurfaceType, treatment_type: TreatmentType, cfg: SimConfig) -> OutcomeTheta:
    mu0 = _scalar_uniform(rng, *cfg.mu0_range)
    muL = _uniform_vec(rng, (cfg.dL,), *cfg.muL_range)
    deltaC = _uniform_vec(rng, (cfg.dC,), *cfg.delta_range)
    deltaD = _uniform_vec(rng, (cfg.dD,), *cfg.delta_range)
    deltaU = _uniform_vec(rng, (cfg.dA if cfg.dA > 1 else 1,), *cfg.delta_u_range)

    if treatment_type == "continuous":
        if surface == "poly":
            # Use a signed quadratic draw and a shared scale rule so scalar
            # poly is closer in curvature to the vector-poly construction.
            quad_scale = _scalar_uniform(rng, *cfg.poly_quad_range) / max(cfg.dA, 2)
            if cfg.dA == 1:
                params = {
                    "beta_a": _scalar_uniform(rng, *cfg.poly_beta_range),
                    "quad_a": _scalar_uniform(rng, -quad_scale, quad_scale),
                }
            else:
                beta_a = _uniform_vec(rng, (cfg.dA,), *cfg.poly_beta_range)
                q_raw = rng.uniform(-quad_scale, quad_scale, size=(cfg.dA, cfg.dA))
                params = {"beta_a": beta_a, "Q": 0.5 * (q_raw + q_raw.T)}
        elif surface == "varying_slope":
            if cfg.dA == 1:
                params = {
                    "tau0": _scalar_uniform(rng, *cfg.vslope_tau0_range),
                    "tauL": _uniform_vec(rng, (cfg.dL,), *cfg.vslope_tauL_range),
                    "lam": _scalar_uniform(rng, *cfg.vslope_lambda_range),
                }
            else:
                params = {
                    "lam0": _uniform_vec(rng, (cfg.dA,), *cfg.vslope_lambda_range),
                    "lamL": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.vslope_tauL_range),
                }
        elif surface == "soft_threshold":
            if cfg.dA == 1:
                params = {
                    "tau0": _scalar_uniform(rng, *cfg.soft_threshold_tau0_range),
                    "tauL": _uniform_vec(rng, (cfg.dL,), *cfg.soft_threshold_tauL_range),
                    "k": _scalar_uniform(rng, *cfg.soft_threshold_k_range),
                    "c0": _scalar_uniform(rng, *cfg.soft_threshold_c0_range),
                    "cL": _uniform_vec(rng, (cfg.dL,), *cfg.soft_threshold_cL_range),
                }
            else:
                params = {
                    "tau0": _uniform_vec(rng, (cfg.dA,), *cfg.soft_threshold_tau0_range),
                    "tauL": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.soft_threshold_tauL_range),
                    "k": _uniform_vec(rng, (cfg.dA,), *cfg.soft_threshold_k_range),
                    "c0": _uniform_vec(rng, (cfg.dA,), *cfg.soft_threshold_c0_range),
                    "cL": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.soft_threshold_cL_range),
                    "scale": 1.0 / math.sqrt(float(cfg.dA)),
                }
        elif surface == "sinusoidal":
            if cfg.dA == 1:
                params = {
                    "lin0": _scalar_uniform(rng, *cfg.sinus_lin0_range),
                    "linL": _uniform_vec(rng, (cfg.dL,), *cfg.sinus_linL_range),
                    "tau0": _scalar_uniform(rng, 0.2, 0.6),
                    "tauL": _uniform_vec(rng, (cfg.dL,), *cfg.sinus_tauL_range),
                    "omega0": _scalar_uniform(rng, 0.4, 0.9),
                    "omegaL": _uniform_vec(rng, (cfg.dL,), -0.15, 0.15),
                    "phase0": _scalar_uniform(rng, *cfg.sinus_phase0_range),
                    "phaseL": _uniform_vec(rng, (cfg.dL,), -0.25, 0.25),
                }
            else:
                params = {
                    "lin0": _uniform_vec(rng, (cfg.dA,), *cfg.sinus_lin0_range),
                    "linL": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.sinus_linL_range),
                    "tau0": _uniform_vec(rng, (cfg.dA,), 0.2, 0.6),
                    "tauL": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.sinus_tauL_range),
                    "omega0": _uniform_vec(rng, (cfg.dA,), 0.4, 0.9),
                    "omegaL": _uniform_vec(rng, (cfg.dA, cfg.dL), -0.15, 0.15),
                    "phase0": _uniform_vec(rng, (cfg.dA,), *cfg.sinus_phase0_range),
                    "phaseL": _uniform_vec(rng, (cfg.dA, cfg.dL), -0.25, 0.25),
                    "scale": 1.0 / math.sqrt(float(cfg.dA)),
                }
        else:
            raise ValueError(f"Unknown continuous surface family: {surface}")
    else:
        if surface == "binary_linear":
            if cfg.dA == 1:
                params = {"tau0": _scalar_uniform(rng, *cfg.binary_tau0_range), "tauL": _uniform_vec(rng, (cfg.dL,), *cfg.binary_tauL_range)}
            else:
                params = {"lam0": _uniform_vec(rng, (cfg.dA,), *cfg.binary_tau0_range), "lamL": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.binary_tauL_range)}
        elif surface == "binary_threshold_modulated":
            if cfg.dA == 1:
                params = {
                    "tau10": _scalar_uniform(rng, *cfg.binary_tau0_range),
                    "tau1L": _uniform_vec(rng, (cfg.dL,), *cfg.binary_tauL_range),
                    "tau20": _scalar_uniform(rng, *cfg.binary_mod_tau0_range),
                    "tau2L": _uniform_vec(rng, (cfg.dL,), *cfg.binary_mod_tauL_range),
                    "gamma": _uniform_vec(rng, (cfg.dL,), *cfg.binary_mod_gamma_range),
                }
            else:
                params = {
                    "lam10": _uniform_vec(rng, (cfg.dA,), *cfg.binary_tau0_range),
                    "lam1L": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.binary_tauL_range),
                    "lam20": _uniform_vec(rng, (cfg.dA,), *cfg.binary_mod_tau0_range),
                    "lam2L": _uniform_vec(rng, (cfg.dA, cfg.dL), *cfg.binary_mod_tauL_range),
                    "gamma": _uniform_vec(rng, (cfg.dL,), *cfg.binary_mod_gamma_range),
                }
        else:
            raise ValueError(f"Unknown binary surface family: {surface}")

    return OutcomeTheta(surface=surface, mu0=mu0, muL=muL, deltaC=deltaC, deltaD=deltaD, deltaU=deltaU, params=params)


def build_dgp_definition(design_seed: int, surface: SurfaceType, cfg: SimConfig, treatment_type: TreatmentType = "continuous") -> DGPDefinition:
    _validate_surface_for_treatment(surface, treatment_type)
    _validate_dims_for_treatment(cfg, treatment_type)

    common_root = _common_design_seedseq(design_seed)
    proxy_ss, mixing_ss = common_root.spawn(2)

    proxy_loading, proxy_attempt = sample_proxy_loading(proxy_ss, cfg)
    mixing = sample_mixing_coeffs(np.random.default_rng(mixing_ss), cfg)

    treat_root = _treatment_design_seedseq(design_seed, treatment_type)
    treat_ss, = treat_root.spawn(1)
    treat_theta = sample_treatment_theta_with_accept_reject(treat_ss, cfg, treatment_type=treatment_type)

    surface_root = _surface_design_seedseq(design_seed, surface, treatment_type)
    outcome_ss, = surface_root.spawn(1)
    outcome_theta = sample_outcome_theta(np.random.default_rng(outcome_ss), surface, treatment_type, cfg)

    return DGPDefinition(
        surface=surface,
        treatment_type=treatment_type,
        design_seed=int(design_seed),
        proxy_loading=proxy_loading,
        proxy_loading_attempt=int(proxy_attempt),
        mixing=mixing,
        treatment=treat_theta,
        outcome=outcome_theta,
    )


def sample_latents(n: int, rng: Generator, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    C = rng.normal(size=(n, cfg.dC))
    V = rng.normal(size=(n, cfg.dV))
    D = rng.normal(size=(n, cfg.dD))
    U = rng.normal(size=(n, cfg.dA if cfg.dA > 1 else 1))
    return C, V, D, U


def sample_covariates(n: int, rng: Generator, cfg: SimConfig) -> np.ndarray:
    return rng.binomial(n=1, p=0.5, size=(n, cfg.dL)).astype(float)


def sample_treatment(C: np.ndarray, V: np.ndarray, L: np.ndarray, U: np.ndarray, rng: Generator, dgp: DGPDefinition, cfg: SimConfig) -> np.ndarray:
    tt = dgp.treatment
    if cfg.dA > 1:
        if tt.B is None:
            raise ValueError("Vector treatment requires B.")
        linear_index = tt.alpha0[None, :] + C @ tt.alphaC.T + L @ tt.alphaL.T + V @ tt.B.T + U * tt.gamma_u[None, :]
        if dgp.treatment_type == "continuous":
            eps_a = rng.normal(loc=0.0, scale=cfg.sigma_a, size=(C.shape[0], cfg.dA))
            raw_a = np.asarray(linear_index + eps_a, dtype=float)
            return raw_a
        if dgp.treatment_type == "binary":
            eps_a = rng.normal(loc=0.0, scale=1.0, size=(C.shape[0], cfg.dA))
            return np.asarray((linear_index + eps_a >= 0.0).astype(float), dtype=float)
        raise ValueError(f"Unknown treatment type: {dgp.treatment_type}")

    betaL = treatment_loading(L, tt)
    linear_index = (
        tt.alpha0[0]
        + C @ tt.alphaC[0][:, None]
        + L @ tt.alphaL[0][:, None]
        + np.sum(betaL * V, axis=1, keepdims=True)
        + tt.gamma_u[0] * U
    )
    if dgp.treatment_type == "continuous":
        eps_a = rng.normal(loc=0.0, scale=cfg.sigma_a, size=(C.shape[0], 1))
        raw_a = linear_index + eps_a
        return raw_a.reshape(-1)
    if dgp.treatment_type == "binary":
        eps_a = rng.normal(loc=0.0, scale=1.0, size=(C.shape[0], 1))
        return ((linear_index + eps_a) >= 0.0).astype(float).reshape(-1)
    raise ValueError(f"Unknown treatment type: {dgp.treatment_type}")


def _baseline_response(L: np.ndarray, outcome_theta: OutcomeTheta) -> np.ndarray:
    return outcome_theta.mu0 + L @ outcome_theta.muL[:, None]


def _varying_slope_effect(L: np.ndarray, outcome_theta: OutcomeTheta) -> np.ndarray:
    tau0 = float(outcome_theta.params["tau0"])
    tauL = np.asarray(outcome_theta.params["tauL"], dtype=float)
    return tau0 + L @ tauL[:, None]


def _binary_effect(L: np.ndarray, outcome_theta: OutcomeTheta) -> np.ndarray:
    tau0 = float(outcome_theta.params["tau0"])
    tauL = np.asarray(outcome_theta.params["tauL"], dtype=float)
    return tau0 + L @ tauL[:, None]


def structural_response(A: np.ndarray | float, L: np.ndarray, dgp: DGPDefinition, cfg: Optional[SimConfig] = None) -> np.ndarray:
    outcome_theta = dgp.outcome
    L = np.asarray(L, dtype=float)
    n = L.shape[0]

    if np.isscalar(A):
        if cfg is not None and cfg.dA > 1:
            A_arr = np.full((n, cfg.dA), float(A), dtype=float)
        else:
            A_arr = np.full((n, 1), float(A), dtype=float)
    else:
        A_arr = np.asarray(A, dtype=float)
        if A_arr.ndim == 1:
            A_arr = A_arr.reshape(-1, 1)

    baseline_response = _baseline_response(L, outcome_theta)

    if dgp.treatment_type == "continuous":
        if outcome_theta.surface == "poly":
            if A_arr.shape[1] == 1 and "quad_a" in outcome_theta.params:
                beta_a = float(outcome_theta.params["beta_a"])
                quad_a = float(outcome_theta.params["quad_a"])
                val = baseline_response + beta_a * A_arr + quad_a * (A_arr ** 2)
            else:
                beta_a = np.asarray(outcome_theta.params["beta_a"], dtype=float)
                Q = np.asarray(outcome_theta.params["Q"], dtype=float)
                linear = A_arr @ beta_a[:, None]
                quad = np.einsum("ni,ij,nj->n", A_arr, Q, A_arr)[:, None]
                val = baseline_response + linear + quad
        elif outcome_theta.surface == "varying_slope":
            if A_arr.shape[1] == 1 and "lam" in outcome_theta.params:
                varying_slope = _varying_slope_effect(L, outcome_theta)
                slope_offset = float(outcome_theta.params["lam"])
                val = baseline_response + (varying_slope - slope_offset) * A_arr
            else:
                lam0 = np.asarray(outcome_theta.params["lam0"], dtype=float)
                lamL = np.asarray(outcome_theta.params["lamL"], dtype=float)
                slopes = lam0[None, :] + L @ lamL.T
                val = baseline_response + np.sum(slopes * A_arr, axis=1, keepdims=True)
        elif outcome_theta.surface == "soft_threshold":
            if A_arr.shape[1] == 1:
                tau = float(outcome_theta.params["tau0"]) + L @ np.asarray(outcome_theta.params["tauL"], dtype=float)[:, None]
                c_loc = float(outcome_theta.params["c0"]) + L @ np.asarray(outcome_theta.params["cL"], dtype=float)[:, None]
                k = float(outcome_theta.params["k"])
                soft_hinge = _softplus_np(k * (A_arr - c_loc)) / max(k, 1e-8)
                val = baseline_response + tau * soft_hinge
            else:
                tau0 = np.asarray(outcome_theta.params["tau0"], dtype=float)
                tauL = np.asarray(outcome_theta.params["tauL"], dtype=float)
                k = np.asarray(outcome_theta.params["k"], dtype=float)
                c0 = np.asarray(outcome_theta.params["c0"], dtype=float)
                cL = np.asarray(outcome_theta.params["cL"], dtype=float)
                scale = float(outcome_theta.params["scale"])
                tau = tau0[None, :] + L @ tauL.T
                c_loc = c0[None, :] + L @ cL.T
                soft_hinge = _softplus_np(k[None, :] * (A_arr - c_loc)) / np.maximum(k[None, :], 1e-8)
                val = baseline_response + scale * np.sum(tau * soft_hinge, axis=1, keepdims=True)
        elif outcome_theta.surface == "sinusoidal":
            if A_arr.shape[1] == 1:
                lin = float(outcome_theta.params["lin0"]) + L @ np.asarray(outcome_theta.params["linL"], dtype=float)[:, None]
                tau = float(outcome_theta.params["tau0"]) + L @ np.asarray(outcome_theta.params["tauL"], dtype=float)[:, None]
                omega = float(outcome_theta.params["omega0"]) + L @ np.asarray(outcome_theta.params["omegaL"], dtype=float)[:, None]
                omega = np.maximum(omega, 0.25)
                phase = float(outcome_theta.params["phase0"]) + L @ np.asarray(outcome_theta.params["phaseL"], dtype=float)[:, None]
                val = baseline_response + lin * A_arr + tau * np.sin(omega * A_arr + phase)
            else:
                lin0 = np.asarray(outcome_theta.params["lin0"], dtype=float)
                linL = np.asarray(outcome_theta.params["linL"], dtype=float)
                tau0 = np.asarray(outcome_theta.params["tau0"], dtype=float)
                tauL = np.asarray(outcome_theta.params["tauL"], dtype=float)
                omega0 = np.asarray(outcome_theta.params["omega0"], dtype=float)
                omegaL = np.asarray(outcome_theta.params["omegaL"], dtype=float)
                phase0 = np.asarray(outcome_theta.params["phase0"], dtype=float)
                phaseL = np.asarray(outcome_theta.params["phaseL"], dtype=float)
                scale = float(outcome_theta.params["scale"])
                lin = lin0[None, :] + L @ linL.T
                tau = tau0[None, :] + L @ tauL.T
                omega = omega0[None, :] + L @ omegaL.T
                omega = np.maximum(omega, 0.25)
                phase = phase0[None, :] + L @ phaseL.T
                val = baseline_response + scale * np.sum(
                    lin * A_arr + tau * np.sin(omega * A_arr + phase),
                    axis=1,
                    keepdims=True,
                )
        else:
            raise ValueError(f"Unknown continuous surface family: {outcome_theta.surface}")
    else:
        if outcome_theta.surface == "binary_linear":
            if A_arr.shape[1] == 1 and "tau0" in outcome_theta.params:
                tau = _binary_effect(L, outcome_theta)
                val = baseline_response + tau * A_arr
            else:
                lam0 = np.asarray(outcome_theta.params["lam0"], dtype=float)
                lamL = np.asarray(outcome_theta.params["lamL"], dtype=float)
                effects = lam0[None, :] + L @ lamL.T
                val = baseline_response + np.sum(effects * A_arr, axis=1, keepdims=True)
        elif outcome_theta.surface == "binary_threshold_modulated":
            gamma = np.asarray(outcome_theta.params["gamma"], dtype=float)
            gate = (L @ gamma[:, None] >= 0.0).astype(float)
            if A_arr.shape[1] == 1 and "tau10" in outcome_theta.params:
                tau1 = float(outcome_theta.params["tau10"]) + L @ np.asarray(outcome_theta.params["tau1L"], dtype=float)[:, None]
                tau2 = float(outcome_theta.params["tau20"]) + L @ np.asarray(outcome_theta.params["tau2L"], dtype=float)[:, None]
                val = baseline_response + (tau1 + tau2 * gate) * A_arr
            else:
                lam10 = np.asarray(outcome_theta.params["lam10"], dtype=float)
                lam1L = np.asarray(outcome_theta.params["lam1L"], dtype=float)
                lam20 = np.asarray(outcome_theta.params["lam20"], dtype=float)
                lam2L = np.asarray(outcome_theta.params["lam2L"], dtype=float)
                eff1 = lam10[None, :] + L @ lam1L.T
                eff2 = lam20[None, :] + L @ lam2L.T
                effects = eff1 + gate * eff2
                val = baseline_response + np.sum(effects * A_arr, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown binary surface family: {outcome_theta.surface}")

    return np.asarray(val, dtype=float).reshape(-1)


def confounder_nuisance(C: np.ndarray, dgp: DGPDefinition) -> np.ndarray:
    return (C @ dgp.outcome.deltaC[:, None]).reshape(-1)


def sample_outcome(
    A: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    U: np.ndarray,
    L: np.ndarray,
    rng: Generator,
    dgp: DGPDefinition,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    response_signal = structural_response(A, L, dgp, cfg=cfg)
    nuisance_signal = confounder_nuisance(C, dgp)
    direct_effect_signal = (D @ dgp.outcome.deltaD[:, None]).reshape(-1)

    if cfg.dA > 1:
        hidden_confounding_signal = np.sum(U * dgp.outcome.deltaU[None, :], axis=1)
        n = A.shape[0]
    else:
        hidden_confounding_signal = dgp.outcome.deltaU[0] * U.reshape(-1)
        n = len(A)

    eps_y = rng.normal(loc=0.0, scale=cfg.sigma_y, size=n)
    y = response_signal + nuisance_signal + direct_effect_signal + hidden_confounding_signal + eps_y
    structural_noise = direct_effect_signal + hidden_confounding_signal + eps_y
    return y, response_signal, nuisance_signal, structural_noise


def simulate_dataset(data_seed: int, dgp: DGPDefinition, cfg: SimConfig) -> SimulatedDataset:
    rng = _rng_from_data_seed(data_seed)
    L = sample_covariates(cfg.n, rng, cfg)
    C, V, D, U = sample_latents(cfg.n, rng, cfg)
    X = mix_source(C, V, D, dgp.mixing)
    M = sample_proxy_measurements(C, rng, dgp, cfg)
    A = sample_treatment(C, V, L, U, rng, dgp, cfg)
    Y, response_signal_obs, nuisance_signal_obs, structural_noise_obs = sample_outcome(A, C, D, U, L, rng, dgp, cfg)
    U_out = U if cfg.dA > 1 else U.reshape(-1)
    return SimulatedDataset(
        X=X,
        M=M,
        L=L,
        A=A,
        Y=Y,
        C=C,
        V=V,
        D=D,
        U=U_out,
        structural_response_obs=response_signal_obs,
        confounder_nuisance_obs=nuisance_signal_obs,
        structural_noise_obs=structural_noise_obs,
    )


def dgp_to_dict(dgp: DGPDefinition) -> Dict[str, Any]:
    payload = asdict(dgp)

    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return convert(payload)


def summarize_dataset(data: SimulatedDataset, dgp: DGPDefinition, cfg: SimConfig) -> Dict[str, Any]:
    proxy_svals = np.linalg.svd(dgp.proxy_loading, compute_uv=False)
    summary = {
        "n": int(cfg.n),
        "surface": dgp.surface,
        "treatment_type": dgp.treatment_type,
        "design_seed": int(dgp.design_seed),
        "dA": int(cfg.dA),
        "proxy_loading_attempt": int(dgp.proxy_loading_attempt),
        "accepted_attempt_treat": int(dgp.treatment.accepted_attempt),
        "A_mean": float(np.mean(data.A)),
        "A_std": float(np.std(data.A)),
        "A_min": float(np.min(data.A)),
        "A_max": float(np.max(data.A)),
        "Y_mean": float(np.mean(data.Y)),
        "Y_std": float(np.std(data.Y)),
        "min_singular_value_Lambda_C": float(proxy_svals[-1]),
        "mix_linear_abs_max": float(np.max(np.abs(dgp.mixing.a))),
        "mix_quadratic_abs_max": float(np.max(np.abs(dgp.mixing.b))),
        "mix_cubic_abs_max": float(np.max(np.abs(dgp.mixing.c))),
    }

    if dgp.treatment.B is not None:
        svals_B = np.linalg.svd(dgp.treatment.B, compute_uv=False)
        row_norms_B = np.linalg.norm(dgp.treatment.B, axis=1)
        summary.update({
            "min_singular_value_B": float(svals_B[-1]),
            "min_row_norm_B": float(row_norms_B.min()),
            "B_fro_norm": float(np.linalg.norm(dgp.treatment.B)),
        })
    else:
        L_anchor = _anchor_L_values(cfg.dL, cfg.dV)
        beta_anchor = treatment_loading(L_anchor, dgp.treatment)
        beta_norms = np.linalg.norm(beta_anchor, axis=1)
        svals = np.linalg.svd(dgp.treatment.S_beta, compute_uv=False)
        summary.update({
            "min_beta_norm_over_anchor_L": float(beta_norms.min()),
            "min_singular_value_S_beta": float(svals[-1]),
        })

    if dgp.treatment_type == "binary" and dgp.treatment.binary_accept_stats:
        if cfg.dA == 1:
            summary.update({
                "binary_signal_ratio": float(dgp.treatment.binary_accept_stats["signal_ratio"]),
                "binary_iv_signal_sd": float(dgp.treatment.binary_accept_stats["iv_signal_sd"]),
                "binary_nuisance_signal_sd": float(dgp.treatment.binary_accept_stats["nuisance_signal_sd"]),
                "binary_global_treated_rate_pilot": float(dgp.treatment.binary_accept_stats["treated_rate_global"]),
                "binary_anchor_treated_rates_pilot": {
                    k: float(v) for k, v in dgp.treatment.binary_accept_stats["anchor_treated_rates"].items()
                },
            })
        else:
            stats = dgp.treatment.binary_accept_stats
            summary.update({
                "binary_signal_ratio_by_coord": [float(v) for v in stats["signal_ratio_by_coord"]],
                "binary_iv_signal_sd_by_coord": [float(v) for v in stats["iv_signal_sd_by_coord"]],
                "binary_nuisance_signal_sd_by_coord": [float(v) for v in stats["nuisance_signal_sd_by_coord"]],
                "binary_treated_rates_by_coord_pilot": [float(v) for v in stats["treated_rates_by_coord"]],
                "binary_min_signal_ratio": float(stats["min_signal_ratio"]),
                "binary_distinct_patterns_pilot": int(stats["distinct_patterns"]),
                "binary_max_pattern_share_pilot": float(stats["max_pattern_share"]),
                "binary_joint_overlap_filter_enabled": bool(cfg.binary_enable_joint_overlap_filter),
                "binary_joint_cell_prob_min": float(stats["joint_cell_prob_min"]),
                "binary_joint_cell_prob_p01": float(stats["joint_cell_prob_p01"]),
                "binary_joint_cell_prob_p05": float(stats["joint_cell_prob_p05"]),
                "binary_joint_cell_prob_p10": float(stats["joint_cell_prob_p10"]),
                "binary_joint_cell_prob_p50": float(stats["joint_cell_prob_p50"]),
                "binary_joint_cell_prob_p95": float(stats["joint_cell_prob_p95"]),
                "binary_joint_cell_prob_max": float(stats["joint_cell_prob_max"]),
                "binary_joint_bad_mass_lt_001": float(stats["joint_bad_mass_lt_001"]),
                "binary_joint_bad_mass_lt_005": float(stats["joint_bad_mass_lt_005"]),
                "binary_joint_bad_mass_lt_010": float(stats["joint_bad_mass_lt_010"]),
                "binary_joint_cell_prob_mean_by_cell": {
                    k: float(v) for k, v in stats["joint_cell_prob_mean_by_cell"].items()
                },
                "binary_joint_cell_prob_p01_by_cell": {
                    k: float(v) for k, v in stats["joint_cell_prob_p01_by_cell"].items()
                },
                "binary_joint_cell_prob_p05_by_cell": {
                    k: float(v) for k, v in stats["joint_cell_prob_p05_by_cell"].items()
                },
                "binary_joint_cell_prob_min_by_cell": {
                    k: float(v) for k, v in stats["joint_cell_prob_min_by_cell"].items()
                },
            })
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate the NPIV benchmark with scalar/vector binary treatment and scalar/vector continuous treatment."
    )
    parser.add_argument("--treatment-type", type=str, choices=["continuous", "binary"], default="continuous", help="Treatment type.")
    parser.add_argument("--surface", dest="surface", type=str, choices=ALL_SURFACES, default="poly", help="Outcome-surface family.")
    parser.add_argument("--design-seed", type=int, default=42, help="Seed used to generate the fixed DGP definition for the chosen surface.")
    parser.add_argument("--data-seed", type=int, default=42, help="Seed used to generate one iid dataset from the fixed DGP.")
    parser.add_argument("--n", type=int, default=5000, help="Sample size.")
    parser.add_argument("--dC", type=int, default=3, help="Dimension of latent confounder C.")
    parser.add_argument("--dV", type=int, default=2, help="Dimension of latent IV block V.")
    parser.add_argument("--dD", type=int, default=2, help="Dimension of latent direct-effect block D.")
    parser.add_argument("--dL", type=int, default=None, help="Dimension of observed covariates L; defaults to dV.")
    parser.add_argument("--dM", type=int, default=None, help="Dimension of proxy M; defaults to dC.")
    parser.add_argument("--dA", type=int, default=1, help="Treatment dimension. Scalar treatment uses dA=1.")
    parser.add_argument("--sigma-m", type=float, default=0.50, help="Proxy noise scale.")
    parser.add_argument("--sigma-a", type=float, default=0.50, help="Continuous-treatment noise scale.")
    parser.add_argument("--sigma-y", type=float, default=0.50, help="Outcome noise scale.")
    parser.add_argument(
        "--min-continuous-confound-ratio",
        type=float,
        default=0.50,
        help="Minimum latent first-stage U share for continuous treatment; set 0 to disable.",
    )

    # Joint-cell overlap / positivity controls.
    parser.add_argument(
        "--binary-disable-joint-overlap-filter",
        action="store_true",
        help="Disable vector-binary joint-cell positivity filtering. Useful for stress-test DGPs.",
    )
    parser.add_argument(
        "--binary-min-joint-cell-prob-p01",
        type=float,
        default=0.0025,
        help="Minimum allowed 1st percentile of oracle joint-cell probabilities for vector-binary treatment.",
    )
    parser.add_argument(
        "--binary-min-joint-cell-prob-p05",
        type=float,
        default=0.0100,
        help="Minimum allowed 5th percentile of oracle joint-cell probabilities for vector-binary treatment.",
    )
    parser.add_argument(
        "--binary-max-joint-cell-prob-p95",
        type=float,
        default=0.9900,
        help="Maximum allowed 95th percentile of oracle joint-cell probabilities for vector-binary treatment.",
    )
    parser.add_argument(
        "--max-treat-tries",
        type=int,
        default=2000,
        help="Maximum treatment-coefficient proposals before failing accept-reject.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = SimConfig(
        n=args.n,
        dC=args.dC,
        dV=args.dV,
        dD=args.dD,
        dL=args.dL,
        dM=args.dM,
        dA=args.dA,
        sigma_m=args.sigma_m,
        sigma_a=args.sigma_a,
        sigma_y=args.sigma_y,
        max_treat_tries=args.max_treat_tries,
        min_continuous_confound_ratio=args.min_continuous_confound_ratio,
        binary_enable_joint_overlap_filter=not args.binary_disable_joint_overlap_filter,
        binary_min_joint_cell_prob_p01=args.binary_min_joint_cell_prob_p01,
        binary_min_joint_cell_prob_p05=args.binary_min_joint_cell_prob_p05,
        binary_max_joint_cell_prob_p95=args.binary_max_joint_cell_prob_p95,
    )

    dgp = build_dgp_definition(args.design_seed, args.surface, cfg, treatment_type=args.treatment_type)
    data = simulate_dataset(args.data_seed, dgp, cfg)
    summary = summarize_dataset(data, dgp, cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
