"""
sim_npiv_with_binary_checked.py

Synthetic NPIV benchmark simulator for scalar treatment with support for
both continuous and binary treatment.

Design split
------------
- ``design_seed`` controls the fixed DGP design for a chosen outcome surface:
      dgp = build_dgp_definition(design_seed, surface, treatment_type, cfg)
  This includes:
      * proxy loading matrix Lambda_C
      * triangular polynomial mixing coefficients for X = f(C, V, D)
      * treatment coefficients
      * outcome coefficients

- ``data_seed`` controls the sampled iid dataset from that fixed DGP:
      data = simulate_dataset(data_seed, dgp, cfg)

The original continuous-treatment path is preserved; binary treatment is added as
an additional branch with its own admissible surface families.

Binary-treatment-only extra accept/reject checks
-----------------------------------------------
In addition to the shared structural checks on proxy informativeness and
loading-matrix rank / anchor strength, the binary-treatment branch adds:

1. Signal-ratio check:
      sd(beta(L)^T V) / sd(alpha_C^T C + alpha_L^T L + gamma_U U + eps_A)
   on a deterministic pilot sample.

2. Positivity checks:
   - global treated-rate check on a pilot sample with random binary L
   - anchor-specific treated-rate checks on the binary anchor set used by the
     scalar-treatment identification argument.

Continuous treatment does not use the signal-ratio check.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
from numpy.random import Generator, SeedSequence


TreatmentType = Literal["continuous", "binary"]
ContinuousSurfaceType = Literal["poly", "tanh", "thresh", "varying_slope"]
BinarySurfaceType = Literal["binary_linear", "binary_threshold_modulated"]
SurfaceType = Literal[
    "poly",
    "tanh",
    "thresh",
    "varying_slope",
    "binary_linear",
    "binary_threshold_modulated",
]

CONTINUOUS_SURFACES = ["poly", "tanh", "thresh", "varying_slope"]
BINARY_SURFACES = ["binary_linear", "binary_threshold_modulated"]
ALL_SURFACES = CONTINUOUS_SURFACES + BINARY_SURFACES

SURFACE_CODE: Dict[str, int] = {
    "poly": 0,
    "tanh": 1,
    "thresh": 2,
    "varying_slope": 3,
    "binary_linear": 4,
    "binary_threshold_modulated": 5,
}


@dataclass
class SimConfig:
    """Configuration for the scalar-treatment synthetic NPIV DGP."""

    # Sample size and dimensions
    n: int = 5000
    dC: int = 3
    dV: int = 3
    dD: int = 2
    dL: Optional[int] = None  # defaults to dV in __post_init__
    dM: Optional[int] = None  # defaults to dC in __post_init__

    # Noise scales (fixed across all DGPs)
    sigma_m: float = 0.50
    sigma_a: float = 1.00  # continuous-treatment noise scale only
    sigma_y: float = 1.00

    # Accept/reject settings for treatment coefficients (shared)
    max_treat_tries: int = 2000
    min_sbeta_singular_value: float = 0.35
    min_beta_norm_anchor: float = 0.35

    # Accept/reject settings for proxy loading
    max_proxy_tries: int = 2000
    min_lambda_singular_value: float = 0.35

    # Binary-treatment-only extra accept/reject settings
    binary_pilot_n: int = 2048
    binary_min_signal_ratio: float = 0.50
    binary_min_global_treated_rate: float = 0.10
    binary_max_global_treated_rate: float = 0.90
    binary_min_anchor_treated_rate: float = 0.15
    binary_max_anchor_treated_rate: float = 0.85

    # Coefficient proposal ranges: treatment and outcome
    alpha0_range: tuple[float, float] = (-0.5, 0.5)
    alpha_range: tuple[float, float] = (-1.0, 1.0)
    gamma_u_range: tuple[float, float] = (0.5, 1.5)

    delta_range: tuple[float, float] = (-1.0, 1.0)
    delta_u_range: tuple[float, float] = (0.5, 1.5)

    mu0_range: tuple[float, float] = (-1.0, 1.0)
    muL_range: tuple[float, float] = (-1.0, 1.0)

    poly_beta_range: tuple[float, float] = (-1.0, 1.0)
    poly_quad_range: tuple[float, float] = (0.2, 1.0)

    tanh_amp_range: tuple[float, float] = (0.5, 2.0)
    tanh_slope_range: tuple[float, float] = (0.5, 2.0)
    tanh_gamma_range: tuple[float, float] = (-1.0, 1.0)

    thresh_tau0_range: tuple[float, float] = (0.5, 2.0)
    thresh_tauL_range: tuple[float, float] = (-1.0, 1.0)
    thresh_w_range: tuple[float, float] = (0.5, 2.0)

    vslope_tau0_range: tuple[float, float] = (0.5, 2.0)
    vslope_tauL_range: tuple[float, float] = (-1.0, 1.0)
    vslope_lambda_range: tuple[float, float] = (-1.0, 1.0)

    # Binary-treatment-only surfaces
    binary_tau0_range: tuple[float, float] = (0.5, 2.0)
    binary_tauL_range: tuple[float, float] = (-1.0, 1.0)
    binary_mod_tau0_range: tuple[float, float] = (0.25, 1.5)
    binary_mod_tauL_range: tuple[float, float] = (-1.0, 1.0)
    binary_mod_gamma_range: tuple[float, float] = (-1.0, 1.0)

    # Proposal ranges: proxy loading and mixing coefficients
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
        if self.dL < self.dV:
            raise ValueError(
                "This construction requires dL >= dV so that the loading matrix S_beta can have full row rank."
            )
        if self.max_treat_tries < 1:
            raise ValueError("max_treat_tries must be positive.")
        if self.max_proxy_tries < 1:
            raise ValueError("max_proxy_tries must be positive.")
        if self.binary_pilot_n < 1:
            raise ValueError("binary_pilot_n must be positive.")


@dataclass
class MixingCoeffs:
    """Coefficients for a triangular polynomial diffeomorphism."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass
class TreatmentTheta:
    """Random treatment-side coefficients for scalar treatment."""

    alpha0: float
    alphaC: np.ndarray
    alphaL: np.ndarray
    b0: np.ndarray
    S_beta: np.ndarray
    gamma_u: float
    accepted_attempt: int
    binary_accept_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeTheta:
    """Random outcome-side coefficients conditional on the chosen surface type."""

    surface: SurfaceType
    mu0: float
    muL: np.ndarray
    deltaC: np.ndarray
    deltaD: np.ndarray
    deltaU: float
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DGPDefinition:
    """Complete DGP definition for one chosen surface type."""

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
    """A simulated dataset plus optional latent ground truth."""

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
    raise RuntimeError(
        "Failed to find an acceptable proxy loading draw. "
        "Increase max_proxy_tries or relax the proxy acceptance threshold."
    )


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


def propose_treatment_theta(rng: Generator, cfg: SimConfig) -> TreatmentTheta:
    return TreatmentTheta(
        alpha0=_scalar_uniform(rng, *cfg.alpha0_range),
        alphaC=_uniform_vec(rng, (cfg.dC,), *cfg.alpha_range),
        alphaL=_uniform_vec(rng, (cfg.dL,), *cfg.alpha_range),
        b0=_uniform_vec(rng, (cfg.dV,), *cfg.alpha_range),
        S_beta=_uniform_vec(rng, (cfg.dV, cfg.dL), *cfg.alpha_range),
        gamma_u=_scalar_uniform(rng, *cfg.gamma_u_range),
        accepted_attempt=-1,
    )


def treatment_loading(L: np.ndarray, treatment_theta: TreatmentTheta) -> np.ndarray:
    return treatment_theta.b0[None, :] + L @ treatment_theta.S_beta.T


def _binary_signal_ratio_and_positivity_stats(
    treatment_theta: TreatmentTheta,
    cfg: SimConfig,
    pilot_rng: Generator,
) -> Dict[str, Any]:
    """Binary-treatment-only pilot diagnostics for accept/reject."""
    n = int(cfg.binary_pilot_n)

    C = pilot_rng.normal(size=(n, cfg.dC))
    V = pilot_rng.normal(size=(n, cfg.dV))
    U = pilot_rng.normal(size=(n, 1))
    eps = pilot_rng.normal(size=(n, 1))  # binary branch uses latent N(0,1) threshold noise
    L = pilot_rng.binomial(n=1, p=0.5, size=(n, cfg.dL)).astype(float)

    betaL = treatment_loading(L, treatment_theta)
    iv_signal = np.sum(betaL * V, axis=1)
    nuis_signal = (
        C @ treatment_theta.alphaC[:, None]
        + L @ treatment_theta.alphaL[:, None]
        + treatment_theta.gamma_u * U
        + eps
    ).reshape(-1)

    iv_sd = float(np.std(iv_signal))
    nuis_sd = float(np.std(nuis_signal))
    signal_ratio = iv_sd / max(nuis_sd, 1e-8)

    eta = treatment_theta.alpha0 + iv_signal + nuis_signal
    treated_rate_global = float(np.mean(eta >= 0.0))

    anchor_rates: Dict[str, float] = {}
    anchors = _anchor_L_values(cfg.dL, cfg.dV)
    for idx, l in enumerate(anchors):
        l_arr = np.repeat(l[None, :], n, axis=0)
        beta_l = treatment_loading(l_arr, treatment_theta)
        iv_l = np.sum(beta_l * V, axis=1)
        nuis_l = (
            C @ treatment_theta.alphaC[:, None]
            + l_arr @ treatment_theta.alphaL[:, None]
            + treatment_theta.gamma_u * U
            + eps
        ).reshape(-1)
        eta_l = treatment_theta.alpha0 + iv_l + nuis_l
        anchor_rates[f"anchor_{idx}"] = float(np.mean(eta_l >= 0.0))

    return {
        "signal_ratio": signal_ratio,
        "iv_signal_sd": iv_sd,
        "nuisance_signal_sd": nuis_sd,
        "treated_rate_global": treated_rate_global,
        "anchor_treated_rates": anchor_rates,
    }


def accept_treatment_theta(
    treatment_theta: TreatmentTheta,
    cfg: SimConfig,
    treatment_type: TreatmentType,
    pilot_rng: Optional[Generator] = None,
) -> bool:
    """Shared structural checks; binary adds positivity and signal-ratio."""
    svals = np.linalg.svd(treatment_theta.S_beta, compute_uv=False)
    if float(svals[-1]) < cfg.min_sbeta_singular_value:
        return False

    L_anchor = _anchor_L_values(cfg.dL, cfg.dV)
    beta_anchor = treatment_loading(L_anchor, treatment_theta)
    beta_norms = np.linalg.norm(beta_anchor, axis=1)
    if float(beta_norms.min()) < cfg.min_beta_norm_anchor:
        return False

    if treatment_type == "continuous":
        treatment_theta.binary_accept_stats = {}
        return True

    if pilot_rng is None:
        raise ValueError("pilot_rng is required for binary-treatment acceptance checks.")

    stats = _binary_signal_ratio_and_positivity_stats(treatment_theta, cfg, pilot_rng)
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


def sample_treatment_theta_with_accept_reject(
    root_ss: SeedSequence,
    cfg: SimConfig,
    treatment_type: TreatmentType,
) -> TreatmentTheta:
    for attempt, ss in enumerate(root_ss.spawn(cfg.max_treat_tries)):
        proposal_ss, pilot_ss = ss.spawn(2)
        rng = np.random.default_rng(proposal_ss)
        theta = propose_treatment_theta(rng, cfg)
        pilot_rng = np.random.default_rng(pilot_ss) if treatment_type == "binary" else None
        if accept_treatment_theta(theta, cfg, treatment_type, pilot_rng=pilot_rng):
            theta.accepted_attempt = attempt
            return theta
    raise RuntimeError(
        "Failed to find an acceptable treatment coefficient draw. "
        "Increase max_treat_tries or relax the treatment acceptance thresholds."
    )


def sample_outcome_theta(
    rng: Generator,
    surface: SurfaceType,
    treatment_type: TreatmentType,
    cfg: SimConfig,
) -> OutcomeTheta:
    mu0 = _scalar_uniform(rng, *cfg.mu0_range)
    muL = _uniform_vec(rng, (cfg.dL,), *cfg.muL_range)

    deltaC = _uniform_vec(rng, (cfg.dC,), *cfg.delta_range)
    deltaD = _uniform_vec(rng, (cfg.dD,), *cfg.delta_range)
    deltaU = _scalar_uniform(rng, *cfg.delta_u_range)

    params: Dict[str, Any]

    if treatment_type == "continuous":
        if surface == "poly":
            params = {
                "beta_a": _scalar_uniform(rng, *cfg.poly_beta_range),
                "quad_a": _scalar_uniform(rng, *cfg.poly_quad_range),
            }
        elif surface == "tanh":
            params = {
                "amp": _scalar_uniform(rng, *cfg.tanh_amp_range),
                "slope": _scalar_uniform(rng, *cfg.tanh_slope_range),
                "gamma": _uniform_vec(rng, (cfg.dL,), *cfg.tanh_gamma_range),
            }
        elif surface == "thresh":
            params = {
                "tau0": _scalar_uniform(rng, *cfg.thresh_tau0_range),
                "tauL": _uniform_vec(rng, (cfg.dL,), *cfg.thresh_tauL_range),
                "w": _scalar_uniform(rng, *cfg.thresh_w_range),
            }
        elif surface == "varying_slope":
            params = {
                "tau0": _scalar_uniform(rng, *cfg.vslope_tau0_range),
                "tauL": _uniform_vec(rng, (cfg.dL,), *cfg.vslope_tauL_range),
                "lam": _scalar_uniform(rng, *cfg.vslope_lambda_range),
            }
        else:
            raise ValueError(f"Unknown continuous surface family: {surface}")
    else:
        if surface == "binary_linear":
            params = {
                "tau0": _scalar_uniform(rng, *cfg.binary_tau0_range),
                "tauL": _uniform_vec(rng, (cfg.dL,), *cfg.binary_tauL_range),
            }
        elif surface == "binary_threshold_modulated":
            params = {
                "tau10": _scalar_uniform(rng, *cfg.binary_tau0_range),
                "tau1L": _uniform_vec(rng, (cfg.dL,), *cfg.binary_tauL_range),
                "tau20": _scalar_uniform(rng, *cfg.binary_mod_tau0_range),
                "tau2L": _uniform_vec(rng, (cfg.dL,), *cfg.binary_mod_tauL_range),
                "gamma": _uniform_vec(rng, (cfg.dL,), *cfg.binary_mod_gamma_range),
            }
        else:
            raise ValueError(f"Unknown binary surface family: {surface}")

    return OutcomeTheta(
        surface=surface,
        mu0=mu0,
        muL=muL,
        deltaC=deltaC,
        deltaD=deltaD,
        deltaU=deltaU,
        params=params,
    )


def build_dgp_definition(
    design_seed: int,
    surface: SurfaceType,
    cfg: SimConfig,
    treatment_type: TreatmentType = "continuous",
) -> DGPDefinition:
    _validate_surface_for_treatment(surface, treatment_type)

    common_root = _common_design_seedseq(design_seed)
    proxy_ss, mixing_ss = common_root.spawn(2)

    proxy_loading, proxy_attempt = sample_proxy_loading(proxy_ss, cfg)
    mixing = sample_mixing_coeffs(np.random.default_rng(mixing_ss), cfg)

    surface_root = _surface_design_seedseq(design_seed, surface, treatment_type)
    treat_ss, outcome_ss = surface_root.spawn(2)
    treat_theta = sample_treatment_theta_with_accept_reject(treat_ss, cfg, treatment_type=treatment_type)
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
    U = rng.normal(size=(n, 1))
    return C, V, D, U


def sample_covariates(n: int, rng: Generator, cfg: SimConfig) -> np.ndarray:
    return rng.binomial(n=1, p=0.5, size=(n, cfg.dL)).astype(float)


def sample_treatment(
    C: np.ndarray,
    V: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    rng: Generator,
    dgp: DGPDefinition,
    cfg: SimConfig,
) -> np.ndarray:
    tt = dgp.treatment
    betaL = treatment_loading(L, tt)
    linear_index = (
        tt.alpha0
        + C @ tt.alphaC[:, None]
        + L @ tt.alphaL[:, None]
        + np.sum(betaL * V, axis=1, keepdims=True)
        + tt.gamma_u * U
    )

    if dgp.treatment_type == "continuous":
        eps_a = rng.normal(loc=0.0, scale=cfg.sigma_a, size=(C.shape[0], 1))
        A = linear_index + eps_a
        return A.reshape(-1)

    if dgp.treatment_type == "binary":
        eps_a = rng.normal(loc=0.0, scale=1.0, size=(C.shape[0], 1))
        A = (linear_index + eps_a >= 0.0).astype(float)
        return A.reshape(-1)

    raise ValueError(f"Unknown treatment type: {dgp.treatment_type}")


def _baseline_response(L: np.ndarray, outcome_theta: OutcomeTheta) -> np.ndarray:
    return outcome_theta.mu0 + L @ outcome_theta.muL[:, None]


def _threshold_effect(L: np.ndarray, outcome_theta: OutcomeTheta) -> np.ndarray:
    tau0 = float(outcome_theta.params["tau0"])
    tauL = np.asarray(outcome_theta.params["tauL"], dtype=float)
    return tau0 + L @ tauL[:, None]


def _varying_slope_effect(L: np.ndarray, outcome_theta: OutcomeTheta) -> np.ndarray:
    tau0 = float(outcome_theta.params["tau0"])
    tauL = np.asarray(outcome_theta.params["tauL"], dtype=float)
    return tau0 + L @ tauL[:, None]


def _binary_effect(L: np.ndarray, outcome_theta: OutcomeTheta) -> np.ndarray:
    tau0 = float(outcome_theta.params["tau0"])
    tauL = np.asarray(outcome_theta.params["tauL"], dtype=float)
    return tau0 + L @ tauL[:, None]


def structural_response(A: np.ndarray | float, L: np.ndarray, dgp: DGPDefinition) -> np.ndarray:
    outcome_theta = dgp.outcome
    L = np.asarray(L, dtype=float)

    if np.isscalar(A):
        A_arr = np.full((L.shape[0], 1), float(A), dtype=float)
    else:
        A_arr = np.asarray(A, dtype=float).reshape(-1, 1)

    baseline_response = _baseline_response(L, outcome_theta)

    if dgp.treatment_type == "continuous":
        if outcome_theta.surface == "poly":
            beta_a = float(outcome_theta.params["beta_a"])
            quad_a = float(outcome_theta.params["quad_a"])
            val = baseline_response + beta_a * A_arr + quad_a * (A_arr ** 2)
        elif outcome_theta.surface == "tanh":
            amp = float(outcome_theta.params["amp"])
            slope = float(outcome_theta.params["slope"])
            gamma = np.asarray(outcome_theta.params["gamma"], dtype=float)
            phase = slope * A_arr + L @ gamma[:, None]
            val = baseline_response + amp * np.tanh(phase)
        elif outcome_theta.surface == "thresh":
            threshold_effect = _threshold_effect(L, outcome_theta)
            threshold_scale = float(outcome_theta.params["w"])
            val = baseline_response + threshold_effect * (threshold_scale * A_arr >= 0.0).astype(float)
        elif outcome_theta.surface == "varying_slope":
            varying_slope = _varying_slope_effect(L, outcome_theta)
            slope_offset = float(outcome_theta.params["lam"])
            val = baseline_response + (varying_slope - slope_offset) * A_arr
        else:
            raise ValueError(f"Unknown continuous surface family: {outcome_theta.surface}")
    else:
        if outcome_theta.surface == "binary_linear":
            tau = _binary_effect(L, outcome_theta)
            val = baseline_response + tau * A_arr
        elif outcome_theta.surface == "binary_threshold_modulated":
            tau1 = float(outcome_theta.params["tau10"]) + L @ np.asarray(outcome_theta.params["tau1L"], dtype=float)[:, None]
            tau2 = float(outcome_theta.params["tau20"]) + L @ np.asarray(outcome_theta.params["tau2L"], dtype=float)[:, None]
            gamma = np.asarray(outcome_theta.params["gamma"], dtype=float)
            gate = (L @ gamma[:, None] >= 0.0).astype(float)
            val = baseline_response + (tau1 + tau2 * gate) * A_arr
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
    response_signal = structural_response(A, L, dgp)
    nuisance_signal = confounder_nuisance(C, dgp)
    direct_effect_signal = (D @ dgp.outcome.deltaD[:, None]).reshape(-1)
    hidden_confounding_signal = dgp.outcome.deltaU * U.reshape(-1)
    eps_y = rng.normal(loc=0.0, scale=cfg.sigma_y, size=A.shape[0])

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

    return SimulatedDataset(
        X=X,
        M=M,
        L=L,
        A=A,
        Y=Y,
        C=C,
        V=V,
        D=D,
        U=U.reshape(-1),
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
    L_anchor = _anchor_L_values(cfg.dL, cfg.dV)
    beta_anchor = treatment_loading(L_anchor, dgp.treatment)
    beta_norms = np.linalg.norm(beta_anchor, axis=1)
    svals = np.linalg.svd(dgp.treatment.S_beta, compute_uv=False)
    proxy_svals = np.linalg.svd(dgp.proxy_loading, compute_uv=False)

    summary = {
        "n": int(cfg.n),
        "surface": dgp.surface,
        "treatment_type": dgp.treatment_type,
        "design_seed": int(dgp.design_seed),
        "proxy_loading_attempt": int(dgp.proxy_loading_attempt),
        "accepted_attempt_treat": int(dgp.treatment.accepted_attempt),
        "A_mean": float(np.mean(data.A)),
        "A_std": float(np.std(data.A)),
        "Y_mean": float(np.mean(data.Y)),
        "Y_std": float(np.std(data.Y)),
        "min_beta_norm_over_anchor_L": float(beta_norms.min()),
        "min_singular_value_S_beta": float(svals[-1]),
        "min_singular_value_Lambda_C": float(proxy_svals[-1]),
        "mix_linear_abs_max": float(np.max(np.abs(dgp.mixing.a))),
        "mix_quadratic_abs_max": float(np.max(np.abs(dgp.mixing.b))),
        "mix_cubic_abs_max": float(np.max(np.abs(dgp.mixing.c))),
    }

    if dgp.treatment_type == "binary" and dgp.treatment.binary_accept_stats:
        summary.update({
            "binary_signal_ratio": float(dgp.treatment.binary_accept_stats["signal_ratio"]),
            "binary_iv_signal_sd": float(dgp.treatment.binary_accept_stats["iv_signal_sd"]),
            "binary_nuisance_signal_sd": float(dgp.treatment.binary_accept_stats["nuisance_signal_sd"]),
            "binary_global_treated_rate_pilot": float(dgp.treatment.binary_accept_stats["treated_rate_global"]),
            "binary_anchor_treated_rates_pilot": {
                k: float(v) for k, v in dgp.treatment.binary_accept_stats["anchor_treated_rates"].items()
            },
        })

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate the scalar-treatment synthetic NPIV benchmark.")
    parser.add_argument("--treatment-type", type=str, choices=["continuous", "binary"], default="continuous",
                        help="Treatment type.")
    parser.add_argument(
        "--surface",
        dest="surface",
        type=str,
        choices=ALL_SURFACES,
        default="poly",
        help="Outcome-surface family.",
    )
    parser.add_argument("--design-seed", type=int, default=42,
                        help="Seed used to generate the fixed DGP definition for the chosen surface.")
    parser.add_argument("--data-seed", type=int, default=42,
                        help="Seed used to generate one iid dataset from the fixed DGP.")
    parser.add_argument("--n", type=int, default=5000, help="Sample size.")
    parser.add_argument("--dC", type=int, default=3, help="Dimension of latent confounder C.")
    parser.add_argument("--dV", type=int, default=2, help="Dimension of latent IV block V.")
    parser.add_argument("--dD", type=int, default=2, help="Dimension of latent direct-effect block D.")
    parser.add_argument("--dL", type=int, default=None, help="Dimension of observed covariates L; defaults to dV.")
    parser.add_argument("--dM", type=int, default=None, help="Dimension of proxy M; defaults to dC.")
    parser.add_argument("--sigma-m", type=float, default=0.30, help="Proxy noise scale.")
    parser.add_argument("--sigma-a", type=float, default=1.00, help="Continuous-treatment noise scale.")
    parser.add_argument("--sigma-y", type=float, default=1.00, help="Outcome noise scale.")
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
        sigma_m=args.sigma_m,
        sigma_a=args.sigma_a,
        sigma_y=args.sigma_y,
    )

    dgp = build_dgp_definition(args.design_seed, args.surface, cfg, treatment_type=args.treatment_type)
    data = simulate_dataset(args.data_seed, dgp, cfg)
    summary = summarize_dataset(data, dgp, cfg)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
