"""Synthetic airline-demands benchmark.

Causal structure
----------------
Observed covariates : L1 (time-of-year), L2 (customer price-sensitivity level)
Latent variables    : C (demand-side confounder), V (fuel-price IV),
                      D (direct demand factor), U (endogenous demand shock)
Treatment           : A  (ticket price)
Outcome             : Y  (airline demand)

Mixed source X = (X1, X2, X3) = f(C, V, D) is a nonlinear differmorphism map.
Proxy M is a noisy observation of C.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


# ── Config ────────────────────────────────────────────────────────

@dataclass
class DemandsConfig:
    n: int = 8_000
    seed: int = 42
    rho: float = 0.85       # corr(eps_A, U); →1 makes U almost pure treatment-shock noise
    # Scale of treatment equation shock eps_A (same draw that enters U via rho).
    # >1 amplifies how much A moves with the shock that is correlated with U.
    coef_eps_a: float = 1.5
    sigma_m: float = 0.5    # proxy noise scale
    sigma_x: float = 0.0    # optional isotropic noise on X after f (0 = deterministic f)
    # Strength of instrument V in treatment equation.
    coef_v_treatment: float = 1.0
    # Strength of confounder C and direct factor D in structural equations.
    coef_c_treatment: float = 1.0   # C enters A = … + coef_c_treatment * C + …
    coef_c_outcome: float = 1.0     # C enters Y = … + coef_c_outcome * C + …
    coef_d_outcome: float = 1.0     # D enters Y = … + coef_d_outcome * D + …
    coef_u_outcome: float = 1.0     # U enters Y = … + coef_u_outcome * U (was implicit 1)


# ── Seasonal effect ───────────────────────────────────────────────

def kappa(L1: np.ndarray) -> np.ndarray:
    """Nonlinear seasonal effect kappa(L1)."""
    x = np.asarray(L1, dtype=np.float64)
    return (2.0 * (
        (x - 5.0) ** 4 / 600.0
        + np.exp(-4.0 * (x - 5.0) ** 2)
        + x / 10.0
        - 2.0
    )).astype(np.float32)


# ── Ground-truth structural response ─────────────────────────────

def structural_response_demands(
    a: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
) -> np.ndarray:
    """True interventional structural response h0(a, L1, L2).

    Under do(A = a), C, D, U are mean-zero and independent of the
    intervention, so:
        h0(a, L1, L2) = 100 + (10 + a) * L2 * kappa(L1) - 2 * a
    """
    a = np.asarray(a, dtype=np.float64)
    L1 = np.asarray(L1, dtype=np.float64)
    L2 = np.asarray(L2, dtype=np.float64)
    return (100.0 + (10.0 + a) * L2 * kappa(L1) - 2.0 * a).astype(np.float32)


def true_marginal_effect_demands(L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """Interventional slope ∂h0/∂a = L2 * kappa(L1) - 2 (does not depend on C, D, U)."""
    L1 = np.asarray(L1, dtype=np.float64)
    L2 = np.asarray(L2, dtype=np.float64)
    kap = np.asarray(kappa(L1), dtype=np.float64)
    return (L2 * kap - 2.0).astype(np.float32)


# ── Diffeomorphic mixing X = f(C, V, D) ───

def diffeomorphic_mix_demands(
    C: np.ndarray, V: np.ndarray, D: np.ndarray,
) -> np.ndarray:
    """
    Map (C, V, D) in R^3 to X in R^3.

    Harder triangular polynomial diffeomorphism:

    * ``z1 = C + 0.15 * C³``;
    * ``z2 = (0.6 + 0.2 * tanh(C)) * V + 0.35 * C + 0.10 * C²``;
    * ``z3 = (0.7 + 0.15 * tanh(C + V)) * D + 0.25 * V + 0.20 * C * V``;

    then ``X = [z1, z2, z3] @ A.T`` with fixed full-rank ``A``.
    """
    c = np.asarray(C, dtype=np.float64).reshape(-1)
    v = np.asarray(V, dtype=np.float64).reshape(-1)
    d = np.asarray(D, dtype=np.float64).reshape(-1)

    z1 = c + 0.15 * (c ** 3)
    z2 = (0.6 + 0.2 * np.tanh(c)) * v + 0.35 * c + 0.10 * (c ** 2)
    z3 = (0.7 + 0.15 * np.tanh(c + v)) * d + 0.25 * v + 0.20 * c * v

    Z = np.column_stack([z1, z2, z3])

    A = np.array(
        [
            [0.85, 0.45, -0.25],
            [-0.35, 0.88, 0.18],
            [0.22, -0.15, 0.92],
        ],
        dtype=np.float64,
    )
    return (Z @ A.T).astype(np.float32)


# ── Nonlinear oracle instruments Z = g(·) for ablations ───────────
# Fixed polynomial maps (diffeomorphisms on their domains) used in place of
# raw concatenations [V], [V,D], [C,V,D] in scripts/train_demands.py.


def oracle_z_nonlinear_v(V: np.ndarray) -> np.ndarray:
    """
    Z in R^1 from scalar V: cubic strictly monotone map (global diffeo R→R).
    """
    v = np.asarray(V, dtype=np.float64).reshape(-1)
    z = v + 0.06 * (v ** 3)
    return z.reshape(-1, 1).astype(np.float32)


def oracle_z_nonlinear_vd(V: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Z in R^2 from (V, D): triangular polynomial map + invertible 2×2 mix.
    """
    v = np.asarray(V, dtype=np.float64).reshape(-1)
    d = np.asarray(D, dtype=np.float64).reshape(-1)
    t1 = v + 0.05 * (v ** 3)
    t2 = d + 0.15 * v + 0.08 * (v ** 2)
    Z = np.column_stack([t1, t2])
    B = np.array(
        [[0.88, 0.42], [-0.28, 0.90]],
        dtype=np.float64,
    )
    return (Z @ B.T).astype(np.float32)


def oracle_z_nonlinear_cvd(C: np.ndarray, V: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Z in R^3 from (C, V, D): polynomial triangular map (coefficients chosen so
    it differs from ``diffeomorphic_mix_demands`` / observed X).
    """
    c = np.asarray(C, dtype=np.float64).reshape(-1)
    v = np.asarray(V, dtype=np.float64).reshape(-1)
    d = np.asarray(D, dtype=np.float64).reshape(-1)
    z1 = c + 0.04 * (c ** 3)
    z2 = v + 0.20 * c + 0.08 * (c ** 2)
    z3 = d + 0.06 * (v ** 2) + 0.10 * c
    Z = np.column_stack([z1, z2, z3])
    B = np.array(
        [
            [0.78, 0.52, -0.22],
            [-0.32, 0.86, 0.20],
            [0.18, -0.18, 0.88],
        ],
        dtype=np.float64,
    )
    return (Z @ B.T).astype(np.float32)


# ── Simulator ─────────────────────────────────────────────────────

def simulate_demands(cfg: DemandsConfig | None = None) -> Dict[str, np.ndarray]:
    """Generate one realisation of the airline-demands synthetic benchmark.

    Returns a dict with keys:
        C, V, D, U          – latent variables (each [N, 1])
        L                   – observed covariates [N, 2]  (L1, L2)
        M                   – proxy for C [N, 1]
        X                   – mixed source [N, 3]
        A                   – treatment (ticket price) [N, 1]
        Y                   – outcome (demand) [N, 1]
        h0                  – true structural response at observed A [N, 1]
    """
    cfg = cfg or DemandsConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n

    # ── Observed covariates ───────────────────────────────────────
    L1 = rng.uniform(0.0, 10.0, n).astype(np.float32)
    # L2 ∈ {1, 2, …, 7}  (discrete price-sensitivity level)
    L2 = rng.integers(1, 8, n).astype(np.float32)

    # ── Latent variables ──────────────────────────────────────────
    C = rng.standard_normal(n).astype(np.float32)
    V = rng.standard_normal(n).astype(np.float32)
    D = rng.standard_normal(n).astype(np.float32)

    # U = rho * eps_A + sqrt(1 - rho^2) * xi  (correlated with treatment noise)
    eps_A = rng.standard_normal(n).astype(np.float32)
    xi = rng.standard_normal(n).astype(np.float32)
    rho = float(cfg.rho)
    U = (rho * eps_A + np.sqrt(1.0 - rho ** 2) * xi).astype(np.float32)

    # ── Seasonal effect ───────────────────────────────────────────
    kap = kappa(L1)  # shape [N]

    # ── Treatment: A = … + coef_c_treatment * C + coef_eps_a * eps_A
    # Larger coef_eps_a × rho jointly strengthen endogeneity via the U–eps_A channel.
    cc = float(cfg.coef_c_treatment)
    ea = float(cfg.coef_eps_a)
    cv = float(cfg.coef_v_treatment)
    A = (25.0 + cv * (V + 3.0) * kap + cc * C + ea * eps_A).astype(np.float32)

    # ── Outcome: Y = … + coef_c_outcome * C + coef_d_outcome * D + coef_u_outcome * U
    cy = float(cfg.coef_c_outcome)
    dy = float(cfg.coef_d_outcome)
    uy = float(cfg.coef_u_outcome)
    Y = (
        100.0 + (10.0 + A) * L2 * kap - 2.0 * A + cy * C + dy * D + uy * U
    ).astype(np.float32)

    # ── True structural response h0(A, L1, L2) ───────────────────
    h0 = structural_response_demands(A, L1, L2)

    # ── Proxy M = C + eta, eta ~ N(0, sigma_M^2) ─────────────────
    eta = (cfg.sigma_m * rng.standard_normal(n)).astype(np.float32)
    M = (C + eta).astype(np.float32)

    # ── Mixed source X = f(C, V, D) (harder nonlinear diffeomorphic mix) ──
    X = diffeomorphic_mix_demands(C, V, D)
    if cfg.sigma_x > 0.0:
        X = (
            X + (cfg.sigma_x * rng.standard_normal(X.shape)).astype(np.float32)
        )

    return {
        "C": C.reshape(-1, 1),
        "V": V.reshape(-1, 1),
        "D": D.reshape(-1, 1),
        "U": U.reshape(-1, 1),
        "L": np.column_stack([L1, L2]).astype(np.float32),
        "M": M.reshape(-1, 1),
        "X": X,
        "A": A.reshape(-1, 1),
        "Y": Y.reshape(-1, 1),
        "h0": h0.reshape(-1, 1),
    }


__all__ = [
    "DemandsConfig",
    "diffeomorphic_mix_demands",
    "oracle_z_nonlinear_v",
    "oracle_z_nonlinear_vd",
    "oracle_z_nonlinear_cvd",
    "kappa",
    "simulate_demands",
    "structural_response_demands",
    "true_marginal_effect_demands",
]
