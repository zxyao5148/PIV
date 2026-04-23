from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np


Geometry = Literal[
    "half_moons", "spiral", "circles", "gmm", "pinwheel",
    "swiss_roll", "eight_gaussians", "checkerboard",
]


# ---------------------------------------------------------------------
# 2D geometry samplers for latent V
# ---------------------------------------------------------------------


def sample_half_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    n_upper = n // 2
    n_lower = n - n_upper

    theta_u = np.pi * rng.uniform(0.0, 1.0, n_upper)
    theta_l = np.pi * rng.uniform(0.0, 1.0, n_lower)

    upper = np.column_stack([np.cos(theta_u), np.sin(theta_u)])
    lower = np.column_stack([1.0 - np.cos(theta_l), 1.0 - np.sin(theta_l) - 0.5])

    pts = np.vstack([upper, lower])
    pts += 0.05 * rng.standard_normal(pts.shape)
    return pts.astype(np.float32)


def sample_spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    n_half = n // 2

    t0 = np.linspace(0.5, 3.0, n_half) + 0.08 * rng.standard_normal(n_half)
    t1 = np.linspace(0.5, 3.0, n - n_half) + 0.08 * rng.standard_normal(n - n_half)

    arm0 = np.column_stack([t0 * np.cos(2.0 * t0), t0 * np.sin(2.0 * t0)])
    arm1 = np.column_stack([-t1 * np.cos(2.0 * t1), -t1 * np.sin(2.0 * t1)])

    pts = np.vstack([arm0, arm1])
    return pts.astype(np.float32)


def sample_circles(n: int, rng: np.random.Generator) -> np.ndarray:
    n_inner = n // 2
    n_outer = n - n_inner

    theta_in = 2.0 * np.pi * rng.uniform(0.0, 1.0, n_inner)
    theta_out = 2.0 * np.pi * rng.uniform(0.0, 1.0, n_outer)

    r_in = 0.9 + 0.06 * rng.standard_normal(n_inner)
    r_out = 2.1 + 0.06 * rng.standard_normal(n_outer)

    inner = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])
    outer = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])

    pts = np.vstack([inner, outer])
    return pts.astype(np.float32)


def sample_gmm(n: int, rng: np.random.Generator, k: int = 5) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    centers = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])

    labels = rng.integers(0, k, size=n)
    pts = centers[labels] + 0.30 * rng.standard_normal((n, 2))
    return pts.astype(np.float32)


def sample_pinwheel(n: int, rng: np.random.Generator, k: int = 5) -> np.ndarray:
    n_per = n // k
    remainder = n - n_per * k

    pieces = []
    for i in range(k):
        ni = n_per + (1 if i < remainder else 0)
        r = np.linspace(0.2, 2.0, ni) + 0.08 * rng.standard_normal(ni)
        theta_base = 0.7 * r
        theta = theta_base + 2.0 * np.pi * i / k
        pts_i = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        pieces.append(pts_i)

    pts = np.vstack(pieces)
    pts += 0.05 * rng.standard_normal(pts.shape)
    return pts.astype(np.float32)


def sample_swiss_roll(n: int, rng: np.random.Generator) -> np.ndarray:
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(0.0, 1.0, n))
    x = t * np.cos(t)
    y = t * np.sin(t)
    pts = np.column_stack([x, y]).astype(np.float32)
    pts += 0.08 * rng.standard_normal(pts.shape).astype(np.float32)
    return pts


def sample_eight_gaussians(n: int, rng: np.random.Generator) -> np.ndarray:
    k = 8
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    centers = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])

    labels = rng.integers(0, k, size=n)
    pts = centers[labels] + 0.20 * rng.standard_normal((n, 2))
    return pts.astype(np.float32)


def sample_checkerboard(n: int, rng: np.random.Generator, k: int = 4) -> np.ndarray:
    pieces = []
    per = n // (k * k // 2)
    remainder = n - per * (k * k // 2)

    idx = 0
    for i in range(k):
        for j in range(k):
            if (i + j) % 2 == 0:
                ni = per + (1 if idx < remainder else 0)
                x = rng.uniform(i, i + 1, ni)
                y = rng.uniform(j, j + 1, ni)
                pieces.append(np.column_stack([x, y]))
                idx += 1

    pts = np.vstack(pieces).astype(np.float32)
    pts -= np.array([k / 2.0, k / 2.0], dtype=np.float32)
    return pts


GEOMETRY_SAMPLERS = {
    "half_moons": sample_half_moons,
    "spiral": sample_spiral,
    "circles": sample_circles,
    "gmm": sample_gmm,
    "pinwheel": sample_pinwheel,
    "swiss_roll": sample_swiss_roll,
    "eight_gaussians": sample_eight_gaussians,
    "checkerboard": sample_checkerboard,
}


# ---------------------------------------------------------------------
# Source X = f(C, V, D)
# ---------------------------------------------------------------------


def diffeomorphic_mix(
    C: np.ndarray, V: np.ndarray, D: np.ndarray,
) -> np.ndarray:
    """
    X = f(C, V, D) in R^4 via triangular nonlinear map + invertible linear
    mix.  D is a scalar direct-effect nuisance that enters X but does not
    affect A or Y through the instrument channel.
    """
    c = C.reshape(-1)
    v1 = V[:, 0]
    v2 = V[:, 1]
    d = D.reshape(-1)

    z1 = c + 0.4 * np.tanh(c)
    z2 = v1 + 0.3 * np.sin(2.0 * c)
    z3 = v2 + 0.2 * np.tanh(v1) + 0.25 * c ** 2
    z4 = d + 0.15 * np.sin(v1 + c)

    Z = np.column_stack([z1, z2, z3, z4]).astype(np.float32)

    A = np.array(
        [
            [0.80,  0.50, -0.30,  0.10],
            [-0.40,  0.90,  0.20, -0.15],
            [0.30, -0.20,  0.85,  0.05],
            [0.10,  0.15, -0.10,  0.90],
        ],
        dtype=np.float32,
    )
    return (Z @ A.T).astype(np.float32)


def generate_proxy(
    C: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.25,
) -> np.ndarray:
    """
    Proxy channel: M | C ~ N(eta(C), sigma^2), eta(c) = c + 0.3 sin(2c).
    This matches the stage-1 proxy-only dependence on C.
    """
    c = C.reshape(-1)
    eta = c + 0.3 * np.sin(2.0 * c)
    M = eta + sigma * rng.standard_normal(size=eta.shape)
    return M.astype(np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------
# Treatment / outcome
# ---------------------------------------------------------------------


def phi_v(V: np.ndarray) -> np.ndarray:
    """
    Simple injective stage-2 feature map. Since V is already 2D, use phi(V)=V.
    """
    return V.astype(np.float32)


def alpha_of_c(C: np.ndarray) -> np.ndarray:
    """
    Baseline alpha(C) in R^2.
    """
    c = C[:, 0]
    alpha1 = 0.8 * c + 0.25 * np.sin(c)
    alpha2 = -0.6 * c + 0.20 * np.tanh(1.5 * c)
    return np.column_stack([alpha1, alpha2]).astype(np.float32)


def loading_matrix_of_c(C: np.ndarray) -> np.ndarray:
    """
    Smooth full-rank B(C) in R^{2x2} for every sample.

    The matrix is diagonally dominant, so it stays invertible across C.
    """
    c = C[:, 0]

    b11 = 1.10 + 0.12 * np.tanh(c)
    b12 = 0.20 + 0.08 * np.sin(c)
    b21 = -0.15 + 0.06 * np.cos(c)
    b22 = 0.95 + 0.10 * np.tanh(0.5 * c)

    B = np.zeros((len(C), 2, 2), dtype=np.float32)
    B[:, 0, 0] = b11
    B[:, 0, 1] = b12
    B[:, 1, 0] = b21
    B[:, 1, 1] = b22
    return B


def treatment_mean(C: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Mean treatment under vector-continuous stage-2 model:
        E[A | C, V] = alpha(C) + B(C) phi(V)
    with d_A = d_V = 2.
    """
    alpha = alpha_of_c(C)          # [N, 2]
    B = loading_matrix_of_c(C)     # [N, 2, 2]
    phi = phi_v(V)                 # [N, 2]
    iv_term = np.einsum("nij,nj->ni", B, phi).astype(np.float32)
    return (alpha + iv_term).astype(np.float32)


def generate_treatment(
    C: np.ndarray,
    V: np.ndarray,
    U: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float = 0.25,
    gamma_u: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Continuous 2D treatment:
        A = alpha(C) + B(C) phi(V) + gamma_u * U + eps_A
    where eps_A ~ N(0, sigma_A^2 I_2) and U is the hidden confounder.
    """
    if gamma_u is None:
        gamma_u = np.array([0.5, -0.35], dtype=np.float32)

    mean = treatment_mean(C, V)
    u_effect = U.reshape(-1, 1) * gamma_u.reshape(1, -1)
    eps = noise_scale * rng.standard_normal(size=mean.shape).astype(np.float32)
    A = (mean + u_effect + eps).astype(np.float32)

    alpha = alpha_of_c(C)
    B = loading_matrix_of_c(C)
    return {
        "A": A,
        "A_mean": mean.astype(np.float32),
        "alpha_C": alpha.astype(np.float32),
        "B_C": B.astype(np.float32),
        "phi_V": phi_v(V).astype(np.float32),
    }


def structural_response(A: np.ndarray) -> np.ndarray:
    """
    Structural response h0(A) for Y = h0(A) + m0(C) + eps_Y.
    """
    a1 = A[:, 0]
    a2 = A[:, 1]
    h = 1.6 * np.tanh(a1) + 0.9 * np.sin(a2) + 0.35 * a1 * a2
    return h.astype(np.float32).reshape(-1, 1)


def nuisance_m0(C: np.ndarray) -> np.ndarray:
    """
    Source-side confounding term m0(C).
    """
    c = C[:, 0]
    m = 0.85 * c + 0.25 * np.sin(1.8 * c)
    return m.astype(np.float32).reshape(-1, 1)


def generate_outcome(
    C: np.ndarray,
    A: np.ndarray,
    U: np.ndarray,
    D: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float = 0.20,
    delta_u: float = 0.6,
    eta_d: float = 0.45,
) -> Dict[str, np.ndarray]:
    """
    Outcome:
        Y = h0(A) + m0(C) + delta_u * U + eta_d * D + eps_Y
    where U is the hidden confounder, D is the direct effect, and eps_Y is
    independent noise.
    """
    h0 = structural_response(A)
    m0 = nuisance_m0(C)
    u_effect = (delta_u * U.reshape(-1, 1)).astype(np.float32)
    d_effect = (eta_d * D.reshape(-1, 1)).astype(np.float32)
    eps_y = noise_scale * rng.standard_normal(size=(len(C), 1)).astype(np.float32)
    Y = (h0 + m0 + u_effect + d_effect + eps_y).astype(np.float32)

    return {
        "Y": Y,
        "h0": h0.astype(np.float32),
        "m0": m0.astype(np.float32),
    }


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class GeomVConfig:
    n: int = 8000
    geometry: Geometry = "pinwheel"
    seed: int = 42
    proxy_sigma: float = 0.25
    treatment_noise_scale: float = 0.25
    outcome_noise_scale: float = 0.20


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def sample_latents(cfg: GeomVConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    C = rng.standard_normal(cfg.n).astype(np.float32).reshape(-1, 1)
    V = GEOMETRY_SAMPLERS[cfg.geometry](cfg.n, rng).astype(np.float32)
    U = rng.standard_normal(cfg.n).astype(np.float32).reshape(-1, 1)
    D = rng.standard_normal(cfg.n).astype(np.float32).reshape(-1, 1)

    return {"C": C, "V": V, "U": U, "D": D}


def simulate_geom_v(cfg: GeomVConfig | None = None) -> Dict[str, np.ndarray]:
    """
    Synthetic benchmark with keys:
        C, V, U, D, X, M, A, Y
    plus helpful latent diagnostics:
        phi_V, alpha_C, B_C, A_mean, h0, m0

    Causal structure:
        C  → X, M, A, Y        (observed confounder)
        V  → X, A              (latent instrument, geometric)
        U  → A, Y              (hidden unmeasured confounder)
        D  → X                 (direct-effect nuisance)
        X  = f(C, V, D) ∈ R⁴
    """
    cfg = cfg or GeomVConfig()
    rng = np.random.default_rng(cfg.seed)

    latents = sample_latents(cfg)
    C = latents["C"]
    V = latents["V"]
    U = latents["U"]
    D = latents["D"]

    X = diffeomorphic_mix(C, V, D)
    M = generate_proxy(C, rng, sigma=cfg.proxy_sigma)

    treatment_dict = generate_treatment(
        C=C,
        V=V,
        U=U,
        rng=rng,
        noise_scale=cfg.treatment_noise_scale,
    )
    outcome_dict = generate_outcome(
        C=C,
        A=treatment_dict["A"],
        U=U,
        D=D,
        rng=rng,
        noise_scale=cfg.outcome_noise_scale,
    )

    out = {
        "C": C.astype(np.float32),
        "V": V.astype(np.float32),
        "U": U.astype(np.float32),
        "D": D.astype(np.float32),
        "X": X.astype(np.float32),
        "M": M.astype(np.float32),
        "A": treatment_dict["A"].astype(np.float32),
        "Y": outcome_dict["Y"].astype(np.float32),
        "phi_V": treatment_dict["phi_V"].astype(np.float32),
        "alpha_C": treatment_dict["alpha_C"].astype(np.float32),
        "B_C": treatment_dict["B_C"].astype(np.float32),
        "A_mean": treatment_dict["A_mean"].astype(np.float32),
        "h0": outcome_dict["h0"].astype(np.float32),
        "m0": outcome_dict["m0"].astype(np.float32),
    }

    det_B = np.linalg.det(out["B_C"])
    if np.any(np.abs(det_B) < 1e-3):
        raise RuntimeError("Encountered nearly singular B(C). Adjust loading_matrix_of_c.")

    return out


__all__ = [
    "GEOMETRY_SAMPLERS",
    "GeomVConfig",
    "Geometry",
    "alpha_of_c",
    "diffeomorphic_mix",
    "generate_outcome",
    "generate_proxy",
    "generate_treatment",
    "loading_matrix_of_c",
    "nuisance_m0",
    "phi_v",
    "sample_latents",
    "simulate_geom_v",
    "structural_response",
    "treatment_mean",
]