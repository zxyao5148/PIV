"""Semi-synthetic IBM/LBIDD NPIV benchmark.

This simulator uses the real mixed-type LBIDD covariate matrix from the IBM
Causal Inference Benchmarking Framework as the empirical source distribution.
Observed covariates ``L`` are selected directly from the covariate table. The
remaining covariates are split into disjoint blocks and compressed with
blockwise mixed-type VAEs to produce continuous latent blocks ``C``,
``V``, and ``D``. Given those blocks, proxy measurements, nonlinear source
mixtures, treatments, and outcomes follow the synthetic NPIV simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sim_npiv import (  # noqa: E402
    SimConfig,
    SimulatedDataset,
    build_arg_parser as build_npiv_arg_parser,
    build_dgp_definition,
    confounder_nuisance,
    mix_source,
    sample_outcome,
    sample_proxy_measurements,
    sample_treatment,
    summarize_dataset,
)


DEFAULT_COVARIATE_PATH = PROJECT_ROOT / "data" / "data_files" / "lbidd_covariates.csv"
DEFAULT_L_COLUMNS = ("mar", "sex", "rf_cesar")
FeatureKind = Literal["binary", "categorical", "continuous", "count"]


@dataclass
class FeatureSpec:
    name: str
    kind: FeatureKind
    input_index: int
    values: list[float] | None = None
    mean: float = 0.0
    scale: float = 1.0


@dataclass
class IBMBlockMetadata:
    columns: list[str]
    feature_specs: list[FeatureSpec]
    latent_dim: int
    reconstruction_loss: float
    kl_loss: float


@dataclass
class IBMSimulatedDataset(SimulatedDataset):
    D_residualized: np.ndarray
    sample_id: np.ndarray
    L_columns: list[str]
    R_C_columns: list[str]
    R_V_columns: list[str]
    R_D_columns: list[str]


def _parse_column_list(raw: str | None, default: tuple[str, ...]) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(default)
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _resolve_covariate_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"IBM covariate file not found: {p}")
    return p


def load_ibm_covariates(path: str | Path = DEFAULT_COVARIATE_PATH) -> pd.DataFrame:
    """Load the IBM/LBIDD covariate table."""
    return pd.read_csv(_resolve_covariate_path(path))


def sample_empirical_rows(df: pd.DataFrame, n: int, data_seed: int) -> pd.DataFrame:
    """Draw rows from the empirical covariate distribution."""
    rng = np.random.default_rng(int(data_seed))
    replace = int(n) > len(df)
    idx = rng.choice(len(df), size=int(n), replace=replace)
    return df.iloc[idx].reset_index(drop=True)


def _binary_indicator(series: pd.Series) -> np.ndarray:
    """Map common LBIDD binary encodings to 0/1 indicators."""
    x = pd.to_numeric(series, errors="coerce")
    vals = set(x.dropna().unique().tolist())
    if vals <= {-1, 0, 1}:
        return (x.to_numpy(dtype=float) == 1.0).astype(float)
    if vals <= {1, 2, 9}:
        return (x.to_numpy(dtype=float) == 1.0).astype(float)
    if vals <= {1, 2}:
        return (x.to_numpy(dtype=float) == 1.0).astype(float)
    if vals <= {0, 1}:
        return (x.to_numpy(dtype=float) == 1.0).astype(float)
    uniq = sorted(v for v in vals if np.isfinite(v))
    if len(uniq) == 2:
        return (x.to_numpy(dtype=float) == uniq[-1]).astype(float)
    raise ValueError(f"{series.name!r} is not binary-like; unique values: {uniq[:10]}")


def build_observed_covariates(df: pd.DataFrame, l_columns: list[str]) -> np.ndarray:
    missing = [c for c in l_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Requested L columns not found in covariate file: {missing}")
    cols = [_binary_indicator(df[c]) for c in l_columns]
    return np.stack(cols, axis=1).astype(float)


def split_remaining_covariates(
    columns: list[str],
    l_columns: list[str],
    *,
    design_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    remaining = [c for c in columns if c != "sample_id" and c not in set(l_columns)]
    rng = np.random.default_rng(int(design_seed))
    perm = list(rng.permutation(remaining))
    blocks = np.array_split(np.asarray(perm, dtype=object), 3)
    r_c = [str(c) for c in blocks[0].tolist()]
    r_v = [str(c) for c in blocks[1].tolist()]
    r_d = [str(c) for c in blocks[2].tolist()]
    return r_c, r_v, r_d


def _is_integer_like(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.all(np.isclose(finite, np.round(finite))))


def _skewness(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 0.0
    scale = float(np.std(finite))
    if scale <= 1e-12:
        return 0.0
    centered = (finite - float(np.mean(finite))) / scale
    return float(np.mean(centered ** 3))


def infer_feature_kind(
    values: np.ndarray,
    *,
    max_categories: int,
    count_max: int,
    count_skew_threshold: float,
) -> FeatureKind:
    finite = values[np.isfinite(values)]
    uniq = np.unique(finite)
    uniq_set = set(uniq.tolist())
    if len(uniq) <= 2:
        return "binary"
    if uniq_set <= {-1.0, 0.0, 1.0} or uniq_set <= {1.0, 2.0, 9.0}:
        return "binary"
    if _is_integer_like(finite):
        if len(uniq) <= int(max_categories):
            return "categorical"
        is_nonnegative_count = np.min(finite) >= 0.0
        is_bounded_count = np.max(finite) <= float(count_max)
        is_highly_skewed = abs(_skewness(finite)) >= float(count_skew_threshold)
        if is_nonnegative_count and is_bounded_count and is_highly_skewed:
            return "count"
    return "continuous"


def prepare_block_matrix(
    df: pd.DataFrame,
    columns: list[str],
    *,
    max_categories: int,
    count_max: int,
    count_skew_threshold: float,
) -> tuple[np.ndarray, list[FeatureSpec]]:
    """Preprocess mixed covariates before VAE fitting.

    Continuous variables are standardized. Highly skewed nonnegative integer
    counts are transformed with ``log1p`` and standardized. Categorical or
    ordinal variables are encoded by empirical levels observed in the fitting
    sample.
    """
    inputs: list[np.ndarray] = []
    specs: list[FeatureSpec] = []
    for name in columns:
        raw = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        kind = infer_feature_kind(
            raw,
            max_categories=max_categories,
            count_max=count_max,
            count_skew_threshold=count_skew_threshold,
        )
        input_index = len(inputs)

        if kind == "binary":
            vals = np.unique(raw[np.isfinite(raw)])
            if set(vals.tolist()) <= {-1.0, 0.0, 1.0}:
                x = (raw == 1.0).astype(float)
            elif set(vals.tolist()) <= {1.0, 2.0, 9.0}:
                x = (raw == 1.0).astype(float)
            elif set(vals.tolist()) <= {1.0, 2.0}:
                x = (raw == 1.0).astype(float)
            else:
                x = (raw == np.max(vals)).astype(float)
            inputs.append(np.nan_to_num(x, nan=0.0))
            specs.append(FeatureSpec(name=name, kind=kind, input_index=input_index, values=vals.astype(float).tolist()))
        elif kind == "categorical":
            vals = np.unique(raw[np.isfinite(raw)]).astype(float)
            mapping = {float(v): i for i, v in enumerate(vals.tolist())}
            codes = np.asarray([mapping.get(float(v), 0) for v in raw], dtype=float)
            denom = max(len(vals) - 1, 1)
            inputs.append(codes / denom)
            specs.append(FeatureSpec(name=name, kind=kind, input_index=input_index, values=vals.tolist()))
        elif kind == "count":
            clean = np.nan_to_num(raw, nan=0.0)
            clean = np.clip(np.round(clean), 0.0, None)
            trans = np.log1p(clean)
            mean = float(np.mean(trans))
            scale = float(np.std(trans) + 1e-6)
            inputs.append((trans - mean) / scale)
            specs.append(FeatureSpec(name=name, kind=kind, input_index=input_index, mean=mean, scale=scale))
        else:
            clean = raw.copy()
            mean = float(np.nanmean(clean))
            clean = np.nan_to_num(clean, nan=mean)
            scale = float(np.std(clean) + 1e-6)
            inputs.append((clean - mean) / scale)
            specs.append(FeatureSpec(name=name, kind=kind, input_index=input_index, mean=mean, scale=scale))

    if not inputs:
        raise ValueError("Cannot train a VAE on an empty covariate block.")
    return np.stack(inputs, axis=1).astype(np.float32), specs


class MixedTypeVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        specs: list[FeatureSpec],
        *,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(max(1, int(depth))):
            enc_layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU()])
            prev = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(prev, latent_dim)
        self.logvar_head = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for _ in range(max(1, int(depth))):
            dec_layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU()])
            prev = hidden_dim
        self.decoder = nn.Sequential(*dec_layers)

        self.heads = nn.ModuleList()
        for spec in specs:
            if spec.kind == "categorical":
                self.heads.append(nn.Linear(prev, len(spec.values or [])))
            else:
                self.heads.append(nn.Linear(prev, 1))

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h).clamp(min=-12.0, max=12.0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        h = self.decoder(z)
        return mu, logvar, z, [head(h) for head in self.heads]


class MLPResidualizer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for _ in range(max(1, int(depth))):
            layers.extend([nn.Linear(prev, int(hidden_dim)), nn.ReLU()])
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _mixed_reconstruction_loss(
    heads: list[Tensor],
    x: Tensor,
    specs: list[FeatureSpec],
    *,
    count_loss: str,
    poisson_weight: float,
    nb_theta: float,
) -> Tensor:
    losses: list[Tensor] = []
    for head, spec in zip(heads, specs):
        target = x[:, spec.input_index]
        if spec.kind == "binary":
            losses.append(nn.functional.binary_cross_entropy_with_logits(head.squeeze(1), target))
        elif spec.kind == "categorical":
            n_cat = max(len(spec.values or []), 1)
            target_idx = torch.clamp(torch.round(target * max(n_cat - 1, 1)).long(), 0, n_cat - 1)
            losses.append(nn.functional.cross_entropy(head, target_idx))
        elif spec.kind == "count":
            raw_count = torch.expm1(target * spec.scale + spec.mean).clamp_min(0.0)
            log_mu = head.squeeze(1)
            if count_loss == "negative_binomial":
                mu = torch.exp(torch.clamp(log_mu, min=-12.0, max=12.0))
                theta = torch.as_tensor(float(nb_theta), device=mu.device, dtype=mu.dtype).clamp_min(1e-6)
                nb_loss = -(
                    torch.lgamma(raw_count + theta)
                    - torch.lgamma(theta)
                    - torch.lgamma(raw_count + 1.0)
                    + theta * (torch.log(theta) - torch.log(theta + mu))
                    + raw_count * (torch.log(mu + 1e-8) - torch.log(theta + mu))
                ).mean()
                losses.append(float(poisson_weight) * nb_loss)
            else:
                losses.append(
                    float(poisson_weight)
                    * nn.functional.poisson_nll_loss(log_mu, raw_count, log_input=True, full=True)
                )
        else:
            losses.append(nn.functional.mse_loss(head.squeeze(1), target))
    return torch.stack(losses).mean()


def _kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def train_block_vae(
    x: np.ndarray,
    specs: list[FeatureSpec],
    latent_dim: int,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    hidden_dim: int,
    depth: int,
    count_loss: str,
    poisson_weight: float,
    nb_theta: float,
    device: torch.device,
    label: str = "block",
) -> tuple[np.ndarray, MixedTypeVAE, float, float]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = MixedTypeVAE(
        x.shape[1],
        int(latent_dim),
        specs,
        hidden_dim=int(hidden_dim),
        depth=int(depth),
    ).to(device)
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
        generator=generator,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    last_recon = float("nan")
    last_kl = float("nan")
    log_every = max(1, int(epochs) // 10)
    for _ in range(int(epochs)):
        ep = _ + 1
        model.train()
        recon_total = 0.0
        kl_total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            mu, logvar, _, heads = model(xb)
            recon = _mixed_reconstruction_loss(
                heads,
                xb,
                specs,
                count_loss=str(count_loss),
                poisson_weight=float(poisson_weight),
                nb_theta=float(nb_theta),
            )
            kl = _kl_standard_normal(mu, logvar)
            loss = recon + float(beta) * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            recon_total += float(recon.detach().cpu()) * xb.shape[0]
            kl_total += float(kl.detach().cpu()) * xb.shape[0]
        last_recon = recon_total / len(dataset)
        last_kl = kl_total / len(dataset)
        if ep == 1 or ep == int(epochs) or ep % log_every == 0:
            total_loss = last_recon + float(beta) * last_kl
            print(
                f"  [VAE {label}] epoch {ep:3d}/{int(epochs)}  "
                f"recon={last_recon:.4f}  kl={last_kl:.4f}  loss={total_loss:.4f}",
            )

    model.eval()
    latents: list[np.ndarray] = []
    eval_loader = DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=False)
    with torch.no_grad():
        for (xb,) in eval_loader:
            mu, _ = model.encode(xb.to(device))
            latents.append(mu.cpu().numpy())
    z_np = np.concatenate(latents, axis=0).astype(float)
    z_np = (z_np - z_np.mean(axis=0, keepdims=True)) / (z_np.std(axis=0, keepdims=True) + 1e-6)
    return z_np, model, float(last_recon), float(last_kl)


def residualize_against_mlp(
    D: np.ndarray,
    covariates: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    depth: int,
    device: torch.device,
) -> np.ndarray:
    """Return residuals from an MLP projection of D on covariates."""
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x = torch.as_tensor(covariates, dtype=torch.float32)
    y = torch.as_tensor(D, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
        generator=generator,
    )

    model = MLPResidualizer(covariates.shape[1], D.shape[1], hidden_dim, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    log_every = max(1, int(epochs) // 10)
    last_loss = float("nan")
    for ep in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
        last_loss = total / len(dataset)
        if ep == 1 or ep == int(epochs) or ep % log_every == 0:
            print(f"  [D residualizer] epoch {ep:3d}/{int(epochs)}  mse={last_loss:.4f}")

    model.eval()
    preds: list[np.ndarray] = []
    eval_loader = DataLoader(TensorDataset(x), batch_size=min(int(batch_size), len(dataset)), shuffle=False)
    with torch.no_grad():
        for (xb,) in eval_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return D - np.concatenate(preds, axis=0).astype(float)


def build_ibm_latent_blocks(
    covariates: pd.DataFrame,
    *,
    l_columns: list[str],
    cfg: SimConfig,
    design_seed: int,
    autoenc_seed: int,
    ae_epochs: int,
    ae_batch_size: int,
    ae_lr: float,
    vae_beta: float,
    ae_hidden_dim: int,
    ae_depth: int,
    ae_max_categories: int,
    ae_count_max: int,
    ae_count_skew_threshold: float,
    ae_count_loss: str,
    ae_poisson_weight: float,
    ae_nb_theta: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, IBMBlockMetadata]]:
    L = build_observed_covariates(covariates, l_columns)
    r_c, r_v, r_d = split_remaining_covariates(
        list(covariates.columns),
        l_columns,
        design_seed=design_seed,
    )
    block_defs = {
        "C": (r_c, int(cfg.dC)),
        "V": (r_v, int(cfg.dV)),
        "D": (r_d, int(cfg.dD)),
    }

    latents: dict[str, np.ndarray] = {}
    metadata: dict[str, IBMBlockMetadata] = {}
    for offset, (name, (columns, latent_dim)) in enumerate(block_defs.items()):
        x_block, specs = prepare_block_matrix(
            covariates,
            columns,
            max_categories=int(ae_max_categories),
            count_max=int(ae_count_max),
            count_skew_threshold=float(ae_count_skew_threshold),
        )
        z_block, _, recon_loss, kl_loss = train_block_vae(
            x_block,
            specs,
            latent_dim,
            seed=int(autoenc_seed) + offset,
            epochs=int(ae_epochs),
            batch_size=int(ae_batch_size),
            lr=float(ae_lr),
            beta=float(vae_beta),
            hidden_dim=int(ae_hidden_dim),
            depth=int(ae_depth),
            count_loss=str(ae_count_loss),
            poisson_weight=float(ae_poisson_weight),
            nb_theta=float(ae_nb_theta),
            device=device,
            label=name,
        )
        latents[name] = z_block
        metadata[name] = IBMBlockMetadata(
            columns=columns,
            feature_specs=specs,
            latent_dim=int(latent_dim),
            reconstruction_loss=float(recon_loss),
            kl_loss=float(kl_loss),
        )
    return L, latents["C"], latents["V"], latents["D"], metadata


def simulate_ibm_dataset(
    data_seed: int,
    dgp,
    cfg: SimConfig,
    *,
    L: np.ndarray,
    C: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    sample_id: np.ndarray,
    l_columns: list[str],
    block_metadata: dict[str, IBMBlockMetadata],
    residualizer_seed: int,
    residualizer_epochs: int,
    residualizer_batch_size: int,
    residualizer_lr: float,
    residualizer_hidden_dim: int,
    residualizer_depth: int,
    device: torch.device,
) -> IBMSimulatedDataset:
    rng = np.random.default_rng(int(data_seed))
    U = rng.normal(size=(cfg.n, cfg.dA if cfg.dA > 1 else 1))
    D_resid = residualize_against_mlp(
        D,
        np.concatenate([C, V, L], axis=1),
        seed=int(residualizer_seed),
        epochs=int(residualizer_epochs),
        batch_size=int(residualizer_batch_size),
        lr=float(residualizer_lr),
        hidden_dim=int(residualizer_hidden_dim),
        depth=int(residualizer_depth),
        device=device,
    )
    D_resid = (D_resid - D_resid.mean(axis=0, keepdims=True)) / (D_resid.std(axis=0, keepdims=True) + 1e-6)

    X = mix_source(C, V, D, dgp.mixing)
    M = sample_proxy_measurements(C, rng, dgp, cfg)
    A = sample_treatment(C, V, L, U, rng, dgp, cfg)
    Y, response_signal_obs, _, structural_noise_obs = sample_outcome(A, C, D_resid, U, L, rng, dgp, cfg)
    nuisance_signal_obs = confounder_nuisance(C, dgp)
    U_out = U if cfg.dA > 1 else U.reshape(-1)
    return IBMSimulatedDataset(
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
        D_residualized=D_resid,
        sample_id=np.asarray(sample_id),
        L_columns=list(l_columns),
        R_C_columns=block_metadata["C"].columns,
        R_V_columns=block_metadata["V"].columns,
        R_D_columns=block_metadata["D"].columns,
    )


def summarize_ibm_dataset(
    data: IBMSimulatedDataset,
    dgp,
    cfg: SimConfig,
    block_metadata: dict[str, IBMBlockMetadata],
) -> dict[str, object]:
    summary = summarize_dataset(data, dgp, cfg)
    summary.update(
        {
            "L_columns": data.L_columns,
            "vae_reconstruction_loss": {
                name: meta.reconstruction_loss for name, meta in block_metadata.items()
            },
            "vae_kl_loss": {
                name: meta.kl_loss for name, meta in block_metadata.items()
            },
            "R_block_sizes": {
                "C": len(data.R_C_columns),
                "V": len(data.R_V_columns),
                "D": len(data.R_D_columns),
            },
            "max_abs_corr_D_resid_with_CVL": float(
                np.max(
                    np.abs(
                        np.corrcoef(
                            np.concatenate([data.D_residualized, data.C, data.V, data.L], axis=1),
                            rowvar=False,
                        )[: data.D_residualized.shape[1], data.D_residualized.shape[1] :]
                    )
                )
            ),
        }
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_npiv_arg_parser()
    parser.description = "Construct a semi-synthetic IBM/LBIDD NPIV benchmark."
    parser.add_argument(
        "--covariate-path",
        type=str,
        default=str(DEFAULT_COVARIATE_PATH.relative_to(PROJECT_ROOT)),
        help="Path to the IBM/LBIDD covariate CSV.",
    )
    parser.add_argument(
        "--l-columns",
        type=str,
        default=",".join(DEFAULT_L_COLUMNS),
        help="Comma-separated covariate names to use as observed L.",
    )
    parser.add_argument("--autoenc-seed", type=int, default=42, help="Seed for VAE compression training.")
    parser.add_argument("--vae-epochs", "--ae-epochs", dest="ae_epochs", type=int, default=40, help="VAE training epochs per block.")
    parser.add_argument("--vae-batch-size", "--ae-batch-size", dest="ae_batch_size", type=int, default=512, help="VAE batch size.")
    parser.add_argument("--vae-lr", "--ae-lr", dest="ae_lr", type=float, default=1e-3, help="VAE learning rate.")
    parser.add_argument("--vae-beta", type=float, default=1.0, help="KL weight for the mixed-type VAE objective.")
    parser.add_argument("--vae-hidden-dim", "--ae-hidden-dim", dest="ae_hidden_dim", type=int, default=128, help="VAE hidden width.")
    parser.add_argument("--vae-depth", "--ae-depth", dest="ae_depth", type=int, default=2, help="VAE encoder/decoder depth.")
    parser.add_argument("--vae-max-categories", "--ae-max-categories", dest="ae_max_categories", type=int, default=20, help="Max unique integer values treated as categorical.")
    parser.add_argument("--vae-count-max", "--ae-count-max", dest="ae_count_max", type=int, default=100, help="Max value for integer count features.")
    parser.add_argument(
        "--vae-count-skew-threshold",
        type=float,
        default=1.0,
        help="Absolute skewness above which nonnegative integer variables are treated as counts.",
    )
    parser.add_argument(
        "--vae-count-loss",
        "--ae-count-loss",
        dest="ae_count_loss",
        type=str,
        choices=["poisson", "negative_binomial"],
        default="poisson",
        help="Likelihood used for integer count reconstruction heads.",
    )
    parser.add_argument("--vae-count-weight", "--ae-poisson-weight", dest="ae_poisson_weight", type=float, default=0.25, help="Relative weight for count losses.")
    parser.add_argument("--vae-nb-theta", "--ae-nb-theta", dest="ae_nb_theta", type=float, default=5.0, help="Negative-binomial dispersion for count heads.")
    parser.add_argument("--resid-epochs", type=int, default=200, help="Epochs for the MLP D residualizer.")
    parser.add_argument("--resid-batch-size", type=int, default=512, help="Batch size for the MLP D residualizer.")
    parser.add_argument("--resid-lr", type=float, default=1e-3, help="Learning rate for the MLP D residualizer.")
    parser.add_argument("--resid-hidden-dim", type=int, default=128, help="Hidden width for the MLP D residualizer.")
    parser.add_argument("--resid-depth", type=int, default=2, help="Depth for the MLP D residualizer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    l_columns = _parse_column_list(args.l_columns, DEFAULT_L_COLUMNS)

    cfg = SimConfig(
        n=args.n,
        dC=args.dC,
        dV=args.dV,
        dD=args.dD,
        dL=len(l_columns) if args.dL is None else args.dL,
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
    if cfg.dL != len(l_columns):
        raise ValueError(f"dL={cfg.dL} but --l-columns has {len(l_columns)} columns.")

    covariates = sample_empirical_rows(
        load_ibm_covariates(args.covariate_path),
        n=cfg.n,
        data_seed=args.data_seed,
    )
    device = torch.device(args.device)
    L, C, V, D, metadata = build_ibm_latent_blocks(
        covariates,
        l_columns=l_columns,
        cfg=cfg,
        design_seed=args.design_seed,
        autoenc_seed=args.autoenc_seed,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_lr=args.ae_lr,
        vae_beta=args.vae_beta,
        ae_hidden_dim=args.ae_hidden_dim,
        ae_depth=args.ae_depth,
        ae_max_categories=args.ae_max_categories,
        ae_count_max=args.ae_count_max,
        ae_count_skew_threshold=args.vae_count_skew_threshold,
        ae_count_loss=args.ae_count_loss,
        ae_poisson_weight=args.ae_poisson_weight,
        ae_nb_theta=args.ae_nb_theta,
        device=device,
    )
    dgp = build_dgp_definition(args.design_seed, args.surface, cfg, treatment_type=args.treatment_type)
    data = simulate_ibm_dataset(
        args.data_seed,
        dgp,
        cfg,
        L=L,
        C=C,
        V=V,
        D=D,
        sample_id=covariates["sample_id"].to_numpy() if "sample_id" in covariates else np.arange(cfg.n),
        l_columns=l_columns,
        block_metadata=metadata,
        residualizer_seed=int(args.autoenc_seed) + 10_000,
        residualizer_epochs=args.resid_epochs,
        residualizer_batch_size=args.resid_batch_size,
        residualizer_lr=args.resid_lr,
        residualizer_hidden_dim=args.resid_hidden_dim,
        residualizer_depth=args.resid_depth,
        device=device,
    )
    print(json.dumps(summarize_ibm_dataset(data, dgp, cfg, metadata), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
