"""Helpers for repo-local YAML model configuration files."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


IV_REPRESENTATION_MODEL_MODULES = {
    "PIV": "piv",
    "Proxy-Orth": "proxy_orth",
    "Conf-Orth": "conf_orth",
    "ZNet": "znet",
    "AutoIV": "autoiv",
    "VIV": "viv",
    "DIV.VAE": "div_vae",
    "TrueIV": "trueiv",
    "NoneIV": "noneiv",
}

NPIV_ESTIMATOR_MODULES = {
    "deepiv": "deepiv",
    "dfiv": "dfiv",
}


def add_model_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_iv_representation: bool,
    include_npiv_estimators: bool,
) -> None:
    """Register CLI overrides for model configs.

    All overrides default to ``None`` so the config files remain the source of
    truth unless the caller explicitly passes a value.
    """
    if include_iv_representation:
        parser.add_argument("--s1_epochs", type=int, default=None)
        parser.add_argument("--s2_epochs", type=int, default=None)
        parser.add_argument("--s1_patience", type=int, default=None)
        parser.add_argument("--s2_patience", type=int, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--lr", type=float, default=None)
        parser.add_argument("--proxy-embed-dim", type=int, default=None, dest="proxy_embed_dim")

    if include_npiv_estimators:
        parser.add_argument("--npiv-restarts", type=int, default=None, dest="npiv_restarts")
        parser.add_argument(
            "--npiv-normalize-wl",
            action=argparse.BooleanOptionalAction,
            default=None,
            dest="npiv_normalize_wl",
        )
        parser.add_argument("--deepiv-epochs", type=int, default=None, dest="deepiv_epochs")
        parser.add_argument("--deepiv-num-components", type=int, default=None, dest="deepiv_num_components")
        parser.add_argument("--deepiv-treatment-hidden", type=int, default=None, dest="deepiv_treatment_hidden")
        parser.add_argument("--deepiv-response-hidden", type=int, default=None, dest="deepiv_response_hidden")
        parser.add_argument("--deepiv-num-layers", type=int, default=None, dest="deepiv_num_layers")
        parser.add_argument("--deepiv-activation", type=str, default=None, dest="deepiv_activation")
        parser.add_argument("--deepiv-dropout", type=float, default=None, dest="deepiv_dropout")
        parser.add_argument("--deepiv-num-samples", type=int, default=None, dest="deepiv_num_samples")
        parser.add_argument("--deepiv-lr", type=float, default=None, dest="deepiv_lr")
        parser.add_argument("--deepiv-weight-decay", type=float, default=None, dest="deepiv_weight_decay")
        parser.add_argument("--deepiv-min-scale", type=float, default=None, dest="deepiv_min_scale")
        parser.add_argument("--deepiv-val-frac", type=float, default=None, dest="deepiv_val_frac")
        parser.add_argument("--deepiv-patience", type=int, default=None, dest="deepiv_patience")

        parser.add_argument("--dfiv-epochs", type=int, default=None, dest="dfiv_epochs")
        parser.add_argument("--dfiv-feature-dim", type=int, default=None, dest="dfiv_feature_dim")
        parser.add_argument("--dfiv-psi-hidden", type=int, default=None, dest="dfiv_psi_hidden")
        parser.add_argument("--dfiv-phi-hidden", type=int, default=None, dest="dfiv_phi_hidden")
        parser.add_argument("--dfiv-xi-hidden", type=int, default=None, dest="dfiv_xi_hidden")
        parser.add_argument("--dfiv-num-layers", type=int, default=None, dest="dfiv_num_layers")
        parser.add_argument("--dfiv-activation", type=str, default=None, dest="dfiv_activation")
        parser.add_argument("--dfiv-dropout", type=float, default=None, dest="dfiv_dropout")
        parser.add_argument("--dfiv-lambda1", type=float, default=None, dest="dfiv_lambda1")
        parser.add_argument("--dfiv-lambda2", type=float, default=None, dest="dfiv_lambda2")
        parser.add_argument("--dfiv-lr", type=float, default=None, dest="dfiv_lr")
        parser.add_argument("--dfiv-lr-phi", type=float, default=None, dest="dfiv_lr_phi")
        parser.add_argument("--dfiv-batch-size", type=int, default=None, dest="dfiv_batch_size")
        parser.add_argument("--dfiv-weight-decay", type=float, default=None, dest="dfiv_weight_decay")
        parser.add_argument("--dfiv-weight-decay-psi", type=float, default=None, dest="dfiv_weight_decay_psi")
        parser.add_argument("--dfiv-weight-decay-phi", type=float, default=None, dest="dfiv_weight_decay_phi")
        parser.add_argument("--dfiv-val-frac", type=float, default=None, dest="dfiv_val_frac")
        parser.add_argument("--dfiv-patience", type=int, default=None, dest="dfiv_patience")


def load_model_config(
    dataset: str,
    group: str,
    model_name: str,
    *,
    surface: str | None = None,
    treatment_type: str | None = None,
    dA: int = 1,
) -> dict[str, object]:
    """Load one model-config dict from the repo-local ``config`` tree.

    If the YAML file has top-level ``scalar:`` / ``vector:`` blocks the
    appropriate one is selected based on ``dA`` (1 → scalar, >1 → vector).
    Otherwise the file is parsed with the legacy ``treatment_types:`` layout.
    """
    if group == "iv_representation_models":
        module_name = IV_REPRESENTATION_MODEL_MODULES[model_name]
    elif group == "npiv_estimators":
        module_name = NPIV_ESTIMATOR_MODULES[model_name]
    else:
        raise ValueError(f"unknown config group: {group!r}")

    config_path = (
        Path(__file__).resolve().parent
        / dataset
        / group
        / f"{module_name}.yaml"
    )
    if not config_path.exists():
        return {}
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a mapping at the top level")

    # scalar/vector split: select the right top-level block
    if "scalar" in raw or "vector" in raw:
        block_key = "scalar" if int(dA) == 1 else "vector"
        raw = copy.deepcopy(raw.get(block_key) or {})
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: '{block_key}' block must be a mapping")

    treatment_types_block = raw.get("treatment_types") or {}
    if treatment_types_block:
        cfg = copy.deepcopy(treatment_types_block.get("continuous") or {})
        if not isinstance(cfg, dict):
            raise ValueError(f"{config_path}: treatment_types.continuous must be a mapping")
        if treatment_type and treatment_type != "continuous":
            treatment_cfg = treatment_types_block.get(treatment_type, {})
            if treatment_cfg:
                if not isinstance(treatment_cfg, dict):
                    raise ValueError(
                        f"{config_path}: treatment_types.{treatment_type} must be a mapping",
                    )
                cfg.update(copy.deepcopy(treatment_cfg))
    else:
        return copy.deepcopy(raw)

    if surface:
        surface_cfg = (raw.get("surfaces") or {}).get(surface, {})
        if surface_cfg:
            if not isinstance(surface_cfg, dict):
                raise ValueError(
                    f"{config_path}: surfaces.{surface} must be a mapping",
                )
            cfg.update(copy.deepcopy(surface_cfg))
    return cfg


def apply_config_defaults(
    args: argparse.Namespace,
    config: dict[str, object],
) -> argparse.Namespace:
    """Fill ``None`` / missing attrs on ``args`` with values from ``config``."""
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, copy.deepcopy(value))
    return args


def clone_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Copy an ``argparse.Namespace`` so per-model configs can diverge safely."""
    return argparse.Namespace(**vars(args))


def configured_model_args(
    args: argparse.Namespace,
    *,
    dataset: str,
    iv_method: str,
    include_npiv_estimators: tuple[str, ...] = ("deepiv", "dfiv"),
    dA: int = 1,
) -> argparse.Namespace:
    """Return a copy of ``args`` with model-config defaults applied."""
    out = clone_namespace(args)
    surface = getattr(out, "surface", None)
    treatment_type = getattr(out, "treatment_type", None)
    apply_config_defaults(
        out,
        load_model_config(
            dataset,
            "iv_representation_models",
            iv_method,
            treatment_type=treatment_type,
            dA=dA,
        ),
    )
    for estimator in include_npiv_estimators:
        apply_config_defaults(
            out,
            load_model_config(
                dataset,
                "npiv_estimators",
                estimator,
                surface=surface,
                treatment_type=treatment_type,
                dA=dA,
            ),
        )
    return out
