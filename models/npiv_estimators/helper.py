"""Shared building blocks for ``npiv_estimators`` (DeepIV and DFIV).

Contains:

* ``_ensure_2d``        — lift shape ``[B]`` inputs to ``[B, 1]``.
* ``get_activation``    — string → ``nn.Module`` factory.
* :class:`MLP`          — plain feed-forward block used throughout.
* :func:`concat_context`/ :func:`concat_wl` — small helpers to glue
  ``(z, w, l)`` or ``(w, l)`` into a single input tensor.
* :class:`DeepIVTreatmentOutput`, :class:`NPIVOutcomeOutput` — light
  dataclasses used by the DeepIV training loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn


# ── Tensor helpers ──────────────────────────────────────────────────

def _ensure_2d(t: Tensor) -> Tensor:
    """``[B] → [B, 1]``; ``[B, D]`` passes through unchanged."""
    return t.unsqueeze(-1) if t.ndim == 1 else t


def concat_wl(w: Tensor, l: Tensor) -> Tensor:
    """Concatenate ``(W, L)`` along the last dim."""
    return torch.cat([w, l], dim=-1)


def concat_context(z: Tensor, w: Tensor, l: Tensor) -> Tensor:
    """Concatenate ``(Z, W, L)`` along the last dim."""
    return torch.cat([z, w, l], dim=-1)


# ── Activations / MLP ───────────────────────────────────────────────

def get_activation(name: str) -> nn.Module:
    """String → PyTorch activation module.

    Supported: ``relu|leaky_relu|gelu|elu|silu|tanh``.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Feed-forward block ``Linear → Activation → … → Linear``.

    Set ``activate_last=True`` when a representation-like output is
    desired (final layer followed by the same non-linearity used between
    hidden layers).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        activate_last: bool = False,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers: List[nn.Module] = []
        act = get_activation(activation)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or activate_last:
                layers.append(act.__class__())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class DeepIVTreatmentOutput:
    """Return value of the DeepIV stage-1 treatment model's ``nll`` call."""

    nll: Tensor
    mean: Tensor
    samples: Optional[Tensor] = None
    extra: Dict[str, Tensor] = field(default_factory=dict)


@dataclass
class NPIVOutcomeOutput:
    """Return value of an NPIV estimator's outcome-loss call (shared schema)."""

    loss: Tensor
    metrics: Dict[str, Tensor] = field(default_factory=dict)


__all__ = [
    "DeepIVTreatmentOutput",
    "MLP",
    "NPIVOutcomeOutput",
    "_ensure_2d",
    "concat_context",
    "concat_wl",
    "get_activation",
]
