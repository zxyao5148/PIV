"""Shared contract for downstream NPIV estimators (DeepIV, DFIV).

Each downstream estimator in ``models.npiv_estimators`` inherits from
:class:`NPIVEstimator` and exposes a uniform :meth:`predict` interface
``h(A, W, L)`` that the upstream evaluation utilities rely on.

"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

TreatmentFamily = Literal["continuous", "binary"]


class NPIVEstimator(nn.Module):
    """Base class declaring the common prediction API.

    Subclasses must implement :meth:`predict` returning
    ``ĥ(A, W, L)`` on the normalised scale the estimator was trained on.
    The pipeline wraps this with :func:`evaluation.predict_hat_a` to
    de-normalise back to the original ``Y`` scale.

    Subclasses are also expected to expose ``a_dim``, ``w_dim``,
    ``l_dim``, ``y_dim`` and ``treatment_family`` attributes so
    downstream loaders can rebuild the right shape.
    """

    a_dim: int
    w_dim: int
    l_dim: int
    y_dim: int
    treatment_family: TreatmentFamily

    @torch.no_grad()
    def predict(self, a: Tensor, w: Tensor, l: Tensor) -> Tensor:
        raise NotImplementedError


__all__ = ["NPIVEstimator", "TreatmentFamily"]
