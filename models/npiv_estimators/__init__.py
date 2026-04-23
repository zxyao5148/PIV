"""Downstream NPIV estimators consuming learned IV representations.

Estimators, all supporting the four ``(a_dim, treatment_family)``
combinations (scalar/vector × continuous/binary):

* :class:`DeepIV`       — Hartford et al. (2017).
* :class:`DFIV`         — Xu et al. (2021).
* :class:`AdversarialGMM` — Bennett et al. / Microsoft AGMM style.
* :class:`DeepGMM`      — Bennett, Kallus, Schnabel (NeurIPS 2019).

All expose a common :meth:`predict(a, w, l) → Tensor` interface
declared by :class:`NPIVEstimator`, consumed by
``evaluation.predict_hat_a`` to produce counterfactual responses on
the original outcome scale.
"""
from models.npiv_estimators.deep_iv import (
    BernoulliTreatmentNet,
    DeepIV,
    GaussianMixtureTreatmentNet,
    StructuralResponseNet,
)
from models.npiv_estimators.adversarial_gmm import AdversarialGMM
from models.npiv_estimators.deep_gmm import DeepGMM
from models.npiv_estimators.df_iv import DFIV
from models.npiv_estimators.helper import (
    MLP,
    DeepIVTreatmentOutput,
    NPIVOutcomeOutput,
    _ensure_2d,
    concat_context,
    concat_wl,
    get_activation,
)
from models.npiv_estimators.parent_class import NPIVEstimator, TreatmentFamily

__all__ = [
    "BernoulliTreatmentNet",
    "AdversarialGMM",
    "DeepGMM",
    "DFIV",
    "DeepIV",
    "DeepIVTreatmentOutput",
    "GaussianMixtureTreatmentNet",
    "MLP",
    "NPIVEstimator",
    "NPIVOutcomeOutput",
    "StructuralResponseNet",
    "TreatmentFamily",
    "_ensure_2d",
    "concat_context",
    "concat_wl",
    "get_activation",
]
