"""Replaceable model interface (spec 9.5).

TriageModel
├── TransparentRulesModel   (primary, used for every assessment)
├── AnomalyDetectionModel   (secondary check; used for disagreement detection)
├── SupervisedModel         (development only until real outcome labels exist; not used by the app)
└── EnsembleModel           (future)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TriageModel(ABC):
    model_version: str = "base"

    @abstractmethod
    def fit(self, reference_cases: list[dict]) -> "TriageModel":
        ...

    @abstractmethod
    def score_cases(self, cases: list[dict]) -> dict[str, float]:
        """case_id -> 0-100 score. Higher means more review priority."""
