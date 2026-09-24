"""Unsupervised secondary checks (spec 9.3).

Used only to detect disagreement with the rules score. An anomaly is never called fraud.
Methods: robust z-score flags, seed-averaged Isolation Forest, Local Outlier Factor (exploratory),
PCA reconstruction error.

Isolation Forest is two-sided: an unusually SMALL, quiet claim is as "anomalous" as a large one.
The rules model therefore only lets an anomaly change a priority when the case is also unusual
in a suspicious direction (a flag is set, or a signal is above the queue norm).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from ..schemas import BINARY_SIGNALS, CONTINUOUS_SIGNALS
from .base import TriageModel

FEATURES = CONTINUOUS_SIGNALS + BINARY_SIGNALS


def feature_matrix(cases: list[dict]) -> np.ndarray:
    return np.array([[float(c.get(f) or 0) for f in FEATURES] for c in cases])


class AnomalyDetectionModel(TriageModel):
    model_version = "anomaly-1.1-seedavg"

    def __init__(self, contamination: float = 0.15, random_state: int = 42, n_seeds: int | None = None):
        self.contamination = contamination
        self.random_state = random_state
        self.n_seeds = n_seeds

    def fit(self, reference_cases: list[dict]) -> "AnomalyDetectionModel":
        X = feature_matrix(reference_cases)
        self.scaler = RobustScaler().fit(X)
        Xs = self.scaler.transform(X)
        # Seed-averaged Isolation Forest: one forest's random splits can move borderline cases between
        # flagged / not flagged; averaging the raw scores of several forests removes that noise.
        seeds = self.n_seeds or (10 if len(reference_cases) <= 20_000 else 3)
        # 10 forests x 100 trees = 1,000 trees averaged (vs 300 in a single forest), at modest cost.
        self.forests = [IsolationForest(n_estimators=100, random_state=self.random_state + k).fit(Xs) for k in range(seeds)]
        n_neighbors = max(2, min(10, len(reference_cases) - 1))
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination, novelty=True).fit(Xs)
        n_comp = max(1, min(3, Xs.shape[1], len(reference_cases) - 1))
        self.pca = PCA(n_components=n_comp, random_state=self.random_state).fit(Xs)
        self._ref_seed_scores = self._if_seed_scores(Xs)                  # (seeds, n_ref)
        self._ref_if = self._ref_seed_scores.mean(axis=0)
        self._if_cut = np.percentile(self._ref_if, 100 * (1 - self.contamination))
        self._ref_pca = self._pca_error(Xs)
        return self

    def _if_seed_scores(self, Xs):
        return np.array([-f.score_samples(Xs) for f in self.forests])    # higher = more unusual

    def _pca_error(self, Xs):
        recon = self.pca.inverse_transform(self.pca.transform(Xs))
        return np.mean((Xs - recon) ** 2, axis=1)

    def analyze(self, cases: list[dict]) -> dict[str, dict]:
        X = feature_matrix(cases)
        Xs = self.scaler.transform(X)
        seed_scores = self._if_seed_scores(Xs)
        if_scores = seed_scores.mean(axis=0)
        if_flag = if_scores >= self._if_cut
        # Seed spread: how far this case's percentile moves between individual forests.
        ref_sorted = np.sort(self._ref_if)
        seed_pct = np.array([100 * np.searchsorted(ref_sorted, row, side="right") / len(ref_sorted) for row in seed_scores])
        lof_flag = self.lof.predict(Xs) == -1
        pca_err = self._pca_error(Xs)
        pca_cut = np.percentile(self._ref_pca, 100 * (1 - self.contamination))
        out = {}
        for i, c in enumerate(cases):
            z_flags = int(np.sum(np.abs(Xs[i, : len(CONTINUOUS_SIGNALS)]) > 3))
            votes = int(if_flag[i]) + int(lof_flag[i]) + int(pca_err[i] > pca_cut)
            out[c["case_id"]] = {
                "isolation_forest_percentile": round(float(100 * np.mean(self._ref_if <= if_scores[i])), 1),
                "isolation_forest_seed_range": round(float(seed_pct[:, i].max() - seed_pct[:, i].min()), 1),
                "isolation_forest_seeds": int(len(self.forests)),
                "isolation_forest_flag": bool(if_flag[i]),
                "lof_flag": bool(lof_flag[i]),
                "pca_error_flag": bool(pca_err[i] > pca_cut),
                "robust_z_flags": z_flags,
                "votes": votes,
                # Unusual only when at least two independent methods agree. Direction is checked by the
                # rules model: an anomaly only counts when the case is unusual in a suspicious direction.
                "is_anomalous": votes >= 2,
                "model_version": self.model_version,
            }
        return out

    def score_cases(self, cases: list[dict]) -> dict[str, float]:
        return {k: v["isolation_forest_percentile"] for k, v in self.analyze(cases).items()}
