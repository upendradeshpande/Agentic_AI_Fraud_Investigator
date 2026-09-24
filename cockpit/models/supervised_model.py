"""DEVELOPMENT ONLY. Supervised challenger models and their evaluation harness (spec 9.4 / 21).

Nothing in the product uses this module: not the scores, the queue, the dashboard, the case view,
the copilot or the batch job. The supplied cases have no outcome labels, and a model trained on
synthetic (made-up) labels is not fit for production decisions.

It exists so the training / testing / comparison pipeline is ready for real labels: once investigator
outcomes accumulate (every final decision is stored), replace the synthetic set with them. Until then
it can be run by hand as a development sanity check:

    python -m cockpit.models.supervised_model      # writes data/dev_synthetic_comparison.json

Preprocessing is fitted on training data only (inside the sklearn Pipeline).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import synthetic
from .anomaly_model import FEATURES, feature_matrix
from .base import TriageModel
from .rules_model import TransparentRulesModel


class SupervisedModel(TriageModel):
    def __init__(self, kind: str = "gradient_boosting", random_state: int = 42):
        self.kind = kind
        self.model_version = f"supervised-{kind}-1.0"
        if kind == "logistic_regression":
            est = LogisticRegression(max_iter=2000, class_weight="balanced")
        else:
            est = GradientBoostingClassifier(random_state=random_state)
        self.pipeline = Pipeline([("scale", StandardScaler()), ("model", est)])

    def fit(self, reference_cases: list[dict]) -> "SupervisedModel":
        X = feature_matrix(reference_cases)
        y = np.array([c["label"] for c in reference_cases])
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, cases: list[dict]) -> np.ndarray:
        return self.pipeline.predict_proba(feature_matrix(cases))[:, 1]

    def score_cases(self, cases: list[dict]) -> dict[str, float]:
        p = self.predict_proba(cases)
        return {c["case_id"]: round(float(v) * 100, 1) for c, v in zip(cases, p)}

    def feature_importance(self) -> dict[str, float]:
        m = self.pipeline.named_steps["model"]
        vals = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_[0])
        return {f: round(float(v), 4) for f, v in sorted(zip(FEATURES, vals), key=lambda t: -t[1])}


def binary_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    div = lambda a, b: round(a / b, 3) if b else None
    return {
        "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
        "false_positive_rate": div(fp, fp + tn), "false_negative_rate": div(fn, fn + tp),
        "negative_predictive_value": div(tn, tn + fn),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def curve_summary(y_true, scores) -> dict:
    p, r, _ = precision_recall_curve(y_true, scores)
    idx = np.linspace(0, len(p) - 1, min(40, len(p))).astype(int)
    return {
        "average_precision": round(float(average_precision_score(y_true, scores)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 3),
        "pr_curve": {"precision": [round(float(p[i]), 3) for i in idx], "recall": [round(float(r[i]), 3) for i in idx]},
    }


def evaluate_all(reference_cases: list[dict], n: int = 3000, seed: int = 7) -> dict:
    """Compare transparent rules vs challengers on the synthetic labelled set."""
    data = synthetic.generate(n=n, seed=seed)
    y = np.array([c["label"] for c in data])
    train, test = train_test_split(data, test_size=0.3, stratify=y, random_state=seed)
    y_test = np.array([c["label"] for c in test])

    rules = TransparentRulesModel().fit(reference_cases)
    results = {"generator_version": synthetic.GENERATOR_VERSION, "n_total": n,
               "n_test": len(test), "prevalence_test": round(float(y_test.mean()), 3), "models": {}}

    rule_scores, levels = [], []
    for c in test:
        a = rules.assess(c)
        rule_scores.append(a.score)
        levels.append(a.priority_level)
    rule_scores = np.array(rule_scores)
    red = np.array([lvl == "red" for lvl in levels]).astype(int)
    red_or_yellow = np.array([lvl in ("red", "yellow") for lvl in levels]).astype(int)
    results["models"]["transparent_rules"] = {
        "model_version": rules.model_version, "ruleset_id": rules.ruleset_id,
        "at_red": binary_metrics(y_test, red),
        "at_red_or_yellow": binary_metrics(y_test, red_or_yellow),
        "green_path_npv": binary_metrics(y_test, red_or_yellow)["negative_predictive_value"],
        **curve_summary(y_test, rule_scores),
    }

    per_scheme = {}
    schemes = sorted({c["scheme"] for c in test})
    for s in schemes:
        mask = np.array([c["scheme"] == s for c in test])
        per_scheme[s] = {"n": int(mask.sum()), "flagged_red_or_yellow": round(float(red_or_yellow[mask].mean()), 3)}
    results["models"]["transparent_rules"]["by_scheme"] = per_scheme

    challengers = {}
    for kind in ["logistic_regression", "gradient_boosting"]:
        m = SupervisedModel(kind).fit(train)
        proba = m.predict_proba(test)
        pred = (proba >= 0.5).astype(int)
        results["models"][kind] = {
            "model_version": m.model_version, "at_0.5": binary_metrics(y_test, pred),
            **curve_summary(y_test, proba), "feature_importance": m.feature_importance(),
        }
        challengers[kind] = m
    return {"results": results, "challengers": challengers}


def main():
    import json
    from ..config import ROOT, settings
    from ..data_loader import load_file, records
    df, _ = load_file(str(settings.sample_csv))
    res = evaluate_all(records(df))["results"]
    res["warning"] = "DEVELOPMENT ONLY: synthetic labels from documented assumptions. Not evidence of real accuracy."
    out = ROOT / "data" / "dev_synthetic_comparison.json"
    out.write_text(json.dumps(res, indent=2))
    print(res["warning"])
    for name, m in res["models"].items():
        print(f"  {name:22s} PR-AUC {m['average_precision']}  ROC-AUC {m['roc_auc']}")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
