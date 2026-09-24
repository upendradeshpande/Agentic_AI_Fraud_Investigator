"""Peer comparison and similar cases (spec section 8).

Peer group = same care type within the loaded queue. With 50 rows that is 8-13 cases,
so the UI always states that comparisons are relative to the current queue.
"""
from __future__ import annotations

import numpy as np

from .schemas import BINARY_SIGNALS, CONTINUOUS_SIGNALS

LABELS = {
    "weekly_visit_frequency": "Weekly visit frequency",
    "member_provider_distance_miles": "Member-provider distance (mi)",
    "prior_claims_last_12mo": "Prior claims (12 mo)",
    "amount_vs_peer_avg_pct": "Amount vs peer average (%)",
    "weekend_billing_ratio": "Weekend billing ratio",
    "round_dollar_billing_ratio": "Round-dollar billing ratio",
    "claim_amount_usd": "Claim amount (USD)",
    "duplicate_service_billed": "Duplicate service billed",
    "shared_contact_with_provider": "Shared contact with provider",
    "recent_policy_change_flag": "Recent policy change",
    "service_overlap_other_provider": "Service overlap other provider",
}


def cohort(cases: list[dict], case: dict, by: str = "care_type") -> list[dict]:
    return [c for c in cases if c.get(by) == case.get(by)]


def _meaning(value, median, p90, spread):
    if spread == 0:
        spread = abs(median) or 1
    if value > p90 and value > median:
        return "Much higher"
    if value > median + 0.5 * spread:
        return "Higher"
    if value < median - 0.5 * spread:
        return "Lower"
    return "Typical"


def compare_to_cohort(cases: list[dict], case: dict, by: str = "care_type") -> dict:
    peers = [c for c in cohort(cases, case, by) if c["case_id"] != case["case_id"]]
    rows = []
    for sig in ["claim_amount_usd"] + CONTINUOUS_SIGNALS:
        vals = np.array([float(c[sig]) for c in peers if c.get(sig) is not None])
        if len(vals) == 0:
            continue
        med, p90 = float(np.median(vals)), float(np.percentile(vals, 90))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        x = float(case[sig])
        # Leave-one-out percentile: the case itself is excluded from its own baseline (mid-rank for ties).
        pct = 100.0 * (np.sum(vals < x) + 0.5 * np.sum(vals == x)) / len(vals)
        rows.append({
            "signal": sig, "label": LABELS[sig], "selected": case.get(sig),
            "peer_median": round(med, 2), "peer_p90": round(p90, 2), "peer_percentile": round(float(pct), 1),
            "meaning": _meaning(x, med, p90, iqr),
        })
    binary_rows = []
    for sig in BINARY_SIGNALS:
        rate = float(np.mean([c[sig] for c in peers])) if peers else 0.0
        binary_rows.append({"signal": sig, "label": LABELS[sig], "selected": case.get(sig),
                            "peer_rate": round(rate, 2)})
    thin = len(peers) < 5
    return {
        "cohort_by": by, "cohort_value": case.get(by), "peer_count": len(peers), "thin_cohort": thin,
        "note": f"Compared with the other {len(peers)} {case.get(by)} cases in the loaded queue (case excluded from its own baseline)."
                + (" Fewer than 5 peers: treat this comparison as indicative only." if thin else ""),
        "continuous": rows, "binary": binary_rows,
    }


def _vectors(cases: list[dict]) -> np.ndarray:
    X = np.array([[float(c.get(s) or 0) for s in CONTINUOUS_SIGNALS + BINARY_SIGNALS] for c in cases])
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr[iqr == 0] = 1.0
    return (X - med) / iqr


def find_similar(cases: list[dict], case_id: str, limit: int = 5,
                 assessments: dict | None = None, decisions: dict | None = None) -> list[dict]:
    ids = [c["case_id"] for c in cases]
    if case_id not in ids:
        return []
    X = _vectors(cases)
    i = ids.index(case_id)
    d = np.linalg.norm(X - X[i], axis=1)
    order = [j for j in np.argsort(d) if j != i][:limit]
    target = cases[i]
    t_on = {s for s in BINARY_SIGNALS if target.get(s) == 1}
    out = []
    for j in order:
        c = cases[j]
        on = {s for s in BINARY_SIGNALS if c.get(s) == 1}
        a = (assessments or {}).get(c["case_id"])
        out.append({
            "case_id": c["case_id"], "care_type": c["care_type"],
            "distance": round(float(d[j]), 2),
            "shared_flags": sorted(LABELS[s] for s in t_on & on),
            "different_flags": sorted(LABELS[s] for s in t_on ^ on),
            "priority_level": a.priority_level if a else None,
            "score": a.score if a else None,
            "investigator_status": (decisions or {}).get(c["case_id"]),
        })
    return out


class CohortCache:
    """Per-care-type signal arrays built once at load, so opening a case does not scan the whole queue."""

    def __init__(self, cases: list[dict], by: str = "care_type"):
        self.by = by
        self.sigs = ["claim_amount_usd"] + CONTINUOUS_SIGNALS
        groups: dict[str, list[dict]] = {}
        for c in cases:
            groups.setdefault(c.get(by), []).append(c)
        self.values = {g: {s: np.array([float(c[s]) for c in cs if c.get(s) is not None]) for s in self.sigs + BINARY_SIGNALS}
                       for g, cs in groups.items()}
        self._all = {s: np.array([float(c[s]) for c in cases if c.get(s) is not None]) for s in self.sigs + BINARY_SIGNALS}

    def column(self, sig: str) -> np.ndarray:
        return self._all[sig]

    @staticmethod
    def _without(vals: np.ndarray, x: float) -> np.ndarray:
        """Leave-one-out: drop one occurrence of the case's own value."""
        idx = np.flatnonzero(vals == x)
        return np.delete(vals, idx[0]) if len(idx) else vals

    def compare(self, case: dict) -> dict:
        g = case.get(self.by)
        vals_by_sig = self.values.get(g, {})
        n_peers = max(0, len(vals_by_sig.get("claim_amount_usd", [])) - 1)
        rows, binary_rows = [], []
        for sig in self.sigs:
            if case.get(sig) is None or sig not in vals_by_sig:
                continue
            x = float(case[sig])
            vals = self._without(vals_by_sig[sig], x)
            if len(vals) == 0:
                continue
            med, p90 = float(np.median(vals)), float(np.percentile(vals, 90))
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            pct = 100.0 * (np.sum(vals < x) + 0.5 * np.sum(vals == x)) / len(vals)
            rows.append({"signal": sig, "label": LABELS[sig], "selected": case.get(sig),
                         "peer_median": round(med, 2), "peer_p90": round(p90, 2), "peer_percentile": round(float(pct), 1),
                         "meaning": _meaning(x, med, p90, iqr)})
        for sig in BINARY_SIGNALS:
            vals = self._without(vals_by_sig.get(sig, np.array([])), float(case.get(sig) or 0))
            binary_rows.append({"signal": sig, "label": LABELS[sig], "selected": case.get(sig),
                                "peer_rate": round(float(vals.mean()), 2) if len(vals) else 0.0})
        thin = n_peers < 5
        return {"cohort_by": self.by, "cohort_value": g, "peer_count": n_peers, "thin_cohort": thin,
                "note": f"Compared with the other {n_peers:,} {g} cases in the loaded queue (case excluded from its own baseline)."
                        + (" Fewer than 5 peers: treat this comparison as indicative only." if thin else ""),
                "continuous": rows, "binary": binary_rows}


class SimilarityIndex:
    """Robust-scaled signal matrix built once; nearest neighbours by Euclidean distance.
    Brute force is fine to a few million rows (one vectorised pass per query); swap for an ANN index beyond that."""

    def __init__(self, cases: list[dict]):
        self.cases = cases
        self.ids = [c["case_id"] for c in cases]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.X = _vectors(cases).astype(np.float32)

    def nearest(self, case_id: str, limit: int = 5) -> list[dict]:
        i = self.pos.get(case_id)
        if i is None:
            return []
        d = np.linalg.norm(self.X - self.X[i], axis=1)
        d[i] = np.inf
        k = min(limit, len(d) - 1)
        if k <= 0:
            return []
        idx = np.argpartition(d, k)[:k]
        idx = idx[np.argsort(d[idx])]
        target = self.cases[i]
        t_on = {s for s in BINARY_SIGNALS if target.get(s) == 1}
        out = []
        for j in idx:
            c = self.cases[j]
            on = {s for s in BINARY_SIGNALS if c.get(s) == 1}
            out.append({"case_id": c["case_id"], "care_type": c["care_type"], "distance": round(float(d[j]), 2),
                        "shared_flags": sorted(LABELS[s] for s in t_on & on),
                        "different_flags": sorted(LABELS[s] for s in t_on ^ on)})
        return out
