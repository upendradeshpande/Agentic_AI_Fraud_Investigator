"""Transparent rules model: the primary triage score (spec section 7).

suspicion_score = sum over groups of min(group cap, sum of signal points in group)
                + min(combination cap, combination bonuses)

The LLM never produces or modifies this number.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import settings
from ..schemas import PRIORITY_LABELS, Assessment, Finding
from .base import TriageModel

MAD_SCALE = 1.4826


def load_ruleset(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or settings.ruleset_path).read_text())


class TransparentRulesModel(TriageModel):
    model_version = "transparent-rules-1.2"

    def __init__(self, ruleset: dict | None = None):
        self.ruleset = ruleset or load_ruleset()
        self.ruleset_id = self.ruleset["ruleset_id"]
        self.signals = self.ruleset["signals"]
        self.continuous = [k for k, v in self.signals.items() if v["type"] == "continuous"]
        self.binary = [k for k, v in self.signals.items() if v["type"] == "binary"]
        self.reference: dict[str, dict] = {}
        self._values: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, reference_cases: list[dict]) -> "TransparentRulesModel":
        for sig in self.continuous:
            vals = np.array([c[sig] for c in reference_cases if c.get(sig) is not None], dtype=float)
            median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - median))) * MAD_SCALE
            if mad == 0:
                mad = float(np.std(vals)) or 1.0
            self.reference[sig] = {
                "median": median,
                "scale": mad,
                "p90": float(np.percentile(vals, 90)),
                "n": int(len(vals)),
            }
            self._values[sig] = np.sort(vals)
        return self

    def robust_z(self, sig: str, value: float) -> float:
        ref = self.reference[sig]
        return (float(value) - ref["median"]) / ref["scale"]

    def percentile(self, sig: str, value: float) -> float:
        vals = self._values[sig]
        return round(100.0 * np.searchsorted(vals, value, side="right") / len(vals), 1)

    # ----------------------------------------------------------------- scoring
    def _raw_points(self, case: dict, excluded: set[str], weights: Optional[dict] = None):
        cfg = self.ruleset["continuous_scoring"]
        z0, z1 = cfg["z_start"], cfg["z_full"]
        raw, zs = {}, {}
        for sig, meta in self.signals.items():
            w = meta["points"] * (weights.get(sig, 1.0) if weights else 1.0)
            val = case.get(sig)
            if val is None or sig in excluded:
                raw[sig] = 0.0
                continue
            if meta["type"] == "binary":
                raw[sig] = w if int(val) == 1 else 0.0
            else:
                z = self.robust_z(sig, val)
                zs[sig] = z
                frac = min(1.0, max(0.0, (z - z0) / (z1 - z0)))
                raw[sig] = w * frac
        return raw, zs

    def _unusual(self, sig, zs, excluded):
        return sig not in excluded and zs.get(sig, 0.0) >= self.ruleset["unusual_z"]

    def _binary_on(self, sig, case, excluded):
        return sig not in excluded and int(case.get(sig) or 0) == 1

    def _combos(self, case, zs, excluded):
        hits = []
        for combo in self.ruleset["combinations"]:
            if combo["id"] in excluded:
                continue
            ok = all(self._binary_on(s, case, excluded) for s in combo.get("requires_binary", []))
            ok = ok and all(self._unusual(s, zs, excluded) for s in combo.get("requires_unusual", []))
            any_list = combo.get("requires_any_unusual")
            if any_list:
                ok = ok and any(self._unusual(s, zs, excluded) for s in any_list)
            count_rule = combo.get("requires_count_unusual")
            if count_rule:
                n = sum(1 for s in count_rule["signals"]
                        if s not in excluded and zs.get(s, 0.0) >= count_rule["z"])
                ok = ok and n >= count_rule["min"]
            if ok:
                hits.append(combo)
        return hits

    def _score(self, case, excluded, weights=None):
        raw, zs = self._raw_points(case, excluded, weights)
        groups = self.ruleset["groups"]
        capped, group_totals = {}, {}
        for gid, g in groups.items():
            if gid == "combination":
                continue
            members = [s for s, m in self.signals.items() if m["group"] == gid]
            total = sum(raw[s] for s in members)
            factor = min(1.0, g["cap"] / total) if total > 0 else 1.0
            for s in members:
                capped[s] = raw[s] * factor
            group_totals[gid] = min(total, g["cap"])
        combos = self._combos(case, zs, excluded)
        combo_raw = sum(c["points"] for c in combos)
        combo_cap = groups["combination"]["cap"]
        combo_factor = min(1.0, combo_cap / combo_raw) if combo_raw else 1.0
        group_totals["combination"] = min(combo_raw, combo_cap)
        score = round(sum(group_totals.values()), 1)
        return score, raw, capped, zs, combos, combo_factor, group_totals

    def _level(self, score, strong):
        t = self.ruleset["thresholds"]
        if strong or score >= t["red"]:
            return "red"
        if score >= t["yellow"]:
            return "yellow"
        return "green"

    def _strong(self, case, excluded):
        return [s["label"] for s in self.ruleset["strong_indicators"]
                if all(self._binary_on(b, case, excluded) for b in s["requires_binary"])]

    def score_cases(self, cases: list[dict]) -> dict[str, float]:
        return {c["case_id"]: self._score(c, set())[0] for c in cases}

    # -------------------------------------------------------------- stability
    def stability(self, case: dict, excluded: set[str], base_level: str) -> float:
        """Share of randomly perturbed weightings (+/- perturbation_pct) that keep the same level."""
        cfg = self.ruleset["confidence"]
        # Exact shortcut: signal weights scaled by factors in [1-p, 1+p] keep every capped group total within
        # [(1-p)g, (1+p)g] and combination points are unaffected, so the perturbed score lies in
        # [(1-p)S, (1+p)S]. If both ends give the same level, every run does, and simulation is unnecessary.
        strong = bool(self._strong(case, excluded))
        score = self._score(case, excluded)[0]
        pct = cfg["perturbation_pct"]
        if self._level(score * (1 - pct), strong) == self._level(score * (1 + pct), strong) == base_level:
            return 1.0
        seed = int(hashlib.md5(case["case_id"].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        same = 0
        runs = cfg["perturbation_runs"]
        for _ in range(runs):
            weights = {s: 1 + rng.uniform(-cfg["perturbation_pct"], cfg["perturbation_pct"])
                       for s in self.signals}
            score = self._score(case, excluded, weights)[0]
            same += self._level(score, strong) == base_level
        return round(same / runs, 3)

    # -------------------------------------------------------------- assessment
    def assess(
        self,
        case: dict,
        *,
        excluded: Optional[set[str]] = None,
        dq_warnings: Optional[list[str]] = None,
        anomaly: Optional[dict] = None,
        cohort_size: Optional[int] = None,
        reject_reasons: Optional[dict[str, str]] = None,
    ) -> Assessment:
        excluded = set(excluded or [])
        dq_warnings = list(dq_warnings or [])
        reject_reasons = reject_reasons or {}
        score, raw, capped, zs, combos, combo_factor, group_totals = self._score(case, excluded)
        strong = self._strong(case, excluded)

        findings: list[Finding] = []
        for sig, meta in self.signals.items():
            val = case.get(sig)
            rejected = sig in excluded
            if meta["type"] == "binary":
                triggered = val is not None and int(val) == 1
                observed_text = "Flagged" if triggered else "Not flagged"
                z = pct = None
            else:
                z = round(self.robust_z(sig, val), 2) if val is not None else None
                pct = self.percentile(sig, val) if val is not None else None
                ref = self.reference[sig]
                triggered = raw[sig] > 0 or (rejected and z is not None and z >= self.ruleset["continuous_scoring"]["z_start"])
                observed_text = (f"{val} {meta['unit']} (queue median {ref['median']:g}, "
                                 f"{z:+.1f} robust SD, {_ordinal(pct)} percentile)") if val is not None else "Missing"
            findings.append(Finding(
                finding_id=sig, label=meta["label"], kind=meta["type"], group=meta["group"],
                observed=val, observed_text=observed_text, why=meta["why"],
                points=round(capped[sig], 1), raw_points=round(raw[sig], 1),
                max_points=meta["points"], triggered=bool(triggered), source_columns=[sig],
                robust_z=z, queue_percentile=pct,
                status="rejected" if rejected else "open",
                reject_reason=reject_reasons.get(sig, ""),
            ))
        combo_ids = {c["id"] for c in combos}
        for combo in self.ruleset["combinations"]:
            rejected = combo["id"] in excluded
            if combo["id"] not in combo_ids and not rejected:
                continue
            cols = (combo.get("requires_binary", []) + combo.get("requires_unusual", [])
                    + combo.get("requires_any_unusual", [])
                    + combo.get("requires_count_unusual", {}).get("signals", []))
            pts = combo["points"] * combo_factor if combo["id"] in combo_ids else 0.0
            findings.append(Finding(
                finding_id=combo["id"], label=combo["label"], kind="combination", group="combination",
                observed=True, observed_text="Pattern present", why=combo["why"],
                points=round(pts, 1), raw_points=combo["points"] if combo["id"] in combo_ids else 0.0,
                max_points=combo["points"], triggered=True, source_columns=cols,
                status="rejected" if rejected else "open", reject_reason=reject_reasons.get(combo["id"], ""),
            ))
        findings.sort(key=lambda f: (-f.points, f.status == "rejected", f.label))

        level = self._level(score, bool(strong))
        reasons: list[str] = []
        conf = 1.0
        anomaly = anomaly or {}
        # An anomaly only counts if the case is unusual in a suspicious direction. Isolation Forest also isolates
        # unusually small, quiet claims; those are not a reason to raise priority.
        suspicious_direction = any(self._binary_on(sg, case, excluded) for sg in self.binary) or \
            any(zs.get(sg, 0.0) >= 1.0 for sg in self.continuous if sg not in excluded)
        if level == "green" and anomaly.get("is_anomalous") and suspicious_direction:
            level = "yellow"
            reasons.append("Rules score is low but the anomaly check marks this case as unusual; routed to Needs attention.")
            conf -= 0.25
        elif level == "green" and anomaly.get("is_anomalous"):
            reasons.append("The anomaly check marks this case as unusual, but only in a benign direction "
                           "(for example an unusually small claim), so its priority was not raised.")
        elif level == "red" and anomaly and not anomaly.get("is_anomalous") and not strong:
            reasons.append("Rules score is high but the anomaly check does not mark the case as unusual.")
            conf -= 0.15
        if dq_warnings:
            conf -= 0.3
            reasons.append("Data-quality warning on this record.")
            if level == "green":
                level = "data_review"
        missing = [s for s in self.signals if case.get(s) is None]
        if missing:
            conf -= 0.2
            reasons.append(f"Missing values: {', '.join(missing)}.")
        min_cohort = self.ruleset["confidence"]["min_cohort_size"]
        if cohort_size is not None and cohort_size < min_cohort:
            conf -= 0.15
            reasons.append(f"Only {cohort_size} cases of this care type in the queue; peer comparison is thin.")
        stab = self.stability(case, excluded, self._level(score, bool(strong)))
        if stab < self.ruleset["confidence"]["stability_threshold"]:
            conf -= 0.2
            reasons.append(f"Priority changes in {100 - stab * 100:.0f}% of runs when weights shift by 20%; score sits near a threshold.")
        conf = round(max(0.0, conf), 2)
        confidence = "High" if conf >= 0.85 else "Medium" if conf >= 0.55 else "Low"
        if not reasons:
            reasons.append("Complete record, stable under weight changes, rules and anomaly checks agree.")

        mitigating = self._mitigating(case, findings, zs)
        a = Assessment(
            case_id=case["case_id"], score=score, priority_level=level,
            priority_label=PRIORITY_LABELS[level], confidence=confidence, confidence_score=conf,
            confidence_reasons=reasons,
            signal_contributions={f.finding_id: f.points for f in findings},
            group_contributions={k: round(v, 1) for k, v in group_totals.items()},
            findings=findings, mitigating=mitigating, strong_indicators=strong,
            data_quality_warnings=dq_warnings, uncertainty="", recommended_next_step="",
            model_version=self.model_version, ruleset_id=self.ruleset_id,
            anomaly=anomaly, stability=stab, excluded_findings=sorted(excluded),
        )
        a.caveats = caveats(a)
        steps = next_steps(a, case)
        a.recommended_next_step = steps[0]
        a.uncertainty = uncertainty_statement(a)
        return a

    def _mitigating(self, case, findings, zs) -> list[str]:
        out = []
        for f in findings:
            if f.status == "rejected":
                out.append(f"{f.label} was rejected by the investigator"
                           + (f": {f.reject_reason}" if f.reject_reason else "."))
        for sig in ["duplicate_service_billed", "service_overlap_other_provider", "shared_contact_with_provider"]:
            if int(case.get(sig) or 0) == 0:
                out.append(f"{self.signals[sig]['label']}: not flagged.")
        for sig in self.continuous:
            z = zs.get(sig)
            if z is not None and z < 1.0:
                ref = self.reference[sig]
                out.append(f"{self.signals[sig]['label']} is within the normal range "
                           f"({case[sig]} vs queue median {ref['median']:g}).")
        if (case.get("amount_vs_peer_avg_pct") or 0) < 0:
            out.append(f"Claim amount is {abs(case['amount_vs_peer_avg_pct'])}% below the peer average.")
        return out[:6]


def _ordinal(value: float) -> str:
    """82 -> '82nd', 71 -> '71st', 13 -> '13th', 97.5 -> '97.5th'."""
    if float(value) != int(value):
        return f"{value:g}th"
    n = int(value)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def next_steps(a: Assessment, case: dict) -> list[str]:
    """Deterministic next-step recommendations derived from the active evidence."""
    active = {f.finding_id for f in a.findings if f.triggered and f.points > 0}
    steps = []
    if a.data_quality_warnings:
        steps.append("Resolve the data-quality warning with claims operations before relying on the signals.")
    if {"shared_contact_with_provider", "service_overlap_other_provider"} <= active and "duplicate_service_billed" not in active:
        steps.append("Check whether the overlapping provider is linked to the member (shared address, phone or family "
                     "relationship) and confirm both providers actually delivered the billed services.")
    if {"duplicate_service_billed", "service_overlap_other_provider"} <= active:
        steps.append("Pull service-line records from both providers and check whether the overlapping services fall on the same dates.")
    elif "duplicate_service_billed" in active:
        steps.append("Request the itemised bill and confirm whether the duplicated service lines are separate visits.")
    elif "service_overlap_other_provider" in active:
        steps.append("Compare visit schedules with the other provider to confirm the overlap is real.")
    if "shared_contact_with_provider" in active:
        steps.append("Verify member and provider contact records independently and check for a family or business relationship.")
    if {"weekly_visit_frequency", "member_provider_distance_miles"} & active:
        steps.append(f"Check visit verification logs: {case.get('weekly_visit_frequency')} visits/week at "
                     f"{case.get('member_provider_distance_miles')} miles needs to be operationally plausible.")
    if "amount_vs_peer_avg_pct" in active:
        steps.append("Compare billed rates with the contracted rate for this care type.")
    if {"round_dollar_billing_ratio", "weekend_billing_ratio"} & active:
        steps.append("Sample line items for round amounts and weekend dates and request matching timesheets.")
    if a.priority_level == "green" and not steps:
        steps.append("No strong indicators. Confirm nothing is missing and approve as lower priority.")
    if not steps:
        steps.append("Review the top-weighted findings and confirm them against source records.")
    return steps


CAVEATS = {
    "duplicate_service_billed": "The file does not show whether both duplicate lines were paid or one was a correction or resubmission.",
    "service_overlap_other_provider": "The overlap flag does not show whether the two providers' services were actually mutually exclusive.",
    "shared_contact_with_provider": "Shared contact details can have an administrative cause, such as a facility or case-manager number.",
    "weekly_visit_frequency": "Frequent visits are expected for members with high care needs; the care plan is not in the data.",
    "prior_claims_last_12mo": "Many prior claims can reflect genuine long-term care needs rather than a pattern of abuse.",
    "member_provider_distance_miles": "Distance uses addresses on file; the provider may deliver care from a closer branch.",
    "amount_vs_peer_avg_pct": "The peer definition behind amount vs peer average is not available here.",
    "weekend_billing_ratio": "Weekend care is normal for some care types and live-in arrangements.",
    "round_dollar_billing_ratio": "Some contracts bill flat daily rates, which produces round amounts legitimately.",
}


def caveats(a: Assessment) -> list[str]:
    """Skeptic view: for each active finding, what the case-level data cannot establish."""
    out = [CAVEATS[f.finding_id] for f in a.findings
           if f.triggered and f.points > 0 and f.status != "rejected" and f.finding_id in CAVEATS]
    return out[:4]


def uncertainty_statement(a: Assessment) -> str:
    parts = ["No service-line records or confirmed investigation outcomes are available, so this is a review priority, not a fraud finding."]
    parts.append("Peer comparisons are relative to the currently loaded queue.")
    if a.confidence != "High":
        parts.append(f"Confidence is {a.confidence.lower()}: {a.confidence_reasons[0]}")
    return " ".join(parts)
