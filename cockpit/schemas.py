"""Shared data contracts. Every scoring model returns an `Assessment` so the UI, agents and
evaluation code never depend on one specific scoring method (spec 9.5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

ID_COLUMNS = ["case_id", "claim_number"]
ATTRIBUTE_COLUMNS = ["claim_date", "care_type", "claim_amount_usd", "state"]
BINARY_SIGNALS = [
    "duplicate_service_billed",
    "shared_contact_with_provider",
    "recent_policy_change_flag",
    "service_overlap_other_provider",
]
CONTINUOUS_SIGNALS = [
    "weekly_visit_frequency",
    "member_provider_distance_miles",
    "prior_claims_last_12mo",
    "weekend_billing_ratio",
    "amount_vs_peer_avg_pct",
    "round_dollar_billing_ratio",
]
RATIO_SIGNALS = ["weekend_billing_ratio", "round_dollar_billing_ratio"]
SIGNAL_COLUMNS = BINARY_SIGNALS + CONTINUOUS_SIGNALS
REQUIRED_COLUMNS = ID_COLUMNS + ATTRIBUTE_COLUMNS + SIGNAL_COLUMNS

PRIORITY_LABELS = {
    "red": "High-priority review",
    "yellow": "Needs attention",
    "green": "Lower priority",
    "data_review": "Data review",
}

CASE_STATUSES = [
    "Unreviewed",
    "In review",
    "Approved / closed",
    "Likely false positive",
    "Needs more information",
    "Escalated",
    "Solved",
]
REJECT_REASONS = [
    "Explained by a legitimate reason",
    "Incorrect or outdated source data",
    "Not relevant for this care type",
    "Duplicate of another finding",
    "Other",
]
FINAL_STATUSES = {"Approved / closed", "Likely false positive", "Escalated", "Solved"}


@dataclass
class Finding:
    finding_id: str            # equals the source column for signal findings, combo id otherwise
    label: str
    kind: str                  # binary | continuous | combination
    group: str
    observed: Any
    observed_text: str
    why: str
    points: float              # points after group cap
    raw_points: float          # points before group cap
    max_points: float
    triggered: bool
    source_columns: list[str]
    robust_z: Optional[float] = None
    queue_percentile: Optional[float] = None
    status: str = "open"       # open | accepted | rejected
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assessment:
    case_id: str
    score: float
    priority_level: str
    priority_label: str
    confidence: str
    confidence_score: float
    confidence_reasons: list[str]
    signal_contributions: dict[str, float]
    group_contributions: dict[str, float]
    findings: list[Finding]
    mitigating: list[str]
    strong_indicators: list[str]
    data_quality_warnings: list[str]
    uncertainty: str
    recommended_next_step: str
    model_version: str
    ruleset_id: str
    anomaly: dict = field(default_factory=dict)
    stability: float = 1.0
    excluded_findings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)   # what the data cannot establish ("skeptic" view)

    @property
    def key_evidence(self) -> list[Finding]:
        return [f for f in self.findings if f.triggered and f.points > 0]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Assessment":
        d = dict(d)
        d["findings"] = [Finding(**f) for f in d["findings"]]
        return cls(**d)
