"""Validation agent, deterministic part (spec 10.4 and 21.4).

Checks every LLM answer against the evidence it was given:
- every cited case ID exists
- every number matches a value in the evidence (with tolerant formatting: 0.44 == 44%)
- every snake_case field reference is a real column, finding id or tool
- no fraud determinations or invented probabilities
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..schemas import REQUIRED_COLUMNS

CASE_ID_RE = re.compile(r"\b[CS]\d{4,5}\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z\d-])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
FIELD_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
FORBIDDEN = [
    (re.compile(r"\b(is|was|are|confirmed|definitely|clearly)\s+(committing\s+)?fraud(ulent)?\b", re.I),
     "States a fraud determination; the tool only sets review priority."),
    (re.compile(r"\d+(\.\d+)?\s*%\s*(probability|likelihood|chance)\s+(of|that)\s+(fraud|it)", re.I),
     "States a fraud probability that the system does not compute."),
    (re.compile(r"\bDr\.\s+[A-Z][a-z]+"), "Names a person who is not in the data."),
]
SMALL_INT_OK = 12  # counts like "3 signals" or "top 5" are fine without a source


@dataclass
class GroundingResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    checked_numbers: int = 0

    def to_dict(self):
        return {"passed": self.passed, "issues": self.issues, "checked_numbers": self.checked_numbers}


def _collect_numbers(obj, out: set):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        return
    if isinstance(obj, str):
        for m in NUMBER_RE.findall(obj):
            v = _to_float(m)
            if v is not None:
                out.add(v)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_numbers(v, out)


def _to_float(token: str):
    t = token.replace("$", "").replace(",", "").rstrip("%").lstrip("+")
    try:
        return float(t)
    except ValueError:
        return None


def _allowed_forms(values: set) -> set:
    forms = set()
    for v in values:
        for x in (v, abs(v), round(v, 1), round(v), round(abs(v), 1), round(abs(v))):
            forms.add(float(x))
        if -1.0 <= v <= 1.0:
            forms.update({round(v * 100, 1), round(abs(v) * 100), round(abs(v) * 100, 1)})
        if abs(v) >= 1000:
            forms.update({round(v / 1000, 1), round(v / 1000), round(v / 1e6, 1), round(v / 1e6, 2)})
    return forms


def _matches(x: float, allowed: set) -> bool:
    for a in allowed:
        if abs(x - a) <= max(0.051, 0.006 * abs(a)):
            return True
    return False


def check(text: str, evidence, *, known_case_ids: set[str], known_fields: set[str] | None = None) -> GroundingResult:
    issues = []
    evidence_text = json.dumps(evidence, default=str)

    for cid in set(CASE_ID_RE.findall(text)):
        if cid not in known_case_ids:
            issues.append(f"Case ID {cid} does not exist in the loaded queue.")

    values: set = set()
    _collect_numbers(evidence, values)
    allowed = _allowed_forms(values)
    checked = 0
    # Ignore numbers that are part of case IDs, claim numbers and dates.
    scrubbed = CASE_ID_RE.sub(" ", text)
    scrubbed = re.sub(r"\b[A-Z]{2,}-\d+\b", " ", scrubbed)
    scrubbed = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", scrubbed)
    scrubbed = re.sub(r"\b(v|rules-v|version\s)\d+(\.\d+)*\b", " ", scrubbed)
    for token in NUMBER_RE.findall(scrubbed):
        x = _to_float(token)
        if x is None:
            continue
        if float(x).is_integer() and 0 <= x <= SMALL_INT_OK and "%" not in token and "$" not in token:
            continue
        checked += 1
        if not _matches(x, allowed):
            issues.append(f"Number {token} is not present in the case evidence.")

    fields = set(known_fields or set()) | set(REQUIRED_COLUMNS)
    for token in set(FIELD_RE.findall(text)):
        if token not in fields and token not in evidence_text:
            issues.append(f"Field '{token}' does not exist in the data.")

    for pattern, msg in FORBIDDEN:
        if pattern.search(text):
            issues.append(msg)

    return GroundingResult(passed=not issues, issues=issues, checked_numbers=checked)
