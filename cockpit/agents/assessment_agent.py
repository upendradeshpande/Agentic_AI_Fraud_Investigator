"""Analysis agent + Assessment agent (spec 10.2, 10.3).

AnalysisAgent  - deterministic: gathers peer comparison, anomaly results and the scored
                 findings into an evidence pack. No LLM needed for this step.
AssessmentAgent - LLM writes the summary, key points, mitigating points, next step and
                 uncertainty FROM the evidence pack. It cannot set or change the score.
                 Output is schema-checked and grounding-checked; one retry on failure;
                 deterministic fallback if the LLM is unavailable or fails twice.
"""
from __future__ import annotations

import json
import time
import uuid

from ..config import settings
from ..llm.providers import LLMError, get_provider, parse_json
from ..models.rules_model import next_steps
from ..schemas import Assessment
from .. import knowledge_base
from . import grounding
from .prompts import ASSESSMENT_SYSTEM

LEVEL_PHRASE = {
    "red": "High-priority review",
    "yellow": "Needs attention",
    "green": "Lower priority",
    "data_review": "Data review before assessment",
}


class AnalysisAgent:
    def __init__(self, service):
        self.svc = service

    def evidence_pack(self, case_id: str, a: Assessment) -> dict:
        case = self.svc.by_id[case_id]
        cohort = self.svc.compare_to_cohort(case_id)
        return {
            "case": case,
            "assessment": {
                "score": a.score, "priority_level": a.priority_level, "priority_label": a.priority_label,
                "confidence": a.confidence, "confidence_reasons": a.confidence_reasons,
                "strong_indicators": a.strong_indicators,
                "group_contributions": a.group_contributions,
                "data_quality_warnings": a.data_quality_warnings,
                "ruleset_id": a.ruleset_id,
            },
            "active_findings": [
                {"finding_id": f.finding_id, "label": f.label, "observed": f.observed_text,
                 "points": f.points, "why": f.why, "status": f.status}
                for f in a.findings if f.triggered and f.points > 0 and f.status != "rejected"
            ],
            "rejected_findings": [
                {"finding_id": f.finding_id, "label": f.label, "reason": f.reject_reason}
                for f in a.findings if f.status == "rejected"
            ],
            "mitigating": a.mitigating,
            "caveats_data_cannot_establish": a.caveats,
            "peer_comparison": {"note": cohort["note"], "rows": [
                {"label": r["label"], "selected": r["selected"], "peer_median": r["peer_median"], "meaning": r["meaning"]}
                for r in cohort["continuous"]]},
            "anomaly_check": {k: v for k, v in a.anomaly.items() if k != "model_version"},
            "candidate_next_steps": next_steps(a, case),
            "guidance": [{"citation": g["citation"], "text": g["text"]} for g in knowledge_base.for_assessment(a, 2)],
            "known_limits": [
                "No service-line records.", "No confirmed investigation outcomes.",
                "Peer groups come from the loaded queue only.",
            ],
        }


def deterministic_narrative(a: Assessment, case: dict) -> dict:
    ev = [f for f in a.key_evidence if f.status != "rejected"]
    if a.priority_level == "data_review":
        summary = (f"{case['care_type']} claim for ${case['claim_amount_usd']:,} with a data-quality issue "
                   "that should be resolved before the signals are relied on.")
    elif ev:
        names = ", ".join(f.label.lower() for f in ev[:3])
        summary = (f"{LEVEL_PHRASE[a.priority_level]} (score {a.score:g}). "
                   f"The strongest evidence is {names}.")
        if a.strong_indicators:
            summary += " This matches a policy-defined strong indicator."
    else:
        summary = f"Lower priority (score {a.score:g}). No signals are above the queue norm."
    return {
        "summary": summary,
        "key_points": [f"{f.label}: {f.observed_text}. {f.why}" for f in ev[:4]],
        "mitigating": (a.mitigating or a.caveats)[:3],
        "next_step": a.recommended_next_step,
        "uncertainty": a.uncertainty,
        "cited_findings": [f.finding_id for f in ev[:4]],
        "source": "deterministic",
        "grounding": {"passed": True, "issues": [], "checked_numbers": 0},
    }


def _schema_issues(out: dict, allowed_findings: set[str]) -> list[str]:
    issues = []
    for key, typ in [("summary", str), ("key_points", list), ("mitigating", list), ("next_step", str),
                     ("uncertainty", str), ("cited_findings", list)]:
        if not isinstance(out.get(key), typ):
            issues.append(f"Field '{key}' missing or not a {typ.__name__}.")
    for fid in out.get("cited_findings", []) or []:
        if fid not in allowed_findings:
            issues.append(f"cited_findings contains '{fid}', which is not an active finding.")
    return issues


class AssessmentAgent:
    def __init__(self, service):
        self.svc = service
        self.provider = get_provider()

    def run(self, case_id: str, a: Assessment) -> tuple[dict, str]:
        case = self.svc.by_id[case_id]
        fallback = deterministic_narrative(a, case)
        if not settings.enable_llm or not self.provider.available:
            return fallback, "deterministic"

        pack = AnalysisAgent(self.svc).evidence_pack(case_id, a)
        allowed = {f["finding_id"] for f in pack["active_findings"]}
        known_fields = {f.finding_id for f in a.findings} | {"priority_level", "risk_score"}
        messages = [{"role": "user", "content": "Evidence pack:\n" + json.dumps(pack, default=str)}]
        run = {"run_id": f"R-{uuid.uuid4().hex[:10]}", "agent": "assessment", "case_id": case_id,
               "provider": self.provider.name, "model": self.provider.model,
               "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
        result, issues = None, []
        for attempt in range(2):  # one retry for invalid output
            try:
                resp = self.provider.chat(ASSESSMENT_SYSTEM, messages, max_tokens=900, json_mode=True)
            except LLMError as e:
                run["error"] = str(e)
                break
            run["input_tokens"] += resp.input_tokens
            run["output_tokens"] += resp.output_tokens
            run["latency_ms"] += resp.latency_ms
            try:
                out = parse_json(resp.text)
            except ValueError as e:
                issues = [f"Output was not valid JSON: {e}"]
            else:
                issues = _schema_issues(out, allowed)
                if not issues:
                    text = " ".join([out["summary"], out["next_step"], out["uncertainty"],
                                     *map(str, out["key_points"]), *map(str, out["mitigating"])])
                    g = grounding.check(text, pack, known_case_ids=self.svc.by_id, known_fields=known_fields)
                    issues = g.issues
                    if g.passed:
                        out["source"] = "llm"
                        out["grounding"] = g.to_dict()
                        result = out
                        break
            messages += [{"role": "assistant", "content": resp.text},
                         {"role": "user", "content": "Your output failed validation:\n- " + "\n- ".join(issues)
                          + "\nReturn corrected JSON only, using only values from the evidence pack."}]

        run["grounding_passed"] = result is not None
        run["grounding_issues"] = issues if result is None else []
        run["fallback_used"] = result is None
        run["answer"] = json.dumps(result or fallback)[:2000]
        self.svc.repo.log_agent_run(run, [])
        if result is None:
            fallback["grounding"] = {"passed": False, "issues": issues or [run.get("error", "LLM unavailable")],
                                     "checked_numbers": 0}
            fallback["source"] = "deterministic (LLM output rejected)" if issues else "deterministic (LLM error)"
            # Distinct model name so the UI does not re-call the LLM on every rerun.
            return fallback, "deterministic (llm fallback)"
        return result, f"{self.provider.name}:{self.provider.model}"


def batch_generate(service, only_missing: bool = True, progress=None) -> int:
    """Pre-compute AI narratives for the whole queue (the 'ready by morning' job)."""
    n = 0
    rows = service.queue()
    for i, r in enumerate(rows):
        cur = service.get_assessment(r["case_id"])
        if only_missing and cur["llm_model"] not in ("deterministic", "pending"):
            continue
        narrative, model = AssessmentAgent(service).run(r["case_id"], cur["assessment"])
        service.repo.update_narrative(cur["assessment_id"], narrative, model)
        n += 1
        if progress:
            progress(i + 1, len(rows))
        time.sleep(0.05)
    return n
