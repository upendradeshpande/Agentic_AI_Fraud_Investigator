"""Allowlisted tools for the agents (spec 10.6).

Namespaces follow the MCP direction (spec 11): claims, triage, analysis, workflow, audit,
knowledge, ui. Tool names exposed to the LLM are plain (Anthropic/OpenAI disallow dots).
Read and analysis tools run immediately. Write tools only return a proposal that the
investigator must confirm in the UI; nothing is written by the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..schemas import CASE_STATUSES, SIGNAL_COLUMNS

FILTER_SCHEMA = {
    "type": "object",
    "description": "Queue filters. All fields optional.",
    "properties": {
        "priority_level": {"type": "array", "items": {"type": "string", "enum": ["red", "yellow", "green", "data_review"]}},
        "care_type": {"type": "array", "items": {"type": "string"}},
        "state": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "array", "items": {"type": "string", "enum": ["High", "Medium", "Low"]}},
        "has_signal": {"type": "array", "items": {"type": "string"}, "description": "Signal columns that must be active findings."},
        "data_quality_warning": {"type": "boolean"},
        "reviewed": {"type": "boolean", "description": "true = any status except Unreviewed; false = Unreviewed only"},
        "min_amount": {"type": "number"}, "max_amount": {"type": "number"},
        "min_score": {"type": "number"}, "max_score": {"type": "number"},
    },
}

CASE_ID = {"type": "string", "description": "Case ID such as C1024"}


@dataclass
class Tool:
    name: str
    namespace: str
    kind: str  # read | analysis | write | ui
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


def _assessment_view(svc, case_id: str) -> dict:
    cur = svc.get_assessment(case_id)
    a = cur["assessment"]
    return {
        "case_id": case_id, "assessment_id": cur["assessment_id"], "version": cur["version"],
        "score": a.score, "priority_level": a.priority_level, "priority_label": a.priority_label,
        "confidence": a.confidence, "confidence_reasons": a.confidence_reasons,
        "strong_indicators": a.strong_indicators, "group_contributions": a.group_contributions,
        "findings": [{"finding_id": f.finding_id, "label": f.label, "observed": f.observed_text,
                      "points": f.points, "max_points": f.max_points, "triggered": f.triggered,
                      "status": f.status, "reject_reason": f.reject_reason, "why": f.why}
                     for f in a.findings if f.triggered or f.status == "rejected"],
        "mitigating": a.mitigating, "caveats": a.caveats, "data_quality_warnings": a.data_quality_warnings,
        "anomaly_check": a.anomaly, "recommended_next_step": a.recommended_next_step,
        "uncertainty": a.uncertainty, "ruleset_id": a.ruleset_id, "model_version": a.model_version,
        "narrative_summary": (cur["narrative"] or {}).get("summary"),
    }


def _check_case(svc, case_id):
    if case_id not in svc.by_id:
        raise ValueError(f"Case {case_id} not found. Valid IDs look like C1001-C1050.")


def build_tools(svc) -> dict[str, Tool]:
    rs = svc.rules.ruleset

    def get_queue_summary():
        return svc.queue_summary()

    def list_cases(filters: dict | None = None, sort: str | None = None, limit: int = 15):
        f = dict(filters or {})
        if sort:
            f["sort"] = sort
        f["limit"] = min(int(limit or 15), 50)
        rows = svc.queue(f)
        return {"count": svc.count({k: v for k, v in f.items() if k not in ("limit", "sort")}),
                "returned": len(rows), "cases": [{k: r[k] for k in (
            "rank", "case_id", "care_type", "state", "claim_amount_usd", "score", "priority_level",
            "confidence", "status", "top_evidence")} for r in rows]}

    def get_case(case_id: str):
        _check_case(svc, case_id)
        return svc.by_id[case_id]

    def get_assessment(case_id: str):
        _check_case(svc, case_id)
        return _assessment_view(svc, case_id)

    def get_signal_definition(signal_name: str):
        meta = rs["signals"].get(signal_name)
        if meta:
            return {"signal": signal_name, **meta, "scoring": rs["continuous_scoring"] if meta["type"] == "continuous" else
                    "Full points when flagged (value 1)."}
        for c in rs["combinations"]:
            if c["id"] == signal_name:
                return c | {"requires_count_unusual": c.get("requires_count_unusual")}
        return {"error": f"Unknown signal {signal_name}", "known": list(rs["signals"])}

    def compare_case_to_cohort(case_id: str, cohort: str = "care_type"):
        _check_case(svc, case_id)
        if cohort not in ("care_type", "state"):
            cohort = "care_type"
        return svc.compare_to_cohort(case_id, cohort)

    def find_similar_cases(case_id: str, limit: int = 5):
        _check_case(svc, case_id)
        return {"case_id": case_id, "similar": svc.similar_cases(case_id, min(int(limit or 5), 10))}

    def get_signal_distribution(signal_name: str, filters: dict | None = None):
        return svc.signal_distribution(signal_name, filters)

    def get_signal_cooccurrence(filters: dict | None = None):
        return {"pairs": svc.cooccurrence(filters)}

    def get_chart_data(chart_id: str, filters: dict | None = None):
        return svc.chart_data(chart_id, filters)

    def get_case_history(case_id: str):
        _check_case(svc, case_id)
        return {"status": svc.repo.get_status(case_id), "audit_trail": svc.repo.audit_trail(case_id)[-25:]}

    def search_knowledge(query: str = "", case_id: str | None = None):
        from .. import knowledge_base
        if case_id:
            _check_case(svc, case_id)
            guidance = knowledge_base.for_assessment(svc.get_assessment(case_id)["assessment"])
            if query:
                guidance = knowledge_base.search(query, (), 2) + guidance
        else:
            guidance = knowledge_base.search(query or "investigation procedure", (), 3)
        feedback = svc.repo.knowledge(query, limit=5) if query else []
        return {"playbook_sections": guidance[:4],
                "investigator_feedback": [{"kind": d["kind"], "case_id": d["case_id"], "title": d["title"], "body": d["body"]}
                                          for d in feedback],
                "note": knowledge_base.DISCLAIMER + " Cite the doc_id and section when you use a playbook."}

    def run_anomaly_analysis(case_id: str):
        _check_case(svc, case_id)
        return {"case_id": case_id, **svc.anomaly[case_id],
                "note": "Anomaly checks are a secondary signal used to detect disagreement. An anomaly is not fraud."}

    def calculate_triage_score(case_id: str, exclude_findings: list[str] | None = None):
        _check_case(svc, case_id)
        return svc.what_if(case_id, list(exclude_findings or []))

    def apply_queue_filter(filters: dict):
        rows = svc.queue({**filters, "limit": 50})
        return {"ui_action": "apply_filter", "filters": filters, "matching_cases": svc.count(filters),
                "case_ids": [r["case_id"] for r in rows]}

    def _proposal(action, **kw):
        cid = kw.get("case_id")
        if cid:
            _check_case(svc, cid)
        return {"ui_action": "proposal", "action": action, "args": kw, "requires_confirmation": True,
                "message": "Proposal created. The investigator must confirm it in the UI."}

    def accept_finding(case_id: str, finding_id: str):
        return _proposal("accept_finding", case_id=case_id, finding_id=finding_id)

    def reject_finding(case_id: str, finding_id: str, reason: str):
        return _proposal("reject_finding", case_id=case_id, finding_id=finding_id, reason=reason)

    def add_case_note(case_id: str, note: str):
        return _proposal("add_case_note", case_id=case_id, note=note)

    def set_case_status(case_id: str, status: str, reason: str = ""):
        if status not in CASE_STATUSES:
            return {"error": f"status must be one of {CASE_STATUSES}"}
        return _proposal("set_case_status", case_id=case_id, status=status, reason=reason)

    signal_enum = {"type": "string", "enum": SIGNAL_COLUMNS}
    tools = [
        Tool("get_queue_summary", "claims", "read", "Counts by priority, statuses and amounts for the whole queue.", _obj({}), get_queue_summary),
        Tool("list_cases", "claims", "read", "List cases in queue order with optional filters and sort (score or claim_amount_usd).",
             _obj({"filters": FILTER_SCHEMA, "sort": {"type": "string", "enum": ["score", "claim_amount_usd"]},
                   "limit": {"type": "integer"}}), list_cases),
        Tool("get_case", "claims", "read", "Raw attributes and signal values for one case.", _obj({"case_id": CASE_ID}, ["case_id"]), get_case),
        Tool("get_assessment", "triage", "read", "Current triage assessment: score, priority, confidence, findings with points, mitigating points, next step.",
             _obj({"case_id": CASE_ID}, ["case_id"]), get_assessment),
        Tool("get_signal_definition", "knowledge", "read", "Definition and scoring rule for a signal or combination id.",
             _obj({"signal_name": {"type": "string"}}, ["signal_name"]), get_signal_definition),
        Tool("compare_case_to_cohort", "analysis", "analysis", "Compare a case with peers of the same care type (or state).",
             _obj({"case_id": CASE_ID, "cohort": {"type": "string", "enum": ["care_type", "state"]}}, ["case_id"]), compare_case_to_cohort),
        Tool("find_similar_cases", "analysis", "analysis", "Nearest cases by signal profile, with their priority and investigator status.",
             _obj({"case_id": CASE_ID, "limit": {"type": "integer"}}, ["case_id"]), find_similar_cases),
        Tool("get_signal_distribution", "analysis", "analysis", "Min, median, p90, max for a signal across the (filtered) queue.",
             _obj({"signal_name": signal_enum, "filters": FILTER_SCHEMA}, ["signal_name"]), get_signal_distribution),
        Tool("get_signal_cooccurrence", "analysis", "analysis", "Most common pairs of active findings across the (filtered) queue.",
             _obj({"filters": FILTER_SCHEMA}), get_signal_cooccurrence),
        Tool("get_chart_data", "analysis", "analysis", "Data behind an Overview chart so you can explain it.",
             _obj({"chart_id": {"type": "string", "enum": ["cases_by_care_type", "exposure_by_priority", "rules_vs_anomaly", "exposure_by_case", "priority_by_care_type",
                                                           "amount_vs_score", "signal_frequency", "signal_cooccurrence"]},
                   "filters": FILTER_SCHEMA}, ["chart_id"]), get_chart_data),
        Tool("get_case_history", "audit", "read", "Status and audit trail (assessments, decisions, notes).",
             _obj({"case_id": CASE_ID}, ["case_id"]), get_case_history),
        Tool("search_knowledge", "knowledge", "read",
             "Investigation playbooks (procedures, false-positive explanations, escalation guidance) and past investigator "
             "feedback. Pass case_id to get guidance routed by that case's findings.",
             _obj({"query": {"type": "string"}, "case_id": CASE_ID}), search_knowledge),
        Tool("run_anomaly_analysis", "analysis", "analysis", "Isolation Forest / LOF / PCA anomaly votes for a case.",
             _obj({"case_id": CASE_ID}, ["case_id"]), run_anomaly_analysis),
        Tool("calculate_triage_score", "triage", "analysis", "What-if: recompute the score excluding some finding ids. Does not save anything.",
             _obj({"case_id": CASE_ID, "exclude_findings": {"type": "array", "items": {"type": "string"}}}, ["case_id"]), calculate_triage_score),
        Tool("apply_queue_filter", "ui", "ui", "Filter the investigator's queue view.", _obj({"filters": FILTER_SCHEMA}, ["filters"]), apply_queue_filter),
        Tool("accept_finding", "workflow", "write", "Propose accepting a finding. Needs investigator confirmation.",
             _obj({"case_id": CASE_ID, "finding_id": {"type": "string"}}, ["case_id", "finding_id"]), accept_finding),
        Tool("reject_finding", "workflow", "write", "Propose rejecting a finding with a reason. Needs investigator confirmation.",
             _obj({"case_id": CASE_ID, "finding_id": {"type": "string"}, "reason": {"type": "string"}}, ["case_id", "finding_id", "reason"]), reject_finding),
        Tool("add_case_note", "workflow", "write", "Propose a case note (for example a drafted summary). Needs investigator confirmation.",
             _obj({"case_id": CASE_ID, "note": {"type": "string"}}, ["case_id", "note"]), add_case_note),
        Tool("set_case_status", "workflow", "write", "Propose a status change. Needs investigator confirmation.",
             _obj({"case_id": CASE_ID, "status": {"type": "string", "enum": CASE_STATUSES}, "reason": {"type": "string"}},
                  ["case_id", "status"]), set_case_status),
    ]
    return {t.name: t for t in tools}


def run_tool(tools: dict[str, Tool], name: str, args: dict) -> tuple[Any, str | None]:
    tool = tools.get(name)
    if tool is None:
        return None, f"Tool '{name}' is not allowlisted."
    try:
        return tool.fn(**(args or {})), None
    except TypeError as e:
        return None, f"Bad arguments for {name}: {e}"
    except Exception as e:  # tools raise ValueError for bad input
        return None, str(e)


def to_json(obj) -> str:
    return json.dumps(obj, default=str)[:12000]
