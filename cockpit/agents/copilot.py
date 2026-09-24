"""Investigator Copilot agent (spec 5.5 and 10.5).

Context-aware: selected case, current tab, active filters, selected chart point/group and
recent feedback are passed in on every turn. The agent uses allowlisted tools only, has a
tool-call cap and a timeout, validates grounding, and never writes directly.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field

from ..config import settings
from ..llm.providers import LLMError, get_provider
from . import grounding
from .prompts import COPILOT_SYSTEM
from .tools import build_tools, run_tool, to_json


@dataclass
class CopilotResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    proposals: list[dict] = field(default_factory=list)
    ui_filters: dict | None = None
    grounding: dict = field(default_factory=dict)
    source: str = "llm"


class CopilotAgent:
    def __init__(self, service, max_history: int = 8):
        self.svc = service
        self.provider = get_provider()
        self.tools = build_tools(service)
        self.max_history = max_history

    def _context_block(self, ctx: dict) -> str:
        c = {
            "selected_case_id": ctx.get("case_id"),
            "current_tab": ctx.get("tab"),
            "active_filters": ctx.get("filters") or {},
            "selected_chart": ctx.get("chart"),
            "selected_chart_point": ctx.get("chart_selection"),
            "queue_rows_visible": ctx.get("visible_count"),
        }
        if ctx.get("case_id"):
            c["recent_feedback_on_case"] = self.svc.repo.finding_decisions(ctx["case_id"])
        return "Context:\n" + json.dumps(c, default=str)

    def ask(self, question: str, ctx: dict, history: list[dict] | None = None) -> CopilotResult:
        if not settings.enable_llm or not self.provider.available:
            return FallbackCopilot(self.svc, self.tools).ask(question, ctx)

        run = {"run_id": f"R-{uuid.uuid4().hex[:10]}", "agent": "copilot", "case_id": ctx.get("case_id"),
               "provider": self.provider.name, "model": self.provider.model, "question": question,
               "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
        msgs = [m for m in (history or [])[-self.max_history:] if m["role"] in ("user", "assistant")]
        msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
        msgs.append({"role": "user", "content": self._context_block(ctx) + "\n\nQuestion: " + question})
        specs = [t.spec() for t in self.tools.values()]
        calls, evidence, proposals, ui_filters = [], [self._context_block(ctx)], [], None
        deadline = time.time() + settings.llm_timeout_s * 2
        answer, retried = "", False

        try:
            while True:
                allow_tools = len(calls) < settings.agent_max_tool_calls and time.time() < deadline
                resp = self.provider.chat(COPILOT_SYSTEM, msgs, tools=specs if allow_tools else None, max_tokens=900)
                run["input_tokens"] += resp.input_tokens
                run["output_tokens"] += resp.output_tokens
                run["latency_ms"] += resp.latency_ms
                if resp.tool_calls and allow_tools:
                    msgs.append({"role": "assistant", "content": resp.text,
                                 "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls]})
                    for tc in resp.tool_calls:
                        result, err = run_tool(self.tools, tc.name, tc.args)
                        payload = {"error": err} if err else result
                        calls.append({"tool": tc.name, "args": tc.args, "result": to_json(payload)[:500]})
                        evidence.append(payload)
                        if isinstance(result, dict) and result.get("ui_action") == "proposal":
                            proposals.append(result)
                        if isinstance(result, dict) and result.get("ui_action") == "apply_filter":
                            ui_filters = result["filters"]
                        msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": to_json(payload)})
                    continue
                answer = resp.text.strip()
                g = grounding.check(answer, evidence, known_case_ids=self.svc.by_id,
                                    known_fields=set(self.tools) | {f for f in self.svc.rules.signals}
                                    | {c["id"] for c in self.svc.rules.ruleset["combinations"]})
                if g.passed or retried:
                    break
                retried = True
                msgs += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": "Validation found unsupported content:\n- " + "\n- ".join(g.issues)
                          + "\nRewrite the answer using only tool results. Call a tool if you need a value."}]
        except LLMError as e:
            run.update({"error": str(e), "fallback_used": True, "grounding_passed": True})
            self.svc.repo.log_agent_run(run, calls)
            res = FallbackCopilot(self.svc, self.tools).ask(question, ctx)
            res.answer = f"The LLM call failed ({e.__class__.__name__}); showing a deterministic answer.\n\n" + res.answer
            return res

        run.update({"answer": answer, "grounding_passed": g.passed, "grounding_issues": g.issues, "fallback_used": False})
        self.svc.repo.log_agent_run(run, calls)
        return CopilotResult(answer=answer, tool_calls=calls, proposals=proposals, ui_filters=ui_filters,
                             grounding=g.to_dict(), source=f"{self.provider.name}:{self.provider.model}")


class FallbackCopilot:
    """Keyword intent router used when no LLM is configured. Same tools, template answers."""

    def __init__(self, service, tools):
        self.svc = service
        self.tools = tools

    def _call(self, calls, name, **args):
        result, err = run_tool(self.tools, name, args)
        calls.append({"tool": name, "args": args, "result": to_json(result if not err else {"error": err})[:500]})
        return result if not err else {"error": err}

    def ask(self, question: str, ctx: dict) -> CopilotResult:
        q = question.lower()
        calls: list[dict] = []
        m = re.search(r"\bC\d{4}\b", question.upper())
        cid = m.group(0) if m else ctx.get("case_id")
        filters = None
        lines: list[str] = []

        if re.search(r"prior (fraud|investigation)|fraud history|previous(ly)? (investigat|fraud)|paid twice|"
                     r"payment record|was .* paid|provider('s)? name|member('s)? name|who is the (provider|member)|diagnos",
                     q):
            lines.append("That is not in the supplied data. The file has case-level signals only: no payment records, "
                         "investigation history, names or diagnoses. Check the claims or case-management system. "
                         "(prior_claims_last_12mo is a claim count, not an investigation history.)")
        elif re.search(r"against|mitigat|false positive|benign|argue|innocent|legitimate", q) and cid:
            a = self._call(calls, "get_assessment", case_id=cid)
            if a["mitigating"]:
                lines.append(f"Points in the data that argue for a lower priority on {cid}:")
                lines += [f"- {x}" for x in a["mitigating"]]
            else:
                lines.append(f"Nothing in the data argues for a lower priority on {cid}; every major signal is elevated.")
            if a["caveats"]:
                lines.append("Alternative explanations to rule out before escalating:")
                lines += [f"- {x}" for x in a["caveats"]]
        elif re.search(r"\b(first|next)\b|do now|what action", q) and cid:
            a = self._call(calls, "get_assessment", case_id=cid)
            lines.append(f"Verify first on {cid}: {a['recommended_next_step']}")
        elif re.search(r"polic|procedure|playbook|guidance|how (do|should|can) i|how to|escalat", q):
            r = self._call(calls, "search_knowledge", query=question, case_id=cid) if cid else \
                self._call(calls, "search_knowledge", query=question)
            lines.append("Relevant guidance (synthetic demonstration playbooks, not approved policy):")
            lines += [f"- **{g['citation']}**: {g['text']}" for g in r["playbook_sections"][:3]]
        elif re.search(r"high amount|large amount|expensive", q) and re.search(r"low|green|lower", q):
            filters = {"priority_level": ["green", "yellow"], "min_amount": 10000}
            r = self._call(calls, "apply_queue_filter", filters=filters)
            lines.append(f"{r['matching_cases']} cases have a claim amount of at least $10,000 but are not high priority: "
                         f"{', '.join(r['case_ids'])}. The queue is filtered to them.")
        elif re.search(r"summar(y|ise|ize)|overview|how many cases", q) and ("queue" in q or not cid):
            s = self._call(calls, "get_queue_summary")
            lines.append(f"The queue has {s['total_cases']} cases: {s['high_priority']} high-priority, "
                         f"{s['needs_attention']} need attention, {s['lower_priority']} lower priority and "
                         f"{s['data_review']} need data review. {s['reviewed']} have been reviewed.")
        elif re.search(r"co-?occur|together|pairs? of (signals|findings)", q):
            r = self._call(calls, "get_signal_cooccurrence", filters=ctx.get("filters") or {})
            lines.append("Findings that most often occur together:")
            lines += [f"- {' + '.join(p['signals'])}: {p['cases']} cases" for p in r["pairs"][:6]]
        elif "chart" in q and ctx.get("chart"):
            d = self._call(calls, "get_chart_data", chart_id=ctx["chart"], filters=ctx.get("filters") or {})
            lines.append(f"Chart data for {ctx['chart']}: {json.dumps(d['data'], default=str)[:600]}")
        elif not cid:
            lines.append("Select a case or mention a case ID (for example C1024), or ask for a queue summary.")
        elif re.search(r"similar|like this", q):
            r = self._call(calls, "find_similar_cases", case_id=cid, limit=5)
            for s in r["similar"]:
                lines.append(f"- {s['case_id']} ({s['care_type']}, {s['priority_level']}, score {s['score']}); "
                             f"shared flags: {', '.join(s['shared_flags']) or 'none'}; status: {s['investigator_status']}")
            lines.insert(0, f"Most similar cases to {cid} by signal profile:")
        elif re.search(r"compare|peer|unusual|typical|care type", q):
            r = self._call(calls, "compare_case_to_cohort", case_id=cid)
            lines.append(r["note"])
            for row in r["continuous"]:
                if row["meaning"] != "Typical":
                    lines.append(f"- {row['label']}: {row['selected']} vs peer median {row['peer_median']} ({row['meaning'].lower()})")
            if len(lines) == 1:
                lines.append("All continuous signals are typical for the peer group.")
        elif re.search(r"verify|next|first|do now|action", q):
            a = self._call(calls, "get_assessment", case_id=cid)
            lines.append(f"Verify first on {cid}: {a['recommended_next_step']}")
        elif re.search(r"data.quality|data issue|duplicate claim", q):
            a = self._call(calls, "get_assessment", case_id=cid)
            dq = a["data_quality_warnings"]
            lines.append(" ".join(dq) if dq else f"No data-quality warnings on {cid}.")
        elif re.search(r"history|audit|what happened", q):
            h = self._call(calls, "get_case_history", case_id=cid)
            lines.append(f"{cid} status: {h['status']}.")
            lines += [f"- {e['kind']}: {e['detail']}" for e in h["audit_trail"][-8:]]
        else:
            a = self._call(calls, "get_assessment", case_id=cid)
            lines.append(f"{cid} is {a['priority_label'].lower()} with score {a['score']} and {a['confidence'].lower()} confidence.")
            top = [f for f in a["findings"] if f["points"] > 0][:4]
            if top:
                lines.append("Largest contributions:")
                lines += [f"- {f['label']}: {f['points']} points ({f['observed']})" for f in top]
            if a["strong_indicators"]:
                lines.append("Policy strong indicator: " + "; ".join(a["strong_indicators"]))
            lines.append(f"Next step: {a['recommended_next_step']}")

        answer = "\n".join(lines)
        run = {"run_id": f"R-{uuid.uuid4().hex[:10]}", "agent": "copilot", "case_id": cid, "provider": "none",
               "model": "deterministic", "question": question, "answer": answer, "grounding_passed": True,
               "fallback_used": True}
        self.svc.repo.log_agent_run(run, calls)
        return CopilotResult(answer=answer, tool_calls=calls, ui_filters=filters,
                             grounding={"passed": True, "issues": [], "checked_numbers": 0}, source="deterministic")


def explain_validation(service) -> str:
    """Data Quality agent (spec 10.1). LLM if available, otherwise deterministic."""
    from .prompts import DATA_QUALITY_SYSTEM
    rep = service.report.to_dict()
    provider = get_provider()
    if settings.enable_llm and provider.available:
        try:
            resp = provider.chat(DATA_QUALITY_SYSTEM, [{"role": "user", "content": json.dumps(rep)}], max_tokens=400)
            service.repo.log_agent_run({"run_id": f"R-{uuid.uuid4().hex[:10]}", "agent": "data_quality",
                                        "provider": provider.name, "model": provider.model, "answer": resp.text,
                                        "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                                        "latency_ms": resp.latency_ms, "grounding_passed": True}, [])
            return resp.text.strip()
        except LLMError:
            pass
    parts = []
    if rep["blocking_errors"]:
        parts.append("Blocking: " + "; ".join(rep["blocking_errors"]) + ".")
    else:
        parts.append(f"All {rep['rows_loaded']} rows loaded and all required fields are present.")
    for num, ids in rep["duplicate_claim_numbers"].items():
        parts.append(f"Claim number {num} appears on {', '.join(ids)}. Confirm with claims operations whether "
                     "this is a keying error or a resubmission. It is not treated as evidence of fraud.")
    if rep["invalid_values"]:
        parts.append(f"{len(rep['invalid_values'])} invalid values need correcting.")
    if rep["missing_values"]:
        parts.append(f"{rep['missing_values']} missing values; affected cases get lower confidence.")
    return " ".join(parts)
