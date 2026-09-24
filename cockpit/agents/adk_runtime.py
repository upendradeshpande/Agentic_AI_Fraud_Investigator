"""Optional Google ADK runtime for the copilot (spec 10: "Google ADK is the agent orchestration layer").

Enable with AGENT_RUNTIME=adk and `pip install google-adk`. Uses the same allowlisted tools,
the same prompt and the same grounding validator as the native runtime, so the UI does not
change. Gemini models run natively; other providers go through ADK's LiteLLM wrapper
(`pip install litellm`), for example LLM_MODEL=anthropic/claude-sonnet-5.

Status: written against the google-adk 1.x API (LlmAgent, Runner, InMemorySessionService).
The native runtime in copilot.py is the tested default.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

from ..config import settings
from ..llm.providers import get_provider
from . import grounding
from .copilot import CopilotResult, FallbackCopilot
from .prompts import COPILOT_SYSTEM
from .tools import build_tools, run_tool, to_json


def _adk_tools(svc, calls: list, proposals: list, ui: dict):
    tools = build_tools(svc)

    def call(name, **args):
        args = {k: v for k, v in args.items() if v not in (None, "", [])}
        result, err = run_tool(tools, name, args)
        payload = {"error": err} if err else result
        calls.append({"tool": name, "args": args, "result": to_json(payload)[:500], "_payload": payload})
        if isinstance(result, dict) and result.get("ui_action") == "proposal":
            proposals.append(result)
        if isinstance(result, dict) and result.get("ui_action") == "apply_filter":
            ui["filters"] = result["filters"]
        return payload

    def _filters(priority_level, care_type, min_amount, max_amount):
        f = {}
        if priority_level:
            f["priority_level"] = [p.strip() for p in priority_level.split(",")]
        if care_type:
            f["care_type"] = [c.strip() for c in care_type.split(",")]
        if min_amount:
            f["min_amount"] = min_amount
        if max_amount:
            f["max_amount"] = max_amount
        return f

    def get_queue_summary() -> dict:
        """Counts by priority, statuses and claim amounts for the whole queue."""
        return call("get_queue_summary")

    def list_cases(priority_level: str = "", care_type: str = "", min_amount: float = 0,
                   max_amount: float = 0, sort: str = "", limit: int = 15) -> dict:
        """List cases in queue order. priority_level and care_type accept comma-separated values
        (priority levels: red, yellow, green, data_review). sort: score or claim_amount_usd."""
        return call("list_cases", filters=_filters(priority_level, care_type, min_amount, max_amount),
                    sort=sort or None, limit=limit)

    def get_case(case_id: str) -> dict:
        """Raw attributes and signal values for one case, e.g. C1024."""
        return call("get_case", case_id=case_id)

    def get_assessment(case_id: str) -> dict:
        """Current triage assessment: score, priority, confidence, findings with points, next step."""
        return call("get_assessment", case_id=case_id)

    def get_signal_definition(signal_name: str) -> dict:
        """Definition and scoring rule for a signal column or combination id."""
        return call("get_signal_definition", signal_name=signal_name)

    def compare_case_to_cohort(case_id: str, cohort: str = "care_type") -> dict:
        """Compare a case with peers of the same care_type or state."""
        return call("compare_case_to_cohort", case_id=case_id, cohort=cohort)

    def find_similar_cases(case_id: str, limit: int = 5) -> dict:
        """Nearest cases by signal profile with their priority and investigator status."""
        return call("find_similar_cases", case_id=case_id, limit=limit)

    def get_chart_data(chart_id: str) -> dict:
        """Data behind a chart: priority_by_care_type, amount_vs_score, signal_frequency, signal_cooccurrence."""
        return call("get_chart_data", chart_id=chart_id)

    def get_case_history(case_id: str) -> dict:
        """Status and audit trail for a case."""
        return call("get_case_history", case_id=case_id)

    def calculate_triage_score(case_id: str, exclude_findings: list[str]) -> dict:
        """What-if score excluding finding ids. Does not save anything."""
        return call("calculate_triage_score", case_id=case_id, exclude_findings=exclude_findings)

    def apply_queue_filter(priority_level: str = "", care_type: str = "", min_amount: float = 0,
                           max_amount: float = 0) -> dict:
        """Filter the investigator's queue view."""
        return call("apply_queue_filter", filters=_filters(priority_level, care_type, min_amount, max_amount))

    def propose_reject_finding(case_id: str, finding_id: str, reason: str) -> dict:
        """Propose rejecting a finding. The investigator must confirm."""
        return call("reject_finding", case_id=case_id, finding_id=finding_id, reason=reason)

    def propose_case_note(case_id: str, note: str) -> dict:
        """Propose a case note. The investigator must confirm."""
        return call("add_case_note", case_id=case_id, note=note)

    def propose_case_status(case_id: str, status: str, reason: str = "") -> dict:
        """Propose a status change. The investigator must confirm."""
        return call("set_case_status", case_id=case_id, status=status, reason=reason)

    return [get_queue_summary, list_cases, get_case, get_assessment, get_signal_definition,
            compare_case_to_cohort, find_similar_cases, get_chart_data, get_case_history,
            calculate_triage_score, apply_queue_filter, propose_reject_finding, propose_case_note,
            propose_case_status]


class ADKCopilotAgent:
    def __init__(self, service):
        self.svc = service

    def _model(self):
        if settings.llm_provider == "gemini":
            return settings.llm_model
        from google.adk.models.lite_llm import LiteLlm  # requires litellm
        name = settings.llm_model if "/" in settings.llm_model else f"{settings.llm_provider}/{settings.llm_model}"
        return LiteLlm(model=name)

    async def _run(self, question: str, ctx: dict, calls, proposals, ui) -> str:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        agent = LlmAgent(name="investigator_copilot", model=self._model(), instruction=COPILOT_SYSTEM,
                         tools=_adk_tools(self.svc, calls, proposals, ui))
        sessions = InMemorySessionService()
        runner = Runner(agent=agent, app_name="cockpit", session_service=sessions)
        session = await sessions.create_session(app_name="cockpit", user_id="investigator")
        context = "Context:\n" + json.dumps({
            "selected_case_id": ctx.get("case_id"), "current_tab": ctx.get("tab"),
            "active_filters": ctx.get("filters") or {}, "selected_chart": ctx.get("chart"),
            "selected_chart_point": ctx.get("chart_selection")}, default=str)
        msg = types.Content(role="user", parts=[types.Part(text=context + "\n\nQuestion: " + question)])
        final = ""
        async for event in runner.run_async(user_id="investigator", session_id=session.id, new_message=msg):
            if len(calls) > settings.agent_max_tool_calls:
                break
            if event.is_final_response() and event.content and event.content.parts:
                final = "".join(p.text or "" for p in event.content.parts)
        return final

    def ask(self, question: str, ctx: dict, history: Optional[list] = None) -> CopilotResult:
        if not settings.enable_llm or not get_provider().available:
            return FallbackCopilot(self.svc, build_tools(self.svc)).ask(question, ctx)
        calls, proposals, ui = [], [], {}
        answer = asyncio.run(asyncio.wait_for(self._run(question, ctx, calls, proposals, ui),
                                              timeout=settings.llm_timeout_s * 2))
        evidence = [c.pop("_payload") for c in calls] + [ctx]
        g = grounding.check(answer, evidence, known_case_ids=self.svc.by_id,
                            known_fields=set(build_tools(self.svc)) | set(self.svc.rules.signals))
        self.svc.repo.log_agent_run({"run_id": f"R-{uuid.uuid4().hex[:10]}", "agent": "copilot",
                                     "case_id": ctx.get("case_id"), "provider": f"adk/{settings.llm_provider}",
                                     "model": settings.llm_model, "question": question, "answer": answer,
                                     "grounding_passed": g.passed, "grounding_issues": g.issues}, calls)
        return CopilotResult(answer=answer, tool_calls=calls, proposals=proposals, ui_filters=ui.get("filters"),
                             grounding=g.to_dict(), source=f"adk:{settings.llm_model}")


def make_copilot(service):
    """Factory used by the UI. AGENT_RUNTIME=adk selects Google ADK, otherwise the native runtime."""
    if settings.agent_runtime == "adk":
        try:
            import google.adk  # noqa: F401
            return ADKCopilotAgent(service)
        except ImportError:
            pass
    from .copilot import CopilotAgent
    return CopilotAgent(service)
