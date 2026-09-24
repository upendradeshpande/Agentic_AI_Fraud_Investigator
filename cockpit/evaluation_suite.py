"""Executable evaluation suite (spec section 21).

    python -m cockpit.evaluation_suite           # deterministic checks, no LLM calls, exit 1 on failure
    python -m cockpit.evaluation_suite --live    # also grade the configured LLM copilot on the golden set (billable)

Layers
  1. Data quality   validation of the supplied file plus adversarial input fixtures
  2. Scoring        per-case invariants (bounds, accounting, caps, source values, strong indicators,
                    rejection monotonicity, determinism)
  3. Agent          golden questions: expected tool, expected content, abstention (spec 21.3)
  4. Grounding      validator must-pass / must-fail fixtures, and every deterministic narrative for every
                    case must pass the validator against its own evidence pack (spec 21.4)
  5. Retrieval      playbook Recall@3 for known queries
Human-feedback metrics (spec 21.5) are reported, not graded: acceptance is not correctness.

What this does NOT measure: fraud accuracy on the supplied data (no labels), answer quality judged by
experts, or time saved. The synthetic-set comparison in the AI evaluation tab covers ranking quality
against documented assumptions only.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

from . import knowledge_base
from .agents import grounding
from .agents.assessment_agent import AnalysisAgent, deterministic_narrative
from .agents.copilot import FallbackCopilot
from .agents.tools import build_tools
from .config import ROOT, settings
from .data_loader import load_file

GOLDEN = [
    # question, case, expected tool (None = must not call a tool), phrases that must appear (any-of), kind
    ("Why is this case high priority?", "C1024", "get_assessment", ["score"], "explain"),
    ("How does it compare with similar cases?", "C1024", "find_similar_cases", ["C10"], "similar"),
    ("Is this signal unusual for this care type?", "C1024", "compare_case_to_cohort", ["peer"], "peer"),
    ("What evidence argues against escalation?", "C1024", "get_assessment", ["rule out", "lower priority"], "counter"),
    ("What should I verify first?", "C1024", "get_assessment", ["<current next step>"], "next_step"),
    ("Could this be a data-quality issue?", "C1001", "get_assessment", ["claim number"], "data_quality"),
    ("How should I verify the duplicate billing?", "C1024", "search_knowledge", ["PB-DUP-01"], "guidance"),
    ("Show cases with high amounts but lower priority", None, "apply_queue_filter", ["cases"], "filter"),
    ("Did this member have prior fraud investigations?", "C1024", None, ["not in the supplied data"], "abstain"),
    ("Was this claim paid twice?", "C1024", None, ["not in the supplied data"], "abstain"),
    ("Summarise the queue", None, "get_queue_summary", ["high-priority"], "queue"),
    ("Which signals co-occur most?", None, "get_signal_cooccurrence", ["cases"], "cooccurrence"),
]

GROUNDING_FIXTURES = [
    # text, should_pass, description
    ("C1024 bills 53% of visits on weekends and has a score of 100.", True, "percent form of a ratio"),
    ("The claim amount is $66,655.", True, "currency formatting"),
    ("C1024 has 37 weekly visits.", False, "invented number"),
    ("Compare it with C9999.", False, "non-existent case"),
    ("Check the provider_license_number field.", False, "non-existent field"),
    ("This case is fraud.", False, "fraud determination"),
    ("There is a 90% probability of fraud.", False, "fraud probability"),
    ("Dr. Smith signed the claim.", False, "invented person"),
]

RETRIEVAL_FIXTURES = [
    ("duplicate line payment correction reversal", "PB-DUP-01"),
    ("shared phone address relationship", "PB-REL-01"),
    ("overlapping services two providers", "PB-OVL-01"),
    ("repeated claim number", "PB-DQ-01"),
    ("high need member many visits", "PB-UTL-01"),
    ("when to escalate or close a case", "PB-ESC-01"),
    ("round dollar weekend billing", "PB-BIL-01"),
]


SAMPLE_CAP = 500


def _sample(cases: list[dict]) -> list[dict]:
    """Every case for small queues; a fixed spread of SAMPLE_CAP cases for large ones."""
    if len(cases) <= SAMPLE_CAP:
        return cases
    step = len(cases) / SAMPLE_CAP
    return [cases[int(i * step)] for i in range(SAMPLE_CAP)]


class Suite:
    def __init__(self):
        self.checks: list[dict] = []

    def check(self, layer: str, name: str, passed: bool, detail: str = ""):
        self.checks.append({"layer": layer, "name": name, "passed": bool(passed), "detail": detail[:300]})

    def summary(self) -> dict:
        layers: dict = {}
        for c in self.checks:
            s = layers.setdefault(c["layer"], {"passed": 0, "total": 0})
            s["total"] += 1
            s["passed"] += c["passed"]
        return layers


def _data_quality(s: Suite):
    L = "1 data quality"
    df, rep = load_file(str(settings.sample_csv))
    s.check(L, "sample: 50 rows loaded", rep.rows_loaded == 50, str(rep.rows_loaded))
    s.check(L, "sample: required fields present", rep.required_fields_ok)
    s.check(L, "sample: repeated claim number detected", "LTC-2034786" in rep.duplicate_claim_numbers)
    s.check(L, "sample: repeated claim number is a warning, not blocking", not rep.blocking_errors)
    text = settings.sample_csv.read_text(encoding="utf-8-sig").splitlines()
    header = text[0].split(",")

    def mutate(col, value):
        rows = [r.split(",") for r in text]
        rows[1][header.index(col)] = value
        return io.StringIO("\n".join(",".join(r) for r in rows))

    fixtures = [
        ("empty file is blocked", io.BytesIO(b""), "e.csv", lambda r: bool(r.blocking_errors)),
        ("missing column is blocked", io.StringIO("case_id,claim_number\nC1,X\n"), "m.csv", lambda r: bool(r.blocking_errors)),
        ("unsupported extension is blocked", io.BytesIO(b"x"), "x.pdf", lambda r: bool(r.blocking_errors)),
        ("ratio above 1 is reported", mutate("weekend_billing_ratio", "1.7"), "r.csv",
         lambda r: any("weekend_billing_ratio" in v for v in r.invalid_values)),
        ("infinite value is reported", mutate("weekly_visit_frequency", "inf"), "i.csv",
         lambda r: any("finite" in v for v in r.invalid_values)),
        ("fractional count is reported", mutate("prior_claims_last_12mo", "2.5"), "f.csv",
         lambda r: any("whole number" in v for v in r.invalid_values)),
        ("non-binary flag is reported", mutate("duplicate_service_billed", "3"), "b.csv",
         lambda r: any("expected 0 or 1" in v for v in r.invalid_values)),
    ]
    for name, src, fname, ok in fixtures:
        _, r = load_file(src, fname)
        s.check(L, name, ok(r), "; ".join(r.blocking_errors + r.invalid_values)[:200])
    leading = mutate("case_id", "00999")
    d2, _ = load_file(leading, "z.csv")
    s.check(L, "leading zeros in case_id preserved", "00999" in set(d2["case_id"]))


def _scoring(s: Suite, svc):
    L = "2 scoring"
    caps = {k: v["cap"] for k, v in svc.rules.ruleset["groups"].items()}
    sample = _sample(svc.cases)
    assessments = {c["case_id"]: svc.get_assessment(c["case_id"])["assessment"] for c in sample}
    rescored = svc.rules.score_cases(sample)
    for c in sample:
        cid = c["case_id"]
        a = assessments[cid]
        base = svc.rules.assess(c)  # without human feedback, for invariants
        ok = 0 <= a.score <= 100
        ok &= abs(sum(base.group_contributions.values()) - base.score) < 0.2
        ok &= all(base.group_contributions[g] <= caps[g] + 1e-6 for g in base.group_contributions)
        ok &= all(f.observed == c[f.finding_id] for f in base.findings if f.kind != "combination")
        ok &= all(col in c for f in base.findings for col in f.source_columns)
        ok &= rescored[cid] == base.score
        if base.strong_indicators:
            ok &= base.priority_level == "red"
        top = next((f for f in base.findings if f.points > 0), None)
        if top:
            ok &= svc.rules.assess(c, excluded={top.finding_id}).score <= base.score
        ok &= a.ruleset_id == svc.rules.ruleset_id and a.model_version == svc.rules.model_version
        s.check(L, f"{cid}: bounds, accounting, caps, source values, strong indicator, rejection monotonic, versions",
                ok, f"score {a.score}")
    c1001 = svc.get_assessment("C1001")["assessment"] if "C1001" in svc.by_id else None
    s.check(L, "C1001 (repeated claim number) routed to data review", c1001 and c1001.priority_level == "data_review")


def _agent(s: Suite, svc):
    L = "3 agent (deterministic router)"
    router = FallbackCopilot(svc, build_tools(svc))
    for q, cid, tool, phrases, kind in GOLDEN:
        if kind == "next_step":  # expected answer follows the case's current state, including investigator feedback
            phrases = [svc.get_assessment(cid)["assessment"].recommended_next_step]
        r = router.ask(q, {"case_id": cid})
        used = [t["tool"] for t in r.tool_calls]
        tool_ok = (not used) if tool is None else (tool in used)
        text_ok = any(p.lower() in r.answer.lower() for p in phrases)
        s.check(L, f"[{kind}] {q}", tool_ok and text_ok, f"tools={used}; answer={r.answer[:120]}")


def _agent_live(s: Suite, svc):
    L = "3b agent (live LLM)"
    from .agents.adk_runtime import make_copilot
    agent = make_copilot(svc)
    for q, cid, tool, phrases, kind in GOLDEN:
        t0 = time.time()
        r = agent.ask(q, {"case_id": cid, "tab": "Analysis"})
        used = [t["tool"] for t in r.tool_calls]
        tool_ok = True if tool is None else tool in used
        s.check(L, f"[{kind}] {q}", tool_ok and r.grounding.get("passed", False),
                f"{int((time.time() - t0) * 1000)} ms; tools={used}; grounded={r.grounding.get('passed')}; {r.answer[:100]}")


def _grounding(s: Suite, svc):
    L = "4 grounding"
    ev = {"case": svc.by_id["C1024"], "score": 100}
    for text, should, desc in GROUNDING_FIXTURES:
        g = grounding.check(text, ev, known_case_ids=set(svc.by_id))
        s.check(L, f"validator: {desc} -> {'pass' if should else 'fail'}", g.passed == should, "; ".join(g.issues))
    known = {f for f in svc.rules.signals} | {c["id"] for c in svc.rules.ruleset["combinations"]}
    analysis = AnalysisAgent(svc)
    for c in _sample(svc.cases):
        a = svc.get_assessment(c["case_id"])["assessment"]
        n = deterministic_narrative(a, c)
        text = " ".join([n["summary"], n["next_step"], n["uncertainty"], *n["key_points"], *n["mitigating"]])
        g = grounding.check(text, analysis.evidence_pack(c["case_id"], a), known_case_ids=set(svc.by_id), known_fields=known)
        s.check(L, f"{c['case_id']}: deterministic narrative is grounded in its evidence pack", g.passed, "; ".join(g.issues))


def _retrieval(s: Suite):
    L = "5 retrieval"
    for q, doc in RETRIEVAL_FIXTURES:
        got = [r["doc_id"] for r in knowledge_base.search(q, (), 3)]
        s.check(L, f"Recall@3 '{q}' -> {doc}", doc in got, str(got))


def run(svc=None, live: bool = False) -> dict:
    from .service import CockpitService
    t0 = time.time()
    if svc is None:
        svc = CockpitService()
        svc.ensure_loaded()
    s = Suite()
    _data_quality(s)
    _scoring(s, svc)
    _agent(s, svc)
    if live:
        _agent_live(s, svc)
    _grounding(s, svc)
    _retrieval(s)
    summary = s.summary()
    result = {
        "suite_version": "eval-suite-1.0",
        "ruleset_id": svc.rules.ruleset_id, "model_version": svc.rules.model_version,
        "llm": "live" if live else "not called",
        "passed": sum(c["passed"] for c in s.checks), "total": len(s.checks),
        "layers": summary, "failures": [c for c in s.checks if not c["passed"]],
        "human_feedback": {k: v for k, v in svc.evaluation_summary().items()
                           if k in ("accepted_findings", "rejected_findings", "human_overrides", "final_dispositions",
                                    "most_rejected_signals", "rejection_reasons", "grounding_pass_rate")},
        "not_measured": ["fraud accuracy on the supplied data (no labels)", "expert-judged answer quality",
                         "investigator time saved"],
        "seconds": round(time.time() - t0, 2), "checks": s.checks,
    }
    svc.repo.save_evaluation("evaluation-suite", {k: v for k, v in result.items() if k != "checks"} | {"checks": s.checks})
    return result


def main():
    ap = argparse.ArgumentParser(description="Run the cockpit evaluation suite.")
    ap.add_argument("--live", action="store_true", help="Also grade the configured LLM copilot (makes API calls).")
    ap.add_argument("--out", default=str(ROOT / "data" / "evaluation_results.json"))
    args = ap.parse_args()
    res = run(live=args.live)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    for layer, v in sorted(res["layers"].items()):
        print(f"  {layer:34s} {v['passed']:>4}/{v['total']}")
    print(f"TOTAL {res['passed']}/{res['total']} in {res['seconds']}s  ->  {args.out}")
    for f in res["failures"][:20]:
        print(f"  FAIL [{f['layer']}] {f['name']}: {f['detail']}")
    sys.exit(0 if not res["failures"] else 1)


if __name__ == "__main__":
    main()
