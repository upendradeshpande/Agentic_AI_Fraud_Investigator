"""Tests. Run with `pytest` or `python -m unittest discover tests`."""
import json
import os
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{TMP}/test.db"
os.environ["LLM_PROVIDER"] = "none"

from cockpit.agents import grounding  # noqa: E402
from cockpit.agents.copilot import CopilotAgent  # noqa: E402
from cockpit.config import settings  # noqa: E402
from cockpit.data_loader import load_file, records  # noqa: E402
from cockpit.db import Repository  # noqa: E402
from cockpit.llm import providers  # noqa: E402
from cockpit.models.rules_model import TransparentRulesModel  # noqa: E402
from cockpit.service import CockpitService  # noqa: E402


def fresh_service():
    path = tempfile.mktemp(suffix=".db", dir=TMP)
    svc = CockpitService(Repository(path))
    svc.ensure_loaded()
    return svc


class TestDataQuality(unittest.TestCase):
    def test_sample_file(self):
        df, rep = load_file(str(settings.sample_csv))
        self.assertEqual(rep.rows_loaded, 50)
        self.assertTrue(rep.required_fields_ok)
        self.assertEqual(rep.duplicate_claim_numbers, {"LTC-2034786": ["C1001", "C1031"]})
        self.assertEqual(rep.status, "ready with warnings")

    def test_invalid_values_reported_not_dropped(self):
        import io
        with open(settings.sample_csv, encoding="utf-8-sig") as fh:
            text = fh.read().splitlines()
        bad = text[1].split(",")
        bad[11] = "1.7"  # weekend ratio > 1
        text[1] = ",".join(bad)
        df, rep = load_file(io.StringIO("\n".join(text)), "x.csv")
        self.assertEqual(rep.rows_loaded, 50)
        self.assertTrue(any("weekend_billing_ratio" in v for v in rep.invalid_values))

    def test_missing_column_blocks(self):
        import io
        df, rep = load_file(io.StringIO("case_id,claim_number\nC1,X\n"), "x.csv")
        self.assertEqual(rep.status, "blocked")


class TestScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, _ = load_file(str(settings.sample_csv))
        cls.cases = records(df)
        cls.model = TransparentRulesModel().fit(cls.cases)

    def test_deterministic_and_bounded(self):
        s1 = self.model.score_cases(self.cases)
        s2 = TransparentRulesModel().fit(self.cases).score_cases(self.cases)
        self.assertEqual(s1, s2)
        self.assertTrue(all(0 <= v <= 100 for v in s1.values()))

    def test_contributions_add_up(self):
        for c in self.cases:
            a = self.model.assess(c)
            self.assertAlmostEqual(sum(a.group_contributions.values()), a.score, delta=0.2)
            self.assertAlmostEqual(sum(f.points for f in a.findings), a.score, delta=0.6)

    def test_group_caps_respected(self):
        caps = {k: v["cap"] for k, v in self.model.ruleset["groups"].items()}
        for c in self.cases:
            a = self.model.assess(c)
            for g, v in a.group_contributions.items():
                self.assertLessEqual(v, caps[g] + 1e-6)

    def test_strong_indicator_forces_red(self):
        c = dict(next(x for x in self.cases if x["case_id"] == "C1002"))
        c.update(duplicate_service_billed=1, service_overlap_other_provider=1)
        a = self.model.assess(c)
        self.assertEqual(a.priority_level, "red")
        self.assertTrue(a.strong_indicators)

    def test_rejection_lowers_or_keeps_score_and_removes_combo(self):
        c = next(x for x in self.cases if x["case_id"] == "C1050")
        base = self.model.assess(c)
        after = self.model.assess(c, excluded={"duplicate_service_billed"})
        self.assertLess(after.score, base.score)
        self.assertNotIn("combo_dup_overlap", [f.finding_id for f in after.findings if f.points > 0])
        self.assertFalse(after.strong_indicators)

    def test_data_quality_routes_green_to_data_review(self):
        c = next(x for x in self.cases if x["case_id"] == "C1001")
        a = self.model.assess(c, dq_warnings=["dup claim"])
        self.assertEqual(a.priority_level, "data_review")
        self.assertNotEqual(a.confidence, "High")


class TestWorkflow(unittest.TestCase):
    def test_reject_recalculates_and_audits(self):
        svc = fresh_service()
        before = svc.get_assessment("C1050")
        after = svc.reject_finding("C1050", "duplicate_service_billed", "Separate visits confirmed")
        self.assertLess(after["assessment"].score, before["assessment"].score)
        self.assertEqual(after["version"], before["version"] + 1)
        kinds = [e["kind"] for e in svc.repo.audit_trail("C1050")]
        self.assertIn("decision", kinds)
        self.assertEqual(kinds.count("assessment"), 2)
        restored = svc.reset_finding("C1050", "duplicate_service_billed")
        self.assertEqual(restored["assessment"].score, before["assessment"].score)

    def test_reject_needs_reason_and_final_status_needs_reason(self):
        svc = fresh_service()
        with self.assertRaises(ValueError):
            svc.reject_finding("C1024", "duplicate_service_billed", " ")
        with self.assertRaises(ValueError):
            svc.set_status("C1024", "Escalated")
        svc.set_status("C1024", "Escalated", "Overlapping dates confirmed")
        self.assertEqual(svc.repo.get_status("C1024"), "Escalated")

    def test_override_counted(self):
        svc = fresh_service()
        svc.set_status("C1024", "Likely false positive", "Known billing system glitch")
        self.assertEqual(svc.evaluation_summary()["human_overrides"], 1)

    def test_cache_reuse(self):
        svc = fresh_service()
        v = svc.get_assessment("C1003")["version"]
        svc.assess_all()
        self.assertEqual(svc.get_assessment("C1003")["version"], v)


class TestGrounding(unittest.TestCase):
    ev = {"case": {"case_id": "C1024", "weekend_billing_ratio": 0.53, "claim_amount_usd": 66655},
          "score": 100}

    def test_pass(self):
        g = grounding.check("C1024 bills 53% on weekends, $66,655 total, score 100.", self.ev, known_case_ids={"C1024"})
        self.assertTrue(g.passed, g.issues)

    def test_invented_number(self):
        g = grounding.check("C1024 has 37 visits.", self.ev, known_case_ids={"C1024"})
        self.assertFalse(g.passed)

    def test_invented_case_and_field(self):
        g = grounding.check("Compare with C9999 and its provider_license_number.", self.ev, known_case_ids={"C1024"})
        self.assertEqual(len(g.issues), 2)

    def test_fraud_claim(self):
        g = grounding.check("This case is fraud.", self.ev, known_case_ids={"C1024"})
        self.assertFalse(g.passed)


class FakeHTTP:
    """Returns a scripted sequence of provider-native JSON responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, url, headers=None, json=None, timeout=None, **kwargs):
        self.bodies.append(json)
        r = mock.Mock()
        r.status_code = 200
        r.json.return_value = self.responses.pop(0)
        return r


def native_responses(provider):
    if provider == "anthropic":
        return [
            {"content": [{"type": "tool_use", "id": "t1", "name": "get_assessment", "input": {"case_id": "C1024"}}],
             "usage": {"input_tokens": 100, "output_tokens": 20}},
            {"content": [{"type": "text", "text": "C1024 is high priority with score 100."}],
             "usage": {"input_tokens": 300, "output_tokens": 30}},
        ]
    if provider == "openai":
        return [
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "get_assessment", "arguments": '{"case_id": "C1024"}'}}]}}],
             "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
            {"choices": [{"message": {"content": "C1024 is high priority with score 100."}}],
             "usage": {"prompt_tokens": 300, "completion_tokens": 30}},
        ]
    return [
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_assessment", "args": {"case_id": "C1024"}}}]}}],
         "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}},
        {"candidates": [{"content": {"parts": [{"text": "C1024 is high priority with score 100."}]}}],
         "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 30}},
    ]


class TestProviders(unittest.TestCase):
    def _run(self, name, env):
        svc = fresh_service()
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(providers.requests, "post", FakeHTTP(native_responses(name))) as fake, \
             mock.patch("cockpit.agents.copilot.settings", mock.Mock(enable_llm=True, llm_timeout_s=30, agent_max_tool_calls=6)):
            agent = CopilotAgent(svc)
            agent.provider = providers.PROVIDERS[name]("test-model")
            res = agent.ask("Why is this high priority?", {"case_id": "C1024", "tab": "Analysis"})
        self.assertEqual(res.tool_calls[0]["tool"], "get_assessment")
        self.assertIn("score 100", res.answer)
        self.assertTrue(res.grounding["passed"], res.grounding)
        return fake.bodies

    def test_anthropic(self):
        bodies = self._run("anthropic", {"ANTHROPIC_API_KEY": "x"})
        self.assertEqual(bodies[1]["messages"][-1]["content"][0]["type"], "tool_result")

    def test_openai(self):
        bodies = self._run("openai", {"OPENAI_API_KEY": "x"})
        self.assertEqual(bodies[1]["messages"][-1]["role"], "tool")

    def test_gemini(self):
        bodies = self._run("gemini", {"GEMINI_API_KEY": "x"})
        self.assertIn("functionResponse", bodies[1]["contents"][-1]["parts"][0])
        self.assertEqual(bodies[0]["tools"][0]["functionDeclarations"][0]["parameters"]["type"], "OBJECT")

    def test_parse_json(self):
        self.assertEqual(providers.parse_json('```json\n{"a": 1}\n```'), {"a": 1})


class TestAssessmentAgent(unittest.TestCase):
    def test_bad_llm_output_falls_back(self):
        from cockpit.agents.assessment_agent import AssessmentAgent
        svc = fresh_service()
        bad = {"content": [{"type": "text", "text": json.dumps({
            "summary": "C1024 is fraud with 97% probability of fraud.", "key_points": [], "mitigating": [],
            "next_step": "x", "uncertainty": "y", "cited_findings": ["made_up"]})}], "usage": {}}
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}), \
             mock.patch.object(providers.requests, "post", FakeHTTP([bad, bad])), \
             mock.patch("cockpit.agents.assessment_agent.settings", mock.Mock(enable_llm=True)):
            agent = AssessmentAgent(svc)
            agent.provider = providers.AnthropicProvider("test-model")
            narrative, model = agent.run("C1024", svc.get_assessment("C1024")["assessment"])
        self.assertTrue(model.startswith("deterministic"))
        self.assertFalse(narrative["grounding"]["passed"])

    def test_good_llm_output_accepted(self):
        from cockpit.agents.assessment_agent import AssessmentAgent
        svc = fresh_service()
        good = {"content": [{"type": "text", "text": json.dumps({
            "summary": "High-priority review with score 100. Duplicate billing and service overlap are both flagged.",
            "key_points": ["Duplicate service billed is flagged."], "mitigating": [],
            "next_step": "Pull service-line records from both providers.",
            "uncertainty": "No service-line records are available.",
            "cited_findings": ["duplicate_service_billed"]})}], "usage": {"input_tokens": 5, "output_tokens": 5}}
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}), \
             mock.patch.object(providers.requests, "post", FakeHTTP([good])), \
             mock.patch("cockpit.agents.assessment_agent.settings", mock.Mock(enable_llm=True)):
            agent = AssessmentAgent(svc)
            agent.provider = providers.AnthropicProvider("test-model")
            narrative, model = agent.run("C1024", svc.get_assessment("C1024")["assessment"])
        self.assertEqual(narrative["source"], "llm")
        self.assertTrue(model.startswith("anthropic"))


class TestDevOnlySupervisedPipeline(unittest.TestCase):
    """The supervised pipeline still works for development, but the product never uses it."""

    def test_offline_pipeline_runs(self):
        from cockpit.models.supervised_model import evaluate_all
        df, _ = load_file(str(settings.sample_csv))
        m = evaluate_all(records(df), n=1500)["results"]["models"]
        self.assertIn("gradient_boosting", m)
        self.assertGreater(m["transparent_rules"]["roc_auc"], 0.7)

    def test_app_never_loads_supervised_models(self):
        import subprocess
        import sys
        code = ("import os, sys, tempfile; os.environ['LLM_PROVIDER']='none'; "
                "os.environ['DATABASE_URL']='sqlite:///'+tempfile.mktemp(suffix='.db'); "
                "from cockpit.service import CockpitService; from cockpit.agents.copilot import CopilotAgent; "
                "from cockpit.evaluation_suite import run; s=CockpitService(); s.ensure_loaded(); "
                "CopilotAgent(s).ask('Why is this case high priority?', {'case_id': 'C1024'}); run(s); "
                "s.evaluation_summary(); s.label_summary(); "
                "bad=[m for m in sys.modules if m.endswith(('supervised_model','models.synthetic'))]; "
                "print('LOADED', bad); sys.exit(1 if bad else 0)")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=240)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-800:])

    def test_label_summary_counts_real_decisions(self):
        svc = fresh_service()
        svc.set_status("C1024", "Escalated", "Overlapping dates confirmed")
        svc.set_status("C1002", "Likely false positive", "Known billing correction")
        svc.set_status("C1003", "In review")
        self.assertEqual(svc.label_summary(), {"decided": 2, "positive": 1, "negative": 1})


if __name__ == "__main__":
    unittest.main()


class TestRuntimeProviders(unittest.TestCase):
    def tearDown(self):
        providers.reset_runtime()

    def test_configure_switches_provider_and_key_stays_in_memory(self):
        p = providers.configure("anthropic", "claude-test", "sk-secret")
        self.assertEqual((p.name, p.model), ("anthropic", "claude-test"))
        self.assertTrue(providers.llm_available())
        self.assertNotIn("sk-secret", providers.describe())
        providers.reset_runtime()
        self.assertFalse(providers.llm_available())

    def test_default_model_used_when_blank(self):
        self.assertEqual(providers.configure("gemini", "", "k").model, "gemini-2.5-flash")

    def test_url_safety(self):
        for bad in ["http://example.com/v1", "https://user:pw@example.com/v1", "https://example.com/v1?key=x"]:
            with self.assertRaises(providers.LLMError):
                providers.configure("openai_compatible", "m", "", bad)
        p = providers.configure("openai_compatible", "llama3", "", "http://localhost:11434/v1")
        self.assertTrue(p.available)
        self.assertEqual(p.url, "http://localhost:11434/v1/chat/completions")

    def test_openai_compatible_needs_base_url(self):
        with self.assertRaises(providers.LLMError):
            providers.configure("openai_compatible", "m", "", "")

    def test_retry_once_on_transient_error(self):
        err = mock.Mock(status_code=503)
        err.json.return_value = {"error": {"message": "overloaded"}}
        ok = mock.Mock(status_code=200)
        ok.json.return_value = {"content": [{"type": "text", "text": "OK"}], "usage": {}}
        with mock.patch.object(providers.requests, "post", side_effect=[err, ok]) as post, \
             mock.patch.object(providers.time, "sleep"):
            r = providers.AnthropicProvider("m", api_key="k").chat("s", [{"role": "user", "content": "x"}])
        self.assertEqual(r.text, "OK")
        self.assertEqual(post.call_count, 2)
        self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_error_message_does_not_leak_key(self):
        bad = mock.Mock(status_code=401, text="denied")
        bad.json.return_value = {"error": {"message": "invalid x-api-key"}}
        with mock.patch.object(providers.requests, "post", return_value=bad):
            with self.assertRaises(providers.LLMError) as cm:
                providers.AnthropicProvider("m", api_key="sk-very-secret").chat("s", [{"role": "user", "content": "x"}])
        self.assertNotIn("sk-very-secret", str(cm.exception))
        self.assertIn("401", str(cm.exception))

    def test_connections_are_private_per_session(self):
        """Two concurrent visitors (Streamlit runs each session on its own thread) never share a key."""
        import threading
        import time as _t
        seen = {}

        def visitor(name, key):
            providers.configure("anthropic", "m", key)
            _t.sleep(0.05)
            seen[name] = providers.get_provider().api_key

        threads = [threading.Thread(target=visitor, args=(n, k)) for n, k in (("a", "sk-a"), ("b", "sk-b"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(seen, {"a": "sk-a", "b": "sk-b"})
        self.assertEqual(providers.get_provider().api_key, "")  # this thread never connected
        providers.use_runtime({"provider": "openai", "model": "x", "api_key": "sk-c", "base_url": ""})
        self.assertEqual(providers.get_provider().name, "openai")
        providers.use_runtime(None)
        self.assertFalse(providers.llm_available())

    def test_check_connection_without_key(self):
        self.assertFalse(providers.check_connection(providers.NullProvider("x"))["ok"])


class TestInputHardening(unittest.TestCase):
    def test_empty_and_corrupt_files_blocked_not_crashing(self):
        import io
        for content, name in [(b"", "a.csv"), (b"\xff\xfe\x00", "b.csv"), (b"x", "c.pdf")]:
            df, rep = load_file(io.BytesIO(content), name)
            self.assertEqual(rep.status, "blocked", name)

    def test_leading_zero_ids_and_nonfinite(self):
        import io
        with open(settings.sample_csv, encoding="utf-8-sig") as fh:
            rows = [r.split(",") for r in fh.read().splitlines()]
        rows[1][0], rows[1][7] = "00123", "inf"
        df, rep = load_file(io.StringIO("\n".join(",".join(r) for r in rows)), "x.csv")
        self.assertIn("00123", set(df["case_id"]))
        self.assertTrue(any("finite" in v for v in rep.invalid_values))


class TestKnowledgeAndSkeptic(unittest.TestCase):
    def test_retrieval_and_routing(self):
        from cockpit import knowledge_base
        self.assertEqual(knowledge_base.search("repeated claim number")[0]["doc_id"], "PB-DQ-01")
        svc = fresh_service()
        routed = knowledge_base.for_assessment(svc.get_assessment("C1024")["assessment"])
        self.assertIn("PB-DUP-01", [g["doc_id"] for g in routed])

    def test_caveats_and_counter_evidence_answer(self):
        svc = fresh_service()
        a = svc.get_assessment("C1024")["assessment"]
        self.assertTrue(a.caveats)
        res = CopilotAgent(svc).ask("What argues against escalation?", {"case_id": "C1024"})
        self.assertIn("rule out", res.answer)

    def test_abstains_on_missing_information(self):
        svc = fresh_service()
        res = CopilotAgent(svc).ask("Was this claim paid twice?", {"case_id": "C1024"})
        self.assertIn("not in the supplied data", res.answer)
        self.assertEqual(res.tool_calls, [])

    def test_reject_category_recorded(self):
        svc = fresh_service()
        svc.reject_finding("C1024", "member_provider_distance_miles", "Branch office nearby", "Incorrect or outdated source data")
        self.assertEqual(svc.evaluation_summary()["rejection_reasons"][0][0], "Incorrect or outdated source data")
        with self.assertRaises(ValueError):
            svc.reject_finding("C1024", "weekend_billing_ratio", "x", "Made-up category")


class TestDashboardFilters(unittest.TestCase):
    def test_reviewed_and_dq_filters(self):
        svc = fresh_service()
        self.assertEqual(len(svc.queue({"reviewed": False})), 50)
        svc.set_status("C1024", "In review")
        self.assertEqual([r["case_id"] for r in svc.queue({"reviewed": True})], ["C1024"])
        self.assertEqual(svc.queue_summary()["unreviewed"], 49)
        self.assertEqual({r["case_id"] for r in svc.queue({"data_quality_warning": True})}, {"C1001", "C1031"})

    def test_restore_all(self):
        svc = fresh_service()
        base = svc.get_assessment("C1050")["assessment"].score
        svc.reject_finding("C1050", "duplicate_service_billed", "x")
        svc.reject_finding("C1050", "service_overlap_other_provider", "y")
        restored = svc.restore_all_findings("C1050")
        self.assertEqual(restored["assessment"].score, base)
        self.assertFalse([f for f in restored["assessment"].findings if f.status == "rejected"])

    def test_new_chart_data(self):
        svc = fresh_service()
        pie = svc.chart_data("cases_by_care_type")["data"]
        self.assertEqual(sum(x["cases"] for x in pie), 50)
        bars = svc.chart_data("exposure_by_case", {"priority_level": ["red"]})["data"]
        self.assertEqual(len(bars), 10)


class TestPaging(unittest.TestCase):
    def test_pages_cover_queue_in_order(self):
        svc = fresh_service()
        full = [r["case_id"] for r in svc.queue()]
        paged = []
        for p in range(1, 4):
            pg = svc.queue_page({}, p, 20)
            paged += [r["case_id"] for r in pg["rows"]]
            self.assertEqual(pg["total"], 50)
            self.assertEqual(pg["pages"], 3)
        self.assertEqual(paged, full)
        self.assertEqual(svc.queue_page({}, 3, 20)["first"], 41)
        self.assertEqual(svc.queue_page({}, 99, 20)["page"], 3)  # clamped

    def test_sql_filters_match_expectations(self):
        svc = fresh_service()
        self.assertEqual(svc.count({"priority_level": ["red"]}), 10)
        self.assertEqual(svc.count({"search": "c102"}), 10)
        dup = svc.count({"has_signal": ["duplicate_service_billed"]})
        self.assertEqual(dup, sum(1 for c in svc.cases if c["duplicate_service_billed"] == 1))
        red_sorted = svc.queue({"priority_level": ["red"], "sort": "claim_amount_usd"})
        amounts = [r["claim_amount_usd"] for r in red_sorted]
        self.assertEqual(amounts, sorted(amounts, reverse=True))

    def test_index_rebuilds_for_older_databases(self):
        path = tempfile.mktemp(suffix=".db", dir=TMP)
        svc = CockpitService(Repository(path))
        svc.ensure_loaded()
        with svc.repo.conn() as c:
            c.execute("DELETE FROM queue_index")
        svc2 = CockpitService(Repository(path))   # simulates upgrading an existing database
        self.assertEqual(svc2.count(), 50)

    def test_stability_shortcut_is_exact(self):
        import hashlib
        import numpy as np
        df, _ = load_file(str(settings.sample_csv))
        cases = records(df)
        m = TransparentRulesModel().fit(cases)
        cfg = m.ruleset["confidence"]
        for c in cases:
            strong = bool(m._strong(c, set()))
            lvl = m._level(m._score(c, set())[0], strong)
            rng = np.random.default_rng(int(hashlib.md5(c["case_id"].encode()).hexdigest()[:8], 16))
            same = sum(m._level(m._score(c, set(), {s: 1 + rng.uniform(-cfg["perturbation_pct"], cfg["perturbation_pct"])
                                                    for s in m.signals})[0], strong) == lvl
                       for _ in range(cfg["perturbation_runs"]))
            self.assertEqual(m.stability(c, set(), lvl), round(same / cfg["perturbation_runs"], 3), c["case_id"])


class TestFriendReviewChanges(unittest.TestCase):
    """Changes adopted after comparing with an independent Isolation Forest analysis."""

    def test_contact_overlap_combination_raises_c1023(self):
        svc = fresh_service()
        a = svc.get_assessment("C1023")["assessment"]
        self.assertEqual(a.priority_level, "red")
        self.assertIn("combo_contact_overlap", [f.finding_id for f in a.findings if f.points > 0])
        self.assertTrue(a.recommended_next_step.startswith("Check whether the overlapping provider is linked"))

    def test_benign_direction_anomaly_is_not_escalated(self):
        df, _ = load_file(str(settings.sample_csv))
        cases = records(df)
        m = TransparentRulesModel().fit(cases)
        quiet = dict(next(c for c in cases if c["case_id"] == "C1013"))   # $858, 2 visits, nothing elevated
        a = m.assess(quiet, anomaly={"is_anomalous": True, "votes": 3})
        self.assertEqual(a.priority_level, "green")
        self.assertIn("benign direction", " ".join(a.confidence_reasons))
        loud = dict(quiet, weekly_visit_frequency=14)
        self.assertEqual(m.assess(loud, anomaly={"is_anomalous": True, "votes": 3}).priority_level, "yellow")

    def test_seed_averaged_isolation_forest(self):
        from cockpit.models.anomaly_model import AnomalyDetectionModel
        df, _ = load_file(str(settings.sample_csv))
        cases = records(df)
        an = AnomalyDetectionModel().fit(cases).analyze(cases)
        self.assertEqual(an["C1024"]["isolation_forest_seeds"], 10)
        self.assertEqual(an, AnomalyDetectionModel().fit(cases).analyze(cases))   # deterministic
        self.assertTrue(all(v["isolation_forest_seed_range"] >= 0 for v in an.values()))

    def test_existing_database_rescored_after_rules_change(self):
        path = tempfile.mktemp(suffix=".db", dir=TMP)
        svc = CockpitService(Repository(path))
        svc.ensure_loaded()
        with svc.repo.conn() as c:   # pretend these assessments came from the previous ruleset
            c.execute("UPDATE assessments SET ruleset_id = 'rules-v1.0' WHERE case_id = 'C1023'")
        svc2 = CockpitService(Repository(path))
        cur = svc2.get_assessment("C1023")
        self.assertEqual(cur["assessment"].ruleset_id, "rules-v1.1")
        self.assertGreaterEqual(cur["version"], 2)


class TestSecondOpinion(unittest.TestCase):
    def test_disagreements_explained(self):
        svc = fresh_service()
        d = {x["case_id"]: x for x in svc.disagreements()}
        self.assertIn("C1011", d)
        self.assertEqual(d["C1011"]["kind"], "unsupervised_flags")
        self.assertIn("Isolation Forest", d["C1011"]["message"])
        self.assertIsNone(svc.disagreement("C1024"))   # both methods agree it is high priority
        pts = svc.chart_data("rules_vs_anomaly")["data"]
        self.assertEqual(len(pts), 50)
        self.assertTrue(all(0 <= p["anomaly_percentile"] <= 100 for p in pts))


class TestEvaluationSuite(unittest.TestCase):
    def test_suite_passes(self):
        from cockpit.evaluation_suite import run
        res = run(fresh_service())
        self.assertEqual(res["failures"], [], res["failures"][:3])
        self.assertGreater(res["total"], 100)
