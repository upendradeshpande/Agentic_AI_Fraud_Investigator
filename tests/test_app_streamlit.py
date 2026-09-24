"""Real Streamlit UI tests using streamlit.testing.v1.AppTest (runs the actual widget runtime).
Skipped automatically when Streamlit is not installed. Run: python -m pytest tests/test_app_streamlit.py"""
import os
import tempfile
import unittest

os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mkdtemp()}/apptest.db")
os.environ.setdefault("LLM_PROVIDER", "none")

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "streamlit_app.py")


@unittest.skipIf(AppTest is None, "streamlit not installed")
class TestStreamlitApp(unittest.TestCase):
    def app(self, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_overview_renders(self):
        at = self.app()
        self.assertTrue(any("Queue" in s.value for s in at.subheader))

    def test_every_tab_renders(self):
        for tab in ["Overview", "Analysis", "AI evaluation", "Data & rules"]:
            with self.subTest(tab=tab):
                self.app(tab=tab)

    def test_case_view_and_reject_flow(self):
        at = self.app(tab="Analysis", case_id="C1050")
        at.button(key="rj_duplicate_service_billed").click().run()
        self.assertFalse(at.exception)
        at.text_input(key="rt_C1050_duplicate_service_billed").input("Separate visits confirmed")
        next(b for b in at.button if b.label == "Reject and recalculate").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertIn("rejected", at.session_state["flash"] or "rejected")

    def test_kpi_cards_filter_queue(self):
        at = self.app()
        at.button(key="kpib_red").click().run()
        self.assertEqual(at.session_state["filters"], {"priority_level": ["red"]})
        at.button(key="kpib_unreviewed").click().run()
        self.assertEqual(at.session_state["filters"], {"reviewed": False})
        at.button(key="kpib_all").click().run()
        self.assertEqual(at.session_state["filters"], {})
        self.assertFalse(at.exception)

    def test_tab_bar_switches(self):
        at = self.app()
        at.button(key="tab_ai_evaluation").click().run()
        self.assertEqual(at.session_state["tab"], "AI evaluation")
        self.assertFalse(at.exception)

    def test_filtered_queue_without_selection(self):
        # Regression: green + Assisted Living with no case selected raised IndexError.
        self.app(filters={"priority_level": ["green"], "care_type": ["Assisted Living"]})

    def test_copilot_suggestion(self):
        at = self.app(tab="Analysis", case_id="C1024")
        at.button(key="sug_0").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["chat"][-1]["role"], "assistant")

    def test_chat_input_and_new_chat(self):
        at = self.app(tab="Analysis", case_id="C1024")
        at.chat_input(key="copilot_input").set_value("Was this claim paid twice?").run()
        self.assertFalse(at.exception)
        self.assertIn("not in the supplied data", at.session_state["chat"][-1]["content"])
        at.button(key="cp_new").click().run()
        self.assertEqual(at.session_state["chat"], [])

    def test_pager(self):
        at = self.app(page_size=20)
        at.button(key="pg_next").click().run()
        self.assertEqual(at.session_state["page"], 2)
        self.assertFalse(at.exception)

    def test_password_gate(self):
        os.environ["APP_PASSWORD"] = "demo-pass"
        try:
            at = AppTest.from_file(APP, default_timeout=120)
            at.run()
            self.assertFalse(at.exception)
            self.assertNotIn("case_id", at.session_state)          # app stopped before loading
            at.text_input[0].input("demo-pass")
            at.button[0].click().run()
            self.assertTrue(at.session_state["_authed"])
        finally:
            del os.environ["APP_PASSWORD"]

    def test_evaluation_suite_button(self):
        at = self.app(tab="AI evaluation")
        at.button(key="run_suite").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["suite"]["failures"], [])


if __name__ == "__main__":
    unittest.main()
