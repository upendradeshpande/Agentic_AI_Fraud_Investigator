"""Headless smoke test for the Streamlit script, for environments without Streamlit.

Injects stub `streamlit` and `plotly` modules, then executes app/streamlit_app.py across
every tab, with a selected case, clicked buttons and chart selections. Catches NameErrors,
bad attribute access and logic errors in the UI code. It does not test visual layout.

    python tests/smoke_ui.py
"""
import os
import runpy
import sys
import tempfile
import types

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/ui.db"
os.environ.setdefault("LLM_PROVIDER", "none")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class Rerun(Exception):
    pass


class Stop(Exception):
    pass


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class Obj:
    """Generic stand-in: callable, context manager, any attribute."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return Obj()

    def __call__(self, *a, **k):
        return Obj()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_st(clicks: set, selections: dict):
    st = types.ModuleType("streamlit")
    st.session_state = SESSION
    calls = {"buttons": []}

    def ctx(*a, **k):
        return Obj(**{name: getattr(st, name) for name in WIDGETS})

    def columns(spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [ctx() for _ in range(n)]

    def button(label, key=None, **k):
        calls["buttons"].append(key or label)
        return (key or label) in clicks

    def form_submit_button(label, **k):
        return label in clicks

    def dataframe(df, key=None, **k):
        return Obj(selection=Obj(rows=selections.get(key, [])))

    def plotly_chart(fig, key=None, **k):
        return Obj(selection=Obj(points=selections.get(key, [])))

    def altair_chart(chart, key=None, **k):
        return {"selection": selections.get(key, {})}

    def cache_resource(fn=None, **k):
        def deco(f):
            cache = {}

            def wrapper(*a):
                if "v" not in cache:
                    cache["v"] = f(*a)
                return cache["v"]
            wrapper.clear = cache.clear
            return wrapper
        return deco(fn) if fn else deco

    def rerun():
        raise Rerun()

    def stop():
        raise Stop()

    st.columns = columns
    st.button = button
    st.form_submit_button = form_submit_button
    st.dataframe = dataframe
    st.plotly_chart = plotly_chart
    st.altair_chart = altair_chart
    st.cache_resource = cache_resource
    st.rerun = rerun
    st.stop = stop
    def pills(label, options, default=None, key=None, on_change=None, args=(), **k):
        if key and key in PILLS_PICK:  # simulate a click on a pill
            SESSION[key] = PILLS_PICK.pop(key)
            if on_change:
                on_change(*args)
        return SESSION.get(key, default)

    st.pills = pills
    st.chat_input = lambda placeholder="", key=None, **k: INPUTS.get("__chat__")
    st.multiselect = lambda label, options, default=None, **k: list(default or [])
    st.selectbox = lambda label, options, index=0, **k: list(options)[index]
    st.radio = lambda label, options, index=0, **k: list(options)[index]
    st.text_input = lambda label, *a, **k: INPUTS.get(label, "")
    st.text_area = lambda label, *a, **k: INPUTS.get(label, "")
    st.number_input = lambda label, value=0, **k: value
    st.file_uploader = lambda *a, **k: None
    st.progress = lambda *a, **k: Obj()
    for name in ["set_page_config", "markdown", "subheader", "caption", "info", "warning", "error", "success",
                 "metric", "code", "write"]:
        setattr(st, name, lambda *a, **k: None)
    for name in ["expander", "container", "form", "chat_message", "spinner"]:
        setattr(st, name, ctx)
    st.column_config = Obj()
    WIDGETS[:] = [n for n in dir(st) if not n.startswith("_")]
    return st, calls


WIDGETS: list = []
PILLS_PICK: dict = {}
SESSION = SessionState()
INPUTS: dict = {}


def install_altair():
    alt = types.ModuleType("altair")
    for name in ["Chart", "layer", "selection_point", "Theta", "Color", "Scale", "Legend", "Opacity", "Tooltip", "X", "Y",
                 "Axis", "value", "condition"]:
        setattr(alt, name, lambda *a, **k: Obj())
    sys.modules["altair"] = alt


def install_plotly():
    plotly = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")

    class Figure(Obj):
        def __init__(self, *a, **k):
            pass
    go.Figure = Figure
    go.Bar = go.Scatter = go.Pie = lambda *a, **k: Obj()
    plotly.graph_objects = go
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = go


def run(clicks=(), selections=None, inputs=None, max_reruns=5):
    INPUTS.clear()
    INPUTS.update(inputs or {})
    st, calls = make_st(set(clicks), selections or {})
    sys.modules["streamlit"] = st
    for _ in range(max_reruns):
        try:
            runpy.run_path(os.path.join(ROOT, "app", "streamlit_app.py"), run_name="__main__")
            return calls
        except Stop:
            return calls
        except Rerun:
            st, calls = make_st(set(), {})  # after a rerun, stop clicking
            sys.modules["streamlit"] = st
    return calls


def qkey():
    """Same key the app uses for the queue table (it changes with filters and page)."""
    sig = repr(sorted(SESSION.get("filters", {}).items()))
    return "queue_table_" + str(abs(hash(sig))) + f"_p{SESSION.get('page', 1)}"


def main():
    install_plotly()
    try:
        import altair  # noqa: F401  (real Altair when available)
    except ImportError:
        install_altair()
    steps = []

    run(); steps.append("initial load (Overview)")
    SESSION["tab"] = "Analysis"; run(); steps.append("Analysis, no case")
    run(selections={qkey(): [0]}); steps.append("select first queue row")
    assert SESSION["case_id"], "case not selected"
    cid = SESSION["case_id"]
    run(); steps.append(f"case view {cid}")
    run(clicks={"rj_duplicate_service_billed"}); steps.append("click Reject")
    assert SESSION["reject_target"] == "duplicate_service_billed"
    run(clicks={"Reject and recalculate"}, inputs={"Why is this finding wrong?": "Separate visits confirmed"})
    steps.append("submit rejection")
    assert "rejected" in (SESSION.get("flash") or "") or SESSION["reject_target"] is None
    run(clicks={"Save"}, inputs={"Add a note": "Requested records", "Reason (required for final decisions)": ""})
    steps.append("save note")
    run(clicks={"sug_0"}); steps.append("copilot suggestion on case")
    assert SESSION["chat"] and SESSION["chat"][-1]["role"] == "assistant"
    run(inputs={"__chat__": "Was this claim paid twice?"}); steps.append("typed question in chat input")
    assert "not in the supplied data" in SESSION["chat"][-1]["content"], SESSION["chat"][-1]["content"][:80]
    run(clicks={"cp_new"}); steps.append("New chat clears the conversation")
    assert SESSION["chat"] == []
    SESSION["case_id"] = None; SESSION["tab"] = "Overview"
    run(clicks={"sug_1"}); steps.append("copilot filter suggestion")
    assert SESSION["filters"], "copilot filter not applied"
    run(clicks={"Clear"}); steps.append("clear filters")
    run(clicks={"kpib_red"}); steps.append("click High priority card")
    assert SESSION["filters"] == {"priority_level": ["red"]}, SESSION["filters"]
    run(clicks={"kpib_unreviewed"}); steps.append("click Unreviewed card")
    assert SESSION["filters"] == {"reviewed": False}
    run(clicks={"kpib_reviewed"}); steps.append("click Reviewed card")
    assert SESSION["filters"] == {"reviewed": True}
    run(clicks={"kpib_all"}); steps.append("click All cases card")
    SESSION["page_size"] = 20
    run(clicks={"pg_next"}); steps.append("pager: next page")
    assert SESSION["page"] == 2, SESSION["page"]
    run(clicks={"pg_prev"}); steps.append("pager: previous page")
    assert SESSION["page"] == 1
    SESSION["page"] = 3; SESSION["filters"] = {"priority_level": ["red"]}
    run(); steps.append("filter change resets to page 1")
    assert SESSION["page"] == 1
    SESSION["filters"] = {}; SESSION["page_size"] = 50
    assert SESSION["filters"] == {}
    run(selections={"chart_care_pie": {"care_pick": [{"care_type": "Assisted Living"}]}}); steps.append("click pie slice")
    assert SESSION["filters"].get("care_type") == ["Assisted Living"], SESSION["filters"]
    run(selections={"chart_level_bar": {"level_pick": [{"priority_level": "green"}]}}); steps.append("click priority bar")
    assert SESSION["filters"] == {"care_type": ["Assisted Living"], "priority_level": ["green"]}, SESSION["filters"]
    run(selections={"chart_care_pie": {"care_pick": [{"care_type": "Assisted Living"}]}})
    steps.append("remembered selection does not re-apply")
    run(clicks={"clear_lvl"}); steps.append("clear priority from chart title")
    assert SESSION["filters"] == {"care_type": ["Assisted Living"]}, SESSION["filters"]
    run(selections={"chart_care_pie": {"care_pick": {"care_type": ["Home Health Aide"]}}})
    steps.append("pie selection, alternate event shape")
    assert SESSION["filters"]["care_type"] == ["Home Health Aide"], SESSION["filters"]
    run(clicks={"clear_care"}); steps.append("clear care type from chart title")
    assert SESSION["filters"] == {}, SESSION["filters"]
    run(selections={"chart_rules_vs_anomaly": {"case_pick": [{"case_id": "C1011"}]}})
    steps.append("click a dot in rules vs Isolation Forest chart opens the case")
    assert SESSION["case_id"] == "C1011" and SESSION["tab"] == "Analysis", (SESSION["case_id"], SESSION["tab"])
    run(); steps.append("case view shows the second-opinion note (C1011)")
    SESSION["case_id"] = None; SESSION["tab"] = "Overview"
    SESSION["filters"] = {"priority_level": ["green"], "care_type": ["Assisted Living"]}
    # Reproduce the reported bug: a stale row index larger than the filtered table.
    run(selections={qkey(): [40]}); steps.append("stale row selection past end of filtered table (reported IndexError)")
    SESSION["filters"] = {}
    SESSION["tab"] = "Overview"
    run(clicks={"tab_analysis"}); steps.append("click Analysis tab")
    assert SESSION["tab"] == "Analysis"
    SESSION["tab"] = "AI evaluation"; run(); steps.append("AI evaluation tab")
    run(clicks={"run_suite"}); steps.append("run evaluation suite from UI")
    assert SESSION["suite"]["failures"] == [], SESSION["suite"]["failures"][:2]
    run(clicks={"mc_connect"}); steps.append("model connection: connect (provider none)")
    run(clicks={"mc_test"}); steps.append("model connection: test")
    run(clicks={"mc_reset"}); steps.append("model connection: back to .env")
    SESSION["tab"] = "Data & rules"; run(); steps.append("Data & rules tab")
    SESSION["tab"] = "Analysis"; SESSION["case_id"] = "C1001"; run(); steps.append("data-review case C1001")
    SESSION["case_id"] = cid
    run(clicks={f"rerun_{cid}"}); steps.append("re-run assessment")
    run(clicks={f"restore_all_{cid}"}); steps.append("restore all rejected findings")
    # ---- hosting: per-session model connection, password gate, question limit
    from cockpit.llm import providers
    SESSION["llm_runtime"] = {"provider": "anthropic", "model": "claude-test", "api_key": "sk-visitor-a", "base_url": ""}
    run(); steps.append("visitor A's own connection is active in A's session")
    assert providers.describe() == "anthropic:claude-test", providers.describe()
    saved_a = dict(SESSION)
    SESSION.clear()
    run(); steps.append("visitor B does not inherit A's connection")
    assert providers.describe().startswith("deterministic"), providers.describe()
    SESSION.clear(); SESSION.update(saved_a); SESSION["llm_runtime"] = None
    os.environ["APP_PASSWORD"] = "demo-pass"
    SESSION.pop("_authed", None)
    run(); steps.append("password gate stops the app")
    assert not SESSION.get("_authed")
    run(clicks={"Enter"}, inputs={"Access password": "wrong"}); steps.append("wrong password rejected")
    assert not SESSION.get("_authed")
    run(clicks={"Enter"}, inputs={"Access password": "demo-pass"}); steps.append("correct password admits")
    assert SESSION.get("_authed")
    del os.environ["APP_PASSWORD"]
    os.environ["SESSION_QUESTION_LIMIT"] = "1"
    SESSION["questions_asked"] = 0; SESSION["chat"] = []; SESSION["case_id"] = "C1024"; SESSION["tab"] = "Analysis"
    run(clicks={"sug_0"}); run(clicks={"sug_1"}); steps.append("question limit enforced")
    assert SESSION["questions_asked"] == 1 and len(SESSION["chat"]) == 2, (SESSION["questions_asked"], len(SESSION["chat"]))
    del os.environ["SESSION_QUESTION_LIMIT"]

    from cockpit.db import Repository
    repo = Repository(os.environ["DATABASE_URL"].replace("sqlite:///", ""))
    acts = [d["action"] for d in repo.decisions(cid)]
    assert "reject_finding" in acts, acts
    assert repo.notes(cid), "note not saved"
    assert len(repo.assessment_history(cid)) >= 2, "no recalculated assessment version"
    steps.append(f"verified in DB: {acts}, {len(repo.assessment_history(cid))} assessment versions")
    for s in steps:
        print("ok  ", s)
    print(f"UI smoke test passed ({len(steps)} steps).")


if __name__ == "__main__":
    main()
