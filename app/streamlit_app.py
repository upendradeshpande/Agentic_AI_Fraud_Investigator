"""AI Investigation Cockpit - Streamlit UI.

Layout (spec section 4):  queue | Overview / Analysis / AI evaluation / Data & rules | copilot
Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hmac
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import altair as alt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from cockpit.agents.assessment_agent import batch_generate  # noqa: E402
from cockpit.agents.adk_runtime import make_copilot  # noqa: E402
from cockpit.agents.copilot import explain_validation  # noqa: E402
from cockpit import knowledge_base  # noqa: E402
from cockpit.config import DEFAULT_MODELS, settings  # noqa: E402
from cockpit.evaluation_suite import run as run_evaluation_suite  # noqa: E402
from cockpit.llm import providers  # noqa: E402
from cockpit.schemas import CASE_STATUSES, FINAL_STATUSES, REJECT_REASONS  # noqa: E402
from cockpit.service import SIGNAL_LABELS, CockpitService  # noqa: E402

st.set_page_config(page_title="Fraud Investigator Tool", page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")

LEVEL_COLOR = {"red": "#A8322D", "yellow": "#B7791F", "green": "#3F7D4E", "data_review": "#5B5FA8"}
LEVEL_NAME = {"red": "High priority", "yellow": "Needs attention", "green": "Lower priority", "data_review": "Data review"}
LEVEL_DOT = {"red": "🔴", "yellow": "🟡", "green": "🟢", "data_review": "🟣"}
TABS = ["Overview", "Analysis", "AI evaluation", "Data & rules"]

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText { font-family: 'IBM Plex Sans', sans-serif; }
  .block-container { padding-top: 3.4rem; padding-bottom: 2rem; max-width: 100%; }
  .stApp { background:
      radial-gradient(1100px 520px at 0% 0%, #DDF0EC 0%, rgba(221,240,236,0) 60%),
      radial-gradient(900px 480px at 100% 0%, #E9E3FB 0%, rgba(233,227,251,0) 55%),
      radial-gradient(900px 600px at 50% 100%, #FCEBDD 0%, rgba(252,235,221,0) 60%),
      #F4F6F8; }
  [data-testid="stHeader"] { background: transparent; }
  /* Header banner */
  .st-key-hero { background: linear-gradient(100deg, #173F4A 0%, #2F5D62 42%, #4E5AA8 100%); border-radius: 18px;
      padding: 1.1rem 1.4rem 0.9rem; margin-bottom: 0.9rem; box-shadow: 0 6px 20px rgba(23,63,74,0.18); }
  .st-key-hero .ck-title { color: #FFFFFF; font-size: 2.05rem; line-height: 1.3; padding-top: 0.1rem; }
  .st-key-hero .ck-sub { color: rgba(255,255,255,0.86); font-size: 0.95rem; margin-top: 0.2rem; }
  .st-key-hero .ck-chip { background: rgba(255,255,255,0.14); border-color: rgba(255,255,255,0.35); color: #FFFFFF; }
  .st-key-hero [data-testid="stExpander"] { background: #FFFFFF; border-radius: 10px; border: none; }
  .st-key-hero [data-testid="stExpander"] summary, .st-key-hero [data-testid="stExpander"] summary p { color: #17262B; }
  /* White card panels */
  .st-key-queuepanel, .st-key-dash { background: #FFFFFF; border: 1px solid #E2E8EC; border-radius: 16px;
      padding: 0.9rem 1rem 0.8rem; box-shadow: 0 2px 12px rgba(31,78,90,0.06); }
  .st-key-queuepanel h3 { font-size: 1.25rem; line-height: 1.3; padding-top: 0.1rem; }
  .ck-legend { display: flex; flex-wrap: wrap; gap: 0.25rem 0.9rem; justify-content: center; font-size: 0.8rem;
      color: #34474C; margin-top: -0.4rem; }
  .ck-legend i { display: inline-block; width: 0.62rem; height: 0.62rem; border-radius: 50%; margin-right: 0.35rem;
      vertical-align: -0.02rem; }
  .st-key-clear_care button, .st-key-clear_lvl button { min-height: 1.8rem; border-radius: 999px !important;
      border: 1px solid #D5DEDB !important; background: #FFFFFF !important; }
  .st-key-clear_care button p, .st-key-clear_lvl button p { font-size: 0.76rem; }
  h1, h2, h3 { font-family: 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif; letter-spacing: -0.01em; }
  .ck-title { font-family: 'IBM Plex Sans Condensed', sans-serif; font-size: 2rem; line-height: 1.3; font-weight: 600; margin: 0; }
  .ck-sub { color: #52656B; font-size: 0.9rem; margin-top: 0.1rem; }
  .ck-badge { display: inline-block; padding: 0.12rem 0.55rem; border-radius: 3px; color: white;
              font-size: 0.8rem; font-weight: 600; }
  .ck-chip { display: inline-block; padding: 0.1rem 0.5rem; border: 1px solid #C7D3D0; border-radius: 3px;
             font-size: 0.78rem; color: #34474C; margin: 0 0.25rem 0.25rem 0; background: #FFFFFF; }
  .ck-evidence { border-left: 4px solid var(--c); padding: 0.1rem 0 0.1rem 0.75rem; margin: 0.2rem 0; }
  .ck-score { font-family: 'IBM Plex Sans Condensed', sans-serif; font-size: 2.6rem; font-weight: 600; line-height: 1; }
  .ck-muted { color: #5E7075; font-size: 0.82rem; }
  .ck-summary { background: #FFFFFF; border: 1px solid #D5DEDB; border-radius: 4px; padding: 0.9rem 1rem; }
  .ck-future { border: 1px dashed #AFC0BC; border-radius: 4px; padding: 0.7rem 0.9rem; color: #52656B; font-size: 0.85rem; }

  /* ---- Copilot, ChatGPT-style ---- */
  .st-key-copilot { background: linear-gradient(180deg, #FCFBFF 0%, #F2EFFD 100%); border: 1px solid #E1DAF7;
      border-radius: 16px; padding: 0.85rem 0.9rem 0.7rem; box-shadow: 0 2px 12px rgba(78,90,168,0.08); }
  .st-key-copilot, .st-key-copilot p, .st-key-copilot li { color: #0D0D0D; font-family: 'IBM Plex Sans', -apple-system,
      BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  .cp-title { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em; }
  .cp-sub { font-size: 0.78rem; color: #8E8E8E; margin-top: 0.05rem; }
  .cp-ctx { display: inline-block; font-size: 0.72rem; color: #4A4870; background: #ECE8FB; border-radius: 999px;
      padding: 0.12rem 0.55rem; margin: 0.35rem 0.3rem 0 0; }
  .cp-empty { text-align: center; color: #5D5D5D; font-size: 1.1rem; font-weight: 500; padding-top: 38%; }
  .cp-meta { font-size: 0.72rem; color: #8E8E8E; margin-top: -0.2rem; }
  .cp-case { font-size: 0.72rem; color: #8E8E8E; }
  .st-key-chatlog { padding-right: 0.2rem; }
  div[class*="st-key-umsg_"] { background: #E6E1FA; border-radius: 18px; padding: 0.55rem 0.9rem 0.45rem;
      margin: 0.5rem 0 0.5rem auto; width: fit-content !important; max-width: 86%; }
  div[class*="st-key-umsg_"] p { font-size: 0.92rem; line-height: 1.5; margin-bottom: 0.1rem; }
  div[class*="st-key-amsg_"] { padding: 0.2rem 0.1rem 0.4rem; margin: 0.3rem 0 0.6rem 0; }
  div[class*="st-key-amsg_"] p, div[class*="st-key-amsg_"] li { font-size: 0.92rem; line-height: 1.6; }
  div[class*="st-key-amsg_"] [data-testid="stExpander"] details { border: none; background: transparent; }
  div[class*="st-key-amsg_"] [data-testid="stExpander"] summary { font-size: 0.75rem; color: #8E8E8E; padding: 0.1rem 0; }
  .st-key-suggest button { border-radius: 999px !important; border: 1px solid #E5E5E5 !important; background: #FFFFFF !important;
      min-height: 2rem; padding: 0.2rem 0.75rem; box-shadow: none !important; }
  .st-key-suggest button p { font-size: 0.76rem; color: #0D0D0D; text-align: left; line-height: 1.25; }
  .st-key-suggest button:hover { background: #F1EDFD !important; border-color: #CFC5F2 !important; }
  .st-key-cp_new button { border-radius: 999px !important; border: 1px solid #E5E5E5 !important; background: #FFFFFF !important;
      min-height: 1.9rem; }
  .st-key-cp_new button p { font-size: 0.76rem; }
  .st-key-copilot [data-testid="stChatInput"] { border-radius: 26px !important; border: 1px solid #E0E0E0 !important;
      background: #FFFFFF !important; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .st-key-copilot [data-testid="stChatInput"] textarea { font-size: 0.92rem; background: transparent !important; }
  .st-key-copilot [data-testid="stChatInputSubmitButton"] { background: #0D0D0D !important; color: #FFFFFF !important;
      border-radius: 50% !important; }
  .st-key-copilot [data-testid="stChatInputSubmitButton"] svg { fill: #FFFFFF !important; color: #FFFFFF !important; }
  .st-key-copilot [data-testid="stChatInputSubmitButton"]:disabled { background: #D7D7D7 !important; }
  .ck-pageinfo { text-align: center; padding-top: 0.45rem; font-size: 0.85rem; color: #52656B; }
  /* Tab bar: buttons styled as tabs (keeps programmatic switching, unlike st.tabs) */
  .st-key-tabbar { border-bottom: 1px solid #D5DEDB; margin-bottom: 0.8rem; }
  .st-key-tabbar button { background: transparent !important; border: none !important; box-shadow: none !important;
      border-bottom: 3px solid transparent !important; border-radius: 0 !important; color: #52656B !important;
      min-height: 2.5rem; padding: 0 0.4rem; }
  .st-key-tabbar button p { font-size: 0.95rem; font-weight: 500; }
  .st-key-tabbar button:hover { color: #17262B !important; border-bottom-color: #AFC0BC !important; }
  .st-key-tabbar button[kind="primary"], .st-key-tabbar button[data-testid="stBaseButton-primary"] {
      color: #17262B !important; border-bottom-color: #2F5D62 !important; }
  .st-key-tabbar button[kind="primary"] p, .st-key-tabbar button[data-testid="stBaseButton-primary"] p { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

KPI_STYLE = {  # card id: (background, accent, text)
    "all": ("#E6EDEE", "#2F5D62", "#17262B"),
    "red": ("#F8E3E1", "#A8322D", "#5C1814"),
    "yellow": ("#FBF0DA", "#B7791F", "#5E3D08"),
    "green": ("#E2F0E5", "#3F7D4E", "#1D4027"),
    "data_review": ("#E8E8F6", "#5B5FA8", "#2A2D63"),
    "unreviewed": ("#ECEFF1", "#6B7C82", "#26363B"),
    "reviewed": ("#DDEFEE", "#2A7F7A", "#123D3B"),
}
st.markdown("<style>" + "".join(f"""
  .st-key-kpi_{k} button {{ background: {bg} !important; color: {fg} !important; border: 1px solid {ac}40 !important;
      border-left: 5px solid {ac} !important; border-radius: 6px !important; min-height: 84px; width: 100%;
      justify-content: flex-start; padding: 0.55rem 0.8rem; box-shadow: none !important; }}
  .st-key-kpi_{k} button p {{ text-align: left; font-size: 0.8rem; line-height: 1.25; margin: 0; color: {fg}; }}
  .st-key-kpi_{k} button strong {{ display: block; font-size: 1.8rem; font-weight: 600; line-height: 1.1;
      font-family: 'IBM Plex Sans Condensed', sans-serif; }}
  .st-key-kpi_{k} button:hover {{ filter: brightness(0.96); }}
  .st-key-kpi_{k} button[kind="primary"], .st-key-kpi_{k} button[data-testid="stBaseButton-primary"] {{
      box-shadow: inset 0 0 0 2px {ac} !important; }}""" for k, (bg, ac, fg) in KPI_STYLE.items()) + "</style>",
            unsafe_allow_html=True)
CARE_COLORS = ["#2F7F86", "#E07A5F", "#3D6FB6", "#F2B84B", "#7FB069", "#9C7BC4", "#F28FAD"]


# ---------------------------------------------------------------- access gate (optional, for hosted demos)
def _password_gate() -> None:
    """If APP_PASSWORD is set (env var or host secret), ask for it once per browser session."""
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("_authed"):
        return
    st.markdown('<div class="ck-title" style="margin-top:2rem">Fraud Investigator Tool</div>', unsafe_allow_html=True)
    with st.form("gate"):
        pw = st.text_input("Access password", type="password")
        ok = st.form_submit_button("Enter", type="primary")
    if ok and hmac.compare_digest(pw.encode(), expected.encode()):
        st.session_state["_authed"] = True
        st.rerun()
    if ok:
        st.error("Incorrect password.")
    st.stop()


_password_gate()


# ---------------------------------------------------------------- state
@st.cache_resource(show_spinner="Loading queue and running assessments...")
def get_service() -> CockpitService:
    svc = CockpitService()
    svc.ensure_loaded()
    return svc


svc = get_service()
ss = st.session_state
ss.setdefault("case_id", None)
ss.setdefault("tab", "Overview")
ss.setdefault("filters", {})
ss.setdefault("chart", None)
ss.setdefault("chart_selection", None)
ss.setdefault("chat", [])
ss.setdefault("proposals", [])
ss.setdefault("reject_target", None)
ss.setdefault("flash", None)
ss.setdefault("llm_runtime", None)
ss.setdefault("questions_asked", 0)
# Activate THIS visitor's model connection for this page run (never shared between sessions).
providers.use_runtime(ss.llm_runtime)
QUESTION_LIMIT = int(os.environ.get("SESSION_QUESTION_LIMIT", "0") or 0)  # 0 = unlimited


def select_case(case_id: str | None, tab: str = "Analysis"):
    ss.case_id = case_id
    ss.tab = tab
    ss.reject_target = None


def badge(level: str) -> str:
    return f'<span class="ck-badge" style="background:{LEVEL_COLOR[level]}">{LEVEL_NAME[level]}</span>'


def ts(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def money(v) -> str:
    return f"${v:,.0f}"


# ---------------------------------------------------------------- header
with st.container(key="hero"):
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown('<div class="ck-title">Fraud Investigator Tool</div>', unsafe_allow_html=True)
        s = svc.queue_summary()
        st.markdown(f'<div class="ck-sub">{s["total_cases"]:,} referrals assessed before you opened the queue. '
                    f'{s["high_priority"]:,} need review first. The AI prepares; you decide.</div>', unsafe_allow_html=True)
    with h2:
        chips = [f"LLM {providers.describe()}", f"Rules {svc.rules.ruleset_id}", f"Data {svc.report.status}",
                 f"Batch {svc.batch['batch_id']}"]
        st.markdown("".join(f'<span class="ck-chip">{c}</span>' for c in chips), unsafe_allow_html=True)
        with st.expander("Model connection"):
            st.caption("Plug in any supported model. The key stays in your browser session on the server only: it is never "
                       "saved, logged, sent back to the browser, or shared with other visitors. For a permanent setup "
                       "use the .env file or the host\'s secrets.")
            names = ["gemini", "anthropic", "openai", "openai_compatible", "none"]
            cur = providers.get_provider()
            pv = st.selectbox("Provider", names, index=names.index(cur.name) if cur.name in names else 0, key="mc_provider")
            model = st.text_input("Model ID", value=cur.model if pv == cur.name else "",
                                  placeholder=DEFAULT_MODELS.get(pv, ""), key=f"mc_model_{pv}")
            key = st.text_input("API key", type="password", placeholder="Leave empty to use the key from .env",
                                key=f"mc_key_{pv}")
            base = st.text_input("Base URL (openai_compatible only, e.g. http://localhost:11434/v1)",
                                 key=f"mc_base_{pv}") if pv == "openai_compatible" else ""
            b1, b2, b3 = st.columns(3)
            if b1.button("Connect", key="mc_connect", use_container_width=True):
                try:
                    providers.configure(pv, model, key, base)
                    ss.llm_runtime = providers.runtime_config()  # this session only
                    ss.flash = f"Now using {providers.describe()} for your session."
                    st.rerun()
                except providers.LLMError as e:
                    st.error(str(e))
            if b2.button("Test", key="mc_test", use_container_width=True):
                try:
                    candidate = providers.build(pv, model, key, base)  # tests what is typed, before Connect
                    with st.spinner("Sending a tiny test request..."):
                        res = providers.check_connection(candidate)
                    if res["ok"] and providers.describe() != f"{candidate.name}:{candidate.model}":
                        res["message"] += " Click Connect to start using it."
                    (st.success if res["ok"] else st.error)(res["message"])
                except providers.LLMError as e:
                    st.error(str(e))
            if b3.button("Use .env", key="mc_reset", use_container_width=True):
                providers.reset_runtime()
                ss.llm_runtime = None
                ss.flash = f"Back to the server settings: {providers.describe()}."
                st.rerun()

if os.environ.get("DEMO_NOTICE"):
    st.info(os.environ["DEMO_NOTICE"])
if ss.flash:
    st.success(ss.flash)
    ss.flash = None

left, mid, right = st.columns([1.15, 2.6, 1.3], gap="medium")

# ---------------------------------------------------------------- queue
with left, st.container(key="queuepanel"):
    st.subheader("Queue")
    total_all = svc.count()
    care_types = [x["care_type"] for x in svc.chart_data("cases_by_care_type")["data"]]
    with st.expander("Filters", expanded=bool(ss.filters)):
        f_level = st.multiselect("Priority", list(LEVEL_NAME), default=ss.filters.get("priority_level", []),
                                 format_func=lambda k: LEVEL_NAME[k])
        f_care = st.multiselect("Care type", care_types, default=[c for c in ss.filters.get("care_type", []) if c in care_types])
        f_status = st.multiselect("Status", CASE_STATUSES, default=ss.filters.get("status", []))
        c1, c2 = st.columns(2)
        if c1.button("Apply", use_container_width=True):
            new = {k: v for k, v in {"priority_level": f_level, "care_type": f_care, "status": f_status}.items() if v}
            for k in ("min_amount", "max_amount", "has_signal", "min_score", "max_score", "reviewed", "data_quality_warning"):
                if k in ss.filters:
                    new[k] = ss.filters[k]
            ss.filters = new
            st.rerun()
        if c2.button("Clear", use_container_width=True):
            ss.filters, ss.chart_selection = {}, None
            st.rerun()
    if ss.filters:
        st.markdown("".join(f'<span class="ck-chip">{k}: {v}</span>' for k, v in ss.filters.items()), unsafe_allow_html=True)

    q = st.text_input("Find case", placeholder="Case ID or claim number", label_visibility="collapsed").strip()
    qf = {**ss.filters, **({"search": q} if q else {})}
    # Paging: only one page of rows is read from the database. Any filter or search change returns to page 1.
    sig = repr(sorted(qf.items()))
    if ss.get("_page_sig") != sig:
        ss._page_sig, ss.page = sig, 1
    page_size = ss.get("page_size", 50)
    pg = svc.queue_page(qf, ss.get("page", 1), page_size)
    ss.page = pg["page"]
    rows = pg["rows"]
    df = pd.DataFrame([{
        "#": r["rank"], "Case": r["case_id"], "Priority": f'{LEVEL_DOT[r["priority_level"]]} {LEVEL_NAME[r["priority_level"]]}',
        "Score": r["score"], "Conf.": r["confidence"], "Amount": r["claim_amount_usd"], "Status": r["status"],
    } for r in rows])
    if df.empty:
        st.info("No cases match these filters. Clear the filters to see the full queue.")
    else:
        # Key depends on filters and page so a stale row selection never points past the end of a shorter table.
        table_key = "queue_table_" + str(abs(hash(sig))) + f"_p{pg['page']}"
        event = st.dataframe(
            df, hide_index=True, use_container_width=True, height=min(560, 38 + 35 * len(df)), on_select="rerun",
            selection_mode="single-row", key=table_key,
            column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                           "Amount": st.column_config.NumberColumn("Amount", format="$%d")},
        )
        sel = event.selection.rows if event and hasattr(event, "selection") else []
        sel = [i for i in sel if 0 <= i < len(df)]
        if sel:
            picked = df.iloc[sel[0]]["Case"]
            if picked != ss.get("_last_table_pick"):
                ss._last_table_pick = picked
                select_case(picked)
                st.rerun()
        with st.container(key="pager"):
            p1, p2, p3 = st.columns([1, 2.2, 1])
            if p1.button("‹ Prev", key="pg_prev", disabled=pg["page"] <= 1, use_container_width=True):
                ss.page = pg["page"] - 1
                st.rerun()
            p2.markdown(f"<div class='ck-pageinfo'>Page <b>{pg['page']:,}</b> of {pg['pages']:,}</div>", unsafe_allow_html=True)
            if p3.button("Next ›", key="pg_next", disabled=pg["page"] >= pg["pages"], use_container_width=True):
                ss.page = pg["page"] + 1
                st.rerun()
        if pg["pages"] > 2:
            jump = st.number_input("Go to page", min_value=1, max_value=pg["pages"], value=pg["page"], step=1,
                                   key=f"pg_jump_{sig}_{pg['page']}")
            if int(jump) != pg["page"]:
                ss.page = int(jump)
                st.rerun()
        st.caption(f"Showing {pg['first']:,}–{pg['last']:,} of {pg['total']:,} matching"
                   + (f" ({total_all:,} in queue)" if pg["total"] != total_all else "")
                   + ". Ordered by strong indicators, priority, uncertainty, score, then amount.")


# ---------------------------------------------------------------- tabs
def overview_tab():
    s = svc.queue_summary()
    cards = [
        ("all", s["total_cases"], "All cases", {}),
        ("red", s["high_priority"], "High priority", {"priority_level": ["red"]}),
        ("yellow", s["needs_attention"], "Needs attention", {"priority_level": ["yellow"]}),
        ("green", s["lower_priority"], "Lower priority", {"priority_level": ["green"]}),
        ("data_review", s["data_review"], "Data review", {"priority_level": ["data_review"]}),
        ("unreviewed", s["unreviewed"], "Unreviewed", {"reviewed": False}),
        ("reviewed", s["reviewed"], "Reviewed", {"reviewed": True}),
    ]
    cols = st.columns(len(cards), gap="small")
    for col, (cid, n, label, flt) in zip(cols, cards):
        active = (not ss.filters) if cid == "all" else ss.filters == flt
        with col.container(key=f"kpi_{cid}"):
            if st.button(f"**{n}** {label}", key=f"kpib_{cid}", type="primary" if active else "secondary",
                         use_container_width=True):
                ss.filters, ss.chart, ss.chart_selection = dict(flt), None, None
                st.rerun()
    st.caption(f"Click a card to filter the queue; All cases resets. Queue exposure {money(s['total_claim_amount_usd'])}, "
               f"of which {money(s['amount_in_high_priority_usd'])} is in high-priority cases. "
               "Claim amount is exposure, not evidence.")

    c1, c2 = st.columns(2, gap="large")
    picked_care = (ss.filters.get("care_type") or [None])[0]
    picked_lvl = (ss.filters.get("priority_level") or [None])[0] if len(ss.filters.get("priority_level") or []) == 1 else None
    with c1:
        st.markdown("**Cases by care type**  \n<span class='ck-muted'>Click a slice to filter the queue.</span>",
                    unsafe_allow_html=True)
        base_f = {k: v for k, v in ss.filters.items() if k != "care_type"}
        d = svc.chart_data("cases_by_care_type", base_f)["data"]
        if d:
            total = sum(x["cases"] for x in d)
            df_pie = pd.DataFrame([{
                "care_type": x["care_type"], "cases": x["cases"], "claimed": money(x["claim_amount_usd"]),
                "label": f"{x['cases']:,} · {100 * x['cases'] / total:.0f}%",
                "op": 1.0 if (not picked_care or x["care_type"] == picked_care) else 0.3} for x in d])
            pick = alt.selection_point(name="care_pick", fields=["care_type"], on="click")
            base = alt.Chart(df_pie).encode(
                theta=alt.Theta("cases:Q", stack=True),
                # No built-in legend: Streamlit fits legend + chart into one box, which shrank (and clipped) the donut.
                color=alt.Color("care_type:N", scale=alt.Scale(domain=list(df_pie["care_type"]), range=CARE_COLORS[:len(df_pie)]),
                                legend=None),
                opacity=alt.Opacity("op:Q", scale=None, legend=None),
                tooltip=[alt.Tooltip("care_type:N", title="Care type"), alt.Tooltip("cases:Q", title="Cases"),
                         alt.Tooltip("claimed:N", title="Claimed")],
            )
            # Radii sized so donut + outside labels (about 250 px) fit the 300 px chart and a narrow column.
            arcs = base.mark_arc(innerRadius=54, outerRadius=94, stroke="white", strokeWidth=2, cursor="pointer").add_params(pick)
            labels = base.mark_text(radius=116, fontSize=12, fontWeight="bold").encode(text="label:N")
            ev = _clickable_chart(alt.layer(arcs, labels).properties(height=CHART_H),
                                  arcs.properties(height=CHART_H), "chart_care_pie", "care_pick")
            _apply_chart_pick(ev, "care_pick", "care_type", "care_type", "cases_by_care_type")
            st.markdown("<div class='ck-legend'>" + "".join(
                f"<span style='opacity:{r.op}'><i style='background:{CARE_COLORS[i]}'></i>{r.care_type}</span>"
                for i, r in enumerate(df_pie.itertuples())) + "</div>", unsafe_allow_html=True)
            _chart_footer(f"{total:,} cases" + (f" · showing {picked_care}" if picked_care else ""),
                          "clear_care", "care_type" if picked_care else None)
        else:
            st.info("No cases for this selection.")
    with c2:
        st.markdown("**Claim exposure by priority**  \n<span class='ck-muted'>Click a bar to filter the queue.</span>",
                    unsafe_allow_html=True)
        base_f = {k: v for k, v in ss.filters.items() if k != "priority_level"}
        d = svc.chart_data("exposure_by_priority", base_f)["data"]
        if d:
            short = {"red": "High priority", "data_review": "Data review", "yellow": "Needs attention", "green": "Lower priority"}
            df_bar = pd.DataFrame([{
                "priority_level": x["priority_level"],
                # One word per line so labels fit narrow bar slots (e.g. "Needs / attention / 8 cases").
                "axis": short[x["priority_level"]].replace(" ", "|") + f"|{x['cases']:,} case{'s' if x['cases'] != 1 else ''}",
                "amount": x["claim_amount_usd"], "cases": x["cases"], "claimed": money(x["claim_amount_usd"]),
                "value": compact_money(x["claim_amount_usd"]),
                "op": 1.0 if (not picked_lvl or x["priority_level"] == picked_lvl) else 0.3} for x in d])
            pick = alt.selection_point(name="level_pick", fields=["priority_level"], on="click")
            base = alt.Chart(df_bar).encode(
                x=alt.X("axis:N", sort=list(df_bar["axis"]), title=None,
                        axis=alt.Axis(labelAngle=0, labelFontSize=11, labelLineHeight=14, labelLimit=0, labelPadding=8,
                                      labelExpr="split(datum.label, '|')", ticks=False, domain=False)),
                y=alt.Y("amount:Q", title=None, axis=alt.Axis(format="$,.0s", grid=True, tickCount=5, domain=False, ticks=False),
                        scale=alt.Scale(domainMax=float(df_bar["amount"].max()) * 1.15 or 1)),
                opacity=alt.Opacity("op:Q", scale=None, legend=None),
                tooltip=[alt.Tooltip("axis:N", title="Priority"), alt.Tooltip("cases:Q", title="Cases", format=","),
                         alt.Tooltip("claimed:N", title="Claimed")],
            )
            bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=54, cursor="pointer").encode(
                color=alt.Color("priority_level:N", scale=alt.Scale(domain=list(LEVEL_COLOR), range=list(LEVEL_COLOR.values())),
                                legend=None)).add_params(pick)
            values = base.mark_text(dy=-9, fontSize=12, fontWeight="bold", color="#17262B").encode(text="value:N")
            ev = _clickable_chart(alt.layer(bars, values).properties(height=CHART_H),
                                  bars.properties(height=CHART_H), "chart_level_bar", "level_pick")
            _apply_chart_pick(ev, "level_pick", "priority_level", "priority_level", "exposure_by_priority")
            _chart_footer("Claim amount is exposure, not evidence." + (f" · showing {LEVEL_NAME[picked_lvl]}" if picked_lvl else ""),
                          "clear_lvl", "priority_level" if picked_lvl else None)
        else:
            st.info("No cases for this selection.")

    # ---- Rules vs unsupervised second opinion
    st.markdown("**Rules vs Isolation Forest (unsupervised second opinion)**  \n<span class='ck-muted'>Each dot is a case. "
                "The shaded corner holds cases the unsupervised model finds unusual while the rules rate them below high "
                "priority. Click a dot to open the case.</span>", unsafe_allow_html=True)
    rva = svc.chart_data("rules_vs_anomaly", ss.filters)
    if rva["data"]:
        red_t = svc.rules.ruleset["thresholds"]["red"]
        top = rva["anomaly_top_pct"]
        df_r = pd.DataFrame([{**x, "level_name": LEVEL_NAME[x["priority_level"]],
                              "note": {"unsupervised_flags": "Unsupervised flags it, rules do not",
                                       "rules_flag": "Rules flag it, unsupervised does not"}.get(x["disagreement"], "Methods agree")}
                             for x in rva["data"] if x["anomaly_percentile"] is not None])
        xs = alt.X("score:Q", title="Rules score (transparent)", scale=alt.Scale(domain=[0, 100]))
        ys = alt.Y("anomaly_percentile:Q", title="Isolation Forest percentile", scale=alt.Scale(domain=[0, 100]))
        zone = alt.Chart(pd.DataFrame([{"score": 0, "score2": red_t, "anomaly_percentile": top, "a2": 100}])).mark_rect(
            color="#F2B84B", opacity=0.16).encode(x=xs, x2="score2:Q", y=ys, y2="a2:Q")
        zone_txt = alt.Chart(pd.DataFrame([{"score": 1.5, "anomaly_percentile": 99}])).mark_text(
            align="left", baseline="top", fontSize=11, color="#8A5A00", text="Unsupervised flags it, rules do not").encode(x=xs, y=ys)
        pick = alt.selection_point(name="case_pick", fields=["case_id"], on="click")
        dots = alt.Chart(df_r).mark_circle(size=110, stroke="white", strokeWidth=1, cursor="pointer").encode(
            x=xs, y=ys,
            color=alt.Color("priority_level:N", scale=alt.Scale(domain=list(LEVEL_COLOR), range=list(LEVEL_COLOR.values())),
                            legend=None),
            tooltip=[alt.Tooltip("case_id:N", title="Case"), alt.Tooltip("level_name:N", title="Rules priority"),
                     alt.Tooltip("score:Q", title="Rules score"), alt.Tooltip("anomaly_percentile:Q", title="Anomaly percentile"),
                     alt.Tooltip("note:N", title="Comparison")],
        ).add_params(pick)
        ids = alt.Chart(df_r[df_r["disagreement"].notna()]).mark_text(
            align="left", dx=9, fontSize=11, fontWeight="bold", color="#17262B").encode(x=xs, y=ys, text="case_id:N")
        ev = _clickable_chart(alt.layer(zone, zone_txt, dots, ids).properties(height=320),
                              dots.properties(height=320), "chart_rules_vs_anomaly", "case_pick")
        _apply_case_pick(ev)
        dis = [x for x in rva["data"] if x["disagreement"]]
        st.caption((f"{len(dis)} disagreement(s): " + ", ".join(x["case_id"] for x in dis) + ". " if dis else
                    "No disagreements in this selection. ")
                   + "The unsupervised model only advises; it never changes a priority by itself.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Most common active findings**  \n<span class='ck-muted'>Click a bar to see cases with that finding.</span>",
                    unsafe_allow_html=True)
        sf = svc.chart_data("signal_frequency")
        labels = list(sf["data"])[::-1]
        fig = go.Figure(go.Bar(x=[sf["data"][l] for l in labels], y=labels, orientation="h", marker_color="#2F5D62",
                               customdata=[[sf["signal_ids"][l]] for l in labels], hovertemplate="%{y}: %{x} cases<extra></extra>"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
        ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart_signals", selection_mode="points")
        _handle_chart(ev, "signal_frequency")
    with c4:
        st.markdown("**Findings that occur together**", unsafe_allow_html=True)
        co = svc.cooccurrence(top=8)
        st.dataframe(pd.DataFrame([{"Finding pair": " + ".join(c["signals"]), "Cases": c["cases"]} for c in co]),
                     hide_index=True, use_container_width=True, height=300)

    f1, f2, f3 = st.columns(3)
    for col, txt in zip((f1, f2, f3), ("Future: provider network view (needs provider IDs)",
                                       "Future: queue trends over time",
                                       "Future: investigator workload view")):
        col.markdown(f'<div class="ck-future">{txt}</div>', unsafe_allow_html=True)


CHART_H = 300


def compact_money(v: float) -> str:
    v = float(v or 0)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:,.1f}{suf}".replace(".0" + suf, suf)
    return f"${v:,.0f}"


def _style(chart):
    """Transparent background and the app's fonts and colours, instead of Streamlit's grey chart theme."""
    return (chart.configure(background="transparent", font="IBM Plex Sans")
                 .configure_view(stroke=None)
                 .configure_axis(labelColor="#52656B", titleColor="#52656B", gridColor="#E9EEF1", labelFont="IBM Plex Sans")
                 .configure_legend(labelColor="#34474C", labelFont="IBM Plex Sans"))


def _clickable_chart(layered, plain, key: str, param: str):
    """Render the labelled (layered) chart with click selection. If this Streamlit version cannot take selections on a
    layered chart, fall back to the same chart without the number labels so clicking always works."""
    try:
        return st.altair_chart(_style(layered), use_container_width=True, theme=None, on_select="rerun",
                               selection_mode=param, key=key)
    except Exception:  # noqa: BLE001  (StreamlitAPIException on older versions)
        return st.altair_chart(_style(plain), use_container_width=True, theme=None, on_select="rerun",
                               selection_mode=param, key=key + "_plain")


def _chart_footer(text: str, clear_key: str, clear_filter: str | None):
    f1, f2 = st.columns([3.2, 1])
    f1.caption(text)
    if clear_filter and f2.button("✕ Clear", key=clear_key, help="Remove this chart's filter"):
        ss.filters = {k: v for k, v in ss.filters.items() if k != clear_filter}
        st.rerun()


def _apply_case_pick(event):
    """A click on a dot in the rules-vs-anomaly chart opens that case."""
    value = _selected_value(event, "case_pick", "case_id")
    if value == ss.get("_last_case_pick"):
        return
    ss["_last_case_pick"] = value
    if value and value in svc.by_id:
        ss.chart, ss.chart_selection = "rules_vs_anomaly", {"case_id": value}
        select_case(value)
        st.rerun()


def _selected_value(event, param: str, field: str):
    """Value picked in an Altair point selection. Handles the dict shapes different Streamlit versions return."""
    try:
        sel = event["selection"] if isinstance(event, dict) else event.selection
        pts = sel.get(param) if hasattr(sel, "get") else getattr(sel, param, None)
    except (KeyError, AttributeError, TypeError):
        return None
    item = pts[0] if isinstance(pts, list) and pts else pts if isinstance(pts, dict) else None
    if not isinstance(item, dict):
        return None
    v = item.get(field)
    return v[0] if isinstance(v, list) and v else v


def _apply_chart_pick(event, param: str, field: str, filter_key: str, chart_id: str):
    """Act only when the chart selection itself changes, so a remembered selection never re-applies a cleared filter."""
    value = _selected_value(event, param, field)
    last_key = f"_last_{param}"
    if value == ss.get(last_key):
        return
    ss[last_key] = value
    if value is None or ss.filters.get(filter_key) == [value]:
        return
    ss.filters = {**{k: v for k, v in ss.filters.items() if k != filter_key}, filter_key: [value]}
    ss.chart, ss.chart_selection = chart_id, {filter_key: value}
    st.rerun()


def _handle_chart(event, chart_id):
    try:
        points = event.selection.points if event else []
    except AttributeError:
        points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
    if not points:
        return
    p = points[0]
    cd = p.get("customdata") or []
    key = (chart_id, str(cd))
    if ss.get("_last_chart_pick") == key:
        return
    ss._last_chart_pick = key
    ss.chart = chart_id
    if chart_id in ("exposure_by_case", "amount_vs_score") and cd:
        ss.chart_selection = {"case_id": cd[0]}
        select_case(cd[0])
    elif chart_id == "signal_frequency" and cd:
        ss.filters = {"has_signal": [cd[0]]}
        ss.chart_selection = {"signal": cd[0]}
    st.rerun()


def queue_analysis():
    st.markdown("Select a case from the queue to see its assessment. Queue-level view:")
    scores = svc.score_values(ss.filters)
    fig = go.Figure()
    for lvl in ["red", "data_review", "yellow", "green"]:
        fig.add_histogram(x=[sc for sc, lv in scores if lv == lvl], name=LEVEL_NAME[lvl],
                          marker_color=LEVEL_COLOR[lvl], xbins=dict(size=5))
    t = svc.rules.ruleset["thresholds"]
    for v, name in ((t["yellow"], "yellow threshold"), (t["red"], "red threshold")):
        fig.add_vline(x=v, line_dash="dot", annotation_text=name)
    fig.update_layout(barmode="stack", height=280, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Triage score",
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Signal distribution across the queue**")
    stats = [svc.signal_distribution(sig, ss.filters) for sig in svc.rules.continuous]
    st.dataframe(pd.DataFrame([{"Signal": d["label"], "Median": d["median"], "90th pct": d["p90"], "Max": d["max"]}
                               for d in stats if d.get("n")]), hide_index=True, use_container_width=True)


def case_analysis(case_id: str):
    case = svc.by_id[case_id]
    cur = svc.ensure_ai_narrative(case_id)
    a, n = cur["assessment"], cur["narrative"] or {}
    status = svc.repo.get_status(case_id)

    top = st.columns([2.2, 1, 1])
    with top[0]:
        st.markdown(f"### {case_id}  {badge(a.priority_level)}", unsafe_allow_html=True)
        st.markdown(f'<span class="ck-muted">{case["care_type"]}, {case["state"]}, {case["claim_date"]}, '
                    f'claim {case["claim_number"]}, {money(case["claim_amount_usd"])}. Status: <b>{status}</b></span>',
                    unsafe_allow_html=True)
    with top[1]:
        st.markdown(f'<div class="ck-score" style="color:{LEVEL_COLOR[a.priority_level]}">{a.score:g}</div>'
                    '<span class="ck-muted">triage score / 100</span>', unsafe_allow_html=True)
    with top[2]:
        st.markdown(f"**Confidence: {a.confidence}**")
        st.markdown(f'<span class="ck-muted">{a.confidence_reasons[0]}</span>', unsafe_allow_html=True)

    for w in a.data_quality_warnings:
        st.warning(w)
    if a.strong_indicators:
        st.error(" ".join(a.strong_indicators))
    second = svc.disagreement(case_id)
    if second:
        (st.warning if second["kind"] == "unsupervised_flags" else st.info)(
            "**Second opinion:** " + second["message"] + "  \n" + second["action"])

    src = n.get("source", "deterministic")
    g = n.get("grounding", {})
    gtxt = "grounding check passed" if g.get("passed") else f"grounding check failed: {'; '.join(g.get('issues', [])[:2])}"
    st.markdown(f'<div class="ck-summary"><b>Assessment</b><br>{n.get("summary", "")}<br><br>'
                f'<b>Recommended next step:</b> {n.get("next_step") or a.recommended_next_step}<br>'
                f'<span class="ck-muted"><b>Uncertainty:</b> {n.get("uncertainty") or a.uncertainty}</span><br>'
                f'<span class="ck-muted">Written by {cur["llm_model"]} ({src}), {gtxt}. Score from {a.ruleset_id}. '
                f'Assessment v{cur["version"]}, {ts(cur["created_at"])}.</span></div>', unsafe_allow_html=True)
    with st.expander("What argues for a lower priority", expanded=a.priority_level != "red"):
        if a.mitigating:
            for m in a.mitigating:
                st.markdown(f"- {m}")
        else:
            st.markdown("Nothing in the data argues for a lower priority.")
        if a.caveats:
            st.markdown("**What the data cannot establish** (rule these out before escalating)")
            for c in a.caveats:
                st.markdown(f"- {c}")
    with st.expander("Related guidance"):
        for g in knowledge_base.for_assessment(a):
            st.markdown(f"**{g['citation']}**  \n{g['text']}")
        st.caption(knowledge_base.DISCLAIMER)

    history = svc.repo.assessment_history(case_id)
    rejected_now = [f for f in a.findings if f.status == "rejected"]
    if len(history) > 1:
        st.info(f"Assessment v{history[-1]['version']}: score {history[0]['score']:g} at import, now {a.score:g} "
                f"after your feedback (latest: {history[-1]['trigger']}). Every version is kept in the audit history below.")
    r1, r2, _ = st.columns([1.2, 1.2, 3])
    if r1.button("Re-run assessment", key=f"rerun_{case_id}", use_container_width=True,
                 help="Recalculate with the current rules and feedback, and rewrite the explanation (uses the LLM if connected)."):
        with st.spinner("Re-running the assessment..."):
            svc.recompute(case_id, trigger="manual re-run")
        ss.flash = f"{case_id} re-assessed. A new version was added to the audit history."
        st.rerun()
    if rejected_now and r2.button(f"Restore all ({len(rejected_now)})", key=f"restore_all_{case_id}", use_container_width=True):
        with st.spinner("Restoring findings and recalculating..."):
            svc.restore_all_findings(case_id)
        ss.flash = f"All rejected findings on {case_id} restored."
        st.rerun()

    st.markdown("#### Evidence")
    groups = svc.rules.ruleset["groups"]
    gfig = go.Figure()
    gfig.add_bar(y=[groups[g]["label"] for g in groups], x=[groups[g]["cap"] for g in groups], orientation="h",
                 marker_color="#E1E8E6", name="Group cap", hoverinfo="skip")
    gfig.add_bar(y=[groups[g]["label"] for g in groups], x=[a.group_contributions.get(g, 0) for g in groups],
                 orientation="h", marker_color=LEVEL_COLOR[a.priority_level], name="Contribution",
                 hovertemplate="%{y}: %{x:.1f} points<extra></extra>")
    gfig.update_layout(barmode="overlay", height=190, margin=dict(l=10, r=10, t=5, b=5), showlegend=False,
                       plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Points (bar length = group cap)")
    st.plotly_chart(gfig, use_container_width=True)
    st.caption("Related signals are grouped and capped so one underlying pattern cannot count several times.")

    active = [f for f in a.findings if f.triggered and f.points > 0 and f.status != "rejected"]
    if not active:
        st.info("No finding contributes points. Every signal is at or below the queue norm.")

    def _card(f):
        color = "#8A9A9E" if f.status == "rejected" else LEVEL_COLOR[a.priority_level]
        with st.container(border=True):
            c1, c2 = st.columns([4, 1.4])
            with c1:
                tag = {"rejected": " · rejected", "accepted": " · accepted"}.get(f.status, "")
                st.markdown(f'<div class="ck-evidence" style="--c:{color}"><b>{f.label}</b>'
                            f'<span class="ck-muted">{tag}</span><br>{f.observed_text}<br>'
                            f'<span class="ck-muted">{f.why}</span><br>'
                            f'<span class="ck-muted">{f.points:g} of {f.max_points:g} points'
                            f'{" (group cap applied)" if f.raw_points > f.points + 0.05 else ""}. '
                            f'Source: {", ".join(f.source_columns)}</span></div>', unsafe_allow_html=True)
                if f.reject_reason:
                    st.caption(f"Rejection reason: {f.reject_reason}")
            with c2:
                if f.status == "rejected":
                    if st.button("Restore", key=f"rs_{f.finding_id}", use_container_width=True):
                        with st.spinner("Recalculating..."):
                            svc.reset_finding(case_id, f.finding_id)
                        ss.flash = f"{f.label} restored. Assessment recalculated."
                        st.rerun()
                else:
                    if f.status != "accepted" and st.button("Accept", key=f"ac_{f.finding_id}", use_container_width=True):
                        svc.accept_finding(case_id, f.finding_id)
                        st.rerun()
                    if st.button("Reject", key=f"rj_{f.finding_id}", use_container_width=True):
                        ss.reject_target = f.finding_id
                        st.rerun()
            if ss.reject_target == f.finding_id:
                with st.form(f"reject_form_{case_id}_{f.finding_id}"):
                    category = st.selectbox("Reason type", REJECT_REASONS, key=f"rc_{case_id}_{f.finding_id}")
                    reason = st.text_input("Why is this finding wrong?", placeholder="Travel exception already verified",
                                           key=f"rt_{case_id}_{f.finding_id}")
                    b1, b2 = st.columns(2)
                    ok = b1.form_submit_button("Reject and recalculate", type="primary")
                    cancel = b2.form_submit_button("Cancel")
                if ok:
                    if not reason.strip():
                        st.error("Add a reason. It is stored in the audit trail and feeds future improvements.")
                    else:
                        before = a.score
                        with st.spinner("Recalculating score and regenerating the explanation..."):
                            new = svc.reject_finding(case_id, f.finding_id, reason, category)
                        ss.reject_target = None
                        na = new["assessment"]
                        ss.flash = (f"{f.label} rejected. Score {before:g} -> {na.score:g}, "
                                    f"priority {LEVEL_NAME[na.priority_level].lower()}. It is listed under 'Rejected by you' "
                                    "and the original assessment is kept in the audit history.")
                        st.rerun()
                if cancel:
                    ss.reject_target = None
                    st.rerun()

    for f in active:
        _card(f)
    if rejected_now:
        st.markdown("#### Rejected by you")
        st.caption("Not counted in the score. Kept here and in the audit history; restore any finding at any time.")
        for f in rejected_now:
            _card(f)

    st.markdown("#### Peer comparison")
    comp = svc.compare_to_cohort(case_id)
    st.caption(comp["note"] + " Peer comparisons are relative to the current loaded queue.")
    st.dataframe(pd.DataFrame([{"Signal": r["label"], "This case": r["selected"], "Peer median": r["peer_median"],
                                "Peer 90th pct": r["peer_p90"], "Percentile among peers": r["peer_percentile"],
                                "Meaning": r["meaning"]} for r in comp["continuous"]]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Most similar cases**")
    sim = svc.similar_cases(case_id)
    sim_df = pd.DataFrame([{"Case": s["case_id"], "Priority": f'{LEVEL_DOT.get(s["priority_level"], "")} {s["score"]}',
                            "Care type": s["care_type"], "Shared flags": ", ".join(s["shared_flags"]) or "none",
                            "Different flags": ", ".join(s["different_flags"]) or "none",
                            "Investigator status": s["investigator_status"]} for s in sim])
    ev = st.dataframe(sim_df, hide_index=True, use_container_width=True, on_select="rerun",
                      selection_mode="single-row", key=f"sim_{case_id}")
    if ev and ev.selection.rows:
        select_case(sim_df.iloc[ev.selection.rows[0]]["Case"])
        st.rerun()

    with st.expander("Anomaly cross-check"):
        an = a.anomaly
        seeds = an.get("isolation_forest_seeds")
        st.markdown(f"Isolation Forest percentile **{an.get('isolation_forest_percentile')}**"
                    + (f" (average of {seeds} forests; individual forests varied by "
                       f"{an.get('isolation_forest_seed_range')} points)" if seeds else "")
                    + f", votes {an.get('votes')}/3 (Isolation Forest {an.get('isolation_forest_flag')}, "
                    f"LOF {an.get('lof_flag')}, PCA error {an.get('pca_error_flag')}). "
                    f"Marked unusual: **{an.get('is_anomalous')}**.")
        st.caption("Used only to detect disagreement with the rules. If rules say lower priority but two methods mark the "
                   "case unusual in a suspicious direction, it is routed to Needs attention. Unusually small or quiet "
                   "claims are not escalated. An anomaly is never called fraud.")
        st.caption(f"Weight stability: priority unchanged in {a.stability * 100:.0f}% of runs with weights shifted by up to 20%.")

    st.markdown("#### Decision")
    suggested = ("Escalated" if a.strong_indicators else "Needs more information") if a.priority_level == "red" else \
        {"yellow": "Needs more information", "green": "Approved / closed", "data_review": "Needs more information"}[a.priority_level]
    st.caption(f"AI suggestion: {suggested}. You make the final call.")
    with st.form(f"decision_{case_id}"):
        new_status = st.selectbox("Status", CASE_STATUSES, index=CASE_STATUSES.index(status))
        reason = st.text_input("Reason (required for final decisions)")
        note = st.text_area("Add a note", height=70)
        saved = st.form_submit_button("Save", type="primary")
    if saved:
        try:
            if note.strip():
                svc.add_note(case_id, note)
            if new_status != status:
                svc.set_status(case_id, new_status, reason)
            ss.flash = f"{case_id} saved."
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    notes = svc.repo.notes(case_id)
    if notes:
        st.markdown("**Notes**")
        for nt in notes:
            st.markdown(f"- {nt['note']}  \n  <span class='ck-muted'>{nt['author']}, {ts(nt['created_at'])}</span>",
                        unsafe_allow_html=True)
    with st.expander(f"Audit history ({len(history)} assessment version{'s' if len(history) != 1 else ''})",
                     expanded=len(history) > 1):
        st.markdown("**Assessment versions**")
        st.dataframe(pd.DataFrame([{"Version": h["version"], "Time": ts(h["created_at"]), "Trigger": h["trigger"],
                                    "Score": h["score"], "Priority": LEVEL_NAME.get(h["priority_level"], h["priority_level"]),
                                    "Confidence": h["confidence"], "Explained by": h["llm_model"]} for h in history]),
                     hide_index=True, use_container_width=True)
        st.markdown("**All events**")
        st.dataframe(pd.DataFrame([{"Time": ts(e["ts"]), "Event": e["kind"], "Detail": e["detail"]}
                                   for e in svc.repo.audit_trail(case_id)]), hide_index=True, use_container_width=True)


def evaluation_tab():
    e = svc.evaluation_summary()
    st.markdown("This view shows whether the system is behaving well. It is not a claim of production accuracy: "
                "the 50 supplied cases have no confirmed outcomes.")
    m = st.columns(6)
    m[0].metric("Assessments", e["assessments_completed"])
    m[1].metric("Findings accepted", e["accepted_findings"])
    m[2].metric("Findings rejected", e["rejected_findings"])
    m[3].metric("Human overrides", e["human_overrides"])
    m[4].metric("Grounding pass rate", f"{e['grounding_pass_rate'] * 100:.0f}%" if e["grounding_pass_rate"] is not None else "n/a")
    m[5].metric("LLM requests", e["llm_requests"])
    st.caption(f"LLM: {e['llm']}. Tokens in/out {e['input_tokens']:,}/{e['output_tokens']:,}. "
               f"Estimated cost ${e['estimated_cost_usd']} (rates from env; verify against provider pricing). "
               f"Average LLM latency {e['avg_latency_ms'] or 'n/a'} ms. Ruleset {e['ruleset_id']}, model {e['model_version']}.")
    if e["most_rejected_signals"]:
        st.markdown("**Most rejected findings**: " + ", ".join(f"{SIGNAL_LABELS.get(k, k)} ({v})" for k, v in e["most_rejected_signals"]))
        st.markdown("**Rejection reasons**: " + ", ".join(f"{k} ({v})" for k, v in e["rejection_reasons"]))

    st.markdown("#### Evaluation suite")
    st.caption("Executable checks across data quality, scoring invariants, agent behaviour (golden questions), grounding "
               "and retrieval. Same code as `python -m cockpit.evaluation_suite`. No LLM calls.")
    if st.button("Run evaluation suite", key="run_suite"):
        with st.spinner("Running checks..."):
            ss.suite = run_evaluation_suite(svc)
    suite = ss.get("suite") or svc.repo.latest_evaluation("evaluation-suite")
    if suite:
        st.dataframe(pd.DataFrame([{"Layer": k, "Passed": v["passed"], "Total": v["total"]}
                                   for k, v in sorted(suite["layers"].items())]), hide_index=True, use_container_width=True)
        if suite["failures"]:
            st.error(f"{len(suite['failures'])} check(s) failed")
            st.dataframe(pd.DataFrame(suite["failures"]), hide_index=True, use_container_width=True)
        else:
            st.success(f"All {suite['total']} checks passed ({suite['ruleset_id']}, {suite['model_version']}).")
        st.caption("Not measured: " + "; ".join(suite["not_measured"]) + ".")

    st.markdown("#### Outcome labels (for future supervised models)")
    lab = svc.label_summary()
    l1, l2, l3 = st.columns(3)
    l1.metric("Cases with a final decision", lab["decided"])
    l2.metric("Escalated / solved", lab["positive"])
    l3.metric("Likely false positive / closed", lab["negative"])
    st.caption("No supervised model is used. The supplied cases have no confirmed outcomes, and a model trained on "
               "made-up labels would not be fit for production. Every final investigator decision is stored and becomes "
               "a real training label; once enough reviewed outcomes exist, a supervised model can be trained on them and "
               "compared against the rules before it is ever allowed to influence priorities.")

    runs = svc.repo.agent_runs()
    if runs:
        with st.expander(f"Agent runs ({len(runs)})"):
            st.dataframe(pd.DataFrame([{"Time": ts(r["created_at"]), "Agent": r["agent"], "Case": r["case_id"],
                                        "Model": f"{r['provider']}:{r['model']}", "Tools": r["tool_calls"],
                                        "Grounded": bool(r["grounding_passed"]), "Fallback": bool(r["fallback_used"]),
                                        "Latency ms": r["latency_ms"]} for r in runs[::-1]]),
                         hide_index=True, use_container_width=True)


def data_tab():
    st.markdown("#### Load data")
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Upload CSV or Excel with the same columns", type=["csv", "xlsx", "xls"])
        if up is not None and st.button("Import file", type="primary"):
            rep = svc.import_file(up, up.name)
            if rep.blocking_errors:
                st.error("Import blocked: " + "; ".join(rep.blocking_errors))
            else:
                ss.flash = f"Imported {rep.rows_loaded} rows from {up.name}."
                select_case(None, "Overview")
                st.rerun()
    with c2:
        if st.button("Reload supplied sample", use_container_width=True):
            with open(settings.sample_csv, "rb") as fh:
                svc.import_file(fh, settings.sample_csv.name)
            ss.flash = "Sample reloaded."
            st.rerun()
        st.markdown('<div class="ck-future">Future: connect a database source</div>', unsafe_allow_html=True)

    rep = svc.report
    st.markdown("#### Validation summary")
    v = st.columns(6)
    v[0].metric("Rows loaded", rep.rows_loaded)
    v[1].metric("Required fields", "valid" if rep.required_fields_ok else "missing")
    v[2].metric("Missing values", rep.missing_values)
    v[3].metric("Duplicate case IDs", len(rep.duplicate_case_ids))
    v[4].metric("Duplicate claim numbers", len(rep.duplicate_claim_numbers))
    v[5].metric("Invalid values", len(rep.invalid_values))
    st.caption(f"Signals found: {rep.signals_found}. Assessment status: {rep.status}.")
    st.info(explain_validation(svc))

    st.markdown("#### AI narratives")
    st.caption("Every case already has a deterministic explanation. With an LLM configured, the full queue can be "
               "narrated in one batch (the overnight job in production). Otherwise narratives are written when a case is opened.")
    if st.button("Generate AI narratives for all cases", disabled=not providers.llm_available()):
        bar = st.progress(0.0)
        n = batch_generate(svc, progress=lambda i, total: bar.progress(i / total))
        ss.flash = f"Generated {n} narratives."
        st.rerun()

    st.markdown("#### Active ruleset")
    rs = svc.rules.ruleset
    st.caption(rs["description"])
    st.dataframe(pd.DataFrame([{"Signal": m["label"], "Column": k, "Type": m["type"], "Group": rs["groups"][m["group"]]["label"],
                                "Max points": m["points"]} for k, m in rs["signals"].items()]), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame([{"Combination": c["label"], "Points": c["points"], "Why": c["why"]} for c in rs["combinations"]]),
                 hide_index=True, use_container_width=True)
    caps_txt = ", ".join(f"{g['label']} {g['cap']}" for g in rs["groups"].values())
    st.caption(f"Group caps: {caps_txt}. "
               f"Thresholds: yellow {rs['thresholds']['yellow']}, red {rs['thresholds']['red']}. "
               f"Continuous signals score from {rs['continuous_scoring']['z_start']} to {rs['continuous_scoring']['z_full']} robust SD above the queue median.")

    with st.expander("Custom rule builder (preview only)"):
        st.caption("Future: safe rule builder with versioning and approval. This preview shows which cases a rule would "
                   "match. It does not change any score.")
        cols = st.columns(4)
        field = cols[0].selectbox("Field", svc.rules.continuous + svc.rules.binary + ["claim_amount_usd"])
        op = cols[1].selectbox("Operator", [">", ">=", "<", "<=", "=="])
        val = cols[2].number_input("Value", value=10.0)
        pts = cols[3].number_input("Points", value=10, min_value=0, max_value=30)
        ops = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b}
        hits = [c["case_id"] for c in svc.cases if c.get(field) is not None and ops[op](float(c[field]), val)]
        st.markdown(f"`IF {field} {op} {val:g} THEN add {pts} points` would match **{len(hits)}** cases: {', '.join(hits[:20])}")

    with st.expander("Feedback corpus (future RAG source)"):
        docs = svc.repo.knowledge()
        if docs:
            st.dataframe(pd.DataFrame([{"Kind": d["kind"], "Case": d["case_id"], "Title": d["title"], "Body": d["body"]} for d in docs]),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("Rejected findings and final dispositions will appear here.")

    with st.expander("Reset"):
        if st.button("Delete all decisions and reload sample"):
            svc.repo.reset()
            get_service.clear()
            for k in list(ss.keys()):
                del ss[k]
            st.rerun()


with mid, st.container(key="dash"):
    with st.container(key="tabbar"):
        tab_cols = st.columns([1, 1, 1.15, 1.1, 1.6], gap="small")
        for col, name in zip(tab_cols, TABS):
            if col.button(name, key="tab_" + name.lower().replace(" ", "_").replace("&", "and"),
                          type="primary" if ss.tab == name else "secondary", use_container_width=True):
                ss.tab = name
                st.rerun()
    if ss.tab == "Overview":
        overview_tab()
    elif ss.tab == "Analysis":
        if ss.case_id and ss.case_id in svc.by_id:
            if st.button("Back to queue view"):
                select_case(None, "Analysis")
                st.rerun()
            case_analysis(ss.case_id)
        else:
            queue_analysis()
    elif ss.tab == "AI evaluation":
        evaluation_tab()
    else:
        data_tab()


# ---------------------------------------------------------------- copilot
def run_proposal(p: dict):
    a = p["args"]
    act = p["action"]
    if act == "accept_finding":
        svc.accept_finding(a["case_id"], a["finding_id"])
    elif act == "reject_finding":
        svc.reject_finding(a["case_id"], a["finding_id"], a.get("reason") or "Rejected via copilot")
    elif act == "add_case_note":
        svc.add_note(a["case_id"], a["note"], author="investigator (drafted by copilot)")
    elif act == "set_case_status":
        svc.set_status(a["case_id"], a["status"], a.get("reason", ""))


with right:
    with st.container(key="copilot"):
        h_a, h_b = st.columns([3, 1.3])
        h_a.markdown("<div class='cp-title'>Copilot</div><div class='cp-sub'>Answers use only the loaded data and tools.</div>",
                     unsafe_allow_html=True)
        if ss.chat and h_b.button("New chat", key="cp_new", use_container_width=True):
            ss.chat, ss.proposals = [], []
            st.rerun()
        ctx_bits = [f"Case {ss.case_id}" if ss.case_id else "No case selected", ss.tab]
        if ss.filters:
            ctx_bits.append(f"{len(ss.filters)} filter{'s' if len(ss.filters) > 1 else ''}")
        if ss.chart_selection:
            ctx_bits.append("chart selection")
        st.markdown("".join(f'<span class="cp-ctx">{b}</span>' for b in ctx_bits), unsafe_allow_html=True)

        with st.container(height=500, border=False, key="chatlog"):
            if not ss.chat:
                st.markdown("<div class='cp-empty'>How can I help with "
                            + (f"<b>{ss.case_id}</b>" if ss.case_id else "the queue") + "?</div>", unsafe_allow_html=True)
            for i, m in enumerate(ss.chat):
                if m["role"] == "user":
                    with st.container(key=f"umsg_{i}"):
                        st.markdown(m["content"], unsafe_allow_html=True)
                    continue
                with st.container(key=f"amsg_{i}"):
                    st.markdown(m["content"])
                    meta = m.get("meta") or {}
                    g = meta.get("grounding", {})
                    status = "✓ grounded in the data" if g.get("passed") else "⚠ unverified: " + "; ".join(g.get("issues", [])[:2])
                    st.markdown(f"<div class='cp-meta'>{meta.get('source', '')} · {status}</div>", unsafe_allow_html=True)
                    if meta.get("tool_calls"):
                        with st.expander(f"Used {len(meta['tool_calls'])} tool{'s' if len(meta['tool_calls']) > 1 else ''}"):
                            for t in meta["tool_calls"]:
                                st.code(f"{t['tool']}({t['args']})", language="text")

        for i, p in enumerate(list(ss.proposals)):
            with st.container(border=True):
                st.markdown(f"**Proposed:** {p['action'].replace('_', ' ')}  \n`{p['args']}`")
                b1, b2 = st.columns(2)
                if b1.button("Confirm", key=f"pc_{i}", type="primary", use_container_width=True):
                    try:
                        run_proposal(p)
                        ss.flash = f"Confirmed: {p['action'].replace('_', ' ')}."
                    except ValueError as e:
                        ss.flash = f"Could not apply: {e}"
                    ss.proposals.pop(i)
                    st.rerun()
                if b2.button("Dismiss", key=f"pd_{i}", use_container_width=True):
                    ss.proposals.pop(i)
                    st.rerun()

        suggestions = (["Why is this case high priority?", "How does it compare with similar cases?",
                        "What evidence argues against escalation?", "What should I verify first?"]
                       if ss.case_id else
                       ["Summarise the queue", "Show cases with high amounts but lower priority",
                        "Which signals co-occur most?", "Explain this chart"])
        picked = None
        with st.container(key="suggest"):
            sc = st.columns(2, gap="small")
            for i, sug in enumerate(suggestions):
                if sc[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                    picked = sug
        typed = st.chat_input("Ask about this case or the queue", key="copilot_input")

    question = picked or ((typed or "").strip() or None)
    if question and QUESTION_LIMIT and ss.questions_asked >= QUESTION_LIMIT:
        ss.flash = f"This demo allows {QUESTION_LIMIT} copilot questions per session."
        st.rerun()
    if question:
        ss.questions_asked += 1
        ss.chat.append({"role": "user", "content": question + (f"  \n<span class='cp-case'>{ss.case_id}</span>" if ss.case_id else ""),
                        "text": question})
        ctx = {"case_id": ss.case_id, "tab": ss.tab, "filters": ss.filters, "chart": ss.chart,
               "chart_selection": ss.chart_selection, "visible_count": pg["total"]}
        with st.spinner("Checking the data..."):
            res = make_copilot(svc).ask(question, ctx, history=[{"role": m["role"], "content": m.get("text", m["content"])}
                                                                for m in ss.chat[:-1]])
        ss.chat.append({"role": "assistant", "content": res.answer,
                        "meta": {"source": res.source, "grounding": res.grounding, "tool_calls": res.tool_calls}})
        ss.proposals.extend(res.proposals)
        if res.ui_filters:
            ss.filters = res.ui_filters
            ss.flash = "Copilot filtered the queue. Use Clear in Filters to reset."
        st.rerun()
