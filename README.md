# AI Investigation Cockpit

An AI-first case review workspace for insurance fraud investigators. When an investigator opens the
app, every referral in the queue already has an evidence-backed assessment: a transparent triage
score, the findings behind it, what argues against it, a peer comparison, a recommended next step
and an honest uncertainty statement. The investigator can question the AI, reject findings (the
score recalculates), add notes and record the final decision. The AI prepares; the human decides.

## Quick start

### Check the files first

After downloading or uploading through the GitHub website, run `python3 verify_install.py`. It lists any
missing file by name (the start scripts run it automatically). Hidden files such as `.env.example`,
`.gitignore`, `.streamlit/config.toml` and `.github/workflows/tests.yml` are part of the project.

### Windows (easiest)

Double-click **`run_windows.bat`**. The first run creates a virtual environment and installs packages
(a few minutes); later runs start immediately. Your browser opens at http://localhost:8501.

If it says Python was not found, install Python 3.12 or newer from https://www.python.org/downloads/ and
tick **"Add python.exe to PATH"** on the first installer screen.

Manual equivalent (Command Prompt or PowerShell, no activation needed):

```cmd
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

### macOS / Linux

```bash
bash run.sh
```

or manually: `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && cp .env.example .env && .venv/bin/python -m streamlit run app/streamlit_app.py`

### Docker (only Docker and a browser needed)

```bash
cp .env.example .env        # Windows: copy .env.example .env
docker compose up --build
```

Open http://localhost:8501. The container runs as a non-root user; decisions persist in a named volume.

On first start the supplied CSV is imported, validated and assessed (about 1 second).

### No API key?

Everything still works: import, validation, scoring, charts, peer comparison, guidance, accept/reject,
notes, decisions, audit and evaluation. Narratives come from deterministic templates and the copilot
uses a keyword router over the same tools. The header shows which mode is active.

## Presentation

The slide deck is in [docs/Fraud_Investigator_Tool.pptx](docs/Fraud_Investigator_Tool.pptx) (19 slides with speaker notes).

## Hosting

See **[docs/DEPLOY.md](docs/DEPLOY.md)**: Streamlit Community Cloud (free reviewer link), Google Cloud Run or
Render/Railway (Docker), why not Vercel, and a public-link checklist. Optional settings for public demos:
`APP_PASSWORD` (access password), `SESSION_QUESTION_LIMIT` (copilot questions per visitor) and `DEMO_NOTICE`.
A model connection made in the app is private to each visitor's session.

## Choosing the LLM

Two ways, no code changes either way:

1. **In the app**: header, then **Model connection**. Pick a provider, enter a model ID and key, click
   **Test**, then **Connect**. The connection is private to your browser session: the key is never saved, logged,
   sent back to the browser or shared with other visitors.
2. **In `.env`**: set one key. The provider is auto-detected; override with `LLM_PROVIDER` and `LLM_MODEL`.

| Provider | Key | Default model | Notes |
|---|---|---|---|
| Google Gemini (spec default) | `GEMINI_API_KEY` | `gemini-2.5-flash` | Native REST, function calling |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | Native REST, tool use |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | Chat Completions, tool calls |
| OpenAI-compatible server | `LLM_BASE_URL` (+ optional `LLM_API_KEY`) | set `LLM_MODEL` | Ollama, vLLM, LM Studio, LiteLLM gateway |
| None | - | deterministic | Default when no key is set |

Check a key from the command line (one tiny billable request, also checks tool calling):

```bash
python -m cockpit.check_provider
python -m cockpit.check_provider --provider anthropic --model claude-sonnet-5
```

All providers share one interface (`cockpit/llm/providers.py`) over plain REST, so no vendor SDK is needed.
Requests retry once on transient errors (429/5xx/timeouts), never follow redirects, and require HTTPS
(plain HTTP only for localhost). Error messages never include keys. Adding a provider means one class
with a `chat()` method.

**Agent runtime.** `AGENT_RUNTIME=native` (default) runs a provider-agnostic tool loop that is fully
tested. `AGENT_RUNTIME=adk` runs the copilot on Google ADK (`pip install -r requirements-optional.txt`)
with the same tools, prompt and grounding checks.

## Demo flow (about 5 minutes)

1. Open the app (the Fraud Investigator Tool). The queue is already assessed and ordered. The coloured
   cards at the top are clickable filters: High priority, Needs attention, Lower priority, Data review,
   Unreviewed, Reviewed; All cases resets.
2. **Data & rules** tab: validation summary. The repeated claim number (C1001 / C1031) is shown as a
   data-quality warning, not as fraud.
3. **Overview**: the "Rules vs Isolation Forest" chart shows each case's rules score against the unsupervised
   anomaly percentile; the shaded corner holds disagreements (C1011, C1019). Click a dot to open the case, where a
   "Second opinion" note explains the disagreement. Then click the High priority card, a slice of the care-type
   donut, and a bar in "Claim exposure by priority". The charts filter each other and the queue, and the copilot receives that context.
   Use ✕ Clear next to a chart title, or the All cases card, to reset.
4. Click a case (C1024). The Analysis view shows the assessment, score breakdown by capped group,
   finding cards with source columns, peer comparison, similar cases and the anomaly cross-check.
5. In the ChatGPT-style copilot panel on the right, ask "Why is this case high priority?", "What argues against escalation?" and
   "How should I verify the duplicate billing?" (answers cite the playbook section). Expand the tool-call
   trace to see what data it read. Ask "Was this claim paid twice?" to see it decline: payment records
   are not in the data.
6. Reject a finding with a reason type and a note (for example distance: "Explained by a legitimate reason",
   "Travel exception already verified"). The score and narrative are recalculated; the finding moves to
   "Rejected by you" (restore any time, or Restore all), a banner shows the score change, and every version
   is listed in the audit history. **Re-run assessment** recalculates on demand, for example after
   connecting an LLM.
7. Add a note and set "Needs more information". Final decisions require a reason.
8. **AI evaluation**: click **Run evaluation suite** (140 checks), then review accepted/rejected findings,
   rejection reasons, overrides, grounding pass rate, LLM cost and latency, and the real outcome labels
   collected so far.

## Architecture

```
CSV / Excel
   │
   ▼
Validation (deterministic) ──► data-quality warnings, blocking errors
   │
   ▼
SQLite: import_batches, cases, case_signals
   │
   ▼
Scoring models (same Assessment contract, spec 9.5)
   ├─ TransparentRulesModel   primary score 0-100, versioned ruleset JSON
   ├─ AnomalyDetectionModel   seed-averaged Isolation Forest + LOF + PCA, disagreement detection only
   └─ SupervisedModel         development only: not used by the app until real outcome labels exist
   │
   ▼
Agents
   ├─ Data Quality agent      explains the validation report
   ├─ Analysis agent          deterministic evidence pack (findings, caveats, peers, anomaly, guidance, next steps)
   ├─ Assessment agent (LLM)  writes summary / key points / mitigating / next step / uncertainty
   ├─ Validation agent        Python grounding check on every LLM output, one retry, then fallback
   └─ Copilot agent (LLM)     context-aware, 19 allowlisted tools, write tools = proposals only
   │
Knowledge layer: 9 versioned playbooks (cockpit/knowledge/*.md), section-level BM25 + routing by active
                 findings, cited by doc id and section; plus the investigator feedback corpus
   │
   ▼
Streamlit UI: queue │ Overview · Analysis · AI evaluation · Data & rules │ copilot
   │
   ▼
SQLite: assessments (versioned), assessment_findings, investigator_decisions, case_notes,
        case_status_history, agent_runs, agent_tool_calls, model_evaluations, knowledge_documents
```

**The LLM never produces a score.** Scores, points, percentiles and priority levels come from Python.
The LLM only explains an evidence pack it is given, and its output is checked before anyone sees it.

### How priority is decided

```
score = Σ over groups  min(group cap, Σ signal points in group)   +   min(14, combination bonuses)
```

| Group | Signals | Cap |
|---|---|---|
| Billing integrity | duplicate service (20), service overlap (18) | 30 |
| Relationship | shared contact (14), recent policy change (6) | 16 |
| Utilization | weekly visits (10), distance (10), prior claims (6) | 20 |
| Billing amount and pattern | amount vs peer (10), weekend ratio (7), round-dollar ratio (7) | 20 |
| Combinations | dup + overlap (8), shared contact + dup (5), shared contact + overlap (5), frequency + distance (5), amount + pattern (5), 3+ moderately unusual signals (6) | 14 |

- Caps sum to 100, so the score never needs clipping and one pattern cannot be counted several times.
- Continuous signals use a robust z-score against the loaded queue: `(value − median) / (1.4826 × MAD)`.
  Points scale linearly from 0.75 to 3.0 robust SD.
- Levels: red ≥ 60 or a policy strong indicator (duplicate billing with service overlap); yellow ≥ 20;
  otherwise green. A data-quality issue moves a green case to Data review.
- If rules say green but two of three anomaly methods say unusual, the case goes to yellow, but only when it is
  unusual in a suspicious direction (a flag is set or a signal is above the norm). Isolation Forest also isolates
  unusually small, quiet claims; those are not escalated. The forest is averaged over 10 seeds for stability.
- **Confidence is separate from suspicion.** It drops for data-quality issues, missing values, thin peer
  groups, rules/anomaly disagreement, and instability: each case is re-scored 200 times with every weight
  shifted by up to 20%, and if the level changes in more than 15% of runs the score is near a threshold.
- **Queue order**: strong indicators, red, data review, yellow with non-high confidence, other yellow,
  green; then score; then claim amount as a tie-breaker. Amount is exposure, not evidence.
- All weights live in `cockpit/rules/ruleset_v1.json`, and each assessment stores the ruleset and model version.

Current result on the supplied file (rules-v1.1): 10 high priority, 7 needs attention, 32 lower priority,
1 data review. See [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) for a case-by-case comparison with an
independent Isolation Forest analysis and the changes it led to.

### Grounding and safety

- Every LLM answer is checked by `cockpit/agents/grounding.py`. Every case ID must exist, every number
  must appear in the evidence (0.44 and 44% both match), and every field name must be real. No
  fraud determinations or fraud probabilities are allowed.
- The assessment agent also checks the JSON schema and that it only cites active findings. On failure
  it gets one retry with the issues listed. If that fails too, the deterministic narrative is shown,
  labelled as such.
- The copilot has at most 6 tool calls per question, a timeout, no arbitrary code or SQL, and one
  grounding retry. Unverified answers are shown with a warning, never silently.
- Write tools (accept, reject, note, status) only create proposals. The investigator confirms them in the UI.
- Every assessment version, decision, note, status change, agent run and tool call is stored.

### Knowledge and the skeptic view

- **Playbooks**: `cockpit/knowledge/*.md` hold procedures and false-positive explanations as versioned files.
  Retrieval is section-level BM25 plus routing by the case's active findings; the copilot and the case view
  cite them as `PB-DUP-01 v1.0 · Verify the indicator`. They never contain scoring rules. The content is
  **synthetic demonstration guidance, not an insurer's approved procedure**.
- **Caveats**: every assessment lists what the case-level data cannot establish for each active finding
  (for example, whether both duplicate lines were paid). They are shown as "rule these out before escalating".
- **Abstention**: questions about data the file does not contain (payment records, investigation history,
  names, diagnoses) get an explicit "not in the supplied data" answer instead of a guess.

### Evaluation

Run `python -m cockpit.evaluation_suite` or click **Run evaluation suite** in the AI evaluation tab.
It runs 140 checks, exits non-zero on any failure (used in CI) and writes `data/evaluation_results.json`:

| Layer (spec 21) | Checks |
|---|---|
| Data quality | sample expectations plus adversarial inputs: empty, corrupt, wrong extension, missing column, ratio > 1, infinite value, fractional count, non-binary flag, leading-zero IDs |
| Scoring | per case: bounds, contributions add up, group caps, observed values equal source data, strong indicator is red, rejecting a finding never raises the score, deterministic, versions stored |
| Agent | golden questions with expected tool and content, including two abstention cases |
| Grounding | validator must-pass and must-fail fixtures, and every case's narrative checked against its own evidence pack |
| Retrieval | playbook Recall@3 for known queries |

`--live` also grades the configured LLM copilot on the golden set (tool use and grounding; makes API calls).

The 50 supplied cases have no outcome labels, so **no accuracy is reported on them and no supervised model is
used**. A model trained on made-up labels is not fit for production decisions. The app shows only real
behaviour: score and priority distribution, the rules-vs-Isolation-Forest comparison (unsupervised, on the real
cases), investigator accept/reject rates and reasons, overrides, grounding pass rate, LLM latency and cost, and
the **outcome labels collected so far** (every final decision is stored as a future training label).

Supervised models (logistic regression, gradient boosting) exist as a **development-only** pipeline, ready for
real labels. They can be run by hand against a documented synthetic set (`cockpit/models/synthetic.py`) as a
sanity check, never by the app:

```bash
python -m cockpit.models.supervised_model   # writes data/dev_synthetic_comparison.json, marked DEVELOPMENT ONLY
```

During development the rule thresholds were also sanity-checked against that synthetic set and against the
priority distribution of the supplied cases. Runtime scoring uses only the versioned ruleset and the real data.

## Scaling to large queues

The queue is read from the database one page at a time; nothing on screen loads the whole queue.

- **Queue index**: every assessed case has one row in `queue_index` holding what the queue sorts and
  filters on (priority group, score, amount, level, confidence, care type, status, flags), with indexes that
  match the default order. Active findings are in `queue_signals`, so "has this finding" filters and signal
  counts are SQL too. Status is kept on the index row and updated on every status change.
- **Paging**: filters are applied first, then the database sorts by strong indicators, priority, uncertainty,
  score and amount, and returns 50 rows with `LIMIT/OFFSET`. Rank numbers are global (page 2 starts at #51).
  Any filter or search change returns to page 1.
- **Counts and charts** (cards, donut, signal counts, co-occurrence) are SQL aggregates. The exposure chart
  shows the first 50 matching cases in queue order.
- **Scoring at import** is chunked with bulk writes, and the confidence stability check skips simulation when
  the score is provably far from a threshold (exact; a unit test compares it with the full simulation).
- Anomaly models fit on a fixed 20,000-case sample and score everything. Similar cases and peer
  comparisons use arrays built once at load.

Measured on 100,000 synthetic cases (SQLite, laptop-class CPU):

| Operation | Time |
|---|---|
| Page 1 (50 rows + total count) | 5 ms |
| Page 1,000 / last page (2,000) | 99 ms / 190 ms |
| Filtered page (priority + care type; Unreviewed) | 6–12 ms |
| Search, or "has finding" filter | ~250 ms |
| All summary cards | 86 ms |
| Open a case (assessment, peers, similar cases) | under 10 ms each |
| Reject a finding, recalculate, re-index | 13 ms |
| Import, validate and score 100,000 cases | about 4 minutes |

Toward a million rows: use `python -m cockpit.batch --file big.csv` (the overnight job) rather than a browser
upload; move to PostgreSQL; switch deep pages to keyset pagination (OFFSET cost grows with page number);
compress stored assessment payloads (about 12 KB per case in SQLite today); keep case records in the database
instead of process memory; and use an approximate-nearest-neighbour index for similar cases.
Import limits are configurable with `MAX_IMPORT_ROWS` and `MAX_UPLOAD_MB`.

## Tests

```bash
python -m pytest -q                  # 55 tests, including real Streamlit AppTest UI tests
python tests/smoke_ui.py             # stubbed headless run of the UI script (works without Streamlit)
python -m cockpit.evaluation_suite   # 140 evaluation checks
```

The tests cover validation and adversarial inputs, scoring invariants, the reject/recompute/audit loop,
grounding, all LLM providers' tool-call formats (mocked HTTP), retry, URL safety, key redaction, runtime
model switching, knowledge retrieval, abstention and the evaluation suite. GitHub Actions
(`.github/workflows/tests.yml`) runs all three on Python 3.12 and 3.13 for every push.

Other entry points:

```bash
python -m cockpit.batch      # overnight job: import, assess, write AI narratives, refresh evaluation
python -m cockpit.mcp_server # MCP server exposing read/analysis tools (pip install mcp)
```

## Project layout

```
app/streamlit_app.py            UI
cockpit/config.py               env-driven settings
cockpit/data_loader.py          CSV/Excel load + validation
cockpit/rules/ruleset_v1.json   versioned rule registry
cockpit/models/                 rules (primary), anomaly cross-check; supervised + synthetic set are development only
cockpit/peer.py                 cohort comparison + similar cases
cockpit/service.py              orchestration, caching, workflow, charts, evaluation summary
cockpit/db.py                   SQLite repository (schema maps 1:1 to PostgreSQL)
cockpit/llm/providers.py        Gemini / Claude / OpenAI / none
cockpit/agents/                 prompts, tools, grounding, assessment agent, copilot, ADK runtime
cockpit/knowledge/, knowledge_base.py   playbooks + section retriever
cockpit/evaluation_suite.py     executable spec-21 evaluation
cockpit/check_provider.py       live key / tool-calling check
cockpit/batch.py, mcp_server.py batch job, MCP boundary
run_windows.bat, run.sh         one-command start
tests/                          unit tests + UI smoke test
```

## Assumptions and deviations from the spec

- **Assumptions**: the fraud engine's referral flags are reliable inputs; 0 means "not triggered";
  `amount_vs_peer_avg_pct` is already peer-relative; with 50 rows the peer group is the loaded queue.
- **Pydantic, SQLAlchemy and Alembic** were replaced by dataclasses and a thin `sqlite3` repository to
  keep dependencies minimal. The schema is PostgreSQL-ready.
- **Google ADK** is an optional runtime rather than the default, so the app runs with any model. The ADK
  path is written against the ADK 1.x API but has not been exercised against a live key. The native
  runtime is the tested path.
- **`uv.lock`** is not included. Use `requirements.txt` or run `uv lock` locally.
- The Streamlit UI was verified with a stubbed headless run in the build environment. The real AppTest UI
  tests run wherever Streamlit is installed (your machine, CI). Click through once in a browser before
  recording the demo.
- No live LLM calls were made during development; provider formats are verified against mocked HTTP.
  Run `python -m cockpit.check_provider` once with your key.

## Roadmap

RAG over investigator decisions and playbooks (the feedback corpus is already collected in
`knowledge_documents`), a safe rule builder with approval (preview exists), supervised models trained
on real investigator outcomes, PostgreSQL and multi-user, provider/member network analysis when
identifiers exist, drift monitoring.
