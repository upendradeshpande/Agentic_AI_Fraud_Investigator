"""SQLite persistence (spec section 16). Plain sqlite3 so it runs anywhere with no server.
The schema maps 1:1 to PostgreSQL for the production path."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS import_batches (
  batch_id TEXT PRIMARY KEY, filename TEXT, file_hash TEXT, rows INTEGER,
  report_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS cases (
  case_id TEXT PRIMARY KEY, batch_id TEXT, case_hash TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS case_signals (
  case_id TEXT, signal TEXT, value REAL, PRIMARY KEY (case_id, signal));
CREATE TABLE IF NOT EXISTS rulesets (
  ruleset_id TEXT PRIMARY KEY, content_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS assessments (
  assessment_id TEXT PRIMARY KEY, case_id TEXT, version INTEGER, trigger TEXT,
  score REAL, priority_level TEXT, confidence TEXT, payload_json TEXT, narrative_json TEXT,
  ruleset_id TEXT, model_version TEXT, llm_model TEXT, case_hash TEXT,
  is_current INTEGER, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_assess_case ON assessments(case_id, is_current);
CREATE TABLE IF NOT EXISTS assessment_findings (
  assessment_id TEXT, finding_id TEXT, label TEXT, points REAL, triggered INTEGER, status TEXT,
  PRIMARY KEY (assessment_id, finding_id));
CREATE TABLE IF NOT EXISTS case_state (
  case_id TEXT PRIMARY KEY, status TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS investigator_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, assessment_id TEXT, action TEXT,
  finding_id TEXT, reason TEXT, note TEXT, ruleset_id TEXT, model_version TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS case_notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, note TEXT, author TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS case_status_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, from_status TEXT, to_status TEXT,
  reason TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS agent_runs (
  run_id TEXT PRIMARY KEY, agent TEXT, case_id TEXT, provider TEXT, model TEXT,
  question TEXT, answer TEXT, input_tokens INTEGER, output_tokens INTEGER, latency_ms INTEGER,
  tool_calls INTEGER, grounding_passed INTEGER, grounding_issues TEXT, fallback_used INTEGER,
  error TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS agent_tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, tool TEXT, args_json TEXT,
  result_preview TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS model_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, model_version TEXT, payload_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS queue_index (
  case_id TEXT PRIMARY KEY, batch_id TEXT, order_group INTEGER, score REAL, amount REAL, priority_level TEXT,
  confidence TEXT, care_type TEXT, state TEXT, claim_number TEXT, claim_date TEXT,
  strong INTEGER, dq INTEGER, anomalous INTEGER, top_evidence TEXT, triggered TEXT, status TEXT);
CREATE INDEX IF NOT EXISTS ix_queue_order ON queue_index(batch_id, order_group, score DESC, amount DESC);
CREATE INDEX IF NOT EXISTS ix_queue_level ON queue_index(batch_id, priority_level);
CREATE INDEX IF NOT EXISTS ix_queue_care ON queue_index(batch_id, care_type);
CREATE INDEX IF NOT EXISTS ix_queue_score ON queue_index(batch_id, score DESC);
CREATE INDEX IF NOT EXISTS ix_queue_amount ON queue_index(batch_id, amount DESC);
CREATE INDEX IF NOT EXISTS ix_queue_status ON queue_index(batch_id, status);
CREATE TABLE IF NOT EXISTS queue_signals (
  case_id TEXT, signal TEXT, PRIMARY KEY (case_id, signal));
CREATE INDEX IF NOT EXISTS ix_qsig_signal ON queue_signals(signal, case_id);
CREATE INDEX IF NOT EXISTS ix_cases_batch ON cases(batch_id, case_id);
CREATE INDEX IF NOT EXISTS ix_state_status ON case_state(status);
CREATE TABLE IF NOT EXISTS knowledge_documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, case_id TEXT, title TEXT, body TEXT, created_at REAL);
"""


class Repository:
    def __init__(self, path: str | None = None):
        self.path = path or settings.sqlite_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            # Migration: queue_index from an earlier version had no batch_id; drop it (it is rebuilt automatically).
            cols = [r[1] for r in c.execute("PRAGMA table_info(queue_index)")]
            if cols and not {"batch_id", "status"} <= set(cols):
                c.executescript("DROP TABLE queue_index; DROP TABLE IF EXISTS queue_signals;")
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------ import
    def save_batch(self, filename, file_hash, cases, report: dict) -> str:
        batch_id = f"B-{uuid.uuid4().hex[:8]}"
        with self.conn() as c:
            c.execute("INSERT INTO import_batches VALUES (?,?,?,?,?,?)",
                      (batch_id, filename, file_hash, len(cases), json.dumps(report), time.time()))
            now = time.time()
            c.executemany("INSERT OR REPLACE INTO cases VALUES (?,?,?,?)",
                          [(case["case_id"], batch_id, case_hash(case), json.dumps(case)) for case in cases])
            c.executemany("INSERT OR REPLACE INTO case_signals VALUES (?,?,?)",
                          [(case["case_id"], k, v) for case in cases for k, v in case.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "claim_amount_usd"])
            c.executemany("INSERT OR IGNORE INTO case_state VALUES (?,?,?)",
                          [(case["case_id"], "Unreviewed", now) for case in cases])
        return batch_id

    def latest_batch(self):
        with self.conn() as c:
            r = c.execute("SELECT * FROM import_batches ORDER BY created_at DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def load_cases(self, batch_id: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT data_json FROM cases WHERE batch_id=? ORDER BY case_id", (batch_id,)).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def save_ruleset(self, ruleset: dict):
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO rulesets VALUES (?,?,?)",
                      (ruleset["ruleset_id"], json.dumps(ruleset), time.time()))

    # ------------------------------------------------------- assessments
    def current_assessment(self, case_id):
        with self.conn() as c:
            r = c.execute("SELECT * FROM assessments WHERE case_id=? AND is_current=1", (case_id,)).fetchone()
        return dict(r) if r else None

    def all_current_assessments(self) -> dict[str, dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM assessments WHERE is_current=1").fetchall()
        return {r["case_id"]: dict(r) for r in rows}

    def assessment_history(self, case_id):
        with self.conn() as c:
            rows = c.execute("SELECT assessment_id, version, trigger, score, priority_level, confidence, "
                             "ruleset_id, model_version, llm_model, created_at FROM assessments "
                             "WHERE case_id=? ORDER BY version", (case_id,)).fetchall()
        return [dict(r) for r in rows]

    def save_assessment(self, assessment: dict, narrative: dict | None, *, trigger: str,
                        llm_model: str, case_hash_value: str, queue_row: dict | None = None) -> str:
        aid = f"A-{uuid.uuid4().hex[:10]}"
        with self.conn() as c:
            if queue_row:
                self._upsert_queue(c, [queue_row])
            prev = c.execute("SELECT MAX(version) v FROM assessments WHERE case_id=?",
                             (assessment["case_id"],)).fetchone()["v"] or 0
            c.execute("UPDATE assessments SET is_current=0 WHERE case_id=?", (assessment["case_id"],))
            c.execute("INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                aid, assessment["case_id"], prev + 1, trigger, assessment["score"],
                assessment["priority_level"], assessment["confidence"], json.dumps(assessment),
                json.dumps(narrative) if narrative else None, assessment["ruleset_id"],
                assessment["model_version"], llm_model, case_hash_value, 1, time.time()))
            for f in assessment["findings"]:
                c.execute("INSERT INTO assessment_findings VALUES (?,?,?,?,?,?)",
                          (aid, f["finding_id"], f["label"], f["points"], int(f["triggered"]), f["status"]))
        return aid

    def save_assessments_bulk(self, items: list[tuple[dict, dict, str, dict]], trigger: str, llm_model: str):
        """items: (assessment_dict, narrative, case_hash, queue_row). One transaction per call."""
        now = time.time()
        with self.conn() as c:
            ids = [a["case_id"] for a, *_ in items]
            prev = {}
            for chunk in _chunks(ids, 900):
                q = f"SELECT case_id, MAX(version) v FROM assessments WHERE case_id IN ({','.join('?' * len(chunk))}) GROUP BY case_id"
                prev.update({r["case_id"]: r["v"] for r in c.execute(q, chunk)})
                c.execute(f"UPDATE assessments SET is_current=0 WHERE case_id IN ({','.join('?' * len(chunk))})", chunk)
            rows, frows = [], []
            for a, narrative, h, _ in items:
                aid = f"A-{uuid.uuid4().hex[:10]}"
                rows.append((aid, a["case_id"], (prev.get(a["case_id"]) or 0) + 1, trigger, a["score"], a["priority_level"],
                             a["confidence"], json.dumps(a), json.dumps(narrative), a["ruleset_id"], a["model_version"],
                             llm_model, h, 1, now))
                frows += [(aid, f["finding_id"], f["label"], f["points"], int(f["triggered"]), f["status"])
                          for f in a["findings"] if f["triggered"] or f["status"] != "open"]
            c.executemany("INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            c.executemany("INSERT OR REPLACE INTO assessment_findings VALUES (?,?,?,?,?,?)", frows)
            self._upsert_queue(c, [q for *_, q in items])

    @staticmethod
    def _upsert_queue(c, rows: list[dict]):
        c.executemany(
            "INSERT OR REPLACE INTO queue_index VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
            "COALESCE((SELECT status FROM case_state WHERE case_id = ?), 'Unreviewed'))", [
                (r["case_id"], r["batch_id"], r["order_group"], r["score"], r["amount"], r["priority_level"],
                 r["confidence"], r["care_type"], r["state"], r["claim_number"], r["claim_date"], int(r["strong"]),
                 int(r["dq"]), int(r["anomalous"]), r["top_evidence"], "," + ",".join(r["triggered"]) + ",",
                 r["case_id"]) for r in rows])
        ids = [r["case_id"] for r in rows]
        for chunk in _chunks(ids, 900):
            c.execute(f"DELETE FROM queue_signals WHERE case_id IN ({','.join('?' * len(chunk))})", chunk)
        c.executemany("INSERT OR IGNORE INTO queue_signals VALUES (?,?)",
                      [(r["case_id"], s) for r in rows for s in r["triggered"]])

    def current_hashes(self) -> dict[str, tuple]:
        """case_id -> (case_hash, ruleset_id, model_version) without loading assessment payloads."""
        with self.conn() as c:
            return {r[0]: (r[1], r[2], r[3]) for r in c.execute(
                "SELECT case_id, case_hash, ruleset_id, model_version FROM assessments WHERE is_current=1")}

    # ------------------------------------------------------- paged queue (SQL)
    ORDERS = {
        "queue": "q.order_group, q.score DESC, q.amount DESC, q.case_id",
        "score": "q.score DESC, q.amount DESC, q.case_id",
        "claim_amount_usd": "q.amount DESC, q.case_id",
    }

    def _where(self, batch_id: str, f: dict) -> tuple[str, list]:
        sql = ["q.batch_id = ?"]
        args: list = [batch_id]

        def inlist(col, vals):
            vals = vals if isinstance(vals, list) else [vals]
            sql.append(f"{col} IN ({','.join('?' * len(vals))})")
            args.extend(vals)
        for key, col in (("priority_level", "q.priority_level"), ("care_type", "q.care_type"), ("state", "q.state"),
                         ("confidence", "q.confidence"), ("status", "q.status"),
                         ("case_ids", "q.case_id")):
            if f.get(key):
                inlist(col, f[key])
        for key, op, col in (("min_amount", ">=", "q.amount"), ("max_amount", "<=", "q.amount"),
                             ("min_score", ">=", "q.score"), ("max_score", "<=", "q.score")):
            if f.get(key) is not None:
                sql.append(f"{col} {op} ?")
                args.append(float(f[key]))
        for sig in (f.get("has_signal") or []) if isinstance(f.get("has_signal"), list) else [f["has_signal"]] if f.get("has_signal") else []:
            sql.append("EXISTS (SELECT 1 FROM queue_signals g WHERE g.case_id = q.case_id AND g.signal = ?)")
            args.append(sig)
        if f.get("data_quality_warning"):
            sql.append("q.dq = 1")
        if f.get("reviewed") is True:
            sql.append("q.status <> 'Unreviewed'")
        elif f.get("reviewed") is False:
            sql.append("q.status = 'Unreviewed'")
        if f.get("search"):
            sql.append("(q.case_id LIKE ? OR q.claim_number LIKE ?)")
            term = f"%{str(f['search']).strip().upper()}%"
            args += [term, term]
        return " AND ".join(sql), args

    FROM = FROM_FAST = "FROM queue_index q"

    @staticmethod
    def _needs_status(f: dict) -> bool:
        return bool(f.get("status")) or f.get("reviewed") in (True, False)

    def queue_rows(self, batch_id: str, f: dict, *, limit: int | None = None, offset: int = 0,
                   order: str = "queue", columns: str = "q.*") -> list[dict]:
        where, args = self._where(batch_id, f)
        q = f"SELECT {columns} {self.FROM} WHERE {where} ORDER BY {self.ORDERS.get(order, self.ORDERS['queue'])}"
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            args = args + [int(limit), int(offset)]
        with self.conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    def queue_count(self, batch_id: str, f: dict) -> int:
        where, args = self._where(batch_id, f)
        frm = self.FROM if self._needs_status(f) else self.FROM_FAST
        with self.conn() as c:
            return c.execute(f"SELECT COUNT(*) {frm} WHERE {where}", args).fetchone()[0]

    def queue_aggregate(self, batch_id: str, f: dict, group_by: str) -> list[dict]:
        where, args = self._where(batch_id, f)
        col = {"priority_level": "q.priority_level", "care_type": "q.care_type",
               "status": "q.status"}[group_by]
        frm = self.FROM if (group_by == "status" or self._needs_status(f)) else self.FROM_FAST
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                f"SELECT {col} AS k, COUNT(*) AS n, SUM(q.amount) AS amount, SUM(q.dq) AS dq "
                f"{frm} WHERE {where} GROUP BY {col} ORDER BY {col}", args)]

    def signal_counts(self, batch_id: str, f: dict) -> list[tuple[str, int]]:
        where, args = self._where(batch_id, f)
        frm = self.FROM if self._needs_status(f) else self.FROM_FAST
        with self.conn() as c:
            return [(r[0], r[1]) for r in c.execute(
                f"SELECT g.signal, COUNT(*) FROM queue_signals g WHERE g.case_id IN "
                f"(SELECT q.case_id {frm} WHERE {where}) GROUP BY g.signal ORDER BY 2 DESC", args)]

    def signal_pairs(self, batch_id: str, f: dict, top: int) -> list[tuple[str, str, int]]:
        where, args = self._where(batch_id, f)
        frm = self.FROM if self._needs_status(f) else self.FROM_FAST
        with self.conn() as c:
            return [(r[0], r[1], r[2]) for r in c.execute(
                f"SELECT a.signal, b.signal, COUNT(*) FROM queue_signals a JOIN queue_signals b "
                f"ON a.case_id = b.case_id AND a.signal < b.signal WHERE a.case_id IN "
                f"(SELECT q.case_id {frm} WHERE {where}) GROUP BY 1, 2 ORDER BY 3 DESC LIMIT ?", args + [top])]

    def indexed_count(self, batch_id: str) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM queue_index WHERE batch_id = ?", (batch_id,)).fetchone()[0]

    def adopt_batch(self, batch_id: str) -> None:
        """Cases re-imported unchanged keep their cached assessment; move their queue rows to the new batch."""
        with self.conn() as c:
            c.execute("UPDATE queue_index SET batch_id = ? WHERE case_id IN "
                      "(SELECT case_id FROM cases WHERE batch_id = ?)", (batch_id, batch_id))

    def assessment_count(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM assessments WHERE is_current=1").fetchone()[0]

    def update_narrative(self, assessment_id, narrative: dict, llm_model: str):
        with self.conn() as c:
            c.execute("UPDATE assessments SET narrative_json=?, llm_model=? WHERE assessment_id=?",
                      (json.dumps(narrative), llm_model, assessment_id))

    # ---------------------------------------------------------- workflow
    def get_status(self, case_id) -> str:
        with self.conn() as c:
            r = c.execute("SELECT status FROM case_state WHERE case_id=?", (case_id,)).fetchone()
        return r["status"] if r else "Unreviewed"

    def all_statuses(self) -> dict[str, str]:
        with self.conn() as c:
            return {r["case_id"]: r["status"] for r in c.execute("SELECT case_id, status FROM case_state")}

    def set_status(self, case_id, status, reason=""):
        old = self.get_status(case_id)
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO case_state VALUES (?,?,?)", (case_id, status, time.time()))
            c.execute("UPDATE queue_index SET status = ? WHERE case_id = ?", (status, case_id))
            c.execute("INSERT INTO case_status_history (case_id, from_status, to_status, reason, created_at) "
                      "VALUES (?,?,?,?,?)", (case_id, old, status, reason, time.time()))

    def add_decision(self, case_id, assessment_id, action, *, finding_id=None, reason="", note="",
                     ruleset_id="", model_version=""):
        with self.conn() as c:
            c.execute("INSERT INTO investigator_decisions (case_id, assessment_id, action, finding_id, reason, note, "
                      "ruleset_id, model_version, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                      (case_id, assessment_id, action, finding_id, reason, note, ruleset_id, model_version, time.time()))

    def finding_decisions(self, case_id) -> dict[str, dict]:
        """Latest accept/reject per finding."""
        with self.conn() as c:
            rows = c.execute("SELECT finding_id, action, reason FROM investigator_decisions WHERE case_id=? "
                             "AND action IN ('accept_finding','reject_finding','reset_finding') ORDER BY id",
                             (case_id,)).fetchall()
        out = {}
        for r in rows:
            if r["action"] == "reset_finding":
                out.pop(r["finding_id"], None)
            else:
                out[r["finding_id"]] = {"action": r["action"], "reason": r["reason"]}
        return out

    def decisions(self, case_id=None) -> list[dict]:
        q, args = "SELECT * FROM investigator_decisions", ()
        if case_id:
            q, args = q + " WHERE case_id=?", (case_id,)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id", args)]

    def add_note(self, case_id, note, author="investigator"):
        with self.conn() as c:
            c.execute("INSERT INTO case_notes (case_id, note, author, created_at) VALUES (?,?,?,?)",
                      (case_id, note, author, time.time()))

    def notes(self, case_id) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM case_notes WHERE case_id=? ORDER BY id", (case_id,))]

    def status_history(self, case_id) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM case_status_history WHERE case_id=? ORDER BY id", (case_id,))]

    def audit_trail(self, case_id) -> list[dict]:
        events = []
        for a in self.assessment_history(case_id):
            events.append({"ts": a["created_at"], "kind": "assessment",
                           "detail": f"v{a['version']} ({a['trigger']}): {a['priority_level']} score {a['score']}, "
                                     f"confidence {a['confidence']}, {a['ruleset_id']} / {a['llm_model']}"})
        for d in self.decisions(case_id):
            txt = d["action"].replace("_", " ")
            if d["finding_id"]:
                txt += f" [{d['finding_id']}]"
            if d["reason"]:
                txt += f" reason: {d['reason']}"
            events.append({"ts": d["created_at"], "kind": "decision", "detail": txt})
        for n in self.notes(case_id):
            events.append({"ts": n["created_at"], "kind": "note", "detail": n["note"]})
        for s in self.status_history(case_id):
            events.append({"ts": s["created_at"], "kind": "status",
                           "detail": f"{s['from_status']} -> {s['to_status']}" + (f" ({s['reason']})" if s["reason"] else "")})
        return sorted(events, key=lambda e: e["ts"])

    # ------------------------------------------------------- telemetry
    def log_agent_run(self, run: dict, tool_calls: list[dict]):
        with self.conn() as c:
            c.execute("INSERT INTO agent_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                run["run_id"], run["agent"], run.get("case_id"), run.get("provider"), run.get("model"),
                run.get("question"), run.get("answer"), run.get("input_tokens", 0), run.get("output_tokens", 0),
                run.get("latency_ms", 0), len(tool_calls), int(run.get("grounding_passed", True)),
                json.dumps(run.get("grounding_issues", [])), int(run.get("fallback_used", False)),
                run.get("error"), time.time()))
            for t in tool_calls:
                c.execute("INSERT INTO agent_tool_calls (run_id, tool, args_json, result_preview, created_at) "
                          "VALUES (?,?,?,?,?)", (run["run_id"], t["tool"], json.dumps(t.get("args", {})),
                                                 str(t.get("result", ""))[:500], time.time()))

    def agent_runs(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM agent_runs ORDER BY created_at")]

    def save_evaluation(self, model_version, payload):
        with self.conn() as c:
            c.execute("INSERT INTO model_evaluations (model_version, payload_json, created_at) VALUES (?,?,?)",
                      (model_version, json.dumps(payload), time.time()))

    def latest_evaluation(self, kind: str = "synthetic-comparison"):
        with self.conn() as c:
            r = c.execute("SELECT * FROM model_evaluations WHERE model_version=? ORDER BY id DESC LIMIT 1",
                          (kind,)).fetchone()
        return json.loads(r["payload_json"]) if r else None

    def add_knowledge(self, kind, case_id, title, body):
        with self.conn() as c:
            c.execute("INSERT INTO knowledge_documents (kind, case_id, title, body, created_at) VALUES (?,?,?,?,?)",
                      (kind, case_id, title, body, time.time()))

    def knowledge(self, query: str = "", limit: int = 20) -> list[dict]:
        with self.conn() as c:
            if query:
                rows = c.execute("SELECT * FROM knowledge_documents WHERE title LIKE ? OR body LIKE ? "
                                 "ORDER BY id DESC LIMIT ?", (f"%{query}%", f"%{query}%", limit))
            else:
                rows = c.execute("SELECT * FROM knowledge_documents ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    def reset(self):
        Path(self.path).unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(self.path + suffix).unlink(missing_ok=True)
        self.__init__(self.path)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def case_hash(case: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(case, sort_keys=True, default=str).encode()).hexdigest()[:16]
