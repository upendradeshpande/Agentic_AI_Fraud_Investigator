"""Application service. The UI and the agents both go through this layer, so every
number shown anywhere comes from the same deterministic code path."""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from typing import Optional

import numpy as np

from . import peer
from .config import settings
from .data_loader import ValidationReport, file_hash, load_file, records
from .db import Repository, case_hash
from .llm.providers import describe, llm_available
from .models.anomaly_model import AnomalyDetectionModel
from .models.rules_model import TransparentRulesModel
from .schemas import (PRIORITY_LABELS, BINARY_SIGNALS, CASE_STATUSES, CONTINUOUS_SIGNALS, FINAL_STATUSES, REJECT_REASONS,
                      SIGNAL_COLUMNS, Assessment)

SIGNAL_LABELS = peer.LABELS


class CockpitService:
    def __init__(self, repo: Optional[Repository] = None):
        self.repo = repo or Repository()
        self.rules = TransparentRulesModel()
        self.repo.save_ruleset(self.rules.ruleset)
        self.cases: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.report: Optional[ValidationReport] = None
        self.batch: Optional[dict] = None
        batch = self.repo.latest_batch()
        if batch:
            self._activate(batch)

    # --------------------------------------------------------------- import
    @property
    def loaded(self) -> bool:
        return bool(self.cases)

    def import_file(self, source, filename: str) -> ValidationReport:
        df, rep = load_file(source, filename)
        if rep.blocking_errors:
            return rep
        cases = records(df)
        batch_id = self.repo.save_batch(filename, file_hash(df), cases, rep.to_dict())
        self._activate(self.repo.latest_batch())
        self.assess_all(trigger="import")
        return rep

    def ensure_loaded(self) -> None:
        if not self.loaded:
            with open(settings.sample_csv, "rb") as f:
                self.import_file(f, settings.sample_csv.name)

    ANOMALY_FIT_SAMPLE = 20_000

    def _activate(self, batch: dict) -> None:
        self.batch = batch
        self.cases = self.repo.load_cases(batch["batch_id"])
        self.by_id = {c["case_id"]: c for c in self.cases}
        rep_dict = json.loads(batch["report_json"])
        rep_dict.pop("status", None)
        self.report = ValidationReport(**rep_dict)
        self.rules.fit(self.cases)
        # Large queues: fit the anomaly models on a fixed random sample, score everything.
        fit_set = self.cases
        if len(self.cases) > self.ANOMALY_FIT_SAMPLE:
            idx = np.random.default_rng(42).choice(len(self.cases), self.ANOMALY_FIT_SAMPLE, replace=False)
            fit_set = [self.cases[i] for i in idx]
        self.anomaly_model = AnomalyDetectionModel().fit(fit_set)
        self.anomaly = self.anomaly_model.analyze(self.cases)
        self.cohort_sizes = Counter(c["care_type"] for c in self.cases)
        self.cohorts = peer.CohortCache(self.cases)
        self._sim_index = None
        self.repo.adopt_batch(batch["batch_id"])
        if self.repo.indexed_count(batch["batch_id"]) < len(self.cases):
            self.rebuild_queue_index()
        # Rules or model version changed since these assessments were made: re-score (cached cases are skipped).
        self.assess_all(trigger=f"rules/model update to {self.rules.ruleset_id} / {self.rules.model_version}")

    def rebuild_queue_index(self) -> None:
        """Rebuild the SQL queue index from current assessments (also upgrades databases from older versions)."""
        rows = self.repo.all_current_assessments()
        batch = []
        for cid, r in rows.items():
            if cid in self.by_id:
                batch.append(self._queue_row(self.by_id[cid], Assessment.from_dict(json.loads(r["payload_json"]))))
            if len(batch) >= 5000:
                with self.repo.conn() as c:
                    self.repo._upsert_queue(c, batch)
                batch = []
        if batch:
            with self.repo.conn() as c:
                self.repo._upsert_queue(c, batch)

    def _queue_row(self, case: dict, a: Assessment) -> dict:
        return {
            "case_id": case["case_id"], "batch_id": self.batch["batch_id"], "order_group": self.order_key(a, 0)[0],
            "score": a.score,
            "amount": float(case["claim_amount_usd"] or 0), "priority_level": a.priority_level,
            "confidence": a.confidence, "care_type": case["care_type"], "state": case["state"],
            "claim_number": case["claim_number"], "claim_date": case["claim_date"],
            "strong": bool(a.strong_indicators), "dq": bool(a.data_quality_warnings),
            "anomalous": bool(a.anomaly.get("is_anomalous")),
            "top_evidence": ", ".join(f.label for f in a.key_evidence[:3]),
            "triggered": [f.finding_id for f in a.key_evidence if f.kind != "combination"],
        }

    # ------------------------------------------------------------ assessment
    def _compute(self, case_id: str) -> Assessment:
        case = self.by_id[case_id]
        decisions = self.repo.finding_decisions(case_id)
        excluded = {fid for fid, d in decisions.items() if d["action"] == "reject_finding"}
        reasons = {fid: d["reason"] for fid, d in decisions.items() if d["action"] == "reject_finding"}
        a = self.rules.assess(
            case, excluded=excluded, reject_reasons=reasons,
            dq_warnings=self.report.case_warnings(case_id, case["claim_number"]),
            anomaly=self.anomaly.get(case_id), cohort_size=self.cohort_sizes[case["care_type"]],
        )
        for f in a.findings:
            d = decisions.get(f.finding_id)
            if d and d["action"] == "accept_finding":
                f.status = "accepted"
        return a

    def assess_all(self, trigger: str = "import", chunk: int = 2000, progress=None) -> int:
        """Assess every case in chunks with bulk writes. Reuses cached assessments when case data,
        ruleset and model version are unchanged (spec 17). Returns the number of cases assessed."""
        from .agents.assessment_agent import deterministic_narrative
        current = self.repo.current_hashes()
        todo = [c for c in self.cases
                if current.get(c["case_id"]) != (case_hash(c), self.rules.ruleset_id, self.rules.model_version)]
        for i in range(0, len(todo), chunk):
            items = []
            for case in todo[i:i + chunk]:
                a = self._compute(case["case_id"])
                items.append((a.to_dict(), deterministic_narrative(a, case), case_hash(case), self._queue_row(case, a)))
            self.repo.save_assessments_bulk(items, trigger=trigger, llm_model="deterministic")
            if progress:
                progress(min(i + chunk, len(todo)), len(todo))
        return len(todo)

    def recompute(self, case_id: str, trigger: str) -> dict:
        from .agents.assessment_agent import AssessmentAgent
        a = self._compute(case_id)
        aid = self.repo.save_assessment(a.to_dict(), None, trigger=trigger, llm_model="pending",
                                        case_hash_value=case_hash(self.by_id[case_id]),
                                        queue_row=self._queue_row(self.by_id[case_id], a))
        narrative, model = AssessmentAgent(self).run(case_id, a)
        self.repo.update_narrative(aid, narrative, model)
        return self.get_assessment(case_id)

    def get_assessment(self, case_id: str) -> dict:
        row = self.repo.current_assessment(case_id)
        if row is None:
            a = self._compute(case_id)
            from .agents.assessment_agent import deterministic_narrative
            self.repo.save_assessment(a.to_dict(), deterministic_narrative(a, self.by_id[case_id]),
                                      trigger="lazy", llm_model="deterministic",
                                      case_hash_value=case_hash(self.by_id[case_id]),
                                      queue_row=self._queue_row(self.by_id[case_id], a))
            row = self.repo.current_assessment(case_id)
        return {
            "assessment_id": row["assessment_id"],
            "version": row["version"],
            "assessment": Assessment.from_dict(json.loads(row["payload_json"])),
            "narrative": json.loads(row["narrative_json"]) if row["narrative_json"] else None,
            "llm_model": row["llm_model"],
            "created_at": row["created_at"],
        }

    def ensure_ai_narrative(self, case_id: str) -> dict:
        """Generate the LLM narrative on first open if it is still the deterministic one."""
        from .agents.assessment_agent import AssessmentAgent
        cur = self.get_assessment(case_id)
        if cur["llm_model"] in ("deterministic", "pending") and llm_available():
            narrative, model = AssessmentAgent(self).run(case_id, cur["assessment"])
            self.repo.update_narrative(cur["assessment_id"], narrative, model)
            cur = self.get_assessment(case_id)
        return cur

    def all_assessments(self) -> dict[str, Assessment]:
        rows = self.repo.all_current_assessments()
        return {cid: Assessment.from_dict(json.loads(r["payload_json"])) for cid, r in rows.items()
                if cid in self.by_id}

    # --------------------------------------------------------------- queue
    @staticmethod
    def order_key(a: Assessment, amount: float):
        if a.strong_indicators:
            group = 0
        elif a.priority_level == "red":
            group = 1
        elif a.priority_level == "data_review":
            group = 2
        elif a.priority_level == "yellow" and a.confidence != "High":
            group = 3
        elif a.priority_level == "yellow":
            group = 4
        else:
            group = 5
        return (group, -a.score, -amount)

    # All queue reads go through SQL: filtering, ordering and paging happen in the database.
    def _row(self, r: dict, rank: int) -> dict:
        return {
            "rank": rank, "case_id": r["case_id"], "claim_number": r["claim_number"], "care_type": r["care_type"],
            "state": r["state"], "claim_date": r["claim_date"], "claim_amount_usd": int(r["amount"]),
            "score": r["score"], "priority_level": r["priority_level"],
            "priority_label": PRIORITY_LABELS[r["priority_level"]], "confidence": r["confidence"],
            "strong_indicator": bool(r["strong"]), "data_quality_warning": bool(r["dq"]),
            "anomalous": bool(r["anomalous"]), "top_evidence": r["top_evidence"], "status": r["status"],
            "triggered_signals": [x for x in (r["triggered"] or "").split(",") if x],
        }

    def queue(self, filters: Optional[dict] = None) -> list[dict]:
        """Matching cases in queue order. Honour filters['limit'] for large queues."""
        f = dict(filters or {})
        limit = f.pop("limit", None)
        order = f.pop("sort", None) or "queue"
        rows = self.repo.queue_rows(self.batch["batch_id"], f, limit=int(limit) if limit else None, order=order)
        return [self._row(r, i) for i, r in enumerate(rows, 1)]

    def queue_page(self, filters: Optional[dict] = None, page: int = 1, page_size: int = 50) -> dict:
        """One page of the queue plus the total count. Only page_size rows leave the database."""
        f = dict(filters or {})
        f.pop("limit", None)
        order = f.pop("sort", None) or "queue"
        total = self.repo.queue_count(self.batch["batch_id"], f)
        pages = max(1, -(-total // page_size))
        page = min(max(1, int(page)), pages)
        offset = (page - 1) * page_size
        rows = self.repo.queue_rows(self.batch["batch_id"], f, limit=page_size, offset=offset, order=order)
        return {"rows": [self._row(r, offset + i) for i, r in enumerate(rows, 1)], "total": total,
                "page": page, "pages": pages, "page_size": page_size, "first": offset + 1 if total else 0,
                "last": offset + len(rows)}

    def count(self, filters: Optional[dict] = None) -> int:
        return self.repo.queue_count(self.batch["batch_id"], dict(filters or {}))

    def queue_summary(self, filters: Optional[dict] = None) -> dict:
        b, f = self.batch["batch_id"], dict(filters or {})
        lv = {r["k"]: r for r in self.repo.queue_aggregate(b, f, "priority_level")}
        st = {r["k"]: r["n"] for r in self.repo.queue_aggregate(b, f, "status")}
        total = sum(r["n"] for r in lv.values())
        unreviewed = st.get("Unreviewed", 0)
        green_open = self.repo.queue_count(b, {**f, "priority_level": ["green"],
                                               "status": [s for s in CASE_STATUSES if s not in FINAL_STATUSES]})
        n = lambda k: lv[k]["n"] if k in lv else 0
        return {
            "total_cases": total, "high_priority": n("red"), "needs_attention": n("yellow"),
            "lower_priority": n("green"), "data_review": n("data_review"),
            "lower_priority_awaiting_approval": green_open,
            "reviewed": total - unreviewed, "unreviewed": unreviewed,
            "data_quality_warnings": int(sum(r["dq"] or 0 for r in lv.values())),
            "total_claim_amount_usd": int(sum(r["amount"] or 0 for r in lv.values())),
            "amount_in_high_priority_usd": int(lv["red"]["amount"]) if "red" in lv else 0,
            "status_counts": st,
        }

    # -------------------------------------------------------------- charts
    TOP_N_CHART = 50

    def chart_data(self, chart_id: str, filters: Optional[dict] = None) -> dict:
        b, f = self.batch["batch_id"], dict(filters or {})
        if chart_id == "cases_by_care_type":
            return {"chart_id": chart_id, "data": [{"care_type": r["k"], "cases": r["n"], "claim_amount_usd": int(r["amount"] or 0)}
                                                   for r in self.repo.queue_aggregate(b, f, "care_type")]}
        if chart_id == "exposure_by_priority":
            agg = {r["k"]: r for r in self.repo.queue_aggregate(b, f, "priority_level")}
            return {"chart_id": chart_id, "note": "Total claim amount per priority level (exposure, not evidence).",
                    "data": [{"priority_level": lvl, "priority_label": PRIORITY_LABELS[lvl], "cases": agg[lvl]["n"],
                              "claim_amount_usd": int(agg[lvl]["amount"] or 0)}
                             for lvl in ("red", "data_review", "yellow", "green") if lvl in agg]}
        if chart_id == "rules_vs_anomaly":
            rows = self.queue({**f, "limit": 2000})
            data = []
            for r in rows:
                an = self.anomaly.get(r["case_id"], {})
                d = self.disagreement(r["case_id"], r["priority_level"], an)
                data.append({"case_id": r["case_id"], "score": r["score"], "priority_level": r["priority_level"],
                             "anomaly_percentile": an.get("isolation_forest_percentile"),
                             "anomaly_votes": an.get("votes"), "disagreement": d["kind"] if d else None})
            return {"chart_id": chart_id, "data": data, "anomaly_top_pct": self.ANOMALY_TOP,
                    "note": "x = rules score, y = seed-averaged Isolation Forest percentile (unsupervised). "
                            "Disagreement = one method rates the case high and the other does not."}
        if chart_id == "exposure_by_case":
            rows = self.queue({**f, "limit": self.TOP_N_CHART})
            total = self.count(f)
            return {"chart_id": chart_id, "total_matching": total, "shown": len(rows),
                    "note": f"First {len(rows)} of {total} cases in queue order; height is claim amount (exposure, not evidence).",
                    "data": [{"rank": r["rank"], "case_id": r["case_id"], "claim_amount_usd": r["claim_amount_usd"],
                              "score": r["score"], "priority_level": r["priority_level"]} for r in rows]}
        if chart_id == "priority_by_care_type":
            with self.repo.conn() as c:
                where, args = self.repo._where(b, f)
                res = c.execute(f"SELECT q.care_type, q.priority_level, COUNT(*) {self.repo.FROM} WHERE {where} "
                                "GROUP BY 1, 2", args).fetchall()
            table: dict = {}
            for care, lvl, n in res:
                table.setdefault(care, {})[lvl] = n
            return {"chart_id": chart_id, "data": dict(sorted(table.items()))}
        if chart_id == "amount_vs_score":
            rows = self.queue({**f, "limit": 2000})
            return {"chart_id": chart_id, "note": "At most 2,000 cases in queue order.", "data": [
                {"case_id": r["case_id"], "claim_amount_usd": r["claim_amount_usd"], "score": r["score"],
                 "priority_level": r["priority_level"], "care_type": r["care_type"]} for r in rows]}
        if chart_id == "signal_frequency":
            cnt = self.repo.signal_counts(b, f)
            return {"chart_id": chart_id, "data": {SIGNAL_LABELS.get(k, k): v for k, v in cnt},
                    "signal_ids": {SIGNAL_LABELS.get(k, k): k for k, _ in cnt}}
        if chart_id == "signal_cooccurrence":
            return {"chart_id": chart_id, "data": self.cooccurrence(filters)}
        raise ValueError(f"Unknown chart_id {chart_id}")

    def cooccurrence(self, filters: Optional[dict] = None, top: int = 10) -> list[dict]:
        return [{"signals": [SIGNAL_LABELS.get(a, a), SIGNAL_LABELS.get(b2, b2)], "signal_ids": [a, b2], "cases": n}
                for a, b2, n in self.repo.signal_pairs(self.batch["batch_id"], dict(filters or {}), top)]

    def score_values(self, filters: Optional[dict] = None, limit: int = 200_000) -> list[tuple[float, str]]:
        rows = self.repo.queue_rows(self.batch["batch_id"], dict(filters or {}), limit=limit,
                                    columns="q.score AS score, q.priority_level AS priority_level")
        return [(r["score"], r["priority_level"]) for r in rows]

    def signal_distribution(self, signal: str, filters: Optional[dict] = None) -> dict:
        if signal not in SIGNAL_COLUMNS + ["claim_amount_usd"]:
            raise ValueError(f"Unknown signal {signal}")
        f = dict(filters or {})
        if f:
            ids = [r["case_id"] for r in self.repo.queue_rows(self.batch["batch_id"], f, columns="q.case_id AS case_id")]
            vals = np.array([float(self.by_id[i][signal]) for i in ids if self.by_id[i].get(signal) is not None])
        else:
            vals = self.cohorts.column(signal)
        if len(vals) == 0:
            return {"signal": signal, "n": 0}
        out = {"signal": signal, "label": SIGNAL_LABELS.get(signal, signal), "n": int(len(vals)),
               "min": float(vals.min()), "median": float(np.median(vals)),
               "p90": float(np.percentile(vals, 90)), "max": float(vals.max()), "mean": round(float(vals.mean()), 3)}
        if signal in BINARY_SIGNALS:
            out["share_flagged"] = round(float(vals.mean()), 3)
        return out

    # ------------------------------------------------ rules vs unsupervised
    ANOMALY_TOP = 85.0   # "unusual" = top 15% of the queue by Isolation Forest (matches the model's contamination)

    def disagreement(self, case_id: str, level: Optional[str] = None, an: Optional[dict] = None) -> Optional[dict]:
        """Where the unsupervised model and the rules disagree. Explains why in plain language."""
        an = an if an is not None else self.anomaly.get(case_id, {})
        if level is None:
            level = self.get_assessment(case_id)["assessment"].priority_level
        pct = an.get("isolation_forest_percentile")
        if pct is None:
            return None
        unusual = pct >= self.ANOMALY_TOP or an.get("is_anomalous")
        if unusual and level in ("green", "yellow", "data_review"):
            a = self.get_assessment(case_id)["assessment"]
            benign = any("benign direction" in r for r in a.confidence_reasons)
            why = ("The case is unusual mainly in a benign direction (for example an unusually small or quiet claim), "
                   "which the rules do not treat as a fraud indicator." if benign or level == "green" and a.score == 0 else
                   "It is unusual across several signals, but none of the known high-weight fraud indicators "
                   "(duplicate billing, overlap with shared contact) is present, so the rules score stays below the "
                   "high-priority threshold.")
            return {"kind": "unsupervised_flags", "percentile": pct, "level": level, "score": a.score,
                    "message": f"Second opinion disagrees: the unsupervised model (Isolation Forest, averaged over "
                               f"{an.get('isolation_forest_seeds', 1)} forests) ranks {case_id} in the top "
                               f"{max(1, round(100 - pct))}% most unusual cases, but the rules rate it "
                               f"{PRIORITY_LABELS[level].lower()} (score {a.score:g}). {why}",
                    "action": "Open the anomaly cross-check and decide whether the pattern needs a closer look; "
                              "the investigator's decision is recorded as feedback."}
        if level == "red" and pct < 50 and not an.get("is_anomalous"):
            a = self.get_assessment(case_id)["assessment"]
            return {"kind": "rules_flag", "percentile": pct, "level": level, "score": a.score,
                    "message": f"The rules rate {case_id} high priority (score {a.score:g}) from known fraud indicators, "
                               f"but the unsupervised model does not find it unusual (percentile {pct:g}). This happens "
                               "when a pattern is common in the queue, so it no longer looks rare.",
                    "action": "Keep the rules priority: known indicators matter even when they are common."}
        return None

    def disagreements(self, filters: Optional[dict] = None) -> list[dict]:
        out = []
        for r in self.queue({**(filters or {}), "limit": 2000}):
            d = self.disagreement(r["case_id"], r["priority_level"])
            if d:
                out.append({"case_id": r["case_id"], **d})
        return out

    # ------------------------------------------------------------- similar
    def similar_cases(self, case_id: str, limit: int = 5) -> list[dict]:
        if self._sim_index is None:
            self._sim_index = peer.SimilarityIndex(self.cases)
        hits = self._sim_index.nearest(case_id, limit)
        info = {r["case_id"]: r for r in self.queue({"case_ids": [h["case_id"] for h in hits]})} if hits else {}
        for h in hits:
            r = info.get(h["case_id"], {})
            h.update({"priority_level": r.get("priority_level"), "score": r.get("score"),
                      "investigator_status": r.get("status")})
        return hits

    def compare_to_cohort(self, case_id: str, by: str = "care_type") -> dict:
        if by == "care_type":
            return self.cohorts.compare(self.by_id[case_id])
        return peer.compare_to_cohort(self.cases, self.by_id[case_id], by)

    def what_if(self, case_id: str, exclude: list[str]) -> dict:
        """Recalculate without persisting. Used by the copilot to answer 'what if' questions."""
        case = self.by_id[case_id]
        a = self.rules.assess(case, excluded=set(exclude),
                              dq_warnings=self.report.case_warnings(case_id, case["claim_number"]),
                              anomaly=self.anomaly.get(case_id), cohort_size=self.cohort_sizes[case["care_type"]])
        return {"case_id": case_id, "excluded": exclude, "score": a.score, "priority_level": a.priority_level,
                "confidence": a.confidence}

    # ------------------------------------------------------------ workflow
    def accept_finding(self, case_id, finding_id):
        cur = self.get_assessment(case_id)
        self.repo.add_decision(case_id, cur["assessment_id"], "accept_finding", finding_id=finding_id,
                               ruleset_id=self.rules.ruleset_id, model_version=self.rules.model_version)
        if self.repo.get_status(case_id) == "Unreviewed":
            self.repo.set_status(case_id, "In review", "Investigator started review")
        a = self._compute(case_id)
        # Accepting does not change the score, so update findings in place without a new LLM call.
        self.repo.save_assessment(a.to_dict(), cur["narrative"], trigger=f"accepted {finding_id}",
                                  llm_model=cur["llm_model"], case_hash_value=case_hash(self.by_id[case_id]),
                                  queue_row=self._queue_row(self.by_id[case_id], a))
        return self.get_assessment(case_id)

    def reject_finding(self, case_id, finding_id, reason: str, category: str = ""):
        if not reason.strip():
            raise ValueError("A reason is required to reject a finding.")
        if category:
            if category not in REJECT_REASONS:
                raise ValueError(f"category must be one of {REJECT_REASONS}")
            reason = f"[{category}] {reason.strip()}"
        cur = self.get_assessment(case_id)
        self.repo.add_decision(case_id, cur["assessment_id"], "reject_finding", finding_id=finding_id,
                               reason=reason, ruleset_id=self.rules.ruleset_id,
                               model_version=self.rules.model_version)
        self.repo.add_knowledge("rejected_finding", case_id, f"Rejected {finding_id} on {case_id}", reason)
        if self.repo.get_status(case_id) == "Unreviewed":
            self.repo.set_status(case_id, "In review", "Investigator started review")
        return self.recompute(case_id, trigger=f"rejected {finding_id}")

    def reset_finding(self, case_id, finding_id):
        cur = self.get_assessment(case_id)
        self.repo.add_decision(case_id, cur["assessment_id"], "reset_finding", finding_id=finding_id,
                               ruleset_id=self.rules.ruleset_id, model_version=self.rules.model_version)
        return self.recompute(case_id, trigger=f"reset {finding_id}")

    def restore_all_findings(self, case_id):
        cur = self.get_assessment(case_id)
        rejected = [fid for fid, d in self.repo.finding_decisions(case_id).items() if d["action"] == "reject_finding"]
        for fid in rejected:
            self.repo.add_decision(case_id, cur["assessment_id"], "reset_finding", finding_id=fid,
                                   ruleset_id=self.rules.ruleset_id, model_version=self.rules.model_version)
        return self.recompute(case_id, trigger=f"restored {len(rejected)} finding(s)")

    def add_note(self, case_id, note: str, author: str = "investigator"):
        if not note.strip():
            raise ValueError("Note is empty.")
        self.repo.add_note(case_id, note.strip(), author)
        cur = self.get_assessment(case_id)
        self.repo.add_decision(case_id, cur["assessment_id"], "add_note", note=note.strip(),
                               ruleset_id=self.rules.ruleset_id, model_version=self.rules.model_version)

    def set_status(self, case_id, status: str, reason: str = ""):
        if status not in CASE_STATUSES:
            raise ValueError(f"Status must be one of {CASE_STATUSES}")
        if status in FINAL_STATUSES and not reason.strip():
            raise ValueError("A reason is required for a final disposition.")
        cur = self.get_assessment(case_id)
        a = cur["assessment"]
        self.repo.set_status(case_id, status, reason)
        override = _is_override(a.priority_level, status)
        self.repo.add_decision(case_id, cur["assessment_id"], "set_status", reason=reason,
                               note=f"{status}; ai_level={a.priority_level}; override={override}",
                               ruleset_id=self.rules.ruleset_id, model_version=self.rules.model_version)
        if status in FINAL_STATUSES:
            self.repo.add_knowledge("disposition", case_id, f"{case_id}: {status}",
                                    f"AI level {a.priority_level}, score {a.score}. Reason: {reason}")

    # ---------------------------------------------------------- evaluation
    def label_summary(self) -> dict:
        """Real outcome labels collected so far: the latest final decision per case. These, not synthetic data,
        are what a future supervised model would be trained on."""
        latest = {}
        for d in self.repo.decisions():
            if d["action"] == "set_status":
                latest[d["case_id"]] = (d["note"] or "").split(";")[0]
        positive = sum(1 for s in latest.values() if s in ("Escalated", "Solved"))
        negative = sum(1 for s in latest.values() if s in ("Likely false positive", "Approved / closed"))
        return {"decided": positive + negative, "positive": positive, "negative": negative}

    def evaluation_summary(self) -> dict:
        decisions = self.repo.decisions()
        runs = self.repo.agent_runs()
        acc = [d for d in decisions if d["action"] == "accept_finding"]
        rej = [d for d in decisions if d["action"] == "reject_finding"]
        status = [d for d in decisions if d["action"] == "set_status"]
        overrides = [d for d in status if "override=True" in (d["note"] or "")]
        llm_runs = [r for r in runs if not r["fallback_used"]]
        in_tok = sum(r["input_tokens"] or 0 for r in runs)
        out_tok = sum(r["output_tokens"] or 0 for r in runs)
        cost = in_tok / 1e6 * settings.price_input_per_mtok + out_tok / 1e6 * settings.price_output_per_mtok
        grounded = [r for r in runs if r["agent"] in ("assessment", "copilot")]
        return {
            "assessments_completed": self.repo.assessment_count(),
            "accepted_findings": len(acc),
            "rejected_findings": len(rej),
            "human_overrides": len(overrides),
            "final_dispositions": len([d for d in status if any(s in (d["note"] or "") for s in FINAL_STATUSES)]),
            "most_rejected_signals": Counter(d["finding_id"] for d in rej).most_common(5),
            "rejection_reasons": Counter(_reason_category(d["reason"]) for d in rej).most_common(),
            "grounding_pass_rate": round(sum(r["grounding_passed"] for r in grounded) / len(grounded), 3) if grounded else None,
            "grounding_checked": len(grounded),
            "avg_latency_ms": int(np.mean([r["latency_ms"] for r in llm_runs])) if llm_runs else None,
            "llm_requests": len(llm_runs),
            "fallback_runs": len(runs) - len(llm_runs),
            "input_tokens": in_tok, "output_tokens": out_tok, "estimated_cost_usd": round(cost, 4),
            "ruleset_id": self.rules.ruleset_id, "model_version": self.rules.model_version,
            "llm": describe(),
        }


def _reason_category(reason: str) -> str:
    if reason and reason.startswith("[") and "]" in reason:
        return reason[1:reason.index("]")]
    return "Uncategorised"


def _is_override(ai_level: str, status: str) -> bool:
    if status in ("Escalated",) and ai_level == "green":
        return True
    if status in ("Likely false positive", "Approved / closed") and ai_level == "red":
        return True
    return False
