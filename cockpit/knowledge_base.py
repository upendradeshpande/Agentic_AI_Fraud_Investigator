"""Knowledge layer (spec 14): investigation playbooks retrieved by section.

Playbooks live in cockpit/knowledge/*.md as versioned files (reviewed like code). They hold
explanations and procedures, never scoring rules: the rule engine executes exact formulas
from the ruleset JSON. Retrieval is lexical (BM25) plus routing from a case's active
findings, which is transparent and sufficient for a small corpus. The interface
`search(query, signals)` is what a vector store would implement later.

All content is SYNTHETIC DEMONSTRATION GUIDANCE, not an insurer's approved procedure.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
STOP = set("the a an and or of to for in on with is are be as by it this that at from not no can when than".split())
DISCLAIMER = "Synthetic demonstration guidance, not approved policy."


@dataclass(frozen=True)
class Section:
    doc_id: str
    title: str
    version: str
    heading: str
    text: str
    signals: tuple[str, ...]

    @property
    def citation(self) -> str:
        return f"{self.doc_id} v{self.version} · {self.heading}"

    def to_dict(self, score: float | None = None) -> dict:
        d = {"doc_id": self.doc_id, "title": self.title, "section": self.heading, "text": self.text,
             "citation": self.citation, "note": DISCLAIMER}
        if score is not None:
            d["score"] = round(score, 2)
        return d


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in STOP]


def _parse(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    if raw.startswith("---"):
        _, head, body = raw.split("---", 2)
        for line in head.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    signals = tuple(s.strip() for s in meta.get("signals", "").split(",") if s.strip())
    out = []
    for block in re.split(r"^## ", body, flags=re.M)[1:]:
        heading, _, text = block.partition("\n")
        out.append(Section(meta.get("doc_id", path.stem), meta.get("title", path.stem), meta.get("version", "1.0"),
                           heading.strip(), " ".join(text.split()), signals))
    return out


@lru_cache(maxsize=1)
def sections() -> tuple[Section, ...]:
    return tuple(s for p in sorted(KNOWLEDGE_DIR.glob("*.md")) for s in _parse(p))


@lru_cache(maxsize=1)
def _index():
    docs = [_tokens(f"{s.title} {s.heading} {s.text}") for s in sections()]
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    avg = sum(len(d) for d in docs) / max(1, len(docs))
    return docs, df, avg


def search(query: str = "", signals: list[str] | tuple[str, ...] = (), limit: int = 3) -> list[dict]:
    """BM25 on the query plus a boost for sections whose playbook covers the given signals."""
    docs, df, avg = _index()
    n = len(docs)
    q = _tokens(query)
    wanted = set(signals)
    scored = []
    for sec, toks in zip(sections(), docs):
        score = 0.0
        for t in q:
            tf = toks.count(t)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(toks) / avg))
        overlap = len(wanted & set(sec.signals))
        if overlap:
            # Specific playbooks (few signals) beat generic ones that list many signals.
            score += 3.0 * overlap / math.sqrt(len(sec.signals))
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda t: -t[0])
    # Diversify: best section of each playbook first, then fill with remaining sections.
    out, docs_used, picked = [], set(), set()
    for pass_ in (1, 2):
        for score, sec in scored:
            if len(out) >= limit:
                return out
            key = (sec.doc_id, sec.heading)
            if key in picked or (pass_ == 1 and sec.doc_id in docs_used):
                continue
            picked.add(key)
            docs_used.add(sec.doc_id)
            out.append(sec.to_dict(score))
    return out


def for_assessment(assessment, limit: int = 3) -> list[dict]:
    """Guidance routed by the case's active findings (and data-quality warnings)."""
    active = sorted((f for f in assessment.findings if f.triggered and f.points > 0 and f.status != "rejected"),
                    key=lambda f: -f.points)[:3]
    signals = [f.finding_id for f in active]
    if assessment.data_quality_warnings:
        signals.insert(0, "data_quality")
    query = " ".join(f.label for f in active)
    if not signals:
        query = "false positive alternative explanations disposition"
    return search(query, signals, limit)
