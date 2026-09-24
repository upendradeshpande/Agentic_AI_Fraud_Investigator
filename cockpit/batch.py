"""Overnight job: import the referral file, assess every case, write AI narratives and
so the queue is ready when investigators log in.

    python -m cockpit.batch [--file path.csv] [--no-llm]
"""
from __future__ import annotations

import argparse
import time

from .agents.assessment_agent import batch_generate
from .config import settings
from .llm.providers import describe, llm_available
from .service import CockpitService


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", help="CSV or Excel file to import (default: supplied sample)")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM narratives")
    args = ap.parse_args()

    t0 = time.time()
    svc = CockpitService()
    path = args.file or str(settings.sample_csv)
    with open(path, "rb") as fh:
        rep = svc.import_file(fh, path)
    if rep.blocking_errors:
        raise SystemExit("Import blocked: " + "; ".join(rep.blocking_errors))
    print(f"Imported {rep.rows_loaded} rows ({rep.status}).")
    s = svc.queue_summary()
    print(f"Assessed: {s['high_priority']} high priority, {s['needs_attention']} needs attention, "
          f"{s['lower_priority']} lower priority, {s['data_review']} data review.")
    if llm_available() and not args.no_llm:
        n = batch_generate(svc, progress=lambda i, total: print(f"\r  narratives {i}/{total}", end=""))
        print(f"\nWrote {n} AI narratives with {describe()}.")
    else:
        print("LLM disabled or not configured: deterministic narratives only.")
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
