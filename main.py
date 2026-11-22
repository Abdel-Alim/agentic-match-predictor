"""CLI entry point: run one daily prediction cycle.

    python main.py                  # today (Africa/Cairo)
    python main.py --date 2026-11-23
    python main.py --no-cache --force-retrain
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from zoneinfo import ZoneInfo

from config import TIMEZONE
from graph import build_graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Agentic daily match predictor")
    ap.add_argument("--date", help="run date YYYY-MM-DD (default: today in configured timezone)")
    ap.add_argument("--no-cache", action="store_true", help="refetch all source data")
    ap.add_argument("--force-retrain", action="store_true", help="retrain even if the model is fresh")
    args = ap.parse_args()

    if args.date:
        run_date = dt.date.fromisoformat(args.date)
    else:
        run_date = dt.datetime.now(ZoneInfo(TIMEZONE)).date()

    initial = {
        "run_date": run_date.isoformat(),
        "no_cache": args.no_cache,
        "force_retrain": args.force_retrain,
        "report": [],
        "warnings": [],
    }
    graph = build_graph()
    final = graph.invoke(initial)

    print("=== run report ===")
    for line in final.get("report", []):
        print(" -", line)
    for w in final.get("warnings", []):
        print(" ! warning:", w)
    print("outputs:", ", ".join(final.get("outputs", [])) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
