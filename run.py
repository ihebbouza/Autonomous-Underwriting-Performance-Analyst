"""
The unattended path. Designed to run from cron/Airflow at 6am with nobody watching.

Exit codes: 0 on success, 1 on a data validation failure (so it's safe to wire into alerting).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from agent import AnalystAgent
from data import DataValidationError


def main():
    parser = argparse.ArgumentParser(description="Mosaic weekly underwriting performance pack")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--min-sustained-weeks", type=int, default=3)
    parser.add_argument("--as-of-week", default=None, help="YYYY-MM-DD; defaults to the latest week in the data")
    parser.add_argument("--output-dir", default="sample_output")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        agent = AnalystAgent(data_dir=args.data_dir, min_sustained_weeks=args.min_sustained_weeks)
        result = agent.run(as_of_week=args.as_of_week, api_key=api_key)
    except DataValidationError as exc:
        print(f"DATA VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "narrative.md").write_text(result["narrative"])

    print(f"As of week: {result['as_of_week']}")
    print(f"Narrative source: {result['narrative_source']}")
    print(f"Top concerns: {[f['lob'] for f in result['top_concerns']]}")
    print(f"Top opportunity: {[f['lob'] for f in result['top_opportunities']]}")
    print(f"Written to {out_dir}/summary.json and {out_dir}/narrative.md")


if __name__ == "__main__":
    main()
