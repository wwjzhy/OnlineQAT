#!/usr/bin/env python3
"""Merge OPD per-step JSONL metrics with checkpoint eval scores.

Example:
  python scripts/merge_opd_timeline.py \\
    --metrics log/distill/Qwen3-1.7B-w2g128-opd-.../opd_step_metrics.jsonl \\
    --eval-root output/eval/Qwen3-1.7B-w2g128-opd-... \\
    --out-dir output/plots/Qwen3-1.7B-w2g128-opd-...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantize.opd_metrics import merge_timeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True, help="opd_step_metrics.jsonl")
    p.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Dir with checkpoint-N eval folders (e.g. output/eval/<tag>)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for timeline.csv / timeline.jsonl (default: beside metrics)",
    )
    args = p.parse_args()
    out_dir = args.out_dir or args.metrics.parent
    out_csv = out_dir / "opd_timeline.csv"
    out_jsonl = out_dir / "opd_timeline.jsonl"
    n = merge_timeline(args.metrics, args.eval_root, out_csv, out_jsonl)
    print(f"wrote {n} rows -> {out_csv} and {out_jsonl}")


if __name__ == "__main__":
    main()
