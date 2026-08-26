#!/usr/bin/env python3
"""Download training + eval data for ReasoningQAT Qwen3-1.7B.

Training (HF cache, used by main_block_qat / main_e2e_distill):
  - open-thoughts/OpenThoughts3-1.2M
  - HuggingFaceFW/fineweb-edu sample-10BT  (first N docs -> data/raw/fineweb_edu_subset.jsonl)

Eval (written to data/eval/):
  - GSM8K test, AIME-120  (OPT-QAT comparison)
  - MATH-500               (paper; also used by evalscope)

Paper evalscope tasks (auto-downloaded on first eval):
  MATH-500, LiveCodeBench, MMLU-Redux, GPQA-Diamond, IFEval

Usage (on the cluster, from repo root):
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=$PWD/hf_cache
  python scripts/download_datasets.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PIP_CONFIG_FILE", "/dev/null")

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
FINEWEB_DOCS = int(os.environ.get("FINEWEB_DOCS", "8000"))


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} -> {path}")


def download_openthoughts() -> None:
    print("==> OpenThoughts3-1.2M (full train split, cached under HF_HOME)")
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    print(f"  cached {len(ds)} rows")


def download_fineweb_subset() -> None:
    out = RAW_DIR / "fineweb_edu_subset.jsonl"
    if out.is_file() and out.stat().st_size > 0:
        print(f"==> FineWeb-Edu subset exists, skip ({out})")
        return
    print(f"==> FineWeb-Edu sample-10BT streaming first {FINEWEB_DOCS} docs")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for sample in ds:
            text = sample.get("text") or ""
            if not text.strip():
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
            if n >= FINEWEB_DOCS:
                break
            if n % 500 == 0:
                print(f"  {n}/{FINEWEB_DOCS}")
    print(f"  wrote {n} -> {out}")


def _field(row: dict, *keys: str, default=""):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def download_gsm8k() -> None:
    print("==> GSM8K test")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for item in ds:
        answer_text = item["answer"]
        final = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
        rows.append(
            {
                "question": item["question"],
                "answer": final,
                "answer_full": answer_text,
            }
        )
    save_jsonl(rows, EVAL_DIR / "gsm8k_test.jsonl")


def _aime_row(problem, answer, year, src, idx) -> dict | None:
    if not problem or answer in (None, ""):
        return None
    return {
        "id": str(idx),
        "year": int(year) if str(year).isdigit() else year,
        "problem": problem,
        "answer": str(answer).strip(),
        "source": src,
    }


def download_aime120() -> None:
    print("==> AIME-120 (2024 + 2025 + 1983-2024 years 2021-2023)")
    rows: list[dict] = []
    seen = set()

    def add(row: dict | None) -> None:
        if row is None:
            return
        key = (row["year"], row["problem"][:80])
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    ds24 = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    for i, item in enumerate(ds24):
        add(
            _aime_row(
                _field(item, "problem", "Problem", "question"),
                _field(item, "answer", "Answer"),
                _field(item, "year", default=2024) or 2024,
                "Maxwell-Jia/AIME_2024",
                _field(item, "id", default=f"2024-{i}"),
            )
        )

    ds25 = load_dataset("math-ai/aime25", split="train")
    for i, item in enumerate(ds25):
        add(
            _aime_row(
                _field(item, "problem", "Problem", "question"),
                _field(item, "answer", "Answer"),
                2025,
                "math-ai/aime25",
                _field(item, "id", default=str(i)),
            )
        )

    ds_hist = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    for i, item in enumerate(ds_hist):
        year = _field(item, "year", "Year")
        try:
            year_i = int(year)
        except (TypeError, ValueError):
            continue
        if year_i not in (2021, 2022, 2023):
            continue
        add(
            _aime_row(
                _field(item, "problem", "Problem", "Question", "question"),
                _field(item, "answer", "Answer"),
                year_i,
                "di-zhang-fdu/AIME_1983_2024",
                _field(item, "id", default=f"{year_i}-{i}"),
            )
        )

    if len(rows) > 120:
        rows = rows[:120]
    if len(rows) != 120:
        print(f"  WARNING: got {len(rows)} AIME items, expected 120")
    save_jsonl(rows, EVAL_DIR / "aime120.jsonl")


def download_math500() -> None:
    print("==> MATH-500")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for item in ds:
        rows.append(
            {
                "problem": item["problem"],
                "solution": item.get("solution", ""),
                "answer": item.get("answer", ""),
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
            }
        )
    save_jsonl(rows, EVAL_DIR / "math500.jsonl")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '(default)')}")
    download_openthoughts()
    download_fineweb_subset()
    download_gsm8k()
    download_aime120()
    download_math500()
    print("done")
    print(f"  eval gsm8k   {EVAL_DIR / 'gsm8k_test.jsonl'}")
    print(f"  eval aime    {EVAL_DIR / 'aime120.jsonl'}")
    print(f"  eval math500 {EVAL_DIR / 'math500.jsonl'}")
    print(f"  fineweb      {RAW_DIR / 'fineweb_edu_subset.jsonl'}")
    print("  paper/evalscope benches are downloaded on first eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
