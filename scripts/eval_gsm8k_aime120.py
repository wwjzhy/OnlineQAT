#!/usr/bin/env python3
"""GSM8K + AIME-120 eval for a converted HuggingFace model (Stage 3 output).

Protocol matches OPT-QAT comparison runs:
  thinking off, greedy, max_new_tokens=2048.

Usage:
  python scripts/eval_gsm8k_aime120.py \
    --model_path ./output/vllm/Qwen3-1.7B-w3g128 \
    --output_dir ./output/eval/Qwen3-1.7B-w3g128
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"

GSM8K_TMPL = (
    "Solve the following math problem step by step. "
    "Put your final answer after ####.\n\nQuestion: {question}\n\nSolution:"
)
AIME_TMPL = (
    "Solve the following math problem. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_chat(tokenizer, user: str) -> str:
    messages = [{"role": "user", "content": user}]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_gsm8k(text: str) -> str | None:
    if "####" in text:
        match = re.search(r"-?[\d.]+", text.split("####")[-1].replace(",", ""))
        if match:
            return match.group()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


def extract_boxed(text: str) -> str | None:
    index = text.rfind("\\boxed{")
    if index < 0:
        return None
    start = index + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start : end - 1].strip() if depth == 0 else None


def norm_num(value: str) -> float | None:
    try:
        return float(value.strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def gsm8k_ok(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    a, b = norm_num(pred), norm_num(gold)
    if a is not None and b is not None:
        return abs(a - b) < 1e-6
    return pred.strip() == gold.strip()


def aime_ok(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    a, b = norm_num(pred), norm_num(gold)
    if a is not None and b is not None:
        return abs(a - b) < 1e-6
    return re.sub(r"\s+", "", pred) == re.sub(r"\s+", "", str(gold))


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: list[str], args) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = model.device
    outs: list[str] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_prompt_len,
            return_tensors="pt",
        ).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        plen = enc["input_ids"].shape[1]
        for seq in gen:
            outs.append(tokenizer.decode(seq[plen:], skip_special_tokens=True))
        print(f"  generated {min(start + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def eval_gsm8k(model, tokenizer, args) -> dict:
    data = load_jsonl(EVAL_DIR / "gsm8k_test.jsonl")
    prompts = [apply_chat(tokenizer, GSM8K_TMPL.format(question=r["question"])) for r in data]
    texts = generate_texts(model, tokenizer, prompts, args)
    correct = 0
    results = []
    for row, text in zip(data, texts):
        pred = extract_gsm8k(text)
        ok = gsm8k_ok(pred, row["answer"])
        correct += int(ok)
        results.append({"gold": row["answer"], "pred": pred, "correct": ok, "output": text[:2000]})
    acc = correct / max(len(data), 1)
    print(f"GSM8K accuracy: {acc:.4f} ({correct}/{len(data)})")
    return {"accuracy": acc, "correct": correct, "total": len(data), "results": results}


def eval_aime120(model, tokenizer, args) -> dict:
    data = load_jsonl(EVAL_DIR / "aime120.jsonl")
    prompts = [apply_chat(tokenizer, AIME_TMPL.format(problem=r["problem"])) for r in data]
    texts = generate_texts(model, tokenizer, prompts, args)
    correct = 0
    results = []
    for row, text in zip(data, texts):
        pred = extract_boxed(text)
        ok = aime_ok(pred, str(row["answer"]))
        correct += int(ok)
        results.append({"gold": row["answer"], "pred": pred, "correct": ok, "output": text[:2000]})
    acc = correct / max(len(data), 1)
    print(f"AIME120 accuracy: {acc:.4f} ({correct}/{len(data)})")
    return {"accuracy": acc, "correct": correct, "total": len(data), "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="./output/eval/Qwen3-1.7B-w3g128")
    p.add_argument("--benchmarks", nargs="+", default=["gsm8k", "aime120"])
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_prompt_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    for name in args.benchmarks:
        fname = "gsm8k_test.jsonl" if name == "gsm8k" else "aime120.jsonl"
        path = EVAL_DIR / fname
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run python scripts/download_datasets.py")

    print(f"loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if "gsm8k" in args.benchmarks:
        summary["gsm8k"] = eval_gsm8k(model, tokenizer, args)
        (out_dir / "gsm8k_results.json").write_text(json.dumps(summary["gsm8k"], ensure_ascii=False, indent=2))
    if "aime120" in args.benchmarks:
        summary["aime120"] = eval_aime120(model, tokenizer, args)
        (out_dir / "aime120_results.json").write_text(json.dumps(summary["aime120"], ensure_ascii=False, indent=2))

    slim = {
        k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
        for k, v in summary.items()
    }
    (out_dir / "summary.json").write_text(json.dumps(slim, indent=2))
    print("summary", json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
