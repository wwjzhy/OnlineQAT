#!/usr/bin/env python3
"""Compare terminal repetition on fixed OpenThoughts prompts across checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from quantize.int_linear_fake import load_quantized_model, opd_generate_context
from quantize.opd_metrics import terminal_repeat_length
from quantize.utils import set_quant_state


def checkpoint_arg(value: str) -> tuple[int, Path]:
    try:
        step, path = value.split("=", 1)
        return int(step), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected STEP=PATH") from exc


def prompt_ids(row: dict, tokenizer) -> list[int]:
    messages = [
        {"role": "user" if item["from"] == "human" else "assistant", "content": item["value"]}
        for item in row["conversations"]
        if item["from"] in ("human", "gpt")
    ]
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )


def fixed_prompts(tokenizer, count: int, seed: int, max_length: int) -> list[dict]:
    dataset = load_dataset(
        "open-thoughts/OpenThoughts3-1.2M",
        split="train",
        verification_mode="no_checks",
    ).shuffle(seed=seed)
    prompts = []
    for shuffled_index, row in enumerate(dataset):
        ids = prompt_ids(row, tokenizer)
        if len(ids) >= max_length:
            continue
        prompts.append({"shuffled_index": shuffled_index, "input_ids": ids})
        if len(prompts) == count:
            return prompts
    raise RuntimeError(f"only found {len(prompts)} usable prompts, requested {count}")


def response_tokens(sequence: torch.Tensor, prompt_len: int, eos_ids: set[int]) -> torch.Tensor:
    response = sequence[prompt_len:]
    for index, token in enumerate(response.tolist()):
        if token in eos_ids:
            return response[: index + 1]
    return response


@torch.no_grad()
def analyze_checkpoint(args, step: int, checkpoint: Path, prompts: list[dict]) -> list[dict]:
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint {step} missing config.json: {checkpoint}")
    model, tokenizer = load_quantized_model(
        str(checkpoint),
        args.wbits,
        args.group_size,
        replace=True,
        strict=True,
        use_flash_attn=True,
        device=torch.device("cpu"),
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(args.device).eval()
    set_quant_state(model, weight_quant=True)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    rows = []
    for start in range(0, len(prompts), args.batch_size):
        batch_seed = args.sampling_seed + start
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        batch = prompts[start : start + args.batch_size]
        encoded = tokenizer.pad(
            {
                "input_ids": [row["input_ids"] for row in batch],
                "attention_mask": [[1] * len(row["input_ids"]) for row in batch],
            },
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        padded_prompt_len = int(encoded["input_ids"].shape[1])
        with opd_generate_context(model):
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_length - padded_prompt_len,
                do_sample=True,
                temperature=args.temperature,
                top_k=0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos,
            )
        for metadata, sequence in zip(batch, generated):
            response = response_tokens(sequence, padded_prompt_len, eos_ids)
            response_len = int(response.numel())
            repeat_len = terminal_repeat_length(response, max_window=None)
            has_eos = bool(response_len and int(response[-1]) in eos_ids)
            rows.append(
                {
                    "step": step,
                    "shuffled_index": metadata["shuffled_index"],
                    "prompt_length": len(metadata["input_ids"]),
                    "response_length": response_len,
                    "terminal_repeat_length": repeat_len,
                    "pre_repeat_length": response_len - repeat_len,
                    "repeat_fraction": repeat_len / max(1, response_len),
                    "has_eos": has_eos,
                    "hit_max_length": (not has_eos and padded_prompt_len + response_len >= args.max_length),
                }
            )
        print(f"step {step}: {min(start + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def summarize(step: int, rows: list[dict]) -> dict:
    repeated = [row for row in rows if row["terminal_repeat_length"] > 0]
    total_response = sum(row["response_length"] for row in rows)
    mean = lambda values: sum(values) / max(1, len(values))
    return {
        "step": step,
        "num_rollouts": len(rows),
        "repetition_rate": len(repeated) / max(1, len(rows)),
        "mean_response_length": mean([row["response_length"] for row in rows]),
        "mean_terminal_repeat_length": mean([row["terminal_repeat_length"] for row in rows]),
        "mean_terminal_repeat_length_repeated": mean([row["terminal_repeat_length"] for row in repeated]),
        "mean_pre_repeat_length": mean([row["pre_repeat_length"] for row in rows]),
        "mean_pre_repeat_length_repeated": mean([row["pre_repeat_length"] for row in repeated]),
        "repeat_token_fraction": sum(row["terminal_repeat_length"] for row in rows) / max(1, total_response),
        "eos_rate": mean([float(row["has_eos"]) for row in rows]),
        "hit_max_length_rate": mean([float(row["hit_max_length"]) for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=checkpoint_arg, required=True, help="repeat STEP=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dataset-seed", type=int, default=2)
    parser.add_argument("--sampling-seed", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.num_prompts <= 0 or args.batch_size <= 0 or args.max_length <= 1:
        parser.error("num-prompts and batch-size must be positive; max-length must be > 1")

    checkpoints = sorted(args.checkpoint)
    if len({step for step, _ in checkpoints}) != len(checkpoints):
        parser.error("checkpoint steps must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1])
    prompts = fixed_prompts(tokenizer, args.num_prompts, args.dataset_seed, args.max_length)
    del tokenizer
    (args.output_dir / "probe_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "open-thoughts/OpenThoughts3-1.2M train",
                "dataset_seed": args.dataset_seed,
                "sampling_seed": args.sampling_seed,
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_k": 0,
                "batch_size": args.batch_size,
                "wbits": args.wbits,
                "group_size": args.group_size,
                "checkpoints": [
                    {"step": step, "path": str(path)} for step, path in checkpoints
                ],
                "repeat_detector": {
                    "period_tokens": [1, 10],
                    "minimum_agreement": 0.9,
                    "minimum_tokens": 64,
                    "minimum_cycles": 3,
                    "full_response_scan": True,
                },
                "prompts": [
                    {"shuffled_index": row["shuffled_index"], "prompt_length": len(row["input_ids"])}
                    for row in prompts
                ],
            },
            indent=2,
        ) + "\n"
    )

    summaries = []
    for step, checkpoint in checkpoints:
        rows = analyze_checkpoint(args, step, checkpoint, prompts)
        summaries.append(summarize(step, rows))
        (args.output_dir / f"checkpoint-{step}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    fields = list(summaries[0])
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
