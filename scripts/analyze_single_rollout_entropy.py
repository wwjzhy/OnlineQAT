#!/usr/bin/env python3
"""Record per-token entropy for one fixed W2 rollout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def entropy_quartiles(values: list[float]) -> list[int]:
    """Assign rank-based Q1..Q4 labels; ties are broken by response position."""
    labels = [0] * len(values)
    for rank, index in enumerate(sorted(range(len(values)), key=lambda i: (values[i], i))):
        labels[index] = min(4, math.ceil((rank + 1) * 4 / len(values)))
    return labels


def self_test() -> None:
    assert entropy_quartiles([0.1, 0.2, 0.3, 0.4]) == [1, 2, 3, 4]
    assert entropy_quartiles([0.4, 0.1, 0.3, 0.2]) == [4, 1, 3, 2]
    assert entropy_quartiles([0.1]) == [4]
    assert entropy_quartiles([]) == []
    print("entropy quartile self-test: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-number", type=int, default=0,
                        help="zero-based index among usable prompts after dataset shuffle")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dataset-seed", type=int, default=2)
    parser.add_argument("--sampling-seed", type=int, default=2)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.prompt_number < 0 or args.max_length <= 1 or args.temperature <= 0:
        parser.error("prompt-number must be >= 0, max-length > 1, and temperature > 0")
    if not (args.checkpoint / "config.json").is_file():
        parser.error(f"checkpoint is missing config.json: {args.checkpoint}")

    import torch
    from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

    from quantize.int_linear_fake import load_quantized_model, opd_generate_context
    from quantize.utils import set_quant_state
    from scripts.analyze_rollout_repetition import fixed_prompts, response_tokens

    class EntropyTemperatureProcessor(LogitsProcessor):
        def __init__(self, temperature: float):
            self.temperature = temperature
            self.entropies: list[float] = []

        def __call__(self, input_ids, scores):
            scaled = scores / self.temperature
            log_probs = torch.log_softmax(scaled.float(), dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            self.entropies.extend(entropy.detach().cpu().tolist())
            return scaled

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    prompt = fixed_prompts(
        tokenizer,
        args.prompt_number + 1,
        args.dataset_seed,
        args.max_length,
    )[args.prompt_number]
    encoded = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(encoded)
    prompt_length = int(encoded.shape[1])

    model, tokenizer = load_quantized_model(
        str(args.checkpoint),
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

    torch.manual_seed(args.sampling_seed)
    torch.cuda.manual_seed_all(args.sampling_seed)
    recorder = EntropyTemperatureProcessor(args.temperature)
    with torch.no_grad(), opd_generate_context(model):
        generated = model.generate(
            input_ids=encoded,
            attention_mask=attention_mask,
            max_new_tokens=args.max_length - prompt_length,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            logits_processor=LogitsProcessorList([recorder]),
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos,
        )

    response = response_tokens(generated[0], prompt_length, eos_ids)
    token_ids = response.detach().cpu().tolist()
    entropies = recorder.entropies[:len(token_ids)]
    if len(entropies) != len(token_ids):
        raise RuntimeError(
            f"entropy/token length mismatch: {len(entropies)} vs {len(token_ids)}"
        )
    quartiles = entropy_quartiles(entropies)
    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    rows = [
        {
            "response_pos": position,
            "token_id": token_id,
            "token": str(token_piece),
            "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
            "entropy_nats": entropy,
            "entropy_bits": entropy / math.log(2),
            "entropy_quartile": quartile,
            "top_25_percent": quartile == 4,
        }
        for position, (token_id, token_piece, entropy, quartile) in enumerate(
            zip(token_ids, token_pieces, entropies, quartiles)
        )
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "response_pos", "token_id", "token", "decoded", "entropy_nats",
        "entropy_bits", "entropy_quartile", "top_25_percent",
    ]
    for name, selected in [
        ("tokens.csv", rows),
        ("top25_high_entropy_tokens.csv", [row for row in rows if row["top_25_percent"]]),
    ]:
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)

    quartile_summary = {}
    for quartile in range(1, 5):
        values = [row["entropy_nats"] for row in rows if row["entropy_quartile"] == quartile]
        quartile_summary[f"Q{quartile}"] = {
            "count": len(values),
            "min_entropy_nats": min(values) if values else None,
            "max_entropy_nats": max(values) if values else None,
        }
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": "open-thoughts/OpenThoughts3-1.2M train",
        "shuffled_index": prompt["shuffled_index"],
        "prompt_number": args.prompt_number,
        "prompt_length": prompt_length,
        "response_length": len(rows),
        "temperature": args.temperature,
        "top_k": 0,
        "sampling_seed": args.sampling_seed,
        "entropy_unit": "nats (bits also written per token)",
        "mean_entropy_nats": sum(entropies) / len(entropies) if entropies else None,
        "quartiles": quartile_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
