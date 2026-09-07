"""OPD training diagnostics: truncation rate, code-jump, timeline merge."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist


def _eos_id_list(eos_token_id) -> list[int]:
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, (list, tuple)):
        return [int(x) for x in eos_token_id]
    return [int(eos_token_id)]


@torch.no_grad()
def rollout_truncation_metrics(
    generated_tokens: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    eos_token_id,
    max_length: int,
    pad_token_id=None,
) -> dict[str, float]:
    """Per-microbatch rollout stats.

    A sample is *truncated* if it never emits EOS in the response and its
    effective length reaches ``max_length`` (hit the generation budget).
    """
    prompt_length = int(prompt_attention_mask.shape[1])
    batch, seq_len = generated_tokens.shape
    response = generated_tokens[:, prompt_length:]
    eos_ids = _eos_id_list(eos_token_id)

    if response.numel() == 0:
        return {
            "truncation_rate": 0.0,
            "mean_response_len": 0.0,
            "mean_seq_len": float(seq_len),
            "hit_max_length_rate": 0.0,
            "eos_rate": 0.0,
        }

    has_eos = torch.zeros(batch, dtype=torch.bool, device=generated_tokens.device)
    for token_id in eos_ids:
        has_eos |= (response == token_id).any(dim=-1)

    if pad_token_id is not None and pad_token_id not in eos_ids:
        non_pad = generated_tokens != pad_token_id
        seq_lens = non_pad.sum(dim=-1)
    else:
        # pad == eos or no pad: use first-EOS index, else full length
        seq_lens = torch.full(
            (batch,), seq_len, dtype=torch.long, device=generated_tokens.device
        )
        if eos_ids:
            is_eos = torch.zeros_like(generated_tokens, dtype=torch.bool)
            for token_id in eos_ids:
                is_eos |= generated_tokens == token_id
            any_eos = is_eos.any(dim=-1)
            first_eos = is_eos.float().argmax(dim=-1)
            seq_lens = torch.where(any_eos, first_eos + 1, seq_lens)

    response_lens = (seq_lens - prompt_length).clamp_min(0)
    hit_cap = seq_lens >= int(max_length)
    truncated = (~has_eos) & hit_cap

    return {
        "truncation_rate": float(truncated.float().mean().item()),
        "mean_response_len": float(response_lens.float().mean().item()),
        "mean_seq_len": float(seq_lens.float().mean().item()),
        "hit_max_length_rate": float(hit_cap.float().mean().item()),
        "eos_rate": float(has_eos.float().mean().item()),
    }


@torch.no_grad()
def snapshot_quant_codes(model) -> dict[str, torch.Tensor]:
    """CPU int16 codes for every QuantLinear (rank-0 diagnostics)."""
    from quantize.int_linear_fake import QuantLinear

    codes = {}
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        if not getattr(module, "use_weight_quant", False):
            continue
        quantizer = module.weight_quantizer
        if not hasattr(quantizer, "integer_codes"):
            continue
        codes[name] = quantizer.integer_codes(module.weight.data).detach().to("cpu")
    return codes


@torch.no_grad()
def code_jump_rate(
    prev_codes: Optional[dict[str, torch.Tensor]],
    new_codes: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Fraction of quantized weights whose integer code changed."""
    if not prev_codes or not new_codes:
        return {
            "code_jump_rate": 0.0,
            "code_jump_count": 0.0,
            "code_total": float(sum(v.numel() for v in new_codes.values())),
            "master_rel_change": 0.0,
        }

    jumped = 0
    total = 0
    for name, cur in new_codes.items():
        old = prev_codes.get(name)
        if old is None or old.shape != cur.shape:
            continue
        jumped += int((cur != old).sum().item())
        total += int(cur.numel())

    rate = float(jumped) / float(max(1, total))
    return {
        "code_jump_rate": rate,
        "code_jump_count": float(jumped),
        "code_total": float(total),
    }


@torch.no_grad()
def master_rel_change(
    prev_weights: Optional[dict[str, torch.Tensor]],
    model,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Mean |ΔW| / (|W|+eps) over QuantLinear masters; also return new snapshot."""
    from quantize.int_linear_fake import QuantLinear

    new_weights: dict[str, torch.Tensor] = {}
    num = 0.0
    den = 0.0
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        w = module.weight.data.detach().to("cpu", dtype=torch.float32)
        new_weights[name] = w
        if prev_weights is None or name not in prev_weights:
            continue
        old = prev_weights[name]
        if old.shape != w.shape:
            continue
        delta = (w - old).abs()
        num += float(delta.sum().item())
        den += float((old.abs() + 1e-8).sum().item())
    rel = num / max(den, 1e-12) if prev_weights else 0.0
    return rel, new_weights


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


_STEP_RE = re.compile(r"checkpoint-(\d+)$")


def _extract_scores(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                low = path.lower()
                if any(
                    tok in low
                    for tok in (
                        "acc",
                        "score",
                        "exact_match",
                        "pass@1",
                        "pass_at",
                        "accuracy",
                        "metric",
                    )
                ):
                    out[path] = float(val)
            elif isinstance(val, (dict, list)):
                out.update(_extract_scores(val, path))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            out.update(_extract_scores(val, f"{prefix}[{i}]"))
    return out


def load_eval_scores_for_step(eval_root: Path, step: int) -> dict[str, float]:
    """Best-effort scrape of evalscope / summary JSON under one checkpoint dir."""
    ckpt_dir = eval_root / f"checkpoint-{step}"
    if not ckpt_dir.is_dir():
        return {}
    scores: dict[str, float] = {}
    candidates = list(ckpt_dir.rglob("*.json"))
    # Prefer compact summaries first.
    candidates.sort(
        key=lambda p: (
            0 if "summary" in p.name.lower() else 1,
            0 if "report" in p.name.lower() else 1,
            len(p.parts),
            str(p),
        )
    )
    for path in candidates[:40]:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        extracted = _extract_scores(data)
        for key, val in extracted.items():
            # Keep first hit per leaf name to avoid duplicates from huge trees.
            leaf = key.split(".")[-1]
            scores.setdefault(f"eval/{leaf}", val)
            scores.setdefault(f"eval/{key}", val)
    # Common dataset short names if present as top-level keys in any summary.
    for path in candidates[:10]:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for ds in ("gsm8k", "math_500", "aime24", "aime25"):
            node = data.get(ds)
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        scores[f"eval/{ds}/{k}"] = float(v)
            elif isinstance(node, (int, float)):
                scores[f"eval/{ds}"] = float(node)
    return scores


def merge_timeline(
    metrics_jsonl: Path,
    eval_root: Path,
    out_csv: Path,
    out_jsonl: Path,
) -> int:
    """Join per-step train metrics with checkpoint eval scores (same step axis)."""
    rows: list[dict[str, Any]] = []
    if metrics_jsonl.is_file():
        with metrics_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    by_step = {int(r["step"]): dict(r) for r in rows if "step" in r}

    if eval_root.is_dir():
        for child in eval_root.iterdir():
            m = _STEP_RE.match(child.name)
            if not m:
                continue
            step = int(m.group(1))
            rec = by_step.setdefault(step, {"step": step})
            rec.update(load_eval_scores_for_step(eval_root, step))

    ordered = [by_step[s] for s in sorted(by_step)]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # CSV: union of keys
    keys: list[str] = []
    seen = set()
    for rec in ordered:
        for k in rec:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write(",".join(keys) + "\n")
        for rec in ordered:
            f.write(
                ",".join(
                    "" if rec.get(k) is None else str(rec.get(k)) for k in keys
                )
                + "\n"
            )
    return len(ordered)
