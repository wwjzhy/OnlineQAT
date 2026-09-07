"""Unit tests for OPD truncation / code-jump helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

from quantize.opd_metrics import (
    code_jump_rate,
    merge_timeline,
    rollout_truncation_metrics,
)
from quantize.quantizer import UniformAffineQuantizer


def test_truncation_hit_max_without_eos():
    # prompt len 2, max_length 6, no EOS in response
    gen = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    prompt_mask = torch.ones(1, 2, dtype=torch.long)
    m = rollout_truncation_metrics(
        generated_tokens=gen,
        prompt_attention_mask=prompt_mask,
        eos_token_id=99,
        max_length=6,
        pad_token_id=0,
    )
    assert m["truncation_rate"] == 1.0
    assert m["eos_rate"] == 0.0
    assert m["hit_max_length_rate"] == 1.0
    assert m["mean_response_len"] == 4.0


def test_truncation_eos_not_counted():
    gen = torch.tensor([[1, 2, 3, 99, 0, 0]], dtype=torch.long)
    prompt_mask = torch.ones(1, 2, dtype=torch.long)
    m = rollout_truncation_metrics(
        generated_tokens=gen,
        prompt_attention_mask=prompt_mask,
        eos_token_id=99,
        max_length=6,
        pad_token_id=0,
    )
    assert m["truncation_rate"] == 0.0
    assert m["eos_rate"] == 1.0


def test_integer_codes_and_jump():
    w = torch.randn(8, 16)
    q = UniformAffineQuantizer(n_bits=2, group_size=16, weight=w)
    codes = q.integer_codes(w)
    assert codes.dtype == torch.int16
    assert codes.shape == w.shape
    assert int(codes.min()) >= q.qmin
    assert int(codes.max()) <= q.qmax

    prev = {"layer": codes.clone()}
    new = {"layer": codes.clone()}
    new["layer"][0, 0] = (new["layer"][0, 0] + 1).clamp(q.qmin, q.qmax)
    if new["layer"][0, 0] == prev["layer"][0, 0]:
        new["layer"][0, 0] = q.qmin if prev["layer"][0, 0] != q.qmin else q.qmax
    jump = code_jump_rate(prev, new)
    assert jump["code_jump_count"] == 1.0
    assert jump["code_total"] == float(codes.numel())
    assert abs(jump["code_jump_rate"] - 1.0 / codes.numel()) < 1e-9


def test_merge_timeline(tmp_path: Path):
    metrics = tmp_path / "opd_step_metrics.jsonl"
    metrics.write_text(
        json.dumps({"step": 5, "grad_norm": 1.2, "opd/truncation_rate": 0.1})
        + "\n"
        + json.dumps({"step": 10, "grad_norm": 2.0, "opd/code_jump_rate": 0.05})
        + "\n"
    )
    eval_root = tmp_path / "eval"
    ckpt = eval_root / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "summary.json").write_text(
        json.dumps({"gsm8k": {"accuracy": 0.42}, "math_500": {"score": 0.21}})
    )
    out_csv = tmp_path / "opd_timeline.csv"
    out_jsonl = tmp_path / "opd_timeline.jsonl"
    n = merge_timeline(metrics, eval_root, out_csv, out_jsonl)
    assert n == 2
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert rows[0]["step"] == 5
    assert rows[1]["step"] == 10
    assert rows[1].get("eval/gsm8k/accuracy") == 0.42
    assert out_csv.is_file()
