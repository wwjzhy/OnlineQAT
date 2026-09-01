"""Two-GPU smoke test for the PV-OPD FullPair forward/backward path.

Example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
    tests/pv_opd_two_gpu_smoke.py --model /path/to/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from main_e2e_distill import PolicyGKDTrainer
from quantize.int_linear_fake import (
    QuantLinear,
    opd_generate_context,
    quantized_precision_view,
    set_op_by_name,
)
from quantize.utils import set_quant_state


def replace_decoder_linears(model, group_size):
    replaced = 0
    for layer in model.model.layers:
        linears = {
            name: module
            for name, module in layer.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        for name, module in linears.items():
            set_op_by_name(
                layer, name, QuantLinear(module, wbits=2, group_size=group_size)
            )
            replaced += 1
    set_quant_state(model, weight_quant=True)
    for name, parameter in model.named_parameters():
        if "zero_point" in name or "embed_tokens" in name:
            parameter.requires_grad_(False)
    return replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(2026 + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    replaced = replace_decoder_linears(student, args.group_size)
    student.to(device)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    teacher.requires_grad_(False)
    student = DistributedDataParallel(student, device_ids=[local_rank])

    prompts = [
        "Solve carefully: what is 17 plus 25?",
        "Write a Python function that returns the square of an integer.",
    ]
    batch = tokenizer(prompts[rank % len(prompts)], return_tensors="pt").to(device)
    prompt_length = batch.input_ids.shape[1]
    with torch.no_grad(), opd_generate_context(student.module):
        tokens = student.module.generate(
            **batch,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    attention_mask = torch.ones_like(tokens)
    labels = tokens.clone()
    labels[:, :prompt_length] = -100
    shifted_labels = labels[:, 1:]

    student.train()
    target_logits = student(
        input_ids=tokens, attention_mask=attention_mask, use_cache=False
    ).logits[:, :-1]
    student_logp = PolicyGKDTrainer.sampled_action_log_probs(
        target_logits, shifted_labels, temperature=0.6
    )
    with torch.no_grad():
        teacher_logits = teacher(
            input_ids=tokens, attention_mask=attention_mask, use_cache=False
        ).logits[:, :-1]
        teacher_logp = PolicyGKDTrainer.sampled_action_log_probs(
            teacher_logits, shifted_labels, temperature=0.6
        )
        del teacher_logits
        with quantized_precision_view(student.module, 4):
            probe_logits = student.module(
                input_ids=tokens, attention_mask=attention_mask, use_cache=False
            ).logits[:, :-1]
            probe_logp = PolicyGKDTrainer.sampled_action_log_probs(
                probe_logits, shifted_labels, temperature=0.6
            )

    valid = shifted_labels != -100
    a_fp = teacher_logp - student_logp.detach()
    a_prec = probe_logp - student_logp.detach()
    gate, _ = PolicyGKDTrainer.build_precision_gate(a_fp, a_prec, valid)
    reverse_advantage = (-a_fp).clamp(-5.0, 5.0)
    loss = (gate * reverse_advantage * student_logp)[valid].mean()
    loss.backward()

    scale_grad = torch.zeros((), device=device)
    weight_grad = torch.zeros((), device=device)
    for name, parameter in student.named_parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().float().square().sum()
        if "scale" in name:
            scale_grad += grad_norm
        elif name.endswith("weight"):
            weight_grad += grad_norm
    metrics = torch.stack(
        [
            loss.detach().float(),
            gate[valid].mean().float(),
            scale_grad.sqrt(),
            weight_grad.sqrt(),
        ]
    )
    dist.all_reduce(metrics)
    metrics /= dist.get_world_size()
    assert torch.isfinite(metrics).all()
    assert metrics[2] > 0 and metrics[3] > 0
    if rank == 0:
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            "PV_OPD_TWO_GPU_SMOKE_OK "
            f"replaced={replaced} loss={metrics[0].item():.6f} "
            f"gate={metrics[1].item():.4f} scale_grad={metrics[2].item():.6f} "
            f"weight_grad={metrics[3].item():.6f} peak_gib={peak_gib:.2f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
