"""Two-GPU Qwen smoke test for fake-quant sampled reverse-KL OPD.

Example:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
      tests/opd_two_gpu_smoke.py --model /path/to/Qwen3-1.7B --wbits 3
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from main_e2e_distill import PolicyGKDTrainer
from quantize.int_linear_fake import QuantLinear, opd_generate_context, set_op_by_name
from quantize.utils import set_quant_state


def replace_decoder_linears(model, wbits: int, group_size: int) -> int:
    replaced = 0
    for layer in model.model.layers:
        linears = {
            name: module
            for name, module in layer.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        for name, module in linears.items():
            set_op_by_name(layer, name, QuantLinear(module, wbits=wbits, group_size=group_size))
            replaced += 1
    set_quant_state(model, weight_quant=True)
    for name, parameter in model.named_parameters():
        if "zero_point" in name:
            parameter.requires_grad_(False)
    return replaced


def timed(device: torch.device, function):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--wbits", type=int, choices=[2, 3], required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(1234 + dist.get_rank())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    student = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    replaced = replace_decoder_linears(student, args.wbits, args.group_size)
    student.to(device)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    teacher.requires_grad_(False)

    student = DistributedDataParallel(student, device_ids=[local_rank])
    prompt_text = [
        "Solve carefully: what is 17 plus 25?",
        "Explain briefly why the sky appears blue.",
    ][dist.get_rank() % 2]
    batch = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_length = batch.input_ids.shape[1]

    def generate():
        with torch.no_grad(), opd_generate_context(student.module):
            return student.module.generate(
                **batch,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    sequences, generate_seconds = timed(device, generate)
    # Each rank generates one unpadded sequence, so EOS is a real token rather
    # than padding even though Qwen commonly uses EOS as its pad token.
    attention_mask = torch.ones_like(sequences)
    labels = sequences.clone()
    labels[:, :prompt_length] = -100

    def train_step():
        student.train()
        student_outputs = student(input_ids=sequences, attention_mask=attention_mask, use_cache=False)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=sequences, attention_mask=attention_mask, use_cache=False).logits
        loss = PolicyGKDTrainer.sampled_reverse_kl_policy_loss(
            student_outputs.logits[:, :-1],
            teacher_logits[:, :-1],
            labels[:, 1:],
            temperature=1.0,
        )
        loss.backward()
        scale_grad_sq = torch.zeros((), device=device)
        other_grad_sq = torch.zeros((), device=device)
        for name, parameter in student.named_parameters():
            if parameter.grad is None:
                continue
            grad_sq = parameter.grad.detach().float().square().sum()
            if "scale" in name:
                scale_grad_sq += grad_sq
            else:
                other_grad_sq += grad_sq
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        return loss.detach(), grad_norm.detach(), scale_grad_sq.sqrt(), other_grad_sq.sqrt()

    (loss, grad_norm, scale_grad_norm, other_grad_norm), train_seconds = timed(device, train_step)
    metrics = torch.tensor(
        [
            loss.float(),
            grad_norm.float(),
            scale_grad_norm.float(),
            other_grad_norm.float(),
            generate_seconds,
            train_seconds,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(metrics)
    metrics /= dist.get_world_size()

    if dist.get_rank() == 0:
        allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            "OPD_TWO_GPU_SMOKE_OK "
            f"wbits={args.wbits} replaced_linears={replaced} "
            f"loss={metrics[0].item():.6f} grad_norm={metrics[1].item():.6f} "
            f"scale_grad_norm={metrics[2].item():.6f} other_grad_norm={metrics[3].item():.6f} "
            f"generate_s={metrics[4].item():.3f} train_s={metrics[5].item():.3f} "
            f"rank0_peak_gib={allocated_gib:.2f}",
            flush=True,
        )

    assert torch.isfinite(metrics).all()
    assert metrics[1].item() > 0
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
