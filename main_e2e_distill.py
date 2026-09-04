import os
import sys
import random
import numpy as np
import torch

# # patch_dist_ops.py  ← 実行スクリプトの一番最初で import
# import os, traceback
# import torch.distributed as dist

# def _wrap(fn, name):
#     def w(*a, **k):
#         print(f"\n== {name} (rank={os.getenv('RANK','?')}) ==", flush=True)
#         print("".join(traceback.format_stack(limit=20)), flush=True)
#         return fn(*a, **k)
#     return w

# for _n in ("all_reduce", "all_gather", "all_gather_into_tensor", "_all_gather_base"):
#     if hasattr(dist, _n) and callable(getattr(dist, _n)):
#         setattr(dist, _n, _wrap(getattr(dist, _n), f"dist.{_n}"))

import time
import dataclasses
from datautils_block import get_loaders, test_ppl
import torch.nn as nn
from tqdm import tqdm
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainerCallback
from quantize.int_linear_fake import (
    load_quantized_model,
    opd_generate_context,
    quantized_precision_view,
    resolve_attn_implementation,
)
from accelerate import infer_auto_device_map, dispatch_model
from trl import GKDConfig, GKDTrainer
from trl.models.utils import unwrap_model_for_generation
from datasets import load_dataset
import copy
import quantize.int_linear_fake as int_linear_fake
from quantize.utils import set_op_by_name, set_quant_state, quant_inplace
import types
from functools import partial
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass

torch.backends.cudnn.benchmark = True

from trl.trainer.utils import DataCollatorForChatML


class ChunkedActionLogProbs(torch.autograd.Function):
    """FP32 log-softmax with chunked backward recomputation.

    Keeping every ``chunk.float()`` in the regular autograd graph retains an
    entire FP32 vocabulary tensor. This function stores only the original
    logits plus a sequence-sized log normalizer and recreates each softmax
    chunk during backward.
    """

    @staticmethod
    def forward(ctx, logits, labels, temperature, vocab_chunk_size):
        valid_mask = labels != -100
        safe_labels = labels.masked_fill(~valid_mask, 0)
        token_logits = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).float()
        token_logits = token_logits / temperature

        log_normalizer = None
        for chunk in logits.split(vocab_chunk_size, dim=-1):
            chunk_lse = torch.logsumexp(chunk.float() / temperature, dim=-1)
            log_normalizer = (
                chunk_lse
                if log_normalizer is None
                else torch.logaddexp(log_normalizer, chunk_lse)
            )
        ctx.temperature = temperature
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.save_for_backward(logits, safe_labels, valid_mask, log_normalizer)
        return token_logits - log_normalizer

    @staticmethod
    def backward(ctx, grad_output):
        logits, safe_labels, valid_mask, log_normalizer = ctx.saved_tensors
        temperature = ctx.temperature
        grad_output = grad_output.float() * valid_mask
        grad_logits = torch.empty_like(logits)

        start = 0
        for chunk in logits.split(ctx.vocab_chunk_size, dim=-1):
            width = chunk.shape[-1]
            probabilities = torch.exp(
                chunk.float() / temperature - log_normalizer.unsqueeze(-1)
            )
            chunk_grad = (
                -grad_output.unsqueeze(-1) * probabilities / temperature
            )
            grad_logits[..., start : start + width] = chunk_grad.to(logits.dtype)
            start += width

        selected_grad = (grad_output / temperature).to(logits.dtype).unsqueeze(-1)
        grad_logits.scatter_add_(-1, safe_labels.unsqueeze(-1), selected_grad)
        return grad_logits, None, None, None


def dft_cross_entropy(
    source: torch.Tensor,   # = logits (..., vocab)
    target: torch.Tensor,   # = labels (…),
    loss_weight: torch.Tensor, # = loss weight,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    dft_alpha: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    # per-token CE（無視トークンは自動で無視＝値は0相当で扱う）
    target = target.to(source.device).long()
    per_tok_ce = F.cross_entropy(source, target, ignore_index=ignore_index, reduction="none")
    valid_mask = (target != ignore_index)

    # --- DFT重み: p(true) を安全に計算（ignore_indexはgatherしない）---
    with torch.no_grad():
        per_tok_weight_ce = F.cross_entropy(loss_weight, target, ignore_index=ignore_index, reduction="none")
        p_true = torch.exp(-per_tok_weight_ce)
        p_true = p_true * dft_alpha + (1 - dft_alpha)
    # -------------------------------------------------------------

    # 重み付け総和（無効トークンは0）
    weighted_sum = (per_tok_ce * p_true)[valid_mask].sum()

    if num_items_in_batch is None:
        # fixed_cross_entropy の mean と同じ：有効トークン数で割る
        denom = valid_mask.sum().clamp_min(1)
        loss = weighted_sum / denom
    else:
        # fixed_cross_entropy の "sum → / num_items_in_batch" と同じ規約
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(source.device)
        loss = weighted_sum / num_items_in_batch

    return loss

def DFTCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast for numerical stability
    logits = logits.float()

    # Shift so that tokens < n predict n
    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device).long()  # long を明示

    loss = dft_cross_entropy(logits, shift_labels, logits, num_items_in_batch, ignore_index, **kwargs)

    return loss

class PolicyGKDTrainer(GKDTrainer):
    def __init__(
        self,
        kl_weight=1.0,
        cross_entropy_weight=1.0,
        use_teacher_weight=False,
        use_dft_loss=False,
        top_k=None,
        kd_loss_type="jsd",
        mean_prob=0,
        beta=0.5,
        opd_mode=False,
        pv_opd_mode=False,
        pv_probe_bits=4,
        pv_gate_mode="full",
        pv_gate_max=2.0,
        pv_adv_clip=0.0,
        pv_adv_clip_warmup_steps=10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.use_teacher_weight = use_teacher_weight
        self.use_dft_loss = use_dft_loss
        self.top_k = top_k
        self.kd_loss_type = kd_loss_type
        self.mean_prob = mean_prob
        self.beta = beta
        self.opd_mode = opd_mode
        self.pv_opd_mode = pv_opd_mode
        self.pv_probe_bits = pv_probe_bits
        self.pv_gate_mode = pv_gate_mode
        self.pv_gate_max = pv_gate_max
        self.pv_adv_clip = pv_adv_clip
        self.pv_adv_clip_warmup_steps = pv_adv_clip_warmup_steps
        self._pv_adv_clip_estimates = []
        self._pv_adv_clip_value = pv_adv_clip if pv_adv_clip > 0 else None
        self._pv_clip_last_step = None
        self._pv_metrics_last_step = None
        # Keep teacher model on GPU - do not move to CPU to avoid memory leaks
        # Teacher model device should be managed by device_map during initialization

    @staticmethod
    def forward_kl_loss(student_logits, teacher_logits, labels=None, temperature=1.0, top_k=None):
        """Token-level forward KL(teacher || student), optionally on teacher top-k."""
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_log_probs, top_k, dim=-1)
            student_log_probs = torch.gather(student_log_probs, -1, top_k_indices)
            teacher_log_probs = torch.gather(teacher_log_probs, -1, top_k_indices)
        kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
        if labels is not None:
            mask = labels != -100
            kl = kl[mask]
            return kl.sum() / mask.sum().clamp(min=1)
        return kl.mean()

    @staticmethod
    def sampled_reverse_kl_policy_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=1.0,
    ):
        """Monte Carlo policy-gradient estimator of KL(student || teacher).

        ``labels`` are tokens sampled from the student policy. The sampled
        reverse-KL value is detached and used as a policy-gradient weight,
        matching slime's OPD objective without materializing full-vocabulary
        probability tensors.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        valid_mask = labels != -100
        safe_labels = labels.masked_fill(~valid_mask, 0).unsqueeze(-1)

        if temperature != 1.0:
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

        student_token_logits = student_logits.gather(-1, safe_labels).squeeze(-1)
        student_log_probs = student_token_logits - torch.logsumexp(student_logits, dim=-1)
        with torch.no_grad():
            teacher_token_logits = teacher_logits.gather(-1, safe_labels).squeeze(-1)
            teacher_log_probs = teacher_token_logits - torch.logsumexp(teacher_logits, dim=-1)
            sampled_reverse_kl = student_log_probs.detach() - teacher_log_probs

        # grad E_{a~pi_s}[log pi_s(a) - log pi_t(a)]
        # = E[(log pi_s(a) - log pi_t(a) + 1) grad log pi_s(a)].
        # The omitted +1 is a constant baseline with zero expected gradient.
        per_token_loss = sampled_reverse_kl * student_log_probs
        return per_token_loss[valid_mask].sum() / valid_mask.sum().clamp_min(1)

    @staticmethod
    def sampled_action_log_probs(
        logits, labels, temperature=1.0, vocab_chunk_size=16384
    ):
        """FP32 action log-probs without a full-vocabulary FP32 copy."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if torch.is_grad_enabled() and logits.requires_grad:
            return ChunkedActionLogProbs.apply(
                logits, labels, float(temperature), int(vocab_chunk_size)
            )
        valid_mask = labels != -100
        safe_labels = labels.masked_fill(~valid_mask, 0).unsqueeze(-1)
        token_logits = logits.gather(-1, safe_labels).squeeze(-1).float()
        if temperature != 1.0:
            token_logits = token_logits / temperature

        log_normalizer = None
        for chunk in logits.split(vocab_chunk_size, dim=-1):
            chunk_fp32 = chunk.float()
            if temperature != 1.0:
                chunk_fp32 = chunk_fp32 / temperature
            chunk_lse = torch.logsumexp(chunk_fp32, dim=-1)
            log_normalizer = (
                chunk_lse
                if log_normalizer is None
                else torch.logaddexp(log_normalizer, chunk_lse)
            )
        return token_logits - log_normalizer

    @staticmethod
    def build_precision_gate(
        a_fp,
        a_prec,
        valid_mask,
        gate_mode="full",
        gate_max=2.0,
        eps=1e-6,
    ):
        """Build the detached PV gate and normalize its valid-token mean."""
        if gate_mode not in {"full", "sign", "shuffled"}:
            raise ValueError(f"unsupported PV gate mode: {gate_mode}")
        same_direction = (a_fp * a_prec) > 0
        gate = same_direction.to(torch.float32)
        if gate_mode in {"full", "shuffled"}:
            recovery = (a_prec.abs() / (a_fp.abs() + eps)).clamp(max=1.0)
            gate = gate * recovery
        gate = gate * valid_mask

        gate_sum = gate.sum()
        gate_count = valid_mask.sum().to(gate.dtype)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stats = torch.stack([gate_sum, gate_count])
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            gate_sum, gate_count = stats[0], stats[1]
        gate_mean = gate_sum / gate_count.clamp_min(1.0)
        gate = (gate / (gate_mean + eps)).clamp(max=gate_max) * valid_mask

        if gate_mode == "shuffled":
            shuffled = gate[valid_mask]
            if shuffled.numel() > 1:
                shuffled = shuffled[torch.randperm(shuffled.numel(), device=gate.device)]
                gate = gate.clone()
                gate[valid_mask] = shuffled
        return gate.detach(), same_direction & valid_mask

    def _update_pv_adv_clip(self, a_fp, valid_mask):
        if self.pv_adv_clip > 0:
            return float(self.pv_adv_clip)
        step = int(getattr(self.state, "global_step", 0))
        valid_abs = a_fp.detach().abs()[valid_mask]
        if valid_abs.numel() == 0:
            return self._pv_adv_clip_value or 1.0
        if (
            self._pv_adv_clip_value is None
            or step < self.pv_adv_clip_warmup_steps
        ) and self._pv_clip_last_step != step:
            estimate = torch.quantile(valid_abs.float(), 0.99)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    estimate, op=torch.distributed.ReduceOp.SUM
                )
                estimate /= torch.distributed.get_world_size()
            self._pv_adv_clip_estimates.append(float(estimate.clamp_min(1e-6).cpu()))
            self._pv_adv_clip_value = sum(self._pv_adv_clip_estimates) / len(
                self._pv_adv_clip_estimates
            )
            self._pv_clip_last_step = step
        return self._pv_adv_clip_value or 1.0

    def _log_pv_metrics(
        self, labels, a_fp, a_prec, gate, same_direction, valid_mask, adv_clip
    ):
        step = int(getattr(self.state, "global_step", 0))
        if self._pv_metrics_last_step == step:
            return
        self._pv_metrics_last_step = step

        def masked_mean(value, mask=valid_mask):
            selected = value[mask]
            return float(selected.float().mean().detach().cpu()) if selected.numel() else 0.0

        metrics = {
            "pv/gate_mean": masked_mean(gate),
            "pv/gate_keep_rate": masked_mean((gate > 0).float()),
            "pv/same_direction_rate": masked_mean(same_direction.float()),
            "pv/a_fp_abs": masked_mean(a_fp.abs()),
            "pv/a_prec_abs": masked_mean(a_prec.abs()),
            "pv/adv_clip": float(adv_clip),
        }

        # The on-policy sampled reverse gap is a Monte Carlo KL estimate.
        positions = valid_mask.long().cumsum(dim=-1) - 1
        lengths = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        quartile = (4 * positions / lengths).long().clamp(0, 3)
        reverse_gap = -a_fp
        for index in range(4):
            pos_mask = valid_mask & (quartile == index)
            metrics[f"pv/sampled_kl_pos_q{index + 1}"] = masked_mean(
                reverse_gap, pos_mask
            )

        token_ids = labels[valid_mask].detach().cpu().tolist()
        token_strings = self.processing_class.convert_ids_to_tokens(token_ids)
        categories = {"number": [], "operator": [], "code": [], "text": []}
        valid_gate = gate[valid_mask].detach().float().cpu().tolist()
        operator_chars = set("+-*/%=<>")
        code_chars = set("{}[]();:_.,\\|&^~")
        for token, value in zip(token_strings, valid_gate):
            token = str(token)
            if any(char.isdigit() for char in token):
                category = "number"
            elif any(char in operator_chars for char in token):
                category = "operator"
            elif any(char in code_chars for char in token):
                category = "code"
            else:
                category = "text"
            categories[category].append(value)
        for category, values in categories.items():
            metrics[f"pv/gate_{category}"] = (
                sum(values) / len(values) if values else 0.0
            )
        self.log(metrics)

    def precision_verified_policy_loss(
        self,
        student_logp,
        teacher_logp,
        probe_logp,
        labels,
    ):
        """Current sampled reverse-KL surrogate weighted by the PV gate."""
        valid_mask = labels != -100
        with torch.no_grad():
            detached_student = student_logp.detach()
            a_fp = teacher_logp.detach() - detached_student
            a_prec = probe_logp.detach() - detached_student
            gate, same_direction = self.build_precision_gate(
                a_fp,
                a_prec,
                valid_mask,
                gate_mode=self.pv_gate_mode,
                gate_max=self.pv_gate_max,
            )
            adv_clip = self._update_pv_adv_clip(a_fp, valid_mask)
            reverse_advantage = (-a_fp).clamp(-adv_clip, adv_clip)
            self._log_pv_metrics(
                labels,
                a_fp,
                a_prec,
                gate,
                same_direction,
                valid_mask,
                adv_clip,
            )
        per_token_loss = gate * reverse_advantage * student_logp
        return per_token_loss[valid_mask].sum() / valid_mask.sum().clamp_min(1)

    @staticmethod
    def build_opd_masks(
        generated_tokens,
        prompt_attention_mask,
        eos_token_id,
        pad_token_id,
    ):
        """Mask prompt/padding while retaining the first generated EOS token."""
        prompt_length = prompt_attention_mask.shape[1]
        response_tokens = generated_tokens[:, prompt_length:]
        response_mask = torch.ones_like(response_tokens, dtype=torch.bool)

        if eos_token_id is not None and response_tokens.numel() > 0:
            eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
            is_eos = torch.zeros_like(response_tokens, dtype=torch.bool)
            for token_id in eos_ids:
                is_eos |= response_tokens == token_id
            eos_seen_before = is_eos.cumsum(dim=-1) - is_eos.to(torch.int64)
            response_mask &= eos_seen_before == 0

        eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
        if pad_token_id is not None and pad_token_id not in eos_ids:
            response_mask &= response_tokens != pad_token_id

        attention_mask = torch.cat(
            [prompt_attention_mask.to(torch.bool), response_mask],
            dim=1,
        ).to(torch.long)
        labels = generated_tokens.clone()
        labels[:, :prompt_length] = -100
        labels[:, prompt_length:].masked_fill_(~response_mask, -100)
        return attention_mask, labels

    def training_step(self, model, inputs, num_items_in_batch=None):
        """OPD: roll out the student before applying sampled reverse KL."""
        if self.opd_mode:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # train() + gradient checkpointing forces use_cache=False (O(T^2) decode).
                # Fake-quant also re-rounds full W every token unless cached.
                with opd_generate_context(unwrapped_model):
                    new_input_ids, _, _ = self.generate_on_policy_outputs(
                        unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                    )
            prompt_attention_mask = inputs.get(
                "prompt_attention_mask",
                torch.ones_like(inputs["prompts"]),
            )
            new_attention_mask, new_labels = self.build_opd_masks(
                generated_tokens=new_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                eos_token_id=self.generation_config.eos_token_id,
                pad_token_id=self.processing_class.pad_token_id,
            )
            inputs = dict(inputs)
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
            return super(GKDTrainer, self).training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Linear warmup from ``warmup_start_lr`` (default 0) to ``learning_rate``."""
        if optimizer is None:
            optimizer = self.optimizer
        warmup_steps = int(getattr(self.args, "warmup_steps", 0) or 0)
        start_lr = float(getattr(self, "warmup_start_lr", 0.0) or 0.0)
        peak_lr = float(self.args.learning_rate)
        if warmup_steps <= 0 or start_lr <= 0.0 or peak_lr <= 0.0:
            return super().create_scheduler(num_training_steps, optimizer=optimizer)

        start_factor = min(1.0, start_lr / peak_lr)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                progress = float(current_step) / float(max(1, warmup_steps))
                return start_factor + (1.0 - start_factor) * progress
            # Match HF linear schedule after warmup: decay to 0 by num_training_steps.
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - warmup_steps)),
            )

        from torch.optim.lr_scheduler import LambdaLR

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        self._created_lr_scheduler = True
        return self.lr_scheduler

    @staticmethod
    def cakld_loss(student_logits, teacher_logits, labels=None, beta_prob=0.5):
        mask = (labels != -100)

        # Clamp epsilon for numerical stability
        # Set to 1e-40 to cover observed minimum (5.57e-35) with safety margin
        eps = 1e-40

        # reverse
        teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)
        # Compute the softmax of the student's logits (approximate distribution)
        student_output_soft = F.softmax(student_logits, dim=2)
        # Clamp to prevent division by zero in gradient computation
        student_output_soft = student_output_soft.clamp(min=eps)
        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        reverse_kl = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="none").sum(-1)

        # forward
        student_output_log_prob = F.log_softmax(student_logits, dim=2)
        teacher_output_soft = F.softmax(teacher_logits, dim=2)
        # Clamp teacher probabilities as well for symmetry
        teacher_output_soft = teacher_output_soft.clamp(min=eps)
        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        forward_kl = F.kl_div(student_output_log_prob, teacher_output_soft, reduction="none").sum(-1)

        kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
        kl_loss *= mask
        # Use same reduction as generalized_jsd_loss: sum over all valid tokens, then divide by number of valid tokens
        kl_loss = kl_loss.sum() / mask.sum().clamp(min=1)
        return kl_loss

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean", top_k=None
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1) of https://arxiv.org/abs/2306.13649 for the definition.

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing loss
            beta: Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature: Softmax temperature (default: 1.0)
            reduction: Specifies the reduction to apply to the output (default: 'batchmean')
            top_k: If provided, only compute KL divergence on top-k logits from teacher model

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Original implementation - compute over all vocabulary
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)


        if top_k is not None and top_k > 0:
            # Memory-efficient approach: extract only top-k logits
            with torch.no_grad():
                # Get top-k indices from raw teacher logits (ordering preserved)
                _, top_k_indices = torch.topk(teacher_log_probs, top_k, dim=-1)
            
            # Extract only top-k logits for both student and teacher
            # Shape: [batch_size, seq_len, vocab_size] -> [batch_size, seq_len, top_k]
            batch_size, seq_len, vocab_size = teacher_logits.shape
            
            # Gather top-k logits
            student_top_k_logits = torch.gather(student_log_probs, -1, top_k_indices)
            teacher_top_k_logits = torch.gather(teacher_log_probs, -1, top_k_indices)

            student_log_probs = student_top_k_logits
            teacher_log_probs = teacher_top_k_logits
            # print(f"student_log_probs: {student_log_probs}, teacher_log_probs: {teacher_log_probs}")

        # Compute the interpolated log probabilities
        interpolated_log_probs = beta * student_log_probs + (1 - beta) * teacher_log_probs

        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        kl_teacher = F.kl_div(interpolated_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(interpolated_log_probs, student_log_probs, reduction="none", log_target=True)

        # Compute the Generalized Jensen-Shannon Divergence
        jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Sum over the last dimension (top-k for memory efficient mode, or vocab_size for full mode)
        jsd = jsd.sum(dim=-1)  # Sum over vocabulary dimension

        # Masking for padding tokens
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        need_ce = self.cross_entropy_weight > 0.0
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"] if need_ce and not (self.use_teacher_weight or self.use_dft_loss) else None,
            use_cache=False,
        )

        # compute teacher output in eval mode (BitDistiller style - no device movement)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        # Extract teacher logits before deleting teacher_outputs
        teacher_logits = teacher_outputs.get("logits")

        # Delete teacher_outputs to free memory (keeps only logits)
        del teacher_outputs

        # currently we only support next token prediction
        # OPD only gathers sampled-token log-probs, so views avoid duplicating
        # the large [batch, sequence, vocabulary] tensors.
        make_contiguous = not self.opd_mode
        shifted_student_logits = student_outputs.logits[:, :-1, :]
        shifted_teacher_logits = teacher_logits[:, :-1, :]
        if make_contiguous:
            shifted_student_logits = shifted_student_logits.contiguous()
            shifted_teacher_logits = shifted_teacher_logits.contiguous()
        shifted_labels = inputs["labels"][:, 1:].contiguous().to(shifted_student_logits.device)

        # Delete original logits to free memory
        del teacher_logits

        # KL divergence loss
        if self.kl_weight > 0.0:
            if self.pv_opd_mode:
                student_logp = self.sampled_action_log_probs(
                    shifted_student_logits,
                    shifted_labels,
                    temperature=self.temperature,
                )
                with torch.no_grad():
                    teacher_logp = self.sampled_action_log_probs(
                        shifted_teacher_logits,
                        shifted_labels,
                        temperature=self.temperature,
                    )
                # Free the teacher vocabulary logits before materializing the
                # W4 probe logits. The student graph remains for backward.
                del shifted_teacher_logits
                shifted_teacher_logits = None
                probe_model = (
                    self.accelerator.unwrap_model(model)
                    if hasattr(self, "accelerator")
                    else model
                )
                with torch.no_grad(), quantized_precision_view(
                    probe_model, self.pv_probe_bits
                ):
                    probe_outputs = probe_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                    )
                    probe_logp = self.sampled_action_log_probs(
                        probe_outputs.logits[:, :-1, :],
                        shifted_labels,
                        temperature=self.temperature,
                    )
                del probe_outputs
                kl_loss = self.precision_verified_policy_loss(
                    student_logp=student_logp,
                    teacher_logp=teacher_logp,
                    probe_logp=probe_logp,
                    labels=shifted_labels,
                )
                del student_logp, teacher_logp, probe_logp
            elif self.opd_mode:
                kl_loss = self.sampled_reverse_kl_policy_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    temperature=self.temperature,
                )
            elif self.kd_loss_type == "forward_kl":
                kl_loss = self.forward_kl_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    top_k=self.top_k,
                )
            elif self.kd_loss_type == "cakld":
                # Convert mean_prob to beta_prob for cakld_loss
                # mean_prob is the average max probability from teacher model
                beta_prob = self.mean_prob.item() if torch.is_tensor(self.mean_prob) else self.mean_prob
                kl_loss = self.cakld_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    beta_prob=beta_prob,
                )
            else:  # "jsd"
                kl_loss = self.generalized_jsd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    beta=self.beta,
                    top_k=self.top_k,
                )
        else:
            kl_loss = 0.0

        # RL like policy loss
        if self.use_dft_loss:
            # Use teacher logits as labels for DFT loss
            teacher_labels = torch.argmax(shifted_teacher_logits, dim=-1).contiguous().detach()
            cross_entropy_loss = dft_cross_entropy(
                shifted_student_logits.view(-1, shifted_student_logits.shape[-1]).float(),
                teacher_labels.view(-1).to(shifted_student_logits.device),
                shifted_student_logits.view(-1, shifted_student_logits.shape[-1]).float(),
                num_items_in_batch,
                ignore_index=-100
            )
            # Delete teacher_labels after use to free memory
            del teacher_labels
        elif self.use_teacher_weight:
            cross_entropy_loss = dft_cross_entropy(shifted_student_logits.view(-1, shifted_student_logits.shape[-1]).float(), shifted_labels.view(-1).to(shifted_student_logits.device), shifted_teacher_logits.view(-1, shifted_teacher_logits.shape[-1]).float(), num_items_in_batch, ignore_index=-100)
        else:
            cross_entropy_loss = student_outputs.loss

        # Delete intermediate tensors after all computations to free memory
        del shifted_student_logits
        if shifted_teacher_logits is not None:
            del shifted_teacher_logits
        del shifted_labels

        if need_ce:
            loss = self.cross_entropy_weight * cross_entropy_loss + self.kl_weight * kl_loss
        else:
            loss = self.kl_weight * kl_loss
        # loss = cross_entropy_loss

        # Delete student_outputs if not returning them
        if not return_outputs:
            del student_outputs

        # Return loss
        return (loss, student_outputs) if return_outputs else loss


@torch.no_grad()
def save_quantized_eval_checkpoint(model, tokenizer, save_dir):
    """Write a convert-ready snapshot without leaving training weights quantized.

    Matches the end-of-training save (fake-quant weights baked into ``.weight``)
    so ``convert_to_hf_vllm_compatible_model.py`` can load it. Master weights
    and ``use_weight_quant`` are restored before returning.
    """
    from quantize.int_linear_fake import QuantLinear

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    backups = []
    prev_states = []
    try:
        for module in model.modules():
            if isinstance(module, QuantLinear):
                backups.append((module, module.weight.data.detach().cpu().clone()))
                prev_states.append((module, module.use_weight_quant))
                quantized = module.weight_quantizer(module.weight.data)
                module.weight.data.copy_(quantized)
        set_quant_state(model, weight_quant=False)
        model.save_pretrained(str(save_path))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(save_path))
        (save_path / ".ready").touch()
    finally:
        for module, weight in backups:
            module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
        for module, was_quant in prev_states:
            module.use_weight_quant = was_quant
        if not prev_states:
            set_quant_state(model, weight_quant=True)


class QuantEvalSnapshotCallback(TrainerCallback):
    """Every ``save_steps`` optimizer steps, dump a vLLM-convertible checkpoint."""

    def __init__(self, save_quant_dir, tokenizer, save_steps):
        self.save_quant_dir = save_quant_dir
        self.tokenizer = tokenizer
        self.save_steps = int(save_steps)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.save_steps <= 0 or not self.save_quant_dir:
            return
        if state.global_step <= 0 or state.global_step % self.save_steps != 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if state.is_world_process_zero and model is not None:
            unwrapped = model.module if hasattr(model, "module") else model
            ckpt_dir = os.path.join(self.save_quant_dir, f"checkpoint-{state.global_step}")
            print(f"[save] quantized eval snapshot step={state.global_step} -> {ckpt_dir}", flush=True)
            save_quantized_eval_checkpoint(unwrapped, self.tokenizer, ckpt_dir)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    '''
    Note: evaluation simply move model to single GPU. 
    Therefor, to evaluate large model such as Llama-2-70B on single A100-80GB,
    please activate '--real_quant'.
irom torch
    '''
    # import pdb;pdb.set_trace()
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    return results

def format_s1k_sample(example):
    question = example.get("question", "")
    thinking_text = example.get("thinking_trajectories", "")[0]
    attempt = example.get("attempt", "")
    
    assistant_content = f"<think>\n{thinking_text}\n</think>\n\n{attempt}".rstrip()

    return {
        "messages": [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }

def format_json_sample(example):
    return {
        "messages": example["processed_messages"]
    }

def format_openthoughts_sample(example):
    """Convert OpenThoughts format to HuggingFace chat format"""
    messages = []
    for item in example:
        if item["from"] == "human":
            messages.append({
                "role": "user",
                "content": item["value"]
            })
        elif item["from"] == "gpt":
            messages.append({
                "role": "assistant",
                "content": item["value"]
            })
    return {
        "messages": messages
    }

def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from LaTeX \\boxed{} format.

    Args:
        text: Text containing \\boxed{answer}

    Returns:
        Extracted answer, or empty string if not found
    """
    import re

    # Pattern to match \boxed{...} with nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Return the last boxed answer (usually the final answer)
        return matches[-1].strip()

    return ""


def process_math_data(sample):
    """
    Process a math sample from OpenThoughts dataset.
    Extracts the problem and answer from \\boxed{} format.

    Args:
        sample: A dictionary containing the sample data with 'conversations' field.

    Returns:
        processed_sample: A dictionary with processed fields including:
            - messages: List of conversation messages
            - answer: The extracted answer from \\boxed{}
    """
    messages = []
    full_response = ""

    conversations = sample.get("conversations", [])

    for item in conversations:
        if item.get("from") == "human":
            messages.append({
                "role": "user",
                "content": item.get("value", "")
            })
        elif item.get("from") == "gpt":
            full_response = item.get("value", "")
            messages.append({
                "role": "assistant",
                "content": full_response
            })

    # Extract answer from \boxed{} format
    answer = extract_boxed_answer(full_response)

    return {
        "messages": messages,
        "answer": answer
    }

@dataclass
class CustomDataCollatorForChatML(DataCollatorForChatML):
    original_key = "conversations"

    def convert_to_messages(self, examples):
        messages = []
        for example in examples:
            messages.append(
                format_openthoughts_sample(example[self.original_key])
            )

        return messages

    def __call__(self, examples):
        messages = self.convert_to_messages(examples)
        return super().__call__(messages)

def add_nan_grad_hook(module, name=""):
    """
    勾配に NaN / Inf が含まれていないかチェックする backward hook を登録する
    """
    def _hook(mod, grad_input, grad_output):
        if grad_input is None:
            print(f"Grad for input is None in {name} grad_input is None")
            return
        for gi in grad_input:
            if gi is not None:
                print(f"{name} grad_input max={gi.max().item()} min={gi.min().item()} finite={torch.isfinite(gi).all()}")
            else:
                print(f"Grad for input is None in {name} grad_input is None")
        for go in grad_output:
            if go is not None:
                print(f"{name} grad_output max={go.max().item()} min={go.min().item()} finite={torch.isfinite(go).all()}")
            else:
                print(f"Grad for output is None in {name} grad_output is None")
    module.register_full_backward_hook(_hook)

def add_nan_grad_for_param(param, name=""):
    def _hook(grad):
        print(f"{name} grad max={grad.max().item()} min={grad.min().item()} finite={torch.isfinite(grad).all()}")
    
    param.register_hook(_hook)

def add_weight_monitor_hook(module, name=""):
    """
    Add forward hook to monitor weight maximum values during forward pass
    """
    def _forward_hook(module, input, output):
        if hasattr(module, 'weight') and module.weight is not None:
            weight_max = module.weight.max().item()
            weight_min = module.weight.min().item()
            scale_max = module.weight_quantizer.scale.max().item()
            scale_min = module.weight_quantizer.scale.min().item()
            weight_finite = torch.isfinite(module.weight).all().item()
            print(f"[WEIGHT MONITOR] {name} - max: {weight_max:.6f}, min: {weight_min:.6f}, scale_max: {scale_max:.6f}, scale_min: {scale_min:.6f}, finite: {weight_finite}")
            
            # Also check for NaN/Inf specifically
            if not weight_finite:
                nan_count = torch.isnan(module.weight).sum().item()
                inf_count = torch.isinf(module.weight).sum().item()
                total_params = module.weight.numel()
                print(f"[WEIGHT MONITOR] {name} - NaN: {nan_count}/{total_params}, Inf: {inf_count}/{total_params}")
    
    handle = module.register_forward_hook(_forward_hook)
    return handle

def scale_param_grad_hook(param, full_param_name):
    def _param_grad_hook(grad):
        nan_mask = torch.isnan(grad)
        grad = torch.where(nan_mask, torch.zeros_like(grad), grad)
        return grad

    return param.register_hook(_param_grad_hook)

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

def apply_chat_template(example, tokenizer):
    messages = example['messages'] # remote completion
    # add system message at first
    # messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    return {'messages': messages}


def validate_pv_trainable_scope(model):
    """Fail fast unless PV-OPD updates full weights plus clipping scales."""
    counts = {"master_weight": 0, "scale": 0, "other": 0}
    forbidden = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "zero_point" in name or "embed_tokens" in name:
            forbidden.append(name)
        if "weight_quantizer.scale" in name:
            counts["scale"] += parameter.numel()
        elif name.endswith(".weight"):
            counts["master_weight"] += parameter.numel()
        else:
            counts["other"] += parameter.numel()
    if forbidden:
        raise ValueError(f"PV-OPD forbidden trainable parameters: {forbidden[:5]}")
    if counts["master_weight"] == 0 or counts["scale"] == 0:
        raise ValueError(
            "PV-OPD FullPair requires trainable master weights and quantizer scales"
        )
    return counts


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--save_steps", type=int, default=0, help="Save a convert-ready quantized snapshot every N optimizer steps (0=final only)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Cap optimizer steps (-1=use epochs)")
    parser.add_argument("--calib_dataset",type=str,default="redpajama",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "random"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=4096, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="input sequence length for evaluating perplexity")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--wbits", type=int, default=4, help="weights quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")
    parser.add_argument("--quantizer_class", type=str, default="UniformAffineQuantizer", help="quantizer class to use (e.g., UniformAffineQuantizer, LogQuantizer)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=-1, help="Linear warmup steps; -1 keeps warmup_ratio=0.2")
    parser.add_argument("--warmup_start_lr", type=float, default=0.0, help="LR at step 0 when warmup_steps>0; 0 starts from 0")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="optimizer")
    parser.add_argument("--max_length", type=int, default=None, help="maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--dataset_size", type=int, default=1024, help="dataset size")
    parser.add_argument("--dataset_start_index", type=int, default=0, help="dataset start index")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="weight for KL divergence loss")
    parser.add_argument("--cross_entropy_weight", type=float, default=1.0, help="weight for cross entropy loss")
    parser.add_argument("--use_dft_loss", action="store_true", help="use DFT Causal LM Loss instead of default loss")
    parser.add_argument("--use_teacher_weight", action="store_true", help="use teacher weight for cross entropy loss")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-0.6B", help="teacher model name or path")
    parser.add_argument("--dataset_type", type=str, choices=["openthoughts", "openthoughts-math"], default="openthoughts", help="Dataset type: 'openthoughts' for open-thoughts/OpenThoughts3-1.2M, 'openthoughts-math' for open-thoughts/OpenThoughts3-1.2M filtered by math domain")
    parser.add_argument("--top_k", type=int, default=None, help="Use top-k logits from teacher model for KL divergence computation (None means use all logits)")
    parser.add_argument("--train_emb", action="store_true", help="Enable training of embedding tokens (embed_tokens). Default is False.")

    parser.add_argument("--kd_loss_type", type=str, default="jsd", choices=["jsd", "cakld", "forward_kl"], help="Knowledge distillation loss type: 'jsd' for generalized_jsd_loss, 'cakld' for cakld_loss, 'forward_kl' for OPD KL(teacher||student)")
    parser.add_argument("--opd", action="store_true", help="On-policy distillation: student rollouts + sampled reverse-KL policy gradient, matching slime OPD.")
    parser.add_argument("--pv_opd", action="store_true", help="PV-OPD FullPair: gate sampled reverse-KL with a shared-range precision probe.")
    parser.add_argument("--pv_probe_bits", type=int, default=4, help="Precision-probe bitwidth for PV-OPD.")
    parser.add_argument("--pv_gate_mode", choices=["full", "sign", "shuffled"], default="full", help="PV gate or gate ablation.")
    parser.add_argument("--pv_gate_max", type=float, default=2.0, help="Maximum normalized PV gate.")
    parser.add_argument("--pv_adv_clip", type=float, default=0.0, help="Fixed |A_FP| clip; 0 calibrates from warmup P99.")
    parser.add_argument("--pv_adv_clip_warmup_steps", type=int, default=10, help="Steps used to calibrate |A_FP| P99.")
    parser.add_argument("--cakld_steps", type=int, default=100, help="Number of steps to calculate mean probability for CAKLD loss")
    parser.add_argument("--enable_efficient_qat", action="store_true", help="Enable efficient QAT mode: only train scale parameters, freeze all other parameters")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # print all environment variables related to DDP
    print(f"RANK: {os.environ.get('RANK', -1)}, LOCAL_RANK: {os.environ.get('LOCAL_RANK', -1)}, WORLD_SIZE: {os.environ.get('WORLD_SIZE', -1)}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # ← 各プロセスは自分のGPUに固定
    device = torch.device("cuda", local_rank)
    # device name print
    args = parser.parse_args()
    if args.opd and args.pv_opd:
        parser.error("--opd and --pv_opd are mutually exclusive")
    if args.pv_opd and args.enable_efficient_qat:
        parser.error("PV-OPD FullPair cannot use --enable_efficient_qat")
    if args.pv_opd and args.pv_probe_bits <= args.wbits:
        parser.error("--pv_probe_bits must be greater than --wbits")
    if args.pv_opd and (
        args.cross_entropy_weight != 0.0
        or args.use_dft_loss
        or args.use_teacher_weight
    ):
        parser.error(
            "PV-OPD requires --cross_entropy_weight 0 and no auxiliary CE loss"
        )
    on_policy_mode = args.opd or args.pv_opd
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)

    # Current AdamW settings with NaN prevention:
    # GKD paper run: lmbda=0 offline JSD on dataset completions.
    # OPD: lmbda=1 student rollouts; max_new_tokens matches max_length budget.
    opd_gen_tokens = args.max_length if args.max_length else 8192
    use_fixed_warmup = args.warmup_steps is not None and args.warmup_steps >= 0
    training_args = GKDConfig(
        beta=1.0, # KL(student || teacher)
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        max_new_tokens=opd_gen_tokens if on_policy_mode else 128,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,  # More optimal for AdamW fine-tuning
        adam_epsilon=1e-6,  # Larger epsilon for bfloat16 stability
        adam_beta1=0.9,     # Default AdamW beta1
        adam_beta2=0.95,   # Default AdamW beta2
        output_dir=args.output_dir,
        save_strategy="no",
        gradient_checkpointing=True,
        seq_kd=False, # enforce supervised KD
        lmbda=1.0 if on_policy_mode else 0.0,
        temperature=0.6,
        logging_steps=1,
        warmup_steps=args.warmup_steps if use_fixed_warmup else 0,
        warmup_ratio=0.0 if use_fixed_warmup else 0.2,
        max_grad_norm=0.5,  # Reduced from 1.0 for more aggressive gradient clipping
        bf16=True,
        # fp16=False,
        optim=args.optim,  # Changed from sgd to adamw (Adam with weight decay)
        # optim=
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        # use_liger_kernel=True,
        # ddp_bucket_cap_mb=200,
    )

    # Log GPU memory before loading
    logger.info(f"[MEMORY] Rank {local_rank} - Before student model load - GPU {local_rank}: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB reserved")

    model, _ = load_quantized_model(args.model, args.wbits, args.group_size, strict=True, replace=True, quantizer_class=args.quantizer_class, device=device)

    logger.info(f"[MEMORY] Rank {local_rank} - After student model load - GPU {local_rank}: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB reserved")

    model.to("cpu")

    # Force GPU memory cleanup after moving to CPU
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    logger.info(f"[MEMORY] Rank {local_rank} - After student model to CPU and cleanup - GPU {local_rank}: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB reserved")

    set_quant_state(model, weight_quant=True)

    # Load teacher model on CPU first to avoid memory issues
    logger.info(f"[MEMORY] Rank {local_rank} - Before teacher model load")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=resolve_attn_implementation(True),
        device_map=None  # Keep on CPU for now
    )
    teacher_model = teacher_model.to("cpu").eval()
    logger.info(f"[MEMORY] Rank {local_rank} - After teacher model load (on CPU)")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    # Set pad tokens for both tokenizers
    tokenizer.pad_token = tokenizer.eos_token

    for name, param in model.named_parameters():
        # if ('embed_tokens' in name) or ('norm' in name):
        #     # print(f"set {name} to False")
        #     param.requires_grad = False
        # else:
        # if ('embed_tokens' in name) or ('zero_point' in name):

        if args.enable_efficient_qat:
            if 'scale' in name:
                param.requires_grad = True
                scale_param_grad_hook(param, name)
            else:
                param.requires_grad = False
        else:
            if 'zero_point' in name:
                param.requires_grad = False
            elif 'embed_tokens' in name:
                param.requires_grad = args.train_emb
            else:
                param.requires_grad = True
            if 'scale' in name:
                scale_param_grad_hook(param, name)
    if args.pv_opd:
        trainable_counts = validate_pv_trainable_scope(model)
        logger.info(f"PV-OPD FullPair trainable parameter counts: {trainable_counts}")

    # replace first layer into the ternary
    dev = model.device

    data_collator = None
    # Load dataset based on dataset_type argument
    if args.dataset_type == "openthoughts":
        dataset = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", verification_mode="no_checks")
        dataset = dataset.shuffle(seed=args.seed)
        logger.info(f"Using OpenThoughts dataset: open-thoughts/OpenThoughts3-1.2M")
        data_collator = CustomDataCollatorForChatML(tokenizer=tokenizer, max_length=args.max_length)

        end_index = args.dataset_start_index + args.dataset_size
        actual_end_index = min(end_index, len(dataset))
        dataset = dataset.select(range(args.dataset_start_index, actual_end_index))
    elif args.dataset_type == "openthoughts-math":
        logger.info(f"Loading OpenThoughts dataset from HuggingFace...")
        dataset = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", verification_mode="no_checks")

        total_cpus = os.cpu_count()
        num_proc = max(1, total_cpus // max(1, torch.cuda.device_count())) if torch.cuda.is_available() else total_cpus

        logger.info(f"Filtering for math domain (using {num_proc} CPUs)...")
        dataset = dataset.filter(
            lambda x: x.get("domain") == "math",
            num_proc=num_proc
        )

        logger.info(f"Processing math data to extract answers (using {num_proc} CPUs)...")
        dataset = dataset.map(
            process_math_data,
            num_proc=num_proc,
            desc="Processing math samples"
        )

        logger.info("Filtering out samples with empty answers...")
        original_size = len(dataset)
        dataset = dataset.filter(
            lambda x: x.get("answer", "") != "",
            num_proc=num_proc
        )
        filtered_size = len(dataset)
        logger.info(f"Filtered: {original_size} -> {filtered_size} samples ({original_size - filtered_size} removed)")

        dataset = dataset.shuffle(seed=args.seed)
        logger.info(f"Using OpenThoughts math dataset: open-thoughts/OpenThoughts3-1.2M (math domain)")
        data_collator = CustomDataCollatorForChatML(tokenizer=tokenizer, max_length=args.max_length)

        end_index = args.dataset_start_index + args.dataset_size
        actual_end_index = min(end_index, len(dataset))
        dataset = dataset.select(range(args.dataset_start_index, actual_end_index))

    logger.info(f"Dataset selection: start_index={args.dataset_start_index}, requested_size={args.dataset_size}")
    logger.info(f"Dataset selection: actual_end_index={actual_end_index}, selected_samples={len(dataset)}")
    # apply chat template to prompt
    # dataset = dataset.map(partial(apply_chat_template, tokenizer=tokenizer))
    # print(dataset[0]["messages"])
    # exit(0)
    # dataset = dataset.map(format_s1k_sample, remove_columns=["question", "thinking_trajectories", "attempt", "solution"])

    # Calculate mean_prob for CAKLD loss if needed
    mean_prob = 0
    if args.kd_loss_type == "cakld":
        logger.info("Calculating mean probability for CAKLD loss...")
        from torch.utils.data import DataLoader
        import torch.distributed as dist
        import gc

        # Determine correct device for this process
        if os.environ.get('LOCAL_RANK') is not None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            compute_device = torch.device(f"cuda:{local_rank}")
        else:
            compute_device = device

        # Log GPU memory before teacher model placement
        logger.info(f"[MEMORY] Rank {local_rank} - Before teacher to GPU {compute_device} - GPU {local_rank}: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB reserved")

        # Move teacher model to correct GPU for mean_prob calculation
        teacher_model = teacher_model.to(compute_device)

        # Verify teacher model is on correct GPU
        logger.info(f"[GPU VERIFICATION] Rank {local_rank} - Teacher model device: {next(teacher_model.parameters()).device}, Expected: {compute_device}")

        # Log memory after teacher model placement
        logger.info(f"[MEMORY] Rank {local_rank} - After teacher to GPU - GPU {local_rank}: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB reserved")

        # Check all GPUs to verify no cross-contamination
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            if gpu_id != local_rank and mem_allocated > 0.1:  # More than 100MB
                logger.warning(f"[GPU LEAK WARNING] Rank {local_rank} - GPU {gpu_id} has {mem_allocated:.2f} GB allocated (should be minimal)")

        probDataloader = DataLoader(
            dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
            drop_last=True,
        )

        prob = 0
        teacher_model.eval()  # Ensure eval mode
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for step, batch in tqdm(enumerate(probDataloader), desc="Computing mean_prob", disable=local_rank != 0 if os.environ.get('LOCAL_RANK') else False):
                if step >= args.cakld_steps:
                    break
                # Remove labels to avoid loss calculation
                input_batch = {
                    'input_ids': batch['input_ids'].to(compute_device),
                    'attention_mask': batch['attention_mask'].to(compute_device)
                }

                # Forward pass without loss calculation
                outputs = teacher_model(**input_batch)
                logits = outputs.logits  # Don't use .contiguous() yet

                # Calculate max probability efficiently
                prob1 = torch.nn.functional.softmax(logits, dim=-1)
                prob1 = torch.max(prob1, dim=-1).values
                prob += prob1.mean().cpu().item()  # Move to CPU immediately

                # Free memory
                del outputs, logits, prob1, input_batch
                torch.cuda.empty_cache()

        mean_prob = prob / min(step + 1, args.cakld_steps)
        mean_prob = torch.tensor(mean_prob, device=compute_device)

        # Synchronize across all processes if using distributed training
        if dist.is_initialized():
            dist.all_reduce(mean_prob, op=dist.ReduceOp.SUM)
            mean_prob = mean_prob / dist.get_world_size()

        logger.info(f"Calculated coefficient (mean_prob): {mean_prob.item()}")

        # Log memory after mean_prob calculation
        logger.info(f"[MEMORY] Rank {local_rank} - After mean_prob calculation - GPU {local_rank}: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB reserved")

        # Clean up DataLoader and data_collator
        del probDataloader
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"[MEMORY] Rank {local_rank} - After mean_prob cleanup - GPU {local_rank}: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB reserved")

    callbacks = []
    if args.save_steps and args.save_steps > 0 and args.save_quant_dir:
        callbacks.append(
            QuantEvalSnapshotCallback(args.save_quant_dir, tokenizer, args.save_steps)
        )
        logger.info(f"Quant eval snapshots every {args.save_steps} steps -> {args.save_quant_dir}/checkpoint-*")

    trainer = PolicyGKDTrainer(
        kl_weight=args.kl_weight,
        cross_entropy_weight=args.cross_entropy_weight,
        use_teacher_weight=args.use_teacher_weight,
        use_dft_loss=args.use_dft_loss,
        top_k=args.top_k,
        kd_loss_type=args.kd_loss_type,
        mean_prob=mean_prob,
        opd_mode=on_policy_mode,
        pv_opd_mode=args.pv_opd,
        pv_probe_bits=args.pv_probe_bits,
        pv_gate_mode=args.pv_gate_mode,
        pv_gate_max=args.pv_gate_max,
        pv_adv_clip=args.pv_adv_clip,
        pv_adv_clip_warmup_steps=args.pv_adv_clip_warmup_steps,
        model=model,
        teacher_model=teacher_model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks or None,
        # optimizers=(optimizer, None),
    )
    trainer.warmup_start_lr = float(args.warmup_start_lr)
    if use_fixed_warmup:
        logger.info(
            f"warmup_steps={args.warmup_steps} warmup_start_lr={args.warmup_start_lr} "
            f"peak_lr={args.learning_rate}"
        )
    if on_policy_mode:
        # Cap prompt+rollout at the same sequence budget as GKD (collator max_length).
        # Do not set max_new_tokens=8192: that would allow prompt_len + 8192 > GKD's 8192.
        trainer.generation_config.max_new_tokens = None
        trainer.generation_config.max_length = args.max_length or 8192
        trainer.generation_config.use_cache = True
        logger.info(
            f"{'PV-OPD' if args.pv_opd else 'OPD'} enabled: student rollout lmbda=1 "
            f"max_length={trainer.generation_config.max_length} kd_loss_type={args.kd_loss_type} "
            f"top_k={args.top_k} ce_weight={args.cross_entropy_weight} "
            f"probe_bits={args.pv_probe_bits if args.pv_opd else 'none'}"
        )

    if trainer.is_world_process_zero():
        print(trainer.model)
        print(trainer.teacher_model)
        print(trainer.accelerator.distributed_type)

    from transformers.trainer_utils import get_last_checkpoint
    #last_checkpoint = get_last_checkpoint(training_args.output_dir)
    last_checkpoint = None
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No valid checkpoint found. Starting training from scratch.")
        trainer.train()


    # save model
    torch.cuda.empty_cache()
    if trainer.is_world_process_zero():
        # アンラップしてからセーブ
        unwrapped_model = trainer.accelerator.unwrap_model(model)
        quant_inplace(unwrapped_model)
        set_quant_state(unwrapped_model, weight_quant=False)
        unwrapped_model.save_pretrained(args.save_quant_dir)
        tokenizer.save_pretrained(args.save_quant_dir)

    if trainer.is_world_process_zero() and (args.eval_ppl or args.eval_tasks):
       logger.info("Starting evaluation on main process...")
       evaluate(unwrapped_model, tokenizer, args, logger)

if __name__ == "__main__":
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        print(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'], os.environ['OMPI_COMM_WORLD_RANK'], os.environ['OMPI_COMM_WORLD_SIZE'], os.environ['LOCAL_RANK'], os.environ['RANK'], os.environ['WORLD_SIZE'])
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']
        os.environ['OMPI_COMM_WORLD_RANK'] = os.environ['RANK']
        #os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ['OMPI_COMM_WORLD_SIZE'] =  os.environ['WORLD_SIZE']
    print(sys.argv)
    main()
