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
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_fake import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
from trl import GKDConfig, GKDTrainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache
from datasets import load_dataset
import copy
import quantize.int_linear_fake as int_linear_fake
from quantize.utils import set_op_by_name, set_quant_state, quant_inplace
import types
from functools import partial
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.trainer_utils import get_last_checkpoint

torch.backends.cudnn.benchmark = True

from trl.trainer.utils import DataCollatorForChatML


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
    def __init__(self, kl_weight=1.0, cross_entropy_weight=1.0, use_teacher_weight=False, use_dft_loss=False, top_k=None, kd_loss_type="jsd", mean_prob=0, beta=0.5, opd_mode=False, *args, **kwargs):
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        """OPD: always roll out the student, then JSD (CE optional). GKD keeps lmbda=0 offline path."""
        if self.opd_mode:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            prompt_len = inputs["prompts"].shape[1]
            new_labels = new_labels.clone()
            new_labels[:, :prompt_len] = -100
            inputs = dict(inputs)
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
            return super(GKDTrainer, self).training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch)

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
            labels=inputs["labels"] if need_ce and not (self.use_teacher_weight or self.use_dft_loss) else None
        )

        # compute teacher output in eval mode (BitDistiller style - no device movement)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # Extract teacher logits before deleting teacher_outputs
        teacher_logits = teacher_outputs.get("logits")

        # Delete teacher_outputs to free memory (keeps only logits)
        del teacher_outputs

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]

        # currently we only support next token prediction
        shifted_student_logits = student_outputs.logits[:, :-1, :].contiguous()
        shifted_teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shifted_labels = inputs["labels"][:, 1:].contiguous().to(shifted_student_logits.device)

        # Delete original logits to free memory
        del teacher_logits

        # KL divergence loss
        if self.kl_weight > 0.0:
            if self.kd_loss_type == "forward_kl":
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

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

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

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
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
    parser.add_argument("--opd", action="store_true", help="On-policy distillation: student rollouts + forward KL. Same data/hparams as GKD; only the Stage-2 state source and KL form change.")
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
    training_args = GKDConfig(
        beta=1.0, # KL(student || teacher)
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        max_new_tokens=opd_gen_tokens if args.opd else 128,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,  # More optimal for AdamW fine-tuning
        adam_epsilon=1e-6,  # Larger epsilon for bfloat16 stability
        adam_beta1=0.9,     # Default AdamW beta1
        adam_beta2=0.95,   # Default AdamW beta2
        output_dir=args.output_dir,
        gradient_checkpointing=True,
        seq_kd=False, # enforce supervised KD
        lmbda=1.0 if args.opd else 0.0,
        temperature=0.6,
        logging_steps=1,
        warmup_ratio=0.2,   # Added warmup to stabilize training
        max_grad_norm=0.5,  # Reduced from 1.0 for more aggressive gradient clipping
        bf16=True,
        # fp16=False,
        optim=args.optim,  # Changed from sgd to adamw (Adam with weight decay)
        # optim=
        debug="underflow_overflow",
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
        attn_implementation="flash_attention_2",
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

    trainer = PolicyGKDTrainer(
        kl_weight=args.kl_weight,
        cross_entropy_weight=args.cross_entropy_weight,
        use_teacher_weight=args.use_teacher_weight,
        use_dft_loss=args.use_dft_loss,
        top_k=args.top_k,
        kd_loss_type=args.kd_loss_type,
        mean_prob=mean_prob,
        opd_mode=args.opd,
        model=model,
        teacher_model=teacher_model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,
        # optimizers=(optimizer, None),
    )
    if args.opd:
        # Cap prompt+rollout at the same sequence budget as GKD (collator max_length).
        # Do not set max_new_tokens=8192: that would allow prompt_len + 8192 > GKD's 8192.
        trainer.generation_config.max_new_tokens = None
        trainer.generation_config.max_length = args.max_length or 8192
        trainer.generation_config.use_cache = True
        logger.info(
            f"OPD enabled: student rollout lmbda=1 max_length={trainer.generation_config.max_length} "
            f"kd_loss_type={args.kd_loss_type} top_k={args.top_k} ce_weight={args.cross_entropy_weight}"
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
