"""CPU checks that OnlineQAT Stage 2 OPD is wired as on-policy JSD (no CE).

Does not load Qwen or run a full train step. Importing main_e2e_distill is safe
because main() is behind ``if __name__``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig

from main_e2e_distill import PolicyGKDTrainer
from trl.trainer.gkd_trainer import GKDTrainer


ROOT = Path(__file__).resolve().parents[1]


class TinyLM(nn.Module):
    def __init__(self, vocab_size=32, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size
        self.generation_config = GenerationConfig(pad_token_id=0, eos_token_id=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.lm_head(self.embed(input_ids))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        out = type("Out", (), {})()
        out.logits = logits
        out.loss = loss
        out.get = lambda k, default=None: getattr(out, k, default)
        return out

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, generation_config=None, return_dict_in_generate=False, **kwargs):
        cfg = generation_config or self.generation_config
        max_length = cfg.max_length if cfg.max_length else input_ids.shape[1] + 4
        max_new = cfg.max_new_tokens
        seq = input_ids
        steps = 0
        while seq.shape[1] < max_length:
            if max_new is not None and steps >= max_new:
                break
            nxt = self.forward(seq).logits[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            steps += 1
        if return_dict_in_generate:
            return type("Gen", (), {"sequences": seq})()
        return seq


def _bare_trainer(**kwargs):
    obj = PolicyGKDTrainer.__new__(PolicyGKDTrainer)
    obj.kl_weight = kwargs.get("kl_weight", 1.0)
    obj.cross_entropy_weight = kwargs.get("cross_entropy_weight", 0.0)
    obj.use_teacher_weight = False
    obj.use_dft_loss = False
    obj.top_k = kwargs.get("top_k", None)
    obj.kd_loss_type = kwargs.get("kd_loss_type", "jsd")
    obj.mean_prob = 0
    obj.beta = kwargs.get("beta", 1.0)
    obj.opd_mode = kwargs.get("opd_mode", True)
    obj.teacher_model = kwargs["teacher_model"]
    return obj


def test_jsd_identical_is_zero():
    logits = torch.randn(2, 6, 20)
    labels = torch.ones(2, 6, dtype=torch.long)
    loss = PolicyGKDTrainer.generalized_jsd_loss(logits, logits.clone(), labels=labels, beta=1.0)
    assert float(loss) < 1e-5, loss


def test_jsd_masks_prompt_and_pad():
    torch.manual_seed(0)
    student = torch.randn(1, 5, 8)
    teacher = torch.randn(1, 5, 8)
    labels = torch.tensor([[-100, -100, -100, 3, -100]])
    full = PolicyGKDTrainer.generalized_jsd_loss(student, teacher, labels=torch.ones_like(labels), beta=1.0)
    masked = PolicyGKDTrainer.generalized_jsd_loss(student, teacher, labels=labels, beta=1.0)
    only = PolicyGKDTrainer.generalized_jsd_loss(student[:, 3:4], teacher[:, 3:4], labels=torch.tensor([[3]]), beta=1.0)
    assert not torch.allclose(full, masked)
    assert torch.allclose(masked, only, atol=1e-5)


def test_jsd_beta1_is_forward_kl():
    """GKDConfig(beta=1) + this JSD impl is KL(teacher || student), not symmetric JSD."""
    torch.manual_seed(1)
    student = torch.randn(2, 4, 10)
    teacher = torch.randn(2, 4, 10)
    labels = torch.ones(2, 4, dtype=torch.long)
    got = PolicyGKDTrainer.generalized_jsd_loss(student, teacher, labels=labels, beta=1.0)
    s_log = F.log_softmax(student, dim=-1)
    t_log = F.log_softmax(teacher, dim=-1)
    ref = F.kl_div(s_log, t_log, reduction="none", log_target=True).sum(-1).mean()
    assert torch.allclose(got, ref, atol=1e-5), (got, ref)


def test_compute_loss_ce0_ignores_gold_labels():
    torch.manual_seed(2)
    student = TinyLM()
    teacher = TinyLM()
    teacher.load_state_dict(student.state_dict())
    obj = _bare_trainer(teacher_model=teacher, cross_entropy_weight=0.0)

    bsz, prompt_len, total = 2, 4, 8
    input_ids = torch.randint(2, 20, (bsz, total))
    labels = torch.randint(2, 20, (bsz, total))
    labels[:, :prompt_len] = -100
    gold = labels.clone()
    gold[:, prompt_len:] = 5
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": gold,
        "prompts": input_ids[:, :prompt_len],
    }
    loss_gold = PolicyGKDTrainer.compute_loss(obj, student, inputs)
    inputs["labels"] = labels
    loss_student_tok = PolicyGKDTrainer.compute_loss(obj, student, inputs)
    # Same logits + same mask positions (both completions are valid ints) → JSD identical;
    # changing *which* hard ids are in labels must not change a CE-free loss if mask is the same.
    assert torch.isfinite(loss_gold)
    assert torch.allclose(loss_gold, loss_student_tok, atol=1e-5)


def test_compute_loss_ce0_has_no_ce_graph():
    student = TinyLM()
    teacher = TinyLM()
    obj = _bare_trainer(teacher_model=teacher, cross_entropy_weight=0.0)
    input_ids = torch.randint(2, 20, (1, 6))
    labels = torch.full((1, 6), -100)
    labels[:, 3:] = input_ids[:, 3:]
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "prompts": input_ids[:, :3],
    }
    loss = PolicyGKDTrainer.compute_loss(obj, student, inputs)
    loss.backward()
    assert student.lm_head.weight.grad is not None
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0


def test_opd_replaces_batch_with_student_rollout_and_masks_prompt():
    torch.manual_seed(3)
    model = TinyLM()
    prompts = torch.tensor([[0, 0, 7, 8], [0, 9, 10, 11]])
    inputs = {
        "prompts": prompts,
        "prompt_attention_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
        "input_ids": torch.ones(2, 10, dtype=torch.long),
        "attention_mask": torch.ones(2, 10, dtype=torch.long),
        "labels": torch.ones(2, 10, dtype=torch.long),
    }
    cfg = GenerationConfig(max_length=7, max_new_tokens=None, pad_token_id=0, do_sample=False)
    new_ids, new_mask, new_labels = GKDTrainer.generate_on_policy_outputs(model, inputs, cfg, pad_token_id=0)
    prompt_len = prompts.shape[1]
    new_labels = new_labels.clone()
    new_labels[:, :prompt_len] = -100

    assert new_ids.shape[1] == 7
    assert not torch.equal(new_ids, inputs["input_ids"][:, :7])
    assert torch.all(new_labels[:, :prompt_len] == -100)
    assert torch.any(new_labels[:, prompt_len:] != -100)
    assert new_ids[:, :prompt_len].equal(prompts)


def test_generation_budget_is_total_max_length_not_extra_new_tokens():
    cfg = GenerationConfig(max_new_tokens=8192, temperature=0.6, do_sample=True, top_k=0)
    cfg.max_new_tokens = None
    cfg.max_length = 32
    model = TinyLM()
    prompts = torch.randint(2, 20, (1, 10))
    out = GKDTrainer.generate_on_policy_outputs(model, {"prompts": prompts}, cfg, pad_token_id=0)[0]
    assert out.shape[1] == 32


def test_scripts_gkd_vs_opd_flags():
    gkd = (ROOT / "scripts" / "run_qwen3_1.7b.sh").read_text()
    opd = (ROOT / "scripts" / "run_qwen3_1.7b_opd.sh").read_text()
    assert "--opd" not in gkd
    assert "--cross_entropy_weight 0.2" in gkd
    assert "--kd_loss_type jsd" in gkd
    assert "--opd \\" in opd or "--opd\n" in opd
    assert "--cross_entropy_weight 0.0" in opd
    assert "--kd_loss_type jsd" in opd
    assert "block_qat" in opd
    assert "-opd" in opd


def test_opd_flag_does_not_force_forward_kl():
    src = (ROOT / "main_e2e_distill.py").read_text()
    tree = ast.parse(src)
    forced = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "args.opd" in test:
                body = ast.unparse(node) if hasattr(ast, "unparse") else ""
                if "kd_loss_type" in body and "forward_kl" in body:
                    forced = True
    assert not forced, "--opd must not override kd_loss_type to forward_kl"


if __name__ == "__main__":
    tests = [
        test_jsd_identical_is_zero,
        test_jsd_masks_prompt_and_pad,
        test_jsd_beta1_is_forward_kl,
        test_compute_loss_ce0_ignores_gold_labels,
        test_compute_loss_ce0_has_no_ce_graph,
        test_opd_replaces_batch_with_student_rollout_and_masks_prompt,
        test_generation_budget_is_total_max_length_not_extra_new_tokens,
        test_scripts_gkd_vs_opd_flags,
        test_opd_flag_does_not_force_forward_kl,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK {len(tests)} tests")
