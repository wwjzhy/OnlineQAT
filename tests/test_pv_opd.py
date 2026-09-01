"""CPU unit tests for the PV-OPD FullPair path."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn
import torch.nn.functional as F

from main_e2e_distill import PolicyGKDTrainer, validate_pv_trainable_scope
from quantize.int_linear_fake import QuantLinear, quantized_precision_view
from quantize.quantizer import UniformAffineQuantizer


ROOT = Path(__file__).resolve().parents[1]


def test_w2_w4_probe_shares_real_clipping_range():
    weight = torch.linspace(-2.0, 1.0, 16).reshape(2, 8)
    quantizer = UniformAffineQuantizer(2, 8, weight=weight)
    target_scale = quantizer.scale.detach()
    target_zp = quantizer.zero_point.detach().round()
    target_min = -target_zp * target_scale
    target_max = (3 - target_zp) * target_scale

    probe_scale = target_scale * (3 / 15)
    probe_zp = target_zp * (15 / 3)
    probe_min = -probe_zp * probe_scale
    probe_max = (15 - probe_zp) * probe_scale

    assert torch.allclose(target_min, probe_min, atol=1e-7)
    assert torch.allclose(target_max, probe_max, atol=1e-7)
    w2 = quantizer(weight)
    w4 = quantizer.fake_quant_at_bits(weight.float(), 4)
    assert F.mse_loss(w4, weight) <= F.mse_loss(w2, weight)


def test_precision_view_restores_target_bits_and_cache():
    torch.manual_seed(0)
    model = nn.Sequential(QuantLinear(nn.Linear(8, 8, bias=False), 2, 8))
    model[0].set_quant_state(True)
    x = torch.randn(2, 8)
    target_before = model(x)
    with quantized_precision_view(model, 4):
        assert model[0]._precision_view_bits == 4
        probe = model(x)
    assert model[0]._precision_view_bits is None
    assert model[0]._cached_weight is None
    target_after = model(x)
    assert torch.allclose(target_before, target_after)
    assert not torch.allclose(target_before, probe)


def test_gate_drops_opposite_direction_tokens():
    a_fp = torch.tensor([[1.0, 1.0, -2.0, -2.0]])
    a_prec = torch.tensor([[0.5, -0.5, -1.0, 1.0]])
    mask = torch.ones_like(a_fp, dtype=torch.bool)
    gate, same = PolicyGKDTrainer.build_precision_gate(
        a_fp, a_prec, mask, gate_mode="full", gate_max=2.0
    )
    assert same.tolist() == [[True, False, True, False]]
    assert gate[0, 1] == 0
    assert gate[0, 3] == 0
    assert 0 < float(gate[mask].mean()) <= 1.0
    assert float(gate.max()) <= 2.0


def test_chunked_fp32_action_logp_matches_reference():
    torch.manual_seed(2)
    logits = torch.randn(2, 4, 37, dtype=torch.bfloat16, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1, 2, -100, 4], [3, 5, 6, 7]])
    got = PolicyGKDTrainer.sampled_action_log_probs(
        logits, labels, temperature=0.6, vocab_chunk_size=8
    )
    safe = labels.masked_fill(labels == -100, 0).unsqueeze(-1)
    expected = F.log_softmax(reference_logits.float() / 0.6, dim=-1).gather(
        -1, safe
    ).squeeze(-1)
    assert torch.allclose(got, expected, atol=1e-5)
    got[labels != -100].mean().backward()
    expected[labels != -100].mean().backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, reference_logits.grad, atol=2e-3)


def test_sign_and_shuffled_gate_preserve_support_size():
    a_fp = torch.tensor([[1.0, 2.0, 3.0, -1.0]])
    a_prec = torch.tensor([[0.1, 1.0, -1.0, -0.5]])
    mask = torch.ones_like(a_fp, dtype=torch.bool)
    sign, _ = PolicyGKDTrainer.build_precision_gate(
        a_fp, a_prec, mask, gate_mode="sign"
    )
    torch.manual_seed(3)
    shuffled, _ = PolicyGKDTrainer.build_precision_gate(
        a_fp, a_prec, mask, gate_mode="shuffled"
    )
    assert int((sign > 0).sum()) == 3
    assert int((shuffled > 0).sum()) == 3


def _bare_pv_trainer():
    trainer = PolicyGKDTrainer.__new__(PolicyGKDTrainer)
    trainer.pv_gate_mode = "full"
    trainer.pv_gate_max = 2.0
    trainer.pv_adv_clip = 2.0
    trainer.pv_adv_clip_warmup_steps = 0
    trainer._pv_adv_clip_estimates = []
    trainer._pv_adv_clip_value = 2.0
    trainer._pv_clip_last_step = None
    trainer._pv_metrics_last_step = 0
    trainer.state = SimpleNamespace(global_step=0)
    return trainer


def test_full_pair_receives_weight_and_scale_gradients():
    torch.manual_seed(4)
    layer = QuantLinear(nn.Linear(8, 6, bias=False), 2, 8)
    layer.set_quant_state(True)
    layer.weight_quantizer.zero_point.requires_grad = False
    x = torch.randn(2, 3, 8)
    labels = torch.tensor([[1, 2, 3], [3, 4, 5]])
    logits = layer(x)
    student_logp = F.log_softmax(logits.float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    teacher_logp = student_logp.detach() + 1.0
    probe_logp = student_logp.detach() + 0.5

    trainer = _bare_pv_trainer()
    loss = PolicyGKDTrainer.precision_verified_policy_loss(
        trainer, student_logp, teacher_logp, probe_logp, labels
    )
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.weight_quantizer.scale.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.weight_quantizer.scale.grad).all()
    assert layer.weight_quantizer.zero_point.grad is None


def test_compute_loss_runs_target_teacher_and_probe_paths():
    class TinyQuantLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.head = QuantLinear(nn.Linear(8, 16, bias=False), 2, 8)
            self.head.set_quant_state(True)
            self.head.weight_quantizer.zero_point.requires_grad = False

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            logits = self.head(self.embed(input_ids))
            return type(
                "Output",
                (),
                {
                    "logits": logits,
                    "loss": None,
                    "get": lambda self, key: getattr(self, key),
                },
            )()

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16, bias=False)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            logits = self.head(self.embed(input_ids))
            return type(
                "Output",
                (),
                {"logits": logits, "get": lambda self, key: getattr(self, key)},
            )()

    model = TinyQuantLM()
    trainer = _bare_pv_trainer()
    trainer.cross_entropy_weight = 0.0
    trainer.kl_weight = 1.0
    trainer.use_teacher_weight = False
    trainer.use_dft_loss = False
    trainer.pv_opd_mode = True
    trainer.opd_mode = True
    trainer.pv_probe_bits = 4
    trainer.temperature = 1.0
    trainer.teacher_model = TinyTeacher()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4, 5]]),
    }
    loss = PolicyGKDTrainer.compute_loss(trainer, model, inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.head.weight_quantizer.scale.grad is not None
    assert model.head._precision_view_bits is None


def test_trainable_scope_accepts_full_pair_and_rejects_scale_only():
    model = nn.Sequential(QuantLinear(nn.Linear(8, 8, bias=False), 2, 8))
    model[0].weight_quantizer.zero_point.requires_grad = False
    counts = validate_pv_trainable_scope(model)
    assert counts["master_weight"] > 0
    assert counts["scale"] > 0

    model[0].weight.requires_grad = False
    try:
        validate_pv_trainable_scope(model)
    except ValueError as error:
        assert "master weights" in str(error)
    else:
        raise AssertionError("scale-only PV-OPD should be rejected")


def test_pv_script_is_isolated_and_has_expected_defaults():
    script = (ROOT / "scripts" / "run_qwen3_1.7b_pv_opd.sh").read_text()
    assert "--pv_opd" in script
    assert 'WBITS=2' in script
    assert 'PROBE_BITS="${PV_PROBE_BITS:-4}"' in script
    assert 'MAX_STEPS="${MAX_STEPS:-100}"' in script
    assert 'SAVE_STEPS="${SAVE_STEPS:-5}"' in script
    assert "-pv-opd" in script
    assert "--enable_efficient_qat" not in script
    assert "--train_emb" not in script


if __name__ == "__main__":
    tests = [
        test_w2_w4_probe_shares_real_clipping_range,
        test_precision_view_restores_target_bits_and_cache,
        test_gate_drops_opposite_direction_tokens,
        test_chunked_fp32_action_logp_matches_reference,
        test_sign_and_shuffled_gate_preserve_support_size,
        test_full_pair_receives_weight_and_scale_gradients,
        test_compute_loss_runs_target_teacher_and_probe_paths,
        test_trainable_scope_accepts_full_pair_and_rejects_scale_only,
        test_pv_script_is_isolated_and_has_expected_defaults,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK {len(tests)} tests")
