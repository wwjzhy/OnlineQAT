import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import math
from contextlib import contextmanager
from logging import getLogger
import importlib

import numpy as np
import torch
import torch.nn as nn
import transformers

import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from tqdm import tqdm
import gc  
import os

logger = getLogger(__name__)


def resolve_attn_implementation(use_flash_attn=True):
    """Prefer FlashAttention-2; fall back to SDPA if the package is missing."""
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            logger.warning("flash_attn not installed; using attn_implementation=sdpa")
    return "sdpa"

# 循環インポートを避けるため、必要な関数を直接定義
def get_named_linears(module, type):
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)



class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64,
        quantizer_class="UniformAffineQuantizer",
        **quantizer_kwargs
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        # initialize quantizer dynamically
        self.weight_quantizer = self._create_quantizer(quantizer_class, wbits, group_size, org_module.weight, **quantizer_kwargs)
        self.use_temporary_parameter = False
        self._cached_weight = None
        self._precision_view_bits = None
        # self.output_scale = nn.Parameter([2.0], dtype=self.weight.dtype, device=self.weight.device)

    def _create_quantizer(self, quantizer_class, wbits, group_size, weight, **kwargs):
        """動的にquantizerクラスを作成する"""
        if isinstance(quantizer_class, str):
            # 文字列の場合は、quantize.quantizerモジュールからインポート
            try:
                module = importlib.import_module("quantize.quantizer")
                quantizer_cls = getattr(module, quantizer_class)
                return quantizer_cls(wbits, group_size, weight=weight, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to import {quantizer_class}, falling back to UniformAffineQuantizer: {e}")
                return UniformAffineQuantizer(wbits, group_size, weight=weight, **kwargs)
        elif hasattr(quantizer_class, '__call__'):
            # クラスオブジェクトの場合は直接インスタンス化
            return quantizer_class(wbits, group_size, weight=weight, **kwargs)
        else:
            logger.warning(f"Invalid quantizer_class: {quantizer_class}, falling back to UniformAffineQuantizer")
            return UniformAffineQuantizer(wbits, group_size, weight=weight, **kwargs)

    
    def quantized_weight(self):
        if self._precision_view_bits is not None:
            if not hasattr(self.weight_quantizer, "fake_quant_at_bits"):
                raise TypeError(
                    f"{type(self.weight_quantizer).__name__} does not support precision probes"
                )
            return self.weight_quantizer.fake_quant_at_bits(
                self.weight, self._precision_view_bits
            )
        return self.weight_quantizer(self.weight)

    def forward(self, input: torch.Tensor):
        if self._cached_weight is not None:
            weight = self._cached_weight
            bias = self.bias
        elif self.use_weight_quant:
            weight = self.quantized_weight()
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        return self.fwd_func(input, weight, bias, **self.fwd_kwargs)

    def cache_quantized_weight(self):
        """Snapshot fake-quant weights. Valid while master weights are frozen (generate)."""
        if self.use_weight_quant:
            with torch.no_grad():
                self._cached_weight = self.quantized_weight()
        else:
            self._cached_weight = None

    def clear_quantized_weight_cache(self):
        self._cached_weight = None

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.clear_quantized_weight_cache()


@contextmanager
def quantized_precision_view(model, n_bits):
    """Temporarily evaluate all QuantLinear modules on a shared-range grid."""
    modules = [m for m in model.modules() if isinstance(m, QuantLinear)]
    previous = [m._precision_view_bits for m in modules]
    try:
        for module in modules:
            module.clear_quantized_weight_cache()
            module._precision_view_bits = n_bits
        yield model
    finally:
        for module, old_bits in zip(modules, previous):
            module.clear_quantized_weight_cache()
            module._precision_view_bits = old_bits


@contextmanager
def freeze_fake_quant_for_generate(model):
    """Quantize each QuantLinear once for the generate loop, then drop the cache."""
    modules = [m for m in model.modules() if isinstance(m, QuantLinear)]
    for m in modules:
        m.cache_quantized_weight()
    try:
        yield
    finally:
        for m in modules:
            m.clear_quantized_weight_cache()


@contextmanager
def opd_generate_context(model):
    """eval() so KV cache is not disabled by gradient checkpointing; freeze fake-quant W."""
    was_training = model.training
    config = getattr(model, "config", None)
    prev_cache = getattr(config, "use_cache", None) if config is not None else None
    model.eval()
    if config is not None:
        config.use_cache = True
    try:
        with freeze_fake_quant_for_generate(model):
            yield
    finally:
        if config is not None and prev_cache is not None:
            config.use_cache = prev_cache
        if was_training:
            model.train()

def load_quantized_model_init(model_path, wbits, group_size, quantizer_class, use_flash_attn, scale=1.0, grad_scale=False):
    print(f"Initializing quantized model from {model_path}, {wbits}, {group_size}, {quantizer_class}")
    attn_impl = resolve_attn_implementation(use_flash_attn)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation=attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(module, wbits, group_size, quantizer_class, scale=scale, grad_scale=grad_scale)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    return model, tokenizer


def load_quantized_model(model_path, wbits, group_size, replace=False, strict=False, quantizer_class="UniformAffineQuantizer", scale=1.0, grad_scale=False, use_flash_attn=True, device=None):
    print(f"Loading quantized model from {model_path}")
    if not os.path.exists(model_path):
        model, tokenizer = load_quantized_model_init(model_path, wbits, group_size, quantizer_class, use_flash_attn)
        return model, tokenizer

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        attn_impl = resolve_attn_implementation(use_flash_attn)
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True, attn_implementation=attn_impl)
    layers = model.model.layers
    print(f"Loading models with {wbits} {group_size}")
    if replace:
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            named_linears = get_named_linears(layer, torch.nn.Linear)
            for name, module in named_linears.items():
                q_linear = QuantLinear(module, wbits, group_size, quantizer_class, scale=scale, grad_scale=grad_scale)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
    else:
        pass
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    # if device is None:
    if device is not None:
        device_map = {"": "cpu"}
    else:
        device_map = infer_auto_device_map(model)
    # else:
    # device_map = {"": device}
    # device_map = infer_auto_device_map(model)
    print(f"initialize model: {model}")
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=False, strict=strict)
    # load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True, strict=strict)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer

__all__ = ["QuantLinear", "quantized_precision_view"]

