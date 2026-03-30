#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import textwrap
from typing import List, Optional
import torch

from quantize.int_linear_fake import load_quantized_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-id", type=str, required=True,
                    help="HF repo id またはローカルディレクトリ")
    ap.add_argument("--save-dir", type=str, required=True,
                    help="保存先ディレクトリ")
    ap.add_argument("--exclude-names", type=str, default="lm_head",
                    help="置換対象名のカンマ区切り。空文字/未指定で全 Linear")
    ap.add_argument("--wbits", type=int, required=True,
                    help="量子化ビット幅（必須）")
    ap.add_argument("--group-size", type=int, required=True,
                    help="グループサイズ（必須）")
    ap.add_argument("--extra", action="append", default=[],
                    help="追加引数（任意）。key=val を複数回指定可。例: --extra enable_weight_quant=true --extra foo=1.2")
    ap.add_argument("--dtype", type=str, default=None,
                    help="from_pretrained の torch_dtype（例: 'float16'）")
    ap.add_argument("--quantizer-class", type=str, default=None,
                    help="量子化クラス名（省略時は model name から自動決定）")
    return ap.parse_args()

def _module_path_from_model_type(model_type: str) -> str:
    return f"transformers.models.{model_type}.modeling_{model_type}"

def _sanitize_name_list(arg: Optional[str]) -> Optional[List[str]]:
    if arg is None:
        return None
    if arg.strip() == "":
        return None
    return [s.strip() for s in arg.split(",") if s.strip()]

def _auto_cast(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "none" or low == "null":
        return None
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        return float(s)
    except ValueError:
        return s

def _parse_extra(kvs):
    d = {}
    for kv in (kvs or []):
        if "=" not in kv:
            raise ValueError(f"--extra は key=val 形式で指定してください: {kv}")
        k, v = kv.split("=", 1)
        d[k.strip()] = _auto_cast(v.strip())
    return d

def _auto_detect_quantizer_class(model_path):
    """モデル名に基づいてquantizer_classを自動決定"""
    model_name = model_path.lower()

    # ディレクトリ名やパスから model name を抽出
    if "/" in model_path:
        model_name = model_path.split("/")[-1].lower()

    # SLT系の判定
    if "-slt" in model_name and "zp" in model_name:
        return "SLTQuantizerZP"
    elif "-slt" in model_name:
        return "SLTQuantizer"
    else:
        return "UniformAffineQuantizer"

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # quantizer_class を決定
    quantizer_class = args.quantizer_class
    if quantizer_class is None:
        quantizer_class = _auto_detect_quantizer_class(args.base_id)
        print(f"Auto-detected quantizer class: {quantizer_class}")
    else:
        print(f"Using specified quantizer class: {quantizer_class}")

    # 1) モデル/トークナイザをロード
    dtype_kwargs = {}
    if args.dtype:
        dtype_kwargs["torch_dtype"] = getattr(torch, args.dtype)

    model, tok = load_quantized_model(args.base_id, args.wbits, args.group_size, use_flash_attn=False, quantizer_class=quantizer_class)

    # convert dtype to bfloat16
    model.to(torch.bfloat16)
    
    # update config.json
    config = json.load(open(os.path.join(args.base_id, "config.json")))
    config["torch_dtype"] = "bfloat16"

    # 3) 保存（重み・tokenizer・config）
    model.save_pretrained(args.save_dir, safe_serialization=True)
    tok.save_pretrained(args.save_dir)
    json.dump(config, open(os.path.join(args.save_dir, "config.json"), "w"))

if __name__ == "__main__":
    main()
