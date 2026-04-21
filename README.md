# Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs

Official implementation for the paper:

**Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs**
Yasuyuki Okoshi, Hikari Otsuka, Daichi Fujiki, Masato Motomura
*The Fourteenth International Conference on Learning Representations (ICLR 2026)*
[[OpenReview]](https://openreview.net/forum?id=Azsd2qyK6C)

## Overview

This repository provides a two-stage quantization-aware training (QAT) pipeline for ultra-low-bit reasoning LLMs. We address the severe performance degradation that standard QAT causes on reasoning benchmarks (e.g., mathematics) and instruction-following tasks.

**Pipeline:**
1. **Block-wise QAT** (`main_block_qat.py`): Block-wise quantization-aware training with mixed-domain calibration data to preserve essential capabilities across domains.
2. **End-to-end Distillation** (`main_e2e_distill.py`): Knowledge distillation from a teacher model using GKD (Generalized Knowledge Distillation) to restore and enhance reasoning capability.

## Installation

```bash
git clone https://github.com/yasu0001/ReasoningQAT.git
cd ReasoningQAT

conda create -n reasoningqat python=3.11
conda activate reasoningqat

pip install -r requirements.txt
```

## Usage

### Step 1: Block-wise QAT

Block-wise quantization-aware training with mixed-domain calibration.

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_qat.py \
    --model <path-to-base-model> \
    --wbits 2 \
    --group_size 128 \
    --calib_dataset sweep_0.8 \
    --train_size 4096 \
    --val_size 64 \
    --epochs 2 \
    --batch_size 2 \
    --weight_lr 2e-5 \
    --quant_lr 1e-4 \
    --save_quant_dir ./output/block_qat/<model-name>-w2g128 \
    --output_dir ./log/block_qat/<model-name>-w2g128
```

Key arguments:
- `--wbits`: Quantization bit width (e.g., 2 for 2-bit)
- `--group_size`: Quantization group size (e.g., 64, 128)
- `--calib_dataset`: Calibration data mixing ratio (`sweep_0.0` to `sweep_1.0`)
- `--weight_lr`: Learning rate for full-precision weights (`2e-5` for 2-bit, `1e-5` for 3-/4-bit)
- `--nblock`: Number of blocks to quantize simultaneously (default: 1)

### Step 2: End-to-end Distillation

Knowledge distillation from a teacher model to the quantized student.

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model <path-to-block-qat-output> \
    --teacher_model <path-to-teacher-model> \
    --wbits 2 \
    --group_size 128 \
    --epochs 2 \
    --learning_rate 1e-6 \
    --kl_weight 0.01 \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --max_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_quant_dir ./output/distill/<model-name> \
    --output_dir ./log/distill/<model-name>
```

Key arguments:
- `--teacher_model`: Path to the teacher model for knowledge distillation
- `--kl_weight`: Weight for KL divergence loss
- `--use_dft_loss`: Enable DFT (difficulty-weighted) cross-entropy loss
- `--use_teacher_weight`: Use teacher logits for cross-entropy weighting
- `--kd_loss_type`: KD loss type (`jsd` or `cakld`)
- `--dataset_type`: Training data (`openthoughts` or `openthoughts-math`)

### Step 3: Convert to vLLM-compatible Format

```bash
python scripts/convert_to_hf_vllm_compatible_model.py \
    --input_dir <path-to-distill-output> \
    --output_dir <path-to-vllm-model>
```

### Step 4: Evaluation

Generate evaluation scripts for cluster execution:

```bash
python scripts/generate_vllm_eval_tsubame_scripts.py \
    --model_path <path-to-vllm-model> \
    --output_dir ./eval_scripts
```

## Project Structure

```
.
├── main_block_qat.py          # Stage 1: Block-wise QAT
├── main_e2e_distill.py        # Stage 2: End-to-end distillation
├── datautils_block.py         # Data loading and preprocessing
├── utils.py                   # General utilities
├── configs/
│   └── accelerate_config.yaml # Multi-GPU training config
├── scripts/
│   ├── convert_to_hf_vllm_compatible_model.py
│   ├── generate_tsubame_distill_scripts.py
│   └── generate_vllm_eval_tsubame_scripts.py
└── quantize/
    ├── block_qat.py           # Block-wise QAT implementation
    ├── int_linear_fake.py     # Fake quantization linear layer
    ├── int_linear_real.py     # Real quantization linear layer
    ├── quantizer.py           # Quantizer implementations
    ├── utils.py               # Quantization utilities
    └── triton_utils/          # Triton kernel implementations
```

## Paper Reproduction Configs

The following commands reproduce the exact hyperparameters reported in the paper for each model and bit-width combination.

### Stage 1: Block-wise QAT

Common settings across all models:
- Calibration samples: 4,096
- Context length: 2,048
- Calibration data: OpenThoughts-1.2M 80% + FineWeb-Edu 20% (`sweep_0.8`)
- Group size: 128
- Quantization parameter LR (scale, zero point): 1e-4

#### Qwen3-1.7B W3

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_qat.py \
    --model Qwen/Qwen3-1.7B \
    --wbits 3 \
    --group_size 128 \
    --calib_dataset sweep_0.8 \
    --train_size 4096 \
    --val_size 64 \
    --training_seqlen 2048 \
    --epochs 2 \
    --batch_size 2 \
    --weight_lr 1e-5 \
    --quant_lr 1e-4 \
    --save_quant_dir ./output/block_qat/Qwen3-1.7B-w3g128 \
    --output_dir ./log/block_qat/Qwen3-1.7B-w3g128
```

#### Qwen3-1.7B W2

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_qat.py \
    --model Qwen/Qwen3-1.7B \
    --wbits 2 \
    --group_size 128 \
    --calib_dataset sweep_0.8 \
    --train_size 4096 \
    --val_size 64 \
    --training_seqlen 2048 \
    --epochs 2 \
    --batch_size 2 \
    --weight_lr 2e-5 \
    --quant_lr 1e-4 \
    --save_quant_dir ./output/block_qat/Qwen3-1.7B-w2g128 \
    --output_dir ./log/block_qat/Qwen3-1.7B-w2g128
```

#### Qwen3-4B W3

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_qat.py \
    --model Qwen/Qwen3-4B \
    --wbits 3 \
    --group_size 128 \
    --calib_dataset sweep_0.8 \
    --train_size 4096 \
    --val_size 64 \
    --training_seqlen 2048 \
    --epochs 2 \
    --batch_size 2 \
    --weight_lr 1e-5 \
    --quant_lr 1e-4 \
    --save_quant_dir ./output/block_qat/Qwen3-4B-w3g128 \
    --output_dir ./log/block_qat/Qwen3-4B-w3g128
```

#### Qwen3-4B W2

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_qat.py \
    --model Qwen/Qwen3-4B \
    --wbits 2 \
    --group_size 128 \
    --calib_dataset sweep_0.8 \
    --train_size 4096 \
    --val_size 64 \
    --training_seqlen 2048 \
    --epochs 2 \
    --batch_size 2 \
    --weight_lr 2e-5 \
    --quant_lr 1e-4 \
    --save_quant_dir ./output/block_qat/Qwen3-4B-w2g128 \
    --output_dir ./log/block_qat/Qwen3-4B-w2g128
```

#### Qwen3-8B W3

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_qat.py \
    --model Qwen/Qwen3-8B \
    --wbits 3 \
    --group_size 128 \
    --calib_dataset sweep_0.8 \
    --train_size 4096 \
    --val_size 64 \
    --training_seqlen 2048 \
    --epochs 2 \
    --batch_size 2 \
    --weight_lr 1e-5 \
    --quant_lr 1e-4 \
    --save_quant_dir ./output/block_qat/Qwen3-8B-w3g128 \
    --output_dir ./log/block_qat/Qwen3-8B-w3g128
```

#### Qwen3-8B W2

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_qat.py \
    --model Qwen/Qwen3-8B \
    --wbits 2 \
    --group_size 128 \
    --calib_dataset sweep_0.8 \
    --train_size 4096 \
    --val_size 64 \
    --training_seqlen 2048 \
    --epochs 2 \
    --batch_size 2 \
    --weight_lr 2e-5 \
    --quant_lr 1e-4 \
    --save_quant_dir ./output/block_qat/Qwen3-8B-w2g128 \
    --output_dir ./log/block_qat/Qwen3-8B-w2g128
```

### Stage 2: End-to-end Distillation

Common settings across all models:
- Training data: 32,768 samples from OpenThoughts-1.2M
- Batch size: 64 (per_device_train_batch_size × gradient_accumulation_steps × num_gpus)
- Optimizer: AdamW
- LR scheduler: Cosine annealing decay
- KL loss: Top-20 probability filtered (`--top_k 20`)
- Cross-entropy weight α: 0.2 (`--cross_entropy_weight 0.2`)
- KL divergence weight β: 1.0 (`--kl_weight 1.0`)
- KD loss type: JSD (`--kd_loss_type jsd`)

> **Note**: Adjust `--per_device_train_batch_size` and `--gradient_accumulation_steps` to achieve an effective batch size of 64 based on your GPU count and memory. The examples below assume 4 GPUs with `per_device_train_batch_size=1` and `gradient_accumulation_steps=16`.

> **`--train_emb`**: When enabled, the embedding layer (`embed_tokens`) is also trained during distillation. Each config below provides both variants. Use `--train_emb` if you observe vocabulary-level degradation; omit it for the default (frozen embeddings).

#### Qwen3-1.7B W3

Without `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-1.7B-w3g128 \
    --teacher_model Qwen/Qwen3-1.7B \
    --wbits 3 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-1.7B-w3g128 \
    --output_dir ./log/distill/Qwen3-1.7B-w3g128
```

With `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-1.7B-w3g128 \
    --teacher_model Qwen/Qwen3-1.7B \
    --wbits 3 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --train_emb \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-1.7B-w3g128-trainemb \
    --output_dir ./log/distill/Qwen3-1.7B-w3g128-trainemb
```

#### Qwen3-1.7B W2

Without `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-1.7B-w2g128 \
    --teacher_model Qwen/Qwen3-1.7B \
    --wbits 2 \
    --group_size 128 \
    --epochs 3 \
    --learning_rate 5e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-1.7B-w2g128 \
    --output_dir ./log/distill/Qwen3-1.7B-w2g128
```

With `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-1.7B-w2g128 \
    --teacher_model Qwen/Qwen3-1.7B \
    --wbits 2 \
    --group_size 128 \
    --epochs 3 \
    --learning_rate 5e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --train_emb \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-1.7B-w2g128-trainemb \
    --output_dir ./log/distill/Qwen3-1.7B-w2g128-trainemb
```

#### Qwen3-4B W3

Without `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-4B-w3g128 \
    --teacher_model Qwen/Qwen3-4B \
    --wbits 3 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-4B-w3g128 \
    --output_dir ./log/distill/Qwen3-4B-w3g128
```

With `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-4B-w3g128 \
    --teacher_model Qwen/Qwen3-4B \
    --wbits 3 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --train_emb \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-4B-w3g128-trainemb \
    --output_dir ./log/distill/Qwen3-4B-w3g128-trainemb
```

#### Qwen3-4B W2

Without `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-4B-w2g128 \
    --teacher_model Qwen/Qwen3-4B \
    --wbits 2 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-4B-w2g128 \
    --output_dir ./log/distill/Qwen3-4B-w2g128
```

With `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-4B-w2g128 \
    --teacher_model Qwen/Qwen3-4B \
    --wbits 2 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --train_emb \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-4B-w2g128-trainemb \
    --output_dir ./log/distill/Qwen3-4B-w2g128-trainemb
```

#### Qwen3-8B W3

Without `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-8B-w3g128 \
    --teacher_model Qwen/Qwen3-8B \
    --wbits 3 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-8B-w3g128 \
    --output_dir ./log/distill/Qwen3-8B-w3g128
```

With `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-8B-w3g128 \
    --teacher_model Qwen/Qwen3-8B \
    --wbits 3 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --train_emb \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-8B-w3g128-trainemb \
    --output_dir ./log/distill/Qwen3-8B-w3g128-trainemb
```

#### Qwen3-8B W2

Without `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-8B-w2g128 \
    --teacher_model Qwen/Qwen3-8B \
    --wbits 2 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-8B-w2g128 \
    --output_dir ./log/distill/Qwen3-8B-w2g128
```

With `--train_emb`:
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    main_e2e_distill.py \
    --model ./output/block_qat/Qwen3-8B-w2g128 \
    --teacher_model Qwen/Qwen3-8B \
    --wbits 2 \
    --group_size 128 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.2 \
    --kd_loss_type jsd \
    --top_k 20 \
    --train_emb \
    --dataset_type openthoughts \
    --dataset_size 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_quant_dir ./output/distill/Qwen3-8B-w2g128-trainemb \
    --output_dir ./log/distill/Qwen3-8B-w2g128-trainemb
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
okoshi2026towards,
title={Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning {LLM}s},
author={Yasuyuki Okoshi and Hikari Otsuka and Daichi Fujiki and Masato Motomura},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Azsd2qyK6C}
}
```

## Acknowledgments

This codebase builds upon [EfficientQAT](https://github.com/OpenGVLab/EfficientQAT) (Chen et al., 2024).
