#!/usr/bin/env python3
"""
Generate Tsubame distillation scripts for different parameter combinations.
"""

import os
import argparse
import re
from itertools import product

def extract_quantization_params(model_path):
    """Extract wbits and group_size from model path containing wxgxxx pattern."""
    # Look for pattern like w2g128, w4g64, etc.
    pattern = r'w(\d+)g(\d+)'
    match = re.search(pattern, model_path)

    if match:
        wbits = int(match.group(1))
        group_size = int(match.group(2))
        return wbits, group_size
    else:
        # Default values if pattern not found
        print(f"Warning: Could not extract quantization parameters from {model_path}")
        print("Using default values: wbits=2, group_size=128")
        return 2, 128

def detect_quantizer_class(model_path):
    """Detect quantizer class from model path containing -slt or -sltzp patterns."""
    model_path_lower = model_path.lower()

    if '-sltzp' in model_path_lower:
        return 'SLTQuantizerZP'
    elif '-slt' in model_path_lower:
        return 'SLTQuantizer'
    else:
        # Default quantizer class
        return 'UniformAffineQuantizer'

def format_sequence_count(seq_count):
    """Convert sequence count to k notation (e.g., 8192 -> 8k)."""
    if seq_count % 1024 == 0:
        return f"{seq_count // 1024}k"
    else:
        return str(seq_count)

def create_tsubame_script(model_path, base_model_name, teacher_model, learning_rate, use_teacher_weight, use_dft_loss, kl_weight, cross_entropy_weight, dataset_size, dataset_type, min_difficulty, top_k, epochs, train_emb, enable_efficient_qat, kd_loss_type, cakld_steps, output_dir):
    """Create a Tsubame script with specified parameters."""

    # Extract quantization parameters from model path
    wbits, group_size = extract_quantization_params(model_path)

    # Detect quantizer class from model path
    quantizer_class = detect_quantizer_class(model_path)

    # Format dataset size for file names
    dataset_size_str = format_sequence_count(dataset_size)

    # Generate script name
    lr_str = f"lr{learning_rate}".replace(".", "_").replace("-", "_")
    teacher_str = "teacher" if use_teacher_weight else "noteacher"
    dft_str = "dft" if use_dft_loss else "nodft"
    kl_str = f"kl{kl_weight}".replace(".", "_")
    ce_str = f"ce{cross_entropy_weight}".replace(".", "_")
    epoch_str = f"ep{epochs}"
    top_k_str = f"_topk{top_k}" if top_k is not None else ""
    train_emb_str = "_trainemb" if train_emb else "_notrainemb"
    efficient_qat_str = "_effqat" if enable_efficient_qat else "_noeffqat"
    kd_loss_str = f"_{kd_loss_type}"
    cakld_steps_str = f"_cakldsteps{cakld_steps}" if kd_loss_type == "cakld" else ""

    if dataset_type == "generated":
        dataset_str = "_generatedQwen8B"
    elif dataset_type == "openthoughts":
        difficulty_str = f"_diff{min_difficulty}" if min_difficulty is not None else ""
        dataset_str = f"_openthoughts{difficulty_str}"
    elif dataset_type == "openthoughts-math":
        dataset_str = "_openthoughts_math"
    else:
        dataset_str = "_original"

    script_name = f"{base_model_name}_{dataset_size_str}_{lr_str}_{epoch_str}_{teacher_str}_{dft_str}_{kl_str}_{ce_str}{top_k_str}{train_emb_str}{efficient_qat_str}{kd_loss_str}{cakld_steps_str}{dataset_str}.sh"

    # Generate output directory names
    save_quant_dir = f"output/distill_sweep/{base_model_name}_{dataset_size_str}_{lr_str}_{epoch_str}_{teacher_str}_{dft_str}_{kl_str}_{ce_str}{top_k_str}{train_emb_str}{efficient_qat_str}{kd_loss_str}{cakld_steps_str}{dataset_str}"
    log_dir = f"output/sft_log/{base_model_name}_{dataset_size_str}_{lr_str}_{epoch_str}_{teacher_str}_{dft_str}_{kl_str}_{ce_str}{top_k_str}{train_emb_str}{efficient_qat_str}{kd_loss_str}{cakld_steps_str}{dataset_str}"
    
    # Base command arguments (without accelerate launch part)
    cmd_args = [
        f"--model {model_path}",
        f"--teacher_model {teacher_model}",
        f"--wbits {wbits}",
        f"--group_size {group_size}",
        f"--quantizer_class {quantizer_class}",
        f"--save_quant_dir {save_quant_dir}",
        f"--learning_rate {learning_rate}",
        "--max_length 10000",
        f"--output_dir {log_dir}",
        "--eval_ppl",
        f"--dataset_size {dataset_size}",
        f"--epochs {epochs}",
        "--gradient_accumulation_steps 4",
        f"--kl_weight {kl_weight}",
        f"--cross_entropy_weight {cross_entropy_weight}",
        f"--dataset_type {dataset_type}"
    ]
    
    # Add min_difficulty parameter if using openthoughts dataset
    if dataset_type == "openthoughts" and min_difficulty is not None:
        cmd_args.append(f"--min_difficulty {min_difficulty}")

    # Add teacher weight flag if needed
    if use_teacher_weight:
        cmd_args.append("--use_teacher_weight")

    # Add DFT loss flag if needed
    if use_dft_loss:
        cmd_args.append("--use_dft_loss")

    # Add top_k parameter if specified
    if top_k is not None:
        cmd_args.append(f"--top_k {top_k}")

    # Add train_emb flag if needed
    if train_emb:
        cmd_args.append("--train_emb")

    # Add enable_efficient_qat flag if needed
    if enable_efficient_qat:
        cmd_args.append("--enable_efficient_qat")

    # Add kd_loss_type parameter
    cmd_args.append(f"--kd_loss_type {kd_loss_type}")

    # Add cakld_steps parameter if using cakld
    if kd_loss_type == "cakld":
        cmd_args.append(f"--cakld_steps {cakld_steps}")

    # Create script content with cluster header
    cluster_header = """#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -m abe
#$ -M okoshi.yasuyuki@artic.iir.titech.ac.jp
#$ -o log/basic_distill/
#$ -e log/basic_distill/

source ~/.bashrc
source ~/.vllmrc
conda activate efficientqat
export PYTHONPATH=$PWD:$PYTHONPATH

"""
    
    script_content = cluster_header

    # Add multinode/single node detection and accelerate launch branching
    script_content += """# Detect multinode environment and set appropriate accelerate launch command
if [ -n "$NUM_PROCESSES" ] && [ -n "$NUM_MACHINE" ] && [ -n "$MASTER_ADDR" ] && [ -n "$MASTER_PORT" ] && [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    # Multinode environment
    echo "Detected multinode environment"
    ACCELERATE_CMD="accelerate launch --config_file config/fsdp_config.yaml \\
--num_processes $NUM_PROCESSES \\
--num_machines $NUM_MACHINE \\
--main_process_ip $MASTER_ADDR \\
--main_process_port $MASTER_PORT \\
--machine_rank $OMPI_COMM_WORLD_RANK \\
--gpu_ids 0,1,2,3"
else
    # Single node environment
    echo "Detected single node environment"
    ACCELERATE_CMD="accelerate launch --num_processes=4"
fi

# Run distillation
$ACCELERATE_CMD main_e2e_distill.py \\
"""

    script_content += " \\\n".join(cmd_args) + "\n"
    
    return script_name, script_content

def main():
    """Generate all Tsubame distillation scripts."""
    
    parser = argparse.ArgumentParser(description="Generate Tsubame distillation scripts")
    parser.add_argument("--model_path", required=True,
                       help="Path to the model (e.g., output/block_ap_models/block_match/Qwen3-1.7B-w2g128-residual-next2-nblock2/)")
    parser.add_argument("--teacher_model", required=True,
                       help="Teacher model path (e.g., Qwen/Qwen3-1.7B)")
    parser.add_argument("--dataset_sizes", nargs="+", type=int, default=[32768],
                       help="List of dataset sizes (e.g., 8192 16384 32768)")
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[1e-6, 5e-6],
                       help="List of learning rates (e.g., 1e-6 5e-6)")
    parser.add_argument("--kl_weights", nargs="+", type=float, default=[0.0, 0.1, 0.05],
                       help="List of KL weights (e.g., 0.0 0.1 0.05)")
    parser.add_argument("--cross_entropy_weights", nargs="+", type=float, default=[1.0],
                       help="List of cross entropy weights (e.g., 1.0 0.5 0.1)")
    parser.add_argument("--output_dir", default="tsubame_scripts/distillation",
                       help="Output directory for scripts")
    parser.add_argument("--dataset_type", choices=["original", "generated", "openthoughts", "openthoughts-math"], default="generated",
                       help="Dataset type: 'original' for open-r1/Mixture-of-Thoughts, 'generated' for local qwen3-8B-generated.jsonl, 'openthoughts' for open-thoughts/OpenThoughts3-1.2M, 'openthoughts-math' for open-thoughts/OpenThoughts3-1.2M filtered by math domain")
    parser.add_argument("--min_difficulties", nargs="+", type=float, default=[None],
                       help="List of minimum difficulty thresholds for openthoughts dataset (e.g., 1.0 2.0 3.0). Use 'None' for no filtering.")
    parser.add_argument("--top_k_values", nargs="+", type=int, default=[None],
                       help="List of top-k values for teacher logits selection (e.g., 100 500 1000). Use 'None' for no top-k filtering.")
    parser.add_argument("--use_teacher_weights", nargs="+", type=lambda x: x.lower() == 'true', default=[False],
                       help="List of use_teacher_weight flags (e.g., false true). Default: [false]")
    parser.add_argument("--use_dft_losses", nargs="+", type=lambda x: x.lower() == 'true', default=[False],
                       help="List of use_dft_loss flags (e.g., false true). Default: [false]")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs (default: 1)")
    parser.add_argument("--train_emb_options", nargs="+", type=lambda x: x.lower() == 'true', default=[False],
                       help="List of train_emb flags (e.g., false true). Default: [false]")
    parser.add_argument("--enable_efficient_qat_options", nargs="+", type=lambda x: x.lower() == 'true', default=[False],
                       help="List of enable_efficient_qat flags (e.g., false true). Default: [false]")
    parser.add_argument("--kd_loss_types", nargs="+", type=str, default=["jsd"],
                       choices=["jsd", "cakld"],
                       help="List of KD loss types (e.g., jsd cakld). Default: [jsd]")
    parser.add_argument("--cakld_steps_values", nargs="+", type=int, default=[100],
                       help="List of cakld_steps values (e.g., 50 100 200). Default: [100]")


    args = parser.parse_args()
    
    # Extract base model name from path
    base_model_name = os.path.basename(args.model_path.rstrip('/'))
    
    # Parameter combinations from arguments
    use_teacher_weights = args.use_teacher_weights
    use_dft_losses = args.use_dft_losses
    train_emb_options = args.train_emb_options
    enable_efficient_qat_options = args.enable_efficient_qat_options

    # Process min_difficulties - convert string 'None' to actual None
    min_difficulties = []
    for diff in args.min_difficulties:
        if isinstance(diff, str) and diff.lower() == 'none':
            min_difficulties.append(None)
        else:
            min_difficulties.append(diff)
    
    # Process top_k_values - convert string 'None' to actual None
    top_k_values = []
    for top_k in args.top_k_values:
        if isinstance(top_k, str) and top_k.lower() == 'none':
            top_k_values.append(None)
        else:
            top_k_values.append(top_k)
    
    # Create output directory
    output_base_dir = os.path.join(args.output_dir, base_model_name)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Also create log directory
    os.makedirs("log", exist_ok=True)
    
    # Generate all combinations, ensuring use_teacher_weights and use_dft_losses are mutually exclusive
    if args.dataset_type == "openthoughts":
        all_combinations = list(product(args.dataset_sizes, args.learning_rates, use_teacher_weights, use_dft_losses, args.kl_weights, args.cross_entropy_weights, min_difficulties, top_k_values, train_emb_options, enable_efficient_qat_options, args.kd_loss_types, args.cakld_steps_values))
        combinations = [(ds, lr, teacher, dft, kl, ce, diff, top_k, train_emb, eff_qat, kd_loss, cakld_steps) for ds, lr, teacher, dft, kl, ce, diff, top_k, train_emb, eff_qat, kd_loss, cakld_steps in all_combinations if not (teacher and dft)]
    else:
        all_combinations = list(product(args.dataset_sizes, args.learning_rates, use_teacher_weights, use_dft_losses, args.kl_weights, args.cross_entropy_weights, top_k_values, train_emb_options, enable_efficient_qat_options, args.kd_loss_types, args.cakld_steps_values))
        combinations = [(ds, lr, teacher, dft, kl, ce, None, top_k, train_emb, eff_qat, kd_loss, cakld_steps) for ds, lr, teacher, dft, kl, ce, top_k, train_emb, eff_qat, kd_loss, cakld_steps in all_combinations if not (teacher and dft)]
    
    print(f"Generating {len(combinations)} Tsubame scripts...")
    print(f"Model: {args.model_path}")
    print(f"Teacher Model: {args.teacher_model}")
    print(f"Base Model Name: {base_model_name}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Dataset sizes: {[format_sequence_count(ds) for ds in args.dataset_sizes]}")
    print(f"Learning rates: {args.learning_rates}")
    print(f"Use teacher weights: {use_teacher_weights}")
    print(f"Use DFT losses: {use_dft_losses}")
    print(f"Train embedding options: {train_emb_options}")
    print(f"Enable efficient QAT options: {enable_efficient_qat_options}")
    print(f"KL weights: {args.kl_weights}")
    print(f"Cross entropy weights: {args.cross_entropy_weights}")
    print(f"KD loss types: {args.kd_loss_types}")
    print(f"CAKLD steps values: {args.cakld_steps_values}")
    if args.dataset_type == "openthoughts":
        print(f"Min difficulties: {min_difficulties}")
    print(f"Top-k values: {top_k_values}")
    print()

    for dataset_size, lr, use_teacher, use_dft, kl, ce, min_difficulty, top_k, train_emb, eff_qat, kd_loss, cakld_steps in combinations:
        script_name, script_content = create_tsubame_script(
            args.model_path, base_model_name, args.teacher_model, lr, use_teacher, use_dft, kl, ce, dataset_size, args.dataset_type, min_difficulty, top_k, args.epochs, train_emb, eff_qat, kd_loss, cakld_steps, output_base_dir
        )
        
        script_path = os.path.join(output_base_dir, script_name)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"Created: {script_path}")
    
    print(f"\nAll {len(combinations)} scripts created successfully!")
    print(f"Scripts location: {output_base_dir}/")

if __name__ == "__main__":
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        print(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'], os.environ['OMPI_COMM_WORLD_RANK'], os.environ['OMPI_COMM_WORLD_SIZE'], os.environ['LOCAL_RANK'], os.environ['RANK'], os.environ['WORLD_SIZE'])
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']
        os.environ['OMPI_COMM_WORLD_RANK'] = os.environ['RANK']
        #os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ['OMPI_COMM_WORLD_SIZE'] =  os.environ['WORLD_SIZE']
    main() 