#!/usr/bin/env python3
"""
VLLM eval用のtsubameスクリプトを自動生成するプログラム
eval/yamls/{base_model}/*.yamlの設定ファイルを基に、tsubame/vllm/{base_model}/*.shを生成
"""

import os
import glob
from pathlib import Path

def get_model_serve_name(model_path):
    return model_path.split("/")[-1]

def extract_billion_param(model_name):
    """モデル名からbillion parameterを抽出 (例: Qwen3-4B -> 4, Qwen3-1.7B -> 1.7)"""
    import re
    # モデル名から数値B部分を抽出
    match = re.search(r'-(\d+(?:\.\d+)?)B', model_name)
    if match:
        return match.group(1)
    return "unknown"

def _discover_qwen3_models():
    """output/vllm/models/ディレクトリからQwen3-*で始まるモデルを自動検出"""
    vllm_models_dir = "output/vllm/"
    if not os.path.exists(vllm_models_dir):
        print(f"Warning: {vllm_models_dir} ディレクトリが存在しません")
        return []

    # Qwen3-*で始まるディレクトリを再帰的に検索
    qwen3_pattern = os.path.join(vllm_models_dir, "**", "Qwen3-*")
    qwen3_models = glob.glob(qwen3_pattern, recursive=True)

    # ディレクトリのみをフィルタリング
    qwen3_models = [model for model in qwen3_models if os.path.isdir(model)]

    # パスを正規化
    qwen3_models = [os.path.normpath(model) for model in qwen3_models]

    print(f"自動検出されたQwen3モデル: {len(qwen3_models)}個")
    for model in qwen3_models:
        print(f"  - {model}")

    return qwen3_models

def extract_path_components(model_path):
    """モデルパスからdirname, base_name, model_nameを抽出"""
    # output/vllm/sweep/Qwen3-1.7B-w3g128-block2-sweep02 のようなパスを想定
    # 出力構造: vllm/{dirname}/{model_name}
    path_parts = model_path.split('/')

    # vllm/ 以降の部分を取得
    try:
        vllm_idx = path_parts.index('vllm')

        # vllm/{dirname}/{model_name} の構造を想定
        if vllm_idx + 2 < len(path_parts):
            # output/vllm/{dirname}/{model_name} の場合
            dirname = path_parts[vllm_idx + 1]  # vllmの次がdirname (sweep, models等)
            model_name = path_parts[-1]  # 最後の部分がmodel_name
        elif vllm_idx + 1 < len(path_parts):
            # output/vllm/{model_name} の場合（dirnameなし）
            dirname = ""
            model_name = path_parts[-1]
        else:
            # パスが短すぎる場合
            dirname = ""
            model_name = path_parts[-1]

        # base_nameは billion parameter を抽出して {base_model}-{billion_param}B 形式
        billion_param = extract_billion_param(model_name)

        # base_modelを推定 (Qwen3, DeepSeek-R1, Llama-3 など)
        if model_name.startswith('Qwen3'):
            base_model = 'Qwen3'
        elif 'DeepSeek-R1' in model_name:
            base_model = 'DeepSeek-R1'
        elif 'Llama-3' in model_name:
            base_model = 'Llama-3'
        else:
            base_model = 'Unknown'

        base_name = f"{base_model}-{billion_param}B"

        return dirname, base_name, model_name

    except (ValueError, IndexError):
        # パスの解析に失敗した場合はデフォルト値を返す
        model_name = model_path.split('/')[-1]
        billion_param = extract_billion_param(model_name)

        if model_name.startswith('Qwen3'):
            base_model = 'Qwen3'
        elif 'DeepSeek-R1' in model_name:
            base_model = 'DeepSeek-R1'
        elif 'Llama-3' in model_name:
            base_model = 'Llama-3'
        else:
            base_model = 'Unknown'

        base_name = f"{base_model}-{billion_param}B"
        return "", base_name, model_name

def create_vllm_eval_tsubame_script(yaml_path, base_model, task_name, tsubame_dir, api_url=None):
    """VLLM eval用のtsubameスクリプトを作成（複数seedに対応）"""
    
    # モデル名のマッピング
    model_mapping = {
        "Qwen3": [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            # evalation for custom quantized models
            "output/vllm/models/Qwen3-0.6B-fromw4g128-w2g128",
            "output/vllm/models/Qwen3-0.6B-s1k-w4g128",
            "output/vllm/models/Qwen3-0.6B-w4g128",
            "output/vllm/models/Qwen3-1.7B-fromw4g128-w2g128",
            "output/vllm/models/Qwen3-1.7B-s1k-w4g128",
            "output/vllm/models/Qwen3-1.7B-w4g128",
            "output/vllm/models/Qwen3-4B-fromw4g128-w2g128",
            "output/vllm/models/Qwen3-4B-s1k-w4g128",
            "output/vllm/models/Qwen3-4B-w4g128",
            "output/block_ap_models/vllm/next_prod/Qwen3-8B-w2g128-residual-next"
        ],
        "DeepSeek-R1": [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        ],
        "Llama-3": [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    }
    
    # Qwen3モデルの場合、自動検出されたモデルをmappingに追加
    if base_model == "Qwen3":
        print("Qwen3モデルが検出されました。自動検出されたモデルをmappingに追加中...")
        discovered_models = _discover_qwen3_models()
        
        # 既存のmappingに新しく検出されたモデルを追加（重複を除去）
        existing_models = set(model_mapping["Qwen3"])
        for model in discovered_models:
            if model not in existing_models:
                model_mapping["Qwen3"].append(model)
                print(f"  + {model} をmappingに追加")
        
        print(f"更新されたQwen3マッピング: {len(model_mapping['Qwen3'])}個のモデル")
    
    # ベースモデルに対応するモデルリストを取得
    models = model_mapping.get(base_model, [])
    
    if not models:
        print(f"Warning: No model mapping found for {base_model}")
        return
    
    # tsubame用のヘッダー
    tsubame_header = """#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -m abe
#$ -M okoshi.yasuyuki@artic.iir.titech.ac.jp
#$ -o log/
#$ -e log/

source ~/.bashrc
conda activate evalscope-python3.12
export HF_HOME=/gs/bs/tga-artic/y-okoshi/.cache/huggingface
export NLTK_DATA=/gs/bs/tga-artic/y-okoshi/.cache/nltk_data
export VLLM_DISABLE_COMPILE_CACHE=1
source ~/.vllmrc

# VLLMサーバー設定 - ポート番号をJOB_IDの下4桁から計算
SUB_JOB_ID=$(echo $JOB_ID | tail -c 5)
VLLM_PORT="5${SUB_JOB_ID}"
VLLM_API_URL="http://127.0.0.1:${VLLM_PORT}/v1/chat/completions"

# VLLMサーバーの疎通確認関数
check_vllm_server() {
    local max_attempts=60  # 最大60回試行 (約5分)
    local attempt=1
    
    # 最初の1分間は待機（VLLMサーバーの起動時間を考慮）
    echo "Waiting 60 seconds for VLLM server to initialize..."
    sleep 60
    
    echo "Starting VLLM server connectivity checks..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "VLLM server is ready! (attempt: $attempt)"
            return 0
        fi
        
        echo "Attempt $attempt: VLLM server not ready yet, waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: VLLM server failed to start after $max_attempts attempts"
    return 1
}

# VLLMサーバーの停止関数
stop_vllm_server() {
    echo "Stopping VLLM server..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        echo "VLLM server stopped (PID: $VLLM_PID)"
    fi
}

# トラップでスクリプト終了時にVLLMサーバーを停止
trap stop_vllm_server EXIT

"""
    
    # 複数のseedを定義（デフォルト42 + 追加4つ）
    seeds = [42, 123, 456, 789, 2024]

    # max_tokensの設定を定義
    max_tokens_configs = [
        {"value": 8192, "label": "8k"},
        {"value": 16384, "label": "16k"},
        {"value": 32768, "label": "32k"}
    ]

    # 各モデルに対してスクリプトを生成
    for i, model in enumerate(models):
        # モデル名からディレクトリ名とサーバー名を生成
        model_dir_name = model.split('/')[-1]
        model_serve_name = get_model_serve_name(model)

        # パス構成要素を抽出
        dirname, base_name, model_name = extract_path_components(model)

        # 各max_tokens設定に対してスクリプトを生成
        for max_tokens_config in max_tokens_configs:
            max_tokens_value = max_tokens_config["value"]
            max_tokens_label = max_tokens_config["label"]

            # 各seedに対してスクリプトを生成
            for seed in seeds:
                # 出力パスを生成（max_tokensラベルとseedを含む）
                if dirname:
                    # サブディレクトリがある場合: tsubame/vllm/{dirname}/{base_name}/{model_name}/{task_name}_{max_tokens_label}_{seed}.sh
                    output_path = tsubame_dir / dirname / base_name / model_name / f"{task_name}_{max_tokens_label}_{seed}.sh"
                else:
                    # サブディレクトリがない場合: tsubame/vllm/{base_name}/{model_name}/{task_name}_{max_tokens_label}_{seed}.sh
                    output_path = tsubame_dir / base_name / model_name / f"{task_name}_{max_tokens_label}_{seed}.sh"

                # 評価結果の出力先パスを生成（max_tokensを含むディレクトリ構造）
                if dirname:
                    # サブディレクトリがある場合: eval/results/{dirname}/{base_name}/{model_name}/{task_name}_{max_tokens_label}
                    work_dir = f"eval/results/{dirname}/{base_name}/{model_name}/{task_name}_{max_tokens_label}"
                else:
                    # サブディレクトリがない場合: eval/results/{base_name}/{model_name}/{task_name}_{max_tokens_label}
                    work_dir = f"eval/results/{base_name}/{model_name}/{task_name}_{max_tokens_label}"

                # スクリプト内容を生成
                script_content = f"""# Model {i+1}: {model} (seed: {seed}, max_tokens: {max_tokens_value})
echo "=========================================="
echo "Starting VLLM evaluation for {model} on {task_name} with seed {seed} and max_tokens {max_tokens_value}"
echo "Model serve name: {model_serve_name}"
echo "Seed: {seed}"
echo "Max tokens: {max_tokens_value}"
echo "=========================================="

# 1. VLLMサーバーの起動
echo "Step 1: Starting VLLM server..."
CUDA_VISIBLE_DEVICES=0 vllm serve {model} \\
    --gpu-memory-utilization 0.9 \\
    --served-model-name {model_serve_name} \\
    --trust_remote_code \\
    --download-dir $HF_HOME \\
    --dtype auto \\
    --compilation-config '{{"cache_dir": "/gs/bs/tga-artic/.cache/vllm"}}' \\
    --port $VLLM_PORT &
VLLM_PID=$!

echo "VLLM server started with PID: $VLLM_PID"

# 2. 疎通確認
echo "Step 2: Checking server connectivity..."
if ! check_vllm_server; then
    echo "ERROR: Failed to start VLLM server for {model}"
    exit 1
fi

# 3. 評価実行
echo "Step 3: Running evaluation with seed {seed} and max_tokens {max_tokens_value}..."
PYTHONPATH=$PWD:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python eval/evalscope_vllm.py {yaml_path} \\
--model {model_serve_name} \\
--work-dir {work_dir} \\
--dataset-dir /gs/bs/tga-artic/y-okoshi/.cache/modelscope/hub/datasets/ \\
--model-id {model_name} \\
--api-url "$VLLM_API_URL" \\
--generation-config '{{"seed": {seed}, "max_tokens": {max_tokens_value}}}'

EVAL_EXIT_CODE=$?

# 4. 結果確認
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Evaluation completed for {model} on {task_name} with seed {seed} and max_tokens {max_tokens_value}"
else
    echo "ERROR: Evaluation failed for {model} on {task_name} with seed {seed} and max_tokens {max_tokens_value} (exit code: $EVAL_EXIT_CODE)"
fi

echo "=========================================="
echo "Completed evaluation for {model} on {task_name} with seed {seed} and max_tokens {max_tokens_value}"
echo "=========================================="

# VLLMサーバーの停止は trap で自動実行される
exit $EVAL_EXIT_CODE

"""

                # 出力ディレクトリを作成
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # スクリプトを保存
                with open(output_path, 'w') as f:
                    f.write(tsubame_header + script_content)

                # 実行権限を付与
                os.chmod(output_path, 0o755)

                print(f"Generated: {output_path}")

def main():
    """メイン処理"""
    
    # ベースディレクトリ
    yamls_dir = Path("eval/yamls/vllm")
    tsubame_dir = Path("tsubame/vllm")
    
    # yamlsディレクトリが存在するかチェック
    if not yamls_dir.exists():
        print(f"Error: {yamls_dir} does not exist")
        return
    
    # 処理対象のベースモデル
    target_base_models = ["Qwen3", "DeepSeek-R1", "Llama-3"]
    
    total_scripts = 0
    
    # 各ベースモデルに対して処理
    for base_model in target_base_models:
        yaml_pattern = yamls_dir / base_model / "*.yaml"
        yaml_files = glob.glob(str(yaml_pattern))
        
        if not yaml_files:
            print(f"No yaml files found in {yamls_dir}/{base_model}")
            continue
        
        print(f"Found {len(yaml_files)} yaml files for {base_model}")
        
        # 各yamlファイルに対してtsubameスクリプトを生成
        for yaml_path in yaml_files:
            yaml_path = Path(yaml_path)
            
            # ベースモデル名とタスク名を抽出
            base_model_name = yaml_path.parent.name
            task_name = yaml_path.stem
            
            # tsubameスクリプトを生成（各モデルに対して個別のファイルを作成）
            create_vllm_eval_tsubame_script(yaml_path, base_model_name, task_name, tsubame_dir)
        
        # モデル数を計算（model_mappingから取得）× seed数 × max_tokens設定数
        model_mapping = {
            "Qwen3": 13,  # 13 models
            "DeepSeek-R1": 3,  # 3 models
            "Llama-3": 3   # 3 models
        }
        model_count = model_mapping.get(base_model, 0)
        seeds_count = 5  # 5つのseed
        max_tokens_count = 3  # 3つのmax_tokens設定 (8k, 16k, 32k)
        total_scripts += len(yaml_files) * model_count * seeds_count * max_tokens_count
    
    print(f"\nVLLM Tsubame scripts generated in: {tsubame_dir}")
    print("Total scripts generated:", total_scripts)

if __name__ == "__main__":
    main() 