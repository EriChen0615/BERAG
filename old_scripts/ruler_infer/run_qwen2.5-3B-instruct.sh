#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --output=logs/ruler_%j.out
#SBATCH --error=logs/ruler_%j.err

#
# RULER inference for Qwen/Qwen2.5-3B-Instruct (128k context).
# Sweeps across available context lengths. Uses qwen2-chat template.
#
# Submit: cd A-RAVQA && sbatch scripts/ruler_infer/run_qwen2.5-3B-instruct.sh
# Run interactively: bash scripts/ruler_infer/run_qwen2.5-3B-instruct.sh
#
# Model: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

set -e

SCRIPT_DIR="scripts/ruler_infer"
REPO_ROOT="."
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"
mkdir -p logs

# ---------- Paths ----------
ROOT_DIR="${REPO_ROOT}/outputs/0326/ruler"

# ---------- Model ----------
MODEL_NAME="Qwen2.5-3B-Instruct"
# MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TOKENIZER_PATH="${MODEL_PATH}"
MODEL_TEMPLATE_TYPE="qwen2-chat"
TOKENIZER_TYPE="hf"

# ---------- Benchmark and tasks (task names must exist in synthetic.yaml) ----------
BENCHMARK="synthetic"
TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
)

# ---------- Sequence lengths (Qwen2.5 3B: 128k context) ----------
SEQ_LENGTHS=(
    2048
    4096
    8192
    16384
    32000
)

# ---------- Data / inference options ----------
NUM_SAMPLES=256
REMOVE_NEWLINE_TAB=""
STOP_WORDS=""
BATCH_SIZE=1
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# ---------- vLLM server ----------
GPUS=1
VLLM_PORT=5000

# ---------- Check if vLLM server is alive ----------
vllm_alive() {
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null | grep -q 200
}

if vllm_alive; then
    echo "[RULER] vLLM server already alive at port ${VLLM_PORT}. Skipping start."
else
    echo "[RULER] vLLM server not alive. Starting..."
    bash "${SCRIPT_DIR}/start_vllm_server.sh" "${MODEL_PATH}" "${VLLM_PORT}" "${GPUS}"
    if ! vllm_alive; then
        echo "[RULER] vLLM server failed to become ready."
        exit 1
    fi
fi

# ---------- Run prepare / call_api / evaluate ----------
cd "${RULER_SCRIPTS_DIR}"
total_time=0

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    echo "[RULER] === Seq length: ${MAX_SEQ_LENGTH} ==="
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"

    for TASK in "${TASKS[@]}"; do
        echo "[RULER] Preparing data for task ${TASK}..."
        python data/prepare.py \
            --save_dir "${DATA_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --tokenizer_type "${TOKENIZER_TYPE}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --model_template_type "${MODEL_TEMPLATE_TYPE}" \
            --num_samples "${NUM_SAMPLES}" \
            ${REMOVE_NEWLINE_TAB}

        start_time=$(date +%s)
        echo "[RULER] Calling API for task ${TASK}..."
        python pred/call_api.py \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --server_type vllm \
            --model_name_or_path "${MODEL_PATH}" \
            --temperature "${TEMPERATURE}" \
            --top_k "${TOP_K}" \
            --top_p "${TOP_P}" \
            --batch_size "${BATCH_SIZE}" \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done

    echo "[RULER] Evaluating benchmark ${BENCHMARK}..."
    python eval/evaluate.py \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}"
done

echo "[RULER] Total time spent on call_api: ${total_time} seconds"
echo "[RULER] Done."
