#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --output=logs/ruler_%j.out
#SBATCH --error=logs/ruler_%j.err

#
# RULER inference runner using BERAG (HF backend).
# Ensures BERAG server is up, then runs prepare / call_api / evaluate.
#
# Submit: cd A-RAVQA && sbatch scripts/ruler_infer/run_berag_qwen2.5-7B.sh
# Run interactively: bash scripts/ruler_infer/run_berag_qwen2.5-7B.sh

set -e

SCRIPT_DIR="scripts/ruler_infer"
REPO_ROOT="."
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"
mkdir -p logs

# ---------- Paths ----------
ROOT_DIR="${REPO_ROOT}/outputs/0326/ruler"

# ---------- Model ----------
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_PATH="${MODEL_PATH}"
MODEL_TEMPLATE_TYPE="base"
TOKENIZER_TYPE="hf"

# ---------- Benchmark and tasks ----------
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

# ---------- Sequence lengths ----------
SEQ_LENGTHS=(
    2048
    4096
    8192
    16384
    32000
)

# ---------- Number of chunks per context length (user-side; one value per SEQ_LENGTHS entry) ----------
# Chunking is done before sending to the engine. Engine only uses chunk_size when context is a raw string.
NUM_CHUNKS=(
    4
    8
    16
    32
    64
)

# ---------- Data / inference options ----------
NUM_SAMPLES=256
REMOVE_NEWLINE_TAB=""
STOP_WORDS=""
BATCH_SIZE=1
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# ---------- BERAG server ----------
BERAG_PORT=5000
CHUNK_SIZE=512
# Beam width for BERAG beam search (1 = greedy). When >1, API returns all candidates, beam_scores, posteriors.
BEAM_WIDTH="${BEAM_WIDTH:-1}"
# Segment length for BERAG-SG (segment-level generation).
SEGMENT_LENGTH="${SEGMENT_LENGTH:-4}"
# Max composite chunk size for BERAG-SG (1 = no composite states, single round).
MAX_COMPOSITE_SIZE="${MAX_COMPOSITE_SIZE:-1}"
# Batch size for batched segment generation (chunks per forward).
SEGMENT_GEN_BATCH_SIZE="${SEGMENT_GEN_BATCH_SIZE:-4}"
# Debug mode for the BERAG engine (default: true). Set to 0 to disable.
export BERAG_DEBUG="${BERAG_DEBUG:-1}"

berag_alive() {
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${BERAG_PORT}/health" 2>/dev/null | grep -q 200
}

if berag_alive; then
    echo "[RULER-BERAG] BERAG server already alive at port ${BERAG_PORT}. Skipping start."
else
    echo "[RULER-BERAG] BERAG server not alive. Starting..."
    bash "${SCRIPT_DIR}/start_berag_server.sh" "${MODEL_PATH}" "${BERAG_PORT}" "${CHUNK_SIZE}" "${BERAG_DEBUG}" "${BEAM_WIDTH}" "${SEGMENT_LENGTH}" "${MAX_COMPOSITE_SIZE}" "${SEGMENT_GEN_BATCH_SIZE}"
    if ! berag_alive; then
        echo "[RULER-BERAG] BERAG server failed to become ready."
        exit 1
    fi
fi

# ---------- Run prepare / call_api / evaluate ----------
cd "${RULER_SCRIPTS_DIR}"
total_time=0

for i in "${!SEQ_LENGTHS[@]}"; do
    MAX_SEQ_LENGTH="${SEQ_LENGTHS[i]}"
    NUM_CHUNKS_CURRENT="${NUM_CHUNKS[i]:-32}"
    echo "[RULER-BERAG] === Seq length: ${MAX_SEQ_LENGTH}, num_chunks: ${NUM_CHUNKS_CURRENT} ==="
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"

    for TASK in "${TASKS[@]}"; do
        echo "[RULER-BERAG] Preparing data for task ${TASK}..."
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
        echo "[RULER-BERAG] Calling API for task ${TASK} (num_chunks=${NUM_CHUNKS_CURRENT})..."
        python pred/call_api.py \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --server_type berag \
            --model_name_or_path "${MODEL_PATH}" \
            --num_chunks "${NUM_CHUNKS_CURRENT}" \
            --decode_method "${BERAG_DECODE_METHOD:-berag_sg}" \
            --temperature "${TEMPERATURE}" \
            --top_k "${TOP_K}" \
            --top_p "${TOP_P}" \
            --batch_size "${BATCH_SIZE}" \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done

    echo "[RULER-BERAG] Evaluating benchmark ${BENCHMARK}..."
    python eval/evaluate.py \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}"
done

echo "[RULER-BERAG] Total time spent on call_api: ${total_time} seconds"
echo "[RULER-BERAG] Done."

