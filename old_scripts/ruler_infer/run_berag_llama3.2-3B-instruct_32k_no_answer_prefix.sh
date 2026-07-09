#!/bin/bash
# Small BERAG validation run for RULER synthetic at 32K without the answer prefix.
# Usage:
#   bash scripts/ruler_infer/run_berag_llama3.2-3B-instruct_32k_no_answer_prefix.sh
# Optional env overrides:
#   TASKS_OVERRIDE="niah_single_1 niah_single_2" NUM_SAMPLES=4 BEAM_WIDTH=4 SEGMENT_LENGTH=10

set -e

SCRIPT_DIR="scripts/ruler_infer"
REPO_ROOT="."
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"
mkdir -p logs

ROOT_DIR="${REPO_ROOT}/outputs/0326/ruler"

MODEL_NAME="Llama-3.2-3B-Instruct"
MODEL_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/models/Llama-3.2-3B-Instruct"
TOKENIZER_PATH="${MODEL_PATH}"
MODEL_TEMPLATE_TYPE="meta-llama3"
TOKENIZER_TYPE="hf"

BENCHMARK="synthetic"
TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
)
if [ -n "${TASKS_OVERRIDE:-}" ]; then
    read -r -a TASKS <<< "${TASKS_OVERRIDE}"
fi

MAX_SEQ_LENGTH=32000
NUM_CHUNKS_CURRENT=4

NUM_SAMPLES="${NUM_SAMPLES:-4}"
REMOVE_NEWLINE_TAB=""
REMOVE_ANSWER_PREFIX="--remove_answer_prefix"
STOP_WORDS=""
BATCH_SIZE=1
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

BERAG_PORT="${BERAG_PORT:-5000}"
CHUNK_SIZE=512
BEAM_WIDTH="${BEAM_WIDTH:-4}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-20}"
MAX_COMPOSITE_SIZE="${MAX_COMPOSITE_SIZE:-1}"
SEGMENT_GEN_BATCH_SIZE="${SEGMENT_GEN_BATCH_SIZE:-4}"
BEAM_SCORE_MODE="${BEAM_SCORE_MODE:-proposal_chunk}"
export BERAG_DEBUG="${BERAG_DEBUG:-0}"

BERAG_HYPER_SUFFIX="b${BEAM_WIDTH}_s${SEGMENT_LENGTH}_mc${MAX_COMPOSITE_SIZE}_sg${SEGMENT_GEN_BATCH_SIZE}"
RESULT_TAG="no_answer_prefix_n${NUM_SAMPLES}"

berag_alive() {
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${BERAG_PORT}/health" 2>/dev/null | grep -q 200
}

if berag_alive; then
    echo "[RULER-BERAG] BERAG server already alive at port ${BERAG_PORT}. Skipping start."
else
    echo "[RULER-BERAG] BERAG server not alive. Starting..."
    bash "${SCRIPT_DIR}/start_berag_server.sh" "${MODEL_PATH}" "${BERAG_PORT}" "${CHUNK_SIZE}" "${BERAG_DEBUG}" "${BEAM_WIDTH}" "${SEGMENT_LENGTH}" "${MAX_COMPOSITE_SIZE}" "${SEGMENT_GEN_BATCH_SIZE}" "${BEAM_SCORE_MODE}"
    if ! berag_alive; then
        echo "[RULER-BERAG] BERAG server failed to become ready."
        exit 1
    fi
fi

cd "${RULER_SCRIPTS_DIR}"
total_time=0

RESULTS_DIR="${ROOT_DIR}/BERAG/${MODEL_NAME}/${BERAG_HYPER_SUFFIX}/${RESULT_TAG}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
mkdir -p "${DATA_DIR}" "${PRED_DIR}"

echo "[RULER-BERAG] === 32K validation run, num_chunks: ${NUM_CHUNKS_CURRENT}, tasks: ${TASKS[*]}, samples: ${NUM_SAMPLES} ==="

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
        ${REMOVE_ANSWER_PREFIX} \
        ${REMOVE_NEWLINE_TAB}

    start_time=$(date +%s)
    echo "[RULER-BERAG] Calling API for task ${TASK} (num_chunks=${NUM_CHUNKS_CURRENT}, beam=${BEAM_WIDTH})..."
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
        --berag_concat_all_beams \
        ${STOP_WORDS}
    end_time=$(date +%s)
    time_diff=$((end_time - start_time))
    total_time=$((total_time + time_diff))
done

echo "[RULER-BERAG] Evaluating benchmark ${BENCHMARK}..."
python eval/evaluate.py \
    --data_dir "${PRED_DIR}" \
    --benchmark "${BENCHMARK}"

echo "[RULER-BERAG] Total time spent on call_api: ${total_time} seconds"
echo "[RULER-BERAG] Results saved under: ${RESULTS_DIR}"
echo "[RULER-BERAG] Done."
