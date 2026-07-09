#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --output=logs/ruler_%j.out
#SBATCH --error=logs/ruler_%j.err

#
# RULER inference using BERAG (HF backend) with Llama-3.2-3B-Instruct.
# Beam width 4, with hyperparameter-specific output folders.
#
# Submit: cd A-RAVQA && sbatch scripts/ruler_infer/run_berag_llama3.2-3B-instruct.sh
# Run interactively: bash scripts/ruler_infer/run_berag_llama3.2-3B-instruct.sh
#
# Model: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

set -e

SCRIPT_DIR="scripts/ruler_infer"
REPO_ROOT="."
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"
mkdir -p logs

# ---------- Paths ----------
ROOT_DIR="${REPO_ROOT}/outputs/0326/ruler"

# ---------- Model ----------
MODEL_NAME="Llama-3.2-3B-Instruct"
# MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
MODEL_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/models/Llama-3.2-3B-Instruct"
TOKENIZER_PATH="${MODEL_PATH}"
MODEL_TEMPLATE_TYPE="meta-llama3"
TOKENIZER_TYPE="hf"

# ---------- Benchmark and tasks ----------
BENCHMARK="synthetic"
TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    # "niah_multikey_1"
    # "niah_multikey_2"
    # "niah_multikey_3"
    # "niah_multivalue"
    # "niah_multiquery"
)

# ---------- Sequence lengths (Llama 3.2 3B: 128k context) ----------
SEQ_LENGTHS=(
    4096
    8192
    16384
    32000
    64000
    # 128000
)

# ---------- Number of chunks per context length (one value per SEQ_LENGTHS entry) ----------
NUM_CHUNKS=(
    2
    4
    8 
    16
    32
    # 16 
)

# ---------- BERAG beam width budget (segment-level beam size) ----------
# Must be >= NUM_CHUNKS_CURRENT to give BERAG enough beam capacity for the passages.
# One value per SEQ_LENGTHS entry (aligned by index).
BEAM_WIDTHS=(
    2
    4
    8
    16 
    32
    # 32
)

# ---------- Data / inference options ----------
NUM_SAMPLES=32
REMOVE_NEWLINE_TAB=""
STOP_WORDS=""
BATCH_SIZE=1
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# ---------- BERAG hyperparameters ----------
BERAG_PORT=5000
CHUNK_SIZE=512
SEGMENT_LENGTH="${SEGMENT_LENGTH:-20}"
MAX_COMPOSITE_SIZE="${MAX_COMPOSITE_SIZE:-1}"
SEGMENT_GEN_BATCH_SIZE="${SEGMENT_GEN_BATCH_SIZE:-4}"
BEAM_SCORE_MODE="${BEAM_SCORE_MODE:-proposal_chunk}"
export BERAG_DEBUG="${BERAG_DEBUG:-0}"

berag_alive() {
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${BERAG_PORT}/health" 2>/dev/null | grep -q 200
}

# Start BERAG once; per-request beam width will be overridden via call_api.py.
BEAM_WIDTH_START="${BEAM_WIDTHS[0]:-1}"
for bw in "${BEAM_WIDTHS[@]}"; do
    if [ "${bw}" -gt "${BEAM_WIDTH_START}" ]; then
        BEAM_WIDTH_START="${bw}"
    fi
done

if berag_alive; then
    echo "[RULER-BERAG] BERAG server already alive at port ${BERAG_PORT}. Skipping start."
else
    echo "[RULER-BERAG] BERAG server not alive. Starting with beam_width=${BEAM_WIDTH_START}..."
    bash "${SCRIPT_DIR}/start_berag_server.sh" "${MODEL_PATH}" "${BERAG_PORT}" "${CHUNK_SIZE}" "${BERAG_DEBUG}" "${BEAM_WIDTH_START}" "${SEGMENT_LENGTH}" "${MAX_COMPOSITE_SIZE}" "${SEGMENT_GEN_BATCH_SIZE}" "${BEAM_SCORE_MODE}"
    if ! berag_alive; then
        echo "[RULER-BERAG] BERAG server failed to become ready."
        exit 1
    fi
fi

# ---------- Run prepare / call_api / evaluate ----------
total_time=0

for i in "${!SEQ_LENGTHS[@]}"; do
    MAX_SEQ_LENGTH="${SEQ_LENGTHS[i]}"
    NUM_CHUNKS_CURRENT="${NUM_CHUNKS[i]:-10}"
    BEAM_WIDTH_CURRENT="${BEAM_WIDTHS[i]:-${BEAM_WIDTH:-4}}"
    # Guardrail: ensure beam budget covers passage budget.
    if [ "${BEAM_WIDTH_CURRENT}" -lt "${NUM_CHUNKS_CURRENT}" ]; then
        echo "[RULER-BERAG] Beam width (${BEAM_WIDTH_CURRENT}) < num_chunks (${NUM_CHUNKS_CURRENT}); bumping beam_width_current to ${NUM_CHUNKS_CURRENT}"
        BEAM_WIDTH_CURRENT="${NUM_CHUNKS_CURRENT}"
    fi

    # ---------- Output path suffix: different hyperparams -> different folders ----------
    # Format: b{beam}_s{seg}_mc{max_comp}_sg{seg_gen_batch}_score{mode}
    BERAG_HYPER_SUFFIX="b${BEAM_WIDTH_CURRENT}_s${SEGMENT_LENGTH}_mc${MAX_COMPOSITE_SIZE}_sg${SEGMENT_GEN_BATCH_SIZE}_score${BEAM_SCORE_MODE}"

    echo "[RULER-BERAG] === Seq length: ${MAX_SEQ_LENGTH}, num_chunks: ${NUM_CHUNKS_CURRENT}, beam_width: ${BEAM_WIDTH_CURRENT}, hyper: ${BERAG_HYPER_SUFFIX} ==="

    # Hyperparameter-specific path: BERAG/MODEL/hyper_suffix/benchmark/seq_len
    RESULTS_DIR="${ROOT_DIR}/BERAG/${MODEL_NAME}/${BERAG_HYPER_SUFFIX}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"

    for TASK in "${TASKS[@]}"; do
        echo "[RULER-BERAG] Preparing data for task ${TASK}..."
        python third_party/RULER/scripts/data/prepare.py \
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
        echo "[RULER-BERAG] Calling API for task ${TASK} (num_chunks=${NUM_CHUNKS_CURRENT}, beam=${BEAM_WIDTH_CURRENT})..."
        python third_party/RULER/scripts/pred/call_api.py \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --server_type berag \
            --model_name_or_path "${MODEL_PATH}" \
            --num_chunks "${NUM_CHUNKS_CURRENT}" \
            --berag_beam_width "${BEAM_WIDTH_CURRENT}" \
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
    python third_party/RULER/scripts/eval/evaluate.py \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}"
done

echo "[RULER-BERAG] Total time spent on call_api: ${total_time} seconds"
echo "[RULER-BERAG] Results saved under: ${ROOT_DIR}/BERAG/${MODEL_NAME}/${BERAG_HYPER_SUFFIX}/"
echo "[RULER-BERAG] Done."
