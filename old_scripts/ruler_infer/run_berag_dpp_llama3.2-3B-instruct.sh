#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --output=logs/ruler_%j.out
#SBATCH --error=logs/ruler_%j.err

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
  "niah_multikey_1"
  "niah_multikey_2"
  "niah_multikey_3"
  "niah_multivalue"
  "niah_multiquery"
)

# SEQ_LENGTHS=(4096 8192 16384 32000 64000 128000)
SEQ_LENGTHS=(64000)
# NUM_CHUNKS=(2 2 2 4 4 8)
NUM_CHUNKS=(8)

NUM_SAMPLES=4
REMOVE_NEWLINE_TAB=""
REMOVE_ANSWER_PREFIX="--remove_answer_prefix"
STOP_WORDS=""
BATCH_SIZE=1
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

GPUS=1
VLLM_PORT="${VLLM_PORT:-5000}"
BERAG_DPP_PORT="${BERAG_DPP_PORT:-5001}"
CHUNK_SIZE=512
NUM_SUBSET_SAMPLES="${NUM_SUBSET_SAMPLES:-4}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-128}"
NUM_LOOK_AHEAD="${NUM_LOOK_AHEAD:-10}"
LOOKAHEAD_ROLLOUT="${LOOKAHEAD_ROLLOUT:-4}"
MAX_SUBSET_SIZE="${MAX_SUBSET_SIZE:-2}"
BETA="${BETA:-1.0}"
export BERAG_DPP_DEBUG="${BERAG_DPP_DEBUG:-1}"
export BERAG_DPP_ADD_ANSWER_PREFIX="${BERAG_DPP_ADD_ANSWER_PREFIX:-0}"

BERAG_DPP_HYPER_SUFFIX="b${NUM_SUBSET_SAMPLES}_s${SEGMENT_LENGTH}_la${NUM_LOOK_AHEAD}_lr${LOOKAHEAD_ROLLOUT}_k${MAX_SUBSET_SIZE}_beta${BETA}"

health_alive() {
  local port="$1"
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q 200
}

if health_alive "${VLLM_PORT}"; then
  echo "[RULER-BERAG-DPP] vLLM server already alive at port ${VLLM_PORT}."
else
  echo "[RULER-BERAG-DPP] Starting vLLM server..."
  bash "${SCRIPT_DIR}/start_vllm_server.sh" "${MODEL_PATH}" "${VLLM_PORT}" "${GPUS}"
fi

if ! health_alive "${VLLM_PORT}"; then
  echo "[RULER-BERAG-DPP] vLLM server failed to become ready."
  exit 1
fi

if health_alive "${BERAG_DPP_PORT}"; then
  echo "[RULER-BERAG-DPP] BERAG-DPP server already alive at port ${BERAG_DPP_PORT}."
else
  echo "[RULER-BERAG-DPP] Starting BERAG-DPP server..."
  bash "${SCRIPT_DIR}/start_berag_dpp_server.sh" \
    "${MODEL_PATH}" "${TOKENIZER_PATH}" "${BERAG_DPP_PORT}" "${VLLM_PORT}" "${CHUNK_SIZE}" "${BERAG_DPP_DEBUG}" \
    "${NUM_SUBSET_SAMPLES}" "${SEGMENT_LENGTH}" "${NUM_LOOK_AHEAD}" "${LOOKAHEAD_ROLLOUT}" "${MAX_SUBSET_SIZE}" "${BETA}" \
    "${BERAG_DPP_ADD_ANSWER_PREFIX}"
fi

if ! health_alive "${BERAG_DPP_PORT}"; then
  echo "[RULER-BERAG-DPP] BERAG-DPP server failed to become ready."
  exit 1
fi

cd "${RULER_SCRIPTS_DIR}"
total_time=0

for i in "${!SEQ_LENGTHS[@]}"; do
  MAX_SEQ_LENGTH="${SEQ_LENGTHS[i]}"
  NUM_CHUNKS_CURRENT="${NUM_CHUNKS[i]:-8}"
  echo "[RULER-BERAG-DPP] === Seq length: ${MAX_SEQ_LENGTH}, num_chunks: ${NUM_CHUNKS_CURRENT}, hyper: ${BERAG_DPP_HYPER_SUFFIX} ==="
  RESULTS_DIR="${ROOT_DIR}/BERAG-DPP/${MODEL_NAME}/${BERAG_DPP_HYPER_SUFFIX}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
  DATA_DIR="${RESULTS_DIR}/data"
  PRED_DIR="${RESULTS_DIR}/pred"
  mkdir -p "${DATA_DIR}" "${PRED_DIR}"

  for TASK in "${TASKS[@]}"; do
    echo "[RULER-BERAG-DPP] Preparing data for task ${TASK}..."
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
    echo "[RULER-BERAG-DPP] Calling API for task ${TASK}..."
    python pred/call_api.py \
      --data_dir "${DATA_DIR}" \
      --save_dir "${PRED_DIR}" \
      --benchmark "${BENCHMARK}" \
      --task "${TASK}" \
      --server_type berag_dpp \
      --server_port "${BERAG_DPP_PORT}" \
      --model_name_or_path "${MODEL_PATH}" \
      --num_chunks "${NUM_CHUNKS_CURRENT}" \
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

  echo "[RULER-BERAG-DPP] Evaluating benchmark ${BENCHMARK}..."
  python eval/evaluate.py --data_dir "${PRED_DIR}" --benchmark "${BENCHMARK}"
done

echo "[RULER-BERAG-DPP] Total time spent on call_api: ${total_time} seconds"
echo "[RULER-BERAG-DPP] Results saved under: ${ROOT_DIR}/BERAG-DPP/${MODEL_NAME}/${BERAG_DPP_HYPER_SUFFIX}/"
