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
# RULER inference runner using BERAG with:
#   decode_method=berag_iterative_rerank
#
# Runs only MAX_SEQ_LENGTH=4096.
#
# Submit:
#   cd A-RAVQA && sbatch scripts/ruler_infer/run_berag_iterative_rerank_qwen2.5-7B_4096.sh
# Run interactively:
#   bash scripts/ruler_infer/run_berag_iterative_rerank_qwen2.5-7B_4096.sh

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

# Sequence length to run first
SEQ_LENGTHS=(
  4096
)

# Number of chunks for 4096 (matches the BERAG 7B runner: 8).
NUM_CHUNKS_CURRENT=8

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
BEAM_WIDTH="${BEAM_WIDTH:-1}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-4}"
MAX_COMPOSITE_SIZE="${MAX_COMPOSITE_SIZE:-1}"
SEGMENT_GEN_BATCH_SIZE="${SEGMENT_GEN_BATCH_SIZE:-4}"

# Debug mode for the BERAG engine (default: true). Set to 0 to disable.
export BERAG_DEBUG="${BERAG_DEBUG:-1}"

BERAG_DECODE_METHOD="berag_iterative_rerank"

# Different hyperparams -> different folders (avoid overwriting other runs)
BERAG_HYPER_SUFFIX="b${BEAM_WIDTH}_s${SEGMENT_LENGTH}_mc${MAX_COMPOSITE_SIZE}_sg${SEGMENT_GEN_BATCH_SIZE}_iterRerank"

berag_alive() {
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${BERAG_PORT}/health" 2>/dev/null | grep -q 200
}

if berag_alive; then
  echo "[RULER-BERAG] BERAG server already alive at port ${BERAG_PORT}. Skipping start."
else
  echo "[RULER-BERAG] BERAG server not alive. Starting..."
  bash "${SCRIPT_DIR}/start_berag_server.sh" \
    "${MODEL_PATH}" \
    "${BERAG_PORT}" \
    "${CHUNK_SIZE}" \
    "${BERAG_DEBUG}" \
    "${BEAM_WIDTH}" \
    "${SEGMENT_LENGTH}" \
    "${MAX_COMPOSITE_SIZE}" \
    "${SEGMENT_GEN_BATCH_SIZE}"

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
  echo "[RULER-BERAG] === Seq length: ${MAX_SEQ_LENGTH}, num_chunks: ${NUM_CHUNKS_CURRENT}, hyper: ${BERAG_HYPER_SUFFIX} ==="

  RESULTS_DIR="${ROOT_DIR}/BERAG/${MODEL_NAME}/${BERAG_HYPER_SUFFIX}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
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
      --decode_method "${BERAG_DECODE_METHOD}" \
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

#!/bin/bash
# RULER inference runner using BERAG latent chunk-selection inference:
#   decode_method=berag_iterative_rerank
#
# Runs only MAX_SEQ_LENGTH=4096 (small run).
#
# Example:
#   bash scripts/ruler_infer/run_berag_iterative_rerank_qwen2.5-7B_4096.sh

set -e

SCRIPT_DIR="scripts/ruler_infer"
REPO_ROOT="."
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"
mkdir -p logs

# ---------- Paths ----------
ROOT_DIR="${REPO_ROOT}/outputs/0326/ruler"

# ---------- Model (defaults) ----------
# You can edit these two lines to point at your local model directory.
MODEL_NAME="Qwen2.5-3B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

# ---------- Server/Decode ----------
PORT=8011
DEBUG=1
BERAG_DECODE_METHOD="berag_iterative_rerank"

# Candidate-generation / inference hyperparams (server-side)
BERAG_BEAM_WIDTH=16
SEGMENT_LENGTH=32
MAX_COMPOSITE_SIZE=4
SEGMENT_GEN_BATCH_SIZE=8
BEAM_SCORE_MODE="product"

# Latent rerank hyperparams (request-side)
# (These are read by `serve_berag.py` when `decode_method` is `berag_iterative_rerank`.)
RERANK_ROUNDS=1
OPT_STEPS=30
OPT_LR=0.05
TAU=1.0
LAMBDA_=0.5

# ---------- RULER benchmark config ----------
# These task names should match what RULER expects in `scripts/pred`.
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

# Sequence length to run first
MAX_SEQ_LENGTH=4096

# Chunks for the first pass. (You can increase this later for higher accuracy.)
NUM_CHUNKS_CURRENT=8

# Generation sampling
TEMPERATURE=0.0
TOP_K=0
TOP_P=1.0
STOP_WORDS=""

# Prompt/model settings used by existing BERAG scripts
MODEL_TEMPLATE_TYPE="base"
TOKENIZER_TYPE="hf"

# ---------- Derived output paths ----------
BERAG_HYPER_SUFFIX="b${BERAG_BEAM_WIDTH}_iterRerank_s${SEGMENT_LENGTH}_mc${MAX_COMPOSITE_SIZE}_sg${SEGMENT_GEN_BATCH_SIZE}_${BEAM_SCORE_MODE}"
PRED_DIR="${ROOT_DIR}/${MODEL_NAME}/${BERAG_HYPER_SUFFIX}/${BENCHMARK}/${MAX_SEQ_LENGTH}"

DATA_DIR="${REPO_ROOT}/third_party/RULER/data"

echo "[run] Starting BERAG iterative rerank run"
echo "[run] MODEL_PATH=${MODEL_PATH}"
echo "[run] PRED_DIR=${PRED_DIR}"

mkdir -p "${PRED_DIR}"
mkdir -p "${PRED_DIR}/logs"

# ---------- Start server ----------
SERVER_LOG="${PRED_DIR}/logs/serve_berag_${PORT}.log"
python3 "${RULER_SCRIPTS_DIR}/pred/serve_berag.py" \
  --model "${MODEL_PATH}" \
  --tokenizer_type "${TOKENIZER_TYPE}" \
  --model_template_type "${MODEL_TEMPLATE_TYPE}" \
  --port "${PORT}" \
  --num_chunks "${NUM_CHUNKS_CURRENT}" \
  --debug "${DEBUG}" \
  --beam_width "${BERAG_BEAM_WIDTH}" \
  --segment_length "${SEGMENT_LENGTH}" \
  --max_composite_size "${MAX_COMPOSITE_SIZE}" \
  --segment_gen_batch_size "${SEGMENT_GEN_BATCH_SIZE}" \
  --beam_score_mode "${BEAM_SCORE_MODE}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "[run] Stopping server pid=${SERVER_PID}"
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[run] Waiting briefly for server..."
sleep 3

# ---------- Run inference ----------
python3 "${RULER_SCRIPTS_DIR}/pred/call_api.py" \
  --data_dir "${DATA_DIR}" \
  --save_dir "${PRED_DIR}" \
  --benchmark "${BENCHMARK}" \
  --task "${TASKS[0]}" \
  --server_type berag \
  --model_name_or_path "${MODEL_PATH}" \
  --num_chunks "${NUM_CHUNKS_CURRENT}" \
  --decode_method "${BERAG_DECODE_METHOD}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --top_p "${TOP_P}" \
  --batch_size 1 \
  --tau "${TAU}" \
  --lambda_ "${LAMBDA_}" \
  --rerank_rounds "${RERANK_ROUNDS}" \
  --opt_steps "${OPT_STEPS}" \
  --opt_lr "${OPT_LR}" \
  --stop_words ${STOP_WORDS}

echo "[run] Done. Results saved to ${PRED_DIR}"

