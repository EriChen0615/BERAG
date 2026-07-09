#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH -J mmneedle_beft_qwen2vl2b
set -euo pipefail

# Train BEFT on curated MMNeedle data using LLaMA-Factory.
# Base model: Qwen/Qwen2-VL-2B-Instruct
#
# Usage examples:
#   N_GRID=1 SEQUENCE_LENGTH=10 K=2 TAKE_N=0 OFFSET=0 bash scripts/mmneedle_curate/train_beft_mmneedle_llava_llama3_8b.sh
#   N_GRID=2 SEQUENCE_LENGTH=10 K=2 TAKE_N=20000 OFFSET=0 bash scripts/mmneedle_curate/train_beft_mmneedle_llava_llama3_8b.sh

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "${ROOT_DIR}"

# if [[ -f "scripts/hpc_activate_env_py310_infer.sh" ]]; then
#   # shellcheck disable=SC1091
#   source scripts/hpc_activate_env_py310_infer.sh
# fi

N_GRID="${N_GRID:-1}"                      # fixed target: 1x1
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"   # fixed target: rag10
K="${K:-2}"                                # fixed target: k=2
TAKE_N="${TAKE_N:-10000}"                  # fixed target: size=10000
OFFSET="${OFFSET:-0}"                      # fixed target: offset=0

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
PRIOR_HEAD_LR="${PRIOR_HEAD_LR:-0.000001}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1.0}"
REPORT_TO="${REPORT_TO:-wandb}"

LLAMAFACTORY_ROOT="third_party/LLaMA-Factory-2502"
DATASET_INFO_PATH="${LLAMAFACTORY_ROOT}/data/dataset_info.json"

CURATED_SRC_JSON="${CURATED_SRC_JSON:-/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/MMNeedle/train/curated/rag10-mmneedle-n1x1-k=2-z0-beft-size=10000-offset=0/train_sharegpt.json}"
CURATED_SRC_DIR="$(dirname "${CURATED_SRC_JSON}")"
DATA_TAG="rag10-mmneedle-n1x1-k=2-z0-beft-size=10000-offset=0"

DATA_REL_DIR="jinghong_chen/mmneedle/train/${DATA_TAG}"
DATA_JSON_REL="${DATA_REL_DIR}/train_sharegpt.json"
DATA_JSON_ABS="${LLAMAFACTORY_ROOT}/data/${DATA_JSON_REL}"
DATASET_KEY="mmneedle-rag10-beft-n1-k=2-z0-size=10000-offset=0"

if [[ ! -f "${CURATED_SRC_JSON}" ]]; then
  echo "Missing curated training data: ${CURATED_SRC_JSON}"
  echo "Run: bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh"
  exit 1
fi

mkdir -p "$(dirname "${DATA_JSON_ABS}")"
cp "${CURATED_SRC_JSON}" "${DATA_JSON_ABS}"
if [[ -f "${CURATED_SRC_DIR}/stats.json" ]]; then
  cp "${CURATED_SRC_DIR}/stats.json" "$(dirname "${DATA_JSON_ABS}")/stats.json"
fi

export DATASET_INFO_PATH DATASET_KEY DATA_JSON_REL

python - <<'PY'
import json
from pathlib import Path
import os

dataset_info_path = Path(os.environ["DATASET_INFO_PATH"])
dataset_key = os.environ["DATASET_KEY"]
data_json_rel = os.environ["DATA_JSON_REL"]

with dataset_info_path.open("r", encoding="utf-8") as f:
    info = json.load(f)

info[dataset_key] = {
    "file_name": data_json_rel,
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "images": "images",
        "gt_passage_idx": "gt_passage_idx",
        "passages": "passages",
        "passage_scores": "passage_scores"
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant"
    }
}

with dataset_info_path.open("w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"Registered dataset key: {dataset_key}")
PY

CFG_PATH="${LLAMAFACTORY_ROOT}/my_configs/mmneedle/beft/beft_mmneedle_qwen2vl_2b_rag10_n1x1_k2_z0_size10000_offset0.yaml"
mkdir -p "$(dirname "${CFG_PATH}")"

cat > "${CFG_PATH}" <<EOF
### model
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct

### method
stage: beft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
ppl_hidden_state_offset: 4
ppl_prior_loss_factor: 1.0
ppl_prior_loss_type: logistic
prior_head_lr: ${PRIOR_HEAD_LR}

use_ppl_loss: true
ppl_loss_type: ensemble
ppl_prior_modeling: mlp_head
use_prior_head_loss: true
ppl_prior_head_num_of_layers: 2
ppl_prior_head_proj_dim: 1024
beft_use_gt_subset_branch: true
beft_gt_subset_loss_factor: 1.0

### dataset
dataset: ${DATASET_KEY}
template: qwen2_vl
cutoff_len: 2500
max_samples: 200000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen2_vl-2b/lora/mmneedle/beft/qwen2_vl-2b-rag10-n1x1-k2-z0-size10000-offset0
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true
report_to: ${REPORT_TO}

### train
per_device_train_batch_size: ${BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM}
learning_rate: ${LEARNING_RATE}
num_train_epochs: ${EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 64
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
EOF

echo "Config written: ${CFG_PATH}"
echo "Starting training..."
(
  cd "${LLAMAFACTORY_ROOT}"
  llamafactory-cli train "my_configs/mmneedle/beft/$(basename "${CFG_PATH}")"
)

