#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH -J evqa_pos_bape_7b
set -euo pipefail

# BAPE / BEFT inference for EVQA gold-doc position datasets (l0h4 7B checkpoint).
# Activate your env before sbatch (e.g. source scripts/hpc_activate_env_py310_infer.sh).

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
cd "${ROOT}"

OUT_ROOT="${ROOT}/outputs/0426/EVQA-positional-analysis"
CURATE_ROOT="${ROOT}/analysis/EVQA-gtdoc-position-datasets"
N="${N:-256}"

INCLUDE_Z0_IN_ENSEMBLE="false"
ENSURE_GT_PASSAGE_IN_ENSEMBLE="false"
TAKE_N=0
DS_OFFSET=0
RETRIEVAL_TOPK_LIST=(20)
PASSAGE_PRIOR="prior_head"
RETRIEVE_FIELD="retrieved_passage"

VARIANT_TAGS=(
  "gtdoc_at_1-4"
  "gtdoc_at_5-8"
  "gtdoc_at_9-12"
  "gtdoc_at_13-16"
  "gtdoc_at_17-20"
)

declare -A exp_7B_rag2_beft_k2_prior_mlp_lr1e_6_l0h4_lora_r64_bs8_epoch1=(
  [model_path]="Qwen/Qwen2-VL-7B-Instruct"
  [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
  [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=0-max=2048/checkpoint-20833"
  [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=0-max=2048/checkpoint-20833/prior_head.pt"
  [hidden_state_offset]=4
  [prompt_template]=""
  [do_eval]="true"
  [use_cache]="true"
)

declare -n exp_cfg="exp_7B_rag2_beft_k2_prior_mlp_lr1e_6_l0h4_lora_r64_bs8_epoch1"

for tag in "${VARIANT_TAGS[@]}"; do
  RETRIEVAL_DS_PATH="${CURATE_ROOT}/EVQA-${N}-${tag}"
  if [[ ! -d "${RETRIEVAL_DS_PATH}" ]]; then
    echo "ERROR: missing curated dataset: ${RETRIEVAL_DS_PATH}" >&2
    exit 1
  fi

  for retrieval_topk in "${RETRIEVAL_TOPK_LIST[@]}"; do
    full_exp_name="BAPE-7B-BEFT-l0h4-epoch1__${tag}__${RETRIEVE_FIELD}-Top${retrieval_topk}-K=${retrieval_topk}-h${exp_cfg[hidden_state_offset]}-prior=${PASSAGE_PRIOR}-${RETRIEVE_FIELD}-TakeN=${TAKE_N}"

    echo "--------------------------------"
    echo "Running BAPE: ${full_exp_name}"
    echo "Dataset: ${RETRIEVAL_DS_PATH}"

    args=(
      --retrieval_ds_path "${RETRIEVAL_DS_PATH}"
      --dataset_name "EVQA"
      --take_n "$TAKE_N"
      --img_basedir "."
      --retrieval_field "${RETRIEVE_FIELD}"
      --retrieval_topk "$retrieval_topk"
      --model_path "${exp_cfg[model_path]}"
      --processor_path "${exp_cfg[processor_path]}"
      --adapter_name_or_path "${ROOT}/${exp_cfg[adapter_path]}"
      --prompt_template "${exp_cfg[prompt_template]}"
      --seed 0
      --batch_size 1
      --exp_name "${OUT_ROOT}/${full_exp_name}"
      --prior_head_path "${ROOT}/${exp_cfg[prior_head_path]}"
      --passage_prior "$PASSAGE_PRIOR"
      --max_batch_size_per_forward 5
      --max_words_per_evidence 512
      --offset "$DS_OFFSET"
    )

    if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
      args+=(--do_eval)
    fi
    if [[ "${exp_cfg[use_cache]}" == "true" ]]; then
      args+=(--use_cache)
    fi
    if [[ "${exp_cfg[hidden_state_offset]}" != "0" ]]; then
      args+=(--hidden_state_offset "${exp_cfg[hidden_state_offset]}")
    fi
    if [[ "$INCLUDE_Z0_IN_ENSEMBLE" == "true" ]]; then
      args+=(--include_z0_in_ensemble)
    fi
    if [[ "$ENSURE_GT_PASSAGE_IN_ENSEMBLE" == "true" ]]; then
      args+=(--ensure_gt_passage_in_ensemble)
    fi

    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_vqa_inference.py "${args[@]}"
    echo "Finished ${full_exp_name}"
    echo "--------------------------------"
  done
done
