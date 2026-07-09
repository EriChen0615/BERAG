#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH -J pred_retrieval_baseline

set -euo pipefail

# Activate your Python env before running (e.g. source scripts/hpc_activate_env_py310_infer.sh).
# No environment activation or cd is embedded here by design.
# Assume this script is submitted from repo root:
#   /home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA
#
# Usage:
#   MODEL_GROUP=qwen2vl_7b bash analysis/run_predict_retrieval_success_baseline.sh
#   MODEL_GROUP=gpt4o_mini TAKE_N=10 bash analysis/run_predict_retrieval_success_baseline.sh
#   FORCE_OVERRIDE=true MODEL_GROUP=qwen2vl_7b bash analysis/run_predict_retrieval_success_baseline.sh

MODEL_GROUP="${MODEL_GROUP:-qwen2vl_7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis/output/predict_retrieval_success}"
FORCE_OVERRIDE="${FORCE_OVERRIDE:-false}"
TAKE_N="${TAKE_N:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
IMG_BASEDIR="${IMG_BASEDIR:-.}"

declare -a infoseek_experiments=(
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=1-h4-prior=prior_head-retrieved_passage-TakeN=256/inference_results_with_predictions.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=2-h4-prior=prior_head-retrieved_passage-TakeN=256/inference_results_with_predictions.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=3-h4-prior=prior_head-retrieved_passage-TakeN=256/inference_results_with_predictions.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=5-h4-prior=prior_head-retrieved_passage-TakeN=256/inference_results_with_predictions.csv"
)

declare -a infoseek_ks=(1 2 3 5)

declare -A exp_qwen2vl_7b=(
    [model_backend]="vllm"
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [report_name]="report_qwen2vl_7b.csv"
    [enabled]="true"
)

declare -A exp_qwen25vl_7b=(
    [model_backend]="vllm"
    [model_path]="Qwen/Qwen2.5-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2.5-VL-7B-Instruct"
    [report_name]="report_qwen25vl_7b.csv"
    [enabled]="true"
)

declare -A exp_gpt4o_mini=(
    [model_backend]="openai"
    [model_path]="gpt-4o-mini-2024-07-18"
    [processor_path]=""
    [report_name]="report_gpt4o_mini.csv"
    [enabled]="true"
)

run_model_group() {
    local group_ref="$1"
    declare -n group="$group_ref"

    if [[ "${group[enabled]}" != "true" ]]; then
        echo "Skipping disabled model group: ${MODEL_GROUP}"
        return 0
    fi

    echo "========================================"
    echo "Running retrieval-success baseline"
    echo "  Model group: ${MODEL_GROUP}"
    echo "  Backend: ${group[model_backend]}"
    echo "  Model path: ${group[model_path]}"
    echo "  Experiments: ${#infoseek_experiments[@]}"
    echo "  Force override: ${FORCE_OVERRIDE}"
    echo "  Take N: ${TAKE_N}"
    echo "========================================"

    args=(
        analysis/predict_retrieval_success_baseline.py
        --experiments "${infoseek_experiments[@]}"
        --Ks "${infoseek_ks[@]}"
        --output_root "${OUTPUT_ROOT}"
        --report_csv_path "${OUTPUT_ROOT}/${group[report_name]}"
        --model_backend "${group[model_backend]}"
        --model_path "${group[model_path]}"
        --batch_size "${BATCH_SIZE}"
        --max_model_len "${MAX_MODEL_LEN}"
        --max_tokens "${MAX_TOKENS}"
        --img_basedir "${IMG_BASEDIR}"
        --take_n "${TAKE_N}"
    )

    if [[ -n "${group[processor_path]}" ]]; then
        args+=(--processor_path "${group[processor_path]}")
    fi

    if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
        args+=(--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}")
    fi

    if [[ "${FORCE_OVERRIDE}" == "true" ]]; then
        args+=(--force_override)
    fi

    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python "${args[@]}"
}

case "${MODEL_GROUP}" in
    qwen2vl_7b)
        run_model_group exp_qwen2vl_7b
        ;;
    qwen25vl_7b)
        run_model_group exp_qwen25vl_7b
        ;;
    gpt4o_mini)
        run_model_group exp_gpt4o_mini
        ;;
    *)
        echo "Unknown MODEL_GROUP: ${MODEL_GROUP}" >&2
        echo "Expected one of: qwen2vl_7b, qwen25vl_7b, gpt4o_mini" >&2
        exit 1
        ;;
esac

echo "All baseline runs complete for ${MODEL_GROUP}."
