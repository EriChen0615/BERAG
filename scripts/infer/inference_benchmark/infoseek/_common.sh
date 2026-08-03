#!/usr/bin/env bash

run_infoseek_benchmark() {
    local method="$1"

    local repo_root="${REPO_ROOT:-/workspace/projects/BERAG}"
    local benchmark_dir
    benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    local inference_script="${repo_root}/src/infer/infoseek_vllm_berag_inference.py"
    local eval_script="${repo_root}/scripts/eval/infoseek/evaluate_experiments.sh"
    local eval_venv="${EVAL_VENV:-${repo_root}/src/eval/.venv}"
    local summarize_script="${benchmark_dir}/summarize_result.py"

    local retrieval_ds="${RETRIEVAL_DS_PATH:-${repo_root}/outputs/jinghong_chen/Infoseek-testfull-with-retrieval}"
    local knowledge_base="${KNOWLEDGE_BASE:-/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_100k_wiki/wiki_100_dict_v4.json}"
    local image_dir="${IMG_BASEDIR:-${repo_root}/src/train/LlamaFactory-0.9.5-beft/data/Infoseek/infoseek_val_images}"
    local tokenizer="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
    local sft_model="${SFT_MODEL:-${repo_root}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-SFT-Infoseek64000}"
    local beft_model="${BEFT_MODEL:-${repo_root}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-Infoseek-ckpt8792}"
    local prior_head="${PRIOR_HEAD_PATH:-${beft_model}/prior_head.pt}"

    local k=50
    local take_n="${TAKE_N:-64}"
    local batch_size="${BATCH_SIZE:-64}"
    local seed="${SEED:-42}"
    local top_p=1.0
    local method_slug
    local method_label
    local model
    local mode
    local max_model_len

    case "${method}" in
        rag)
            method_slug="rag"
            method_label="RAG[@K=50]"
            model="${sft_model}"
            mode="rag"
            max_model_len="${RAG_MAX_MODEL_LEN:-32768}"
            ;;
        berag)
            method_slug="berag"
            method_label="BERAG[@K=50]"
            model="${beft_model}"
            mode="berag"
            max_model_len="${BERAG_MAX_MODEL_LEN:-32768}"
            ;;
        berag_topp)
            method_label="BERAG-TopP[@K=50]"
            model="${beft_model}"
            mode="berag"
            max_model_len="${BERAG_MAX_MODEL_LEN:-32768}"
            # TopP = 1 - 1 / (2K) = 1 - 1 / 100.
            top_p="${PRUNING_TOP_P:-0.99}"
            printf -v method_slug "berag-topp-p=%.3f" "${top_p}"
            ;;
        *)
            echo "Unknown InfoSeek benchmark method: ${method}" >&2
            return 2
            ;;
    esac

    local output_root="${OUTPUT_ROOT:-${repo_root}/outputs/inference_benchmark/infoseek}"
    local exp_dir="${output_root}/qwen3-vl-8b/${method_slug}/K=${k}-TakeN=${take_n}"
    local predictions="${exp_dir}/predictions.jsonl"
    local metrics="${exp_dir}/run_metrics.json"

    for required in \
        "${inference_script}" \
        "${eval_script}" \
        "${summarize_script}" \
        "${retrieval_ds}/state.json" \
        "${knowledge_base}" \
        "${eval_venv}/bin/activate" \
        "${eval_venv}/bin/python"; do
        [[ -e "${required}" ]] || {
            echo "Required benchmark dependency not found: ${required}" >&2
            return 1
        }
    done
    [[ -d "${image_dir}" ]] || {
        echo "InfoSeek image directory not found: ${image_dir}" >&2
        return 1
    }
    [[ -f "${model}/config.json" ]] || {
        echo "Merged model not found: ${model}" >&2
        return 1
    }
    if [[ "${mode}" == "berag" && ! -f "${prior_head}" ]]; then
        echo "BEFT prior head not found: ${prior_head}" >&2
        return 1
    fi
    if [[ -e "${predictions}" && "${OVERWRITE_BENCHMARK:-0}" != "1" ]]; then
        echo "Benchmark output already exists: ${predictions}" >&2
        echo "Use a different OUTPUT_ROOT or set OVERWRITE_BENCHMARK=1." >&2
        return 1
    fi

    mkdir -p "${exp_dir}"

    local common_args=(
        --mode "${mode}"
        --model "${model}"
        --tokenizer-path "${tokenizer}"
        --retrieval-ds-path "${retrieval_ds}"
        --knowledge-base "${knowledge_base}"
        --img-basedir "${image_dir}"
        --retrieval-field "${RETRIEVAL_FIELD:-retrieved_passage}"
        --retrieval-topk "${k}"
        --take-n "${take_n}"
        --seed "${seed}"
        --batch-size "${batch_size}"
        --max-model-len "${max_model_len}"
        --max-words-per-evidence "${MAX_WORDS_PER_EVIDENCE:-2048}"
        --max-tokens "${MAX_TOKENS:-64}"
        --dtype "${DTYPE:-bfloat16}"
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
        --output-path "${predictions}"
        --metrics-output-path "${metrics}"
        --no-save-full-prompts
        --no-resume-from-jsonl
        --write-on-batch-complete
        --stop-on-error
    )

    [[ "${CONTINUE_GENERATION_FROM_ANS:-1}" == "1" ]] \
        && common_args+=(--continue-generation-from-ans)
    [[ -n "${MAX_NUM_SEQS:-}" ]] \
        && common_args+=(--max-num-seqs "${MAX_NUM_SEQS}")
    [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] \
        && common_args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
    [[ "${ENFORCE_EAGER:-0}" == "1" ]] \
        && common_args+=(--enforce-eager)

    if [[ "${mode}" == "berag" ]]; then
        common_args+=(
            --prior-mode module
            --prior-head-path "${prior_head}"
            --prior-modeling "${PRIOR_MODELING:-mlp_head}"
            --prior-head-num-layers "${PRIOR_HEAD_NUM_LAYERS:-2}"
            --prior-head-proj-dim "${PRIOR_HEAD_PROJ_DIM:-1024}"
            --default-prior-token-offset "${DEFAULT_PRIOR_TOKEN_OFFSET:--5}"
            --num-accumulator-rows "${NUM_ACCUMULATOR_ROWS:-4096}"
            --pruning-top-p "${top_p}"
        )
    fi

    echo "[InfoSeek benchmark] Method: ${method_label}"
    echo "[InfoSeek benchmark] Sample: N=${take_n}, seed=${seed}, batch=${batch_size}"
    echo "[InfoSeek benchmark] K=${k}, TopP=${top_p}, max_model_len=${max_model_len}"
    echo "[InfoSeek benchmark] Output: ${exp_dir}"

    "${PYTHON_BIN:-python}" "${inference_script}" "${common_args[@]}"

    # Run subset scoring only after the vLLM process has released the model.
    source "${eval_venv}/bin/activate"
    PYTHON="${eval_venv}/bin/python" bash "${eval_script}" "${exp_dir}"
    "${eval_venv}/bin/python" "${summarize_script}" \
        --experiment-dir "${exp_dir}" \
        --method "${method_label}" \
        --top-p "${top_p}" \
        --max-model-len "${max_model_len}"
}
