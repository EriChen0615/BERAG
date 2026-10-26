# BERAG

Train Qwen3-VL-8B-Instruct with Bayesian Ensemble Fine-Tuning (BEFT) on Infoseek or E-VQA. For inference and evaluation, use [BERAG-demo](https://github.com/EriChen0615/BERAG-demo).

## 1. Install the training environment

Requires Linux, Python 3.11+, and an NVIDIA GPU with BF16 support. The commands below use one GPU.

```bash
git clone https://github.com/EriChen0615/BERAG.git
cd BERAG
export BERAG_ROOT="$PWD"
export TRAIN_DIR="$BERAG_ROOT/src/train/LlamaFactory-0.9.5-beft"

python3.11 -m venv "$TRAIN_DIR/.venv"
source scripts/activate_train_env.sh
python -m pip install -e "$TRAIN_DIR" unsloth packaging ninja
python -m pip install flash-attn --no-build-isolation
```

Install this repository's BEFT fork of LLaMA-Factory. Infoseek enables Unsloth; both recipes use FlashAttention-2. The repository does not include a frozen training environment.

## 2. Prepare training data

Training retrieval results are not yet published; reproduce retrieval using the upstream projects below. BERAG-demo downloads test data only. Replace `/absolute/path/...` with your artifacts and run preparation from `$BERAG_ROOT`.

### Infoseek

Follow [EchoSight](https://github.com/Go2Heart/EchoSight)'s data and retrieval instructions: use **EVA-CLIP-8B** against the **100K InfoSeek KB**, without its Q-Former reranker. Retrieve the top 100 documents for a 64K-example training sample (seed 42). Download the training CSV, KB, and OVEN images using the links in that repository.

The converter expects JSONL records with `question_id`, `ground_truth_url`, and `retrieved_pages`, whose entries contain `knowledge_base_index` and `score`. Preserve the KB ordering used to build the retrieval index; adapt upstream output to this format.

```bash
cd "$BERAG_ROOT"
RETRIEVAL_JSONL=/absolute/path/infoseek_train_evaclip_top100.sample64000.seed42.jsonl \
ECHOSIGHT_TRAIN_CSV=/absolute/path/infoseek_train_filtered.csv \
ECHOSIGHT_KB_JSON=/absolute/path/wiki_100_dict_v4.json \
IMG_BASEDIR=/absolute/path/infoseek_train_images \
bash scripts/curate/infoseek/echosight_rag2_beft_data.sh
```

This creates pairs containing one gold document and one random hard negative from the top five retrieved documents. The output is:

```text
src/train/LlamaFactory-0.9.5-beft/data/jinghong_chen/infoseek/echosight-rag2-hardneg-beft-size=0-max=32768/train_sharegpt.json
```

### E-VQA

Follow [PreFLMR](https://github.com/LinWeizheDragon/FLMR)'s environment, indexing, and retrieval instructions with **`LinWeizheDragon/PreFLMR_ViT-L`**, the **M2KR E-VQA training split and passages**, and `openai/clip-vit-large-patch14`. Download the E-VQA images through its dataset instructions.

Save training questions with `retrieved_passage` entries (`passage_id`, `score`) using `Dataset.save_to_disk()`. Keep the original `pos_item_ids` gold labels. BERAG's retrieval settings are in `config/EVQA/Retrieval_PreFLMR-L-train.jsonnet` and `config/retrievers.libsonnet`; the curator downloads the M2KR passages automatically.

```bash
cd "$BERAG_ROOT"
HF_DATASET_PATH=/absolute/path/EVQA-with-retrieval \
IMG_BASEDIR=/absolute/path/EVQA/images \
bash scripts/curate/evqa/beft_data.sh
```

This keeps two documents per example, ensuring that a gold document is present. It uses all training examples before filtering and writes:

```text
src/train/LlamaFactory-0.9.5-beft/data/jinghong_chen/evqa/beft_K=2-size=0-max=2048/train_sharegpt.json
```

Both outputs are already registered in LLaMA-Factory's `data/dataset_info.json`. Image paths in the generated JSON must point to existing files.

## 3. Run BEFT training

Set `TASK` to the dataset prepared above:

```bash
export TASK=infoseek  # or evqa
cd "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  "my_configs/$TASK/qwen3-vl-8B/beft/beft_k2_prior_mlp.yaml" \
  media_dir="$BERAG_ROOT" \
  tokenized_path="$BERAG_ROOT/outputs/tokenized/$TASK-qwen3-vl-8b-beft" \
  output_dir="$BERAG_ROOT/outputs/train/$TASK-qwen3-vl-8b-beft" \
  report_to=none
```

The checked-in recipes use one epoch, effective batch size 8, LoRA rank 64 / alpha 128, learning rate `1e-5`, and prior-head learning rate `1e-6`. The prior head has two layers, projection size 1024, and hidden-state offset 4. No separate SFT warm-up is required.

The command overrides machine-specific paths in the YAML. Use a new `tokenized_path` after changing the dataset or tokenization settings. Checkpoints are written under `outputs/train/$TASK-qwen3-vl-8b-beft/`.

## 4. Export for inference

Use the final training output, or set `ADAPTER_DIR` to a specific `checkpoint-*` directory:

```bash
export ADAPTER_DIR="$BERAG_ROOT/outputs/train/$TASK-qwen3-vl-8b-beft"
export EXPORT_DIR="$BERAG_ROOT/outputs/models/Qwen3-VL-8B-Instruct-BEFT-$TASK"

llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --adapter_name_or_path "$ADAPTER_DIR" \
  --template qwen3_vl_nothink \
  --trust_remote_code true \
  --infer_dtype bfloat16 \
  --export_dir "$EXPORT_DIR" \
  --export_device cpu

cp "$ADAPTER_DIR/prior_head.pt" "$EXPORT_DIR/prior_head.pt"
```

Keep `prior_head.pt` alongside the merged model weights. For BERAG inference, use two prior-head layers, projection size 1024, and `prior_token_offset=-5`, matching training's hidden-state offset 4.

## TODO

- [ ] Publish the E-VQA and Infoseek training retrieval results and prepared BEFT datasets.
- [ ] Release retrieval-to-training export scripts, including the Infoseek 64K sample IDs.
- [ ] Publish a pinned training environment and training-image download instructions.
