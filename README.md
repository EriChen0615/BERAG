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

Download the published training inputs; no retrieval run is required:

| Dataset | Training examples | Included resources |
| --- | ---: | --- |
| [E-VQA](https://huggingface.co/datasets/erichen-0615/EVQA-train-with-PreFLMR-L-retrieval) | 167,369 | PreFLMR-L Top-500, training images, passage corpus, retrieval index |
| [InfoSeek](https://huggingface.co/datasets/erichen-0615/Infoseek-train64000-with-PreFLMR-L-retrieval) | 64,000 | PreFLMR-L Top-500, original fixed 64K subset |

**Both training datasets use PreFLMR-L.** EchoSight EVA-CLIP-8B is used for InfoSeek evaluation. The released InfoSeek rankings require M2KR `Infoseek_passages/train_passages`, not EchoSight's 100K KB. InfoSeek images are available separately through [M2KR_Images](https://huggingface.co/datasets/BByrneLab/M2KR_Images); place the training images in one directory, retaining the filenames in `img_path`.

### Download questions and passages

Choose one dataset. The commands pin the published revisions, preserve row order, and save the local Arrow datasets expected by the curator.

```bash
cd "$BERAG_ROOT"
export TASK=infoseek  # or evqa

python - <<'PY'
import json
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

task = os.environ["TASK"]
releases = {
    "evqa": ("erichen-0615/EVQA-train-with-PreFLMR-L-retrieval", "9355b4d43b22ff4497e27f511382e53f00108235"),
    "infoseek": ("erichen-0615/Infoseek-train64000-with-PreFLMR-L-retrieval", "ea1c465d4f5830e09f576347db154cf427973570"),
}
repo, revision = releases[task]
root = Path(os.environ["BERAG_ROOT"])
raw = root / "outputs/data" / task
raw.mkdir(parents=True, exist_ok=True)
load_dataset(repo, "default", split="train", revision=revision).save_to_disk(str(raw / "questions"))
if task == "evqa":
    passages = load_dataset(repo, "passages", split="train", revision=revision)
    snapshot_download(repo, repo_type="dataset", revision=revision,
                      allow_patterns=["images/*"], local_dir=str(raw))
else:
    passages = load_dataset(
        "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
        "Infoseek_passages", split="train_passages",
        revision="5d8cf441601d964ef1e7eb2c191d81f20323f6b0",
    )
passages.save_to_disk(str(raw / "passages"))

# Register generated BEFT data separately from the existing EchoSight recipe.
data_dir = root / "outputs/data/beft"
data_dir.mkdir(parents=True, exist_ok=True)
registry = data_dir / "dataset_info.json"
info = json.loads(registry.read_text()) if registry.exists() else {}
source = json.loads((Path(os.environ["TRAIN_DIR"]) / "data/dataset_info.json").read_text())
entry = source["evqa_beft_k2_prior"].copy()
entry["file_name"] = f"{task}/train_sharegpt.json"
info[f"{task}_beft_preflmr"] = entry
registry.write_text(json.dumps(info, indent=2) + "\n")
PY
```

### Build BEFT examples

```bash
if [ "$TASK" = evqa ]; then
  for archive in "$BERAG_ROOT"/outputs/data/evqa/images/evqa_train_images-*.tar; do
    tar -xf "$archive" -C "$BERAG_ROOT/outputs/data/evqa"
  done
  export IMG_BASEDIR="$BERAG_ROOT/outputs/data/evqa/images"
  MAX_TOKENS=2048
else
  export IMG_BASEDIR=/absolute/path/to/infoseek_train_images
  MAX_TOKENS=32768
fi

python src/curate/ragk_answer_ppl.py \
  --hf_dataset_path "$BERAG_ROOT/outputs/data/$TASK/questions" \
  --passage_set_name "$BERAG_ROOT/outputs/data/$TASK/passages" \
  --mode sft --output_style beft --topk_docs 2 \
  --ensure_gt_passage_in_topk --sample_size 0 --seed 42 \
  --img_basedir "$IMG_BASEDIR" \
  --output_dir "$BERAG_ROOT/outputs/data/beft/$TASK" \
  --drop_max_tokens "$MAX_TOKENS"
```

This selects the top two passages, inserting a gold passage if absent, then filters long examples. It uses the published questions without resampling; image paths must resolve to existing files. Output: `outputs/data/beft/$TASK/train_sharegpt.json`.

To rerun retrieval, follow [PreFLMR](https://github.com/LinWeizheDragon/FLMR) with [`PreFLMR_ViT-L`](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L), `openai/clip-vit-large-patch14`, and the matching M2KR training corpus. E-VQA's released index is optional when using cached rankings; see the dataset cards for index and checksum details.

## 3. Run BEFT training

Keep `TASK` set to the dataset prepared above:

```bash
cd "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  "my_configs/$TASK/qwen3-vl-8B/beft/beft_k2_prior_mlp.yaml" \
  dataset="${TASK}_beft_preflmr" \
  dataset_dir="$BERAG_ROOT/outputs/data/beft" \
  media_dir="$BERAG_ROOT" \
  tokenized_path="$BERAG_ROOT/outputs/tokenized/$TASK-qwen3-vl-8b-beft-preflmr" \
  output_dir="$BERAG_ROOT/outputs/train/$TASK-qwen3-vl-8b-beft-preflmr" \
  report_to=none
```

The checked-in recipes use one epoch, effective batch size 8, LoRA rank 64 / alpha 128, learning rate `1e-5`, and prior-head learning rate `1e-6`. The prior head has two layers, projection size 1024, and hidden-state offset 4. No separate SFT warm-up is required.

The command overrides machine-specific paths and dataset registration in the YAML, including InfoSeek's existing EchoSight dataset default. Use a new `tokenized_path` after changing the dataset or tokenization settings. Checkpoints are written under `outputs/train/$TASK-qwen3-vl-8b-beft-preflmr/`.

## 4. Export for inference

Use the final training output, or set `ADAPTER_DIR` to a specific `checkpoint-*` directory:

```bash
export ADAPTER_DIR="$BERAG_ROOT/outputs/train/$TASK-qwen3-vl-8b-beft-preflmr"
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

- [x] Publish E-VQA and InfoSeek PreFLMR-L Top-500 training retrieval, preserving the original InfoSeek 64K subset.
- [x] Publish E-VQA training images, passage corpus, and retrieval index.
- [ ] Publish prepared, model-specific BEFT datasets and a pinned training environment.
