# EVQA gold-document position datasets

Curated subsets and position-swap variants are produced by:

```bash
# From repo root, with your Python env already activated:
sbatch scripts/analysis/slurm_curate_evqa_gtdoc_position.sh
```

Or directly:

```bash
python scripts/analysis/curate_evqa_gtdoc_position.py --help
python scripts/analysis/curate_evqa_gtdoc_position.py
```

After curation, run inference (GPU; activate env first):

```bash
sbatch scripts/evqa_vllm/run_positional_analysis_7b.sh
sbatch scripts/evqa_bape/run_positional_analysis_7b_beft_l0h4.sh
```

Outputs go to `outputs/0426/EVQA-positional-analysis/`. If you curated with a different `--n_sample`, set `N` when running the GPU scripts (default `256`).

**Sanity checks**

```bash
python -c "from datasets import load_from_disk; d=load_from_disk('analysis/EVQA-gtdoc-position-datasets/EVQA-256-gtdoc_at_1-4'); print(len(d))"
```

```bash
python - <<'PY'
from datasets import load_from_disk
def rank(ex):
    gt = str(ex["pos_item_ids"][0])
    for i, p in enumerate(ex["retrieved_passage"][:20]):
        if str(p["passage_id"]) == gt:
            return i + 1
    return None
d = load_from_disk("analysis/EVQA-gtdoc-position-datasets/EVQA-256-gtdoc_at_1-4")
assert all(1 <= rank(ex) <= 4 for ex in d)
print("ok")
PY
```

Repeat the rank bounds `1–4` for other folders: `5–8`, `9–12`, `13–16`, `17–20`.
