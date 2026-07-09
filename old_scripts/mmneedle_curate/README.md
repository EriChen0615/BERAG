# MMNeedle Curation + BEFT Training

This folder contains a staged pipeline to:

1. build stitched images (`1x1`, `2x2`) from COCO train2014,
2. generate MMNeedle-style single-needle annotations,
3. convert those annotations into BEFT-compatible `train_sharegpt.json`,
4. launch BEFT training for `xtuner/llava-llama-3-8b` in LLaMA-Factory.

## Prerequisites

- COCO train images at:
  - `/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MSCOCO2014/train2014`
- COCO captions at:
  - `../vqa_data/MSCOCO2014/annotations/captions_train2014.json`
- Optional env activation:
  - `scripts/hpc_activate_env_py310_infer.sh`

## Stage 1: Create stitched images

Single N:

```bash
N_GRID=1 NUM_IMAGES=200 bash scripts/mmneedle_curate/run_sample_stitched_images.sh
N_GRID=2 NUM_IMAGES=200 bash scripts/mmneedle_curate/run_sample_stitched_images.sh
```

Both N=1 and N=2:

```bash
RUN_BOTH=true NUM_IMAGES=200 bash scripts/mmneedle_curate/run_sample_stitched_images.sh
```

Outputs:

- `../vqa_data/MMNeedle/train/images_stitched/1_1/`
- `../vqa_data/MMNeedle/train/images_stitched/2_2/`
- `../vqa_data/MMNeedle/train/metadata_stitched/1_1.json`
- `../vqa_data/MMNeedle/train/metadata_stitched/2_2.json`

Inspect these artifacts before continuing.

## Stage 2: Sample single-needle annotations

Single N:

```bash
N_GRID=1 NUM_SEQUENCES=1000 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_sample_single_needle_nxn.sh
N_GRID=2 NUM_SEQUENCES=1000 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_sample_single_needle_nxn.sh
```

Both N=1 and N=2:

```bash
RUN_BOTH=true NUM_SEQUENCES=1000 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_sample_single_needle_nxn.sh
```

Outputs:

- `../vqa_data/MMNeedle/train/metadata_stitched/annotations_10_1_1.json`
- `../vqa_data/MMNeedle/train/metadata_stitched/annotations_10_2_2.json`

Inspect these artifacts before continuing.

## Stage 3: Curate BEFT training data

Single N:

```bash
N_GRID=1 TAKE_N=0 OFFSET=0 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh
N_GRID=2 TAKE_N=0 OFFSET=0 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh
```

Both N=1 and N=2:

```bash
RUN_BOTH=true TAKE_N=0 OFFSET=0 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh
```

Outputs (examples):

- `../vqa_data/MMNeedle/train/curated/rag10-mmneedle-n1x1-beft-size=0-offset=0/train_sharegpt.json`
- `../vqa_data/MMNeedle/train/curated/rag10-mmneedle-n2x2-beft-size=0-offset=0/train_sharegpt.json`
- `stats.json` in each curated directory

Confirm curated outputs before training.

## Stage 4: Launch BEFT training (LLaVA-Llama-3-8B)

```bash
N_GRID=1 SEQUENCE_LENGTH=10 TAKE_N=0 OFFSET=0 bash scripts/mmneedle_curate/train_beft_mmneedle_llava_llama3_8b.sh
```

or

```bash
N_GRID=2 SEQUENCE_LENGTH=10 TAKE_N=0 OFFSET=0 bash scripts/mmneedle_curate/train_beft_mmneedle_llava_llama3_8b.sh
```

This script:

- registers/updates a dataset entry in `third_party/LLaMA-Factory-2502/data/dataset_info.json`,
- writes a generated YAML under `third_party/LLaMA-Factory-2502/my_configs/mmneedle/beft/`,
- runs `llamafactory-cli train ...`.

## Parameter Cheatsheet

- `N_GRID`: `1` or `2`
- `NUM_IMAGES`: stitched image count for Stage 1
- `NUM_SEQUENCES`: annotation sample count for Stage 2
- `SEQUENCE_LENGTH`: number of stitched images per sample
- `TAKE_N`: cap curated training size (`0` = all)
- `OFFSET`: start offset before `TAKE_N`
- `SEED`: random seed

