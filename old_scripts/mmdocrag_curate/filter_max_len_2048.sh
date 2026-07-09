#!/bin/bash
set -euo pipefail

MAX_LEN=2048
INPUT=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/third_party/LLaMA-Factory-2502/data/jinghong_chen/mmdocrag/rag8-mmdocrag-beft-size=0-offset=0-multimodal/train_sharegpt.json
OUTPUT=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/third_party/LLaMA-Factory-2502/data/jinghong_chen/mmdocrag/rag8-mmdocrag-beft-size=0-offset=0-multimodal/train_sharegpt_max_len=${MAX_LEN}.json

python3 /home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/scripts/mmdocrag_curate/length_distribution.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --max-len ${MAX_LEN} \
  --patch-size 14 \
  --merge-size 2
