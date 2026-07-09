#!/usr/bin/env bash
set -euo pipefail

# Sample 10k MMNeedle annotations for 8x8.
N_GRID=8 \
SEQUENCE_LENGTH=10 \
NUM_SEQUENCES=10000 \
POSITIVE_RATIO=0.5 \
SEED="${SEED:-0}" \
bash scripts/mmneedle_curate/run_sample_single_needle_nxn.sh

