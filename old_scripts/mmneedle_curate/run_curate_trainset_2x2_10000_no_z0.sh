#!/usr/bin/env bash
set -euo pipefail

# Curate 10k training samples for 2x2 (no z0).
N_GRID=2 \
SEQUENCE_LENGTH=10 \
TAKE_N=10000 \
OFFSET=0 \
K=2 \
ADD_Z0=false \
SEED="${SEED:-0}" \
bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh

