#!/usr/bin/env bash
# Run BERAG ESF beam log visualization. Activates py310 inference env and runs visualize_berag_eft.py.
#
# Usage:
#   ./analysis/run_visualize_berag_eft.sh <beam_log.jsonl> [--output out.png] [--no-show]
#   ./analysis/run_visualize_berag_eft.sh outputs/0226/SimpleNIAH/results/simple_niah_berag_esf_len_100000_depth_0_results.beam_log.jsonl -o analysis/output/vis.png --no-show

python analysis/visualize_berag_eft.py \
    outputs/0226/SimpleNIAH/results/simple_niah_berag_esf_len_100000_depth_4800_results.beam_log.jsonl
#    outputs/0226/SimpleNIAH/results/simple_niah_berag_esf_len_100000_depth_0_results.beam_log.jsonl \
