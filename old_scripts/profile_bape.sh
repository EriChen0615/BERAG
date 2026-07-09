#!/bin/bash

# Profile BAPE inference with small dataset
echo "Starting BAPE inference profiling..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

# Run profiling with small dataset
python src/profile_bape_inference.py \
    --take_n 3 \
    --max_tokens 30

echo "Profiling completed!"
