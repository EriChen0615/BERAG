# InfoSeek inference benchmark

These launchers benchmark Qwen3-VL-8B on the same deterministic 64-example
InfoSeek sample with parent batch size 64 and K=50:

```bash
bash scripts/infer/inference_benchmark/infoseek/run_rag_k50.sh
bash scripts/infer/inference_benchmark/infoseek/run_berag_k50.sh
bash scripts/infer/inference_benchmark/infoseek/run_berag_topp_k50.sh
```

To compare BERAG and Top-P with one parent request in the engine at a time, run:

```bash
bash scripts/infer/inference_benchmark/infoseek/run_berag_batch1_comparison.sh
```

This writes to `outputs/inference_benchmark/infoseek_batch1/` by default. Top-P
directories include the threshold to three decimal places, for example
`qwen3-vl-8b/berag-topp-p=0.990/K=50-TakeN=64/`.

The Top-P run uses `1 - 1 / (2K)`, which is `0.99` at K=50. RAG defaults to
`max_model_len=262144`, matching Qwen3-VL's native text context limit; override
it with `RAG_MAX_MODEL_LEN` if necessary.
BERAG defaults to 4096 accumulator rows so all 64×50 branches can be admitted;
this uses roughly 2 GiB for Qwen3's vocabulary-sized row workspace. Override
`NUM_ACCUMULATOR_ROWS` if the GPU needs a smaller workspace.
Each run performs subset evaluation afterward and writes `run_metrics.json`,
`score.json`, and `benchmark_row.json` below
`outputs/inference_benchmark/infoseek/qwen3-vl-8b/`.

For the table's Score column, use `subset_score` (`mean_example_score`) from
`benchmark_row.json`. The official split-balanced score is retained separately
but is not the appropriate aggregate for an arbitrary 64-example subset.
`wall_time_s` measures the vLLM generation call only; it excludes model loading,
dataset preparation, and evaluation. Existing benchmark predictions are not
resumed because that would invalidate timing. Set `OVERWRITE_BENCHMARK=1` to
replace a previous run deliberately.
