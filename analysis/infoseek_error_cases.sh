# MARKED_RESULTS_CSV="analysis/gpt-4o-mini-2024-07-18/outputs/20250122-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_pretrained_PreFLMR-L/marked_results.csv"
# MARKED_RESULTS_CSV="analysis/gpt-4o-mini-2024-07-18/outputs/20250124-14-InfoseekNew_valid_m2kr-256_FullPassageOracleRetrieve[TopK]-Read_RetrieveTopK=1_GPT4o-mini_pretrained_PreFLMR-L/marked_results.csv"
MARKED_RESULTS_CSV="analysis/gpt-4o-mini-2024-07-18/outputs/20250124-14-InfoseekNew_valid_m2kr-256_FullPassageOracleRetrieve[TopK]-Read_RetrieveTopK=1_GPT4o-mini_pretrained_PreFLMR-L/makred_results.csv"
OUTPUT_DIR="analysis/infoseek_error_cases"

python analysis/infoseek_error_cases.py $MARKED_RESULTS_CSV \
    --threshold 10.0 \
    --output_dir $OUTPUT_DIR
