#!/bin/bash

# Script to run prior logits abstention analysis for multiple experiments
# Usage: 
#   ./run_prior_logits_abstention_analysis.sh
#   ./run_prior_logits_abstention_analysis.sh --experiments path1.csv path2.csv --Ks 1 2 [--dataset Infoseek|EVQA] [--deflection-mechanism max_prior_logits|weighted_prior_logits|max_z0_logits]
# 
# Default: Uses dictionary-like experiment group configuration
# max_z0_logits: If z0 logit (last passage) is maximum, deflect (model not using passages)

# Set the base directory (adjust as needed)
BASE_DIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
cd "$BASE_DIR" || exit 1

# Check if experiments are provided as command-line arguments
if [ "$#" -gt 0 ] && [ "$1" == "--experiments" ]; then
    # Use command-line arguments
    echo "Running analysis with provided experiments..."
    python analysis/analyze_prior_logits_abstention.py "$@"
    exit 0
fi

# ============================================================================
# Experiment Group Configuration (Dictionary-like)
# ============================================================================

# Group 1: Experiments without z0
declare -a group1_experiments=(
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=1-h4-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=2-h4-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=3-h4-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=5-h4-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
)

declare -a group1_ks=(1 2 3 5)

declare -A group1=(
    [name]="beft_no_z0"
    [dataset]="Infoseek"
    [deflection_mechanism]="weighted_prior_logits"
    [report_name]="prior_logits_abstention_report_beft_no_z0_n=256"
    [enabled]="true"
)

# Group 2: Experiments with z0
declare -a group2_experiments=(
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-z0-data=64000-K=1-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-z0-data=64000-K=2-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-z0-data=64000-K=3-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-z0-data=64000-K=5-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
)

declare -a group2_ks=(1 2 3 5)

declare -A group2=(
    [name]="beft_with_z0"
    [dataset]="Infoseek"
    [deflection_mechanism]="max_z0_logits"
    [report_name]="prior_logits_abstention_report_beft_with_z0_n=0"
    [enabled]="true"  # Set to "true" to enable this group
)

# Group 3: Experiments with z0, model continued training
declare -a group3_experiments=(
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=1-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=2-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=3-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=5-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=256/marked_inference_results.csv"
)

declare -a group3_ks=(1 2 3 5)

declare -A group3=(
    [name]="beft_cont_with_z0"
    [dataset]="Infoseek"
    [deflection_mechanism]="max_z0_logits"
    [report_name]="prior_logits_abstention_report_beft_cont_with_z0_n=256"
    [enabled]="true"  # Set to "true" to enable this group
)

# Group 3: Experiments with z0, model continued training
declare -a group4_experiments=(
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=1-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=2-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=3-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
    "outputs/1225/BAPE/Infoseek/InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000-K=5-h4-withZ0-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"
)

declare -a group4_ks=(1 2 3 5)

declare -A group4=(
    [name]="beft_cont_with_z0_n=0"
    [dataset]="Infoseek"
    [deflection_mechanism]="max_z0_logits"
    [report_name]="prior_logits_abstention_report_beft_cont_with_z0_n=0"
    [enabled]="true"  # Set to "true" to enable this group
)

# ============================================================================
# Function to run analysis for an experiment group
# ============================================================================

run_group_analysis() {
    local group_ref="$1"
    local experiments_ref="$2"
    local ks_ref="$3"
    declare -n group="$group_ref"
    declare -n experiments="$experiments_ref"
    declare -n ks="$ks_ref"
    
    # Check if group is enabled
    if [[ "${group[enabled]}" != "true" ]]; then
        echo "Skipping group: ${group[name]} (disabled)"
        return 0
    fi
    
    echo "========================================"
    echo "Running analysis for group: ${group[name]}"
    echo "  Deflection mechanism: ${group[deflection_mechanism]}"
    echo "  Dataset: ${group[dataset]}"
    echo "  Report name: ${group[report_name]}"
    echo "  Number of experiments: ${#experiments[@]}"
    echo "========================================"
    
    # Validate arrays have same length
    if [ ${#experiments[@]} -ne ${#ks[@]} ]; then
        echo "Error: Number of experiments (${#experiments[@]}) does not match number of K values (${#ks[@]}) in group ${group[name]}"
        return 1
    fi
    
    # Run the analysis
    python analysis/analyze_prior_logits_abstention.py \
        --experiments "${experiments[@]}" \
        --Ks "${ks[@]}" \
        --dataset "${group[dataset]}" \
        --deflection-mechanism "${group[deflection_mechanism]}" \
        --report_csv_path "analysis/output/${group[report_name]}.csv"
    
    echo ""
    echo "Group ${group[name]} analysis complete!"
    echo "Results saved to: analysis/output/${group[report_name]}.csv"
    echo "JSON files saved in each experiment directory (with _${group[deflection_mechanism]} suffix)"
    echo ""
}

# ============================================================================
# Main execution: Run all enabled groups
# ============================================================================

echo "Starting prior logits abstention analysis..."
echo ""

# Run each group
run_group_analysis group1 group1_experiments group1_ks
run_group_analysis group2 group2_experiments group2_ks
# run_group_analysis group3 group3_experiments group3_ks
# run_group_analysis group4 group4_experiments group4_ks

echo "All analyses complete!"

