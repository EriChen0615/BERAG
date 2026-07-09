#!/usr/bin/env bash

# Small helper to run the RULER table reporter.
# Usage:
#   ./analysis/run_ruler_table.sh [ROOT_DIR] [OUT_CSV]
# Examples:
#   ./analysis/run_ruler_table.sh
#   ./analysis/run_ruler_table.sh third_party/RULER/scripts/outputs/0326/ruler analysis/output/ruler_table_0326.csv

set -euo pipefail

# Change to project root (this script lives in analysis/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}" || exit 1

# Default arguments
ROOT_DIR="${1:-third_party/RULER/scripts/outputs/0326/ruler}"
OUT_CSV="${2:-analysis/output/ruler_table.csv}"

echo "Running report_ruler_table.py"
echo "  ROOT_DIR = ${ROOT_DIR}"
echo "  OUT_CSV  = ${OUT_CSV}"

python3 analysis/report_ruler_table.py "${ROOT_DIR}" -o "${OUT_CSV}"

