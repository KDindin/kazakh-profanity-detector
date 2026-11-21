#!/usr/bin/env bash
# End-to-end rebuild on Linux / macOS.  Usage: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
    ./.venv/bin/pip install -e .
fi

for step in 01_fetch_sources 02_mine 03_build_pool 04_build_dataset \
            05_train_baseline 06_train_transformer 07_evaluate 08_error_analysis 15_record_env; do
    echo "=== $step"
    ./.venv/bin/python "scripts/$step.py"
done

echo "Done. See reports/metrics.json and reports/error_analysis.md"
