#!/bin/bash
# Generate paper figures from outputs/<experiment>_results.json files.
# Produces:
#   figures/per_class_auroc.png  - grouped bar chart with delta-vs-baseline panel
#   figures/macro_auroc.png      - macro AUROC across model variants
#   figures/summary_table.csv    - per-model metrics for the paper table
#
# Run evaluate.py for each (config, seed) first; this script aggregates the JSON outputs.

set -e

RESULTS_DIR="${RESULTS_DIR:-outputs}"
FIGURES_DIR="${FIGURES_DIR:-figures}"

# Order matters: the first model becomes the baseline in the delta-AUROC panel.
MODELS=(
    baseline_prenorm
    baseline_postnorm
    full_attnres
    block_attnres_2
    block_attnres_3
)

python analysis/plot_paper_figures.py \
    --results_dir "$RESULTS_DIR" \
    --figures_dir "$FIGURES_DIR" \
    --models "${MODELS[@]}"
