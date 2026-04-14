#!/bin/bash
# =============================================================================
# Tucker-CAM Ablation Study -- Full Orchestration Script
#
# Runs both experiments:
#   1. Component ablation (Table: what matters?)
#   2. Rank sensitivity study (Figure: what rank to pick?)
#
# These are SEPARATE experiments with SEPARATE purposes.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-5}"
MAX_ITER="${MAX_ITER:-80}"

echo "======================================================================"
echo " Tucker-CAM Ablation Study for NeurIPS"
echo "======================================================================"
echo " Device: $DEVICE"
echo " Seeds:  $SEEDS"
echo " Max iterations per window: $MAX_ITER"
echo ""

# -------------------------------------------
# Experiment 1: Component Ablation
# -------------------------------------------
echo "----------------------------------------------------------------------"
echo " [1/2] Component Ablation Study"
echo "        (Tests contribution of each model component)"
echo "----------------------------------------------------------------------"
echo ""

python -m ablation.run_component_ablation \
    --entities machine-1-1 machine-2-1 machine-3-1 \
    --seeds "$SEEDS" \
    --device "$DEVICE" \
    --max-iter "$MAX_ITER" \
    --output-dir results/ablation_component

echo ""
echo " Component ablation complete. Results in results/ablation_component/"
echo ""

# -------------------------------------------
# Experiment 2: Rank Sensitivity Study
# -------------------------------------------
echo "----------------------------------------------------------------------"
echo " [2/2] Tensor Rank Sensitivity Study"
echo "        (Hyperparameter tuning -- NOT an ablation)"
echo "----------------------------------------------------------------------"
echo ""

python -m ablation.run_rank_study \
    --entity machine-1-1 \
    --ranks-w 5 10 15 20 30 40 \
    --ranks-a 5 10 15 20 30 \
    --seeds "$SEEDS" \
    --device "$DEVICE" \
    --max-iter "$MAX_ITER" \
    --output-dir results/ablation_rank_study

echo ""
echo " Rank study complete. Results in results/ablation_rank_study/"
echo ""

# -------------------------------------------
# Summary
# -------------------------------------------
echo "======================================================================"
echo " All experiments complete."
echo ""
echo " Outputs:"
echo "   results/ablation_component/"
echo "     - ablation_raw_results.csv    (per-seed, per-entity raw data)"
echo "     - ablation_aggregated.json    (mean/std per variant)"
echo "     - ablation_table.tex          (LaTeX table for paper)"
echo ""
echo "   results/ablation_rank_study/"
echo "     - rank_study_raw.csv          (per-seed, per-rank raw data)"
echo "     - rank_study_summary.json     (aggregated rank sensitivity)"
echo ""
echo " Quick test (single entity, 2 seeds, 2 variants):"
echo "   python -m ablation.run_component_ablation \\"
echo "     --entities machine-1-1 --seeds 2 --variants full linear"
echo "======================================================================"
