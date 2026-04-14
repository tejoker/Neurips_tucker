#!/bin/bash
# run_ablation_suite.sh
# Orchestrates synthetic structural ablations (graph recovery metrics only).
# NOTE: This pipeline reports structural precision/recall/F1 against known ground truth
# and should not be mixed with anomaly detection PA-F1/Std-F1 used in the paper.

set -e

# Setup directories
DATA_DIR="results/ablation_metrics/data"
OUT_DIR="results/ablation_metrics/eval"
mkdir -p $DATA_DIR
mkdir -p $OUT_DIR

D=20
T=1000
PREFIX="${DATA_DIR}/synth_${D}d"

echo "=========================================="
echo "1. Generating Synthetic Data (d=${D}, T=${T})"
echo "=========================================="
python scripts/generate_ablation_data.py --d $D --t $T --output $PREFIX

DATA_NPY="${PREFIX}_data.npy"
COLS_NPY="${PREFIX}_columns.npy"
LAGS_NPY="${PREFIX}_lags.npy"
TRUE_EDGES="${PREFIX}_true_edges.csv"

echo ""
echo "=========================================="
echo "2. Running Architectural Ablations"
echo "=========================================="

# Define variants
declare -A VARIANTS=(
    ["baseline"]=""
    ["no_smoothness"]="--disable_smoothness"
    ["no_core_sparsity"]="--disable_core_sparsity"
    ["no_orthogonality"]="--disable_orthogonality"
    ["dense_tucker_disabled"]="--disable_tucker"
)

for VAR_NAME in "${!VARIANTS[@]}"; do
    FLAGS=${VARIANTS[$VAR_NAME]}
    echo "------------------------------------------"
    echo "Running Variant: ${VAR_NAME}"
    echo "------------------------------------------"
    
    # Train and extract structure
    python scripts/train_tucker_cam_ablation_metrics.py \
        --dataset $DATA_NPY \
        --columns $COLS_NPY \
        --lags $LAGS_NPY \
        --R_w 5 \
        --R_a 5 \
        --w_threshold 0.1 \
        --max_windows 2 \
        --output_dir $OUT_DIR \
        --identifier $VAR_NAME \
        $FLAGS
        
    # Evaluate against ground truth
    PRED_EDGES="${OUT_DIR}/predicted_edges_${VAR_NAME}.csv"
    STATS_JSON="${OUT_DIR}/stats_${VAR_NAME}.json"
    
    python scripts/evaluate_ablation_metrics.py \
        --predicted_csv $PRED_EDGES \
        --true_csv $TRUE_EDGES \
        --stats_json $STATS_JSON
done

echo ""
echo "=========================================="
echo "3. Generating NeurIPS Plots"
echo "=========================================="
python scripts/plot_ablation_results.py --results_dir $OUT_DIR --output_dir paper_plots/ablation

echo "Ablation Suite Completed Successfully."
