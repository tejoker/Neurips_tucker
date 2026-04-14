#!/bin/bash
# Tensor Rank Ablation Orchestration Script
# Generates synthetic data and runs the train_tucker_cam_ablation.py script
# across a careful parameter grid to measure time/memory complexity.

set -e

# Directories
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DATA_DIR="${SCRIPTS_DIR}/../data/ablation"
OUT_DIR="${SCRIPTS_DIR}/../results/ablation"

mkdir -p "$DATA_DIR"
mkdir -p "$OUT_DIR"

echo "======================================"
echo " Starting Tensor Rank Ablation Study"
echo "======================================"

# 1. Generate Datasets
echo "[1/3] Generating Synthetic Datasets..."
python3 "${SCRIPTS_DIR}/generate_ablation_data.py" --d 55 --t 200 --output "${DATA_DIR}/msl_mock"
python3 "${SCRIPTS_DIR}/generate_ablation_data.py" --d 500 --t 150 --output "${DATA_DIR}/highd_mock"

# Arrays of ranks
declare -a RANKS_MSL=(5 10 20 30 40)
declare -a RANKS_HIGHD=(10 20 30 50 80)

MAX_WINDOWS=5 # Keep it short for time/memory profiling

# 2. Run MSL (d=55) Grid
echo ""
echo "[2/3] Running Ablation on MSL Datasets (d=55)"
for r in "${RANKS_MSL[@]}"; do
    # Vary R_w (Keep R_a fixed at 10)
    echo "  -> Profiling MSL: R_w=$r, R_a=10"
    python3 "${SCRIPTS_DIR}/train_tucker_cam_ablation.py" \
        --dataset "${DATA_DIR}/msl_mock_data.npy" \
        --columns "${DATA_DIR}/msl_mock_columns.npy" \
        --lags "${DATA_DIR}/msl_mock_lags.npy" \
        --R_w $r --R_a 10 \
        --max_windows $MAX_WINDOWS \
        --output_dir "${OUT_DIR}/msl"
        
    # Vary R_a (Keep R_w fixed at 20)
    # Skip if r=10 since we just did (20, 10) above or will do it.
    if [ "$r" -ne 10 ] && [ "$r" -ne 20 ]; then
        echo "  -> Profiling MSL: R_w=20, R_a=$r"
        python3 "${SCRIPTS_DIR}/train_tucker_cam_ablation.py" \
            --dataset "${DATA_DIR}/msl_mock_data.npy" \
            --columns "${DATA_DIR}/msl_mock_columns.npy" \
            --lags "${DATA_DIR}/msl_mock_lags.npy" \
            --R_w 20 --R_a $r \
            --max_windows $MAX_WINDOWS \
            --output_dir "${OUT_DIR}/msl"
    fi
done

# 3. Run High-D (d=500) Grid
echo ""
echo "[3/3] Running Ablation on High-D Datasets (d=500)"
for r in "${RANKS_HIGHD[@]}"; do
    # Vary R_w (Keep R_a fixed at 10)
    echo "  -> Profiling High-D: R_w=$r, R_a=10"
    python3 "${SCRIPTS_DIR}/train_tucker_cam_ablation.py" \
        --dataset "${DATA_DIR}/highd_mock_data.npy" \
        --columns "${DATA_DIR}/highd_mock_columns.npy" \
        --lags "${DATA_DIR}/highd_mock_lags.npy" \
        --R_w $r --R_a 10 \
        --max_windows $MAX_WINDOWS \
        --output_dir "${OUT_DIR}/highd"
        
    # Vary R_a (Keep R_w fixed at 20)
    if [ "$r" -ne 10 ] && [ "$r" -ne 20 ]; then
        echo "  -> Profiling High-D: R_w=20, R_a=$r"
        python3 "${SCRIPTS_DIR}/train_tucker_cam_ablation.py" \
            --dataset "${DATA_DIR}/highd_mock_data.npy" \
            --columns "${DATA_DIR}/highd_mock_columns.npy" \
            --lags "${DATA_DIR}/highd_mock_lags.npy" \
            --R_w 20 --R_a $r \
            --max_windows $MAX_WINDOWS \
            --output_dir "${OUT_DIR}/highd"
    fi
done

echo ""
echo "======================================"
echo " Ablation Grid Search Complete!     "
echo " Results saved to ${OUT_DIR}"
echo "======================================"
