#!/bin/bash
# Canonical NeurIPS ablation pipeline:
# 1) Component ablation on real SMD entities (with CP vs Tucker, multi-seed stats)
# 2) Rank sensitivity on real SMD
# 3) Bundle outputs for paper provenance

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-5}"
MAX_ITER="${MAX_ITER:-80}"
DATA_ROOT="${DATA_ROOT:-ServerMachineDataset}"
ENTITIES="${ENTITIES:-machine-1-1 machine-2-1 machine-3-1}"
RANK_ENTITY="${RANK_ENTITY:-machine-1-1}"

echo "==============================================================="
echo " NeurIPS Canonical Ablation Pipeline"
echo "==============================================================="
echo "Device:      $DEVICE"
echo "Seeds:       $SEEDS"
echo "Max iter:    $MAX_ITER"
echo "Data root:   $DATA_ROOT"
echo "Entities:    $ENTITIES"
echo "Rank entity: $RANK_ENTITY"
echo ""

python -m ablation.run_component_ablation \
  --profile neurips \
  --entities $ENTITIES \
  --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" \
  --device "$DEVICE" \
  --max-iter "$MAX_ITER" \
  --output-dir results/ablation_component

python -m ablation.run_rank_study \
  --entity "$RANK_ENTITY" \
  --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" \
  --device "$DEVICE" \
  --max-iter "$MAX_ITER" \
  --output-dir results/ablation_rank_study

python scripts/compile_neurips_ablation.py \
  --component-dir results/ablation_component \
  --rank-dir results/ablation_rank_study \
  --out-dir results/neurips_ablation_bundle

echo ""
echo "Done. Paper-ready bundle: results/neurips_ablation_bundle/"
