## Server Runbook (Real Datasets)

This runbook is for executing Tucker-CAM and NeurIPS ablations on a remote Linux server.

### 1) Environment

```bash
cd /path/to/tucker-cam
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended exports:

```bash
export PROJECT_ROOT="$(pwd)"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export DATA_ROOT="$PROJECT_ROOT/ServerMachineDataset"
export PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python3"
export USE_TUCKER_CAM=true
export USE_PARALLEL=true
export N_WORKERS=4
```

### 2) Datasets

- SMD:
  - Expected structure: `ServerMachineDataset/train`, `ServerMachineDataset/test`, `ServerMachineDataset/test_label`.
  - Optional conversion for launcher pipelines:
    ```bash
    python scripts/prepare_smd_npy.py
    ```
- Telemanom (SMAP/MSL):
  - Requires Kaggle API key at `~/.kaggle/kaggle.json`.
  - Download + preprocess:
    ```bash
    cd telemanom
    bash download_data.sh
    python prepare_datasets.py
    cd ..
    ```

### 3) Canonical NeurIPS Ablation

Run in a resilient session:

```bash
tmux new -s neurips_ablation
# inside tmux:
DEVICE=cpu SEEDS=5 MAX_ITER=80 DATA_ROOT=ServerMachineDataset \
  bash scripts/run_neurips_ablation.sh | tee logs/neurips_ablation.log
```

Detached alternative:

```bash
nohup bash scripts/run_neurips_ablation.sh > logs/neurips_ablation.log 2>&1 &
```

Outputs:

- `results/ablation_component`
- `results/ablation_rank_study`
- `results/neurips_ablation_bundle`

### 4) Full Benchmark Pipeline (Optional)

Single entity:

```bash
./scripts/run_entity_pipeline.sh machine-1-6 50
```

All entities:

```bash
python scripts/run_full_smd_benchmark_v9.py
```

### 5) Failure Recovery Checklist

- If interrupted, keep `RESULTS_ROOT` unchanged and rerun.
- Confirm data paths first:
  - `ls ServerMachineDataset/train | head`
  - `ls ServerMachineDataset/test_label | head`
- Confirm python binary:
  - `echo "$PYTHON_EXEC"`
  - `$PYTHON_EXEC --version`
- Tail logs:
  - `tail -f logs/neurips_ablation.log`

### 6) Reporting Rules for Paper

- Report anomaly metrics only from `ablation.run_component_ablation`.
- Do not mix structural graph F1 from synthetic pipelines with anomaly PA-F1/Std-F1.
- Use outputs in `results/neurips_ablation_bundle` as the table/figure source-of-truth.
