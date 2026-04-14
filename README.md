# Tucker-CAM: Nonlinear Causal Discovery via Tucker-Decomposed P-Splines

Detects anomalies in high-dimensional multivariate time series by learning **Non-Linear Dynamic Bayesian Networks (DBNs)** and tracking structural changes in the causal graph.

**Tucker-CAM** combines P-splines for nonlinear functional modeling with Tucker Tensor Decomposition to compress the parameter space from $O(d^2 K)$ to $O(dr)$, enabling scalable causal discovery on thousands of variables.

Acyclicity is enforced via NOTEARS: $h(W) = \mathrm{tr}(e^{W \circ W}) - d = 0$.

---

## Validation

| Metric | Value | Setting |
|---|---|---|
| PA F1 | 0.701 ± 0.088 | Ablation, n=15 runs |
| PA F1 (best rank r=40) | 0.912 ± 0.027 | Rank study, 5 seeds |
| Tucker vs. Dense | +0.051 PA F1 | No Tucker: 0.650 |

See `tier_1_tier_2_validation.ipynb` for live demo + full pre-computed results.

---

## Datasets

- **SMD** (Server Machine Dataset): IT operations monitoring, multi-dimensional KPIs from server machines.
- **NASA SMAP/MSL (Telemanom)**: Spacecraft telemetry with expert-verified anomaly labels.

---

## Quick Start

```bash
pip install -r requirements.txt

# Single run
python executable/launcher.py \
    --baseline data/Golden/golden_period_dataset_mean_channel.csv \
    --test data/Anomaly/telemanom/isolated_anomaly_001_P-1_seq1.csv \
    --output results/my_run

# SMD full entity pipeline
./scripts/run_entity_pipeline.sh machine-1-6 50
```

---

## How It Works

1. **Preprocessing** (`preprocessing_no_mi.py`)  
   ADF/KPSS stationarity tests, differencing, standardization, lag selection via AutoReg AIC.

2. **Causal discovery** (`dynotears_tucker_cam.py`)  
   Tucker-CAM-DAG: Tucker-decomposed P-spline coefficients + NOTEARS acyclicity constraint (augmented Lagrangian). Outputs contemporaneous W and lagged A matrices.

3. **Graph comparison** (`binary_detection_metrics.py`)  
   Computes 4 metrics between baseline and test graphs: SHD, Frobenius norm, spectral radius, max edge change.

4. **Anomaly detection**  
   Voting ensemble: ≥2 of 4 metrics above threshold → anomaly. Thresholds learned from baseline distribution (μ + k·σ).

5. **Root cause analysis** (`root_cause_analysis.py`)  
   Ranks variables by contribution to anomaly score via edge importance.

---

## Repository Layout

```
executable/
├── final_pipeline/
│   ├── cam_model_tucker.py          # Tucker-CAM model (core math)
│   ├── dynotears_tucker_cam.py      # Tucker-CAM-DAG optimizer (NOTEARS)
│   ├── preprocessing_no_mi.py       # Stationarity, lag selection
│   ├── dynotears.py                 # Base DynoTEARS
│   ├── structuremodel.py
│   ├── transformers.py
│   └── window_by_window_detection.py
├── experiments/                     # Ablation tools
└── launcher.py                      # Main orchestrator

tier_1_tier_2_validation.ipynb       # Validation notebook (Tier 1 + Tier 2)

scripts/                             # Batch pipeline runners
config/
├── default.yaml                     # Hyperparameters
└── config_manager.py
docs/
└── SERVER_REAL_DATASET_RUNBOOK.md   # GPU server runbook
tests/
results/                             # Pre-computed benchmark outputs (gitignored)
data/                                # Datasets (gitignored)
```

---

## Key Parameters

```yaml
# Tucker ranks (expressiveness vs. memory)
rank_w: 20          # Contemporaneous edges
rank_a: 10          # Lagged edges

# Acyclicity
h_tol: 1e-8         # NOTEARS tolerance
max_iter: 100       # Optimization iterations
lr: 0.01            # Learning rate

# Penalties
lambda_smooth: 0.01  # P-spline smoothness
lambda_core: 0.01    # Core tensor sparsity (prevents smearing)
lambda_orth: 0.001   # Factor orthogonality

# Detection
threshold_multiplier: 2.5   # μ + k·σ
voting_threshold: 2          # Min metrics for anomaly (out of 4)
```

---

## Setup

Python 3.9+, PyTorch. GPU optional but recommended for large datasets.

```bash
pip install -r requirements.txt

# GPU (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Output Structure

```
results/my_run/
├── preprocessing/
│   ├── differenced_data.npy
│   └── optimal_lags.npy
├── causal_discovery/
│   ├── window_edges.npy     # Tucker-CAM edge weights per window
│   └── progress.txt
├── detection/
│   └── metric_scores.csv
└── root_cause/
    └── edge_importance.csv
```

---

## Limitations

- Anomaly detection assumes anomalies manifest as causal structure changes.
- Tucker compression gains are significant at d≥100 (34x). At d=15 (demo): 1.6x.
- Short CPU runs (150 iters) give recall-biased results; full GPU pipeline needed for precision.
- Detection thresholds are dataset-specific.

---

## References

- Zheng et al. (2018). *DAGs with NO TEARS*. NeurIPS.
- DynoTEARS: dynamic extension for time series DBNs.
