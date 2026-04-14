#!/usr/bin/env python3
"""
Component Ablation Study for Tucker-CAM (NeurIPS).

Runs all ablation variants on SMD entities with multiple seeds,
computes PA-F1 / standard F1 / AUC-PR, and reports statistics
with confidence intervals and paired t-tests.

Usage:
    # Full ablation (all variants, 5 seeds, 3 entities)
    python -m ablation.run_component_ablation

    # Quick test (single entity, 2 seeds, 2 variants)
    python -m ablation.run_component_ablation \
        --entities machine-1-1 \
        --seeds 2 \
        --variants full linear

    # Specific device
    python -m ablation.run_component_ablation --device cuda
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.train import (
    VARIANTS,
    run_rolling_windows,
    compute_anomaly_scores,
    window_labels_from_point_labels,
)
from ablation.evaluate import (
    full_evaluation,
    aggregate_seeds,
    paired_t_test,
    format_result_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default entities: diverse across machine groups for representativeness
DEFAULT_ENTITIES = ["machine-1-1", "machine-2-1", "machine-3-1"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
NEURIPS_VARIANTS = [
    "full",
    "no_tucker",
    "cp_decomposition",
    "single_metric",
    "l1_sparsity",
]


def load_smd_entity(entity: str, data_root: str = "ServerMachineDataset"):
    """Load train/test/labels for one SMD entity."""
    root = Path(data_root)
    train = np.loadtxt(root / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(root / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(root / "test_label" / f"{entity}.txt", delimiter=",")
    return train, test, labels


def run_single_experiment(
    variant_name: str,
    entity: str,
    seed: int,
    train_data: np.ndarray,
    test_data: np.ndarray,
    labels: np.ndarray,
    p: int = 5,
    window_size: int = 100,
    stride: int = 10,
    rank_w: int = 20,
    rank_a: int = 10,
    n_knots: int = 5,
    max_iter: int = 80,
    device: str = "cpu",
) -> dict:
    """
    Run a single (variant, entity, seed) experiment.

    Returns dict with all evaluation metrics + timing info.
    """
    variant_config = VARIANTS[variant_name]
    detection_mode = variant_config.get("detection_mode", "multi")
    threshold_mode = variant_config.get("threshold_mode", "adaptive")

    logger.info(
        f"  [{variant_name}] entity={entity} seed={seed} "
        f"(model={variant_config['model_type']})"
    )

    t0 = time.time()

    # Train: rolling windows on normal data
    golden = run_rolling_windows(
        train_data,
        variant_name=variant_name,
        p=p,
        window_size=window_size,
        stride=stride,
        rank_w=rank_w,
        rank_a=rank_a,
        n_knots=n_knots,
        max_iter=max_iter,
        device=device,
        seed=seed,
    )

    if len(golden) == 0:
        logger.warning(f"  No golden windows produced for {entity}")
        return None

    # Test: rolling windows on test data
    test_results = run_rolling_windows(
        test_data,
        variant_name=variant_name,
        p=p,
        window_size=window_size,
        stride=stride,
        rank_w=rank_w,
        rank_a=rank_a,
        n_knots=n_knots,
        max_iter=max_iter,
        device=device,
        seed=seed,
    )

    if len(test_results) == 0:
        logger.warning(f"  No test windows produced for {entity}")
        return None

    total_time = time.time() - t0

    # Compute anomaly scores
    scoring = compute_anomaly_scores(
        golden, test_results,
        detection_mode=detection_mode,
        threshold_mode=threshold_mode,
    )

    # Convert point labels to window labels
    n_test_windows = len(test_results)
    y_true = window_labels_from_point_labels(
        labels, n_test_windows,
        window_size=window_size, stride=stride, p=p,
    )

    # Evaluate
    y_score = scoring["scores"][:n_test_windows]
    y_true = y_true[:n_test_windows]

    metrics = full_evaluation(y_score, y_true)

    # Also evaluate at the model's own threshold (not swept)
    # This captures the fixed vs adaptive threshold difference
    from ablation.evaluate import compute_f1_at_threshold
    own_th = scoring["threshold"]
    own_pa = compute_f1_at_threshold(y_score, y_true, own_th, use_pa=True)
    own_std = compute_f1_at_threshold(y_score, y_true, own_th, use_pa=False)
    metrics["own_threshold_pa_f1"] = own_pa["f1"]
    metrics["own_threshold_std_f1"] = own_std["f1"]

    metrics["time_total_s"] = total_time
    metrics["n_golden_windows"] = len(golden)
    metrics["n_test_windows"] = n_test_windows
    metrics["avg_window_time_s"] = total_time / (len(golden) + n_test_windows)
    metrics["anomaly_rate"] = float(np.mean(y_true))

    # Memory estimate (parameter count)
    from ablation.train import create_model
    d = train_data.shape[1]
    tmp_model, _ = create_model(
        variant_name, d, p, n_knots=n_knots,
        rank_w=rank_w, rank_a=rank_a, device="cpu",
    )
    metrics["param_count"] = tmp_model.count_parameters()
    del tmp_model

    return metrics


def run_full_ablation(args):
    """Run the complete component ablation study."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entities = args.entities
    seeds = DEFAULT_SEEDS[: args.seeds]
    if args.variants:
        variants = args.variants
    elif args.profile == "neurips":
        variants = NEURIPS_VARIANTS
    else:
        variants = list(VARIANTS.keys())

    logger.info("=" * 70)
    logger.info("TUCKER-CAM COMPONENT ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(f"Variants: {variants}")
    logger.info(f"Entities: {entities}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    # Collect all results
    # Structure: {variant: {entity: [seed_results]}}
    all_results = {}
    raw_records = []

    for entity in entities:
        logger.info(f"Loading SMD entity: {entity}")
        try:
            train_data, test_data, labels = load_smd_entity(
                entity, data_root=args.data_root
            )
        except FileNotFoundError:
            logger.error(f"  Data not found for {entity}, skipping")
            continue

        d = train_data.shape[1]
        logger.info(f"  d={d}, train_T={len(train_data)}, test_T={len(test_data)}")

        for variant_name in variants:
            if variant_name not in all_results:
                all_results[variant_name] = {}
            all_results[variant_name][entity] = []

            for seed in seeds:
                try:
                    metrics = run_single_experiment(
                        variant_name=variant_name,
                        entity=entity,
                        seed=seed,
                        train_data=train_data,
                        test_data=test_data,
                        labels=labels,
                        p=args.p,
                        window_size=args.window_size,
                        stride=args.stride,
                        rank_w=args.rank_w,
                        rank_a=args.rank_a,
                        n_knots=args.n_knots,
                        max_iter=args.max_iter,
                        device=args.device,
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {variant_name}/{entity}/seed={seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    metrics = None

                if metrics is not None:
                    all_results[variant_name][entity].append(metrics)
                    raw_records.append(
                        {
                            "variant": variant_name,
                            "entity": entity,
                            "seed": seed,
                            **metrics,
                        }
                    )
                    logger.info(
                        f"    PA-F1={metrics['pa_f1']:.3f} "
                        f"Std-F1={metrics['std_f1']:.3f} "
                        f"AUC-PR={metrics['auc_pr']:.3f} "
                        f"time={metrics['time_total_s']:.1f}s"
                    )

    # Save raw results
    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(output_dir / "ablation_raw_results.csv", index=False)
    logger.info(f"\nRaw results saved to {output_dir / 'ablation_raw_results.csv'}")

    # Aggregate across seeds and entities
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATED RESULTS (mean +/- std across seeds and entities)")
    logger.info("=" * 70)

    aggregated = {}
    for variant_name in variants:
        # Flatten all seed results across entities
        all_seed_metrics = []
        for entity in entities:
            if entity in all_results.get(variant_name, {}):
                all_seed_metrics.extend(all_results[variant_name][entity])

        if all_seed_metrics:
            aggregated[variant_name] = aggregate_seeds(all_seed_metrics)

    # Print summary table
    table = format_result_table(aggregated, baseline_key="full")
    print("\n" + table)

    # Detailed per-variant statistics with significance tests
    logger.info("\n" + "-" * 70)
    logger.info("SIGNIFICANCE TESTS (paired t-test vs full model)")
    logger.info("-" * 70)

    baseline_key = "full"
    if baseline_key in aggregated:
        for variant_name, agg in aggregated.items():
            if variant_name == baseline_key:
                continue
            for metric in ["pa_f1", "std_f1", "auc_pr"]:
                if metric in agg and metric in aggregated[baseline_key]:
                    tt = paired_t_test(
                        aggregated[baseline_key][metric]["values"],
                        agg[metric]["values"],
                    )
                    delta = agg[metric]["mean"] - aggregated[baseline_key][metric]["mean"]
                    sig_marker = "*" if tt["significant"] else ""
                    logger.info(
                        f"  {variant_name:25s} {metric:10s}: "
                        f"delta={delta:+.3f} p={tt['p_value']:.4f} {sig_marker}"
                    )

    # Save aggregated results as JSON
    agg_serializable = {}
    for vn, agg in aggregated.items():
        agg_serializable[vn] = {
            k: {
                "mean": v["mean"],
                "std": v["std"],
                "ci95": v.get("ci95", 0.0),
                "n": v.get("n", len(v["values"])),
            }
            for k, v in agg.items()
        }
    with open(output_dir / "ablation_aggregated.json", "w") as f:
        json.dump(agg_serializable, f, indent=2)

    # Generate LaTeX table
    _write_latex_table(aggregated, output_dir / "ablation_table.tex", baseline_key)

    logger.info(f"\nAll outputs saved to {output_dir}/")
    return aggregated


def _write_latex_table(
    aggregated: dict, output_path: Path, baseline_key: str = "full"
):
    """Write a publication-ready LaTeX table."""
    metrics = ["pa_f1", "std_f1", "auc_pr", "param_count"]
    metric_labels = {
        "pa_f1": "PA-F1",
        "std_f1": "Std-F1",
        "auc_pr": "AUC-PR",
        "param_count": "Params",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Component ablation results (mean $\pm$ std with 95\% CI across seeds and entities).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Variant & " + " & ".join(metric_labels[m] for m in metrics) + r" \\",
        r"\midrule",
    ]

    # Find best values for bolding
    best = {}
    for m in metrics:
        vals = [
            agg[m]["mean"]
            for agg in aggregated.values()
            if m in agg
        ]
        if vals and m != "param_count":
            best[m] = max(vals)
        elif vals:
            best[m] = min(vals)

    for variant_name, agg in aggregated.items():
        display_name = VARIANTS.get(variant_name, {}).get("name", variant_name)
        # Escape underscores for LaTeX
        display_name = display_name.replace("_", r"\_")

        parts = [display_name]
        for m in metrics:
            if m not in agg:
                parts.append("--")
                continue
            mean = agg[m]["mean"]
            std = agg[m]["std"]
            if m == "param_count":
                # Format as integer with comma separator
                parts.append(f"{int(mean):,}")
            else:
                ci95 = agg[m].get("ci95", 0.0)
                cell = f"{mean:.3f} $\\pm$ {std:.3f} [{ci95:.3f}]"
                if m in best and abs(mean - best[m]) < 1e-6:
                    cell = r"\textbf{" + cell + "}"
                # Significance marker
                if variant_name != baseline_key and baseline_key in aggregated:
                    if m in aggregated[baseline_key]:
                        tt = paired_t_test(
                            aggregated[baseline_key][m]["values"],
                            agg[m]["values"],
                        )
                        if tt["significant"]:
                            cell += r"$^\dagger$"
                parts.append(cell)

        lines.append(" & ".join(parts) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"{\footnotesize $^\dagger$ Statistically significant difference vs.\ full model (paired $t$-test, $p<0.05$).}",
            r"\end{table}",
        ]
    )

    output_path.write_text("\n".join(lines))
    logger.info(f"LaTeX table written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tucker-CAM Component Ablation Study"
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        default=DEFAULT_ENTITIES,
        help="SMD entities to evaluate on",
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of random seeds (max 5)"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variants to run (default: all)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["full", "neurips"],
        default="full",
        help="Preset variant profile (neurips focuses on paper-critical ablations)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="ServerMachineDataset",
        help="Path to SMD dataset root",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--p", type=int, default=5, help="Lag order")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--rank-w", type=int, default=20)
    parser.add_argument("--rank-a", type=int, default=10)
    parser.add_argument("--n-knots", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation_component",
    )
    args = parser.parse_args()

    run_full_ablation(args)


if __name__ == "__main__":
    main()
