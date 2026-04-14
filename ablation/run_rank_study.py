#!/usr/bin/env python3
"""
Tensor Rank Sensitivity Study (Hyperparameter Tuning).

SEPARATE from the component ablation. This answers:
"What ranks should I pick?" -- not "Does Tucker help?"

Sweeps R_w and R_a independently, measuring:
- F1 performance (PA-F1, standard F1)
- Memory footprint (parameter count)
- Wall-clock time per window

Reports results as dual-axis plots (F1 vs compute).

Usage:
    # Full rank study
    python -m ablation.run_rank_study

    # Quick test
    python -m ablation.run_rank_study --ranks-w 5 10 20 --ranks-a 5 10 --seeds 2
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.train import (
    run_rolling_windows,
    compute_anomaly_scores,
    window_labels_from_point_labels,
    create_model,
)
from ablation.evaluate import full_evaluation, aggregate_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_smd_entity(entity, data_root="ServerMachineDataset"):
    root = Path(data_root)
    train = np.loadtxt(root / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(root / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(root / "test_label" / f"{entity}.txt", delimiter=",")
    return train, test, labels


def run_rank_experiment(
    entity, train_data, test_data, labels,
    rank_w, rank_a, seed, args,
):
    """Run a single (rank_w, rank_a, seed) experiment."""
    d = train_data.shape[1]

    t0 = time.time()

    golden = run_rolling_windows(
        train_data, variant_name="full",
        p=args.p, window_size=args.window_size, stride=args.stride,
        rank_w=rank_w, rank_a=rank_a,
        n_knots=args.n_knots, max_iter=args.max_iter,
        device=args.device, seed=seed,
    )

    test_results = run_rolling_windows(
        test_data, variant_name="full",
        p=args.p, window_size=args.window_size, stride=args.stride,
        rank_w=rank_w, rank_a=rank_a,
        n_knots=args.n_knots, max_iter=args.max_iter,
        device=args.device, seed=seed,
    )

    total_time = time.time() - t0

    if len(golden) == 0 or len(test_results) == 0:
        return None

    scoring = compute_anomaly_scores(golden, test_results)
    n_test = len(test_results)
    y_true = window_labels_from_point_labels(
        labels, n_test,
        window_size=args.window_size, stride=args.stride, p=args.p,
    )
    y_score = scoring["scores"][:n_test]
    y_true = y_true[:n_test]

    metrics = full_evaluation(y_score, y_true)

    # Parameter count
    model, _ = create_model(
        "full", d, args.p, n_knots=args.n_knots,
        rank_w=rank_w, rank_a=rank_a, device="cpu",
    )
    metrics["param_count"] = model.count_parameters()
    metrics["time_total_s"] = total_time
    metrics["avg_window_time_s"] = total_time / (len(golden) + n_test)
    del model

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Tucker-CAM Tensor Rank Sensitivity Study"
    )
    parser.add_argument(
        "--entity", type=str, default="machine-1-1",
        help="SMD entity to use",
    )
    parser.add_argument(
        "--ranks-w", nargs="+", type=int, default=[5, 10, 15, 20, 30, 40],
        help="R_w values to sweep (while R_a=10 fixed)",
    )
    parser.add_argument(
        "--ranks-a", nargs="+", type=int, default=[5, 10, 15, 20, 30],
        help="R_a values to sweep (while R_w=20 fixed)",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--data-root", type=str, default="ServerMachineDataset")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--n-knots", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument(
        "--output-dir", type=str, default="results/ablation_rank_study",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 1024][: args.seeds]

    logger.info("=" * 70)
    logger.info("TENSOR RANK SENSITIVITY STUDY (HYPERPARAMETER TUNING)")
    logger.info("=" * 70)
    logger.info(f"Entity: {args.entity}")
    logger.info(f"R_w sweep: {args.ranks_w} (R_a=10 fixed)")
    logger.info(f"R_a sweep: {args.ranks_a} (R_w=20 fixed)")
    logger.info(f"Seeds: {seeds}")
    logger.info("")

    train_data, test_data, labels = load_smd_entity(
        args.entity, data_root=args.data_root
    )
    d = train_data.shape[1]
    logger.info(f"Data: d={d}, train_T={len(train_data)}, test_T={len(test_data)}")

    records = []

    # Sweep R_w (fix R_a=10)
    logger.info("\n--- Sweeping R_w (R_a=10 fixed) ---")
    for rw in args.ranks_w:
        for seed in seeds:
            logger.info(f"  R_w={rw}, R_a=10, seed={seed}")
            try:
                metrics = run_rank_experiment(
                    args.entity, train_data, test_data, labels,
                    rank_w=rw, rank_a=10, seed=seed, args=args,
                )
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                metrics = None

            if metrics:
                records.append({
                    "sweep": "R_w",
                    "R_w": rw, "R_a": 10,
                    "seed": seed,
                    **metrics,
                })
                logger.info(
                    f"    PA-F1={metrics['pa_f1']:.3f} "
                    f"params={metrics['param_count']:,} "
                    f"time={metrics['time_total_s']:.1f}s"
                )

    # Sweep R_a (fix R_w=20)
    logger.info("\n--- Sweeping R_a (R_w=20 fixed) ---")
    for ra in args.ranks_a:
        for seed in seeds:
            logger.info(f"  R_w=20, R_a={ra}, seed={seed}")
            try:
                metrics = run_rank_experiment(
                    args.entity, train_data, test_data, labels,
                    rank_w=20, rank_a=ra, seed=seed, args=args,
                )
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                metrics = None

            if metrics:
                records.append({
                    "sweep": "R_a",
                    "R_w": 20, "R_a": ra,
                    "seed": seed,
                    **metrics,
                })
                logger.info(
                    f"    PA-F1={metrics['pa_f1']:.3f} "
                    f"params={metrics['param_count']:,} "
                    f"time={metrics['time_total_s']:.1f}s"
                )

    # Save raw results
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "rank_study_raw.csv", index=False)

    # Aggregate and print summary
    logger.info("\n" + "=" * 70)
    logger.info("RANK SENSITIVITY SUMMARY")
    logger.info("=" * 70)

    for sweep_name in ["R_w", "R_a"]:
        subset = df[df["sweep"] == sweep_name]
        if subset.empty:
            continue

        rank_col = sweep_name
        logger.info(f"\n--- {sweep_name} sweep ---")
        logger.info(f"{'Rank':>6s} | {'PA-F1':>12s} | {'Std-F1':>12s} | {'Params':>10s} | {'Time(s)':>10s}")
        logger.info("-" * 65)

        for rank_val in sorted(subset[rank_col].unique()):
            rows = subset[subset[rank_col] == rank_val]
            pa_f1_mean = rows["pa_f1"].mean()
            pa_f1_std = rows["pa_f1"].std()
            std_f1_mean = rows["std_f1"].mean()
            std_f1_std = rows["std_f1"].std()
            params = rows["param_count"].iloc[0]
            time_mean = rows["time_total_s"].mean()
            logger.info(
                f"{rank_val:>6d} | "
                f"{pa_f1_mean:.3f} +/- {pa_f1_std:.3f} | "
                f"{std_f1_mean:.3f} +/- {std_f1_std:.3f} | "
                f"{int(params):>10,} | "
                f"{time_mean:>10.1f}"
            )

    # Save summary JSON
    summary = {}
    for sweep_name in ["R_w", "R_a"]:
        subset = df[df["sweep"] == sweep_name]
        sweep_summary = {}
        rank_col = sweep_name
        for rank_val in sorted(subset[rank_col].unique()):
            rows = subset[subset[rank_col] == rank_val]
            sweep_summary[int(rank_val)] = {
                "pa_f1_mean": float(rows["pa_f1"].mean()),
                "pa_f1_std": float(rows["pa_f1"].std()),
                "std_f1_mean": float(rows["std_f1"].mean()),
                "std_f1_std": float(rows["std_f1"].std()),
                "param_count": int(rows["param_count"].iloc[0]),
                "time_mean_s": float(rows["time_total_s"].mean()),
                "n_seeds": len(rows),
            }
        summary[sweep_name] = sweep_summary

    with open(output_dir / "rank_study_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
