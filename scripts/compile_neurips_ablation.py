#!/usr/bin/env python3
"""
Collect canonical ablation outputs into one publication bundle.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--rank-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    component_dir = Path(args.component_dir)
    rank_dir = Path(args.rank_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    component_agg = _load_json(component_dir / "ablation_aggregated.json")
    rank_raw_path = rank_dir / "rank_study_raw.csv"

    manifest = {
        "component_aggregated_json": str(component_dir / "ablation_aggregated.json"),
        "component_raw_csv": str(component_dir / "ablation_raw_results.csv"),
        "component_latex_table": str(component_dir / "ablation_table.tex"),
        "rank_raw_csv": str(rank_raw_path),
        "rank_summary_json": str(rank_dir / "rank_study_summary.json"),
    }

    # Flatten Table 2 style export
    table_rows = []
    if component_agg:
        for variant, metrics in component_agg.items():
            row = {"variant": variant}
            for m in ("pa_f1", "std_f1", "auc_pr", "param_count", "time_total_s"):
                if m in metrics:
                    row[f"{m}_mean"] = metrics[m].get("mean")
                    row[f"{m}_std"] = metrics[m].get("std")
                    row[f"{m}_ci95"] = metrics[m].get("ci95")
                    row[f"{m}_n"] = metrics[m].get("n")
            table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(out_dir / "table2_component_ablation.csv", index=False)

    # Rank study compact summaries
    if rank_raw_path.exists():
        rank_df = pd.read_csv(rank_raw_path)
        summary_rows = []
        for sweep in ("R_w", "R_a"):
            subset = rank_df[rank_df["sweep"] == sweep]
            if subset.empty:
                continue
            col = sweep
            for value, group in subset.groupby(col):
                summary_rows.append(
                    {
                        "sweep": sweep,
                        "rank_value": int(value),
                        "pa_f1_mean": float(group["pa_f1"].mean()),
                        "pa_f1_std": float(group["pa_f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                        "std_f1_mean": float(group["std_f1"].mean()),
                        "time_total_s_mean": float(group["time_total_s"].mean()),
                        "param_count": int(group["param_count"].iloc[0]),
                        "n": int(len(group)),
                    }
                )
        pd.DataFrame(summary_rows).to_csv(out_dir / "rank_sensitivity_summary.csv", index=False)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report_lines = [
        "# NeurIPS Ablation Bundle",
        "",
        "Generated artifacts:",
        f"- `manifest.json`",
        f"- `table2_component_ablation.csv`",
        f"- `rank_sensitivity_summary.csv` (if rank raw CSV exists)",
        "",
        "Upstream sources:",
    ]
    for key, value in manifest.items():
        report_lines.append(f"- `{key}`: `{value}`")
    (out_dir / "README.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
