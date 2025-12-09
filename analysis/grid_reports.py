#!/usr/bin/env python3
"""
Utility to summarize SAE grid runs and draw quick heatmaps.

Expected input: grid_results.jsonl produced by train_sae.py under
<runs>/<model_dir>/layer_<L>/<signal>/grid_results.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_results(path: Path) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def metric_value(row: dict, metric: str) -> float:
    stats = row.get("detailed_stats", {})
    if metric == "recon_loss":
        return row["best_val_recon"]
    if metric == "dead_rate":
        return stats.get("dead_rate")
    if metric == "mean_firing":
        return stats.get("mean_firing")
    if metric == "global_sparsity":
        return stats.get("global_sparsity")
    if metric == "active_mean":
        return stats.get("active_mean")
    if metric == "pca_code_256":
        cum = stats.get("code_pca_cumvar", [])
        return cum[255] if len(cum) > 255 else None
    if metric == "pca_input_256":
        cum = stats.get("input_pca_cumvar", [])
        return cum[255] if len(cum) > 255 else None
    raise ValueError(f"Unknown metric: {metric}")


def to_dataframe(rows: List[dict], metric: str) -> pd.DataFrame:
    records = []
    for r in rows:
        val = metric_value(r, metric)
        if val is None:
            continue
        records.append(
            {
                "hidden_factor": r["hidden_factor"],
                "k_frac": r["k_frac"],
                "l1_lambda": r["l1_lambda"],
                "eq_alpha": r["eq_alpha"],
                "seed": r["seed"],
                metric: val,
                "mean_firing": r["detailed_stats"].get("mean_firing"),
                "dead_rate": r["detailed_stats"].get("dead_rate"),
            }
        )
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg = (
        df.groupby(["hidden_factor", "k_frac", "l1_lambda", "eq_alpha"])
        .agg(
            mean_metric=(metric, "mean"),
            stderr_metric=(metric, "sem"),
            mean_firing=("mean_firing", "mean"),
            dead_rate=("dead_rate", "mean"),
        )
        .reset_index()
    )
    return agg


def plot_heatmap(table: pd.DataFrame, metric: str, output: Path):
    table = table.sort_index()
    if HAS_SEABORN:
        ax = sns.heatmap(table, annot=True, fmt=".3g", cmap="viridis")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(table.values, cmap="viridis")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f"{table.values[i, j]:.3g}", ha="center", va="center", color="w")
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns)
        fig.colorbar(im, ax=ax)
    plt.title(metric)
    plt.xlabel("k_frac")
    plt.ylabel("hidden_factor")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def filter_good_regime(df: pd.DataFrame, firing_min: float, firing_max: float, dead_rate_max: float, recon_quantile: float, metric: str) -> pd.DataFrame:
    recon_thresh = df[metric].quantile(recon_quantile)
    mask = (
        (df["mean_firing"].between(firing_min, firing_max))
        & (df["dead_rate"] < dead_rate_max)
        & (df[metric] <= recon_thresh)
    )
    return df[mask]


def main():
    parser = argparse.ArgumentParser(description="Summarize SAE grid results")
    parser.add_argument("--root", type=str, required=True, help="Path to layer/signal directory containing grid_results.jsonl")
    parser.add_argument("--metric", type=str, default="recon_loss", help="Metric to aggregate (recon_loss, dead_rate, mean_firing, global_sparsity, active_mean, pca_code_256)")
    parser.add_argument("--l1", type=float, default=None, help="Filter l1_lambda")
    parser.add_argument("--eq", type=float, default=None, help="Filter eq_alpha")
    parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for output files")
    parser.add_argument("--firing-min", type=float, default=0.02)
    parser.add_argument("--firing-max", type=float, default=0.10)
    parser.add_argument("--dead-max", type=float, default=0.3)
    parser.add_argument("--recon-quantile", type=float, default=0.5, help="Keep runs within this quantile for recon loss when filtering good regime")
    args = parser.parse_args()

    root = Path(args.root)
    results_path = root / "grid_results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found")

    rows = load_results(results_path)
    df = to_dataframe(rows, args.metric)

    if args.l1 is not None:
        df = df[np.isclose(df["l1_lambda"], args.l1)]
    if args.eq is not None:
        df = df[np.isclose(df["eq_alpha"], args.eq)]

    agg = aggregate(df, args.metric)

    pivot = agg.pivot(index="hidden_factor", columns="k_frac", values="mean_metric")
    out_prefix = Path(args.output_prefix) if args.output_prefix else root / f"{args.metric}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    plot_heatmap(pivot, args.metric, output=out_prefix.with_suffix(".png"))
    agg.to_csv(out_prefix.with_suffix(".csv"), index=False)

    # Good regime filtering
    good = filter_good_regime(
        agg.rename(columns={"mean_metric": args.metric}),
        firing_min=args.firing_min,
        firing_max=args.firing_max,
        dead_rate_max=args.dead_max,
        recon_quantile=args.recon_quantile,
        metric=args.metric,
    )
    good.to_csv(out_prefix.with_name(out_prefix.name + "_good.csv"), index=False)
    print(f"Saved heatmap to {out_prefix.with_suffix('.png')}")
    print(f"Saved aggregate CSV to {out_prefix.with_suffix('.csv')}")
    print(f"Good-regime subset saved to {out_prefix.with_name(out_prefix.name + '_good.csv')}")


if __name__ == "__main__":
    main()
