"""Aggregate per-model evaluation JSONs into paper-ready figures.

Reads outputs/<experiment_name>_results.json (one per model variant) and produces:
    figures/per_class_auroc.png   - grouped bar chart, baseline = first model
    figures/macro_auroc.png       - macro AUROC bar chart with bootstrap CIs
    figures/summary_table.csv     - macro AUROC, F-max, CI per model

If multiple files match a glob pattern (e.g. *_seed*_results.json), averages
per-class AUROC across seeds and uses std as the error-bar half-width.

Usage:
    python analysis/plot_paper_figures.py \
        --results_dir outputs \
        --figures_dir figures \
        --models baseline_postnorm baseline_prenorm full_attnres block_attnres_2 block_attnres_3
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.viz import plot_per_class_comparison

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

DISPLAY_NAMES = {
    "baseline_postnorm": "Baseline (PostNorm)",
    "baseline_prenorm": "Baseline (PreNorm)",
    "full_attnres": "Full AttnRes",
    "block_attnres_2": "Block AttnRes (N=2)",
    "block_attnres_3": "Block AttnRes (N=3)",
}


def find_result_files(results_dir: Path, model: str) -> list[Path]:
    """Return all JSONs matching <model>_results.json or <model>_seed*_results.json."""
    candidates = sorted(results_dir.glob(f"{model}_seed*_results.json"))
    if candidates:
        return candidates
    single = results_dir / f"{model}_results.json"
    return [single] if single.exists() else []


def aggregate(files: list[Path]) -> dict:
    """Mean per-class AUROC + std across seeds. Also macro mean and CI from any one file."""
    per_class_runs = defaultdict(list)
    macro_runs, fmax_runs = [], []
    last_ci = None

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        for cls in SUPERCLASSES:
            v = data["per_class_auroc"].get(cls)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                per_class_runs[cls].append(v)
        macro_runs.append(data["macro_auroc"])
        fmax_runs.append(data["f_max"])
        last_ci = data.get("ci")

    per_class_mean = {c: float(np.mean(per_class_runs[c])) for c in SUPERCLASSES if per_class_runs[c]}
    per_class_std = {c: float(np.std(per_class_runs[c])) for c in SUPERCLASSES if per_class_runs[c]}

    return {
        "per_class": per_class_mean,
        "per_class_err": per_class_std if len(files) > 1 else None,
        "macro_mean": float(np.mean(macro_runs)),
        "macro_std": float(np.std(macro_runs)) if len(macro_runs) > 1 else 0.0,
        "fmax_mean": float(np.mean(fmax_runs)),
        "ci": last_ci,
        "n_seeds": len(files),
    }


def plot_macro_auroc(agg: dict[str, dict], save_path: Path):
    names = list(agg.keys())
    means = [agg[m]["macro_mean"] for m in names]

    if all(agg[m]["n_seeds"] > 1 for m in names):
        errs = [agg[m]["macro_std"] for m in names]
        err_label = "± std across seeds"
    else:
        errs = []
        for m in names:
            ci = agg[m].get("ci")
            if ci and "auroc" in ci:
                half = (ci["auroc"]["upper"] - ci["auroc"]["lower"]) / 2
                errs.append(half)
            else:
                errs.append(0.0)
        err_label = "± 95% bootstrap CI"

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4.5))
    colors = plt.cm.Set2(np.linspace(0, 0.7, len(names)))
    bars = ax.bar(range(len(names)), means, yerr=errs, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.6,
                  error_kw={"linewidth": 1, "color": "gray"})

    for bar, m, e in zip(bars, means, errs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + e + 0.002,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Macro AUROC", fontsize=11)
    ax.set_title(f"Macro AUROC by Model Variant ({err_label})", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    y_min = max(0, min(means) - max(errs + [0.02]) - 0.02)
    ax.set_ylim(y_min, min(1.0, max(means) + max(errs + [0.02]) + 0.03))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved macro AUROC bar chart to {save_path}")


def write_summary_csv(agg: dict[str, dict], save_path: Path):
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "n_seeds", "macro_auroc_mean", "macro_auroc_std",
                  "f_max_mean", "ci_lower", "ci_upper"] + SUPERCLASSES
        w.writerow(header)
        for name, a in agg.items():
            ci = a.get("ci") or {}
            ci_a = ci.get("auroc", {})
            row = [
                name, a["n_seeds"],
                f"{a['macro_mean']:.4f}", f"{a['macro_std']:.4f}",
                f"{a['fmax_mean']:.4f}",
                f"{ci_a.get('lower', float('nan')):.4f}",
                f"{ci_a.get('upper', float('nan')):.4f}",
            ]
            for cls in SUPERCLASSES:
                v = a["per_class"].get(cls, float("nan"))
                row.append(f"{v:.4f}")
            w.writerow(row)
    print(f"Saved summary table to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="outputs")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument(
        "--models", nargs="+",
        default=["baseline_prenorm", "baseline_postnorm",
                 "full_attnres", "block_attnres_2", "block_attnres_3"],
        help="Order matters: first model is treated as baseline in the delta panel.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg = {}
    for model in args.models:
        files = find_result_files(results_dir, model)
        if not files:
            print(f"[skip] no results JSON found for {model}")
            continue
        display = DISPLAY_NAMES.get(model, model)
        agg[display] = aggregate(files)
        print(f"  {display}: {len(files)} seed(s), macro AUROC = {agg[display]['macro_mean']:.4f}")

    if not agg:
        raise SystemExit("No result files found. Run evaluate.py first.")

    per_class = {name: a["per_class"] for name, a in agg.items()}
    errors = None
    if all(a["per_class_err"] is not None for a in agg.values()):
        errors = {name: a["per_class_err"] for name, a in agg.items()}

    plot_per_class_comparison(
        results=per_class,
        class_names=SUPERCLASSES,
        errors=errors,
        save_path=str(figures_dir / "per_class_auroc.png"),
    )

    plot_macro_auroc(agg, figures_dir / "macro_auroc.png")
    write_summary_csv(agg, figures_dir / "summary_table.csv")


if __name__ == "__main__":
    main()
