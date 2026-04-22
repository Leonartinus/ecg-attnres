"""Plotting helpers for figures and analysis."""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_per_class_comparison(results: dict, class_names: list,
                              errors: dict = None, save_path: str = None):
    """Grouped bar chart: per-class AUROC for each model variant.
 
    Optionally shows error bars (from bootstrap CI) and a delta annotation
    showing the improvement of each model over the first model listed.
 
    Args:
        results: dict mapping model name to per-class AUROC values.
                 Values can be either:
                   - a list of floats (one per class, same order as class_names)
                   - a dict mapping class name to float
                 Example: {"Baseline": [0.94, 0.91, 0.92, 0.91, 0.89]}
                 Example: {"Baseline": {"NORM": 0.94, "MI": 0.91, ...}}
 
        class_names: list of class name strings for x-axis labels
                     e.g. ["NORM", "MI", "STTC", "CD", "HYP"]
 
        errors: optional dict with same structure as results, containing
                the half-width of the 95% CI for each bar.
                If None, no error bars are drawn.
 
        save_path: if provided, save figure to this path instead of showing
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_classes = len(class_names)
 
    # Normalize results to lists
    def to_list(values):
        if isinstance(values, dict):
            return [values[c] for c in class_names]
        return list(values)
 
    values_by_model = {m: to_list(results[m]) for m in model_names}
 
    errors_by_model = None
    if errors is not None:
        errors_by_model = {}
        for m in model_names:
            if m in errors:
                errors_by_model[m] = to_list(errors[m])
            else:
                errors_by_model[m] = None
 
    # Colors
    colors = plt.cm.Set2(np.linspace(0, 0.7, n_models))
 
    # Layout
    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=(max(8, n_classes * 1.8), 7),
        gridspec_kw={"height_ratios": [3, 2]},
        sharex=True,
    )
 
    x = np.arange(n_classes)
    bar_width = 0.8 / n_models
 
    # ── Top panel: absolute AUROC per class ──
    for i, model_name in enumerate(model_names):
        vals = values_by_model[model_name]
        offset = (i - n_models / 2 + 0.5) * bar_width
 
        err = None
        if errors_by_model and errors_by_model.get(model_name) is not None:
            err = errors_by_model[model_name]
 
        bars = ax_main.bar(
            x + offset, vals, bar_width,
            label=model_name,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
            yerr=err,
            capsize=3,
            error_kw={"linewidth": 1, "color": "gray"},
        )
 
        # Value labels
        for bar, val in zip(bars, vals):
            ax_main.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.005 if err is None else max(err) + 0.005),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )
 
    ax_main.set_ylabel("AUROC", fontsize=11)
    ax_main.set_title("Per-Class AUROC by Model Variant", fontsize=13, fontweight="bold")
    ax_main.legend(fontsize=9, loc="lower left")
    ax_main.grid(axis="y", alpha=0.3)
 
    # Zoom y-axis to show differences clearly
    all_vals = [v for vals in values_by_model.values() for v in vals]
    y_min = max(0, min(all_vals) - 0.04)
    y_max = min(1.0, max(all_vals) + 0.03)
    ax_main.set_ylim(y_min, y_max)
 
    # ── Bottom panel: delta AUROC vs first model (baseline) ──
    baseline_name = model_names[0]
    baseline_vals = values_by_model[baseline_name]
 
    for i, model_name in enumerate(model_names):
        vals = values_by_model[model_name]
        deltas = [v - b for v, b in zip(vals, baseline_vals)]
        offset = (i - n_models / 2 + 0.5) * bar_width
 
        bar_colors = []
        for d in deltas:
            if d > 0.001:
                bar_colors.append("#4CAF50")   # green for improvement
            elif d < -0.001:
                bar_colors.append("#F44336")   # red for degradation
            else:
                bar_colors.append("#9E9E9E")   # gray for negligible
 
        bars = ax_delta.bar(
            x + offset, deltas, bar_width,
            color=bar_colors if i > 0 else [colors[0]] * n_classes,
            edgecolor="white",
            linewidth=0.5,
        )
 
        # Delta labels
        for bar, d in zip(bars, deltas):
            if abs(d) > 0.001:
                sign = "+" if d > 0 else ""
                ax_delta.text(
                    bar.get_x() + bar.get_width() / 2,
                    d + (0.002 if d >= 0 else -0.006),
                    f"{sign}{d:.3f}",
                    ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=7, fontweight="bold",
                )
 
    ax_delta.axhline(y=0, color="black", linewidth=0.8)
    ax_delta.set_ylabel(f"\u0394 AUROC vs {baseline_name}", fontsize=11)
    ax_delta.set_title(
        f"Improvement Over {baseline_name} (Green = Better)",
        fontsize=12, fontweight="bold",
    )
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(class_names, fontsize=10)
    ax_delta.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved per-class comparison to {save_path}")
    else:
        plt.show()


def plot_depth_attention_heatmap(alphas: np.ndarray, class_names: list,
                                 layer_names: list, save_path: str = None):
    """Heatmap: rows = layers, columns = previous layers, cells = mean attention weight.

    alphas: shape (n_layers, n_prev_layers)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(alphas, cmap='viridis', aspect='auto')
    
    ax.set_xticks(np.arange(alphas.shape[1]))
    ax.set_yticks(np.arange(alphas.shape[0]))
    ax.set_xticklabels([f'Prev {i}' for i in range(alphas.shape[1])])
    ax.set_yticklabels(layer_names)
    
    plt.colorbar(im, ax=ax)
    ax.set_title('Attention Residual Weights (α)')
    ax.set_xlabel('Previous Layers')
    ax.set_ylabel('Current Layer')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_dynamics(hidden_norms: dict, save_path: str = None):
    """Line plot: hidden state L2 norm vs layer index, showing how norms evolve during training.
 
    One subplot per model variant. Within each subplot, one line per training epoch.
    This visualizes the key claim from the AttnRes paper: standard PreNorm causes
    hidden state norms to grow unboundedly with depth, while AttnRes stabilizes them.
 
    Args:
        hidden_norms: nested dict structured as:
            {
                model_name: {
                    epoch_label: [norm_layer0, norm_layer1, ..., norm_layerN],
                    ...
                },
                ...
            }
            Example epoch_labels: "epoch_1", "epoch_10", "epoch_50"
            Each list has length n_layers, containing the mean L2 norm at that layer.
 
        save_path: if provided, save figure to this path instead of showing
    """
    model_names = list(hidden_norms.keys())
    n_models = len(model_names)
 
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]
 
    # Use a colormap that goes from light (early epochs) to dark (late epochs)
    for ax, model_name in zip(axes, model_names):
        epochs = hidden_norms[model_name]
        epoch_labels = list(epochs.keys())
        n_epochs = len(epoch_labels)
        colors = cm.Blues(np.linspace(0.3, 1.0, n_epochs))
 
        for i, (epoch_label, norms) in enumerate(epochs.items()):
            n_layers = len(norms)
            ax.plot(
                range(n_layers),
                norms,
                marker="o",
                markersize=4,
                color=colors[i],
                label=epoch_label,
                linewidth=1.5,
            )
 
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer index")
        ax.set_xticks(range(n_layers))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
 
    axes[0].set_ylabel("Hidden state L2 norm")
    fig.suptitle(
        "Hidden State Norms Across Depth and Training",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved training dynamics plot to {save_path}")
    else:
        plt.show()
 
 
def plot_lead_ablation(results: dict, lead_subsets: list, save_path: str = None):
    """Grouped bar chart: AUROC under different lead masking conditions.
 
    Shows two things at once:
    1. Absolute AUROC for each model under each lead subset
    2. The relative degradation from all_12 (how robust is each model?)
 
    Args:
        results: dict structured as:
            {
                model_name: {
                    subset_name: auroc_value,
                    ...
                },
                ...
            }
            Example: {"Baseline": {"all_12": 0.92, "limb": 0.87, ...}, ...}
 
        lead_subsets: ordered list of subset names for the x-axis
            Example: ["all_12", "limb", "precordial", "lead_I"]
 
        save_path: if provided, save figure to this path
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_subsets = len(lead_subsets)
 
    # Readable labels for the x-axis
    subset_display = {
        "all_12": "All 12 leads",
        "limb": "Limb only\n(I,II,III,aVR,aVL,aVF)",
        "precordial": "Precordial only\n(V1–V6)",
        "lead_I": "Lead I only\n(wearable)",
    }
 
    # Color palette: one color per model
    colors = plt.cm.Set2(np.linspace(0, 0.6, n_models))
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n_subsets * 2.5), 8),
                                    gridspec_kw={"height_ratios": [3, 2]})
 
    # ── Top panel: absolute AUROC ──
    bar_width = 0.8 / n_models
    x = np.arange(n_subsets)
 
    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(s, 0) for s in lead_subsets]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, values, bar_width, label=model_name,
                       color=colors[i], edgecolor="white", linewidth=0.5)
 
        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)
 
    ax1.set_ylabel("Macro AUROC", fontsize=11)
    ax1.set_title("Absolute Performance Under Lead Masking", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([subset_display.get(s, s) for s in lead_subsets], fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
 
    # Set y-axis to start near the minimum value for better visual contrast
    all_values = [results[m].get(s, 0) for m in model_names for s in lead_subsets]
    y_min = max(0, min(all_values) - 0.05)
    ax1.set_ylim(y_min, min(1.0, max(all_values) + 0.03))
 
    # ── Bottom panel: degradation from all_12 ──
    for i, model_name in enumerate(model_names):
        baseline_auroc = results[model_name].get("all_12", 1.0)
        drops = []
        for s in lead_subsets:
            auroc = results[model_name].get(s, 0)
            drop = baseline_auroc - auroc
            drops.append(drop)
 
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax2.bar(x + offset, drops, bar_width, label=model_name,
                       color=colors[i], edgecolor="white", linewidth=0.5)
 
        for bar, val in zip(bars, drops):
            if val > 0.001:  # only label nonzero drops
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f"–{val:.3f}", ha="center", va="bottom", fontsize=8)
 
    ax2.set_ylabel("AUROC drop from all 12 leads", fontsize=11)
    ax2.set_title("Degradation (Lower = More Robust)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([subset_display.get(s, s) for s in lead_subsets], fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, None)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved lead ablation plot to {save_path}")
    else:
        plt.show()