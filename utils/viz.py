"""Plotting helpers for figures and analysis."""
import matplotlib.pyplot as plt
import numpy as np


def plot_per_class_comparison(results: dict, class_names: list, save_path: str = None):
    """Bar chart: per-class AUROC for each model variant.

    results: dict like {'Baseline': [auroc_per_class], 'AttnRes': [...]}
    """
    raise NotImplementedError


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
    """Line plot: hidden state L2 norm vs layer index, one line per training step."""
    raise NotImplementedError


def plot_lead_ablation(results: dict, lead_subsets: list, save_path: str = None):
    """Bar chart: AUROC degradation under different lead masking conditions."""
    raise NotImplementedError
