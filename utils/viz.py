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
    """Heatmap: rows = conditions, columns = layers, cells = mean attention weight.

    alphas: shape (n_classes, n_layers)
    """
    raise NotImplementedError


def plot_training_dynamics(hidden_norms: dict, save_path: str = None):
    """Line plot: hidden state L2 norm vs layer index, one line per training step."""
    raise NotImplementedError


def plot_lead_ablation(results: dict, lead_subsets: list, save_path: str = None):
    """Bar chart: AUROC degradation under different lead masking conditions."""
    raise NotImplementedError
