"""Evaluation metrics for multi-label ECG classification."""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def macro_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged AUROC across classes. Skips classes with no positives."""
    # TODO: per-class AUROC, average only over classes with >0 positives
    raise NotImplementedError


def per_class_auroc(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list) -> Dict[str, float]:
    """Per-class AUROC as a dict."""
    raise NotImplementedError


def bootstrap_auroc(y_true: np.ndarray, y_pred: np.ndarray,
                    n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Returns (mean, lower_95ci, upper_95ci) of macro AUROC via bootstrapping."""
    # TODO: resample indices with replacement n_bootstrap times
    # TODO: compute macro AUROC each time; report 2.5 and 97.5 percentiles
    raise NotImplementedError


def paired_bootstrap_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                          y_pred_b: np.ndarray, n_bootstrap: int = 1000) -> float:
    """p-value for H0: AUROC_A == AUROC_B, via paired bootstrap."""
    raise NotImplementedError


def compute_macro_auroc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Compute macro-averaged AUROC across the 5 superclasses using sklearn, skipping classes with no positives."""
    aurocs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            auroc = roc_auc_score(y_true[:, i], y_probs[:, i])
            aurocs.append(auroc)
    return np.mean(aurocs) if aurocs else 0.0


def compute_f_max(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Compute the optimal threshold that maximizes the F1-score for each class, then average."""
    f1_scores = []
    for i in range(y_true.shape[1]):
        thresholds = np.linspace(0, 1, 100)
        f1s = [f1_score(y_true[:, i], (y_probs[:, i] > t).astype(int)) for t in thresholds]
        f1_scores.append(max(f1s))
    return np.mean(f1_scores)


def compute_bootstrap_ci(y_true: np.ndarray, y_probs: np.ndarray, n_bootstraps: int = 1000) -> Dict[str, Dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for AUROC and F_max."""
    aurocs = []
    f_maxes = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_probs_boot = y_probs[indices]
        auroc = compute_macro_auroc(y_true_boot, y_probs_boot)
        f_max_val = compute_f_max(y_true_boot, y_probs_boot)
        aurocs.append(auroc)
        f_maxes.append(f_max_val)
    
    auroc_mean = np.mean(aurocs)
    auroc_lower = np.percentile(aurocs, 2.5)
    auroc_upper = np.percentile(aurocs, 97.5)
    
    f_max_mean = np.mean(f_maxes)
    f_max_lower = np.percentile(f_maxes, 2.5)
    f_max_upper = np.percentile(f_maxes, 97.5)
    
    return {
        'auroc': {'mean': auroc_mean, 'lower': auroc_lower, 'upper': auroc_upper},
        'f_max': {'mean': f_max_mean, 'lower': f_max_lower, 'upper': f_max_upper}
    }