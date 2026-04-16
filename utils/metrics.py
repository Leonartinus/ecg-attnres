"""Evaluation metrics for multi-label ECG classification."""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


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
