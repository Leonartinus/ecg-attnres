"""Evaluation metrics for multi-label ECG classification.

All functions expect:
    y_true: (n_samples, n_classes) binary array
    y_pred: (n_samples, n_classes) predicted probabilities (after sigmoid)
"""
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _valid_classes(y_true: np.ndarray) -> List[int]:
    """Return indices of classes that have at least one positive AND one negative sample.
    AUROC is undefined for classes with all-positive or all-negative labels."""
    valid = []
    for c in range(y_true.shape[1]):
        n_pos = y_true[:, c].sum()
        n_neg = len(y_true) - n_pos
        if n_pos > 0 and n_neg > 0:
            valid.append(c)
    return valid


def macro_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged AUROC across classes. Skips classes with no positives or no negatives.

    Args:
        y_true: (n_samples, n_classes) binary ground truth
        y_pred: (n_samples, n_classes) predicted probabilities

    Returns:
        float: macro-averaged AUROC over valid classes
    """
    valid = _valid_classes(y_true)
    if len(valid) == 0:
        return float("nan")

    aurocs = []
    for c in valid:
        aurocs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))

    return float(np.mean(aurocs))


def per_class_auroc(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list = None) -> Dict[str, float]:
    """Per-class AUROC as a dict.

    Args:
        y_true: (n_samples, n_classes) binary ground truth
        y_pred: (n_samples, n_classes) predicted probabilities
        class_names: list of class name strings; defaults to SUPERCLASSES

    Returns:
        dict mapping class name to AUROC (or NaN if class has no pos/neg)
    """
    if class_names is None:
        class_names = SUPERCLASSES[:y_true.shape[1]]

    results = {}
    for c, name in enumerate(class_names):
        n_pos = y_true[:, c].sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            results[name] = float("nan")
        else:
            results[name] = float(roc_auc_score(y_true[:, c], y_pred[:, c]))

    return results


def bootstrap_auroc(y_true: np.ndarray, y_pred: np.ndarray,
                    n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for macro AUROC.

    Resamples patients with replacement, computes macro AUROC on each
    bootstrap sample, returns the mean and 95% CI.

    Args:
        y_true: (n_samples, n_classes) binary ground truth
        y_pred: (n_samples, n_classes) predicted probabilities
        n_bootstrap: number of bootstrap iterations
        seed: random seed for reproducibility

    Returns:
        (mean_auroc, lower_95ci, upper_95ci)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        # Skip bootstrap samples where any valid class loses all positives/negatives
        score = macro_auroc(y_true_boot, y_pred_boot)
        if not np.isnan(score):
            scores.append(score)

    scores = np.array(scores)

    return (
        float(np.mean(scores)),
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )


def paired_bootstrap_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                          y_pred_b: np.ndarray, n_bootstrap: int = 1000,
                          seed: int = 42) -> float:
    """Two-sided paired bootstrap test for H0: AUROC_A == AUROC_B.

    Computes the difference in macro AUROC on each bootstrap sample.
    The p-value is the proportion of bootstrap samples where the observed
    difference changes sign (i.e., the fraction where the winner flips).

    Args:
        y_true: (n_samples, n_classes) binary ground truth
        y_pred_a: (n_samples, n_classes) predicted probs from model A
        y_pred_b: (n_samples, n_classes) predicted probs from model B
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        float: two-sided p-value. Small p → significant difference.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    # Observed difference on the full test set
    observed_diff = macro_auroc(y_true, y_pred_a) - macro_auroc(y_true, y_pred_b)

    count_against = 0
    count_valid = 0

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        y_true_boot = y_true[idx]

        score_a = macro_auroc(y_true_boot, y_pred_a[idx])
        score_b = macro_auroc(y_true_boot, y_pred_b[idx])

        if np.isnan(score_a) or np.isnan(score_b):
            continue

        count_valid += 1
        boot_diff = score_a - score_b

        # Two-sided: count when bootstrap difference opposes observed direction
        if observed_diff >= 0 and boot_diff < 0:
            count_against += 1
        elif observed_diff < 0 and boot_diff >= 0:
            count_against += 1

    if count_valid == 0:
        return float("nan")

    # Two-sided p-value: double the one-sided proportion
    p_value = 2.0 * count_against / count_valid
    return min(p_value, 1.0)  # cap at 1.0


def f_max(y_true: np.ndarray, y_pred: np.ndarray, n_thresholds: int = 100) -> float:
    """Threshold-optimized macro F1 score (F_max).

    Sweeps thresholds per class independently to find the best F1 per class,
    then macro-averages. This is the secondary metric used in the PTB-XL benchmark.

    Args:
        y_true: (n_samples, n_classes) binary ground truth
        y_pred: (n_samples, n_classes) predicted probabilities
        n_thresholds: number of thresholds to sweep between 0 and 1

    Returns:
        float: macro-averaged F1 at optimal per-class thresholds
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)[1:-1]  # exclude 0 and 1
    n_classes = y_true.shape[1]
    best_f1s = []

    for c in range(n_classes):
        n_pos = y_true[:, c].sum()
        if n_pos == 0:
            continue

        best_f1 = 0.0
        for t in thresholds:
            y_hat = (y_pred[:, c] >= t).astype(float)
            tp = ((y_hat == 1) & (y_true[:, c] == 1)).sum()
            fp = ((y_hat == 1) & (y_true[:, c] == 0)).sum()
            fn = ((y_hat == 0) & (y_true[:, c] == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            best_f1 = max(best_f1, f1)

        best_f1s.append(best_f1)

    return float(np.mean(best_f1s)) if best_f1s else float("nan")


def summarize_results(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: list = None,
                      n_bootstrap: int = 1000) -> Dict:
    """Compute all metrics and return a summary dict.

    Convenient for saving results to JSON after evaluation.

    Args:
        y_true: (n_samples, n_classes) binary ground truth
        y_pred: (n_samples, n_classes) predicted probabilities
        class_names: list of class name strings
        n_bootstrap: number of bootstrap samples for CI

    Returns:
        dict with macro_auroc, per_class_auroc, bootstrap_ci, f_max
    """
    if class_names is None:
        class_names = SUPERCLASSES[:y_true.shape[1]]

    mean_auroc, lower, upper = bootstrap_auroc(y_true, y_pred, n_bootstrap=n_bootstrap)

    return {
        "macro_auroc": macro_auroc(y_true, y_pred),
        "macro_auroc_bootstrap_mean": mean_auroc,
        "macro_auroc_95ci_lower": lower,
        "macro_auroc_95ci_upper": upper,
        "per_class_auroc": per_class_auroc(y_true, y_pred, class_names),
        "f_max": f_max(y_true, y_pred),
        "n_samples": len(y_true),
        "n_classes": y_true.shape[1],
        "class_names": class_names,
    }