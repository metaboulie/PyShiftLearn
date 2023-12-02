import numpy as np

import torch
from torcheval.metrics.functional import multiclass_precision, multiclass_recall

from imblearn.metrics import geometric_mean_score


def weighted_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.float32:
    """
    The weighted macro F1-score.

    Parameters:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        torch.float32: Weighted macro F1-score.
    """
    # Compute the number of instances for each class
    n_instances = len(y_true)
    n_classes = len(torch.unique(y_true))
    n_instances_per_class = torch.bincount(y_true)

    # Compute the weights for each class
    weights = [(n_instances - n) / ((n_classes - 1) * n_instances) for n in n_instances_per_class]
    weights = torch.tensor(weights, dtype=torch.float32)

    # Compute the class-specific metrics
    ppv = multiclass_precision(y_pred, y_true, average=None, num_classes=n_classes)
    tpr = multiclass_recall(y_pred, y_true, average=None, num_classes=n_classes)

    weighted_ppv = (ppv * weights).sum(dim=0) / weights.sum(dim=0)
    weighted_tpr = (tpr * weights).sum(dim=0) / weights.sum(dim=0)

    wmf1 = 2 * (weighted_ppv * weighted_tpr) / (weighted_ppv + weighted_tpr)

    return wmf1.item()


def geometric_mean_accuracy(
    y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray,
    weights: torch.Tensor | np.ndarray = None, *args
) -> torch.float32:
    """
    The geometric mean accuracy.

    Parameters:
        y_true (torch.Tensor | np.ndarray): True labels.
        y_pred (torch.Tensor | np.ndarray): Predicted labels.
        weights (torch.Tensor | np.ndarray): Weights for each sample.

    Returns:
        torch.float32: Geometric mean accuracy.
    """
    return torch.tensor(geometric_mean_score(
        y_true=y_true, y_pred=y_pred, sample_weight=weights, *args
    ), dtype=torch.float32)
