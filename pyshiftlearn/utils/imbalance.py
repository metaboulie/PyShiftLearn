"""Functions for Cost-Sensitive learning and imbalance learning"""

import numpy as np
import torch
from torcheval.metrics.functional import multiclass_precision, multiclass_recall


def class_weights_by_label_proportion(labels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Calculate the class weights for a given dataset by label proportion.

    Args:
        labels (np.ndarray | torch.Tensor): The input labels.

    Returns:
        torch.Tensor: The class weights.
    """
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.int)

    # Count the number of occurrences of each class label
    class_counts = torch.bincount(labels)

    return torch.max(class_counts) / class_counts if len(class_counts) > 0 else torch.tensor([])


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


def calculate_cost_matrix(labels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Calculates the cost matrix based on the given labels.

    Parameters:
        labels (torch.Tensor): Array of labels.

    Returns:
        torch.Tensor: The calculated cost matrix.
    """
    # Count the unique labels and their occurrences
    labels = torch.tensor(labels, dtype=torch.int)
    unique_labels, label_counts = torch.unique(labels, return_counts=True)
    num_classes = len(unique_labels)

    # Initialize the cost matrix with all ones
    cost_matrix = torch.ones((num_classes, num_classes), dtype=torch.float32)

    # Calculate the cost for each pair of classes
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                # Set the cost to 0 for the same class
                cost_matrix[i, j] = 0
            else:
                if label_counts[i] > label_counts[j]:
                    # Calculate the cost based on the label occurrences
                    cost_matrix[i, j] = label_counts[i] / label_counts[j]

    return cost_matrix


def predict_min_expected_cost_class(cost_matrix: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Predicts the class with the minimum expected cost based on the given cost matrix and logits.

    Args:
        cost_matrix (torch.Tensor): A tensor representing the cost matrix.
        logits (torch.Tensor): A tensor representing the logits.

    Returns:
        torch.Tensor: A tensor representing the class with the minimum expected cost.
    """
    # Calculate the expected costs
    expected_costs = torch.matmul(cost_matrix, logits.T).T

    return torch.argmin(expected_costs, dim=1)
