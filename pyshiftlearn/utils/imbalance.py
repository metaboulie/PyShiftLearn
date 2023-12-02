"""Functions for Cost-Sensitive learning and imbalance learning"""

import numpy as np
import torch


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
