from abc import ABC, abstractmethod

import numpy as np
import torch


class BassWeightsCalculator(ABC):
    def __init__(self, labels: torch.Tensor | np.ndarray, data: torch.Tensor | np.ndarray = None):
        self.labels: torch.Tensor | np.ndarray = labels
        self.data: torch.Tensor | np.ndarray = data
        self.weights: torch.Tensor = self.calculate_weights()

    @abstractmethod
    def calculate_weights(self) -> torch.Tensor:
        pass


class SimpleWeightsCalculator(BassWeightsCalculator):
    """
    Calculate the weights based on the number of samples in each class
    """
    def __init__(self, labels: torch.Tensor | np.ndarray, data: torch.Tensor | np.ndarray = None):
        super(SimpleWeightsCalculator, self).__init__(labels, data)
        self.weights: torch.Tensor = self.calculate_weights()

    def calculate_weights(self) -> torch.Tensor:
        """
        Calculate the class weights for a given dataset by label proportion.

        Returns:
            torch.Tensor: The class weights.
        """
        # Convert labels to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.int)

        # Count the number of occurrences of each class label
        class_counts = torch.bincount(self.labels)

        return torch.max(class_counts) / class_counts if len(class_counts) > 0 else torch.tensor([])
