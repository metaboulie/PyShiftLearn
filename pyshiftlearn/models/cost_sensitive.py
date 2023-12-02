"""Make models in neural_networks cost-sensitive"""
from abc import ABC, abstractmethod
import torch


class BassCostSensitive(ABC):
    """Base class for cost-sensitive learning"""

    @abstractmethod
    def calculate_cost_matrix(self) -> torch.Tensor:
        """Calculate the cost matrix"""
        pass

    @abstractmethod
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict the class with the minimum expected cost"""
        pass


class SimpleCostSensitive(BassCostSensitive):
    """Simple cost-sensitive learning"""

    def __init__(
        self,
    ):
        pass

    def calculate_cost_matrix(self) -> torch.Tensor:
        return NotImplemented

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return NotImplemented
