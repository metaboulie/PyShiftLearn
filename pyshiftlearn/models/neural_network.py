"""Loss functions and Layers for Neural Network"""

from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from pyshiftlearn.models.weight import BassWeightsCalculator


class BaseLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args):
        pass


class BaseNeuralNetwork(nn.Module, ABC):
    def __init__(self, model: nn.Module, loss_func: BaseLoss):
        super().__init__()
        self.model = model
        self.loss_func = loss_func

    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def forward(self):
        pass


class WeightedCrossEntropyLoss(BaseLoss):
    def __init__(self, labels: torch.Tensor | np.ndarray, data: torch.Tensor | np.ndarray = None,
                 WeightsCalculator: Type[BassWeightsCalculator] = BassWeightsCalculator, *args):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.data: torch.Tensor | np.ndarray = data
        self.labels: torch.Tensor | np.ndarray = labels
        self.weights: torch.Tensor = self.calculate_weights(WeightsCalculator)
        self.loss_func: nn.Module = CrossEntropyLoss(weight=self.weights, *args)

    def calculate_weights(self, WeightsCalculator: Type[BassWeightsCalculator]):
        return WeightsCalculator(self.labels, self.data).weights

    def forward(self, logits: torch.Tensor, y: torch.Tensor):
        return self.loss_func(logits, y)


class MLP(BaseNeuralNetwork):
    def __init__(self, model: nn.Module, loss_func: BaseLoss):
        super(MLP, self).__init__(model, loss_func)

    def fit(self, X: torch.Tensor, y: torch.Tensor, *args):
        pass

    def forward(self):
        pass
