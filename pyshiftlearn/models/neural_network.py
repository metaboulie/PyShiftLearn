"""Loss functions and Layers for Neural Network"""

from abc import abstractmethod, ABC

from torch import nn


class BaseLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass


class WeightedCrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class BaseNeuralNetwork(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass
