from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence
from typing import TypeVar

import numpy as np
import torch
from imblearn.metrics import geometric_mean_score
from torch import nn
from tqdm import tqdm

from pyshiftlearn.config import ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_LR, BATCH_SIZE, EARLY_STOPPING_PATIENCE, \
    EARLY_STOPPING_THRESHOLD, N_EPOCHS, N_STEPS_TO_PRINT, SAVED_MODELS_PATH
from pyshiftlearn.models.neural_network import BaseLoss, BaseNeuralNetwork, SimpleWeightsCalculator, \
    WeightedCrossEntropyLoss
from pyshiftlearn.models.sample import BaseSample
from pyshiftlearn.utils.imbalance import calculate_cost_matrix, predict_min_expected_cost_class, weighted_macro_f1


def feature_label_split(data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the given dataset into feature-matrix and label-series, i.e. X and y.

    Parameters
    ----------
    data : np.ndarray
        The data to be split.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        X, y
    """
    # Ensure that the input data is of type np.ndarray
    assert isinstance(data, np.ndarray), "The type of the input data must be numpy.ndarray"

    # Split the data into feature-matrix and label-series
    X = data[:, :-1]
    y = data[:, -1]

    # Convert the feature-matrix and label-series to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y, dtype=torch.long)  # The dtype of y must be `torch.long`

    return X_tensor, y_tensor


MODEL_TYPE = TypeVar("MODEL_TYPE")
DATA_TYPE = TypeVar("DATA_TYPE")


class BaseTrainer(ABC):
    """Base abstract class for all trainers"""

    def __init__(self, train_model: MODEL_TYPE):
        self.train_model = train_model  # maybe the model should invoke a model class
        pass

    @abstractmethod
    def train(self, train_data: DATA_TYPE, valid_data: DATA_TYPE, test_data: DATA_TYPE, *args):
        pass

    @abstractmethod
    def evaluate(self, val_data: DATA_TYPE, *args):
        pass

    @abstractmethod
    def early_stopping(
        self, metric_records: dict[str, Sequence[float]], thresholds: dict[str, float],
        patience: int = EARLY_STOPPING_PATIENCE, *args
    ):
        pass

    @abstractmethod
    def save_model(self, path, *args):
        pass


class NNTrainer(BaseTrainer):
    """Trainer for Neural Network"""

    def __init__(self, train_model: BaseNeuralNetwork, cost_sensitive: bool = False):
        super().__init__(train_model)
        self.loss_func: BaseLoss | nn.Module = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler = None
        self.cost_sensitive = cost_sensitive
        self.cost_matrix: torch.Tensor | None = None
        self.metric_records = defaultdict(list)
        self.thresholds: dict[str, float] | None = None
        self.results = defaultdict(float)

    def __repr__(self):
        pass

    def compile(
        self, data: np.ndarray, weighted_loss: bool = False, optimizer: str = None,
        scheduler: str = None, thresholds: dict[str, float] = None
    ):
        if weighted_loss:
            self.loss_func = WeightedCrossEntropyLoss(data[:, -1], data, SimpleWeightsCalculator)

        if self.cost_sensitive:
            self.cost_matrix = calculate_cost_matrix(data[:, -1])

        self.thresholds = thresholds

        match optimizer:
            case 'adam':
                self.optimizer = torch.optim.Adam(
                    self.train_model.model.parameters(), lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS
                )
            case 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.train_model.model.parameters(), lr=ADAM_LR, momentum=0.9
                )
            case 'adamw':
                self.optimizer = torch.optim.AdamW(
                    self.train_model.model.parameters(), lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS
                )
            case 'adamax':
                self.optimizer = torch.optim.Adamax(
                    self.train_model.model.parameters(), lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS
                )
            case _:
                print(f"Optimizer {optimizer} not supported, set to adam")
                self.optimizer = torch.optim.Adam(
                    self.train_model.model.parameters(), lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS
                )

        match scheduler:
            case 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=N_STEPS_TO_PRINT, gamma=0.1)
            case 'exp':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)
            case 'cos':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=N_EPOCHS)
            case 'poly':
                self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, power=0.9)
            case 'plot':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, "min", patience=EARLY_STOPPING_PATIENCE, threshold=EARLY_STOPPING_THRESHOLD
                )
            case _:
                print(f"Scheduler {scheduler} not supported, set to plot")
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, "min", patience=EARLY_STOPPING_PATIENCE, threshold=EARLY_STOPPING_THRESHOLD
                )

    def train(
        self, train_data: np.ndarray, valid_data: np.ndarray, test_data: np.ndarray, dataset_name: str = None,
        batch_size: int = BATCH_SIZE, sample_model: BaseSample | None = None, n_epochs: int = N_EPOCHS, *args
    ):
        # Split features and labels
        X, y = feature_label_split(train_data)
        n_steps = X.shape[0] // batch_size
        # Set the model to training mode
        self.train_model.model.train()
        break_looping = True
        for epoch in tqdm(range(N_EPOCHS)):
            count = 0
            print(f"Epoch {epoch + 1}\n-------------------------------")

            for _ in tqdm(range(n_steps)):
                count += 1
                # Sample the data
                X_train, y_train = sample_model.sample()
                # Train the model
                self.optimizer.zero_grad()
                pred = self.train_model.model(X_train)
                loss = self.loss_func(pred, y_train)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if count % N_STEPS_TO_PRINT == 0:
                    break_looping = self.evaluate(
                        train_data, 'step', self.cost_matrix
                    ) | self.evaluate(valid_data, 'valid', self.cost_matrix)

                    if break_looping:
                        break

            if break_looping:
                break
            self.evaluate(train_data, 'train', self.cost_matrix)
            self.evaluate(test_data, 'test', self.cost_matrix)

        self.evaluate(train_data, 'model', self.cost_matrix)
        self.evaluate(test_data, 'model', self.cost_matrix)
        # Save the trained model
        try:
            torch.save(self.train_model.model, SAVED_MODELS_PATH)
            print("Model is saved")
        except Exception as e:
            print(f"Error: {e}")

    def evaluate(self, val_data: np.ndarray, mode: str = 'step', cost_matrix: torch.Tensor = None, *args, **kwargs):
        X, y = feature_label_split(val_data)
        self.train_model.model.eval()
        with torch.no_grad():
            logits = self.train_model.model(X)
            if self.cost_sensitive:
                y_pred = predict_min_expected_cost_class(cost_matrix, logits)
            else:
                y_pred = torch.argmax(logits, dim=1)
            # The geometric mean of accuracy for each label
            correct = geometric_mean_score(y_pred, y)
            loss = self.loss_func(logits, y).item()

        match mode:
            case 'step':
                print(f'Current training loss: {loss:.4f} \t Accuracy: {correct:.4f}')
                if self.early_stopping(metric_values=(correct, loss)):
                    return True

            case 'train':
                print(f'Current performance on training set: \n Accuracy: {correct:.4f} \t Loss: {loss:.4f}')

            case 'valid':
                print(f'Current performance on validation set: \n Accuracy: {correct:.4f} \t Loss: {loss:.4f}')

            case 'test':
                print(f'Current performance on test set: \n Accuracy: {correct:.4f} \t Loss: {loss:.4f}')

            case 'model':
                weighted_cross_entropy_loss = WeightedCrossEntropyLoss(y, val_data, SimpleWeightsCalculator).forward(
                    logits, y).item()
                weighted_macro_F1 = weighted_macro_f1(y, y_pred)

                self.results = {
                    "g_mean_accuracy": correct,
                    "weighted_macro_f1": weighted_macro_F1,
                    "weighted_cross_entropy_loss": weighted_cross_entropy_loss,
                    **kwargs,
                }

            # Raise ValueError for invalid mode
            case _:
                raise ValueError(
                    "Please enter a solid value for parameter `mode`, "
                    "options are ('step', 'train', 'valid', 'test', 'model')"
                )

    def early_stopping(
        self, metric_names: Sequence[str] = ('g_mean', 'loss'), metric_values: Sequence[float] = (0.0, 0.0),
        patience: int = EARLY_STOPPING_PATIENCE, *args
    ):
        for metric_name, metric_value in zip(metric_names, metric_values):
            self.metric_records[metric_name].append(metric_value)

        for key, values in self.metric_records.items():
            try:
                diff = np.abs(np.diff(values)[-patience:])
                threshold = self.thresholds[key]
                if np.all(diff <= threshold):
                    print(f"Training has converged for the metric: {key}")
                    return True
            except IndexError:
                return False
            except KeyError:
                raise KeyError(f"Threshold not found for the metric: {key}")
            except Exception as e:
                raise e
            finally:
                print("Training has not converged yet.")
                return False

    def save_model(self, path: SAVED_MODELS_PATH, *args):
        pass


# def train_phase_one(
#     data: np.ndarray,
#     test_data: np.ndarray,
#     model: BaseNeuralNetwork,
#     loss_fn: nn.Module,
#     optimizer,
#     n_steps: int,
#     cost_matrix: torch.Tensor,
#     sample_model,
# ):
#     for epoch in tqdm(range(N_EPOCHS)):
#         count = 0
#         print(f"Epoch {epoch + 1}\n-------------------------------")
#         model.train()
#
#         for _ in tqdm(range(n_steps)):
#             count += 1
#             X_train, y_train = sample_model.sample()
#             optimizer.zero_grad()
#             pred = model(X_train)
#             loss = loss_fn(pred, y_train)
#             loss.backward()
#             optimizer.step()
#             if count % N_STEPS_TO_PRINT == 0:
#                 evaluate(data=data, model=model, loss_fn=loss_fn, mode="step", cost_matrix=cost_matrix)
#
#         evaluate(data=data, model=model, loss_fn=loss_fn, mode="train", cost_matrix=cost_matrix)
#         evaluate(data=test_data, model=model, loss_fn=loss_fn, mode="test", cost_matrix=cost_matrix)
#
#
# def train_phase_two(
#     data: np.ndarray,
#     test_data: np.ndarray,
#     model: BaseNeuralNetwork,
#     loss_fn: nn.Module,
#     optimizer,
#     n_steps: int,
#     cost_matrix: torch.Tensor,
#     sample_model,
# ):
#     X, y = data[:, :-1], data[:, -1]
#     model.layers[0].frozen()
#     active_layers = model.layers[1]
#     X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
#     y = torch.tensor(y, dtype=torch.long)
#     X = model.layers[0].forward(X)
#     for epoch in tqdm(range(N_EPOCHS)):
#         count = 0
#         print(f"Epoch {epoch + 1}\n-------------------------------")
#         active_layers.train()
#
#         for _ in tqdm(range(n_steps)):
#             count += 1
#             X_train, y_train = sample_model.sample()
#             optimizer.zero_grad()
#             pred = active_layers(X_train)
#             loss = loss_fn(pred, y_train)
#             loss.backward()
#             optimizer.step()
#             if count % N_STEPS_TO_PRINT == 0:
#                 model.layers[1] = active_layers
#                 evaluate(data=data, model=model, loss_fn=loss_fn, mode="step", cost_matrix=cost_matrix)
#
#         model.layers[1] = active_layers
#
#         evaluate(data=data, model=model, loss_fn=loss_fn, mode="train", cost_matrix=cost_matrix)
#         evaluate(data=test_data, model=model, loss_fn=loss_fn, mode="test", cost_matrix=cost_matrix)
#
#
# def two_phase_train(
#     data: np.ndarray,
#     test_data: np.ndarray,
#     model: BaseNeuralNetwork,
#     loss_fn: nn.Module,
#     cost_matrix: torch.Tensor,
#     sample_model_one,
#     sample_model_two,
# ):
#     n_steps = data.shape[0] // BATCH_SIZE
#     optimizer = torch.optim.Adam(model.parameters(), ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
#
#     train_phase_one(data, test_data, model, loss_fn, optimizer, n_steps, cost_matrix, sample_model_one)
#
#     optimizer = torch.optim.Adam(model.parameters(), ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
#     train_phase_two(data, test_data, model, loss_fn, optimizer, n_steps, cost_matrix, sample_model_two)
