from abc import ABC, abstractmethod

import numpy as np
import scipy
import torch
from imblearn.combine import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn.linear_model import LogisticRegression

from pyshiftlearn.config import BATCH_SIZE
from pyshiftlearn.utils.train import feature_label_split


class BaseSample(ABC):
    def __init__(self, data: np.ndarray, batch_size: int = BATCH_SIZE):
        self.batch_size: int = batch_size
        self.data: np.ndarray = data
        self.size = data.shape[0]

    @abstractmethod
    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class SimpleSample(BaseSample):
    def __init__(self, data: np.ndarray, batch_size: int = BATCH_SIZE, replace: bool = False):
        super(SimpleSample, self).__init__(data, batch_size)
        self.choices = np.random.choice(range(self.size), self.batch_size, replace)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return feature_label_split(self.data[self.choices])


class ImblearnSample(BaseSample):
    def __init__(self, data: np.ndarray, batch_size: int = BATCH_SIZE, method: str = "over",
                 sampler: str = "RandomOverSampler", **kwargs):
        super(ImblearnSample, self).__init__(data, batch_size)
        self.sample_model = self.get_sampler(method, sampler, **kwargs)

    @staticmethod
    def get_sampler(method: str, sampler: str, **kwargs):
        match method:
            case "over":
                match sampler:
                    case "RandomOverSampler":
                        return RandomOverSampler(**kwargs)
                    case "SMOTE":
                        return SMOTE(n_jobs=-1, **kwargs)
                    case "SMOTENC":
                        return SMOTENC(n_jobs=-1, **kwargs)
                    case "SMOTEN":
                        return SMOTEN(n_jobs=-1, **kwargs)
                    case "ADASYN":
                        return ADASYN(n_jobs=-1, **kwargs)
                    case "BorderlineSMOTE1":
                        return BorderlineSMOTE(n_jobs=-1, kind="borderline-1", **kwargs)
                    case "BorderlineSMOTE2":
                        return BorderlineSMOTE(n_jobs=-1, kind="borderline-2", **kwargs)
                    case "KMeansSMOTE":
                        return KMeansSMOTE(n_jobs=-1, **kwargs)
                    case "SVMSMOTE":
                        return SVMSMOTE(n_jobs=-1, **kwargs)
                    case _:
                        raise ValueError(f"Invalid sampler :{sampler}")

            case "under":
                match sampler:
                    case "ClusterCentroids":
                        return ClusterCentroids(**kwargs)
                    case "CondensedNearestNeighbour":
                        return CondensedNearestNeighbour(n_jobs=-1, **kwargs)
                    case "EditedNearestNeighbours":
                        return EditedNearestNeighbours(n_jobs=-1, **kwargs)
                    case "RepeatedEditedNearestNeighbours":
                        return RepeatedEditedNearestNeighbours(n_jobs=-1, **kwargs)
                    case "AllKNN":
                        return AllKNN(n_jobs=-1, **kwargs)
                    case "InstanceHardnessThreshold":
                        # Check if the `estimator` parameter is set
                        if "estimator" not in kwargs:
                            kwargs["estimator"] = LogisticRegression(max_iter=1000000, solver="lbfgs")
                        return InstanceHardnessThreshold(n_jobs=-1, **kwargs)
                    case "NearMiss1":
                        return NearMiss(version=1, n_jobs=-1, **kwargs)
                    case "NearMiss2":
                        return NearMiss(version=2, n_jobs=-1, **kwargs)
                    case "NearMiss3":
                        return NearMiss(version=3, n_jobs=-1, **kwargs)
                    case "NeighbourhoodCleaningRule":
                        return NeighbourhoodCleaningRule(n_jobs=-1, **kwargs)
                    case "OneSidedSelection":
                        return OneSidedSelection(n_jobs=-1, **kwargs)
                    case "RandomUnderSampler":
                        return RandomUnderSampler(**kwargs)
                    case "TomekLinks":
                        return TomekLinks(n_jobs=-1, **kwargs)
                    case _:
                        raise ValueError(f"Invalid sampler :{sampler}")
            case "combine":
                match sampler:
                    case "SMOTEENN":
                        return SMOTEENN(n_jobs=-1, **kwargs)
                    case "SMOTETomek":
                        return SMOTETomek(n_jobs=-1, **kwargs)
                    case _:
                        raise ValueError(f"Invalid sampler: {sampler}")
            case _:
                raise ValueError(f"Invalid method: {method}")
        pass

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = self.data[:, :-1], self.data[:, -1]
        X_batch, y_batch = self.sample_model.fit_resample(X, y)
        X_tensor = torch.tensor(X_batch, dtype=torch.float32, requires_grad=True)
        y_tensor = torch.tensor(y_batch, dtype=torch.long)
        return X_tensor, y_tensor


class Resample(BaseSample):
    """This class inherits from BaseSample, see the doc of BaseSample for more details

    Utilize a given distribution to generate weights for each observation of the input data and use these
    weights to sample the data

    Parameters:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    distribution: scipy.stats.rv_continuous, optional
        The distribution to generate each probability from, by default scipy.stats.uniform

    Attributes:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    distribution: scipy.stats.rv_continuous, optional
        The distribution to generate each probability from, by default scipy.stats.uniform
    size: int
        The count of observations in the data
    num_labels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    weights: np.ndarray
        The weights for each observation

    Examples
    ------
    resampleModel = Resample(data=data)
    X_batch, y_batch = resampleModel.sample(distribution=distribution, *args)
    """

    def __init__(self, data: np.ndarray, batch_size: int = BATCH_SIZE,
                 **kwargs) -> None:
        super().__init__(data, batch_size)
        self.distribution = self.init_distribution(**kwargs)
        self.size = data.shape[0]
        self.num_labels = len(np.unique(data[:, -1]))
        self.weights: np.ndarray[float] = np.ndarray(data.shape[0])
        self.choices: np.ndarray[int] = np.ndarray(batch_size)

    @staticmethod
    def init_distribution(**kwargs) -> scipy.stats.rv_continuous:
        """Initialize the distribution of the weights"""
        return scipy.stats.rv_continuous(**kwargs)

    def generate_probabilities(
        self,
        **kwargs,
    ) -> np.ndarray:
        """Generate an 1D-array filled with probabilities from a given distribution

        Returns
        -------
        np.ndarray
            An 1D-array filled with probabilities from the given distribution
        """
        return scipy.special.softmax(self.distribution.rvs(size=self.size, **kwargs), axis=0)

    def sample(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the generated weights to sample the data, overrides from the `sample` method from the superClass Sample

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        self.weights = self.generate_probabilities(**kwargs)
        self.choices = np.random.choice(range(self.size), self.batch_size, False, self.weights)

        return feature_label_split(self.data[self.choices])


class Bootstrap(BaseSample):
    """This class inherits from BaseSample, see the doc of BaseSample for more details

    This class split batch_size into num_labels groups and sample each label relevant count of observations by Bootstrap

    Parameters:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled

    Attributes:
    ------------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    size: int
        The count of observations in the data
    num_labels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    change_indexes: list
        The list of the indexes where the label of the sorted data changes

    Examples
    ------
    bootstrapModel = Bootstrap(data=data)
    X_batch, y_batch = bootstrapModel.sample()
    """

    def __init__(self, data: np.ndarray, batch_size: int = BATCH_SIZE) -> None:
        super().__init__(data, batch_size)
        self.data = self.data[self.data[:, -1].argsort()]
        self.choices: np.ndarray[int] = np.ndarray(batch_size)
        self.num_labels = len(np.unique(self.data[:, -1]))
        self.change_indexes: list = list(np.where(np.diff(self.data[:, -1]))[0] + 1)
        self.change_indexes.append(self.data.shape[0])
        self.change_indexes.insert(0, 0)

    @property
    def get_num(self) -> np.ndarray:
        """Split the range of batch_size to num_labels groups, the count of numbers in each group accords
        how many observations should be sampled by Bootstrap for each label

        Returns
        -------
        np.ndarray
            The array of how many observations should be sampled by Bootstrap for each label
        """
        nums = sorted(np.random.choice(range(1, self.batch_size), self.num_labels - 1, False))
        nums.append(self.batch_size)
        nums.insert(0, 0)
        return np.diff(nums)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Utilize the generated list to sample the data by Bootstrap,
        the number of sampled observations for each label should be equal to the relevant number in the list

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        nums = self.get_num
        for i in range(len(self.change_indexes) - 1):
            self.choices += list(
                np.random.choice(
                    range(self.change_indexes[i], self.change_indexes[i + 1]),
                    nums[i],
                    True,
                )
            )
        return feature_label_split(self.data[self.choices])


class SampleWithImputation(BaseSample):
    """Impute data and insert them into the trainSet to make trainSet balanced

    Parameters:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled

    Attributes:
    ------------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    size: int
        The count of observations in the data
    num_labels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    change_indexes: list
        The list of the indexes where the label of the sorted data changes
    max_num: int
        The count of observations of the most frequent label

    Examples
    -------
    SWIModel = SampleWithImputation(data=data)
    SWIModel.iterLabels()
    X_Batch, y_Batch = SWIModel.sample()
    """

    def __init__(self, data: np.ndarray, batch_size: int = BATCH_SIZE) -> None:
        super().__init__(data, batch_size)
        self.data = self.data[self.data[:, -1].argsort()]
        self.choices: np.ndarray[int] = np.ndarray(batch_size)
        self.change_indexes: list = list(np.where(np.diff(self.data[:, -1]))[0] + 1)
        self.change_indexes.append(self.data.shape[0])
        self.change_indexes.insert(0, 0)
        self.max_num: int = max(np.diff(self.change_indexes))

    def iter_labels(self) -> None:
        """Iterate each label, and impute data"""
        for i in range(len(self.change_indexes) - 1):
            num_imputation = (
                self.max_num - self.change_indexes[i + 1] + self.change_indexes[i]
            )  # The number of the data to be imputed equals self.max_num minus the count of observations of this label
            label_mean, label_std = self.feature_stats_agg(label_counter=i)
            imputedData = self.impute_data(i, num_imputation, label_mean, label_std)
            self.data = np.concatenate((self.data, imputedData), axis=0)
        self.size = self.data.shape[0]  # Update the size of the data
        return None

    @staticmethod
    def impute_data(encoded_label: int, num: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Impute data for each label according to its means and stds for each feature

        Parameters
        ----------
        encoded_label : int
            The encoded label for each label
        num : int
            How many observations should be imputed
        mean : np.ndarray
            An array storing mean values for each feature
        std : np.ndarray
            An array storing std values for each feature

        Returns
        -------
        np.ndarray
            Imputed data
        """
        return np.concatenate(
            (
                np.random.normal(loc=mean, scale=std, size=(num, len(mean))),
                np.full((num, 1), encoded_label),
            ),
            axis=1,
        )

    def feature_stats_agg(self, label_counter: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the mean and std for each feature given a set of data with same label

        Parameters
        ----------
        label_counter: int
            A mark of the current label

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
             The mean and std for each feature of this label
        """
        label_data = self.data[
            range(self.change_indexes[label_counter], self.change_indexes[label_counter + 1])
        ]  # self.data is sorted, see the __post_init__ of Bootstrap for details
        return label_data.mean(axis=0)[:-1], label_data.std(axis=0)[:-1]

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the balanced dataset

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        self.choices = np.random.choice(range(self.size), self.batch_size, True)
        return feature_label_split(self.data[self.choices])
