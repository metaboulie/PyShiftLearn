import os
from abc import abstractmethod, ABC
from functools import cache
from typing import TypeVar

import anndata as ad
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyshiftlearn.config import DATASET_PATH, RESULTS_PATH

T = TypeVar("T")


class BaseInStream(ABC):
    def __init__(self, path: str = DATASET_PATH):
        self.path = path
        self.datasets: dict[str, T] = dict()
        self.unique_labels: dict[str, list[str]] = dict()
        self.np_datasets: dict[str, np.ndarray] = dict()

    @cache
    @abstractmethod
    def read_datasets(self, data_type_suffix: str) -> dict[str, T]:
        """Read all datasets inside a given folder and use a dictionary to store them

        Returns
        -------
        dict[str, T]
            A dictionary whose keys being the name of each dataset, values being the data of each dataset
        """
        pass

    @cache
    @abstractmethod
    def get_unique_label(self, *args):
        """Get the unique labels of the dataset"""
        pass

    @cache
    @abstractmethod
    def convert_data_to_numpy(self, *args):
        """Convert data to numpy array"""
        pass


class BaseOutStream(ABC):
    def __init__(self, path: str = RESULTS_PATH, _object: T = None):
        self.path = path
        self._object = _object

    def write_results(self)]:
        """Write all results inside a given folder and use a dictionary to store them"""
        pass


class H5adInStream(BaseInStream):
    def __init__(self, path: str = DATASET_PATH):
        super().__init__(path)

    def read_datasets(self, data_type_suffix=".h5ad") -> dict[str, ad.AnnData]:
        """Read all datasets inside a given folder and use a dictionary to store them

        Returns:
            dict[str, ad.AnnData]
                A dictionary whose keys being the name of each dataset, values being the data of each dataset
        """
        self.datasets = {
            os.path.splitext(file)[0]: ad.read_h5ad(os.path.join(root, file))
            for root, _, files in os.walk(self.path)
            for file in files
            if file.endswith(data_type_suffix)
        }
        return self.datasets

    def get_unique_label(self, obs_name: str) -> dict[str, list[str]]:
        """Get the unique labels of the dataset"""
        self.unique_labels = {
            key: self.datasets[key].obs[obs_name].unique()
            for key in self.datasets.keys()
        }
        return self.unique_labels

    def convert_data_to_numpy(self, obs_name: str):
        """Convert data to numpy array"""
        for dataset_name, adata in self.datasets.items():
            # Initialize label encoder
            label_encoder = LabelEncoder()
            # Encode 'cell_type' to numeric labels
            encoded_values = label_encoder.fit_transform(adata.obs[obs_name])
            # Append encoded values to the data
            self.np_datasets[dataset_name] = np.concatenate((np.array(adata.X), encoded_values[:, None]), axis=1)
        return self.np_datasets
