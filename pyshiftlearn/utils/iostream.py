"""Read and output data"""

import os
from abc import ABC, abstractmethod
from functools import cache
from typing import TypeVar

import anndata as ad
import numpy as np
from sklearn.preprocessing import LabelEncoder

from pyshiftlearn.config import DATASET_PATH, RESULTS_PATH

DATESET_TYPE = TypeVar("DATESET_TYPE")


def model_exist(directory: str, dataset_name: str) -> tuple[bool, str]:
    """
    Check if a model file exists in the specified directory.

    Parameters:
        directory (str): The directory where the model file is located.
        dataset_name (str): The name of the dataset used to train the model.

    Returns:
        tuple[bool, str]: A tuple containing a boolean value indicating whether
        the model file exists and the path to the model file.
    """
    # Remove the "_train" suffix from dataset_name
    model_name = dataset_name.replace("_train", "") + ".pt"

    # Construct the full path to the model file
    model_path = os.path.join(directory, model_name)

    # Print the model path for debugging
    print(f"Model path: {model_path}")

    # Check if the model file exists
    model_exists = os.path.exists(model_path)

    # Print whether the model file exists or not for debugging
    print(f"Model exists: {model_exists}")

    return model_exists, model_path


class BaseInStream(ABC):
    """
    Base class for reading datasets

    Parameters
    ----------
    path : str
        The path of the datasets

    Attributes
    ----------
    path : str
        The path of the datasets
    datasets : dict[str, T]
        A dictionary whose keys being the name of each dataset, values being the data of each dataset
    unique_labels : dict[str, list[str]]
        A dictionary whose keys being the name of each dataset, values being the unique labels of each dataset
    np_datasets : dict[str, np.ndarray]
        A dictionary whose keys being the name of each dataset, values being the data of each dataset in numpy format

    Methods
    -------
    read_datasets(self, data_type_suffix: str)
        Read all datasets inside a given folder and use a dictionary to store them

    get_unique_label(self, obs_name: str)
        Get the unique labels of the dataset

    convert_data_to_numpy(self, obs_name: str)
        Convert data to numpy array
    """
    def __init__(self, path: str = DATASET_PATH):
        self.path = path
        self.datasets: dict[str, DATESET_TYPE] = dict()
        self.unique_labels: dict[str, list[str]] = dict()
        self.np_datasets: dict[str, np.ndarray] = dict()

    @cache
    @abstractmethod
    def read_datasets(self, data_type_suffix: str) -> dict[str, DATESET_TYPE]:
        """Read all datasets inside a given folder and use a dictionary to store them

        Parameters
        ----------
        data_type_suffix : str
            The suffix of the data type

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
    """
    Base class for writing results

    Parameters
    ----------
    path : str
        The path of the results

    Attributes
    ----------
    path : str
        The path of the results
    _object : T
        The object to be written

    Methods
    -------
    write_results(self)
        Write all results inside a given folder and use a dictionary to store them
    """
    def __init__(self, path: str = RESULTS_PATH, _object: DATESET_TYPE = None):
        self.path = path
        self._object = _object

    def write_results(self):
        """Write all results inside a given folder and use a dictionary to store them"""
        pass

    @staticmethod
    def model_exist(directory: str, dataset_name: str) -> tuple[bool, str]:
        """
        Check if a model file exists in the specified directory.

        Parameters:
            directory (str): The directory where the model file is located.
            dataset_name (str): The name of the dataset used to train the model.

        Returns:
            tuple[bool, str]: A tuple containing a boolean value indicating whether
            the model file exists and the path to the model file.
        """
        return model_exist(directory, dataset_name)


class H5adInStream(BaseInStream):
    """Read all .h5ad datasets inside a given folder and use a dictionary to store them

    Parameters
    ----------
    path : str
        The path of the datasets

    Attributes
    ----------
    path : str
        The path of the datasets
    datasets : dict[str, ad.AnnData]
        A dictionary whose keys being the name of each dataset, values being the data of each dataset
    unique_labels : dict[str, list[str]]
        A dictionary whose keys being the name of each dataset, values being the unique labels of each dataset
    np_datasets : dict[str, np.ndarray]
        A dictionary whose keys being the name of each dataset, values being the data of each dataset in numpy format

    Methods
    -------
    read_datasets(self, data_type_suffix: str)
        Read all datasets inside a given folder and use a dictionary to store them

    get_unique_label(self, obs_name: str)
        Get the unique labels of the dataset

    convert_data_to_numpy(self, obs_name: str)
        Convert data to numpy array
    """
    def __init__(self, path: str = DATASET_PATH):
        super().__init__(path)

    def read_datasets(self, data_type_suffix=".h5ad") -> dict[str, ad.AnnData]:
        """Read all datasets inside a given folder and use a dictionary to store them

        Parameters
        ----------
        data_type_suffix : str
            The suffix of the data type

        Returns
        -------
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
        """Get the unique labels of the dataset

        Parameters
        ----------
        obs_name : str
            The name of the observation
        Returns
        -------
        dict[str, list[str]]
            A dictionary whose keys being the name of each dataset, values being the unique labels of each dataset
        """
        self.unique_labels = {
            key: self.datasets[key].obs[obs_name].unique()
            for key in self.datasets.keys()
        }
        return self.unique_labels

    def convert_data_to_numpy(self, obs_name: str):
        """Convert data to numpy array

        Parameters
        ----------
        obs_name : str
            The name of the observation
        Returns
        -------
        dict[str, np.ndarray]
            A dictionary whose keys being the name of each dataset,
            values being the data of each dataset in numpy format
        """
        for dataset_name, adata in self.datasets.items():
            # Initialize label encoder
            label_encoder = LabelEncoder()
            # Encode 'cell_type' to numeric labels
            encoded_values = label_encoder.fit_transform(adata.obs[obs_name])
            # Append encoded values to the data
            self.np_datasets[dataset_name] = np.concatenate((np.array(adata.X), encoded_values[:, None]), axis=1)
        return self.np_datasets
