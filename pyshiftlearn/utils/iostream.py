"""Read and output data"""

import os
from abc import ABC, abstractmethod
from functools import cache
from typing import TypeVar
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pyshiftlearn.config import DATASET_PATH, RESULTS_PATH

DATESET_TYPE = TypeVar("DATESET_TYPE")


class BaseInStream(ABC):
    """
    Base class for reading datasets

    Parameters:
        path : str
            The path of the datasets

    Attributes:
        path : str
            The path of the datasets.
        datasets : dict[str, DATESET_TYPE]
            A dictionary whose keys being the name of each dataset, values being the data of each dataset.
        unique_labels : dict[str, tuple[str]]
            A dictionary whose keys being the name of each dataset, values being the unique labels of each dataset.
        np_datasets : dict[str, np.ndarray]
            A dictionary whose keys being the name of each dataset,
            values being the data of each dataset in numpy format.

    Methods:
        read_datasets(self, data_type_suffix: str)
            Read all datasets inside a given folder and use a dictionary to store them

        Get_unique_label(self, *args)
            Get the unique labels of the dataset

        Convert_data_to_numpy(self, *args)
            Convert data to a numpy array
    """
    def __init__(self, path: str = DATASET_PATH):
        self.path = path
        self.datasets: dict[str, DATESET_TYPE] = defaultdict(DATESET_TYPE)
        self.unique_labels: dict[str, tuple[str]] = defaultdict(tuple[str])
        self.np_datasets: dict[str, np.ndarray] = defaultdict(np.ndarray)

    @cache
    @abstractmethod
    def read_datasets(self, data_type_suffix: str):
        """Read all datasets inside a given folder and use a dictionary to store them

        Parameters:
            data_type_suffix : str
                The suffix of the data type
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

    Parameters:
        path : str
            The path of the results

    Attributes:
        path : str
            The path of the results
        _object : T
            The object to be written

    Methods:
        write_results(self)
            Write all results inside a given folder and use a dictionary to store them
    """
    def __init__(self, path: str = RESULTS_PATH, _object: DATESET_TYPE = None):
        self.path = path
        self._object = _object

    def write_results(self):
        """Write all results inside a given folder and use a dictionary to store them"""
        pass


class H5adInStream(BaseInStream):
    """
    Read all .h5ad datasets inside a given folder and use a dictionary to store them

    Parameters:
        path : str
            The path of the datasets.

    Attributes:
        path : str
            The path of the datasets.
        datasets : dict[str, ad.AnnData]
            A dictionary whose keys being the name of each dataset, values being the data of each dataset.
        unique_labels : dict[str, tuple[str]]
            A dictionary whose keys being the name of each dataset, values being the unique labels of each dataset.
        np_datasets : dict[str, np.ndarray]
            A dictionary whose keys being the name of each dataset,
            values being the data of each dataset in numpy format.

    Methods:
        read_datasets(self, data_type_suffix: str)
            Read all datasets inside a given folder and use a dictionary to store them

        Get_unique_label(self, obs_name: str)
            Get the unique labels of the dataset

        Convert_data_to_numpy(self, obs_name: str)
            Convert data to a numpy array
    """
    def __init__(self, path: str = DATASET_PATH):
        super().__init__(path)

    def read_datasets(self, data_type_suffix=".h5ad") -> None:
        """
        Read all .h5ad datasets inside a given folder and use a dictionary to store them

        Parameters:
            data_type_suffix : str
                The suffix of the data type
        """
        self.datasets = {
            os.path.splitext(file)[0]: ad.read_h5ad(os.path.join(root, file))
            for root, _, files in os.walk(self.path)
            for file in files
            if file.endswith(data_type_suffix)
        }
        return None

    def get_unique_label(self, obs_name: str) -> None:
        """Get the unique labels of the dataset

        Parameters:
            obs_name : str
                The name of the observation
        """
        self.unique_labels = {
            key: tuple(self.datasets[key].obs[obs_name].unique())
            for key in self.datasets.keys()
        }
        return None

    def convert_data_to_numpy(self, obs_name: str) -> None:
        """
        Convert data to a numpy ndarray

        Parameters:
            obs_name : str
                The name of the observation
        """
        for dataset_name, adata in self.datasets.items():
            # Initialize label encoder
            label_encoder = LabelEncoder()
            # Encode 'cell_type' to numeric labels
            encoded_values = label_encoder.fit_transform(adata.obs[obs_name])
            # Append encoded values to the data
            self.np_datasets[dataset_name] = np.concatenate((np.array(adata.X), encoded_values[:, None]), axis=1)
        return None


class PdInStream(BaseInStream):
    """
    Read all datasets that can be read by `pandas` inside a given folder and use a dictionary to store them

    Parameters:
        path : str
            The path of the datasets.

    Attributes:
        path : str
            The path of the datasets.
        datasets : dict[str, pd.DataFrame]
            A dictionary whose keys being the name of each dataset, values being the data of each dataset.
        unique_labels : dict[str, tuple[str]]
            A dictionary whose keys being the name of each dataset, values being the unique labels of each dataset.
        np_datasets : dict[str, np.ndarray]
            A dictionary whose keys being the name of each dataset,
            values being the data of each dataset in numpy format.

    Methods:
        read_datasets(self, data_type_suffix: str)
            Read all datasets that can be read by `pandas` inside a given folder and use a dictionary to store them

        Get_unique_label(self, target_name: str)
            Get the unique labels of the dataset

        Convert_data_to_numpy(self, target_name: str)
            Convert data to a numpy array
    """
    def __init__(self, path: str = DATASET_PATH):
        super().__init__(path)

    def read_datasets(self, data_type_suffix=".csv", **kwargs) -> None:
        """
        Read all datasets that can be read by `pandas` inside a given folder and use a dictionary to store them

        Parameters:
            data_type_suffix : str
                The suffix of the data type
        """
        match data_type_suffix:
            case ".csv":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_csv(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case ".tsv":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_csv(os.path.join(root, file), sep="\t", **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case ".xlsx" | ".xls" | ".xlsm" | ".xlsb" | ".odf" | ".ods" | ".odt":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_excel(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case ".parquet":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_parquet(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case ".feather":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_feather(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case ".json":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_json(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case ".pkl" | ".pickle":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_pickle(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case "html":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_html(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case "xml":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_xml(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case "orc":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_orc(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case "xport" | "sas7bdat":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_sas(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case "sav":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_spss(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case "dta":
                self.datasets = {
                    os.path.splitext(file)[0]: pd.read_stata(os.path.join(root, file), **kwargs)
                    for root, _, files in os.walk(self.path)
                    for file in files
                    if file.endswith(data_type_suffix)
                }
            case _:
                raise ValueError(f"Unsupported data type: {data_type_suffix}")
        return None

    def get_unique_label(self, target_name: str) -> None:
        """
        Get the unique labels of the dataset

        Parameters:
            target_name : str
                The name of the target
        """
        self.unique_labels = {
            key: tuple(self.datasets[key][target_name].unique())
            for key in self.datasets.keys()
        }
        return None

    def convert_data_to_numpy(self, target_name: str) -> None:
        """
        Convert data to a numpy ndarray

        Parameters:
            target_name : str
                The name of the target
        """
        for dataset_name, df in self.datasets.items():
            # Initialize label encoder
            label_encoder = LabelEncoder()
            # Encode 'cell_type' to numeric labels
            encoded_values = label_encoder.fit_transform(df[target_name])
            # Append encoded values to the data
            self.np_datasets[dataset_name] = np.concatenate(
                (np.array(df.drop(target_name, axis=1)), encoded_values.reshape(-1, 1)), axis=1
            )
        return None
