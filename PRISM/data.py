import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .utils import free_memory


class DataProcessor:
    """
    A utility class for processing and manipulating data for machine learning tasks.

    This class provides static methods for various data processing operations,
    including loading data, scaling, creating data lists, and preparing data loaders.
    """

    @staticmethod
    def load_data(file_path: str) -> Dict:
        """
        Load data from a file using PyTorch's load function.

        Args:
            file_path (str): Path to the file to be loaded.

        Returns:
            Dict: Loaded data as a dictionary.
        """
        data = torch.load(file_path, weights_only=False, map_location="cpu")
        free_memory()
        return data

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        df = pd.read_csv(file_path)
        free_memory()
        return df

    @staticmethod
    def create_data_list(data_dict: Dict, keys: List) -> List:
        """
        Create a list of data items from a dictionary using specified keys.

        Args:
            data_dict (Dict): Dictionary containing the data.
            keys (List): List of keys to extract data from the dictionary.

        Returns:
            List: List of data items corresponding to the given keys.
        """
        data_list = [data_dict[i] for i in tqdm(keys, desc="Creating data list")]
        free_memory()
        return data_list

    @staticmethod
    def create_gene_data_list(
        data_dict: Dict,
        keys: List,
        nscs: List,
        cell_lines: List,
        dti: pd.DataFrame,
        use_data_types: Optional[List[str]] = None,
    ) -> List:
        """
        Create a list of gene data items with additional information.

        Args:
            data_dict (Dict): Dictionary containing gene data.
            keys (List): List of keys to extract data from the dictionary.
            nscs (List): List of NSC identifiers.
            cell_lines (List): List of cell line identifiers.
            dti (pd.DataFrame): DataFrame containing DTI (Drug-Target Interaction) information.
            use_data_types (Optional[List[str]]): List of data types to include. Defaults to None.

        Returns:
            List: List of gene data items with additional information.
        """
        data = []
        data_type_indices = {"exp": 0, "met": 1, "cop": 2, "mut": 3}

        dti_values_dict = {
            nsc: torch.tensor(dti.loc[nsc].values, dtype=torch.float32).unsqueeze(1)
            for nsc in tqdm(nscs, desc="Calculating DTI values")
        }
        free_memory()

        for i, j, cell_line in tqdm(
            zip(keys, nscs, cell_lines), desc="Creating gene data list", total=len(keys)
        ):
            tmp = data_dict[i]
            original_x = tmp.x

            if use_data_types:
                selected_indices = [
                    data_type_indices[dt]
                    for dt in use_data_types
                    if dt in data_type_indices
                ]
                original_x = original_x[:, selected_indices]

            new_tmp = copy.copy(tmp)
            new_tmp.x = original_x
            new_tmp.dti = dti_values_dict[j]
            new_tmp.nsc = j
            new_tmp.cell_line = cell_line

            data.append(new_tmp)

        free_memory()
        return data

    @staticmethod
    def load_target(file_path: str) -> torch.Tensor:
        """
        Load target data from a NumPy file and convert it to a PyTorch tensor.

        Args:
            file_path (str): Path to the NumPy file containing target data.

        Returns:
            torch.Tensor: Target data as a PyTorch tensor.
        """
        target = torch.tensor(np.load(file_path), dtype=torch.float32)
        free_memory()
        return target

    @staticmethod
    def create_data_loader(
        data, batch_size: int, num_workers: int = 2, pin_memory: bool = True
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for the given data.

        Args:
            data: Data to be loaded into the DataLoader.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading. Defaults to 2.
            pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory. Defaults to True.

        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        loader = DataLoader(
            data,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        free_memory()
        return loader

    @staticmethod
    def create_data_for_nsc(
        nsc: str,
        gene_dict: Dict,
        drug_dict: Dict,
        dti: pd.DataFrame,
        cell_lines: List[str],
        use_data_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Create data for a specific NSC.

        Args:
            nsc (str): NSC identifier.
            gene_dict (Dict): Dictionary containing gene data.
            drug_dict (Dict): Dictionary containing drug data.
            dti (pd.DataFrame): DataFrame containing DTI information.
            cell_lines (List[str]): List of cell line identifiers.
            use_data_types (Optional[List[str]]): List of data types to include. Defaults to None.

        Returns:
            Dict: Dictionary containing drug and gene data for the specified NSC.
        """
        drug_data = drug_dict[nsc]

        gene_data = DataProcessor.create_gene_data_list(
            gene_dict,
            cell_lines,
            [nsc] * len(cell_lines),
            cell_lines,
            dti,
            use_data_types,
        )

        return {
            "drug": drug_data,
            "gene": gene_data,
        }


class DataManager:
    """
    A class for managing and processing data for machine learning tasks.

    This class handles loading, processing, and creating data loaders for
    train, validation, and test datasets.
    """

    def __init__(self, config):
        """
        Initialize the DataManager.

        Args:
            batch_size (int): Number of samples per batch. Defaults to 10.
            n_samples (int): Number of samples to use from each dataset. Defaults to 100.
            use_data_types (Optional[List[str]]): List of data types to include. Defaults to None.
        """
        self.batch_size = config["BATCH_SIZE"]
        self.n_samples = config["NUM_SAMPLES"]
        self.use_data_types = config["USE_DATA_TYPES"]
        self.dp = DataProcessor()

    def load_and_process_data(self):
        """
        Load and process data for train, validation, and test datasets.

        This method loads gene and drug data, DTI information, and processes
        the datasets accordingly.
        """
        print("Starting data loading and processing...")
        self.gene_dict = self.dp.load_data("data/gene_dict.pt")
        self.drug_dict = self.dp.load_data("data/data_dict.pt")
        self.DTI = pd.read_csv("data/dti.csv.gz", index_col=0)

        datasets = {
            "train": self.dp.load_csv("data/train_IC50.csv"),
            "valid": self.dp.load_csv("data/valid_IC50.csv"),
            "test": self.dp.load_csv("data/test_IC50.csv"),
        }

        for name, dataset in datasets.items():
            dataset = dataset.iloc[: self.n_samples]
            print(f"{name.capitalize()} data size: {len(dataset)}")
            self._process_dataset(name, dataset)
            free_memory()

        print("Data loading and processing completed.")

    def _process_dataset(self, dataset_name: str, dataset: pd.DataFrame):
        """
        Process a single dataset (train, validation, or test).

        Args:
            dataset_name (str): Name of the dataset (e.g., "train", "valid", "test").
            dataset (pd.DataFrame): DataFrame containing the dataset information.
        """
        print(f"Creating {dataset_name} data...")
        setattr(
            self,
            f"{dataset_name}_drug",
            self.dp.create_data_list(self.drug_dict, dataset["NSC"]),
        )
        setattr(
            self,
            f"{dataset_name}_gene",
            self.dp.create_gene_data_list(
                self.gene_dict,
                dataset["CELL_NAME"],
                dataset["NSC"],
                dataset["CELL_NAME"],
                self.DTI,
                self.use_data_types,
            ),
        )
        setattr(
            self,
            f"{dataset_name}_target",
            self.dp.load_target(f"data/{dataset_name}_IC50_labels.npy")[
                : self.n_samples
            ],
        )
        print(f"{dataset_name.capitalize()} data created.")
        free_memory()

    def create_data_loaders(self):
        """
        Create data loaders for train, validation, and test datasets.

        This method creates separate data loaders for drug data, gene data,
        and target data for each dataset.
        """
        print("Starting creation of data loaders...")

        for dataset in ["train", "valid", "test"]:
            for data_type in ["drug", "gene"]:
                setattr(
                    self,
                    f"{dataset}_{data_type}_loader",
                    self.dp.create_data_loader(
                        getattr(self, f"{dataset}_{data_type}"), self.batch_size
                    ),
                )

            setattr(
                self,
                f"{dataset}_target_loader",
                self.dp.create_data_loader(
                    getattr(self, f"{dataset}_target"), self.batch_size
                ),
            )
            free_memory()

        print("Creation of data loaders completed.")
        free_memory()

    def create_data_for_nsc(self, nsc: str, cell_lines: List[str]) -> Dict:
        """
        Create data for a specific NSC.

        Args:
            nsc (str): NSC identifier.
            cell_lines (List[str]): List of cell line identifiers.

        Returns:
            Dict: Dictionary containing drug and gene data for the specified NSC.
        """
        return self.dp.create_data_for_nsc(
            nsc,
            self.gene_dict,
            self.drug_dict,
            self.DTI,
            cell_lines,
            self.use_data_types,
        )

    def get_data_loaders(self):
        """
        Get all data loaders.

        Returns:
            Dict: A dictionary containing all data loaders for train, validation, and test sets.
        """
        return {
            "train": {
                "drug": self.train_drug_loader,
                "gene": self.train_gene_loader,
                "target": self.train_target_loader,
            },
            "valid": {
                "drug": self.valid_drug_loader,
                "gene": self.valid_gene_loader,
                "target": self.valid_target_loader,
            },
            "test": {
                "drug": self.test_drug_loader,
                "gene": self.test_gene_loader,
                "target": self.test_target_loader,
            },
        }

    def get_batch_size(self):
        """
        Get the batch size used for creating data loaders.

        Returns:
            int: The batch size.
        """
        return self.batch_size

    def get_use_data_types(self):
        """
        Get the types of data being used.

        Returns:
            List[str]: List of data types being used (e.g., ['exp', 'met', 'cop', 'mut']).
        """
        return self.use_data_types

    # Usage:
    # config = {
    #     "NSC": "123456",  # Specify the NSC number
    #     ...  # Other configuration parameters
    # }
    # data_manager = DataManager(config)
    # data_manager.load_and_process_data()
    # data_manager.create_data_loaders()
    #
    # # Get data for the specified NSC
    # nsc_data = data_manager.get_nsc_data()
    #
    # # Get all data loaders
    # loaders = data_manager.get_data_loaders()
    # train_gene_loader = loaders['train']['gene']
    #
    # # Get batch size
    # batch_size = data_manager.get_batch_size()
    #
    # # Get data types being used
    # data_types = data_manager.get_use_data_types()
