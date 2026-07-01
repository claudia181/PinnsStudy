"""
phy_sys_dataset.py
===========

This module implements the dataset containing the informations relative to a physical system
(subcalssing the torch.utils.data.Dataset class).

Attributes:
    - cols: a dictionary {(col_name: str, col_values: torch.Tensor)}, where each element represent 
            a column of the dataset and the column vaues are all tensors of the same length, 
            which is the number of rows (# data points);
    - length: the length of the dataset, i.e. the number of rows (# data points);
    - subkeys: some columns may be tensors of vector values (e.g. the vector of spatio-temporal 
               coordinates or the vector of parameters of the physical system); any of the entries 
               of these vectors is associated with a subkey, which is a string; subkeys is a dictionary
               {(col_key: str, subkey_list: List[str])}, where col_key is the column and subkey_list is the 
               associated (ordered) list of subkeys. E.g. for the spatio-temporal coordinates
               col_key = "spacetime", subkeys = ["x", "y", "t"].

'Public' methods:
    - columns: None -> List[Tensor].
        It returns the list of column tensors.
    - get_column: str, Callable[Tensor, bool] -> Tensor.
        It returns the elements of the column identified by the string for which the condition callable returns True.
    - add_column: str, Tensor, List[str] -> None.
        It add the column named with the string, valued with the tensor and with subkeys the list of strings.
    - set subkeys: str, List[str] -> None
        It sets the subkeys for the column identifyed by the string.
    - subsample: Tensor -> PhySysDataset
        It subsamples the rows/points corresponding to the indices in the tensor.
    - merge: PhySysDataset -> None.
        It merges the passed dataset with the one of the object. 
    - copy: None -> PhySysDataset.
        Shallow copy method: the tensors representing columns are shared.
    - deep_copy: None -> PhySysDataset.
        Deep copy method: Nothing shared.
    - save: str -> None
        It stores the dataset in the file identified by the path string.

Class methods:
    - load: str -> PhySysDataset
        It load the dataset in the file identified by the string.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Self

class PhySysDataset(Dataset):
    # ------------ Subclassing methods ------------
    def __init__(self, cols: List[Tuple[str, list|torch.Tensor]] | dict) -> None:
        self.cols = {}
        if type(cols) is list:
            if cols == []:
                self.length = 0
            else:
                self.length = len(cols[0][1])
            for col in cols:
                key, items = col[0], col[1]
                if len(items) != self.length:
                    raise ValueError(f"Length of column {key} is {len(items)}, but is expected to be {self.length}).")
            for col in cols:
                key, items = col[0], col[1]
                self.cols[key] = items
        elif type(cols) is dict:
            if cols == {}:
                self.length = 0
            else:
                self.length = len(list(cols.values())[0])
            for key, col in cols.items():
                if len(col) != self.length:
                    raise ValueError(f"Length of column {key} is {len(col)}, but is expected to be {self.length}.")
            self.cols = cols
        self.subkeys = {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict:
        item = {}
        for key in self.cols.keys():
            item[key] = self.cols[key][idx]
        return item
    
    # ------------ Class methods ------------
    def columns(self) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        None

        Returns
        -------
        List[torch.Tensor]
            The list of columns.
        """
        return [self.cols[key] for key in self.cols.keys()]
    
    def get_column(self, key: str, condition: Callable[[torch.Tensor], bool] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        key : str
            Column identifier.
        condition : Callable[[torch.Tensor], bool] = None
            Condition for the points to satisfy in order to be returned in the column tensor.

        Returns
        -------
        torch.Tensor
        """
        if key not in self.cols.keys():
            raise ValueError(f"Column of key {key} not in dataset.")
        col = self.cols[key]
        if condition is None:
            return col
        elif type(col) is torch.Tensor:
            return torch.stack([item for item in col if condition(item)])
        else:
            return [item for item in col if condition(item)]
    
    def add_column(self, key: str, col: torch.Tensor, subkeys: List[str] = []) -> None:
        """
        Parameters
        ----------
        key : str
            Column identifier.
        col : torch.Tensor
            Column vector.
        subkeys : List[str] = []
            Identifiers of the column dimentions.

        Returns
        -------
        None
        """
        if len(col) != self.length:
            raise ValueError(f"Column of length {len(col)}, but expected of length {self.length}.")
        if key in self.cols.keys():
            raise ValueError(f"Column {key} already present in the dataset.")
        self.cols[key] = col
        if subkeys != []:
            self.subkeys[key] = subkeys
    
    def set_subkeys(self, key: str, subkeys: List[str]) -> None:
        """
        Parameters
        ----------
        key : str
            Column identifier.
        subkeys : List[str]
            Subkeys for the column.

        Returns
        -------
        None
        """
        if key not in self.cols.keys():
            raise ValueError(f"Column of key {key} not in dataset.")
        if len(subkeys) != len(self.cols[key][0]):
            raise ValueError(f"Items of column {key} have {len(self.cols[key][0])} elements each, but {len(subkeys)} subkeys are passed.")
        self.subkeys[key] = subkeys
    
    def index(self, key: str, subkey: str) -> int:
        """
        Parameters
        ----------
        key : str
            The column key.
        subkey : str
            The column subkey.
        Returns
        -------
        int
            The index of the column subkey.
        """
        for i, sk in enumerate(self.subkeys[key]):
            if sk == subkey:
                return i
        raise ValueError(f"{subkey} not in column {key}.")
    
    def subsample(self, indices: torch.Tensor) -> Self:
        """
        Parameters
        ----------
        indices : torch.Tensor
            Indices of the points/rows to subsample.

        Returns
        -------
        PhySysDataset
            Dataset with subsampled columns.
        """
        subsampled_cols = {}
        for key, col in self.cols.items():
            subsampled_cols[key] = col[indices]
        subsampled_ds = PhySysDataset(subsampled_cols)
        subsampled_ds.subkeys = self.subkeys.copy()
        return subsampled_ds
    
    def merge(self, dataset: Self) -> None:
        """
        Parameters
        ----------
        dataset : PhySysDataset
            Dataset to merge with the current one.

        Returns
        -------
        None
        """
        if len(dataset.cols.keys()) != len(self.cols.keys()):
            raise ValueError(f"Different columns to merge: {self.cols.keys()} != {dataset.cols.keys()}.")
        for key, col in dataset.cols.items():
            if key not in self.cols.keys():
                raise ValueError(f"Column {key} not in {self.cols.keys()}.")
            self.cols[key] = torch.cat((self.cols[key], col))
        self.length += dataset.length
    
    def copy(self) -> Self:
        """
        Parameters
        ----------
        None

        Returns
        -------
        PhySysDataset
        """
        ds_copy = PhySysDataset(self.cols.copy())
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy
    
    def deep_copy(self) -> Self:
        """
        Parameters
        ----------
        None

        Returns
        -------
        PhySysDataset
        """
        new_cols = {key: col.clone() for key, col in self.cols.items()}
        ds_copy = PhySysDataset(new_cols)
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy
    
    def save(self, dst_file: str) -> None:
        """
        Parameters
        ----------
        dst_file : str
            Filepath where to save the dataset.

        Returns
        -------
        None
        """
        d = {
            "cols": self.cols,
            "subkeys": self.subkeys
        }
        torch.save(d, dst_file)
    
    @staticmethod
    def load(src_file: str) -> Self:
        """
        Parameters
        ----------
        src_file : str
            Filepath of the dataset to load.

        Returns
        -------
        """
        d = torch.load(src_file, weights_only=False)
        dataset = PhySysDataset(cols=d["cols"])
        for key in d["subkeys"]:
            dataset.set_subkeys(key=key, subkeys=d["subkeys"][key])
        return dataset