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

Methods:
    - weight, containing the weight for the loss term of the task in the multi-objective loss;
    - loss_value, optionally filled with the last loss value obtained for the task;
    - grad_norm, optionally filled with the last gradient norm of the task loss term;
    - grad, optionally filled with the last gradient of the task loss term;
    - conflict, optionally filled with the last cosine similarity between the gradient of the 
      task loss term and a reference gradient vector;
    - id, an identifier string for the task;
    - device.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Self

class PhySysDataset(Dataset):
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
    
    def columns(self) -> list:
        return [self.cols[key] for key in self.cols.keys()]
    
    def set_subkeys(self, key: str, subkeys: List[str]) -> None:
        if key not in self.cols.keys():
            raise ValueError(f"Column of key {key} not in dataset.")
        if len(subkeys) != len(self.cols[key][0]):
            raise ValueError(f"Items of column {key} have {len(self.cols[key][0])} elements each, but {len(subkeys)} subkeys are passed.")
        self.subkeys[key] = subkeys
    
    def get_column(self, key: str, condition: Callable[[torch.Tensor], bool] = None) -> torch.Tensor:
        if key not in self.cols.keys():
            raise ValueError(f"Column of key {key} not in dataset.")
        col = self.cols[key]
        if condition is None:
            return col
        elif type(col) is torch.Tensor:
            return torch.stack([item for item in col if condition(item)])
        else:
            return [item for item in col if condition(item)]
    
    def subsample(self, indices: torch.Tensor) -> Self:
        subsampled_cols = {}
        for key, col in self.cols.items():
            subsampled_cols[key] = col[indices]
        subsampled_ds = PhySysDataset(subsampled_cols)
        subsampled_ds.subkeys = self.subkeys.copy()
        return subsampled_ds
    
    def merge(self, dataset: Self):
        if len(dataset.cols.keys()) != len(self.cols.keys()):
            raise ValueError(f"Different columns to merge: {self.cols.keys()} != {dataset.cols.keys()}.")
        for key, col in dataset.cols.items():
            if key not in self.cols.keys():
                raise ValueError(f"Column {key} not in {self.cols.keys()}.")
            self.cols[key] = torch.cat((self.cols[key], col))
        self.length += dataset.length
    
    def copy(self) -> Self:
        ds_copy = PhySysDataset(self.cols.copy())
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy
    
    def deep_copy(self) -> Self:
        new_cols = {key: col.clone() for key, col in self.cols.items()}
        ds_copy = PhySysDataset(new_cols)
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy

    def add_column(self, key: str, col: torch.Tensor, subkeys: List[str] = []) -> None:
        if len(col) != self.length:
            raise ValueError(f"Column of length {len(col)}, but expected of length {self.length}.")
        if key in self.cols.keys():
            raise ValueError(f"Column {key} already present in the dataset.")
        self.cols[key] = col
        if subkeys != []:
            self.subkeys[key] = subkeys
    
    def save(self, dst_file: str) -> None:
        d = {
            "cols": self.cols,
            "subkeys": self.subkeys
        }
        torch.save(d, dst_file)
    
    @staticmethod
    def load(src_file: str) -> Self:
        d = torch.load(src_file, weights_only=False)
        dataset = PhySysDataset(cols=d["cols"])
        for key in d["subkeys"]:
            dataset.set_subkeys(key=key, subkeys=d["subkeys"][key])
        return dataset