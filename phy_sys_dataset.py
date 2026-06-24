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