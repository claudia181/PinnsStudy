import torch
from torch.utils.data import Dataset
from typing import List, Callable, Any, Tuple

class PhySysDataset(Dataset):
    def __init__(self, cols: List[Tuple[str, list|torch.Tensor]]) -> None:
        self.cols = {}
        self.length = len(cols[0][1])
        for col in cols:
            key, items = col[0], col[1]
            if len(items) != self.length:
                raise ValueError(f"Column {key} has a different length (len({key} = {len(items)}, while it is expected to be {self.length}).")
        for col in cols:
            key, items = col[0], col[1]
            self.cols[key] = items
        self.subkeys = {}

    def __len__(self) -> int:
        return len(self.length)

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
    
    def save(self, dst_file: str) -> None:
        d = {
            "cols": self.cols,
            "subkeys": self.subkeys
        }
        torch.save(d, dst_file)
    
    @staticmethod
    def load(src_file: str) -> Dataset:
        d = torch.load(src_file, weights_only=False)
        dataset = PhySysDataset(cols=d["cols"])
        for key in d["subkeys"]:
            dataset.set_subkeys(key=key, subkeys=d["subkeys"][key])
        return dataset