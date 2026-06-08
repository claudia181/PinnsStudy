"""
data_utils.py
===========

This module provides some functions for datasets manipulation.
"""

import torch
from torch.utils.data import TensorDataset, ConcatDataset
from generate import X, U, DU, D2U
from typing import Tuple, List

DIM_LABEL_DICT = {"x": 0, "y": 1, "r": 2}


def get_key(idx: int) -> str:
    """
    DIM_LABEL_DICT key corresponding to the index idx.

    Parameters
    ----------
    idx : int

    Returns
    -------
    str
    """
    for key in DIM_LABEL_DICT.keys():
        if DIM_LABEL_DICT[key] == idx: return key
    raise ValueError(f"Index {idx} not a valid dimention index.")

def sort_keys(keys: list) -> list:
    """
    Sort the list of keys keys according to the DIM_LABEL_DICT ordering.

    Parameters
    ----------
    keys : list

    Returns
    -------
    list
    """
    sorted_key_indexes = sorted([DIM_LABEL_DICT[key] for key in keys if key in DIM_LABEL_DICT.keys()])
    sorted_keys = [get_key(idx) for idx in sorted_key_indexes]
    return sorted_keys

def filter_tensors(
        columns: list[torch.Tensor], 
        spatial_ranges: dict|List[dict], 
        mode: str, 
        shape: str = "rectangle",
    ) -> list[torch.Tensor]:
    """
    Filter columns keeping elements within spatial_ranges.

    Parameters
    ----------
    columns : list[torch.Tensor]
    spatial_ranges : dict
    mode : str
        Closed or open.
    shape : str
        "rectangle"|"circle", default = "rectangle";

        if "rectangle", each key in spatial_ranges is a model a side;

        if "circle", entry of key "r" is the radius and the other keys encode the center coordinates.

    Returns
    -------
    list[torch.Tensor]
        The list of filtered columns.
    """
    if columns == []: return []
    if type(spatial_ranges) is dict:
        spatial_ranges = [spatial_ranges]
    masks = []
    for subset in spatial_ranges:
        mask = torch.ones(len(columns[0]), dtype=bool)
        sorted_keys = sort_keys(subset.keys())
        if shape == "rectangle":
            subset = [subset[key] for key in sorted_keys]
            for i in range(len(subset)):
                xmin = subset[i][0]
                xmax = subset[i][1]
                x = columns[0][:, i]
                if mode == "closed":
                    mask = mask & (x >= xmin) & (x <= xmax)
                elif mode == "open":
                    mask = mask & (x > xmin) & (x < xmax)
                else:
                    raise ValueError(f"Unrecognized mode {mode}.")
            
        elif shape == "circle":
            center_coords = [subset[key] for key in sorted_keys if key != "r"]
            rmin = subset["r"][0]
            rmax = subset["r"][1]
            center = torch.tensor(center_coords)
            x = columns[0][:, :2]
            if mode == "closed":
                mask = mask & (torch.linalg.norm(x - center, axis=1) >= rmin) & (torch.linalg.norm(x - center, axis=1) <= rmax)
            elif mode == "open":
                mask = mask & (torch.linalg.norm(x - center, axis=1) > rmin) & (torch.linalg.norm(x - center, axis=1) < rmax)
            else:
                raise ValueError(f"Unrecognized mode {mode}.")
        else:
            raise ValueError(f"Unrecognized shape {shape}.")

        masks.append(mask)  
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return [c[mask] for c in columns]

def extract_TensorDataset(
        dataset: ConcatDataset|TensorDataset, 
        time_indexes: list = None, 
        spatial_ranges: dict|List[dict] = {},
        shape: str = "rectangle"
    ) -> TensorDataset:
    """
    Construct a TensorDataset of filtered columns according to spatial_ranges.

    Parameters
    ----------
    dataset : ConcatDataset|TensorDataset
    time_indexes : list
        The time instants where to filter.
    spatial_ranges : dict
        The spatial subset to consider.
    shape : str
        "rectangle" | "circle".

    Returns
    -------
    TensorDataset
        The filtered TensorDataset.
    """
    if dataset is None:
        return None
    if type(dataset) is TensorDataset: 
        datasets = [dataset]
    else:
        datasets = dataset.datasets
    if time_indexes is None:
        time_indexes = [i for i in range(len(datasets))]
    tds = [datasets[i] for i in time_indexes]
    cols = []
    for td in tds:
        tensors = td.tensors
        if cols == []:
            cols += tensors
        else:
            cols = [torch.cat((c1, c2)) for c1, c2 in zip(cols, tensors)]
    cols = filter_tensors(columns=cols, spatial_ranges=spatial_ranges, mode="closed", shape=shape)
    return TensorDataset(*cols)

def replace_labels(dataset: TensorDataset, labels: torch.Tensor) -> TensorDataset:
    """
    Replace the labels column of the TensorDataset.

    Parameters
    ----------
    dataset : TensorDataset
    labels : torch.Tensor
        The new labels for the replacement.

    Returns
    -------
    TensorDataset
        The modified TensorDataset.
    """
    if dataset is None:
        return None
    cols = list(dataset.tensors)
    cols[U] = labels
    return TensorDataset(*cols)

def extract_boundary(
    dataset: ConcatDataset|TensorDataset, 
    shape: str = "rectangle",
    cell_size: float = 0.0,
    center: list = [0.0, 0.0],
    radius: float = 1.0,
    t: int = 0
) -> TensorDataset:
    """
    Extract the boundary points from the dataset for a given time instant.

    Parameters
    ----------
    dataset : ConcatDataset|TensorDataset
    shape : str
        "rectangle" | "circle".
    cell_size : float
    t : int
        Time index.

    Returns
    -------
    TensorDataset
        The TensorDataset containing the boundary points at t.
    """
    if dataset is None:
        return None
    if type(dataset) is ConcatDataset:
        cols = dataset.datasets[t].tensors
    else: # type(dataset) is TensorDataset:
        cols = dataset.tensors

    xmin = cols[0][:, 0].min()
    xmax = cols[0][:, 0].max()
    ymin = cols[0][:, 1].min()
    ymax = cols[0][:, 1].max()
    
    bd_cols = []
    if shape == "rectangle":
        boundary = [
            {"x": [xmin, xmin], "y": [ymin, ymax]},
            {"x": [xmax, xmax], "y": [ymin, ymax]},
            {"x": [xmin, xmax], "y": [ymin, ymin]},
            {"x": [xmin, xmax], "y": [ymax, ymax]}
        ]
        outward_normal_vectors = [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]
        for r, n in zip(boundary, outward_normal_vectors):
            new_cols = filter_tensors(columns=cols, spatial_ranges=r, mode="closed", shape=shape)
            n_tensor = torch.tensor(n)
            n_col = n_tensor.repeat(len(new_cols[0]), 1)
            new_cols.append(n_col)
            if bd_cols == []:
                bd_cols += new_cols
            else:
                bd_cols = [torch.cat((c1, c2)) for c1, c2 in zip(bd_cols, new_cols)]
    elif shape == "circle": # shape = "circle"
        boundary = {"x": center[0], "y": center[1], "r": [radius-0.5*cell_size, radius+0.5*cell_size]}#[radius-cell_size, radius]}
        new_cols = filter_tensors(columns=cols, spatial_ranges=boundary, mode="closed", shape=shape)
        center = torch.tensor([center[0], center[1]]).repeat(len(new_cols[0]), 1)
        out_vect = new_cols[0][:, :2] - center
        outward_normal_vectors = out_vect / torch.linalg.norm(out_vect, dim=1, keepdim=True)
        new_cols.append(outward_normal_vectors)
        bd_cols += new_cols

    else:
        raise ValueError(f"Unrecognized {shape} boundary shape.")
    
    return TensorDataset(*bd_cols)

def extract_interior(
        dataset: ConcatDataset|TensorDataset, 
        t: int = 0, 
        shape: str = "rectangle",
        cell_size: float = 0.0,
        center: list = [0.0, 0.0],
        radius: float = 1.0
    ) -> TensorDataset:
    """
    Extract the interior points from the dataset for a given time instant.

    Parameters
    ----------
    dataset : ConcatDataset|TensorDataset
    t : int
        Time index.
    shape : str
        "rectangle" | "circle".
    cell_size : float

    Returns
    -------
    TensorDataset
        The TensorDataset containing the interior points at t.
    """
    if dataset is None:
        return None
    if type(dataset) is TensorDataset:
        td = dataset
    else:
        td = dataset.datasets[t]
    cols = td.tensors
    xmin = cols[0][:, 0].min()
    xmax = cols[0][:, 0].max()
    ymin = cols[0][:, 1].min()
    ymax = cols[0][:, 1].max()

    if shape == "rectangle":
        ranges = {"x": [xmin, xmax], "y": [ymin, ymax]}
    elif shape == "circle":
        ranges = {"x": center[0], "y": center[1], "r": [-1.0, radius-0.5*cell_size]}
    else:
        raise ValueError(f"Unrecognized {shape} boundary shape.")
    cols = filter_tensors(columns=cols, spatial_ranges=ranges, mode="open", shape=shape)
    return TensorDataset(*cols)


def prepare_dataset(datasets: list[TensorDataset], samples_per_dataset: int, seed: int = 42) -> ConcatDataset:
    """
    Randomly permute and subsample datasets (seed for reproducibility), and then combine the result in a ConcatDataset.

    Parameters
    ----------
    datasets : list[TensorDataset]
    samples_per_dataset : int
    seed : int

    Returns
    -------
    ConcatDataset
    """
    seeds = [seed+i for i in range(len(datasets))] 
    cols = None   
    for ds, seed in zip(datasets, seeds):
        torch.manual_seed(seed)
        indices = torch.randperm(len(ds))[:samples_per_dataset]
        new_cols = [col[indices] for col in ds.tensors]
        if cols is None:
            cols = new_cols
        else:
            for i, col in enumerate(new_cols):
                cols[i] = torch.cat([cols[i], col])
    return ConcatDataset([TensorDataset(*cols)])

def subsample(
    dataset: TensorDataset,
    n_samples: int,
    seed: int = 42
    ) -> TensorDataset:
    
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))
    indices = indices[:n_samples]
    reduced_cols = [col[indices] for col in dataset.tensors]
    reduced_dataset = TensorDataset(*reduced_cols)
    return reduced_dataset

def subsample_normal(dataset: TensorDataset, mean: torch.Tensor, stddev: float, n_samples: int) -> TensorDataset:
    d = len(mean)
    X = dataset.tensors[0][:, :d]

    # compute the Gaussian bump weights
    # w = exp(-||x - center||^2 / (2 * stddev^2))
    weights = torch.exp(-torch.sum((X - mean) ** 2, dim=1) / (2 * (stddev ** 2)))

    # sample indices based on the weights
    sampled_indices = torch.multinomial(weights, n_samples, replacement=False)

    return TensorDataset(*dataset[sampled_indices])


def get_grid(xmin_list: List[float], xmax_list: List[float], dx_list: List[float]) -> torch.Tensor:
    x_list = []
    for xmin, xmax, dx in zip(xmin_list, xmax_list, dx_list):
        if xmin == xmax:
            x_list.append([xmin])
        else:
            # x = [xmin+dx, xmin+2dx, ..., xmin+Ndx <= xmax]
            x_list.append(torch.arange(xmin, xmax+0.5*dx, dx))
    cols = torch.meshgrid(*x_list)
    cols = [col.reshape(-1, 1) for col in cols]
    # points = [(x1, y1, ...), ..., (x[N^2], y[N^2], ...)], shape (N^2, 2)
    points = torch.column_stack(cols)
    return points

def get_circle(radius: float, dx_list: List[float]) -> torch.Tensor:
    x_list = []
    xmin = - radius
    xmax = radius
    for dx in dx_list:
        # x = [xmin+dx, xmin+2dx, ..., xmin+Ndx <= xmax]
        x_list.append(torch.arange(xmin, xmax+0.5*dx, dx))

    cols = torch.meshgrid(*x_list)
    mask = cols[0]**2 + cols[1]**2 <= radius**2
    cols = [axis[mask] for axis in cols]
    # grid_coords[0] = x_pts = [x1, ..., xN, x1, ..., xN, ..., x1, ..., xN], shape (N^2, 1)
    # grid_coords[1] = y_pts = [y1, ..., yN, y1, ..., yN, ..., y1, ..., yN], shape (N^2, 1)
    # ...
    cols = [arr.reshape(-1, 1) for arr in cols]
    # points = [(x1, y1, ...), ..., (x[N^2], y[N^2], ...)], shape (N^2, 2)
    points = torch.column_stack(cols)
    return points

def get_normal(n_samples: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.normal(mean=mean.repeat(n_samples, 1), std=std.repeat(n_samples, 1))

def add_column(dataset: TensorDataset|ConcatDataset, column: torch.Tensor|List[torch.Tensor]) -> TensorDataset|ConcatDataset:
    if type(dataset) is TensorDataset and type(column) is torch.Tensor:
        return TensorDataset(*dataset.tensors, column)
    elif type(dataset) is ConcatDataset and type(column) is list:
        if len(dataset.datasets) != len(column):
            raise ValueError(f"Size mismatch: dataset.datasets has length {len(dataset.datasets)}, while column has length {len(column)}.")
        new_dataset = []
        for tds, col in zip(dataset.datasets, column):
            new_dataset.append(TensorDataset(*tds.tensors, col))
        return ConcatDataset(new_dataset)