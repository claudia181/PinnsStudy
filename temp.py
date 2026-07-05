"""
data_utils.py
===========

This module provides some functions for datasets manipulation.
"""

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from typing import Tuple, List, Iterator
from itertools import cycle
import random
from phy_sys_dataset import PhySysDataset
import copy

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
        if subset != {}:
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

def extract_Dataset(
        dataset: ConcatDataset|PhySysDataset, 
        time_indexes: list = None, 
        spatial_ranges: dict|List[dict] = {},
        shape: str = "rectangle"
    ) -> PhySysDataset:
    """
    Construct a PhySysDataset of filtered columns according to spatial_ranges.

    Parameters
    ----------
    dataset : ConcatDataset|PhySysDataset
    time_indexes : list
        The time instants where to filter.
    spatial_ranges : dict
        The spatial subset to consider.
    shape : str
        "rectangle" | "circle".

    Returns
    -------
    PhySysDataset
        The filtered PhySysDataset.
    """
    if dataset is None:
        return None
    if type(dataset) is PhySysDataset: 
        datasets = [dataset]
    else:
        datasets = dataset.datasets
    if time_indexes is None:
        time_indexes = [i for i in range(len(datasets))]
    ds_list = [datasets[i] for i in time_indexes]
    cols = []
    for ds in ds_list:
        tensors = ds.columns()
        if cols == []:
            cols += tensors
        else:
            cols = [torch.cat((c1, c2)) for c1, c2 in zip(cols, tensors)]
    cols = filter_tensors(columns=cols, spatial_ranges=spatial_ranges, mode="closed", shape=shape)
    new_ds = PhySysDataset([(s, c) for s, c in zip(datasets[0].cols.keys(), cols)])
    for key in datasets[0].subkeys.keys():
        new_ds.set_subkeys(key, datasets[0].subkeys[key])
    return new_ds

def extract_boundary(
    dataset: PhySysDataset, 
    shape: str = "rectangle",
    cell_size: float = 0.0,
    center: list = [0.0, 0.0],
    radius: float = 1.0
) -> PhySysDataset:
    """
    Extract the boundary points from the dataset for a given time instant.

    Parameters
    ----------
    dataset : ConcatDataset|PhySysDataset
    shape : str
        "rectangle" | "circle".
    cell_size : float
    t : int
        Time index.

    Returns
    -------
    PhySysDataset
        The PhySysDataset containing the boundary points at t.
    """
    cols = dataset.columns()
    keys = list(dataset.cols.keys())
    subkeys = dataset.subkeys.copy()
    keys.append("outward_normal")
    subkeys["outward_normal"] = ["x", "y"]

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
    
    boundary_ds = PhySysDataset([(key, col) for key, col in zip(keys, bd_cols)])
    for k in subkeys.keys():
        boundary_ds.set_subkeys(k, subkeys[k])
    return boundary_ds


def extract_interior(
        dataset: PhySysDataset, 
        shape: str = "rectangle",
        cell_size: float = 0.0,
        center: list = [0.0, 0.0],
        radius: float = 1.0
    ) -> PhySysDataset:
    """
    Extract the interior points from the dataset for a given time instant.

    Parameters
    ----------
    dataset : ConcatDataset|PhySysDataset
    t : int
        Time index.
    shape : str
        "rectangle" | "circle".
    cell_size : float

    Returns
    -------
    PhySysDataset
        The PhySysDataset containing the interior points at t.
    """
    cols = dataset.columns()
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

    interior_ds = PhySysDataset([(key, col) for key, col in zip(dataset.cols.keys(), cols)])
    for k in dataset.subkeys.keys():
        interior_ds.set_subkeys(k, dataset.subkeys[k])

    return interior_ds