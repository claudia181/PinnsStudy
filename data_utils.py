"""
data_utils.py
===========

This module provides some functions for datasets manipulation.
"""

import torch
from torch.utils.data import TensorDataset, ConcatDataset
import os
from generate import X, U, DU, D2U, RESIDUAL_KEYS, RESIDUAL_VALUES, PDE_VALUES, IC_VALUES, TIMES
from model import PdeNet
from load_store_utils import resume_model
from pde_utils import Pde, key_idx, key_str, ic_key_idx
from typing import Tuple, List

DIM_LABEL_DICT = {"x": 0, "y": 1, "r": 2}


def get_dictionary(keys: torch.Tensor, values: torch.Tensor, pde_name: str) -> dict:
    """
    Construct a dictionary from keys indexes and values for a given PDE identified by pde_name.

    Parameters
    ----------
    keys : torch.Tensor
        Indexes corresponding to keys.
    values : torch.Tensor
        Values.
    pde_name : str
        PDE string identifier.

    Returns
    -------
    dict
        The derived dictionary.
    """
    dictionary = {}
    keys = keys[0].tolist()
    values = [values[:, i] for i in range(values.shape[1])]
    for key, value in zip([key_str(k, pde_name) for k in keys], [v for v in values]):
        if len(value) == 1: value = value.item()
        dictionary[key] = value
    return dictionary

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
            x = columns[0]
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
        spatial_ranges: dict = {},
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

def include_time_in_input(dataset: TensorDataset) -> TensorDataset:
    """
    Includes the temporal coordinates in the input column.

    Parameters
    ----------
    dataset : TensorDataset

    Returns
    -------
    TensorDataset
        The modified TensorDataset.
    """
    if dataset is None:
        return None
    cols = list(dataset.tensors)
    cols[0] = torch.cat([cols[X], cols[TIMES].unsqueeze(1)], dim=1)
    return TensorDataset(*cols)

def exclude_space_in_input(dataset: TensorDataset) -> TensorDataset:
    """
    Excludes the spatial coordinates from the input column.

    Parameters
    ----------
    dataset : TensorDataset

    Returns
    -------
    TensorDataset
        The modified TensorDataset.
    """
    if dataset is None:
        return None
    cols = list(dataset.tensors)
    cols[0] = cols[0][:, 2].reshape(-1, 1)
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
        boundary = {"x": center[0], "y": center[1], "r": [radius-cell_size, radius]}
        new_cols = filter_tensors(columns=cols, spatial_ranges=boundary, mode="closed", shape=shape)
        center = torch.tensor([center[0], center[1]]).repeat(len(new_cols[0]), 1)
        out_vect = new_cols[0] - center
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
        ranges = {"x": center[0], "y": center[1], "r": [-1.0, radius-cell_size]}
    else:
        raise ValueError(f"Unrecognized {shape} boundary shape.")
    cols = filter_tensors(columns=cols, spatial_ranges=ranges, mode="open", shape=shape)
    return TensorDataset(*cols)


def compute_prediction_difference(
        model: str|PdeNet,
        dataset: str|TensorDataset,
        signed: bool = True,
        device="cpu"
    ) -> TensorDataset:
    """
    Computes the differences btw the model predictions on dataset inputs and the dataset labels (u, du, d2u).

    Parameters
    ----------
    model : str|PdeNet
    dataset : str|TensorDataset
    signed : bool
    device : str

    Returns
    -------
    TensorDataset
        The TensorDataset containing the differences and the residual for each input.
    """
    if type(model) is str:
        model = resume_model(model_path=model)

    if type(dataset) is str:
        if not os.path.exists(dataset):
            raise ValueError(f"'File {dataset}' not found.")
        dataset = torch.load(os.path.join(dataset), weights_only=False)

    if model.time_in_input:
        dataset = include_time_in_input(dataset)
    if not model.space_in_input:
        dataset = exclude_space_in_input(dataset)
    tensors = dataset.tensors
    tensors = dataset.tensors
    x = tensors[X].to(device).float()
    model.eval()
    with torch.no_grad():
        params_values_in_input = None
        if model.pde_params_in_input is not None:
            param_values = tensors[PDE_VALUES].to(device).float()
            pde_params_in_input_indexes = [key_idx(key, model.pde) for key in model.pde_params_in_input]
            params_values_in_input = param_values[:, pde_params_in_input_indexes]

        if model.ic_params_in_input is not None:
            ic_values = tensors[IC_VALUES].to(device).float()
            ic_params_in_input_indexes = [ic_key_idx(key, model.pde) for key in model.ic_params_in_input]
            ic_values_in_input = ic_values[:, ic_params_in_input_indexes]
            if params_values_in_input is not None:
                params_values_in_input = torch.cat([params_values_in_input, ic_values_in_input], dim=-1)
            else:
                params_values_in_input = ic_values_in_input

        u = model.forward(x, params_values_in_input)
        du = model.derivative(order=1, x=x, pde_params=params_values_in_input)
        d2u = model.derivative(order=2, x=x, pde_params=params_values_in_input)
        
        u_diff = u - tensors[U].to(device).float()
        du_diff = du - tensors[DU].to(device).float()
        d2u_diff = d2u - tensors[D2U].to(device).float()

        if not signed:
            u_diff = torch.abs(u_diff)
            du_diff = torch.abs(du_diff)
            d2u_diff = torch.abs(d2u_diff)

        residual_info_keys = tensors[RESIDUAL_KEYS].to(device).int()
        residual_info_values = tensors[RESIDUAL_VALUES].to(device).float()
        residual_data = get_dictionary(residual_info_keys, residual_info_values, model.pde)
        res = Pde.residual(pde_name=model.pde, u=u, du=du, d2u=d2u, **residual_data)
        tensors = [x, u_diff, du_diff[:, 0], du_diff[:, 1], d2u_diff[:, 0, 0], d2u_diff[:, 0, 1], d2u_diff[:, 1, 0], d2u_diff[:, 1, 1], res]

    return TensorDataset(*tensors)

def compute_prediction(model: str|PdeNet, dataset: str|TensorDataset, device="cpu") -> TensorDataset:
    """
    Computes the predictions of model on dataset inputs (u, du, d2u, residual).

    Parameters
    ----------
    model : str|PdeNet
    dataset : str|TensorDataset
    device : str

    Returns
    -------
    TensorDataset
        The TensorDataset containing the model predictions for each input ([x, u, du, d2u, res]).
    """
    if type(model) is str:
        model = resume_model(model_path=model)

    if type(dataset) is str:
        if not os.path.exists(dataset):
            raise ValueError(f"'File {dataset}' not found.")
        dataset = torch.load(os.path.join(dataset), weights_only=False)

    if model.time_in_input:
        dataset = include_time_in_input(dataset)
    if not model.space_in_input:
        dataset = exclude_space_in_input(dataset)
    tensors = dataset.tensors
    x = tensors[X].to(device).float()
    model.eval()
    with torch.no_grad():
        params_values_in_input = None
        if model.pde_params_in_input is not None and model.pde_params_in_input != []:
            param_values = tensors[PDE_VALUES].to(device).float()
            pde_params_in_input_indexes = [key_idx(key, model.pde) for key in model.pde_params_in_input]
            params_values_in_input = param_values[:, pde_params_in_input_indexes]

        if model.ic_params_in_input is not None and model.ic_params_in_input != []:
            ic_values = tensors[IC_VALUES].to(device).float()
            ic_params_in_input_indexes = [ic_key_idx(key, model.pde) for key in model.ic_params_in_input]
            ic_values_in_input = ic_values[:, ic_params_in_input_indexes]
            if params_values_in_input is not None:
                params_values_in_input = torch.cat([params_values_in_input, ic_values_in_input], dim=-1)
            else:
                params_values_in_input = ic_values_in_input

        u = model.forward(x, params_values_in_input)
        du = model.derivative(order=1, x=x, pde_params=params_values_in_input)
        d2u = model.derivative(order=2, x=x, pde_params=params_values_in_input)
        
        residual_info_keys = tensors[RESIDUAL_KEYS].to(device).int()
        residual_info_values = tensors[RESIDUAL_VALUES].to(device).float()
        residual_data = get_dictionary(residual_info_keys, residual_info_values, model.pde)
        res = Pde.residual(pde_name=model.pde, u=u, du=du, d2u=d2u, **residual_data)
        if model.input_units >= 2 and model.pde != "Pendulum": #TODO: add 3 in
            tensors = [x, u, du[:, 0], du[:, 1], d2u[:, 0, 0], d2u[:, 0, 1], d2u[:, 1, 0], d2u[:, 1, 1], res]
        else:
            tensors = [x, u, du, d2u, res]

    return TensorDataset(*tensors)

def extract_targets(dataset: str|TensorDataset, device="cpu") -> TensorDataset:
    """
    Extract targets/labels from dataset (x, u, du, d2u, residual).

    Parameters
    ----------
    dataset : str|TensorDataset
    device : str

    Returns
    -------
    TensorDataset
        The TensorDataset containing the model predictions for each input ([x, u, du, d2u, res]).
    """
    if type(dataset) is str:
        if not os.path.exists(dataset):
            raise ValueError(f"'File {dataset}' not found.")
        dataset = torch.load(os.path.join(dataset), weights_only=False)

    tensors = dataset.tensors
    x = tensors[X].to(device).float()
    u = tensors[U].to(device).float()
    du = tensors[DU].to(device).float()
    d2u = tensors[D2U].to(device).float()
    res = torch.zeros_like(u)
    tensors = [x, u, du[:, 0], du[:, 1], d2u[:, 0, 0], d2u[:, 0, 1], d2u[:, 1, 0], d2u[:, 1, 1], res]

    return TensorDataset(*tensors)

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