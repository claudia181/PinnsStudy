
"""
data_utils.py
===========

This module provides some functions for datasets manipulation.
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List, Iterator
from itertools import cycle
import random
from phy_sys_dataset import PhySysDataset
import copy

def filter_points(
        dataset: PhySysDataset, 
        ranges: dict|List[dict], 
        mode: str,
        shape: str = "rectangle",
        eps: float = 1e-6
    ) -> PhySysDataset:
    """
    Filter columns keeping elements within ranges and return the relative dataset.

    Parameters
    ----------
    columns : PhySysDataset
    ranges : dict|List[dict]
    mode : str
        Closed or open.
    shape : str
        "rectangle"|"circle", default = "rectangle";

        if "rectangle", each key in spatial_ranges is a model a side;

        if "circle", entry of key "r" is the radius and the other keys encode the center coordinates.

    Returns
    -------
    PhySysDataset
        The filtered dataset.
    """
    if type(ranges) is dict:
        ranges = [ranges]
    masks = []
    for subset in ranges:
        mask = torch.ones(dataset.length, dtype=bool)
        if shape == "rectangle":
            for key in subset.keys():
                xmin = subset[key][0] - eps
                xmax = subset[key][1] + eps
                x = dataset.cols["spacetime"][:, dataset.index(key="spacetime", subkey=key)]
                if mode == "closed":
                    mask = mask & (x >= xmin) & (x <= xmax)
                elif mode == "open":
                    mask = mask & (x > xmin) & (x < xmax)
                else:
                    raise ValueError(f"Unrecognized mode {mode}.")
        elif shape == "circle":
            center_coords = [subset[key] for key in subset.keys() if key != "r"]
            coords_indexes = [dataset.index(key="spacetime", subkey=key) for key in subset.keys() if key != "r"]
            rmin = subset["r"][0] - eps
            rmax = subset["r"][1] + eps
            x = dataset.cols["spacetime"][:, coords_indexes]
            center = torch.tensor(
                center_coords,
                dtype=x.dtype,
                device=x.device
            )
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
    
    cols = {}
    for key in dataset.cols.keys():
        cols[key] = dataset.cols[key][mask]

    filtered_dataset = PhySysDataset(cols=cols)
    filtered_dataset.subkeys = dataset.subkeys
    return filtered_dataset

def merge(datasets: List[PhySysDataset]) -> PhySysDataset:
    merged_ds = datasets[0].copy()
    for ds in datasets[1:]:
        merged_ds.merge(ds)
    return merged_ds

def replace_labels(dataset: PhySysDataset, labels: torch.Tensor, key: str) -> None:
    """
    Replace the labels column of the PhySysDataset.

    Parameters
    ----------
    dataset : PhySysDataset
    labels : torch.Tensor
        The new labels for the replacement.

    Returns
    -------
    PhySysDataset
        The modified PhySysDataset.
    """
    if key not in dataset.cols.keys():
        raise ValueError(f"Column {key} not in dataset.")
    if len(labels) != dataset.length:
        raise ValueError(f"Wrong length of label tensor ({len(labels)} instead of {dataset.length}).")
    dataset.cols[key] = labels

def get_boundary(
    dataset: PhySysDataset, 
    shape: str = "rectangle",
    cell_size: float = 0.0,
    center: list = [0.0, 0.0],
    radius: float = 1.0,
    insert_out_normal: bool = True,
    eps: float = 1e-6
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
    spatial_keys = [key for key in dataset.subkeys["spacetime"] if key != "t"]
    if len(spatial_keys) == 0:
        raise ValueError(f"0-dimentional spatial domain.")
    if shape == "rectangle":
        ranges = {}
        for key in spatial_keys:
            x = dataset.cols["spacetime"][:, dataset.index("spacetime", key)]
            ranges[key] = [x.min(), x.max()]
        
        boundary = []
        outward_normal_vectors = []
        for key in spatial_keys:
            xmin = ranges[key][0]
            side = copy.deepcopy(ranges)
            side[key] = [xmin, xmin]
            boundary.append(side)
            if insert_out_normal:
                if len(spatial_keys) == 1:
                    outward_normal_vectors.append(-1.)
                else:
                    outward_normal_vector = [0. for _ in spatial_keys]
                    outward_normal_vector[dataset.index("spacetime", key)] = -1.
                    outward_normal_vectors.append(outward_normal_vector)

            xmax = ranges[key][1]
            side = copy.deepcopy(ranges)
            side[key] = [xmax, xmax]
            boundary.append(side)
            if insert_out_normal:
                if len(spatial_keys) == 1:
                    outward_normal_vectors.append(1.)
                else:
                    outward_normal_vector = [0. for _ in spatial_keys]
                    outward_normal_vector[dataset.index("spacetime", key)] = 1.
                    outward_normal_vectors.append(outward_normal_vector)

            # boundary = [
            #     {"x": [xmin, xmin], "y": [ymin, ymax], "z": [zmin, zmax]},
            #     {"x": [xmax, xmax], "y": [ymin, ymax], "z": [zmin, zmax]},
            #     {"x": [xmin, xmax], "y": [ymin, ymin], "z": [zmin, zmax]},
            #     {"x": [xmin, xmax], "y": [ymax, ymax], "z": [zmin, zmax]}
            # ]

            # outward_normal_vectors = [-1., 1.]
            # outward_normal_vectors = [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]
            # outward_normal_vectors = [[-1., 0., 0.], [1., 0., 0.], [0., -1., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]]

        filtered_ds = filter_points(dataset=dataset, ranges=boundary[0], mode="closed", shape=shape, eps=eps)
        if insert_out_normal:
            n_col = torch.tensor(outward_normal_vectors[0]).repeat(filtered_ds.length, 1)

        for side, n in zip(boundary[1:], outward_normal_vectors):
            side_ds = filter_points(dataset=dataset, ranges=side, mode="closed", shape=shape, eps=eps)
            filtered_ds.merge(filter_points(dataset=dataset, ranges=side, mode="closed", shape=shape, eps=eps))
            if insert_out_normal:
                n_col = torch.cat((n_col, torch.tensor(n).repeat(side_ds.length, 1)))

        if insert_out_normal:
            filtered_ds.add_column(key="n", col=n_col, subkeys=spatial_keys)

    elif shape == "circle":
        for key in spatial_keys:
            boundary[key] = center[dataset.index("spacetime", key)]
        boundary["r"] = [radius-0.5*cell_size, radius+0.5*cell_size] # [radius-cell_size, radius]
        # boundary = {
        #   "x": center[ix], 
        #   "y": center[iy], 
        #   "z": center[iz], 
        #   "r": [radius-0.5*cell_size, radius+0.5*cell_size]
        # }
        filtered_ds = filter_points(dataset=dataset, ranges=boundary, mode="closed", shape=shape, eps=eps)
        if insert_out_normal:
            center = torch.tensor(center).repeat(filtered_ds.length, 1)
            spatial_indexes = [dataset.index("spacetime", key) for key in spatial_keys]
            out_vect = filtered_ds.cols["spacetime"][:, spatial_indexes] - center
            outward_normal_vectors = out_vect / torch.linalg.norm(out_vect, dim=1, keepdim=True)
            filtered_ds.add_column("n", outward_normal_vectors, spatial_keys, spatial_keys)
    else:
        raise ValueError(f"Unrecognized {shape} boundary shape.")
    
    return filtered_ds

def get_interior(
        dataset: PhySysDataset, 
        shape: str = "rectangle",
        cell_size: float = 0.0,
        center: list = [0.0, 0.0],
        radius: float = 1.0,
        eps: float = 1e-6
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
    spatial_keys = [key for key in dataset.subkeys["spacetime"] if key != "t"]
    ranges = {}
    if shape == "rectangle":
        for key in spatial_keys:
            x = dataset.cols["spacetime"][:, dataset.index("spacetime", key)]
            ranges[key] = [x.min(), x.max()]
        # ranges = {"x": [xmin, xmax], "y": [ymin, ymax]}
    elif shape == "circle":
        for key in spatial_keys:
            ranges[key] = center[dataset.index("spacetime", key)]
        ranges["r"] = [-1.0, radius-0.5*cell_size]
        # ranges = {"x": center[0], "y": center[1], "r": [-1.0, radius-0.5*cell_size]}
    else:
        raise ValueError(f"Unrecognized {shape} boundary shape.")
    return filter_points(dataset=dataset, ranges=ranges, mode="open", shape=shape, eps=eps)

def subsample(datasets: list[PhySysDataset], samples_per_dataset: int, seed: int = 42) -> PhySysDataset:
    """
    Randomly permute and subsample datasets (seed for reproducibility), and then insert the resulting samples in a PhySysDataset.

    Parameters
    ----------
    datasets : list[PhySysDataset]
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
        indices = torch.randperm(ds.length)[:samples_per_dataset]
        new_cols = [col[indices] for col in ds.columns()]
        if cols is None:
            cols = new_cols
        else:
            for i, col in enumerate(new_cols):
                cols[i] = torch.cat([cols[i], col])
    
    new_ds = PhySysDataset([(key, col) for key, col in zip(datasets[0].cols.keys(), cols)])
    for k in datasets[0].subkeys.keys():
        new_ds.set_subkeys(k, datasets[0].subkeys[k])
    return new_ds

def subsample(
    dataset: PhySysDataset,
    n_samples: int,
    seed: int = 42
    ) -> PhySysDataset:
    
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))
    indices = indices[:n_samples]
    return dataset.subsample(indices)

def split(
    dataset: PhySysDataset,
    n_samples: List[int],
    seed: int = 42
) -> PhySysDataset:
    torch.manual_seed(seed)
    permutation = torch.randperm(len(dataset))
    split = []
    count = 0
    for n in n_samples:
        indices = permutation[count:count+n]
        split.append(dataset.subsample(indices))
        count += n
        if count > len(dataset):
            raise ValueError(f"The total number of samples ({sum(n_samples)}) required exceeds the dataset size ({len(dataset)}).")
    return split

def subsample_normal(dataset: PhySysDataset, mean: torch.Tensor, stddev: float, n_samples: int) -> PhySysDataset:
    d = len(mean) # d <= dim(spacetime)
    X = dataset.cols["spacetime"][:, :d]

    # compute the Gaussian bump weights
    # w = exp(-||x - center||^2 / (2 * stddev^2))
    weights = torch.exp(-torch.sum((X - mean) ** 2, dim=1) / (2 * (stddev ** 2)))

    # sample indices based on the weights
    indices = torch.multinomial(weights, n_samples, replacement=False)

    return dataset.subsample(indices)

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

def get_normal(n_samples: int, mean: torch.Tensor | float, std: torch.Tensor | float, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    if type(mean) is float:
        mean = torch.tensor(mean)
        std = torch.tensor(std)
    return torch.normal(mean=mean.repeat(n_samples, 1), std=std.repeat(n_samples, 1))

def get_uniform(n_samples: int, a: torch.Tensor | float, b: torch.Tensor | float, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    if type(a) is float or a.shape.numel() == 1:
        return torch.rand(n_samples) * (b - a) + a
    return torch.rand(n_samples, len(a)) * (b - a) + a
    
def get_iterators(datas: List[PhySysDataset], batch_size: float, seed: int) -> Tuple[List[Iterator], int]:
    torch.manual_seed(seed)
    random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    batch_sizes = []
    data_sizes = []
    max_len_idx = 0
    max_len = -1

    for i, task_data in enumerate(datas):
        N = len(task_data)
        if N > max_len:
            max_len = N
            max_len_idx = i

        if batch_size < N:
            N = N - N % batch_size
            batch_sizes.append(batch_size)

        else:
            batch_sizes.append(N)

        data_sizes.append(N)
        print(f"batch size of task {i} = {batch_sizes[-1]}")
        print(f"dataset size of task {i}) = {data_sizes[-1]}")

    for i, data_size in enumerate(data_sizes):
        datas[i] = Subset(datas[i], list(range(data_size)))
    
    iterators = [None for _ in datas]
    
    for i, task_data in enumerate(datas):
        dataloader = DataLoader(task_data, batch_sizes[i], generator=gen, shuffle=True)#, drop_last=True)
        if i == max_len_idx:
            iterators[i] = dataloader
            steps_per_epoch = len(dataloader)
        else:
            iterators[i] = cycle(dataloader)
    return iterators, steps_per_epoch