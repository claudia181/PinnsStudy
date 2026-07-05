"""
generate.py
===========

This module implements data generation.
"""

import torch
from phy_sys_dataset import PhySysDataset
from StationaryAllenCahn.allen_cahn import AllenCahn
from phy_sys_dataset import PhySysDataset
from data_utils import get_uniform, get_grid
from typing import Tuple, List

def sample_points(mode: str, n_samples: int, ranges: List[Tuple[float]], steps: List[float] = None, seed: int = 42) -> torch.Tensor:
    if mode == "uniform":
        a = torch.tensor([r[0] for r in ranges])
        b = torch.tensor([r[1] for r in ranges])
        return get_uniform(n_samples=n_samples, a=a, b=b, seed=seed)
    elif mode == "grid":
        a = [r[0] for r in ranges]
        b = [r[1] for r in ranges]
        return get_grid(xmin_list=a, xmax_list=b, dx_list=steps)
    else:
        raise ValueError(f"Invalid mode '{mode}'. It must be in ['uniform', 'grid'].")


def generate_AllenCahn(
        n_samples: int,
        mode: str,
        x_range: Tuple[float],
        y_range: Tuple[float],
        lam: float, 
        xi: list, 
        include_lam: bool = False, 
        include_xi: bool = False,
        dx: float = None, dy: float = None,
        seed: int = 42
) -> PhySysDataset:
    X = sample_points(mode=mode, n_samples=n_samples, ranges=[x_range, y_range], steps=[dx, dy], seed=seed)
    #get_uniform(n_samples=n_samples, a=torch.tensor([x_range[0], y_range[0]]), b=torch.tensor([x_range[1], y_range[1]]), seed=seed)
    x = X[:, 0]
    y = X[:, 1]

    pde = AllenCahn(lam=lam, force_params=xi)
    pde.set_spatial_points(x=x, y=y)
    #points = torch.stack([pde.x, pde.y], dim=1)
    pde.solve()

    params = []
    param_keys = []
    if include_lam:
        params.append(lam)
        param_keys.append("lam")
    if include_xi:
        for i, item in enumerate(xi):
            params.append(item)
            param_keys.append(f"xi{i}")
    if params != []:
        params = torch.tensor(params).repeat(len(X), 1)

        dataset = PhySysDataset(cols=[
            ("spacetime", X),
            ("u", pde.u),
            ("du", pde.du),
            ("d2u", pde.d2u),
            ("param", params)
        ])
        dataset.set_subkeys("param", param_keys)
    else:
        dataset = PhySysDataset(cols=[
            ("spacetime", X),
            ("u", pde.u),
            ("du", pde.du),
            ("d2u", pde.d2u)
        ])
    dataset.set_subkeys("spacetime", ["x", "y"])
    return dataset

def generate_AllenCahn_unlabeled(
        n_samples: int,
        mode: str,
        x_range: Tuple[float],
        y_range: Tuple[float], 
        lam: float = None, 
        xi: list = None, 
        include_lam: bool = False, 
        include_xi: bool = False,
        dx: float = None, dy: float = None,
        seed: int = 42
) -> PhySysDataset:
    X = sample_points(n_samples=n_samples, mode=mode, ranges=[x_range, y_range], steps=[dx, dy], seed=seed)
    params = []
    param_keys = []
    if include_lam:
        if lam is None:
            raise ValueError("Missing lambda parameter (lam = None).")
        else:
            params.append(lam)
            param_keys.append("lam")
    if include_xi:
        if xi is None:
            raise ValueError("Missing xi parameter (xi = None).")
        else:
            for i, item in enumerate(xi):
                params.append(item)
                param_keys.append(f"xi{i}")
    if params == []:
       dataset = PhySysDataset([("spacetime", X)])
    else:
        params = torch.tensor(params).repeat(len(X), 1)
        dataset = PhySysDataset([
            ("spacetime", X), 
            ("param", params)
        ])
        dataset.set_subkeys("param", param_keys)
    dataset.set_subkeys("spacetime", ["x", "y"])
    return dataset