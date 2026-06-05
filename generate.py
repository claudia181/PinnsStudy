"""
generate.py
===========

This module implements data generation.

Functions:
- some functions used only inside this module.
- generate: implements data generaition.
- generate_from_config:
    given a configuration file/dictionary, prepares the arguments for the generate function, 
    call the generate, save the dataset, if required, and return the dataset.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset
import random
import os
import yaml
from typing import Any, Callable, List, Tuple, Set
from allen_cahn import AllenCahn
from advection_reaction_diffusion import AdvectionReactionDiffusion

X               = 0
U               = 1
DU              = 2
D2U             = 3
OUTWARD_NORMAL  = 4

def generate_AllenCahn(lam: float, xi: list, X: torch.Tensor) -> TensorDataset:
    pde = AllenCahn(lam=lam, force_params=xi)
    pde.set_spatial_points(x=X[:, 0], y=X[:, 1])
    #points = torch.stack([pde.x, pde.y], dim=1)
    pde.solve()
    #param_values = [lam] + xi
    #torch.tensor(param_values).repeat(len(X), 1)

    dataset = TensorDataset(
            X, # X = 0
            pde.u, # U = 1
            pde.du, # DU = 2
            pde.d2u # D2U = 3
            #param_values # PDE_VALUES = 4
            )
    return dataset

def generate_AllenCahn_unlabeled(X: torch.Tensor) -> TensorDataset:
    dataset = TensorDataset(X)
    return dataset

def help_rectangle() -> None:
    print("""
"x_range": Tuple[float, float],
"y_range": Tuple[float, float],
"dx": float,
"dy": float
    """)

def help_rectangle_bc() -> None:
    print("""
"left": Tuple[str, float],
"top": Tuple[str, float],
"right": Tuple[str, float],
"bottom": Tuple[str, float]
    """)

def help_circle() -> None:
    print("""
"cell_size": float,
"radius": float
    """)

def help_circle_bc() -> None:
    print("""
"mode": str,
"value": float
    """)

def help_ic() -> None:
    print("""
"gaussian": bool,
"periodic_circles": bool,
"periodic_valleys": bool,
"periodic_stripes": bool,
"periodic_grid": bool,
"uniform_noise": bool,
"u0": np.ndarray,
"centers": List[Tuple[float, float]], "amps": List[float], "sigmas": List[float],
"A": float, "Ax": float, "Ay": float,
"B": float, "Bx": float, "By": float,
"Cx": float, "Cy": float,
"D": float,
"min_noise": float, "max_noise": float
    """)

def generate_AdvectionReactionDiffusion(
        velocity: Callable,
        diffusion_coeff: float,
        source: Callable,
        implicit_source: Callable,
        #implicit_source: str, 
        #A: float, 
        #B: float,

        shape: str,
        spatial_region: dict,
        bc: dict,
        ic: dict,

        t0: float,
        tN: float,
        dt: float,

        snapshots: Set[float] = None,
        n_snapshots: int = None,
        snapshot_start: float = 0.0,
        vmin: float = None,
        vmax: float = None,
        cmap: str = "inferno",
        figsize: tuple = (3.5, 3.5)
) -> ConcatDataset:

    pde = AdvectionReactionDiffusion(
        velocity = velocity,
        diffusion_coeff = diffusion_coeff,
        source = source,
        implicit_source = implicit_source
    )

    pde.set_spatial_points(mode=shape, **spatial_region)
    X = torch.stack([torch.from_numpy(pde.x), torch.from_numpy(pde.y)], dim=1)
    x = torch.from_numpy(pde.x)
    y = torch.from_numpy(pde.y)
    pde.set_IC(**ic)
    pde.set_BC(**bc)
    pde.solve(
        t0=t0, tN=tN, dt=dt, 
        snapshots=snapshots, 
        n_snapshots=n_snapshots, snapshot_start=snapshot_start, 
        vmin=vmin, vmax=vmax, 
        cmap=cmap, 
        figsize=figsize
    )
    u = [torch.from_numpy(u_snapshot) for u_snapshot in pde.u]
    du = [torch.from_numpy(du_snapshot) for du_snapshot in pde.du]
    d2u = [torch.from_numpy(d2u_snapshot) for d2u_snapshot in pde.d2u]

    datasets = []
    for i, time in enumerate(pde.t):
        t = torch.flatten(torch.tensor(time).repeat(len(x), 1))
        X = torch.stack([x, y, t], dim=1)
        #vx = torch.from_numpy(pde.velocity[i][0])
        #vy = torch.from_numpy(pde.velocity[i][1])
        #s = torch.from_numpy(pde.source[i])
        #a = torch.tensor(A).repeat(len(x), 1)
        #b = torch.tensor(B).repeat(len(x), 1)
        #diff_coeff = torch.tensor(diffusion_coeff).repeat(len(x), 1)
        #param_values = torch.stack([diff_coeff, vx, vy, s, a, b], dim=1)
        dataset = TensorDataset(
            X,
            u[i],
            du[i],
            d2u[i]
            )
        datasets.append(dataset)
    return ConcatDataset(datasets)

def generate_AdvectionReactionDiffusion_unlabeled(X: torch.Tensor, snapshots: Set[float]) -> ConcatDataset:
    x = X[:, 0]
    y = X[:, 1]
    
    datasets = []
    for time in snapshots:
        t = torch.flatten(torch.tensor(time).repeat(len(x), 1))
        X = torch.stack([x, y, t], dim=1)
        dataset = TensorDataset(X)
        datasets.append(dataset)
    return ConcatDataset(datasets)