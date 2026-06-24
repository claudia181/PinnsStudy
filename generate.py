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

import torch
from torch.utils.data import ConcatDataset
from phy_sys_dataset import PhySysDataset
from typing import Callable, Set
from allen_cahn import AllenCahn
from advection_reaction_diffusion import AdvectionReactionDiffusion
from phy_sys_dataset import PhySysDataset

def generate_AllenCahn(
        X: torch.Tensor, 
        lam: float, 
        xi: list, 
        include_lam: bool = False, 
        include_xi: bool = False
) -> PhySysDataset:
    pde = AllenCahn(lam=lam, force_params=xi)
    pde.set_spatial_points(x=X[:, 0], y=X[:, 1])
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
        X: torch.Tensor, 
        lam: float = None, 
        xi: list = None, 
        include_lam: bool = False, 
        include_xi: bool = False
) -> PhySysDataset:
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

        shape: str,
        spatial_region: dict,
        bc: dict,
        ic: dict,

        t0: float,
        tN: float,
        dt: float,

        include_diffusion_coeff: bool = False,
        include_velocity_values: bool = False,
        include_source_values: bool = False,
        include_implicit_source_A: bool = False,
        include_implicit_source_B: bool = False,
        A: float = None, # if the source is implicit and you want to save its params values
        B: float = None, # if the source is implicit and you want to save its params values

        include_bc: bool = False,

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
    #X = torch.stack([torch.from_numpy(pde.x), torch.from_numpy(pde.y)], dim=1)
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
        params = []
        param_keys = []
        if include_diffusion_coeff:
            diff_coeff = torch.flatten(torch.tensor(diffusion_coeff).repeat(len(x), 1))
            params.append(diff_coeff)
            param_keys.append("D")
        if include_velocity_values:
            vx = torch.from_numpy(pde.velocity[i][0])
            params.append(vx)
            param_keys.append("vx")
            vy = torch.from_numpy(pde.velocity[i][1])
            params.append(vy)
            param_keys.append("vy")
        if include_source_values:
            s = torch.from_numpy(pde.source[i])
            params.append(s)
            param_keys.append("s")
        if include_implicit_source_A:
            if A is None:
                raise ValueError(f"Missing implicit source param 'A'.")
            a = torch.flatten(torch.tensor(A).repeat(len(x), 1))
            params.append(a)
            param_keys.append("A")
        if include_implicit_source_B:
            if B is None:
                raise ValueError(f"Missing implicit source param 'B'.")
            b = torch.flatten(torch.tensor(B).repeat(len(x), 1))
            params.append(b)
            param_keys.append("B")

        if params != []:
            params = torch.stack(params, dim=1)

        bcs = None
        if include_bc:
            if shape == "rectangle":
                bcs = torch.tensor([bc[key][1] for key in ["left", "top", "right", "bottom"]]).repeat(len(x), 1)
            elif shape == "circle":
                bcs = torch.tensor(bc["value"]).repeat(len(x), 1)
            else:
                raise ValueError(f"Unknown domain shape '{shape}'.")

        if params != [] and bcs is not None:
            dataset = PhySysDataset([
                ("spacetime", X),
                ("u", u[i]),
                ("du", du[i]),
                ("d2u", d2u[i]),
                ("param", params),
                ("bc", bcs)
            ])
            if shape == "rectangle":
                dataset.set_subkeys("bc", ["left", "top", "right", "bottom"])
            dataset.set_subkeys("param", param_keys)
        elif params != []:
            dataset = PhySysDataset([
                ("spacetime", X),
                ("u", u[i]),
                ("du", du[i]),
                ("d2u", d2u[i]),
                ("param", params)
            ])
            dataset.set_subkeys("param", param_keys)
        elif bcs is not None:
            dataset = PhySysDataset([
                ("spacetime", X),
                ("u", u[i]),
                ("du", du[i]),
                ("d2u", d2u[i]),
                ("bc", bcs)
            ])
            if shape == "rectangle":
                dataset.set_subkeys("bc", ["left", "top", "right", "bottom"])
        else:
            dataset = PhySysDataset([
                ("spacetime", X),
                ("u", u[i]),
                ("du", du[i]),
                ("d2u", d2u[i])
            ])
        dataset.set_subkeys("spacetime", ["x", "y", "t"])
        datasets.append(dataset)

    return ConcatDataset(datasets)

def generate_AdvectionReactionDiffusion_unlabeled(
        X: torch.Tensor, 
        snapshots: Set[float],  
        
        velocity: Callable = None,
        diffusion_coeff: float = None,
        source: Callable = None,

        shape: str = None,
        bc: dict = None,

        include_diffusion_coeff: bool = False,
        include_velocity_values: bool = False,
        include_source_values: bool = False,
        include_implicit_source_A: bool = False,
        include_implicit_source_B: bool = False,
        A: float = None, # if the source is implicit and you want to save its params values
        B: float = None, # if the source is implicit and you want to save its params values

        include_bc: bool = False
) -> ConcatDataset:
    x = X[:, 0]
    y = X[:, 1]
    
    datasets = []
    for time in snapshots:
        t = torch.flatten(torch.tensor(time).repeat(len(x), 1))
        X = torch.stack([x, y, t], dim=1)

        params = []
        param_keys = []
        if include_diffusion_coeff:
            diff_coeff = torch.flatten(torch.tensor(diffusion_coeff).repeat(len(x), 1))
            params.append(diff_coeff)
            param_keys.append("D")
        if include_velocity_values:
            if velocity is None:
                raise ValueError(f"Missing velocity vector field.")
            vx, vy = velocity(x, y, t)
            vx = torch.from_numpy(vx)
            vy = torch.from_numpy(vy)
            params.append(vx)
            params.append(vy)
            param_keys.append("vx")
            param_keys.append("vy")
        if include_source_values:
            if source is None:
                raise ValueError(f"Missing source scalar field.")
            s = source(x, y, t)
            params.append(s)
            param_keys.append("s")
        if include_implicit_source_A:
            if A is None:
                raise ValueError(f"Missing implicit source param 'A'.")
            a = torch.flatten(torch.tensor(A).repeat(len(x), 1))
            params.append(a)
            param_keys.append("A")
        if include_implicit_source_B:
            if B is  None:
                raise ValueError(f"Missing implicit source param 'B'.")
            b = torch.flatten(torch.tensor(B).repeat(len(x), 1))
            params.append(b)
            param_keys.append("B")

        if params != []:
            params = torch.stack(params, dim=1)
    
        bcs = None
        if include_bc:
            if shape == "rectangle":
                bcs = torch.tensor([bc[key][1] for key in ["left", "top", "right", "bottom"]]).repeat(len(x), 1)
            elif shape == "circle":
                bcs = torch.tensor(bc["value"]).repeat(len(x), 1)
            else:
                raise ValueError(f"Unknown domain shape '{shape}'.")

        if params != [] and bcs is not None:
            dataset = PhySysDataset([
                ("spacetime", X), 
                ("param", params), 
                ("bc", bcs)
            ])
            dataset.set_subkeys("param", param_keys)
            if shape == "rectangle":
                dataset.set_subkeys("bc", ["left", "top", "right", "bottom"])
        elif params != []:
            dataset = PhySysDataset([
                ("spacetime", X), 
                ("param", params)
            ])
            dataset.set_subkeys("param", param_keys)
        elif bcs is not None:
            dataset = PhySysDataset([
                ("spacetime", X),
                ("bc", bcs)
            ])
            if shape == "rectangle":
                dataset.set_subkeys("bc", ["left", "top", "right", "bottom"])
        else:
            dataset = PhySysDataset([
                ("spacetime", X)
            ])
        dataset.set_subkeys("spacetime", ["x", "y", "t"])
        datasets.append(dataset)

    return ConcatDataset(datasets)