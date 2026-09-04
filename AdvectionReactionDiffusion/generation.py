"""
generate.py
===========

This module implements data generation for advection-reaction-diffusion systems.
"""

import torch
from torch.utils.data import ConcatDataset
from phy_sys_dataset import PhySysDataset
from typing import Set
from AdvectionReactionDiffusion.advection_reaction_diffusion import AdvectionReactionDiffusion
from AdvectionReactionDiffusion.advection_velocity import Velocity
from AdvectionReactionDiffusion.reaction_source import Source
from AdvectionReactionDiffusion.boundary_condition import BoundaryCondition
from AdvectionReactionDiffusion.initial_condition import InitialCondition

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
        raise ValueError(f"Invalid mode '{mode}'. It must be in ['uniform', 'grid']")

def generate_AdvectionReactionDiffusion(
        shape: str,
        spatial_region: dict,
        #bc: dict,
        #ic: dict,
        bc: BoundaryCondition,
        ic: InitialCondition,

        t0: float,
        tN: float,
        dt: float,

        n_samples: int,
        seed: int = 42,

        velocity: Velocity = None,
        diffusion_coeff: float = None,
        source: Source = None,
        implicit_source: Source = None,

        include_diffusion_coeff: bool = False,
        include_velocity_values: bool = False,
        include_source_values: bool = False,
        include_implicit_source_A: bool = False,
        include_implicit_source_B: bool = False,
        #A: float = None, # if the source is implicit and you want to save its params values
        #B: float = None, # if the source is implicit and you want to save its params values

        include_bc: bool = False,

        snapshot_times: Set[float] = None,
        
        vmin: float = None,
        vmax: float = None,
        cmap: str = "inferno",
        figsize: tuple = (3.5, 3.5)
) -> Tuple[PhySysDataset, PhySysDataset]:

    if velocity is None:
        velocity = Velocity.null_velocity()
    if source is None:
        source = Source.null_source()
    if implicit_source is None:
        implicit_source = Source.null_source()
    if diffusion_coeff is None:
        diffusion_coeff = 0.0

    pde = AdvectionReactionDiffusion(
        velocity = velocity,
        diffusion_coeff = diffusion_coeff,
        source = source,
        implicit_source = implicit_source
    )

    pde.set_spatial_points(mode=shape, **spatial_region)
    #pde.set_IC(**ic)
    #pde.set_BC(**bc)
    pde.set_IC(initial_condition=ic)
    pde.set_BC(boundary_condition=bc)
    pde.solve(
        t0=t0, tN=tN, dt=dt, 
        n_samples=n_samples,
        seed=seed,
        snapshot_times=snapshot_times,
        vmin=vmin, vmax=vmax, 
        cmap=cmap, 
        figsize=figsize
    )

    trajectory_ds = None
    snapshots_ds = None

    for seq, timeline in [("trajectory", pde.trajectory.t), ("snapshots", pde.trajectory.t_full)]:

        for i, time in enumerate(timeline):
            if seq == "trajectory":
                x = pde.trajectory.x[i]
                y = pde.trajectory.y[i]
                u = pde.trajectory.f[i]
                if shape == "rectangle":
                    du = pde.trajectory.df[i]
                    d2u = pde.trajectory.d2f[i]
                velocity = pde.trajectory.velocity[i]
                source = pde.trajectory.source[i]
            else:
                x = pde.trajectory.x_full
                y = pde.trajectory.y_full
                u = pde.trajectory.f_full[i]
                if shape == "rectangle":
                    du = pde.trajectory.df_full[i]
                    d2u = pde.trajectory.d2f_full[i]
                velocity = pde.trajectory.velocity_full[i]
                source = pde.trajectory.source_full[i]

            t = torch.Tensor(time.repeat(len(x)))
            spacetime = torch.stack([x, y, t], dim=1)

            params = []
            param_keys = []
            if include_diffusion_coeff:
                diff_coeff = torch.tensor(diffusion_coeff).repeat(len(x))
                params.append(diff_coeff)
                param_keys.append("D")
            if include_velocity_values:
                vx = velocity[:, 0]
                params.append(vx)
                param_keys.append("vx")
                vy = velocity[:, 1]
                params.append(vy)
                param_keys.append("vy")
            if include_source_values:
                params.append(source)
                param_keys.append("s")
            if include_implicit_source_A:
                if implicit_source.A is None:
                    raise ValueError(f"Missing implicit source param 'A'.")
                a = torch.tensor(implicit_source.A).repeat(len(x))
                params.append(a)
                param_keys.append("A")
            if include_implicit_source_B:
                if implicit_source.B is None:
                    raise ValueError(f"Missing implicit source param 'B'.")
                b = torch.tensor(implicit_source.B).repeat(len(x))
                params.append(b)
                param_keys.append("B")

            if params != []:
                params = torch.stack(params, dim=1)

            bcs = None
            if include_bc:
                if shape == "rectangle":
                    bcs = torch.tensor([bc.left[1], bc.top[1], bc.right[1], bc.bottom[1]]).repeat(len(x), 1)
                elif shape == "circle":
                    bcs = torch.tensor(bc.circumference[1]).repeat(len(x), 1)
                else:
                    raise ValueError(f"Unknown domain shape '{shape}'.")

            if params != [] and bcs is not None:
                if shape == "rectangle":
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u),
                        ("du", du),
                        ("d2u", d2u),
                        ("param", params),
                        ("bc", bcs)
                    ])
                    frame_ds.set_subkeys("bc", ["left", "top", "right", "bottom"])
                else:
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u),
                        ("param", params),
                        ("bc", bcs)
                    ])
                frame_ds.set_subkeys("param", param_keys)
            elif params != []:
                if shape == "rectangle":
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u),
                        ("du", du),
                        ("d2u", d2u),
                        ("param", params)
                    ])
                else:
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u),
                        ("param", params)
                    ])
                frame_ds.set_subkeys("param", param_keys)
            elif bcs is not None:
                if shape == "rectangle":
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u),
                        ("du", du),
                        ("d2u", d2u),
                        ("bc", bcs)
                    ])
                    frame_ds.set_subkeys("bc", ["left", "top", "right", "bottom"])
                else:
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u),
                        ("bc", bcs)
                    ])
            else:
                if shape == "rectangle":
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u),
                        ("du", du),
                        ("d2u", d2u)
                    ])
                else:
                    frame_ds = PhySysDataset([
                        ("spacetime", spacetime),
                        ("u", u)
                    ])
            frame_ds.set_subkeys("spacetime", ["x", "y", "t"])

            if seq == "trajectory":
                if trajectory_ds is None:
                    trajectory_ds = frame_ds
                else:
                    trajectory_ds.merge(frame_ds)
            else:
                if snapshots_ds is None:
                    snapshots_ds = frame_ds
                else:
                    snapshots_ds.merge(frame_ds)

    return trajectory_ds, snapshots_ds

def generate_AdvectionReactionDiffusion_unlabeled(
        n_samples: int,
        mode: str,
        x_range: Tuple[float],
        y_range: Tuple[float],
        t_range: Tuple[float],
        
        velocity: Velocity = None,
        diffusion_coeff: float = None,
        source: Source = None,
        implicit_source: Source = None,

        shape: str = None,
        bc: BoundaryCondition = None,

        include_implicit_source_A: bool = False,
        include_implicit_source_B: bool = False,
        #A: float = None, # if the source is implicit and you want to save its params values
        #B: float = None, # if the source is implicit and you want to save its params values

        dx: float = None, dt: float = None,
        seed: int = 42
) -> PhySysDataset:
    X = sample_points(n_samples=n_samples, mode=mode, ranges=[x_range, y_range, t_range], steps=[dx, dx, dt], seed=seed)
    x = X[:, 0]
    y = X[:, 1]
    t = X[:, 2]

    include_diffusion_coeff = (diffusion_coeff != None)
    include_velocity_values = (velocity != None)
    include_source_values = (source != None)
    include_bc = (bc != None)
    
    params = []
    param_keys = []
    if include_diffusion_coeff:
        diff_coeff = torch.flatten(torch.tensor(diffusion_coeff).repeat(len(x), 1))
        params.append(diff_coeff)
        param_keys.append("D")
    if include_velocity_values:
        vx, vy = velocity(x, y, t)
        vx = torch.from_numpy(vx)
        vy = torch.from_numpy(vy)
        params.append(vx)
        params.append(vy)
        param_keys.append("vx")
        param_keys.append("vy")
    if include_source_values:
        s = source(x, y, t)
        params.append(s)
        param_keys.append("s")
    if include_implicit_source_A:
        a = torch.flatten(torch.tensor(implicit_source.A).repeat(len(x), 1))
        params.append(a)
        param_keys.append("A")
    if include_implicit_source_B:
        b = torch.flatten(torch.tensor(implicit_source.B).repeat(len(x), 1))
        params.append(b)
        param_keys.append("B")

    bcs = None
    if include_bc:
        if shape == "rectangle":
            bcs = torch.tensor([bc.left[1], bc.top[1], bc.right[1], bc.bottom[1]]).repeat(len(x), 1)
        elif shape == "circle":
            bcs = torch.tensor(bc.circumference[1]).repeat(len(x), 1)
        else:
            raise ValueError(f"Unknown domain shape '{shape}'.")
        
    if params != []:
        params = torch.stack(params, dim=1)
        
    if params != [] and include_bc:
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
    elif include_bc:
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
    return dataset