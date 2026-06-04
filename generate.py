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
from pde_utils import Pde, key_idx, ic_key_idx, key_str, ic_key_str, n_pde_params, n_ic_params
from typing import Any, Callable, List, Tuple, Set
from allen_cahn import AllenCahn
from advection_reaction_diffusion import AdvectionReactionDiffusion

X               = 0
U               = 1
DU              = 2
D2U             = 3
PDE_KEYS        = 4
PDE_VALUES      = 5
IC_KEYS         = 6
IC_VALUES       = 7
RESIDUAL_KEYS   = 8
RESIDUAL_VALUES = 9
TIMES           = 10
OUTWARD_NORMAL  = 11 # added successively by data_utils

DIM_LABEL_DICT = {"x": 0, "y": 1}
STEP_LABEL_DICT = {"dx": 0, "dy": 1}

def get_key(idx: int) -> str:
    """
    Returns the DIM_LABEL_DICT key relative to the index idx.
    """
    for key in DIM_LABEL_DICT.keys():
        if DIM_LABEL_DICT[key] == idx: return key
    raise ValueError(f"Error: generate: get_key: index {idx} not found.")

def get_step_key(idx: int) -> str:
    """
    Returns the STEP_LABEL_DICT key relative to the index idx.
    """
    for key in STEP_LABEL_DICT.keys():
        if STEP_LABEL_DICT[key] == idx: return key
    raise ValueError(f"Error: generate: get_key: index {idx} not found.")

def sort_keys(keys: list) -> list:
    """
    Returns the sorted keys list according to the DIM_LABEL_DICT ordering.
    """
    sorted_key_indexes = sorted([DIM_LABEL_DICT[key] for key in keys])
    sorted_keys = [get_key(idx) for idx in sorted_key_indexes]
    return sorted_keys

def sort_step_keys(keys: list) -> list:
    """
    Returns the sorted keys list according to the STEP_LABEL_DICT ordering.
    """
    sorted_key_indexes = sorted([STEP_LABEL_DICT[key] for key in keys])
    sorted_keys = [get_step_key(idx) for idx in sorted_key_indexes]
    return sorted_keys
    

# Data generation ----------------------------------------------------------------------------------------
def generate(
        pde_name: str,
        mode: str,
        shape: str,
        ranges: list,
        pde_param_keys: list = [],
        pde_param_values: list = [],
        ic_keys: list = [],
        ic_values: list = [],
        bc_dict: dict = {},
        steps: list = [],
        cell_size: float = None,
        radius: float = None,
        n_rand_points: int = 1000,
        t0: float = 0.0,
        tN: float = None,
        dt: float = None,
        snapshots: list = None,
        n_snapshots: int = None,
        snapshot_start: float = None,
        vmin: float = None,
        vmax: float = None,
        cmap: str = "inferno",
        figsize = (3.5, 3.5)
        ) -> ConcatDataset:
    '''
    Generate a dataset for training/test according to the passed args.

    Parameters
    ----------
    pde_name : str
        PDE name.
    mode : str
        Describes spatial arrangement of points. Allowed values: grid | random.
    shape : str
        Shape of the system domain: "rectangle" | "circle" for ARD eqs.
    ranges : tuple
        Spatial domain ranges [xmin, xmax], [ymin, ymax].
    pde_param_keys : list
        Keys of the PDE parameters.
    pde_param_values : list
        Values of the PDE parameters.
    ic_keys : list
        Keys of the PDE initial condition parameters.
    ic_values : list
        Values of the PDE initial condition parameters.
    bc_dict : dict
        Dictionary describing the boundary conditions.
    steps : list
        Steps [dx, dy] in the spatial domain (for grid mode).
    cell_size : float
        Cell size for circular domain grid.
    radius : float
        Radius of the circular domain grid.
    n_rand_points : int
        Number of random points in the spatial domain (for random mode).
    t0 : float
        Initial time.
    tN : float
        Final time.
    dt : float
        Time step for simulation.
    snapshots : list
        Time snapshots to return (list of time values).
    n_snapshots : int
        Number of equispaced time snapshots to return.
    snapshot_start : float
        When to start to store snapshots.
    vmin : float
        For visualization.
    vmax : float
        For visualization.
    cmap : str
        For visualization.
    figsize : tuple
        For visualization.

    Returns
    -------
    torch.ConcatDataset
        A torch ConcatDataset s.t. each component correspond to a time snapshot (increasingly ordered by time values).
    '''
    if "grid" in mode:
        n_dimentions = len(ranges)
        coordinate_vectors = []
        ranges_dict = {}
        keys = DIM_LABEL_DICT.keys()
        keys = [f"{key}_range" for key in keys]
        steps_dict = {}
        for i in range(n_dimentions):
            ranges_dict[keys[i]] = ranges[i]
            xmin = ranges[i][0]
            xmax = ranges[i][1]
            dx = steps[i]
            steps_dict[get_step_key(i)] = dx

            if xmin == xmax:
                coordinate_vectors.append([xmin])
            else:
                # x = [xmin+dx, xmin+2dx, ..., xmin+Ndx <= xmax]
                coordinate_vectors.append(np.arange(xmin, xmax+0.5*dx, dx))

        grid_coords = np.meshgrid(*coordinate_vectors)

        if shape == "circle":
            mask = grid_coords[0]**2 + grid_coords[1]**2 <= radius**2
            grid_coords = [axis[mask] for axis in grid_coords]

        # grid_coords[0] = x_pts = [x1, ..., xN, x1, ..., xN, ..., x1, ..., xN], shape (N^2, 1)
        # grid_coords[1] = y_pts = [y1, ..., yN, y1, ..., yN, ..., y1, ..., yN], shape (N^2, 1)
        # ...
        reshaped_columns = [arr.reshape(-1, 1) for arr in grid_coords]

        # points = [(x1, y1, ...), ..., (x[N^2], y[N^2], ...)], shape (N^2, 2)
        points = np.column_stack(reshaped_columns)

        circle_dict = {"cell_size": cell_size, "radius": radius}

        pde = Pde(name=pde_name, param_keys=pde_param_keys, param_values=pde_param_values)
        points = torch.tensor(points, dtype = torch.float32)

        if "unlabeled" not in mode:
            grid_infos = ranges_dict | steps_dict | circle_dict | {"mode": shape}
            pde.set_spatial_points(X=points, **grid_infos)
            pde.set_IC(keys=ic_keys, values=ic_values)
            pde.set_BC(bc_dict=bc_dict)
            pde.solve(t0=t0, tN=tN, dt=dt, snapshots=snapshots, n_snapshots=n_snapshots, snapshot_start=snapshot_start, vmin=vmin, vmax=vmax, cmap=cmap, figsize=figsize)
            points = pde.points
            times = pde.times
            solution = pde.solution
            der = pde.der
        else:
            points = torch.stack([points[:, 0], points[:, 1]], dim=1)
            times = np.arange(start=t0, stop=tN, step=dt)
            if len(times) == 0:
                times = [0.0]
            solution = [torch.tensor([torch.nan for _ in range(len(points))]) for _ in times]
            der = [solution, solution, solution]

    elif mode == "random_unlabeled":
        vectors = [np.random.rand(n_rand_points) * (r[1] - r[0]) + r[0] for r in ranges]
        points = np.column_stack(vectors)
        pde = Pde(name=pde_name, param_keys=pde_param_keys, param_values=pde_param_values)

        points = torch.tensor(points, dtype = torch.float32)
        points = torch.stack([points[:, 0], points[:, 1]], dim=1)
        times = np.arange(start=t0, stop=tN, step=dt)
        if len(times) == 0:
            times = [0.0]
        solution = [torch.tensor([torch.nan for _ in range(len(points))]) for _ in times]
        der = [solution, solution, solution]

    param_keys = torch.tensor(pde_param_keys).repeat(len(points), 1)
    param_values = torch.tensor(pde_param_values).repeat(len(points), 1)

    ic_keys = torch.tensor(ic_keys).repeat(len(points), 1)
    ic_values = torch.tensor(ic_values).repeat(len(points), 1)

    additional_info_keys_list = []
    additional_info_values_list = []
    for t in range(len(times)):
        additional_info_keys, additional_info_values = pde.additional_info(t, x=points[:, 0], y=points[:, 1])
        additional_info_keys_list.append(additional_info_keys)
        additional_info_values_list.append(additional_info_values)
    
    datasets = []
    for t in range(len(times)):
        dataset = TensorDataset(
            points,                                                             # X                 = 0
            solution[t],                                                    # U                 = 1
            der[1][t],                                                      # DU                = 2
            der[2][t],                                                      # D2U               = 3
            param_keys,                                                         # PDE_KEYS          = 4
            param_values,                                                       # PDE_VALUES        = 5
            ic_keys,                                                            # IC_KEYS           = 6
            ic_values,                                                          # IC_VALUES         = 7
            additional_info_keys_list[t].repeat(len(points), 1),                # RESIDUAL_KEYS     = 8
            additional_info_values_list[t],                                     # RESIDUAL_VALUES   = 9
            torch.flatten(torch.tensor(times[t]).repeat(len(points), 1)),   # TIMES             = 10
            )
        datasets.append(dataset)

    return ConcatDataset(datasets)

# --------------------------------------------------------------------------------------
def get_param(
        config_section_dict: dict,
        config_key: str,
        default_val: Any = None,
        type_func: Callable = None
        ) -> Any:
        """
        Function to extract dictionary arguments.
        """
        # Try to get from config file
        if config_section_dict and config_key in config_section_dict:
            val = config_section_dict[config_key]
            return type_func(val) if type_func else val
        # Use default
        else:
            return default_val

# --------------------------------------------------------------------------------------
def generate_from_config(
        configuration: str|dict,
        vmin: float = None,
        vmax: float = None,
        cmap: str = "inferno",
        figsize: tuple = (3.5, 3.5)
        ) -> ConcatDataset:
    """
    Generate a dataset given a configuration filepath/dictionary.

    Parameters
    ----------
    configuration : str|dict
        The YAML configuration filepath or the configuration dictionary.
    vmin : float
        Visualization parameter.
    vmax : float
        Visualization parameter.
    cmap : str
        Visualization parameter.
    figsize : tuple
        Visualization parameter.

    Returns
    -------
    torch.ConcatDataset
        Generated dataset.
    """
    if type(configuration) is str:
        # Load configuration from YAML file
        gen_config = {}
        if os.path.exists(configuration):
            with open(configuration, 'r') as f:
                gen_config = yaml.safe_load(f)
        else:
            raise RuntimeError(f"Config file '{configuration}' not found.")
    else:
        gen_config = configuration

    # Get the section
    params_dict         = gen_config.get("parameters", {})
    ic_dict             = gen_config.get("initial_conditions", {})
    bc_dict             = gen_config.get("boundary_conditions", {})
    time_dict           = gen_config.get("time", {})
    space_dict          = gen_config.get("space", {})
    steps_dict          = space_dict.get("steps", {})
    ranges_dict         = space_dict.get("ranges", {})
    circle_dict         = space_dict.get("circle", {})
    save_options_dict   = gen_config.get("save_options", {})

    # Get the seed
    seed = get_param(gen_config, "seed", default_val=30, type_func=int)

    # Set the seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    if save_options_dict == {}:
        save = False
    else:
        save =True

    # Get the data directory
    data_dir = get_param(save_options_dict, "directory", default_val="data")

    if not os.path.exists(data_dir):
        # Create the directory 'data'
        os.makedirs(data_dir)
    
    # Get the filename
    filename = get_param(save_options_dict, "filename", default_val="")

    pde_name = get_param(gen_config, "PDE", default_val="")

    if filename == "":
        filename = f"{pde_name}"

    param_keys = [key_idx(key, pde_name) for key in params_dict.keys()]
    param_keys.sort()
    param_keys_full = []
    for i in range(n_pde_params(pde_name)):
        if i in param_keys:
            param_keys_full.append(i)
        else:
            param_keys_full.append(-1)
    param_keys = param_keys_full

    param_values = []
    for key in param_keys:
        if key != -1:
            p = params_dict[key_str(key, pde_name)]
            if type(p) is str:
                param_values.append(key_idx(p, "Additional info"))
            else: # type(p) is float
                param_values.append(float(p))
        else:
            param_values.append(torch.nan)

    ic_keys = [ic_key_idx(key, pde_name) for key in ic_dict.keys()]
    ic_keys.sort()
    ic_keys_full = []
    for i in range(n_ic_params(pde_name)):
        if i in ic_keys:
            ic_keys_full.append(i)
        else:
            ic_keys_full.append(-1)
    ic_keys = ic_keys_full

    ic_values = []
    for key in ic_keys:
        if key != -1:
            v = ic_dict[ic_key_str(key, pde_name)]
            ic_values.append(v)
        else:
            ic_values.append(torch.nan)
    
    shape = space_dict.get("shape", "rectangle")
    
    if shape == "rectangle":
        bc = {"left": ["Neumann", 0.0], "top": ["Neumann", 0.0], "right": ["Neumann", 0.0], "bottom": ["Neumann", 0.0]}
        for side in bc.keys():
            side_pair = get_param(bc_dict, side, type_func=list)
            if side_pair is not None:
                bc[side] = [str(side_pair[0]), float(side_pair[1])]
    elif shape == "circle":
        bc = {"mode": "Neumann", "value": 0.0}
        if bc_dict is not None:
            bc = bc_dict

    if pde_name != "Pendulum":
        mode = get_param(space_dict, "mode", default_val=None) # grid or random_unlabeled
        if mode is None:
            raise ValueError("Missing 'mode': 'grid'|'grid_unlabeled'|'random_unlabeled'.")

        if mode not in ["grid", "grid_unlabeled", "random_unlabeled"]:
            raise ValueError(f"Unrecognized '{mode}' mode. Allowed values: 'grid', 'grid_unlabeled', 'random_unlabeled'.")

        sorted_keys = sort_step_keys(steps_dict.keys())
        steps = []
        for key in sorted_keys:
            step = float(steps_dict[key])
            steps.append(step)

        if "grid" in mode and steps == [] and shape == "rectangle":
            raise ValueError("Missing grid setting: 'steps'.")

        n_rand = get_param(space_dict, 'n_rand', default_val=None, type_func=int)

        if mode =="random_unlabeled" and n_rand is None:
            raise ValueError("Missing 'n_rand' for 'random_unlabeled' mode.")

        sorted_keys = sort_keys(ranges_dict.keys())
        if pde_name == "Allen-Cahn" or (pde_name == "Advection-Reaction-Diffusion" and shape == "rectangle"):
            missing_keys = [k for k in DIM_LABEL_DICT.keys() if k not in sorted_keys]
            if missing_keys != []:
                raise ValueError(f"Missing range for {missing_keys}.")

        ranges = []
        for key in sorted_keys:
            r = list(ranges_dict[key])
            ranges.append([float(r[0]), float(r[1])])

        if ranges == []: ranges = [[0.0, 0.0], [0.0, 0.0]]

        if shape == "circle":
            cell_size = get_param(circle_dict, "cell_size", type_func=float)
            radius = get_param(circle_dict, "radius", type_func=float)
            if cell_size is None:
                raise ValueError("Missing circle setting: 'cell_size'.")
            if radius is None:
                raise ValueError("Missing circle setting: 'radius'.")
            ranges = [[-radius, radius], [-radius, radius]]
            steps = [cell_size, cell_size]
        else:
            cell_size = None
            radius = None
    
    else:
        mode = "grid"
        ranges = [[0.0, 0.0], [0.0, 0.0]]
        steps = [1.0, 1.0]
        n_rand = None
        cell_size = None
        radius = None

    snapshots_str = get_param(time_dict, "snapshots", default_val=None, type_func=list)
    if snapshots_str is not None:
        snapshots = [float(s) for s in snapshots_str]
    else:
        snapshots = None
    n_snapshots = get_param(time_dict, "n_snapshots", default_val=None, type_func=int)
    snapshot_start = get_param(time_dict, "snapshot_start", default_val=None, type_func=float)

    if pde_name != "Pendulum" and snapshots is None and n_snapshots is None:
        n_snapshots = 2
    
    time_range = get_param(time_dict, "t", default_val=[0.0, 0.0], type_func=list)
    t0, tN = time_range
    dt = get_param(time_dict, "dt", default_val=1.0, type_func=float)
     
    dataset = generate(
        mode=mode,
        ranges=ranges,
        shape=shape,
        pde_name=pde_name,
        pde_param_keys=param_keys,
        pde_param_values=param_values,
        t0=t0, tN=tN, dt=dt,
        ic_keys=ic_keys,
        ic_values=ic_values,
        bc_dict=bc,
        steps=steps,
        cell_size=cell_size,
        radius=radius,
        n_rand_points=n_rand,
        snapshots=snapshots,
        n_snapshots=n_snapshots,
        snapshot_start=snapshot_start,
        vmin=vmin, vmax=vmax, cmap=cmap, figsize=figsize
        )
    
    if save:
        torch.save(dataset, f"{data_dir}/{filename}.pth")
        print(f"Dataset {filename}.pth stored in {data_dir}.")
    
    return dataset


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


def generate_AllenCahn(lam: float, xi: list, X: torch.Tensor) -> TensorDataset:
    pde = AllenCahn(lam=lam, force_params=xi)
    pde.set_spatial_points(x=X[:, 0], y=X[:, 1])
    #points = torch.stack([pde.x, pde.y], dim=1)
    pde.solve()
    param_values = [lam] + xi
    torch.tensor(param_values).repeat(len(X), 1)

    dataset = TensorDataset(
            X, # X = 0
            pde.u, # U = 1
            pde.du, # DU = 2
            pde.d2u, # D2U = 3
            param_values # PDE_VALUES = 4
            )
    return dataset

rectangle = {
    "x_range": Tuple[float, float],
    "y_range": Tuple[float, float],
    "dx": float, 
    "dy": float
}
rectangle_bc = {
    "left": Tuple[str, float],
    "top": Tuple[str, float],
    "right": Tuple[str, float],
    "bottom": Tuple[str, float]
}

circle = {
    "cell_size": float,
    "radius": float
}

circle_bc = {
    "mode": str,
    "value": float
}

ic = {
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
}

def generate_AdvectionReactionDiffusion(
        velocity: Callable,
        diffusion_coeff: float,
        source: Callable,
        implicit_source: str, 
        A: float, 
        B: float,

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
        implicit_source = implicit_source,
        A = A,
        B = B
    )

    pde.set_spatial_points(mode="rectangle", **spatial_region)
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
    u = [torch.from_numpy(u_snapshot) for u_snapshot in self.pde.u]
    du = [torch.from_numpy(du_snapshot) for du_snapshot in self.pde.du]
    d2u = [torch.from_numpy(d2u_snapshot) for d2u_snapshot in self.pde.d2u]

    datasets = []
    for i, time in enumerate(pde.t):
        t = torch.flatten(torch.tensor(time).repeat(len(x), 1))
        X = torch.stack([x, y, t], dim=1)
        dataset = TensorDataset(
            X,                                                             # X                 = 0
            pde.u[i],                                                    # U                 = 1
            der[1][i],                                                      # DU                = 2
            der[2][i],                                                      # D2U               = 3
            param_keys,                                                         # PDE_KEYS          = 4
            param_values,                                                       # PDE_VALUES        = 5
            ic_keys,                                                            # IC_KEYS           = 6
            ic_values,                                                          # IC_VALUES         = 7
            additional_info_keys_list[i].repeat(len(points), 1),                # RESIDUAL_KEYS     = 8
            additional_info_values_list[i],                                     # RESIDUAL_VALUES   = 9
            torch.flatten(torch.tensor(times[i]).repeat(len(points), 1)),   # TIMES             = 10
            )
        datasets.append(dataset)

    return ConcatDataset(datasets)


    for 
    torch.flatten(torch.tensor(times[t]).repeat(len(points), 1))
    X = torch.stack([torch.from_numpy(pde.x), torch.from_numpy(pde.y), ], dim=1)

    times = pde.t
    solution = pde.solution
    der = pde.der


#################################################################################################
    

        if "unlabeled" not in mode:
            grid_infos = ranges_dict | steps_dict | circle_dict | {"mode": shape}
            pde.set_spatial_points(X=points, **grid_infos)
            pde.set_IC(keys=ic_keys, values=ic_values)
            pde.set_BC(bc_dict=bc_dict)
            pde.solve(t0=t0, tN=tN, dt=dt, snapshots=snapshots, n_snapshots=n_snapshots, snapshot_start=snapshot_start, vmin=vmin, vmax=vmax, cmap=cmap, figsize=figsize)
            points = pde.points
            times = pde.times
            solution = pde.solution
            der = pde.der