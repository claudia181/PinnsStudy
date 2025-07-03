import numpy as np
import torch
from torch.utils.data import TensorDataset
import random
import os
import sys
import argparse
import yaml
from utils import Pde

# Generate a set of points ----------------------------------------------------------------------------------------
def generate(mode: str, ranges: list, pde_name: str, pde_params: list, steps: list = [], n_rand_points: int = 1000):
    if mode == 'grid':
        n_dimentions = len(ranges)
        coordinate_vectors = []
        for i in range(n_dimentions):
            xmin = ranges[i][0]
            xmax = ranges[i][1]
            dx = steps[i]

            if xmin == xmax:
                coordinate_vectors.append([xmin])
            else:
                # x = [xmin+dx, xmin+2dx, ..., xmin+Ndx <= xmax]
                coordinate_vectors.append(np.arange(xmin, xmax, dx))

        grid_coords = np.meshgrid(*coordinate_vectors)

        # grid_coords[0] = x_pts = [x1, ..., xN, x1, ..., xN, ..., x1, ..., xN], shape (N^2, 1)
        # grid_coords[1] = y_pts = [y1, ..., yN, y1, ..., yN, ..., y1, ..., yN], shape (N^2, 1)
        # ...
        reshaped_columns = [arr.reshape(-1, 1) for arr in grid_coords]

        # points = [(x1, y1, ...), ..., (x[N^2], y[N^2], ...)], shape (N^2, 2)
        points = np.column_stack(reshaped_columns)

    else: # mode == 'random'
        vectors = [np.random.rand(n_rand_points) * (r[1] - r[0]) + r[0] for r in ranges]
        points = np.column_stack(vectors)

    pde = Pde(name=pde_name, params=pde_params)
    points = torch.tensor(points, dtype=torch.float32)
    u_vals = pde.solution(points).reshape((-1, 1)) # (N^2, 1)
    pdv_vals = pde.der[1](points) # (N^2, 2) (ux, uy)
    hes_vals = pde.der[2](points) # (N^2, 2x2) (uxx, uxy,
                                  #             uxy, uyy)
    force_vals = pde.force(points) # (N^2, 1)
    params_vals = torch.tensor(pde_params).repeat(len(points), 1)

    print(f'u_vals.shape: {u_vals.shape}')
    print(f'points.shape: {points.shape}')
    print(f'pdv_vals.shape: {pdv_vals.shape}')
    print(f'hes_vals.shape: {hes_vals.shape}')
    print(f'force_vals.shape: {force_vals.shape}')
    print(f'params_vals.shape: {params_vals.shape}')

    return points, u_vals, pdv_vals, hes_vals, force_vals, params_vals

# Domain generation ----------------------------------------------------------------------------------
def generate_domain(mode: str, ranges: list, pde_name: str, pde_params: list, steps: list = [], n_rand_points: int = 1000, filename=''):
    points, u_vals, pdv_vals, hes_vals, force_vals, params_vals = generate(
        mode=mode,
        ranges=ranges,
        pde_name=pde_name,
        pde_params=pde_params,
        steps=steps,
        n_rand_points=n_rand_points
    )

    dataset = TensorDataset(points, u_vals, pdv_vals, hes_vals, force_vals, params_vals)
    
    # save the full dataset
    torch.save(dataset, filename)

    return points, u_vals, pdv_vals, hes_vals, force_vals, params_vals

# Boundary generation ----------------------------------------------------------------------------------
def generate_boundary(mode: str, ranges: list, pde_name: str, pde_params: list, steps: list = [], n_rand_points: int = 1000, filename=''):
    bd_points = []
    bd_u_vals = []
    bd_pdv_vals = []
    bd_hes_vals = []
    bd_force_vals = []
    bd_params_vals = []
    for i in range(len(ranges)):
        xmin = ranges[i][0]
        xmax = ranges[i][1]
        points, u_vals, pdv_vals, hes_vals, force_vals, params_vals = generate(
            mode=mode,
            ranges=ranges[:i] + [[xmin, xmin]] + ranges[i+1:],
            pde_name=pde_name,
            pde_params=pde_params,
            steps=steps,
            n_rand_points=n_rand_points
            )
        bd_points.append(points)
        bd_u_vals.append(u_vals)
        bd_pdv_vals.append(pdv_vals)
        bd_hes_vals.append(hes_vals)
        bd_force_vals.append(force_vals)
        bd_params_vals.append(params_vals)
        
        points, u_vals, pdv_vals, hes_vals, force_vals, params_vals = generate(
            mode=mode,
            ranges=ranges[:i] + [[xmax, xmax]] + ranges[i+1:],
            pde_name=pde_name,
            pde_params=pde_params,
            steps=steps,
            n_rand_points=n_rand_points
            )
        bd_points.append(points)
        bd_u_vals.append(u_vals)
        bd_pdv_vals.append(pdv_vals)
        bd_hes_vals.append(hes_vals)
        bd_force_vals.append(force_vals)
        bd_params_vals.append(params_vals)

    points = torch.cat(bd_points)
    u_vals = torch.cat(bd_u_vals)
    pdv_vals = torch.cat(bd_pdv_vals)
    hes_vals = torch.cat(bd_hes_vals)
    force_vals = torch.cat(bd_force_vals)
    params_vals = torch.cat(bd_params_vals)

    dataset = TensorDataset(points, u_vals, pdv_vals, hes_vals, force_vals, params_vals)
    
    # save the full dataset        
    torch.save(dataset, filename)

    return points, u_vals, pdv_vals, hes_vals, force_vals, params_vals


# Store a portion of the full dataset ------------------------------------------------------------
def store_subset(pts, u_vals, pdv_vals, hes_vals, force_vals, params_vals, ranges, filename):
    mask = torch.ones(len(pts), dtype=bool)
    for i in range(len(ranges)):
        xmin = ranges[i][0]
        xmax = ranges[i][1]
        mask = mask & (pts[:, i] > xmin) & (pts[:, i] < xmax)

    subset_pts = pts[mask]
    subset_u = u_vals[mask]
    subset_pdv = pdv_vals[mask]
    subset_hes = hes_vals[mask]
    subset_force = force_vals[mask]
    subset_params = params_vals[mask]

    # make the TensorDataset
    sub_dataset = TensorDataset(
        subset_pts,
        subset_u,
        subset_pdv,
        subset_hes,
        subset_force,
        subset_params
        )
    
    # save the subset dataset
    torch.save(sub_dataset, filename)

    return subset_pts, subset_u, subset_pdv, subset_hes, subset_force, subset_params

# ==========================================================================================

if __name__ == "__main__":
    # Init parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_gen.yaml', type=str, help='Path to the configuration file (YAML)')

    # Parse command-line arguments
    cli_args = parser.parse_args()

    # Load configuration from YAML file
    config_params = {}
    if os.path.exists(cli_args.config):
        with open(cli_args.config, 'r') as f:
            config_params = yaml.safe_load(f)
    else:
        print(f"Error: Config file '{cli_args.config}' not found.\nUse generate.py --config <config_file_name>.yaml")
        sys.exit(1)

    def get_param(config_section_dict, config_key, default_val=None, type_func=None):   
        # Try to get from config file
        if config_section_dict and config_key in config_section_dict:
            val = config_section_dict[config_key]
            return type_func(val) if type_func else val

        # Use default
        else:
            return default_val

    # Get the section    
    gen_config = config_params.get('generation', {})
    subsets_to_store = config_params.get('subsets', {})
    options = config_params.get('options', {})

    # Get the seed
    seed = get_param(gen_config, 'seed', default_val=30, type_func=int)

    # Set the seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Get the data directory
    data_dir = get_param(options, 'data_dir', default_val='data')

    # create the directory 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    params_id_list = get_param(options, 'params_id', default_val=[0], type_func=list)
    for params_id in params_id_list:
        if not os.path.exists(f'{data_dir}/{params_id}'):
            os.makedirs(f'{data_dir}/{params_id}')

    pde_name = get_param(gen_config, 'pde_name', default_val='')

    pde_params_list_str = get_param(gen_config, 'pde_params', default_val=[], type_func=list)
    pde_params_list = []
    for pde_params_str in pde_params_list_str:
        pde_params_list.append([float(s) for s in list(pde_params_str)])

    dom_mode = get_param(gen_config, 'dom_mode', default_val='grid')
    bd_mode = get_param(gen_config, 'bd_mode', default_val='grid')
    boundary = get_param(gen_config, 'boundary', default_val=True, type_func=bool)
    domain = get_param(gen_config, 'domain', default_val=True, type_func=bool)

    steps_str = get_param(gen_config, 'dom_steps', default_val=[], type_func=list)
    dom_steps = [float(s) for s in steps_str]

    steps_str = get_param(gen_config, 'bd_steps', default_val=[], type_func=list)
    bd_steps = [float(s) for s in steps_str]

    n_rand_bound = get_param(gen_config, 'n_rand_bound', default_val=1000, type_func=int)
    n_rand_dom = get_param(gen_config, 'n_rand_dom', default_val=1000, type_func=int)
    n_rand_bound_sub = get_param(gen_config, 'n_rand_bound_sub', default_val=1000, type_func=int)

    ranges_dict = gen_config['ranges']
    ranges = []
    for key in gen_config['ranges'].keys():
        r = list(ranges_dict[key])
        ranges.append([float(r[0]), float(r[1])])
    
    for i in range(len(pde_params_list)):
        pde_params = pde_params_list[i]
        params_id = params_id_list[i]

        if not os.path.exists(f'{data_dir}/{params_id}/full'):
            os.makedirs(f'{data_dir}/{params_id}/full')

        if boundary:
            print('\nGenerating boundary ...')            
            generate_boundary(mode=bd_mode, ranges=ranges, pde_name=pde_name, pde_params=pde_params, steps=bd_steps, n_rand_points=n_rand_bound, filename=f'{data_dir}/{params_id}/full/{bd_mode}_bd.pth')

        if domain:
            print('\nGenerating domain ...')
            pts, u_vals, pdv_vals, hes_vals, force_vals, params_vals = generate_domain(mode=dom_mode, ranges=ranges, pde_name=pde_name, pde_params=pde_params, steps=dom_steps, n_rand_points=n_rand_dom, filename=f'{data_dir}/{params_id}/full/{dom_mode}_dom.pth')

        for subset in subsets_to_store:
            ranges_sub = []
            for r_str in list(subsets_to_store[subset]):
                r = list(r_str)
                ranges_sub.append([float(r[0]), float(r[1])])

            if not os.path.exists(f'{data_dir}/{params_id}/{subset}'):
                os.makedirs(f'{data_dir}/{params_id}/{subset}')

            print(f'\nGenerating {subset}...')

            generate_boundary(mode=bd_mode, ranges=ranges_sub, pde_name=pde_name, pde_params=pde_params, steps=bd_steps, n_rand_points=n_rand_bound_sub, filename=f'{data_dir}/{params_id}/{subset}/{bd_mode}_bd.pth')

            store_subset(pts=pts, u_vals=u_vals, pdv_vals=pdv_vals, hes_vals=hes_vals, force_vals=force_vals, params_vals=params_vals, ranges=ranges_sub, filename=f'{data_dir}/{params_id}/{subset}/{dom_mode}_dom.pth')