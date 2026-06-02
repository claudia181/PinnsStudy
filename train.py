"""
train.py
===========

This module contains:
- start_train (*)
- get_param (used by start_train)

This module implements the interface to start a training process under given configurations.
It is possible to pass the training configurations through a dictionary argument to start_train.
"""

import torch
import argparse
import numpy as np
import random
import os
import yaml
import optuna
from optuna.trial import TrialState
import shutil
import copy

from optuna_objective import Objective
from model import PdeNet
from data_utils import subsample, replace_labels, extract_TensorDataset, extract_boundary, extract_interior, include_time_in_input, exclude_space_in_input
from load_store_utils import resume_model
from generate import X, U

# ============================================ Function to get value ============================================
def get_param(config_section_dict, config_key, default_val=None, type_func=None):
    # Try to get from config file
    if config_section_dict and config_key in config_section_dict:
        val = config_section_dict[config_key]
        return type_func(val) if type_func else val
    
    # Use default
    else:
        return default_val

# =============================================== Run configuration ===============================================
def start_train(config: dict|str):
    if type(config) is str:
        # Load configuration from YAML file
        config_dict = {}
        if os.path.exists(config):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Config file '{config}' not found.")
    else:
        config_dict = config


    # Get config sections
    # PDE considered
    pde_name = get_param(config_dict, "PDE", type_func=str)
    if pde_name is None:
        raise ValueError("Missing PDE information.")
    
    actual_dict     = config_dict.get("Actual", {})
    unlabeled_dict  = config_dict.get("Unlabeled", {})
    distill_dict    = config_dict.get("Distillation", {})
    ewc_dict        = config_dict.get("EWC", {})
    dwa_dict        = config_dict.get("DWA", {})

    actual_mode                 = get_param(actual_dict, "mode", default_val="Output")
    actual_model                = get_param(actual_dict, "model", default_val="") # path
    layers                      = [int(layer) for layer in get_param(actual_dict, "layers", default_val=[50, 50, 50, 50], type_func=list)]
    time_in_input               = get_param(actual_dict, "time_in_input", default_val=False, type_func=bool)
    space_in_input              = get_param(actual_dict, "space_in_input", default_val=True, type_func=bool)
    fourier_features            = get_param(actual_dict, "fourier_features", default_val=-1)
    frequency_variance          = get_param(actual_dict, "frequency_variance", default_val=1.0)
    pde_params_to_take_keys     = get_param(actual_dict, "pde_params_in_input", default_val=[], type_func=list)
    ic_params_to_take_keys      = get_param(actual_dict, "ic_params_in_input", default_val=[], type_func=list)
    train_dataset               = get_param(actual_dict, "train_dataset", default_val="") # path
    val_dataset                 = get_param(actual_dict, "val_dataset", default_val="") # path
    actual_subset               = get_param(actual_dict, "subset", default_val={})
    actual_shape                = get_param(actual_dict, "shape", default_val={"shape": "rectangle", "cell_size": None, "center": None, "radius": None})
    boundary_mode               = get_param(actual_dict, "boundary")
    bc_mode                     = get_param(actual_dict, "BC", type_func=str, default_val="Dirichlet")
    pde_at_bd                   = get_param(actual_dict, "pde_at_bd", type_func=bool, default_val=True)
    pde_at_t0                   = get_param(actual_dict, "pde_at_t0", type_func=bool, default_val=True)
    initial_time_mode           = get_param(actual_dict, "initial_time") # Excluded, Separated, Joined
    actual_bc_weight            = get_param(actual_dict, "bc_weight", default_val=1.0, type_func=float)
    actual_ic_weight            = get_param(actual_dict, "ic_weight", default_val=1.0, type_func=float)
    actual_out_weight           = get_param(actual_dict, "out_weight", default_val=1.0, type_func=float)
    actual_der_weight           = get_param(actual_dict, "der_weight", default_val=1.0, type_func=float)
    actual_dert_weight          = get_param(actual_dict, "dert_weight", default_val=1.0, type_func=float)
    actual_derx_weight          = get_param(actual_dict, "derx_weight", default_val=1.0, type_func=float)
    actual_hes_weight           = get_param(actual_dict, "hes_weight", default_val=1.0, type_func=float)
    actual_hesx_weight          = get_param(actual_dict, "hesx_weight", default_val=1.0, type_func=float)
    actual_hest_weight          = get_param(actual_dict, "hest_weight", default_val=1.0, type_func=float)
    actual_res_weight           = get_param(actual_dict, "res_weight", default_val=1.0, type_func=float)
    monitor_conflicts           = get_param(actual_dict, "monitor_conflicts", default_val=False, type_func=bool)
    
    nl_dataset                  = get_param(unlabeled_dict, "dataset", default_val="") # path
    nl_subset                   = get_param(unlabeled_dict, "subset", default_val={})
    memory_buffer_size_nl       = get_param(unlabeled_dict, "buffer_size", default_val=None)
    memory_buffer_size_nl_bc    = get_param(unlabeled_dict, "buffer_size_bc", default_val=None)
    memory_buffer_size_nl_ic    = get_param(unlabeled_dict, "buffer_size_ic", default_val=None)
    nl_weight                   = get_param(unlabeled_dict, "weight", default_val=1.0, type_func=float)
    nl_bc_weight                = get_param(unlabeled_dict, "bc_weight", default_val=1.0, type_func=float)
    nl_ic_weight                = get_param(unlabeled_dict, "ic_weight", default_val=1.0, type_func=float)
    nl_shape                    = get_param(unlabeled_dict, "shape", default_val={"shape": "rectangle", "cell_size": None, "center": None, "radius": None})
    boundary_mode_nl            = get_param(unlabeled_dict, "boundary")
    initial_time_mode_nl        = get_param(unlabeled_dict, "initial_time")

    distill_mode                = get_param(distill_dict, "mode", default_val="Forgetting")
    distill_model               = get_param(distill_dict, "model", default_val="") # path
    distill_dataset             = get_param(distill_dict, "dataset", default_val="") # path
    distill_subset              = get_param(distill_dict, "subset", default_val={})
    memory_buffer_size_distill  = get_param(distill_dict, "buffer_size", default_val=None)
    distill_out_weight          = get_param(distill_dict, "out_weight", default_val=1.0, type_func=float)
    distill_der_weight          = get_param(distill_dict, "der_weight", default_val=1.0, type_func=float)
    distill_derx_weight         = get_param(distill_dict, "derx_weight", default_val=1.0, type_func=float)
    distill_dert_weight         = get_param(distill_dict, "dert_weight", default_val=1.0, type_func=float)
    distill_hes_weight          = get_param(distill_dict, "hes_weight", default_val=1.0, type_func=float)
    distill_hesx_weight         = get_param(distill_dict, "hesx_weight", default_val=1.0, type_func=float)
    distill_hest_weight         = get_param(distill_dict, "hest_weight", default_val=1.0, type_func=float)

    ewc_mode                = get_param(ewc_dict, "mode", default_val="Off")
    ewc_model               = get_param(ewc_dict, "model", default_val="") # path
    ewc_dataset             = get_param(ewc_dict, "dataset", default_val="") # path
    ewc_subset              = get_param(ewc_dict, "subset", default_val={})
    memory_buffer_size_ewc  = get_param(ewc_dict, "buffer_size", default_val=None)
    ewc_weight              = get_param(ewc_dict, "weight", default_val=1.0, type_func=float)
    ewc_auto_weighting      = get_param(ewc_dict, "auto_weighting", default_val=False, type_func=bool)
    ewc_warm_up             = get_param(ewc_dict, "warm_up", default_val=0, type_func=int)
    ewc_decay_factor        = get_param(ewc_dict, "decay", default_val=1.0)
    ewc_src_fisher_file     = get_param(ewc_dict, "src_fisher_diag_file", default_val="")
    ewc_dst_fisher_file     = get_param(ewc_dict, "dst_fisher_diag_file", default_val="")
    ewc_moving_avg_factor   = get_param(ewc_dict, "moving_avg_factor", default_val=0.5, type_func=float)

    dwa_mode                    = get_param(dwa_dict, "mode", default_val="Off") # Off, Std, Norm1, NormK
    dwa_moving_avg_factor       = get_param(dwa_dict, "moving_avg_factor", default_val=0.9, type_func=float)
    dwa_moving_avg_frequency    = get_param(dwa_dict, "moving_avg_frequency", default_val=1, type_func=int)
    dwa_warm_up                 = get_param(dwa_dict, "warm_up", default_val=0, type_func=int)
    delayed_weights             = get_param(dwa_dict, "delayed_weights", type_func=bool, default_val=False)
    
    pruner              = get_param(config_dict, "pruner", default_val="median", type_func=str)
    threshold           = get_param(config_dict, "threshold", default_val=1.0, type_func=float)
    n_warmup_steps      = get_param(config_dict, "n_warmup_steps", default_val=0, type_func=int)
    n_trials            = get_param(config_dict, "n_trials", default_val=10, type_func=int)
    train_steps         = get_param(config_dict, "train_steps", default_val=-1, type_func=int)
    epochs              = get_param(config_dict, "epochs", default_val=100, type_func=int)
    early_stop_value    = get_param(config_dict, "early_stop_value", default_val=None, type_func=float)
    eval_every          = get_param(config_dict, "eval_every", default_val=1, type_func=int)
    seed                = get_param(config_dict, "seed", default_val=42, type_func=int)
    device              = get_param(config_dict, "device", default_val="cpu") #cuda:0
    lr_init             = get_param(config_dict, "learning_rate")
    scheduler           = get_param(config_dict, "scheduler", type_func=str, default_val=None)
    batch_size          = get_param(config_dict, "batch_size")
    clip_grad           = get_param(config_dict, "clip_grad", type_func=bool, default_val=True)
    models_dir          = get_param(config_dict, "models_dir", default_val=f"./{pde_name.replace('-', '')}/models")
    suggestions         = get_param(config_dict, "suggestions", default_val="On")

    if lr_init is None or (type(lr_init) is list and lr_init == []):
        lr_init = [1e-3]
    if type(lr_init) is list:
        lr_init = [float(el) for el in lr_init]
    else:
        lr_init = [float(lr_init)]
    
    if batch_size is None or (type(batch_size) is list and batch_size == []):
        batch_size = [1024]
    if type(batch_size) is list:
        batch_size = [int(el) for el in batch_size]
    else:
        batch_size = [int(batch_size)]

    if fourier_features is None or (type(fourier_features) is list and fourier_features == []):
        fourier_features = [-1]
    if type(fourier_features) is list:
        fourier_features = [int(el) for el in fourier_features]
    else:
        fourier_features = [int(fourier_features)]
    
    if frequency_variance is None or (type(frequency_variance) is list and frequency_variance == []):
        frequency_variance = [1.0]
    if type(frequency_variance) is not list:
        frequency_variance = [frequency_variance]
    
    if scheduler == "":
        scheduler = None
    
    if scheduler is None:
        if actual_mode == "Output":
            scheduler = False
        else:
            scheduler = True
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Loading datasets

    if type(actual_subset) is dict:
        actual_subset = [actual_subset]
    if type(nl_subset) is dict:
        nl_subset = [nl_subset]
    if type(distill_subset) is dict:
        distill_subset = [distill_subset]
    if type(ewc_subset) is dict:
        ewc_subset = [ewc_subset]

    if actual_shape["shape"] == "rectangle":
        actual_shape["cell_size"] = None
        actual_shape["center"] = None
        actual_shape["radius"] = None
    elif actual_shape["shape"] == "circle":
        if "cell_size" not in actual_shape.keys():
            raise ValueError(f"Missing argument 'cell_size' for domain shape 'circle'.")
        if "center" not in actual_shape.keys():
            raise ValueError(f"Missing argument 'center' for domain shape 'circle'.")
        if "radius" not in actual_shape.keys():
            raise ValueError(f"Missing argument 'radius' for domain shape 'circle'.")
    else:
        raise ValueError(f"Not valid shape '{actual_shape['shape']}'.")
    
    if nl_shape["shape"] == "rectangle":
        nl_shape["cell_size"] = None
        nl_shape["center"] = None
        nl_shape["radius"] = None
    elif nl_shape["shape"] == "circle":
        if "cell_size" not in nl_shape.keys():
            raise ValueError(f"Missing argument 'cell_size' for domain shape 'circle'.")
        if "center" not in nl_shape.keys():
            raise ValueError(f"Missing argument 'center' for domain shape 'circle'.")
        if "radius" not in nl_shape.keys():
            raise ValueError(f"Missing argument 'radius' for domain shape 'circle'.")
    else:
        raise ValueError(f"Not valid shape '{nl_shape['shape']}'.")

    if type(train_dataset) is str:
        if train_dataset == "":
            raise ValueError("Required train_dataset arg not specified.")
        if not os.path.exists(train_dataset):
            raise ValueError(f"Dataset file '{train_dataset}' not found.")
        train_dataset = torch.load(train_dataset, weights_only=False)

    #if input_units is None:
    input_units = space_in_input * len(train_dataset.datasets[0][0][X]) + time_in_input + len(pde_params_to_take_keys) + len(ic_params_to_take_keys)
    if type(val_dataset) is str:
        if val_dataset == "":
            val_dataset = None
        elif not os.path.exists(val_dataset):
            raise ValueError(f"Dataset file '{val_dataset}' not found.")
        else:
            val_dataset = torch.load(val_dataset, weights_only=False)

    t = actual_subset[0].get("t")
    t_points = actual_subset[0].get("t_points")
    if t is None:
        t = [0, len(train_dataset.datasets)-1]
    if t_points is None:
        t_points = np.arange(start=t[0], stop=t[1]+1, step=1)
    else:
        t_points = [i for i in t_points if i >= t[0] and i <= t[1]]

    if initial_time_mode is None or len(train_dataset.datasets) == 1:
        if actual_mode == "Output" or len(train_dataset.datasets) == 1:
            initial_time_mode = "Joined"
        else:
            raise ValueError("Missing initial mode.")
    if initial_time_mode == "Excluded":
        t_points = t_points[1:]
        train_initial = None
        val_initial = None
        train_dataset = extract_TensorDataset(train_dataset, time_indexes=t_points[1:], spatial_ranges=actual_subset)
        val_dataset = extract_TensorDataset(val_dataset, time_indexes=t_points[1:], spatial_ranges=actual_subset)
    elif initial_time_mode == "Joined":
        train_initial = None
        val_initial = None
        train_dataset = extract_TensorDataset(train_dataset, time_indexes=t_points, spatial_ranges=actual_subset)
        val_dataset = extract_TensorDataset(val_dataset, time_indexes=t_points, spatial_ranges=actual_subset)
    else: # initial_time_mode == "Separated"
        train_initial = extract_TensorDataset(train_dataset, time_indexes=[t_points[0]], spatial_ranges=actual_subset)
        val_initial = extract_TensorDataset(val_dataset, time_indexes=[t_points[0]], spatial_ranges=actual_subset)

        if pde_at_t0:
            train_dataset = extract_TensorDataset(train_dataset, time_indexes=t_points, spatial_ranges=actual_subset)
            val_dataset = extract_TensorDataset(val_dataset, time_indexes=t_points, spatial_ranges=actual_subset)
        else:
            train_dataset = extract_TensorDataset(train_dataset, time_indexes=t_points[1:], spatial_ranges=actual_subset)
            val_dataset = extract_TensorDataset(val_dataset, time_indexes=t_points[1:], spatial_ranges=actual_subset)

    if boundary_mode is None and space_in_input:
        raise ValueError("Missing boundary mode.")
        
    if boundary_mode is not None:
        if boundary_mode == "Global":
            train_boundary = extract_boundary(
                dataset=train_dataset,
                shape=actual_shape["shape"],
                cell_size=actual_shape["cell_size"],
                center=actual_shape["center"],
                radius=actual_shape["radius"]
            )
            val_boundary = extract_boundary(
                dataset=val_dataset,
                shape=actual_shape["shape"],
                cell_size=actual_shape["cell_size"],
                center=actual_shape["center"],
                radius=actual_shape["radius"]
            )
            if not pde_at_bd:
                train_interior = extract_interior(
                    dataset=train_dataset,
                    shape=actual_shape["shape"],
                    cell_size=actual_shape["cell_size"],
                    center=actual_shape["center"],
                    radius=actual_shape["radius"]
                )
                val_interior = extract_interior(
                    dataset=val_dataset,
                    shape=actual_shape["shape"],
                    cell_size=actual_shape["cell_size"],
                    center=actual_shape["center"],
                    radius=actual_shape["radius"]
                )

                train_dataset = extract_TensorDataset(train_interior, spatial_ranges=actual_subset)
                val_dataset = extract_TensorDataset(val_interior, spatial_ranges=actual_subset)
            else:
                train_dataset = extract_TensorDataset(train_dataset, spatial_ranges=actual_subset)
                val_dataset = extract_TensorDataset(val_dataset, spatial_ranges=actual_subset)

        elif "Local" in boundary_mode:
            train_dataset = extract_TensorDataset(train_dataset, spatial_ranges=actual_subset)
            val_dataset = extract_TensorDataset(val_dataset, spatial_ranges=actual_subset)

            train_boundary = extract_boundary(
                dataset=train_dataset,
                shape=actual_shape["shape"],
                cell_size=actual_shape["cell_size"],
                center=actual_shape["center"],
                radius=actual_shape["radius"]
            )
            val_boundary = extract_boundary(
                dataset=val_dataset,
                shape=actual_shape["shape"],
                cell_size=actual_shape["cell_size"],
                center=actual_shape["center"],
                radius=actual_shape["radius"]
            )

            if not pde_at_bd:
                train_interior = extract_interior(
                    dataset=train_dataset,
                    shape=actual_shape["shape"],
                    cell_size=actual_shape["cell_size"],
                    center=actual_shape["center"],
                    radius=actual_shape["radius"]
                )
                val_interior = extract_interior(
                    dataset=val_dataset,
                    shape=actual_shape["shape"],
                    cell_size=actual_shape["cell_size"],
                    center=actual_shape["center"],
                    radius=actual_shape["radius"]
                )
                train_dataset = extract_TensorDataset(train_interior, spatial_ranges=actual_subset)
                val_dataset = extract_TensorDataset(val_interior, spatial_ranges=actual_subset)

            train_boundary = extract_TensorDataset(train_boundary, spatial_ranges=actual_subset)
            val_boundary = extract_TensorDataset(val_boundary, spatial_ranges=actual_subset)
    else:
        train_dataset = extract_TensorDataset(train_interior, spatial_ranges=actual_subset)
        val_dataset = extract_TensorDataset(val_interior, spatial_ranges=actual_subset)
        train_boundary = None
        val_boundary = None

    if type(nl_dataset) is str:
        if nl_dataset == "":
            nl_dataset = None
        elif not os.path.exists(nl_dataset):
            raise ValueError(f"Dataset file '{nl_dataset}' not found.")
        else:
            nl_dataset = torch.load(nl_dataset, weights_only=False)

    if nl_dataset is not None:
        t = nl_subset[0].get("t")
        t_points = nl_subset[0].get("t_points")
        if t is None:
            t = [0, len(nl_dataset.datasets)-1]
        if t_points is None:
            t_points = np.arange(start=t[0], stop=t[1]+1, step=1)
        else:
            t_points = [i for i in t_points if i >= t[0] and i <= t[1]]

        if initial_time_mode_nl is None and len(nl_dataset.datasets) == 1:
            initial_time_mode_nl = "Joined"
        elif initial_time_mode_nl is None:
            raise ValueError("Missing nl initial mode.")

        if initial_time_mode_nl == "Excluded":
            t_points = t_points[1:]
            nl_initial = None
            nl_dataset = extract_TensorDataset(nl_dataset, time_indexes=t_points[1:], spatial_ranges=nl_subset)
        elif initial_time_mode_nl == "Joined":
            nl_initial = None
            nl_dataset = extract_TensorDataset(nl_dataset, time_indexes=t_points, spatial_ranges=nl_subset)
        else: # initial_time_mode_nl == "Separated"
            nl_initial = extract_TensorDataset(nl_dataset, time_indexes=[t_points[0]], spatial_ranges=nl_subset)
            if pde_at_t0:
                nl_dataset = extract_TensorDataset(nl_dataset, time_indexes=t_points, spatial_ranges=nl_subset)
            else:
                nl_dataset = extract_TensorDataset(nl_dataset, time_indexes=t_points[1:], spatial_ranges=nl_subset)

        if boundary_mode_nl is not None:
            if boundary_mode_nl == "Global":
                nl_boundary = extract_boundary(
                    dataset=nl_dataset,
                    shape=nl_shape["shape"],
                    cell_size=nl_shape["cell_size"],
                    center=nl_shape["center"],
                    radius=nl_shape["radius"]
                )
                if not pde_at_bd:
                    nl_interior = extract_interior(
                        dataset=nl_dataset,
                        shape=nl_shape["shape"],
                        cell_size=nl_shape["cell_size"],
                        center=nl_shape["center"],
                        radius=nl_shape["radius"]
                    )
                    nl_dataset = extract_TensorDataset(nl_interior, spatial_ranges=nl_subset)
                else:
                    nl_dataset = extract_TensorDataset(nl_dataset, spatial_ranges=nl_subset)
            elif "Local" in boundary_mode_nl:
                nl_dataset = extract_TensorDataset(nl_dataset, spatial_ranges=nl_subset)
                nl_boundary = extract_boundary(
                    dataset=nl_dataset,
                    shape=nl_shape["shape"],
                    cell_size=nl_shape["cell_size"],
                    center=nl_shape["center"],
                    radius=nl_shape["radius"]
                )
                if not pde_at_bd:
                    nl_interior = extract_interior(
                        dataset=nl_dataset,
                        shape=nl_shape["shape"],
                        cell_size=nl_shape["cell_size"],
                        center=nl_shape["center"],
                        radius=nl_shape["radius"]
                    )
                    nl_dataset = extract_TensorDataset(nl_interior, spatial_ranges=nl_subset)
                nl_boundary = extract_TensorDataset(nl_boundary, spatial_ranges=nl_subset)
        else:
            nl_dataset = extract_TensorDataset(nl_dataset, spatial_ranges=nl_subset)
            nl_boundary = None
    else:
        nl_boundary = None
        nl_initial = None
        
    if nl_dataset is not None and memory_buffer_size_nl is not None and len(nl_dataset) > memory_buffer_size_nl:
        nl_dataset = subsample(
            dataset=nl_dataset,
            n_samples=memory_buffer_size_nl,
            seed=seed
        )
    if nl_boundary is not None and memory_buffer_size_nl_bc is not None and len(nl_boundary) > memory_buffer_size_nl_bc:
        nl_boundary = subsample(
            dataset=nl_boundary,
            n_samples=memory_buffer_size_nl_bc,
            seed=seed
        )
    if nl_initial is not None and memory_buffer_size_nl_ic is not None and len(nl_initial) > memory_buffer_size_nl_ic:
        nl_initial = subsample(
            dataset=nl_initial,
            n_samples=memory_buffer_size_nl_ic,
            seed=seed
        )
    if nl_dataset is not None:
        print(f"Unlabeled subsample size: {len(nl_dataset)}")
    if nl_boundary is not None:
        print(f"Unlabeled subsample bd size: {len(nl_boundary)}")
    if nl_initial is not None:
        print(f"Unlabeled subsample ic size: {len(nl_initial)}")

    t = distill_subset[0].get("t")
    t_points = distill_subset[0].get("t_points")
    if t is not None:
        if t_points is None:
            t_points = np.arange(start=t[0], stop=t[1]+1, step=1)
        else:
            t_points = [i for i in t_points if i >= t[0] and i <= t[1]]

    if type(distill_dataset) is str:
        if distill_dataset == "":
            distill_dataset = None
        elif not os.path.exists(distill_dataset):
            raise ValueError(f"Dataset file '{distill_dataset}' not found.")
        else:
            distill_dataset = torch.load(distill_dataset, weights_only=False)
            distill_dataset = extract_TensorDataset(distill_dataset, time_indexes=t_points, spatial_ranges=distill_subset)
    else:
        distill_dataset = extract_TensorDataset(distill_dataset, time_indexes=t_points, spatial_ranges=distill_subset)
    
    if distill_dataset is not None and memory_buffer_size_distill is not None and len(distill_dataset) > memory_buffer_size_distill:
        distill_dataset = subsample(
            dataset=distill_dataset,
            n_samples=memory_buffer_size_distill,
            seed=seed
        )
    if distill_dataset is not None:
        print(f"Distillation subsample size: {len(distill_dataset)}")


    if distill_model == "" or distill_dataset is None:
        distill_model = None
    elif not os.path.exists(distill_model):
        raise ValueError(f"Model file '{distill_model}' not found.")
    if distill_model is not None:
        distill_model = resume_model(
            model_path=distill_model,
            device=device
            )
        temp_distill_dataset = distill_dataset
        if distill_model.time_in_input:
            temp_distill_dataset = include_time_in_input(distill_dataset)
        if not distill_model.space_in_input:
            temp_distill_dataset = exclude_space_in_input(temp_distill_dataset)
        temp_distill_dataset = distill_model.label(temp_distill_dataset)
        distill_dataset = replace_labels(distill_dataset, temp_distill_dataset.tensors[U])

    t = ewc_subset[0].get("t")
    t_points = ewc_subset[0].get("t_points")
    if t is not None:
        if t_points is None:
            t_points = np.arange(start=t[0], stop=t[1]+1, step=1)
        else:
            t_points = [i for i in t_points if i >= t[0] and i <= t[1]]

    if type(ewc_dataset) is str:
        if ewc_dataset == "":
            ewc_dataset = None
        elif not os.path.exists(ewc_dataset):
            raise ValueError(f"Dataset file '{ewc_dataset}' not found.")
        else:
            ewc_dataset = torch.load(ewc_dataset, weights_only=False)
            ewc_dataset = extract_TensorDataset(ewc_dataset, time_indexes=t_points, spatial_ranges=ewc_subset)
    else:
        ewc_dataset = extract_TensorDataset(ewc_dataset, time_indexes=t_points, spatial_ranges=ewc_subset)

    if ewc_dataset is not None and memory_buffer_size_ewc is not None and len(ewc_dataset) > memory_buffer_size_ewc:
        ewc_dataset = subsample(
            dataset=ewc_dataset,
            n_samples=memory_buffer_size_ewc,
            seed=seed
        )
    if ewc_dataset is not None:
        print(f"EWC subsample size: {len(ewc_dataset)}")


     # EWC
    if ewc_model == "" or ewc_dataset is None:
        ewc_model = None
        if ewc_mode == "On":
            raise ValueError(f"EWC on but missing information: ewc_model = {ewc_model}, ewc_dataset = {ewc_dataset}.")
    elif not os.path.exists(ewc_model):
        raise ValueError(f"Model file '{ewc_model}' not found.")
    elif ewc_dataset is not None:
        ewc_model = resume_model(model_path=ewc_model, device=device, dwa_mode="Off", distill_mode="Forgetting", ewc_mode="Off")
        if ewc_model.time_in_input:
            ewc_dataset = include_time_in_input(ewc_dataset)
        if not ewc_model.space_in_input:
            ewc_dataset = exclude_space_in_input(ewc_dataset)
        temp_ewc_dataset = ewc_model.label(ewc_dataset)
        ewc_dataset = replace_labels(ewc_dataset, temp_ewc_dataset.tensors[U])

    if actual_model != "":
        if not os.path.exists(actual_model):
            raise ValueError(f"Model file '{actual_model}' not found.") 
        actual_model = resume_model(
            model_path=actual_model,
            device=device,
            sys_mode=actual_mode,
            distill_mode=distill_mode,
            ewc_mode=ewc_mode,
            dwa_mode=dwa_mode,
            bc_mode=bc_mode,
            monitor_conflicts=monitor_conflicts,
            alpha=dwa_moving_avg_factor,
            weighted_avg_frequency=dwa_moving_avg_frequency,
            dwa_warm_up=dwa_warm_up,
            bc_weight=actual_bc_weight,
            ic_weight=actual_ic_weight,
            out_weight=actual_out_weight,
            der_weight=actual_der_weight,
            derx_weight=actual_derx_weight,
            dert_weight=actual_dert_weight,
            hes_weight=actual_hes_weight,
            hesx_weight=actual_hesx_weight,
            hest_weight=actual_hest_weight,
            res_weight=actual_res_weight,
            nl_bc_weight=nl_bc_weight,
            nl_ic_weight=nl_ic_weight,
            nl_weight=nl_weight,
            distill_out_weight=distill_out_weight,
            distill_der_weight=distill_der_weight,
            distill_derx_weight=distill_derx_weight,
            distill_dert_weight=distill_dert_weight,
            distill_hes_weight=distill_hes_weight,
            distill_hesx_weight=distill_hesx_weight,
            distill_hest_weight=distill_hest_weight,
            ewc_weight=ewc_weight,
            ewc_auto_weighting=ewc_auto_weighting,
            ewc_warm_up=ewc_warm_up,
            ewc_decay=ewc_decay_factor
            ).to(device)
    else:
        actual_model = PdeNet(
            pde=pde_name,
            time_in_input=time_in_input,
            space_in_input=space_in_input,
            #fourier_features=fourier_features,
            pde_params_in_input=pde_params_to_take_keys,
            ic_params_in_input=ic_params_to_take_keys,
            sys_mode=actual_mode,
            bc_weight=actual_bc_weight,
            ic_weight=actual_ic_weight,
            out_weight=actual_out_weight,
            der_weight=actual_der_weight,
            derx_weight=actual_derx_weight,
            dert_weight=actual_dert_weight,
            hes_weight=actual_hes_weight,
            hesx_weight=actual_hesx_weight,
            hest_weight=actual_hest_weight,
            res_weight=actual_res_weight,
            nl_bc_weight=nl_bc_weight,
            nl_ic_weight=nl_ic_weight,
            nl_weight=nl_weight,
            distill_mode=distill_mode,
            distill_out_weight=distill_out_weight,
            distill_der_weight=distill_der_weight,
            distill_derx_weight=distill_derx_weight,
            distill_dert_weight=distill_dert_weight,
            distill_hes_weight=distill_hes_weight,
            distill_hesx_weight=distill_hesx_weight,
            distill_hest_weight=distill_hest_weight,
            ewc_mode=ewc_mode,
            ewc_weight=ewc_weight,
            ewc_auto_weighting=ewc_auto_weighting,
            ewc_warm_up=ewc_warm_up,
            ewc_decay=ewc_decay_factor,
            dwa_mode=dwa_mode,
            bc_mode=bc_mode,
            monitor_conflicts=monitor_conflicts,
            alpha=dwa_moving_avg_factor,
            moving_avg_frequency=dwa_moving_avg_frequency,
            dwa_warm_up=dwa_warm_up,
            input_units=input_units,
            hidden_units=layers,
            device=device,
            activation=torch.nn.Tanh()
            ).to(device)
        
    if actual_model.time_in_input:
        train_dataset = include_time_in_input(train_dataset)
        if nl_dataset is not None:
            nl_dataset = include_time_in_input(nl_dataset)
        if val_dataset is not None:
            val_dataset = include_time_in_input(val_dataset)
        if distill_dataset is not None:
            distill_dataset = include_time_in_input(distill_dataset)

        if train_boundary is not None:
            train_boundary = include_time_in_input(train_boundary)
        if val_boundary is not None:
            val_boundary = include_time_in_input(val_boundary)
        if nl_boundary is not None:
            nl_boundary = include_time_in_input(nl_boundary)
        if train_initial is not None:
            train_initial = include_time_in_input(train_initial)
        if val_initial is not None:
            val_initial = include_time_in_input(val_initial)
        if nl_initial is not None:
            nl_initial = include_time_in_input(nl_initial)

    if not actual_model.space_in_input:
        train_dataset = exclude_space_in_input(train_dataset)
        if nl_dataset is not None:
            nl_dataset = exclude_space_in_input(nl_dataset)
        if val_dataset is not None:
            val_dataset = exclude_space_in_input(val_dataset)
        if distill_dataset is not None:
            distill_dataset = exclude_space_in_input(distill_dataset)

        if train_boundary is not None:
            train_boundary = exclude_space_in_input(train_boundary)
        if val_boundary is not None:
            val_boundary = exclude_space_in_input(val_boundary)
        if nl_boundary is not None:
            nl_boundary = exclude_space_in_input(nl_boundary)
        if train_initial is not None:
            train_initial = exclude_space_in_input(train_initial)
        if val_initial is not None:
            val_initial = exclude_space_in_input(val_initial)
        if nl_initial is not None:
            nl_initial = exclude_space_in_input(nl_initial)


    if ewc_model is not None:
        if ewc_model.hidden_units != actual_model.hidden_units or ewc_model.input_units != actual_model.input_units:
            raise ValueError(f"EWC requires that the models have the same architecture: EWC model has hidden units {ewc_model.hidden_units} and {input_units} input units, while the actual model has {actual_model.hidden_units} hidden units and {actual_model.input_units} input units.")
        if ewc_src_fisher_file != "":
            prev_fisher_diag = torch.load(ewc_src_fisher_file).to(device)
        else:
            prev_fisher_diag = None
        with torch.no_grad():
            actual_model.ewc_model_weights = ewc_model.get_weights()
        actual_model.ewc_fisher_diag = ewc_model.get_fisher_diag(dataset=ewc_dataset)
        if prev_fisher_diag is not None:
            actual_model.ewc_fisher_diag = ewc_moving_avg_factor * actual_model.ewc_fisher_diag + (1 - ewc_moving_avg_factor) * prev_fisher_diag
        if ewc_dst_fisher_file != "":
            torch.save(actual_model.ewc_fisher_diag, ewc_dst_fisher_file)
        else:
            print("Warning: Fisher diagonal not stored (missing dst filepath 'dst_fisher_file' in the EWC dictionary).")
    else:
        actual_model.ewc_mode = "Off"
        actual_model.ewc_model_weights = None
        actual_model.ewc_fisher_diag = None

    # Models directory
    os.makedirs(models_dir, exist_ok=True) # Ensure models directory exists

    if type(lr_init) is float:
        lr_init = [lr_init]
    if type(batch_size) is int:
        batch_size = [batch_size]

    config_dict = copy.deepcopy(config_dict)
    config_dict["Actual"]["train_dataset"] = "TensorDataset object" if "<" in str(train_dataset) else train_dataset
    config_dict["Actual"]["nl_dataset"] = "TensorDataset object" if "<" in str(nl_dataset) else nl_dataset
    config_dict["Actual"]["val_dataset"] = "TensorDataset object" if "<" in str(val_dataset) else val_dataset
    config_dict["Actual"]["model"] = (None if actual_model == None else "PdeNet object")
    config_dict["Actual"]["res_weight"] = actual_res_weight
    config_dict["Actual"]["out_weight"] = actual_out_weight
    config_dict["Actual"]["der_weight"] = actual_der_weight
    config_dict["Actual"]["derx_weight"] = actual_derx_weight
    config_dict["Actual"]["dert_weight"] = actual_dert_weight
    config_dict["Actual"]["hes_weight"] = actual_hes_weight
    config_dict["Actual"]["hesx_weight"] = actual_hesx_weight
    config_dict["Actual"]["hest_weight"] = actual_hest_weight
    config_dict["Actual"]["bc_weight"] = actual_bc_weight
    config_dict["Actual"]["ic_weight"] = actual_ic_weight
    config_dict["Actual"]["monitor_conflicts"] = monitor_conflicts
    config_dict["Actual"]["mode"] = actual_mode
    config_dict["Actual"]["layers"] = layers
    config_dict["Actual"]["fourier_features"] = actual_model.fourier_features
    config_dict["Actual"]["frequency_variance"] = actual_model.frequency_variance
    config_dict["Actual"]["pde_params_in_input"] = actual_model.pde_params_in_input
    config_dict["Actual"]["ic_params_in_input"] = actual_model.ic_params_in_input
    config_dict["Actual"]["boundary"] = boundary_mode
    config_dict["Actual"]["pde_at_bd"] = pde_at_bd
    config_dict["Actual"]["pde_at_t0"] = pde_at_t0
    config_dict["Actual"]["initial_time"] = initial_time_mode

    if config_dict.get("Unlabeled") is None:
        config_dict["Unlabeled"] = {}
    config_dict["Unlabeled"]["dataset"] = "TensorDataset object" if "<" in str(nl_dataset) else nl_dataset
    config_dict["Unlabeled"]["buffer_size"] = memory_buffer_size_nl
    config_dict["Unlabeled"]["buffer_size_bc"] = memory_buffer_size_nl_bc
    config_dict["Unlabeled"]["buffer_size_ic"] = memory_buffer_size_nl_ic
    config_dict["Unlabeled"]["weight"] = nl_weight
    config_dict["Unlabeled"]["bc_weight"] = nl_bc_weight
    config_dict["Unlabeled"]["ic_weight"] = nl_ic_weight
    config_dict["Unlabeled"]["boundary"] = boundary_mode_nl
    config_dict["Unlabeled"]["initial_time"] = initial_time_mode_nl

    if config_dict.get("Distillation") is None:
        config_dict["Distillation"] = {}
    config_dict["Distillation"]["dataset"] = "TensorDataset object" if "<" in str(distill_dataset) else distill_dataset
    config_dict["Distillation"]["buffer_size"] = memory_buffer_size_distill
    config_dict["Distillation"]["model"] = (None if distill_model == None else "PdeNet object")
    config_dict["Distillation"]["mode"] = distill_mode
    config_dict["Distillation"]["out_weight"] = distill_out_weight
    config_dict["Distillation"]["der_weight"] = distill_der_weight
    config_dict["Distillation"]["derx_weight"] = distill_derx_weight
    config_dict["Distillation"]["dert_weight"] = distill_dert_weight
    config_dict["Distillation"]["hes_weight"] = distill_hes_weight
    config_dict["Distillation"]["hesx_weight"] = distill_hesx_weight
    config_dict["Distillation"]["hest_weight"] = distill_hest_weight

    if config_dict.get("EWC") is None:
        config_dict["EWC"] = {}
    config_dict["EWC"]["dataset"] = "TensorDataset object" if "<" in str(ewc_dataset) else ewc_dataset
    config_dict["EWC"]["buffer_size"] = memory_buffer_size_ewc
    config_dict["EWC"]["model"] = (None if ewc_model == None else "PdeNet object")
    config_dict["EWC"]["mode"] = ewc_mode
    config_dict["EWC"]["weight"] = ewc_weight
    config_dict["EWC"]["auto_weighting"] = ewc_auto_weighting
    config_dict["EWC"]["warm_up"] = ewc_warm_up
    config_dict["EWC"]["decay"] = ewc_decay_factor
    config_dict["EWC"]["src_fisher_diag_file"] = ewc_src_fisher_file
    config_dict["EWC"]["dst_fisher_diag_file"] = ewc_dst_fisher_file
    config_dict["EWC"]["moving_avg_factor"] = ewc_moving_avg_factor
    

    if config_dict.get("DWA") is None:
        config_dict["DWA"] = {}
    config_dict["DWA"]["mode"] = dwa_mode
    config_dict["DWA"]["moving_avg_factor"] = dwa_moving_avg_factor
    config_dict["DWA"]["moving_avg_frequency"] = dwa_moving_avg_frequency
    config_dict["DWA"]["dwa_warm_up"] = dwa_warm_up
    config_dict["DWA"]["delayed_weights"] = delayed_weights

    config_dict["learning_rate"] = lr_init
    config_dict["batch_size"] = batch_size
    config_dict["clip_grad"] = clip_grad
    config_dict["epochs"] = epochs
    config_dict["early_stop_value"] = early_stop_value

    config_dict["pruner"] = pruner
    config_dict["threshold"] = threshold
    config_dict["n_warmup_steps"] = n_warmup_steps
    config_dict["n_trials"] = n_trials
    config_dict["train_steps"] = train_steps
    config_dict["eval_every"] = eval_every
    config_dict["seed"] = seed
    config_dict["device"] = device
    config_dict["scheduler"] = scheduler
    config_dict["models_dir"] = models_dir

    if type(config) is str:
        # Copy the config file in the model directory
        shutil.copy(config, models_dir)
    else:
        config = "config.yaml"

    with open(f"{models_dir}/{os.path.basename(config)}", "w") as file:
        yaml.dump(config_dict, file, sort_keys=False)
    os.rename(f"{models_dir}/{os.path.basename(config)}", f"{models_dir}/config.yaml")

    objective = Objective(
        seed = seed,
        model = actual_model,
        train_steps = train_steps,
        epochs = epochs,
        early_stop_value = early_stop_value,
        eval_every = eval_every,
        train_dataset = train_dataset,
        nl_dataset = nl_dataset,
        val_dataset = val_dataset,
        train_bc_dataset = train_boundary,
        train_ic_dataset = train_initial,
        nl_bc_dataset = nl_boundary,
        nl_ic_dataset = nl_initial,
        val_bc_dataset = val_boundary,
        val_ic_dataset = val_initial,
        distill_dataset = distill_dataset,
        batch_size = batch_size,
        clip_grad = clip_grad,
        lr_init = lr_init,
        fourier_features=fourier_features,
        frequency_variance=frequency_variance,
        scheduler_mode=scheduler,
        delayed_weights=delayed_weights,
        device = device,
        models_dir = models_dir,
        suggestions=suggestions,
        reset=True
        )

    if pruner == "Hyperband":
        opt_pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=epochs, reduction_factor=3)
    elif pruner == "Median":
        opt_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=n_warmup_steps, interval_steps=1)
    elif pruner == "Threshold":
        opt_pruner = optuna.pruners.ThresholdPruner(upper=threshold, n_warmup_steps=n_warmup_steps)
    else:
        raise ValueError(f"Unknown pruner '{pruner}'.")
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction = "minimize", pruner=opt_pruner, sampler=sampler)
    study.optimize(objective, n_trials = n_trials)

    print("Best trial params:", study.best_trial.params)
    print("Best trial value:", study.best_trial.value)

    # Sort the trials by their objective value
    all_trials = [t for t in study.trials if t.state != TrialState.PRUNED]
    sorted_trials = sorted_trials = sorted(
        all_trials,
        key=lambda t: t.value
        )

    # Assign a new 'rank' attribute to each trial
    for i, trial in enumerate(sorted_trials):
        rank = i
        old_name = trial.user_attrs.get("trial_name")
        new_name = f"trial{rank}"
        trial.set_user_attr("rank", rank)
        trial.set_user_attr("old_trial_name", old_name)
        trial.set_user_attr("trial_name", new_name)

        new_filename = f"{models_dir}/{new_name}"
        old_filename = f"{models_dir}/{old_name}"
        if os.path.exists(new_filename):
            if os.path.isfile(new_filename):
                os.remove(new_filename)
            elif os.path.isdir(new_filename):
                shutil.rmtree(new_filename)
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
            print(f"{old_name} --> {new_name}: Objective Value = {trial.value:.4f}")

# ===================================================== Main =====================================================
if __name__ == "__main__":
    # Init parser for command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config.yaml", type=str, help="Path to the configuration file (YAML)")

    # Parse command-line arguments
    cli_args = parser.parse_args()

    start_train(cli_args.config)