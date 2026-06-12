"""
load_store_utils.py
===========

This module implements the loading and storing of train statistics and PINN models.

Functions:
- save_stats
- load_stats
- save_model
- resume_model
- store_split
"""

import torch
from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np
import os
from model2 import Pinn, LOSS_TERMS

def save_stats(stats_dict: dict, directory: str) -> None:
    """
    Save the dictionary stats_dict as a set of csv and npy files in the directory directory.

    For each key in stats_dict create
    - {key}_keys.csv, containing the (ordered) keys in the subdictionary stats_dict[key];
    - {key}_stats.npy, containing the (ordered) curves corresponding to the keys in the subdictionary stats_dict[key].

    Parameters
    ----------
    stats_dict : dict
    directory : str

    Returns
    -------
    None
    """
    for key in stats_dict.keys():
        curves = []
        if key not in ["times", "weights", "conflicts", "grad_norms", "train_loss", "train_loss_grad_norm"]:
            with open(f"{directory}/{key}_keys.csv", 'w') as f:
                for k in LOSS_TERMS + ["weighted_loss", "step_list"]:
                    curves.append(torch.tensor(stats_dict[key][k]).cpu().numpy())
                    f.write(k + ",")
        elif key in ["weights", "conflicts", "grad_norms"]:
            with open(f"{directory}/{key}_keys.csv", 'w') as f:
                for k in LOSS_TERMS + ["step_list"]:
                    if None not in stats_dict[key][k]:
                        curves.append(torch.tensor(stats_dict[key][k]).cpu().numpy())
                        f.write(k + ",")
        else:
            curves.append(torch.tensor(stats_dict[key]).cpu().numpy())

        # Stack loss curves
        stacked_curves = np.column_stack(curves)
        # Save loss curves
        with open(f"{directory}/{key}_stats.npy", "wb") as f:
            np.save(f, stacked_curves)

def load_stats(directory: str, key_list: list = None) -> dict:
    """
    Return the dictionary constructed from the {directory}/{key}_keys.csv files, key in key_list.

    If key_list is not passed, then ["train", "val", "weights", "conflicts", "grad_norms", "times", "train_loss", "train_loss_grad_norm"] is used.

    Parameters
    ----------
    directory : str
    key_list : list

    Returns
    -------
    dict
    """
    if key_list is None:
        key_list = ["train", "val", "weights", "conflicts", "grad_norms", "times", "train_loss", "train_loss_grad_norm"]
    stats_dict = {}
    for key in key_list:
        stats_dict[key] = {}
        if key not in ["times", "train_loss", "train_loss_grad_norm"]:
            with open(f"{directory}/{key}_keys.csv", "r") as file:
                stats_keys = file.readlines()[0].strip().split(",")
            stats_data = np.load(f"{directory}/{key}_stats.npy")
            for i in range(len(stats_keys)-1):
                stats_dict[key][stats_keys[i]] = stats_data[:, i]
        else:
            stats_dict[key] = np.load(f"{directory}/{key}_stats.npy")
    return stats_dict

def save_model(
        model: PdeNet,
        filepath: str # .pth
    ) -> None:
    """
    Save the model as a .pth file in {models_dir}/{name}/model.pth, that contains model state and training hyperparameters.

    Parameters
    ----------
    model : PdeNet
        The model to save.
    filepath : str
        Filepath of the model file.
    
    Returns
    -------
    None
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "fourier_features": model.fourier_features,
        "frequency_variance": model.frequency_variance,
        "input_units": model.input_units,
        "hidden_units": model.hidden_units,
        "device": model.device,
        "ewc_weight": model.ewc_weight,
        "ewc_auto_weighting": model.ewc_auto_weighting,
        "ewc_warm_up": model.ewc_warm_up,
        "ewc_decay": model.ewc_decay,
        "alpha": model.alpha,
        "weighted_avg_frequency": model.moving_avg_frequency,
        "dwa_warm_up": model.dwa_warm_up,
        "sys_mode": model.sys_mode,
        "distill_mode": model.distill_mode,
        "ewc_mode": model.ewc_mode,
        "dwa_mode": model.dwa_mode,
        "monitor_conflicts": model.monitor_conflicts,
        "activation": model.activation
    }
    # Save the checkpoint dictionary
    torch.save(checkpoint, filepath)

def resume_model(
        filepath: str,
        device: str = "cpu",
        dwa_mode: str = None,
        monitor_conflicts: bool = None,
        alpha: float = None,
        weighted_avg_frequency: int = None,
        dwa_warm_up: int = None,
        ewc_mode: str = None,
        ewc_weight: float = None,
        ewc_auto_weighting: bool = None,
        ewc_warm_up: int = None,
        ewc_decay: float = None
        ) -> Pinn:
    """
    Load a saved model from a .pth file.
    """
    if not os.path.exists(filepath):
            raise ValueError(f"File '{filepath}' not found.")

    # Load the entire checkpoint dictionary
    checkpoint = torch.load(filepath, weights_only=False)

    if dwa_mode is None: dwa_mode = checkpoint["dwa_mode"]
    if alpha is None: alpha = checkpoint["alpha"]
    if weighted_avg_frequency is None: weighted_avg_frequency = checkpoint["weighted_avg_frequency"]
    if dwa_warm_up is None: dwa_warm_up = checkpoint["dwa_warm_up"]

    if ewc_mode is None: ewc_mode = checkpoint["ewc_mode"]
    if ewc_importance is None: ewc_importance = checkpoint["ewc_importance"]
    if ewc_weight is None: ewc_weight = checkpoint["ewc_weight"]
    if ewc_auto_weighting is None:
        if "ewc_auto_weighting" in checkpoint.keys():
            ewc_auto_weighting = checkpoint["ewc_auto_weighting"]
        else:
            ewc_auto_weighting = False
    if ewc_warm_up is None:
        if "ewc_warm_up" in checkpoint.keys():
            ewc_warm_up = checkpoint["ewc_warm_up"]
        else:
            ewc_warm_up = 0
    if ewc_decay is None:
        if "ewc_decay" in checkpoint.keys():
            ewc_decay = checkpoint["ewc_decay"]
        else:
            ewc_decay = 1.0
    if bc_mode is None:
        bc_mode = checkpoint["bc_mode"]
    
    if monitor_conflicts is None:
        if "monitor_conflicts" in checkpoint.keys():
            monitor_conflicts = checkpoint["monitor_conflicts"]
        else:
            monitor_conflicts = False
    
    if "frequency_variance" in checkpoint.keys():
        frequency_variance = checkpoint["frequency_variance"]
    else:
        frequency_variance = 1.0
    
    # Create a new instance of the distillation model architecture using the loaded parameters
    model = Pinn(
        input_units=checkpoint["input_units"],
        hidden_units=checkpoint["hidden_units"],
        activation=checkpoint["activation"],
        fourier_features=checkpoint["fourier_features"],
        frequency_variance=frequency_variance,
        device=device,
        ewc_weight=ewc_weight,
        ewc_auto_weighting=ewc_auto_weighting,
        ewc_warm_up=ewc_warm_up,
        ewc_decay=ewc_decay,
        alpha=alpha,
        ewc_mode=ewc_mode,
        dwa_mode=dwa_mode,
        moving_avg_frequency=weighted_avg_frequency,
        dwa_warm_up=dwa_warm_up,
        monitor_conflicts=monitor_conflicts,
        ).to(device)

    # Load the distillation model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    return model