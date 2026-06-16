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
        model: Pinn,
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
        "B": model.B,
        "spatial_input": model.spatial_input,
        "temporal_input": model.temporal_input,
        "param_input": model.param_input,
        "hidden_units": model.hidden_units,
        "activation": model.activation
    }
    # Save the checkpoint dictionary
    torch.save(checkpoint, filepath)

def resume_model(
        filepath: str,
        device: str = "cpu"
        ) -> Pinn:
    """
    Load a saved model from a .pth file.
    """
    if not os.path.exists(filepath):
        raise ValueError(f"File '{filepath}' not found.")

    checkpoint = torch.load(filepath, weights_only=False)
    
    model = Pinn(
        device = device,
        hidden_units = checkpoint["hidden_units"],
        activation = checkpoint["activation"],
        temporal_input = checkpoint["temporal_input"],
        spatial_input = checkpoint["spatial_input"],
        fourier_features = checkpoint["fourier_features"],
        frequency_variance = checkpoint["frequency_variance"],
        param_input = checkpoint["param_input"],
        task_list = [],
        eval_task_list = [],
        ewc_mode = "Off",
        dwa_mode = "Off",
        monitor_conflicts = False
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.B = checkpoint["B"]
    
    return model