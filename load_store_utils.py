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