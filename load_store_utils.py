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
from model import PdeNet, LOSS_TERMS

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
        name: str,
        lr_init: float,
        batch_size: float,
        clip_grad: bool,
        scheduler: str,
        delayed_weights: bool,
        models_dir: str
    ) -> None:
    """
    Save the model as a .pth file in {models_dir}/{name}/model.pth, that contains model state and training hyperparameters.

    Parameters
    ----------
    model : PdeNet
        The PINN to save.
    name : str
        The name in the math.
    lr_init : float
        The initial learning rate used to train the model.
    batch_size : int
        The batch size used to train the model.
    clip_grad : bool
        Gradient clipping to 1.
    scheduler : str
        Learning rate scheduler.
    delayed_weights : bool
        It True, delayed weights are used in training.
    models_dir : str
        models directory in the path.
    
    Returns
    -------
    None
    """
    checkpoint = {
        "model_state_dict"      : model.state_dict(),
        "time_in_input"         : model.time_in_input,
        "space_in_input"        : model.space_in_input,
        "fourier_features"      : model.fourier_features,
        "frequency_variance"    : model.frequency_variance,
        "pde_params_in_input"   : model.pde_params_in_input,
        "ic_params_in_input"    : model.ic_params_in_input,
        "batch_size"            : batch_size,
        "pde"                   : model.pde,
        "input_units"           : model.input_units,
        "hidden_units"          : model.hidden_units,
        "lr_init"               : lr_init,
        "clip_grad"             : clip_grad,
        "scheduler"             : scheduler,
        "delayed_weights"       : delayed_weights,
        "device"                : model.device,
        "bc_weight"             : model.bc_weight,
        "ic_weight"             : model.ic_weight,
        "res_weight"            : model.res_weight,
        "out_weight"            : model.out_weight,
        "der_weight"            : model.der_weight,
        "derx_weight"            : model.derx_weight,
        "dert_weight"            : model.dert_weight,
        "hes_weight"            : model.hes_weight,
        "hesx_weight"            : model.hesx_weight,
        "hest_weight"            : model.hest_weight,
        "nl_weight"             : model.nl_weight,
        "nl_bc_weight"          : model.nl_bc_weight,
        "nl_ic_weight"          : model.nl_ic_weight,
        "distill_out_weight"    : model.distill_out_weight,
        "distill_der_weight"    : model.distill_der_weight,
        "distill_derx_weight"    : model.distill_derx_weight,
        "distill_dert_weight"    : model.distill_dert_weight,
        "distill_hes_weight"    : model.distill_hes_weight,
        "distill_hesx_weight"    : model.distill_hesx_weight,
        "distill_hest_weight"    : model.distill_hest_weight,
        "ewc_weight"            : model.ewc_weight,
        "ewc_auto_weighting"    : model.ewc_auto_weighting,
        "ewc_warm_up"           : model.ewc_warm_up,
        "ewc_decay"             : model.ewc_decay,
        "alpha"                 : model.alpha,
        "weighted_avg_frequency": model.moving_avg_frequency,
        "dwa_warm_up"           : model.dwa_warm_up,
        "sys_mode"              : model.sys_mode,
        "distill_mode"          : model.distill_mode,
        "ewc_mode"              : model.ewc_mode,
        "dwa_mode"              : model.dwa_mode,
        "bc_mode"               : model.bc_mode,
        "monitor_conflicts"     : model.monitor_conflicts,
        "sys_importance"        : model.sys_importance,
        "bc_importance"         : model.bc_importance,
        "ic_importance"         : model.ic_importance,
        "nl_importance"         : model.nl_importance,
        "nl_bc_importance"      : model.nl_bc_importance,
        "nl_ic_importance"      : model.nl_ic_importance,
        "distill_importance"    : model.distill_importance,
        "ewc_importance"        : model.ewc_importance,
        "activation"            : model.activation
    }
    # Save the checkpoint dictionary
    torch.save(checkpoint, f"{models_dir}/{name}/model.pth")

def resume_model(
        model_path: str,
        device: str = "cpu",
        dwa_mode: str = None,
        monitor_conflicts: bool = None,
        alpha: float = None,
        weighted_avg_frequency: int = None,
        dwa_warm_up: int = None,
        sys_mode: str = None,
        sys_importance: float = None,
        bc_importance: float = None,
        ic_importance: float = None,
        bc_weight: float = None,
        ic_weight: float = None,
        res_weight: float = None,
        out_weight: float = None,
        der_weight: float = None,
        derx_weight: float = None,
        dert_weight: float = None,
        hes_weight: float = None,
        hesx_weight: float = None,
        hest_weight: float = None,
        nl_importance: float = None,
        nl_weight: float = None,
        nl_bc_importance: float = None,
        nl_bc_weight: float = None,
        nl_ic_importance: float = None,
        nl_ic_weight: float = None,
        distill_mode: str = None,
        distill_importance: float = None,
        distill_out_weight: float = None,
        distill_der_weight: float = None,
        distill_derx_weight: float = None,
        distill_dert_weight: float = None,
        distill_hes_weight: float = None,
        distill_hesx_weight: float = None,
        distill_hest_weight: float = None,
        ewc_mode: str = None,
        ewc_importance: float = None,
        ewc_weight: float = None,
        ewc_auto_weighting: bool = None,
        ewc_warm_up: int = None,
        ewc_decay: float = None,
        bc_mode: str = None
        ) -> PdeNet:
    """
    Load a saved model from a .pth file.

    The model state values that are not passed are taken from the file.

    Parameters
    ----------
    model_path : str
    device : str
    dwa_mode : str
    monitor_conflicts : bool
    alpha : float
    weighted_avg_frequency : int
    dwa_warm_up : int
    sys_mode : str
    sys_importance : float
    bc_importance : float
    ic_importance : float
    bc_weight : float
    ic_weight : float
    res_weight : float
    out_weight : float
    der_weight : float
    derx_weight : float
    dert_weight : float
    hes_weight : float
    nl_importance : float
    nl_weight : float
    distill_mode : str
    distill_importance : float
    distill_out_weight : float
    distill_der_weight : float
    distill_derx_weight : float
    distill_dert_weight : float
    distill_hes_weight : float
    ewc_mode : str
    ewc_importance : float
    ewc_weight : float
    ewc_auto_weighting : bool
    ewc_warm_up : int
    ewc_decay : float
    bc_mode : str

    Returns
    -------
    PdeNet
        The PINN model.
    """
    if not os.path.exists(model_path):
            raise ValueError(f"File '{model_path}' not found.")

    # Load the entire checkpoint dictionary
    checkpoint = torch.load(model_path, weights_only=False)

    if dwa_mode is None: dwa_mode                               = checkpoint["dwa_mode"]
    if alpha is None: alpha                                     = checkpoint["alpha"]
    if weighted_avg_frequency is None: weighted_avg_frequency   = checkpoint["weighted_avg_frequency"]
    if dwa_warm_up is None: dwa_warm_up                         = checkpoint["dwa_warm_up"]
    if sys_mode is None: sys_mode                               = checkpoint["sys_mode"]
    if sys_importance is None: sys_importance                   = checkpoint["sys_importance"]
    if bc_importance is None: bc_importance                     = checkpoint["bc_importance"]
    if ic_importance is None: ic_importance                     = checkpoint["ic_importance"]
    if bc_weight is None: bc_weight                             = checkpoint["bc_weight"]
    if ic_weight is None: ic_weight                             = checkpoint["ic_weight"]
    if res_weight is None: res_weight                           = checkpoint["res_weight"]
    if out_weight is None: out_weight                           = checkpoint["out_weight"]
    if der_weight is None: der_weight                           = checkpoint["der_weight"]

    if derx_weight is None:
        if "derx_weight" in checkpoint.keys():
            derx_weight = checkpoint["derx_weight"]
        else:
            derx_weight = 1.0
    if dert_weight is None:
        if "dert_weight" in checkpoint.keys():
            dert_weight = checkpoint["dert_weight"]
        else:
            dert_weight = 1.0

    if hes_weight is None: hes_weight                           = checkpoint["hes_weight"]

    if hesx_weight is None:
        if "hesx_weight" in checkpoint.keys():
            hesx_weight = checkpoint["hesx_weight"]
        else:
            hesx_weight = 1.0
    if hest_weight is None:
        if "hest_weight" in checkpoint.keys():
            hest_weight = checkpoint["hest_weight"]
        else:
            hest_weight = 1.0

    if nl_importance is None: nl_importance                     = checkpoint["nl_importance"]
    if nl_weight is None: nl_weight                             = checkpoint["nl_weight"]
    if nl_bc_importance is None:
        if "nl_bc_importance" in checkpoint.keys():
            nl_bc_importance = checkpoint["nl_bc_importance"]
        else:
            nl_bc_importance = 1.0
    if nl_bc_weight is None:
        if "nl_bc_weight" in checkpoint.keys():
            nl_bc_weight = checkpoint["nl_bc_weight"]
        else:
            nl_bc_weight = 1.0
    if nl_ic_importance is None:
        if "nl_ic_importance" in checkpoint.keys():
            nl_ic_importance = checkpoint["nl_ic_importance"]
        else:
            nl_ic_importance = 1.0
    if nl_ic_weight is None:
        if "nl_ic_weight" in checkpoint.keys():
            nl_ic_weight = checkpoint["nl_ic_weight"]
        else:
            nl_ic_weight = 1.0
    if distill_mode is None: distill_mode                       = checkpoint["distill_mode"]
    if distill_importance is None: distill_importance           = checkpoint["distill_importance"]
    if distill_out_weight is None: distill_out_weight           = checkpoint["distill_out_weight"]
    if distill_der_weight is None: distill_der_weight           = checkpoint["distill_der_weight"]

    if distill_derx_weight is None:
        if "distill_derx_weight" in checkpoint.keys():
            distill_derx_weight = checkpoint["distill_derx_weight"]
        else:
            distill_derx_weight = 1.0
    
    if distill_dert_weight is None:
        if "distill_dert_weight" in checkpoint.keys():
            distill_dert_weight = checkpoint["distill_dert_weight"]
        else:
            distill_dert_weight = 1.0

    if distill_hes_weight is None: distill_hes_weight           = checkpoint["distill_hes_weight"]

    if distill_hesx_weight is None:
        if "distill_hesx_weight" in checkpoint.keys():
            distill_hesx_weight = checkpoint["distill_hesx_weight"]
        else:
            distill_hesx_weight = 1.0
    
    if distill_hest_weight is None:
        if "distill_hest_weight" in checkpoint.keys():
            distill_hest_weight = checkpoint["distill_hest_weight"]
        else:
            distill_hest_weight = 1.0

    if ewc_mode is None: ewc_mode                               = checkpoint["ewc_mode"]
    if ewc_importance is None: ewc_importance                   = checkpoint["ewc_importance"]
    if ewc_weight is None: ewc_weight                           = checkpoint["ewc_weight"]
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
    
    if dwa_mode == "On":
        dwa_mode = "NormK"
    
    # Create a new instance of the distillation model architecture using the loaded parameters
    model = PdeNet(
        pde=checkpoint["pde"],
        input_units=checkpoint["input_units"],
        hidden_units=checkpoint["hidden_units"],
        activation=checkpoint["activation"],
        time_in_input=checkpoint["time_in_input"],
        space_in_input=checkpoint["space_in_input"],
        fourier_features=checkpoint["fourier_features"],
        frequency_variance=frequency_variance,
        pde_params_in_input=checkpoint["pde_params_in_input"],
        ic_params_in_input=checkpoint["ic_params_in_input"],
        device=device,
        bc_weight=bc_weight,
        ic_weight=ic_weight,
        res_weight=res_weight,
        out_weight=out_weight,
        der_weight=der_weight,
        derx_weight=derx_weight,
        dert_weight=dert_weight,
        hes_weight=hes_weight,
        hesx_weight=hesx_weight,
        hest_weight=hest_weight,
        nl_weight=nl_weight,
        nl_bc_weight=nl_bc_weight,
        nl_ic_weight=nl_ic_weight,
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
        ewc_decay=ewc_decay,
        alpha=alpha,
        sys_mode=sys_mode,
        distill_mode=distill_mode,
        ewc_mode=ewc_mode,
        dwa_mode=dwa_mode,
        moving_avg_frequency=weighted_avg_frequency,
        dwa_warm_up=dwa_warm_up,
        bc_mode=bc_mode,
        monitor_conflicts=monitor_conflicts,
        sys_importance=sys_importance,
        distill_importance=distill_importance,
        ewc_importance=ewc_importance,
        bc_importance=bc_importance,
        ic_importance=ic_importance,
        nl_importance=nl_importance,
        nl_bc_importance=nl_bc_importance,
        nl_ic_importance=nl_ic_importance
        ).to(device)

    # Load the distillation model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    return model