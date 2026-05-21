"""
train_loop.py
===========

This module contains:
- train_loop (*)
- get_dictionary (used by train_loop)

This module implements the training loop logic and is used by the optuna_objective.py module.
The train_loop function takes in input the necessary informations 
to execute the training loop and store training statistics in a dictionary 
that is returned.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import random
from itertools import cycle
from model import PdeNet, LOSS_TERMS
from torch.optim import LBFGS, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR
import time
import math
import itertools
from itertools import cycle
import optuna
from pde_utils import key_str, key_idx, ic_key_idx
from generate import X, U, DU, D2U, OUTWARD_NORMAL, PDE_KEYS, PDE_VALUES, IC_KEYS, IC_VALUES, RESIDUAL_KEYS, RESIDUAL_VALUES

def get_dictionary(keys: torch.Tensor, values: torch.Tensor, pde_name: str) -> dict:
    """
    Build a dictionary given a list of keys and a list of values.

    Parameters
    ----------
    keys : torch.Tensor
    values : torch.Tensor
    pde_name : str

    Returns
    -------
    dict.
    """
    dictionary = {}
    keys = keys[0].tolist()
    values = [values[:, i] for i in range(values.shape[1])]
    for key, value in zip([key_str(k, pde_name) for k in keys], [v for v in values]):
        if len(value) == 1: value = value.item()
        dictionary[key] = value
    return dictionary

def train_loop(
        model: PdeNet,
        train_steps: int,
        epochs: int,
        early_stop_value: float,
        eval_every: int,
        lr_init: float,
        batch_size: int,
        clip_grad: bool,
        scheduler_mode: str,
        delayed_weights: bool,
        trial: optuna.trial.Trial,
        train_dataset: TensorDataset,
        nl_dataset: TensorDataset = None,
        val_dataset: TensorDataset = None,
        train_bc_dataset: TensorDataset = None,
        train_ic_dataset: TensorDataset = None,
        nl_bc_dataset: TensorDataset = None,
        nl_ic_dataset: TensorDataset = None,
        val_bc_dataset: TensorDataset = None,
        val_ic_dataset: TensorDataset = None,
        distill_dataset: TensorDataset = None,
        device: str = "cpu",
        seed: int = 42
        ) -> dict:
    """
    Perform the training loop.

    Parameters
    ----------
    model : PdeNet
        PINN to train.
    train_steps : int
        Number of train steps.
    epochs : int
        Number of epochs.
    early_stop_value : float
        Value for the grad norm that determine an early stop of the training process.
        If None, no early stopping is performed.
    eval_every : int
        Evaluation period.
    lr_init : float
        Initial learning rate.
    batch_size : int
        Batch size.
    clip_grad : bool
        Gradient clipping to 1.
    scheduler_mode : str
        Learning rate scheduling.
    delayed_weights : bool
        If True, delayed weights are used for dwa.
    trial : optuna.trial.Trial
        Optuna trial object.
    train_dataset : TensorDataset
        Labeled training set in D.
    nl_dataset : TensorDataset
        Unlabeled training set in D.
    val_dataset : TensorDataset
        Validation set in D.
    train_bc_dataset : TensorDataset
        Training boundary dataset (points in bd(D)).
    train_ic_dataset : TensorDataset
        Training initial condition dataset (points in D at t0).
    val_bc_dataset : TensorDataset
        Validation boundary dataset (points in bd(D)).
    val_ic_dataset : TensorDataset
        Validation initial condition dataset (points in D at t0).
    distill_dataset : TensorDataset
        Distillation set.
    device : str
    seed : int

    Returns
    -------
    dict
        Dictionary containing training statistics.
    """
    # Stats dictionary
    stats_dict = {"train": {}, "val": {}, "weights": {}, "conflicts": {}, "grad_norms": {}, "train_loss": [], "train_loss_grad_norm": [], "times": []}
    for k in LOSS_TERMS:
        stats_dict["train"][k] = []
        stats_dict["val"][k] = []
        stats_dict["weights"][k] = []
        stats_dict["conflicts"][k] = []
        stats_dict["grad_norms"][k] = []

    stats_dict["grad_norms"]["weighted_loss"] = []
    stats_dict["train"]["weighted_loss"] = []
    stats_dict["val"]["weighted_loss"] = []
    stats_dict["train"]["step_list"] = []
    stats_dict["val"]["step_list"] = []
    stats_dict["weights"]["step_list"] = []
    stats_dict["conflicts"]["step_list"] = []
    stats_dict["grad_norms"]["step_list"] = []

    # Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Load data and build dataloaders
    N_train = len(train_dataset)
    if batch_size < N_train:
        N_train = N_train - N_train % batch_size
        batch_size_train = batch_size
    else:
        batch_size_train = N_train
    print(f"batch_size_train = {batch_size_train}")
    train_dataset = Subset(train_dataset, list(range(N_train)))
    print(f"len(train_dataset) = {len(train_dataset)}")

    
    if distill_dataset is not None:
        N_distill = len(distill_dataset)
        if batch_size < N_distill:
            N_distill = N_distill - N_distill % batch_size
            batch_size_distill = batch_size
        else:
            batch_size_distill = N_distill
        print(f"batch_size_distill = {batch_size_distill}")
        distill_dataset = Subset(distill_dataset, list(range(N_distill)))
        print(f"len(distill_dataset) = {len(distill_dataset)}")

       
    if nl_dataset is not None:
        N_nl = len(nl_dataset)
        if batch_size < N_nl:
            N_nl = N_nl - N_nl % batch_size
            batch_size_nl = batch_size
        else:
            batch_size_nl = N_nl
        print(f"batch_size_nl = {batch_size_nl}")
        nl_dataset = Subset(nl_dataset, list(range(N_nl)))
        print(f"len(nl_dataset) = {len(nl_dataset)}")
    

    if val_dataset is not None:
        N_val = len(val_dataset)
        if batch_size < N_val:
            N_val = N_val - N_val % batch_size
            batch_size_val = batch_size
        else:
            batch_size_val = N_val
        print(f"batch_size_val = {batch_size_val}")
        val_dataset = Subset(val_dataset, list(range(N_val)))
        print(f"len(val_dataset) = {len(val_dataset)}")
        

    if train_ic_dataset is not None:
        N_train_ic = len(train_ic_dataset)
        if batch_size < N_train_ic:
            N_train_ic = N_train_ic - N_train_ic % batch_size
            batch_size_train_ic = batch_size
        else:
            batch_size_train_ic = N_train_ic
        print(f"batch_size_train_ic = {batch_size_train_ic}")
        train_ic_dataset = Subset(train_ic_dataset, list(range(N_train_ic)))
        print(f"len(train_ic_dataset) = {len(train_ic_dataset)}")


    if train_bc_dataset is not None:
        N_train_bc = len(train_bc_dataset)
        if batch_size < N_train_bc:
            N_train_bc = N_train_bc - N_train_bc % batch_size
            batch_size_train_bc = batch_size
        else:
            batch_size_train_bc = N_train_bc
        print(f"batch_size_train_bc = {batch_size_train_bc}")
        train_bc_dataset = Subset(train_bc_dataset, list(range(N_train_bc)))
        print(f"len(train_bc_dataset) = {len(train_bc_dataset)}")

    
    if nl_ic_dataset is not None:
        N_nl_ic = len(nl_ic_dataset)
        if batch_size < N_nl_ic:
            N_nl_ic = N_nl_ic - N_nl_ic % batch_size
            batch_size_nl_ic = batch_size
        else:
            batch_size_nl_ic = N_nl_ic
        print(f"batch_size_nl_ic = {batch_size_nl_ic}")
        nl_ic_dataset = Subset(nl_ic_dataset, list(range(N_nl_ic)))
        print(f"len(nl_ic_dataset) = {len(nl_ic_dataset)}")


    if nl_bc_dataset is not None:
        N_nl_bc = len(nl_bc_dataset)
        if batch_size < N_nl_bc:
            N_nl_bc = N_nl_bc - N_nl_bc % batch_size
            batch_size_nl_bc = batch_size
        else:
            batch_size_nl_bc = N_nl_bc
        print(f"batch_size_nl_bc = {batch_size_nl_bc}")
        nl_bc_dataset = Subset(nl_bc_dataset, list(range(N_nl_bc)))
        print(f"len(nl_bc_dataset) = {len(nl_bc_dataset)}")

    
    if val_ic_dataset is not None:
        N_val_ic = len(val_ic_dataset)
        if batch_size < N_val_ic:
            N_val_ic = N_val_ic - N_val_ic % batch_size
            batch_size_val_ic = batch_size
        else:
            batch_size_val_ic = N_val_ic
        print(f"batch_size_val_ic = {batch_size_val_ic}")
        val_ic_dataset = Subset(val_ic_dataset, list(range(N_val_ic)))
        print(f"len(val_ic_dataset) = {len(val_ic_dataset)}")


    if val_bc_dataset is not None:
        N_val_bc = len(val_bc_dataset)
        if batch_size < N_val_bc:
            N_val_bc = N_val_bc - N_val_bc % batch_size
            batch_size_val_bc = batch_size
        else:
            batch_size_val_bc = N_val_bc
        print(f"batch_size_val_bc = {batch_size_val_bc}")
        val_bc_dataset = Subset(val_bc_dataset, list(range(N_val_bc)))
        print(f"len(val_bc_dataset) = {len(val_bc_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size_train, generator=gen, shuffle=True)#, drop_last=True)
    if nl_dataset is None:
        N = len(train_dataloader)
        nl_iter = itertools.repeat((None, None, None))
        train_iter = train_dataloader
    else:
        nl_dataloader = DataLoader(nl_dataset, batch_size_nl, generator=gen, shuffle=True)

        if len(train_dataloader) > len(nl_dataloader):
            N = len(train_dataloader)
            nl_iter = cycle(nl_dataloader)
            train_iter = train_dataloader
        else:
            N = len(nl_dataloader)
            nl_iter = nl_dataloader
            train_iter = cycle(train_dataloader)

    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size_val, generator=gen, shuffle=True)
    else:
        val_dataloader = None

    if train_bc_dataset is None:
        train_bc_iter = itertools.repeat((None, None))
    else:
        train_bc_dataloader = DataLoader(train_bc_dataset, batch_size_train_bc, generator=gen, shuffle=True)
        train_bc_iter = cycle(train_bc_dataloader)

    if train_ic_dataset is None:
        train_ic_iter = itertools.repeat((None, None))
    else:
        train_ic_dataloader = DataLoader(train_ic_dataset, batch_size_train_ic, generator=gen, shuffle=True)
        train_ic_iter = cycle(train_ic_dataloader)

    
    if nl_bc_dataset is None:
        nl_bc_iter = itertools.repeat((None, None))
    else:
        nl_bc_dataloader = DataLoader(nl_bc_dataset, batch_size_nl_bc, generator=gen, shuffle=True)
        nl_bc_iter = cycle(nl_bc_dataloader)

    if nl_ic_dataset is None:
        nl_ic_iter = itertools.repeat((None, None))
    else:
        nl_ic_dataloader = DataLoader(nl_ic_dataset, batch_size_nl_ic, generator=gen, shuffle=True)
        nl_ic_iter = cycle(nl_ic_dataloader)
    

    if val_dataset is None:
        val_bc_iter = None
    elif val_bc_dataset is None:
        val_bc_iter = itertools.repeat((None, None))
    else:
        val_bc_dataloader = DataLoader(val_bc_dataset, batch_size_val_bc, generator=gen, shuffle=True)
        val_bc_iter = cycle(val_bc_dataloader)

    if val_dataset is None:
        val_ic_iter = None
    elif val_ic_dataset is None:
        val_ic_iter = itertools.repeat((None, None))
    else:
        val_ic_dataloader = DataLoader(val_ic_dataset, batch_size_val_ic, generator=gen, shuffle=True)
        val_ic_iter = cycle(val_ic_dataloader)
    
    if val_dataset is not None:
        to_eval = [
            ("train", train_dataloader, train_bc_iter, train_ic_iter),
            ("val", val_dataloader, val_bc_iter, val_ic_iter)
            ]
    else:
        to_eval = [("train", train_dataloader, train_bc_iter, train_ic_iter)]

    if distill_dataset is None:
        distill_iter = itertools.repeat((None, None, None, None))
    else:
        distill_dataloader = DataLoader(distill_dataset, batch_size_distill, generator=gen, shuffle=True)
        distill_iter = cycle(distill_dataloader)
    
    if model.pde_params_in_input is not None:
        pde_params_in_input = [key_idx(k, model.pde) for k in model.pde_params_in_input]
        pde_params_in_input.sort()
    else:
        pde_params_in_input = []
    
    if model.ic_params_in_input is not None:
        ic_params_in_input = [ic_key_idx(k, model.pde) for k in model.ic_params_in_input]
        ic_params_in_input.sort()
    else:
        ic_params_in_input = []

    loss_prefixes = []
    items = model.sys_mode.split("+")
    if "PINN" in items:
        loss_prefixes.append("res")
    elif "Output" in items:
        loss_prefixes.append("out")
    elif "Derivative" in items:
        loss_prefixes.append("der")
    elif "Derivative_x" in items:
        loss_prefixes.append("derx")
    elif "Derivative_t" in items:
        loss_prefixes.append("dert")
    elif "Hessian" in items:
        loss_prefixes.append("hes")
    elif "Hessian_x" in items:
        loss_prefixes.append("hesx")
    elif "Hessian_t" in items:
        loss_prefixes.append("hest")
    
    #n = 80  # Stop decaying after 1000 epochs
    #gamma = 0.95  # decay factor

    #def lr_lambda(epoch):
    #    if epoch < n:
    #        return gamma ** epoch
    #    else:
    #        return gamma ** n  # fixed at last decayed value

    # Set the optimizer and the lr scheduler
    optimizer = Adam(params=model.parameters(), lr=lr_init)
    if scheduler_mode == "ExpDec":
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_mode == "CosAnn":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    early_stopping = False

    for epoch in range(epochs):

        # ----------------------------------- Start of epoch -----------------------------------
        # Put the model in training mode
        model.train()

        if train_steps < 0:
            step_prefix = epoch * N
        else:
            step_prefix = epoch * min(N, train_steps)
        start_time = time.time()

        print(f'\nEpoch: {epoch}, step_prefix: {step_prefix}')

        for step, (train_data, nl_data, bc_data, ic_data, nl_bc_data, nl_ic_data, distill_data) in enumerate(zip(train_iter, nl_iter, train_bc_iter, train_ic_iter, nl_bc_iter, nl_ic_iter, distill_iter)):
            if train_steps >= 0 and step > train_steps:
                break

            # Load batches from dataloaders:
            # ---- Domain data ----
            x_train = train_data[X].to(device).float().requires_grad_(True)
            u_train = train_data[U].to(device).float()
            du_train = train_data[DU].to(device).float()
            d2u_train = train_data[D2U].to(device).float()
            pde_param_values = train_data[PDE_VALUES].to(device).float() # assumed sorted
            # pde_params_in_input sorted
            if pde_params_in_input is not []:
                pde_param_values_train = pde_param_values[:, pde_params_in_input]
                if torch.any(torch.isnan(pde_param_values_train)):
                    raise ValueError("Some pde parameters required in model input are not provided by train dataset.")
            else:
                pde_param_values_train = None
            
            ic_param_values = train_data[IC_VALUES].to(device).float() # assumed sorted
            # ic_params_in_input sorted
            if ic_params_in_input is not []:
                ic_param_values_train = ic_param_values[:, ic_params_in_input]
                if torch.any(torch.isnan(ic_param_values_train)):
                    raise ValueError("Some ic parameters required in model input are not provided by train dataset.")
            else:
                ic_param_values_train = None

            residual_info_keys = train_data[RESIDUAL_KEYS].to(device).int()
            residual_info_values = train_data[RESIDUAL_VALUES].to(device).float()
            residual_info_dict = get_dictionary(residual_info_keys, residual_info_values, model.pde)

            if nl_data[X] is not None:
                x_nl = nl_data[X].to(device).float().requires_grad_(True)
                pde_param_values = nl_data[PDE_VALUES].to(device).float() # assumed sorted
                # pde_params_in_input sorted
                if pde_params_in_input is not []:
                    pde_param_values_nl = pde_param_values[:, pde_params_in_input]
                    if torch.any(torch.isnan(pde_param_values_nl)):
                        raise ValueError("Some pde parameters required in model input are not provided by train dataset.")
                else:
                    pde_param_values_nl = None

                ic_param_values = nl_data[IC_VALUES].to(device).float() # assumed sorted
                # pde_params_in_input sorted
                if ic_params_in_input is not []:
                    ic_param_values_nl = ic_param_values[:, ic_params_in_input]
                    if torch.any(torch.isnan(ic_param_values_nl)):
                        raise ValueError("Some ic parameters required in model input are not provided by unlabeled train dataset.")
                else:
                    ic_param_values_nl = None

                residual_info_keys = nl_data[RESIDUAL_KEYS].to(device).int()
                residual_info_values = nl_data[RESIDUAL_VALUES].to(device).float()
                residual_info_dict_nl = get_dictionary(residual_info_keys, residual_info_values, model.pde)
            else:
                x_nl = None
                pde_param_values_nl = None
                ic_param_values_nl = None
                residual_info_dict_nl = None

            # ---- Boundary data ----
            # -- train --
            if bc_data[X] is not None:
                x_bc = bc_data[X].to(device).float()
                u_bc = bc_data[U].to(device).float()
                du_bc = bc_data[DU].to(device).float()
                outward_normal_bc = bc_data[OUTWARD_NORMAL].to(device).float()
                if pde_params_in_input != []:
                    pde_param_values_bc = bc_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                else:
                    pde_param_values_bc = None
                
                if ic_params_in_input != []:
                    ic_param_values_bc = bc_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                else:
                    ic_param_values_bc = None
                
            else:
                x_bc = None
                u_bc = None
                du_bc = None
                outward_normal_bc = None
                pde_param_values_bc = None
                ic_param_values_bc = None
            
            # -- nl --
            if nl_bc_data[X] is not None:
                x_nl_bc = nl_bc_data[X].to(device).float()
                u_nl_bc = nl_bc_data[U].to(device).float()
                du_nl_bc = nl_bc_data[DU].to(device).float()
                outward_normal_nl_bc = nl_bc_data[OUTWARD_NORMAL].to(device).float()
                if pde_params_in_input != []:
                    pde_param_values_nl_bc = nl_bc_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                else:
                    pde_param_values_nl_bc = None
                
                if ic_params_in_input != []:
                    ic_param_values_nl_bc = nl_bc_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                else:
                    ic_param_values_nl_bc = None
                
            else:
                x_nl_bc = None
                u_nl_bc = None
                du_nl_bc = None
                outward_normal_nl_bc = None
                pde_param_values_nl_bc = None
                ic_param_values_nl_bc = None

            # ---- Initial data ----
            # -- train --
            if ic_data[X] is not None:
                x_ic = ic_data[X].to(device).float()
                u_ic = ic_data[U].to(device).float()
                if pde_params_in_input != []:
                    pde_param_values_ic = ic_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                else:
                    pde_param_values_ic = None

                if ic_params_in_input != []:
                        ic_param_values_ic = ic_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                else:
                    ic_param_values_ic = None

            else:
                x_ic = None
                u_ic = None
                pde_param_values_ic = None
                ic_param_values_ic = None
            
            # -- nl --
            if nl_ic_data[X] is not None:
                x_nl_ic = nl_ic_data[X].to(device).float()
                u_nl_ic = nl_ic_data[U].to(device).float()
                if pde_params_in_input != []:
                    pde_param_values_nl_ic = nl_ic_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                else:
                    pde_param_values_nl_ic = None

                if ic_params_in_input != []:
                        ic_param_values_nl_ic = nl_ic_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                else:
                    ic_param_values_nl_ic = None

            else:
                x_nl_ic = None
                u_nl_ic = None
                pde_param_values_nl_ic = None
                ic_param_values_nl_ic = None

            # ---- Distill data ----
            if distill_data[X] is None:
                x_distill = None
                u_distill = None
                du_distill = None
                d2u_distill = None
                pde_param_values_distill = None
                ic_param_values_distill = None
            else:
                x_distill = distill_data[X].to(device).float().requires_grad_(True)
                u_distill = distill_data[U].to(device).float().requires_grad_(True)
                du_distill = distill_data[DU].to(device).float().requires_grad_(True)
                d2u_distill = distill_data[D2U].to(device).float().requires_grad_(True)
                if pde_params_in_input != []:
                    #.requires_grad_(True) TODO
                    pde_param_values_distill = distill_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                    if torch.any(torch.isnan(pde_param_values_distill)):
                        raise ValueError("Some pde parameters required in distill model input are not provided by distill dataset.")
                else:
                    pde_param_values_distill = None
                
                if ic_params_in_input != []:
                    #.requires_grad_(True) TODO
                    ic_param_values_distill = distill_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                    if torch.any(torch.isnan(ic_param_values_distill)):
                        raise ValueError("Some ic parameters required in distill model input are not provided by distill dataset.")
                else:
                    ic_param_values_distill = None
            
            optimizer.zero_grad()
            
            loss = model.loss_fn(
                x = x_train,
                pde_params=pde_param_values_train,
                ic_params=ic_param_values_train,
                u = u_train,
                du = du_train,
                d2u = d2u_train,
                residual_data=residual_info_dict,
                x_bc = x_bc,
                n = outward_normal_bc,
                pde_params_bc=pde_param_values_bc,
                ic_params_bc=ic_param_values_bc,
                u_bc = u_bc,
                du_bc = du_bc,
                x_ic = x_ic,
                pde_params_ic=pde_param_values_ic,
                ic_params_ic=ic_param_values_ic,
                u_ic = u_ic,

                x_nl = x_nl,
                pde_params_nl=pde_param_values_nl,
                ic_params_nl=ic_param_values_nl,
                residual_data_nl=residual_info_dict_nl,
                x_nl_bc = x_nl_bc,
                n_nl = outward_normal_nl_bc,
                pde_params_nl_bc=pde_param_values_nl_bc,
                ic_params_nl_bc=ic_param_values_nl_bc,
                u_nl_bc = u_nl_bc,
                du_nl_bc = du_nl_bc,
                x_nl_ic = x_nl_ic,
                pde_params_nl_ic=pde_param_values_nl_ic,
                ic_params_nl_ic=ic_param_values_nl_ic,
                u_nl_ic = u_nl_ic,
                
                x_distill = x_distill,
                pde_params_distill=pde_param_values_distill,
                ic_params_distill=ic_param_values_distill,
                u_distill = u_distill,
                du_distill = du_distill,
                d2u_distill = d2u_distill,
                delayed_weights=delayed_weights
                )
            
            # --- Backward pass ---
            loss.backward()
            stats_dict["train_loss"].append(loss.item())

            # --- Gradient clipping ---
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if step % 4 == 0:
                # --- Compute total gradient norm ---
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += (p.grad.data.norm(2).item())**2
                total_grad_norm = total_grad_norm**0.5
                stats_dict["train_loss_grad_norm"].append(total_grad_norm)

                # --- Early stopping ---
                if early_stop_value is not None and total_grad_norm < early_stop_value:
                    print(f"Early stopping at epoch {epoch} (grad norm={total_grad_norm:.3e})")
                    early_stopping = True
                    break
            
            # Call the optimizer
            optimizer.step()

        if scheduler_mode is not None:
            scheduler.step()
        
        # ----------------------------------- End of epoch -----------------------------------

        # ------------------------------ Epoch evaluation ------------------------------

        # Append the epoch time
        stop_time = time.time()
        epoch_time = stop_time-start_time
        print(f'Epoch time: {epoch_time}')
        stats_dict["times"].append(epoch_time)

        stats_dict["weights"]["step_list"].append(epoch)#step_prefix + step)
        stats_dict["weights"]["res_loss"].append(model.res_weight)
        stats_dict["weights"]["out_loss"].append(model.out_weight)
        stats_dict["weights"]["der_loss"].append(model.der_weight)
        stats_dict["weights"]["derx_loss"].append(model.derx_weight)
        stats_dict["weights"]["dert_loss"].append(model.dert_weight)
        stats_dict["weights"]["hes_loss"].append(model.hes_weight)
        stats_dict["weights"]["hesx_loss"].append(model.hesx_weight)
        stats_dict["weights"]["hest_loss"].append(model.hest_weight)
        stats_dict["weights"]["bc_loss"].append(model.bc_weight)
        stats_dict["weights"]["ic_loss"].append(model.ic_weight)
        stats_dict["weights"]["nl_loss"].append(model.nl_weight)
        stats_dict["weights"]["nl_bc_loss"].append(model.nl_bc_weight)
        stats_dict["weights"]["nl_ic_loss"].append(model.nl_ic_weight)
        stats_dict["weights"]["distill_out_loss"].append(model.distill_out_weight)
        stats_dict["weights"]["distill_der_loss"].append(model.distill_der_weight)
        stats_dict["weights"]["distill_derx_loss"].append(model.distill_derx_weight)
        stats_dict["weights"]["distill_dert_loss"].append(model.distill_dert_weight)
        stats_dict["weights"]["distill_hes_loss"].append(model.distill_hes_weight)
        stats_dict["weights"]["distill_hesx_loss"].append(model.distill_hesx_weight)
        stats_dict["weights"]["distill_hest_loss"].append(model.distill_hest_weight)
        stats_dict["weights"]["ewc_loss"].append(model.ewc_weight)

        stats_dict["conflicts"]["step_list"].append(epoch)#(step_prefix + step)
        stats_dict["conflicts"]["res_loss"].append(1.0)
        stats_dict["conflicts"]["out_loss"].append(model.res_conflicts["out"])
        stats_dict["conflicts"]["der_loss"].append(model.res_conflicts["der"])
        stats_dict["conflicts"]["derx_loss"].append(model.res_conflicts["derx"])
        stats_dict["conflicts"]["dert_loss"].append(model.res_conflicts["dert"])
        stats_dict["conflicts"]["hes_loss"].append(model.res_conflicts["hes"])
        stats_dict["conflicts"]["hesx_loss"].append(model.res_conflicts["hesx"])
        stats_dict["conflicts"]["hest_loss"].append(model.res_conflicts["hest"])
        stats_dict["conflicts"]["bc_loss"].append(model.res_conflicts["bc"])
        stats_dict["conflicts"]["ic_loss"].append(model.res_conflicts["ic"])
        stats_dict["conflicts"]["nl_loss"].append(model.res_conflicts["nl"])
        stats_dict["conflicts"]["nl_bc_loss"].append(model.res_conflicts["nl_bc"])
        stats_dict["conflicts"]["nl_ic_loss"].append(model.res_conflicts["nl_ic"])
        stats_dict["conflicts"]["distill_out_loss"].append(model.res_conflicts["distill_out"])
        stats_dict["conflicts"]["distill_der_loss"].append(model.res_conflicts["distill_der"])
        stats_dict["conflicts"]["distill_derx_loss"].append(model.res_conflicts["distill_derx"])
        stats_dict["conflicts"]["distill_dert_loss"].append(model.res_conflicts["distill_dert"])
        stats_dict["conflicts"]["distill_hes_loss"].append(model.res_conflicts["distill_hes"])
        stats_dict["conflicts"]["distill_hesx_loss"].append(model.res_conflicts["distill_hesx"])
        stats_dict["conflicts"]["distill_hest_loss"].append(model.res_conflicts["distill_hest"])
        stats_dict["conflicts"]["ewc_loss"].append(model.res_conflicts["ewc"])
        
        stats_dict["grad_norms"]["step_list"].append(epoch)#(step_prefix + step)
        stats_dict["grad_norms"]["res_loss"].append(model.res_last_norm)
        stats_dict["grad_norms"]["out_loss"].append(model.out_last_norm)
        stats_dict["grad_norms"]["der_loss"].append(model.der_last_norm)
        stats_dict["grad_norms"]["derx_loss"].append(model.derx_last_norm)
        stats_dict["grad_norms"]["dert_loss"].append(model.dert_last_norm)
        stats_dict["grad_norms"]["hes_loss"].append(model.hes_last_norm)
        stats_dict["grad_norms"]["hesx_loss"].append(model.hesx_last_norm)
        stats_dict["grad_norms"]["hest_loss"].append(model.hest_last_norm)
        stats_dict["grad_norms"]["bc_loss"].append(model.bc_last_norm)
        stats_dict["grad_norms"]["ic_loss"].append(model.ic_last_norm)
        stats_dict["grad_norms"]["nl_loss"].append(model.nl_last_norm)
        stats_dict["grad_norms"]["nl_bc_loss"].append(model.nl_bc_last_norm)
        stats_dict["grad_norms"]["nl_ic_loss"].append(model.nl_ic_last_norm)
        stats_dict["grad_norms"]["distill_out_loss"].append(model.distill_out_last_norm)
        stats_dict["grad_norms"]["distill_der_loss"].append(model.distill_der_last_norm)
        stats_dict["grad_norms"]["distill_derx_loss"].append(model.distill_derx_last_norm)
        stats_dict["grad_norms"]["distill_dert_loss"].append(model.distill_dert_last_norm)
        stats_dict["grad_norms"]["distill_hes_loss"].append(model.distill_hes_last_norm)
        stats_dict["grad_norms"]["distill_hesx_loss"].append(model.distill_hesx_last_norm)
        stats_dict["grad_norms"]["distill_hest_loss"].append(model.distill_hest_last_norm)
        stats_dict["grad_norms"]["ewc_loss"].append(model.ewc_last_norm)
        stats_dict["grad_norms"]["weighted_loss"].append(stats_dict["train_loss_grad_norm"][-1])

        if (step_prefix + step) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                for key, dataloader, bc_iter, ic_iter in to_eval:
                    # Compute and average the loss over the val dataloader
                    sum_dict = {}
                    for k in LOSS_TERMS + ["weighted_loss"]:
                        sum_dict[k] = 0.0

                    for data, bc_data, ic_data, nl_data, nl_bc_data, nl_ic_data, distill_data in zip(dataloader, bc_iter, ic_iter, nl_iter, nl_bc_iter, nl_ic_iter, distill_iter):
                        # Load batches from dataloaders
                        x = data[X].to(device).float().requires_grad_(True)
                        u = data[U].to(device).float()
                        du = data[DU].to(device).float()
                        d2u = data[D2U].to(device).float()
                        if pde_params_in_input != []:
                            pde_param_values = data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                        else:
                            pde_param_values = None
                        
                        if ic_params_in_input != []:
                            ic_param_values = data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                        else:
                            ic_param_values = None

                        residual_info_keys = data[RESIDUAL_KEYS].to(device).int()
                        residual_info_values = data[RESIDUAL_VALUES].to(device).float()
                        residual_info_dict = get_dictionary(residual_info_keys, residual_info_values, model.pde)

                        # Unlabeled data -----
                        if nl_data[X] is not None:
                            x_nl = nl_data[X].to(device).float().requires_grad_(True)
                            if pde_params_in_input != []:
                                pde_param_values_nl = nl_data[PDE_VALUES].to(device).float()[:, pde_params_in_input] # assumed sorted
                            else:
                                pde_param_values_nl = None

                            if ic_params_in_input != []:
                                ic_param_values_nl = nl_data[IC_VALUES].to(device).float()[:, ic_params_in_input] # assumed sorted
                            else:
                                ic_param_values_nl = None

                            residual_info_keys = nl_data[RESIDUAL_KEYS].to(device).int()
                            residual_info_values = nl_data[RESIDUAL_VALUES].to(device).float()
                            residual_info_dict_nl = get_dictionary(residual_info_keys, residual_info_values, model.pde)

                        # Boundary data -------
                        if bc_data[X] is not None:
                            x_bc = bc_data[X].to(device).float()
                            u_bc = bc_data[U].to(device).float()
                            du_bc = bc_data[DU].to(device).float()
                            outward_normal_bc = bc_data[OUTWARD_NORMAL].to(device).float()
                            if pde_params_in_input != []:
                                pde_param_values_bc = bc_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                            else:
                                pde_param_values_bc = None
                            
                            if ic_params_in_input != []:
                                ic_param_values_bc = bc_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                            else:
                                ic_param_values_bc = None

                        else:
                            x_bc = None
                            u_bc = None
                            du_bc = None
                            outward_normal_bc = None
                            pde_param_values_bc = None
                            ic_param_values_bc = None

                        if ic_data[X] is not None:
                            x_ic = ic_data[X].to(device).float()
                            u_ic = ic_data[U].to(device).float()
                            if pde_params_in_input != []:
                                pde_param_values_ic = ic_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                            else:
                                pde_param_values_ic = None
                            
                            if ic_params_in_input != []:
                                ic_param_values_ic = ic_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                            else:
                                ic_param_values_ic = None

                        else:
                            x_ic = None
                            u_ic = None
                            pde_param_values_ic = None
                            ic_param_values_ic = None
                        
                        # nl boundary data -------
                        if nl_bc_data[X] is not None:
                            x_nl_bc = nl_bc_data[X].to(device).float()
                            u_nl_bc = nl_bc_data[U].to(device).float()
                            du_nl_bc = nl_bc_data[DU].to(device).float()
                            outward_normal_nl_bc = nl_bc_data[OUTWARD_NORMAL].to(device).float()
                            if pde_params_in_input != []:
                                pde_param_values_nl_bc = nl_bc_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                            else:
                                pde_param_values_nl_bc = None
                            
                            if ic_params_in_input != []:
                                ic_param_values_nl_bc = nl_bc_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                            else:
                                ic_param_values_nl_bc = None

                        else:
                            x_nl_bc = None
                            u_nl_bc = None
                            du_nl_bc = None
                            outward_normal_nl_bc = None
                            pde_param_values_nl_bc = None
                            ic_param_values_nl_bc = None

                        if nl_ic_data[X] is not None:
                            x_nl_ic = nl_ic_data[X].to(device).float()
                            u_nl_ic = nl_ic_data[U].to(device).float()
                            if pde_params_in_input != []:
                                pde_param_values_nl_ic = nl_ic_data[PDE_VALUES].to(device).float()[:, pde_params_in_input]
                            else:
                                pde_param_values_nl_ic = None
                            
                            if ic_params_in_input != []:
                                ic_param_values_nl_ic = nl_ic_data[IC_VALUES].to(device).float()[:, ic_params_in_input]
                            else:
                                ic_param_values_nl_ic = None

                        else:
                            x_nl_ic = None
                            u_nl_ic = None
                            pde_param_values_nl_ic = None
                            ic_param_values_nl_ic = None

                        # ---- Distill data ----
                        if distill_data[X] is None:
                            x_distill = None
                            u_distill = None
                            du_distill = None
                            d2u_distill = None
                            pde_param_values_distill = None
                            ic_param_values_distill = None
                        else:
                            x_distill = distill_data[X].to(device).float().requires_grad_(True)
                            u_distill = distill_data[U].to(device).float().requires_grad_(True)
                            du_distill = distill_data[DU].to(device).float().requires_grad_(True)
                            d2u_distill = distill_data[D2U].to(device).float().requires_grad_(True)
                            if pde_params_in_input != []:
                                pde_param_values_distill = distill_data[5].to(device).float()[:, pde_params_in_input]
                            else:
                                pde_param_values_distill = None
                            
                            if ic_params_in_input != []:
                                ic_param_values_distill = distill_data[5].to(device).float()[:, ic_params_in_input]
                            else:
                                ic_param_values_distill = None
                            
                        # Evaluate the evaluation loss on val data
                        eval_dict = model.eval_losses(
                            x = x,
                            pde_params = pde_param_values,
                            ic_params = ic_param_values,
                            u = u,
                            du = du,
                            d2u = d2u,
                            residual_data = residual_info_dict,
                            x_bc = x_bc,
                            n = outward_normal_bc,
                            pde_params_bc = pde_param_values_bc,
                            ic_params_bc = ic_param_values_bc,
                            u_bc = u_bc,
                            du_bc = du_bc,
                            x_ic = x_ic,
                            pde_params_ic = pde_param_values_ic,
                            ic_params_ic = ic_param_values_ic,
                            u_ic = u_ic,

                            x_nl = x_nl,
                            pde_params_nl = pde_param_values_nl,
                            ic_params_nl = ic_param_values_nl,
                            residual_data_nl = residual_info_dict_nl,
                            x_nl_bc = x_nl_bc,
                            n_nl = outward_normal_nl_bc,
                            pde_params_nl_bc = pde_param_values_nl_bc,
                            ic_params_nl_bc = ic_param_values_nl_bc,
                            u_nl_bc = u_nl_bc,
                            du_nl_bc = du_nl_bc,
                            x_nl_ic = x_nl_ic,
                            pde_params_nl_ic = pde_param_values_nl_ic,
                            ic_params_nl_ic = ic_param_values_nl_ic,
                            u_nl_ic = u_nl_ic,
                            
                            x_distill = x_distill,
                            pde_params_distill = pde_param_values_distill,
                            ic_params_distill = ic_param_values_distill,
                            u_distill = u_distill,
                            du_distill = du_distill,
                            d2u_distill = d2u_distill,
                            delayed_weights = delayed_weights
                            )

                        # Accumulate val loss values
                        for k in eval_dict.keys():
                            sum_dict[k] += eval_dict[k].item()

                    # Append val loss average values
                    for k in sum_dict.keys():
                        stats_dict[key][k].append(sum_dict[k] / len(dataloader))

                    stats_dict[key]["step_list"].append(epoch)#(step_prefix + step)

                    print(f"{key} weighted loss: {stats_dict[key]['weighted_loss'][-1]}")
                    if math.isnan(stats_dict[key]['weighted_loss'][-1]):
                        #raise ArithmeticError("You get a nan value (instablility issue).\nSuggestion: Repeat this run using an ubber bounded weighting scheme (e.g. Norm1).\n")
                        #to_report = torch.inf
                        #trial.report(to_report, step=epoch)
                        #raise optuna.exceptions.TrialPruned()
                        return None
                    print(f"{key} out loss: {stats_dict[key]['out_loss'][-1]}")
                    if model.ewc_mode == "On":
                        print(f"{key} ewc loss: {stats_dict[key]['ewc_loss'][-1]}")

                # Report intermediate result to Optuna
                if bc_data[X] is not None and "bc" not in loss_prefixes:
                    loss_prefixes.append("bc")
                if ic_data[X] is not None and "ic" not in loss_prefixes:
                    loss_prefixes.append("ic")

                if val_dataset is not None:
                    to_report = sum([stats_dict["val"][f"{prefix}_loss"][-1] for prefix in loss_prefixes])
                else:
                    to_report = sum([stats_dict["train"][f"{prefix}_loss"][-1] for prefix in loss_prefixes])
                trial.report(to_report, step=epoch)
        
                # Check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
        if early_stopping:
            break
    return stats_dict