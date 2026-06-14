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
from torch.utils.data import TensorDataset, Subset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from model2 import Pinn
from data_utils import subsample, replace_labels, extract_TensorDataset, extract_boundary, extract_interior, include_time_in_input, exclude_space_in_input
from load_store_utils import resume_model
from physics_task import PhysicsTask, AdvectionReactionDiffusionTask, StationaryAllenCahnTask, OutputTask, DerivativeTask, Derivative2Task, SpatialDerivativeTask, SpatialDerivative2Task, TemporalDerivativeTask, TemporalDerivative2Task, ICTask, NeumannBCTask, DirichletBCTask
from typing import List, Callable, Iterator, Tuple
import itertools
from itertools import cycle
import time

X = 0
U = 1
DU = 2
D2U = 3
PARAMS = 4
BC = 5

def get_iterators(datas: List[TensorDataset], batch_size: float, seed: int) -> Tuple[List[Iterator], int]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    batch_sizes = []
    data_sizes = []
    max_len_idx = 0
    max_len = -1

    for i, task_data in enumerate(datas):
        N = len(task_data)
        if N > max_len:
            max_len = N
            max_len_idx = i

        if batch_size < N:
            N = N - N % batch_size
            batch_sizes.append(batch_size)

        else:
            batch_sizes.append(N)

        data_sizes.append(N)
        print(f"batch size of task {i} = {batch_sizes[-1]}")
        print(f"dataset size of task {i}) = {data_sizes[-1]}")

    for i, data_size in enumerate(data_sizes):
        datas[i] = Subset(datas[i], list(range(data_size)))
    
    iterators = [None for _ in datas]
    
    for i, task_data in enumerate(datas):
        dataloader = DataLoader(task_data, batch_sizes[i], generator=gen, shuffle=True)#, drop_last=True)
        if i == max_len_idx:
            iterators[i] = dataloader
            steps_per_epoch = len(dataloader)
        else:
            iterators[i] = cycle(dataloader)
    return iterators, steps_per_epoch

# =============================================== Run configuration ===============================================

def train_ARD(
        model: Pinn,

        ff_size: int,
        ff_frequency_var: float,

        n_param_input: int,

        new_tasks: List[PhysicsTask],
        new_datas: List[TensorDataset],
        new_weights: List[float],

        recall_tasks: List[PhysicsTask],
        recall_datas: List[TensorDataset],
        recall_weights: List[float],

        val_tasks: List[PhysicsTask],
        val_datas: TensorDataset,

        monitoring_tasks: List[PhysicsTask],
        monitoring_datas: TensorDataset,
        
        dwa: bool,
        dwa_mode: str,
        dwa_warm_up: int,
        dwa_moving_avg_factor: float,
        dwa_moving_avg_frequency: int,

        ewc: bool,
        fisher_diag_avg: torch.Tensor,
        fisher_diag_new: torch.Tensor,
        ewc_weight: float,
        ewc_auto_weighting: bool,
        ewc_warm_up: int,
        ewc_decay_factor: float,
        ewc_moving_avg_factor: float,

        monitor_conflicts: bool,
        conflict_reference_task_idx: int,
        monitor_weights: bool,
        monitor_grads: bool,
        monitor_grad_norms: bool,

        clip_grad: bool,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        steps: int,
        seed=42,
        device="cpu"
):
    # Set the model never model-selected (fixed) attributes
    model.dwa_mode = dwa_mode
    model.dwa_warm_up = dwa_warm_up
    model.moving_avg_frequency = dwa_moving_avg_frequency

    model.ewc_auto_weighting = ewc_auto_weighting

    def objective(trial):
        # Sample hyperparameters
        if type(batch_size) is list:
            trial_batch_size = trial.suggest_categorical("batch_size", batch_size)
        else:
            trial_batch_size = batch_size
        
        train_iterators, _ = get_iterators(datas = new_datas + recall_datas, batch_size = trial_batch_size, seed = seed)
        eval_iterators, _ = get_iterators(datas = val_datas + monitoring_datas, batch_size = trial_batch_size, seed = seed)
        
        if type(learning_rate) is list:
            trial_learning_rate = trial.suggest_categorical("learning_rate", learning_rate)
        else:
            trial_learning_rate = learning_rate

        if type(ff_frequency_var) is list:
            trial_ff_frequency_var = trial.suggest_categorical("ff_frequency_var", ff_frequency_var)
        else:
            trial_ff_frequency_var = ff_frequency_var
        
        if type(ff_size) is list:
            trial_ff_size = trial.suggest_categorical("ff_size", ff_size)
        else:
            trial_ff_size = ff_size
        
        model.set_ff(fourier_features=trial_ff_size, frequency_variance=trial_ff_frequency_var)

        if ewc:
            model.ewc_mode = "On"

            if type(ewc_weight) is list:
                model.ewc_weight = trial.suggest_categorical("ewc_weight", ewc_weight)
            else:
                model.ewc_weight = ewc_weight

            if type(ewc_warm_up) is list:
                model.ewc_warm_up = trial.suggest_categorical("ewc_warm_up", ewc_warm_up)
            else:
                model.ewc_warm_up = ewc_warm_up

            if type(ewc_decay_factor) is list:
                model.decay = trial.suggest_categorical("ewc_decay", ewc_decay)
            else:
                model.decay = ewc_decay_factor
            
            if type(ewc_moving_avg_factor) is list:
                trial_ewc_moving_avg_factor = trial.suggest_categorical("ewc_moving_avg_factor", ewc_moving_avg_factor)
            else:
                trial_ewc_moving_avg_factor = ewc_moving_avg_factor
            
            model.ewc_fisher_diag = trial_ewc_moving_avg_factor * fisher_diag_new + (1 - trial_ewc_moving_avg_factor) * fisher_diag_avg
        else:
            model.ewc_mode = "Off"
        
        if dwa:
            if type(dwa_moving_avg_factor) is list:
                model.alpha = trial.suggest_categorical("dwa_moving_avg_factor", dwa_moving_avg_factor)
            else:
                model.alpha = dwa_moving_avg_factor
        
        for i, weight in enumerate(new_weights):
            if type(weight) is list:
                new_tasks[i].weight = trial.suggest_categorical(f"new_weight{i}", weight)
            else:
                new_tasks[i].weight = weight
        
        for i, weight in enumerate(recall_weights):
            if type(weight) is list:
                recall_tasks[i].weight = trial.suggest_categorical(f"recall_weight{i}", weight)
            else:
                recall_tasks[i].weight = weight
        
        for task in new_tasks + recall_tasks + val_tasks + monitoring_tasks:
            task.grad_norm = None
            task.grad = None
            task.conflict = None
            task.loss_value = None
        
        model.task_list = new_tasks + recall_tasks
        model.eval_task_list = val_tasks + monitoring_tasks

        optimizer = Adam(params=model.parameters(), lr=trial_learning_rate)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        stats_dict = {}
        #stats_dict["train_steps"] = []
        stats_dict["epochs"] = []
        stats_dict["times"] = []
        stats_dict["loss"]["weighted_loss"] = []

        for epoch in range(epochs):

            # ----------------------------------- Start of epoch -----------------------------------

            # Put the model in training mode
            model.train()

            if steps < 0:
                step_prefix = epoch * N
            else:
                step_prefix = epoch * min(N, steps)
            start_time = time.time()

            stats_dict["epochs"].append(epoch)
            #stats_dict["train_loss_per_step"] = []

            for step, batches in enumerate(zip(*train_iterators)):
                if steps >= 0 and step > steps:
                    break

                #stats_dict["train_steps"].append(step_prefix + step)
                print(f'\nEpoch: {epoch}, step_prefix: {step_prefix}')

                x_list = []
                labels = {
                    "u": [],
                    "du": [],
                    "d2u": []
                }
                input_param_list = []
                bc_list = []

                for batch in batches:
                    x_list.append(batch[X].to(device).float().requires_grad_(True))
                    labels["u"].append(batch[U].to(device).float())
                    labels["du"].append(batch[DU].to(device).float())
                    labels["d2u"].append(batch[D2U].to(device).float())
                    input_param_list.append(batch[PARAMS].to(device).float())
                    bc_list.append(batch[BC].to(device).float())

                optimizer.zero_grad()

                loss = model.train_loss(
                    x_list = x_list,
                    input_param_list = input_param_list,
                    labels = labels
                )

                # --- Backward pass ---
                loss.backward()
                #stats_dict["train_loss_per_step"].append(loss.item())

                # --- Gradient clipping ---
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # --- Compute total gradient norm ---
                #total_grad_norm = 0.0
                #for p in model.parameters():
                #    if p.grad is not None:
                #        total_grad_norm += (p.grad.data.norm(2).item())**2
                #total_grad_norm = total_grad_norm ** 0.5
                #stats_dict["train_loss_grad_norm"].append(total_grad_norm)

                # Call the optimizer
                optimizer.step()

            # ----------------------------------- End of epoch -----------------------------------

            lr_scheduler.step()

            stop_time = time.time()
            epoch_time = stop_time - start_time
            print(f'Epoch time: {epoch_time}')
            stats_dict["times"].append(epoch_time)
            #stats_dict["loss"]["weighted_loss"].append(torch.mean(stats_dict["train_loss_per_step"]))#[step_prefix : step_prefix + step]))

            # ------------------------------ Epoch evaluation ------------------------------

            if (step_prefix + step) % eval_every == 0:
                stats_dict["eval_epochs"].append(epoch)
                model.eval()
                with torch.set_grad_enabled(monitor_grads or monitor_grad_norms or monitor_conflicts): #TODO: save eval stats correctly
                    weighted_loss_per_batch = []
                    for task in model.eval_task_list:
                        stats_dict["loss_per_batch"][task.id] = []
                        stats_dict["conflicts_per_batch"][task.id] = []
                        stats_dict["grad_norms_per_batch"][task.id] = []
                        stats_dict["grads_per_batch"][task.id] = []

                    for batches in zip(*eval_iterators):
                        x_list = []
                        labels = {
                            "u": [],
                            "du": [],
                            "d2u": []
                        }      

                        input_param_list = []
                        bc_list = []

                        for batch in batches:
                            x_list.append(batch[X].to(device).float().requires_grad_(True))
                            labels["u"].append(batch[U].to(device).float())
                            labels["du"].append(batch[DU].to(device).float())
                            labels["d2u"].append(batch[D2U].to(device).float())
                            input_param_list.append(batch[PARAMS].to(device).float())
                            bc_list.append(batch[BC].to(device).float())

                        model.eval_loss(
                            x_list = x_list,
                            input_param_list = input_param_list,
                            labels = labels
                        )

                        weighted_loss = 0.0
                        for val_task, train_task in zip(val_tasks, model.task_list):
                            weighted_loss += train_task.weight * val_task.loss_value

                        weighted_loss_per_batch.append(weighted_loss)

                        for task in model.eval_task_list:
                            stats_dict["loss_per_batch"][task.id].append(task.loss_value)
                            stats_dict["conflicts_per_batch"][task.id].append(task.conflict)
                            stats_dict["grad_norms_per_batch"][task.id].append(task.grad_norm)
                            stats_dict["grads_per_batch"][task.id].append(task.grad)

                    stats_dict["val_weighted_loss"].append(torch.mean(weighted_loss_per_batch))
                    for task in model.eval_task_list:
                        stats_dict["loss"][task.id].append(torch.mean(stats_dict["loss_per_batch"][task.id]))
                        stats_dict["conflicts"][task.id].append(torch.mean(stats_dict["conflicts_per_batch"][task.id]))
                        stats_dict["grad_norms"][task.id].append(torch.mean(stats_dict["grad_norms_per_batch"][task.id]))
                        stats_dict["grads"][task.id].append(torch.mean(stats_dict["grads_per_batch"][task.id]))


                    print(f"Validation weighted loss: {stats_dict['val_weighted_loss'][-1]}")
                    if math.isnan(stats_dict['val_weighted_loss'][-1]):
                        #raise ArithmeticError("You get a nan value.")
                        #to_report = torch.inf
                        #trial.report(to_report, step=epoch)
                        #raise optuna.exceptions.TrialPruned()
                        return None
                    print(f"Output loss: {stats_dict['loss']['u'][-1]}")

                    # Report intermediate result to Optuna
                    to_report = sum([stats_dict["loss"][task.id] for task in val_tasks])
                    trial.report(to_report, step=epoch)

                    # Check if the trial should be pruned
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
        return stats_dict

    
def train_step_ARD(
        
        batch_list: List[torch.Tensor],# for tuple in zip(*list)
        input_param_list: List[torch.Tensor],
        labels: dict,
):

    x_list = [batch[X_ARD]
    input_param_list = 

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
            param_values_train = None
            if pde_params_in_input is not []:
                pde_param_values_train = pde_param_values[:, pde_params_in_input]
                if torch.any(torch.isnan(pde_param_values_train)):
                    raise ValueError("Some pde parameters required in model input are not provided by train dataset.")
                param_values_train = pde_param_values_train
            else:
                pde_param_values_train = None
            
            ic_param_values = train_data[IC_VALUES].to(device).float() # assumed sorted
            # ic_params_in_input sorted
            if ic_params_in_input is not []:
                ic_param_values_train = ic_param_values[:, ic_params_in_input]
                if torch.any(torch.isnan(ic_param_values_train)):
                    raise ValueError("Some ic parameters required in model input are not provided by train dataset.")
                if param_values_train is not None:
                    param_values_train = torch.cat([param_values_train, ic_param_values_train], dim=-1)
                else:
                    param_values_train = ic_param_values
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