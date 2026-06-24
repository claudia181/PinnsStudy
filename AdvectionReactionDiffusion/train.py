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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
import argparse
import os
import math
import optuna
from optuna.trial import TrialState
import shutil

from model2 import Pinn
from data_utils import extract_boundary, extract_interior, get_iterators
from load_store_utils import resume_model, save_model
from physics_task import PhysicsTask
from phy_sys_dataset import PhySysDataset

from typing import List
import time

def train(
        model: Pinn,

        ff: bool,
        dwa: bool,
        ewc: bool,

        new_tasks: List[PhysicsTask],
        new_datas: List[PhySysDataset],
        new_weights: List[float] | List[List[float]],

        recall_tasks: List[PhysicsTask],
        recall_datas: List[PhySysDataset],
        recall_weights: List[float] | List[List[float]],

        val_tasks: List[PhysicsTask],
        val_datas: List[PhySysDataset],

        monitoring_tasks: List[PhysicsTask],
        monitoring_datas: List[PhySysDataset],

        monitor_conflicts: bool,
        monitor_weights: bool,
        monitor_grad_norms: bool,
        eval_every: int,

        clip_grad: bool,
        batch_size: int | List[int],
        learning_rate: float | List[float],
        epochs: int,
        steps: int,
        destination_folder: str,

        n_trials: int = 1,
        seed: int = 42,
        device: str = "cpu",

        ff_size: int | List[int] = -1,
        ff_frequency_var: float | List[float] = 1,

        dwa_mode: str = "Off",
        dwa_warm_up: int = 3,
        dwa_moving_avg_factor: float | List[float] = 0.9,
        dwa_moving_avg_frequency: int = 1,

        fisher_diag_avg: torch.Tensor = None,
        fisher_diag_new: torch.Tensor = None,
        ewc_weight: float | List[float] = 0.0,
        ewc_auto_weighting: bool = False,
        ewc_warm_up: int | List[int] = 1,
        ewc_decay_factor: float | List[float] = 1.0,
        ewc_moving_avg_factor: float | List[float] = 1.0,

        conflict_reference_task_idx: int = 0
):
    # Set the model never model-selected (fixed) attributes
    model.conflict_reference_task = conflict_reference_task_idx
    model.monitor_conflicts = monitor_conflicts
    if dwa:
        model.dwa_mode = dwa_mode
        model.dwa_warm_up = dwa_warm_up
        model.moving_avg_frequency = dwa_moving_avg_frequency
    if ewc:
        model.ewc_auto_weighting = ewc_auto_weighting
    model.device = device

    # ************************************** Define the objective function to run **************************************
    def objective(trial):
        # Sample hyperparameters
        if type(batch_size) is list:
            trial_batch_size = trial.suggest_categorical("batch_size", batch_size)
        else:
            trial_batch_size = batch_size
        
        train_iterators, train_steps_per_epoch = get_iterators(datas = new_datas + recall_datas, batch_size = trial_batch_size, seed = seed)
        eval_iterators, eval_steps_per_epoch = get_iterators(datas = val_datas + monitoring_datas, batch_size = trial_batch_size, seed = seed)
        
        if type(learning_rate) is list:
            trial_learning_rate = trial.suggest_categorical("learning_rate", learning_rate)
        else:
            trial_learning_rate = learning_rate

        if ff:
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
                model.decay = trial.suggest_categorical("ewc_decay", ewc_decay_factor)
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
        stats_dict["train_epochs"] = []
        stats_dict["eval_epochs"] = []
        stats_dict["times"] = []
        stats_dict["train_loss"] = []

        stats_dict["weights"] = {}
        for task in model.task_list:
            stats_dict["weights"][task.id] = []

        stats_dict["eval_loss"] = {}
        stats_dict["eval_loss"]["weighted_loss"] = []
        stats_dict["eval_conflicts"] = {}
        stats_dict["eval_grad_norms"] = {}

        for task in model.eval_task_list:
            stats_dict["eval_loss"][task.id] = []
            stats_dict["eval_conflicts"][task.id] = []
            stats_dict["eval_grad_norms"][task.id] = []


        for epoch in range(epochs):

            # ----------------------------------- Start of epoch -----------------------------------

            # Put the model in training mode
            model.train()

            if steps < 0:
                step_prefix = epoch * train_steps_per_epoch
            else:
                step_prefix = epoch * min(train_steps_per_epoch, steps)
            start_time = time.time()

            stats_dict["train_epochs"].append(epoch)

            train_loss_epoch = 0.0

            for step, batches in enumerate(zip(*train_iterators)):
                if steps >= 0 and step > steps:
                    break

                print(f"\nepoch: {epoch}, batch: {step}, step: {step_prefix + step}")

                x_list = []
                labels = {
                    "u": [],
                    "du": [],
                    "d2u": []
                }
                input_param_list = []
                bc_list = []

                for i, batch in enumerate(batches):
                    for key in batch.keys():
                        if key == "spacetime":
                            x_list.append(batch[key].to(device).float().requires_grad_(True))
                        elif key == "param":
                            input_param_list.append(batch[key].to(device).float())
                        elif key in ["u", "du", "d2u"]:
                            labels[key].append(batch[key].to(device).float())
                        elif key == "bc":
                            bc_list.append(batch[key].to(device).float())
                        else:
                            raise ValueError(f"Unknown key '{key}'.")

                optimizer.zero_grad()

                loss = model.train_loss(
                    x_list = x_list,
                    input_param_list = input_param_list,
                    labels = labels
                )

                # --- Backward pass ---
                loss.backward()
                train_loss_epoch += loss.item()

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
            stats_dict["train_loss"].append(train_loss_epoch / train_steps_per_epoch)

            # ------------------------------ Epoch evaluation ------------------------------

            if (step_prefix + step) % eval_every == 0:
                stats_dict["eval_epochs"].append(epoch)
                model.eval()
                with torch.set_grad_enabled(monitor_grad_norms or monitor_conflicts):
                    if monitor_weights:
                        for task in model.task_list:
                            stats_dict["weights"][task.id].append(task.weight)
                    loss_epoch = {}
                    if monitor_conflicts:
                        conflict_epoch = {}
                    if monitor_grad_norms:
                        grad_norm_epoch = {}

                    eval_weighted_loss_epoch = 0.0
                    for task in model.eval_task_list:
                        loss_epoch[task.id] = 0.0
                        if monitor_conflicts:
                            conflict_epoch[task.id] = 0.0
                        if monitor_grad_norms:
                            grad_norm_epoch[task.id] = 0.0

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
                            for key in batch.keys():
                                if key == "spacetime":
                                    x_list.append(batch[key].to(device).float().requires_grad_(True))
                                elif key == "param":
                                    input_param_list.append(batch[key].to(device).float())
                                elif key in ["u", "du", "d2u"]:
                                    labels[key].append(batch[key].to(device).float())
                                elif key == "bc":
                                    bc_list.append(batch[key].to(device).float())
                                else:
                                    raise ValueError(f"Unknown key '{key}'.")

                        model.eval_loss(
                            x_list = x_list,
                            input_param_list = input_param_list,
                            labels = labels
                        )

                        weighted_loss = 0.0
                        for val_task, train_task in zip(val_tasks, model.task_list):
                            weighted_loss += train_task.weight * val_task.loss_value

                        eval_weighted_loss_epoch += weighted_loss

                        for task in model.eval_task_list:
                            loss_epoch[task.id] += task.loss_value
                            if monitor_conflicts:
                                conflict_epoch[task.id] += task.conflict
                            if monitor_grad_norms:
                                grad_norm_epoch[task.id] += task.grad_norm

                    stats_dict["eval_loss"]["weighted_loss"].append(eval_weighted_loss_epoch / eval_steps_per_epoch)
                    for task in model.eval_task_list:
                        stats_dict["eval_loss"][task.id].append(loss_epoch[task.id] / eval_steps_per_epoch)
                        if monitor_conflicts:
                            stats_dict["eval_conflicts"][task.id].append(conflict_epoch[task.id] / eval_steps_per_epoch)
                        if monitor_grad_norms:
                            stats_dict["eval_grad_norms"][task.id].append(grad_norm_epoch[task.id] / eval_steps_per_epoch)

                    print(f"Val weighted loss: {stats_dict["eval_loss"]["weighted_loss"][-1]}")
                    if math.isnan(stats_dict["eval_loss"]["weighted_loss"][-1]):
                        #raise ArithmeticError("You get a nan value.")
                        #to_report = torch.inf
                        #trial.report(to_report, step=epoch)
                        #raise optuna.exceptions.TrialPruned()
                        return None
                    if "u" in stats_dict["eval_loss"].keys():
                        print(f"Val output loss: {stats_dict['eval_loss']['u'][-1]}")

                    # Report intermediate result to Optuna
                    to_report = sum([stats_dict["eval_loss"][task.id] for task in val_tasks])
                    trial.report(to_report, step=epoch)

                    # Check if the trial should be pruned
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        # Model in evaluation mode
        model.eval()

        if not trial.should_prune():
            # Create the folder to store trials if it does not exixt
            name = f"trialN{trial.number}"
            trial.set_user_attr("trial_name", name)
            os.makedirs(f"{destination_folder}/models/{name}", exist_ok=True)

            # Save the model
            save_model(model=model, filepath=f"{destination_folder}/models/{name}/model.pth")

            # Save model stats
            torch.save(stats_dict, f"{destination_folder}/models/{name}/stats.pth")

        # Compute final validation performances:   
        with torch.no_grad():
            # Init
            val_loss = {}
            loss_epoch = {}
            eval_weighted_loss_epoch = 0.0
            for task in model.eval_task_list:
                loss_epoch[task.id] = 0.0

            # Scan the batches
            for batches in zip(*eval_iterators):
                x_list = []
                labels = {
                    "u": [],
                    "du": [],
                    "d2u": []
                }      
                input_param_list = []
                bc_list = []

                # Scan the tasks
                for batch in batches:
                    for key in batch.keys():
                        if key == "spacetime":
                            x_list.append(batch[key].to(device).float().requires_grad_(True))
                        elif key == "param":
                            input_param_list.append(batch[key].to(device).float())
                        elif key in ["u", "du", "d2u"]:
                            labels[key].append(batch[key].to(device).float())
                        elif key == "bc":
                            bc_list.append(batch[key].to(device).float())
                        else:
                            raise ValueError(f"Unknown key '{key}'.")
                
                # Evaluate the model on the eval_tasks
                model.eval_loss(
                    x_list = x_list,
                    input_param_list = input_param_list,
                    labels = labels
                )

                # Accumulate the weighted loss
                weighted_loss = 0.0
                for val_task, train_task in zip(val_tasks, model.task_list):
                    weighted_loss += train_task.weight * val_task.loss_value
                eval_weighted_loss_epoch += weighted_loss

                # Accumulate per-task losses
                for task in model.eval_task_list:
                    loss_epoch[task.id] += task.loss_value
            
            # Average across batches
            val_loss["weighted_loss"] = eval_weighted_loss_epoch / eval_steps_per_epoch
            for task in model.eval_task_list:
                val_loss[task.id] = loss_epoch[task.id] / eval_steps_per_epoch
                print(f"Val {task.id} loss: {val_loss[task.id]}")
            print(f"Val weighted loss: {val_loss['weighted_loss']}")

            # Save the final validation performances of the trial
            torch.save(val_loss, f"{destination_folder}/models/{name}/final_validation_performances.pth")

            # Compute the trial value
            trial_loss_value = sum([val_loss[task.id] for task in val_tasks])

        return trial_loss_value
    # ******************************************************************************************************************
    
    # Set the sampler and the pruner for model selection
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    #opt_pruner = optuna.pruners.ThresholdPruner(upper=threshold, n_warmup_steps=3)

    # Create the study
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)

    # Start the model selection and training
    study.optimize(objective, n_trials = n_trials)

    print("best hyperparameters:", study.best_trial.params)
    print("best value:", study.best_trial.value)

    # Sort the (not pruned) trials by their objective value
    all_trials = [t for t in study.trials if t.state != TrialState.PRUNED]
    sorted_trials = sorted_trials = sorted(
        all_trials,
        key=lambda t: t.value
        )

    # Rename trial folders according to their final value: 0 -> best model, ..., n_trials -> worst (not pruned) model
    for i, trial in enumerate(sorted_trials):
        rank = i
        # Define the new trial name according to the rank
        old_name = trial.user_attrs.get("trial_name")
        new_name = f"trial{rank}"
        # Assign a rank attribute to the trial
        trial.set_user_attr("rank", rank)
        # Update the trial_name attribute
        trial.set_user_attr("trial_name", new_name)
        # Update the trial filename
        new_filename = f"{destination_folder}/models/{new_name}"
        old_filename = f"{destination_folder}/models/{old_name}"
        if os.path.exists(new_filename):
            if os.path.isfile(new_filename):
                os.remove(new_filename)
            elif os.path.isdir(new_filename):
                shutil.rmtree(new_filename)
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
            print(f"{old_name} --> {new_name}: objective value = {trial.value:.4f}")


def train_full():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--train_data", type=str, default="")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--space", nargs="+", type=int, default=0)
    parser.add_argument("--time", nargs="+", type=int, default=0)
    parser.add_argument("--params", nargs="+", type=int, default=0)
    parser.add_argument("--destination", type=str, default="./experiment")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.epochs == -1:
        raise ValueError(f"Specify the number of epochs.")
    
    trajectory = PhySysDataset.load(args.train_data).datasets
    boundary = [extract_boundary(dataset=snapshot, shape="rectangle") for snapshot in trajectory]
    interior = [extract_interior(dataset=snapshot, shape="rectangle") for snapshot in trajectory]

    param_keys = trajectory[0].subkeys["param"]
    param_values = 
    param_dict = {}
    for k in param_keys:
        param_dict[k] = 

    train_tasks = [PhysicsTask(
        task_id="ge",
        parameters=
    )]
    
    model = Pinn(
        device = args.device,
        hidden_units = [50, 50, 50, 50],
        activation = nn.Tanh(),
        temporal_input = 1,
        spatial_input = 2,
        param_input = 0,

    )

    train_ARD(
        model = model,
        ff = False,

        dwa = True,
        dwa_mode = "Std",
        dwa_warm_up = 3,
        dwa_moving_avg_factor = 0.9,
        dwa_moving_avg_frequency = 1,

        ewc = False,
        monitor_conflicts = False,
        monitor_weights = True,
        monitor_grad_norms= True,
        eval_every = 10,
        clip_grad = True,
        batch_size = 1024,
        learning_rate = [1e-2, 1e-3, 1e-4],
        epochs = args.epochs,
        steps = 1000000,
        device="cpu"
    )

def train_s():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--train_data", type=str, default="")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--destination", type=str, default="./experiment")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    trajectory = torch.load(args.train_data, weights_only=False)
    timeline = [t for t in range(len(trajectory.datasets))]
    boundary = [extract_boundary(dataset=trajectory, shape="rectangle", t=t) for t in timeline]
    interior = [extract_interior(dataset=trajectory, shape="rectangle", t=t) for t in timeline]

    if args.epochs == -1:
        raise ValueError(f"Specify the number of epochs.")
    
    if args.model == "":
        model = Pinn(
            device = "cpu",
            hidden_units = [50, 50, 50, 50],
            activation = nn.Tanh(),
            temporal_input = 1,
            spatial_input = 2,
            param_input = 0
        )
    else:
        model = resume_model(
            filepath = args.model,
            device = "cpu"
        )
    
    train_ARD(
        model = model,
        ff = False,

        dwa = True,
        dwa_mode = "Std",
        dwa_warm_up = 3,
        dwa_moving_avg_factor = 0.9,
        dwa_moving_avg_frequency = 1,

        ewc = False,
        monitor_conflicts = False,
        monitor_weights = True,
        monitor_grad_norms= True,
        eval_every = 10,
        clip_grad = True,
        batch_size = 1024,
        learning_rate = [1e-2, 1e-3, 1e-4],
        epochs = args.epochs,
        steps = 1000000,
        device="cpu"
    )