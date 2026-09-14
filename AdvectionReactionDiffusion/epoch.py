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

from model import Pinn
from data_utils import get_boundary, get_interior, get_iterators
from load_store_utils import resume_model, save_model

from advection_velocity import Velocity
from reaction_source import Source

from physics_task import PhysicsTask, \
    AdvectionReactionDiffusionTask, StationaryAllenCahnTask, \
    NeumannBCTask, DirichletBCTask, \
    ICTask, \
    OutputTask, \
    DerivativeTask, SpatialDerivativeTask, TemporalDerivativeTask, \
    Derivative2Task, SpatialDerivative2Task, TemporalDerivative2Task
from loss_functions import attach_loss_function

from phy_sys_dataset import PhySysDataset, AdvectionReactionDiffusionDataset

from typing import List, Iterator
import time

class Epoch:
    def __init__(
            self, 
            model: Pinn, 
            total_steps: int, 
            steps_per_epoch: int, 
            stats_dict: dict, 
            train_iterators: List[Iterator], 
            eval_every: int,
            clip_grad: bool = False
    ):
        self.model = model
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.stats_dict = stats_dict
        self.train_iterators = train_iterators
        self.eval_every = eval_every
        self.clip_grad = clip_grad
        self.last_step_index = 0

    def __call__(self, epoch_index: int):
        # ----------------------------------- Start of epoch -----------------------------------

        # Put the model in training mode
        self.model.train()

        if self.total_steps < 0:
            step_prefix = epoch_index * self.steps_per_epoch
        else:
            step_prefix = epoch_index * min(self.steps_per_epoch, self.total_steps)
        start_time = time.time()

        self.stats_dict["train_epochs"].append(epoch_index)

        loss_epoch = 0.0

        for step, batches in enumerate(zip(*self.train_iterators)):
            if self.total_steps >= 0 and step > self.total_steps:
                break

            print(f"\nepoch: {epoch_index}, batch: {step}, step: {step_prefix + step}")

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
                        x_list.append(batch[key].to(self.model.device).float().requires_grad_(True))
                    elif key == "param":
                        input_param_list.append(batch[key].to(self.model.device).float())
                    elif key in ["u", "du", "d2u"]:
                        labels[key].append(batch[key].to(self.model.device).float())
                    elif key == "bc":
                        bc_list.append(batch[key].to(self.model.device).float())
                    else:
                        raise ValueError(f"Unknown key '{key}'.")

            self.model.optimizer.zero_grad()

            loss = self.model.train_loss(
                x_list = x_list,
                input_param_list = input_param_list,
                labels = labels
            )

            # --- Backward pass ---
            loss.backward()
            loss_epoch += loss.item()

            # --- Gradient clipping ---
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # --- Compute total gradient norm ---
                #total_grad_norm = 0.0
                #for p in model.parameters():
                #    if p.grad is not None:
                #        total_grad_norm += (p.grad.data.norm(2).item())**2
                #total_grad_norm = total_grad_norm ** 0.5
                #stats_dict["train_loss_grad_norm"].append(total_grad_norm)

                # Call the optimizer
                self.model.optimizer.step()

        # ----------------------------------- End of epoch -----------------------------------

        self.model.lr_scheduler.step()

        stop_time = time.time()
        epoch_time = stop_time - start_time
        print(f'Epoch time: {epoch_time}')
        self.stats_dict["times"].append(epoch_time)
        self.stats_dict["train_loss"].append(loss_epoch / self.steps_per_epoch)
        self.last_step_index = step_prefix + step
        

    def eval(self, epoch_index: int):
        self.stats_dict["eval_epochs"].append(epoch_index)
        self.model.eval()
        with torch.set_grad_enabled(monitor_grad_norms or monitor_conflicts):
            if monitor_weights:
                for task in model.train_task_list:
                    self.stats_dict["weights"][task.id].append(task.weight)
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
                        for val_task, train_task in zip(val_tasks, model.train_task_list): #TODO
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