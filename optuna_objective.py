"""
optuna_objective.py
===========

This module contains:
- the callable class Objective (*)

This module implements the layer between the interface of train.py 
and the train_loop.py logic that allows to perform model selection, if required (suggestions = "On").
The Objective class takes the parameters to set the train_loop as arguments in the constructor 
and then call the train_loop function of the train_loop.py module.
Finally the Objective call function save the learned model 
and the relative training loop statistics.
"""

import torch
from torch.utils.data import TensorDataset
import os
from model import PdeNet
from load_store_utils import save_model, save_stats
from train_loop import train_loop
import optuna
import copy

class Objective:
    """
    Class representing an objective wrt which the optuna engine perform model selection.
    """
    def __init__(
            self,
            model: PdeNet,
            train_steps: int,
            epochs: int,
            early_stop_value: float,
            eval_every: int,
            batch_size: list,
            lr_init: list,
            clip_grad: bool,
            fourier_features: list,
            frequency_variance: list,
            scheduler_mode: str,
            train_dataset: TensorDataset,
            nl_dataset: TensorDataset,
            val_dataset: TensorDataset,
            train_bc_dataset: TensorDataset = None,
            train_ic_dataset: TensorDataset = None,
            nl_bc_dataset: TensorDataset = None,
            nl_ic_dataset: TensorDataset = None,
            val_bc_dataset: TensorDataset = None,
            val_ic_dataset: TensorDataset = None,
            distill_dataset: TensorDataset = None,
            models_dir: str = "models",
            device: str = "cpu",
            seed: int = 42,
            suggestions: str = "On",
            reset: bool = False
            ):
        """
        Constructor.

        Parameters
        ----------
        model : PdeNet
            PINN to train.
        train_steps : int
            Number of training steps.
        epochs : int
            Number of epochs.
        early_stop_value : float
            Value of loss grad norm for the early stopping.
            If None, no early stopping is performed.
        eval_every :  int
            Evaluation period.
        batch_size : list
            Batch size.
        lr_init : list
            Initial learning rate.
        clip_grad : bool
            If True, gradient norm is clipped to 1.
        fourier_features : list
            Fourier features.
        frequency_variance : list
            Variance of the 0-centered Gaussian distribution 
            from which the frequency mtx for Fourier features are sampled.
        scheduler_mode : bool
            Learning rate scheduler ("ExpDec" | "CosAnn" | None).
        train_dataset : TensorDataset
            Labeled training set.
        nl_dataset : TensorDataset
            Unlabeled training set.
        val_dataset : TensorDataset
            Validation set.
        train_bc_dataset : TensorDataset
            Boundary training set.
        train_ic_dataset : TensorDataset
            Initial condition training set.
        nl_bc_dataset : TensorDataset
            Unlabeled boundary training set.
        nl_ic_dataset : TensorDataset
            Unlabeled initial condition training set.
        val_bc_dataset : TensorDataset
            Boundary validation set.
        val_ic_dataset : TensorDataset
            Initial condition validation set.
        distill_dataset : TensorDataset
            Distillation set.
        models_dir : str
            Folder where to store the models files.
        device: str
        seed : int
        suggestions : str
            If 'On', hyperparameter search is performed.
        reset : bool
            If True, reset the model after the trial.
        """
        self.model = model
        self.train_steps = train_steps
        self.epochs = epochs
        self.early_stop_value = early_stop_value
        self.eval_every = eval_every
        self.device = device
        self.seed = seed
        self.train_dataset = train_dataset
        self.nl_dataset = nl_dataset
        self.val_dataset = val_dataset
        self.train_bc_dataset = train_bc_dataset
        self.train_ic_dataset = train_ic_dataset
        self.nl_bc_dataset = nl_bc_dataset
        self.nl_ic_dataset = nl_ic_dataset
        self.val_bc_dataset = val_bc_dataset
        self.val_ic_dataset = val_ic_dataset
        self.distill_dataset = distill_dataset
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.clip_grad = clip_grad
        self.fourier_features = fourier_features
        self.frequency_variance = frequency_variance
        self.scheduler_mode = scheduler_mode
        self.models_dir = models_dir
        self.models = []
        if suggestions == "On":
            self.suggestions = True
        elif suggestions == "Off":
            self.suggestions = False
        else:
            raise ValueError(f"Unknown value {suggestions} for 'suggestions' parameter. Allowed values are 'On'|'Off'.")
        self.reset = reset
        self.model_cpy = copy.deepcopy(self.model)
        self.count = 0
        
    def __call__(self, trial: optuna.trial.Trial) -> float:
        """
        Call method for the callable object.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        float
            The last output validation loss.
        """
        if self.reset:
            self.model = self.model_cpy

        # Define hyperparameters to optimize
        if len(self.batch_size) > 1 and self.suggestions:
            batch_size = trial.suggest_categorical("batch_size", self.batch_size)
        else:
            batch_size = self.batch_size[min(self.count, len(self.batch_size)-1)]
        
        if len(self.lr_init) > 1 and self.suggestions:
            lr_init = trial.suggest_categorical("lr_init", self.lr_init)
        else:
            lr_init = self.lr_init[min(self.count, len(self.lr_init)-1)]
        
        if len(self.fourier_features) > 1 and self.suggestions:
            fourier_features = trial.suggest_categorical("fourier_features", self.fourier_features)
        else:
            fourier_features = self.fourier_features[min(self.count, len(self.fourier_features)-1)]

        if len(self.frequency_variance) > 1 and self.suggestions:
            frequency_variance = trial.suggest_categorical("frequency_variance", self.frequency_variance)
        else:
            frequency_variance = self.frequency_variance[min(self.count, len(self.frequency_variance)-1)]
        
        self.model.set_fourier_features(fourier_features, frequency_variance)
        
        self.count += 1

        # Run the training loop and get the statistics
        stats_dict = train_loop(
            model = self.model,
            train_steps = self.train_steps,
            lr_init=lr_init,
            batch_size=batch_size,
            clip_grad = self.clip_grad,
            scheduler_mode=self.scheduler_mode,
            epochs = self.epochs,
            early_stop_value = self.early_stop_value,
            train_dataset = self.train_dataset,
            train_bc_dataset = self.train_bc_dataset,
            train_ic_dataset = self.train_ic_dataset,
            nl_dataset = self.nl_dataset,
            nl_bc_dataset = self.nl_bc_dataset,
            nl_ic_dataset = self.nl_ic_dataset,
            val_dataset = self.val_dataset,
            val_bc_dataset = self.val_bc_dataset,
            val_ic_dataset = self.val_ic_dataset,
            distill_dataset = self.distill_dataset,
            device = self.device,
            eval_every = self.eval_every,
            trial = trial,
            seed=self.seed
            )

        torch.cuda.empty_cache()

        # Put the model in evaluation mode
        self.model.eval()

        if stats_dict is None:
            self.model = self.model_cpy
            return torch.inf

        if not trial.should_prune():
            # Save the model and train/test statistics
            name = f"trialN{trial.number}"
            trial.set_user_attr("trial_name", name)
            os.makedirs(f"{self.models_dir}/{name}", exist_ok=True)
            save_model(model=self.model,
                       name=name,
                       lr_init=lr_init,
                       batch_size=batch_size,
                       clip_grad=self.clip_grad,
                       scheduler=self.scheduler_mode,
                       delayed_weights=self.delayed_weights,
                       models_dir=self.models_dir
                       )        
            save_stats(stats_dict=stats_dict, directory=f"{self.models_dir}/{name}")


        # Return the last output val loss
        items = self.model.sys_mode.split("+")
        value = 0
        if self.val_dataset is not None:
            key = "val"
        else:
            key = "train"
        if "PINN" in items:
            value += stats_dict[key]["res_loss"][-1]
            if stats_dict[key]["bc_loss"] != []:
                value += stats_dict[key]["bc_loss"][-1]
            if stats_dict[key]["ic_loss"] != []:
                value += stats_dict[key]["ic_loss"][-1]
        if "Output" in items:
            value += stats_dict[key]["out_loss"][-1]
        if "Derivative" in items:
            value += stats_dict[key]["der_loss"][-1]
        if "Derivative_x" in items:
            value += stats_dict[key]["derx_loss"][-1]
        if "Derivative_t" in items:
            value += stats_dict[key]["dert_loss"][-1]
        if "Hessian" in items:
            value += stats_dict[key]["hes_loss"][-1]
        if "Hessian_x" in items:
            value += stats_dict[key]["hesx_loss"][-1]
        if "Hessian_t" in items:
            value += stats_dict[key]["hest_loss"][-1]
        return value