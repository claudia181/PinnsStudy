import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from itertools import cycle
from model import PdeNet, resume_model
from torch.optim import LBFGS, Adam
import time
import yaml
from utils import Pde
import sys
import itertools
from itertools import cycle
import optuna
import shutil

# =================================== TRAINING LOOP DEFINITION ===================================
def train_loop(
        model: PdeNet,
        train_steps: int,
        epochs: int,
        optim: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        trial,
        with_pde_params: bool = False,
        sys_mode: str = 'Output',
        distill_mode: str = 'Forgetting',
        ewc_mode: str = 'Off',
        ewc_params: torch.Tensor = None,
        ewc_fisher_diag: torch.Tensor = None,
        train_bc_dataloader: DataLoader = None,
        train_ic_dataloader: DataLoader = None,
        eval_bc_dataloader: DataLoader = None,
        eval_ic_dataloader: DataLoader = None,
        distill_dataloader: DataLoader = None,
        distill_on_actual_data: bool = False,
        distill_with_pde_params: bool = False,
        device: str = 'cpu',
        eval_every: int = 100
        ):
    # Stats dictionary
    stats_dict = {
        "train": {
            "step_list": [],
            "out_losses": [],
            "der_losses": [],
            "hes_losses": [],
            "pde_losses": [],
            "tot_losses": [],
            "bc_losses": [],
            "ic_losses": [],
            "losses": [],
            "times": []
        },
        "test": {
            "step_list": [],
            "out_losses": [],
            "der_losses": [],
            "hes_losses": [],
            "pde_losses": [],
            "tot_losses": [],
            "bc_losses": [],
            "ic_losses": [],
            "losses": []
        }
    }
    
    # Prepare Iterators for the Loop (handling empty DataLoaders)

    if train_bc_dataloader is not None:
        train_bc_iter = cycle(train_bc_dataloader)
    else:
        train_bc_iter = itertools.repeat((None, None))

    if train_ic_dataloader is not None:
        train_ic_iter = cycle(train_ic_dataloader)
    else:
        train_ic_iter = itertools.repeat((None, None))
    
    if eval_bc_dataloader is not None:
        eval_bc_iter = cycle(eval_bc_dataloader)
    else:
        eval_bc_iter = itertools.repeat((None, None))

    if eval_ic_dataloader is not None:
        eval_ic_iter = cycle(eval_ic_dataloader)
    else:
        eval_ic_iter = itertools.repeat((None, None))

    if distill_dataloader is not None:
        distill_iter = cycle(distill_dataloader)
    else:
        distill_iter = itertools.repeat((None, None, None, None))

    for epoch in range(epochs):

        # ----------------------------------- Start of epoch -----------------------------------

        # Put the model in training mode
        model.train()

        if train_steps < 0:
            step_prefix = epoch * len(train_dataloader)
        else:
            step_prefix = epoch * min(len(train_dataloader), train_steps)
        start_time = time.time()

        print(f'\nEpoch: {epoch}, step_prefix: {step_prefix}')

        train_reminder = len(train_dataloader)

        for step, (train_data, bc_data, ic_data, distill_data) in enumerate(zip(train_dataloader, train_bc_iter, train_ic_iter, distill_iter)):

            if train_steps >= 0 and step > train_steps:
                break
            
            # Load batches from dataloaders:
            # ---- Domain data ----
            x_train = train_data[0].to(device).float().requires_grad_(True)

            # Note: Generally labels (0th order information) are not available for the data inside the domain.
            #       DERL, PINN, HESL, DERL+HESL do not use them.
            #       For DERL, PINN, HESL, DERL+HESL, we use lables (we know a closed form solution)
            #       only to assess the method, not to fit the model.
            #       In the case of OUTL, SOB, OUTL+PINN, SOB+HES instead, labels inside the domain are needed.
            y_train = train_data[1].to(device).float()

            # We assume instead that labels for the derivative are available
            Dy_train = train_data[2].to(device).float()
            D2y_train = train_data[3].to(device).float()
            force_train = train_data[4].to(device).float()
            if with_pde_params:
                pde_params_train = train_data[5].to(device).float()
            else:
                pde_params_train = None

            # ---- Boundary data ----
            if bc_data[0] is not None and train_reminder <= len(train_bc_dataloader):
                x_bc = bc_data[0].to(device).float()
                y_bc = bc_data[1].to(device).float()
                if with_pde_params:
                    pde_params_bc = bc_data[5].to(device).float()
                else:
                    pde_params_bc = None
            else:
                x_bc = None
                y_bc = None
                pde_params_bc = None

            # ---- Initial data ----
            if ic_data[0] is not None:
                x_ic = ic_data[0].to(device).float()
                y_ic = ic_data[1].to(device).float()
                if with_pde_params:
                    pde_params_ic = ic_data[5].to(device).float()
                else:
                    pde_params_ic = None
            else:
                x_ic = None
                y_ic = None
                pde_params_ic = None

            # ---- Distill data ----
            if distill_data[0] is None:
                x_distill = None
                y_distill = None
                Dy_distill = None
                D2y_distill = None
                force_distill = None
                pde_params_distill = None
            else:
                x_distill = distill_data[0].to(device).float().requires_grad_(True)
                y_distill = distill_data[1].to(device).float().requires_grad_(True)
                Dy_distill = distill_data[2].to(device).float().requires_grad_(True)
                D2y_distill = distill_data[3].to(device).float().requires_grad_(True)
                force_distill = distill_data[4].to(device).float().requires_grad_(True)
                if distill_with_pde_params:
                    pde_params_distill= distill_data[5].to(device).float().requires_grad_(True)
                else:
                    pde_params_distill = None
            
            train_reminder -= len(x_train)

            # Closure
            def closure():
                model.opt.zero_grad()
                loss = model.loss_fn(
                    x_new = x_train,
                    params_new=pde_params_train,
                    y_new = y_train,
                    Dy_new = Dy_train,
                    Hy_new = D2y_train,
                    force_new = force_train,
                    x_bc = x_bc,
                    params_bc=pde_params_bc,
                    y_bc = y_bc,
                    x_ic = x_ic,
                    params_ic=pde_params_ic,
                    y_ic = y_ic,
                    x_distill = x_distill,
                    params_distill=pde_params_distill,
                    y_distill = y_distill,
                    Dy_distill = Dy_distill,
                    Hy_distill = D2y_distill,
                    force_distill = force_distill,
                    distill_on_actual_data = distill_on_actual_data,
                    ewc_params = ewc_params,
                    ewc_fisher_diag = ewc_fisher_diag,
                    sys_mode = sys_mode,
                    distill_mode = distill_mode,
                    ewc_mode = ewc_mode
                    )
                loss.backward()
                return loss
            
            # Call the optimizer
            optim.step(closure=closure)
        
        # ----------------------------------- End of epoch -----------------------------------

        # ------------------------------ Epoch evaluation ------------------------------

        # Append the epoch time
        stop_time = time.time()
        epoch_time = stop_time-start_time
        print(f'Epoch time: {epoch_time}')
        stats_dict["train"]["times"].append(epoch_time)

        if (step_prefix + step) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                for key, dataloader, bc_iter, ic_iter in [("train", train_dataloader, train_bc_iter, train_ic_iter), ("test", eval_dataloader, eval_bc_iter, eval_ic_iter)]:
                    # Compute and average the loss over the test dataloader
                    out_loss = 0.0
                    der_loss = 0.0
                    hes_loss = 0.0
                    pde_loss = 0.0
                    tot_loss = 0.0
                    bc_loss = 0.0
                    ic_loss = 0.0
                    loss = 0.0

                    for step, (data, bc_data, ic_data, distill_data) in enumerate(zip(dataloader, bc_iter, ic_iter, distill_iter)):
                        step_prefix = epoch * len(dataloader)
                        # Load batches from dataloaders
                        x = data[0].to(device).float().requires_grad_(True)
                        y = data[1].to(device).float()
                        Dy = data[2].to(device).float()
                        D2y = data[3].to(device).float()
                        force = data[4].to(device).float()
                        if with_pde_params:
                            pde_params = data[5].to(device).float()
                        else:
                            pde_params = None

                        if bc_data[0] is not None:
                            x_bc = bc_data[0].to(device).float()
                            y_bc = bc_data[1].to(device).float()
                            if with_pde_params:
                                pde_params_bc = bc_data[5].to(device).float()
                            else:
                                pde_params_bc = None
                        else:
                            x_bc = None
                            y_bc = None
                            pde_params_bc = None

                        if ic_data[0] is not None:
                            x_ic = ic_data[0].to(device).float()
                            y_ic = ic_data[1].to(device).float()
                            if with_pde_params:
                                pde_params_ic = ic_data[5].to(device).float()
                            else:
                                pde_params_ic = None
                        else:
                            x_ic = None
                            y_ic = None
                            pde_params_ic = None

                        # ---- Distill data ----
                        if distill_data[0] is None:
                            x_distill = None
                            y_distill = None
                            Dy_distill = None
                            D2y_distill = None
                            force_distill = None
                            if distill_with_pde_params:
                                pde_params_distill = distill_data[5].to(device).float()
                            else:
                                pde_params_distill = None
                        else:
                            x_distill = distill_data[0].to(device).float().requires_grad_(True)
                            y_distill = distill_data[1].to(device).float().requires_grad_(True)
                            Dy_distill = distill_data[2].to(device).float().requires_grad_(True)
                            D2y_distill = distill_data[3].to(device).float().requires_grad_(True)
                            force_distill = distill_data[4].to(device).float().requires_grad_(True)
                            if distill_with_pde_params:
                                pde_params_distill = distill_data[5].to(device).float()
                            else:
                                pde_params_distill = None

                        loss += model.loss_fn(
                            x_new = x,
                            params_new = pde_params,
                            y_new = y,
                            Dy_new = Dy,
                            Hy_new = D2y,
                            force_new = force,
                            x_bc = x_bc,
                            params_bc = pde_params_bc,
                            y_bc = y_bc,
                            x_ic = x_ic,
                            params_ic = pde_params_ic,
                            y_ic = y_ic,
                            x_distill = x_distill,
                            params_distill = pde_params_distill,
                            y_distill = y_distill,
                            Dy_distill = Dy_distill,
                            Hy_distill = D2y_distill,
                            force_distill = force_distill,
                            distill_on_actual_data = distill_on_actual_data,
                            ewc_params = ewc_params,
                            ewc_fisher_diag = ewc_fisher_diag,
                            sys_mode = sys_mode,
                            distill_mode = distill_mode,
                            ewc_mode = ewc_mode
                        ).item()

                        # Evaluate the evaluation loss on test data
                        out_loss_t, der_loss_t, hes_loss_t, pde_loss_t, bc_loss_t, ic_loss_t, tot_loss_t = model.eval_losses(
                            x = x,
                            params = pde_params,
                            y = y,
                            Dy = Dy,
                            Hy = D2y,
                            force = force,
                            x_bc = x_bc,
                            params_bc = pde_params_bc,
                            y_bc = y_bc,
                            x_ic = x_ic,
                            params_ic = pde_params_ic,
                            y_ic = y_ic,
                            sys_mode = sys_mode
                            )

                        # Accumulate test loss values
                        out_loss += out_loss_t.item()
                        der_loss += der_loss_t.item()
                        hes_loss += hes_loss_t.item()
                        pde_loss += pde_loss_t.item()
                        tot_loss += tot_loss_t.item()
                        bc_loss += bc_loss_t.item()
                        ic_loss += ic_loss_t.item()

                    # Average
                    loss /= len(dataloader)
                    out_loss /= len(dataloader)
                    der_loss /= len(dataloader)
                    hes_loss /= len(dataloader)
                    pde_loss /= len(dataloader)
                    bc_loss /= len(dataloader)
                    ic_loss /= len(dataloader)
                    tot_loss /= len(dataloader)

                    # Append test loss average values
                    stats_dict[key]["step_list"].append(step_prefix + step)
                    stats_dict[key]["tot_losses"].append(tot_loss)
                    stats_dict[key]["out_losses"].append(out_loss)
                    stats_dict[key]["der_losses"].append(der_loss)
                    stats_dict[key]["hes_losses"].append(hes_loss)
                    stats_dict[key]["pde_losses"].append(pde_loss)
                    stats_dict[key]["bc_losses"].append(bc_loss)
                    stats_dict[key]["ic_losses"].append(ic_loss)
                    stats_dict[key]["losses"].append(loss)

                    #print(f"Average output loss: {out_loss}")
                    #print(f"Average derivative loss: {der_loss}")
                    #print(f"Average hessian loss: {hes_loss}")
                    #print(f"Average PDE loss: {pde_loss}")
                    #print(f"Average bc loss: {bc_loss}")
                    #print(f"Average ic loss: {ic_loss}")
                    print(f"Average {key} loss: {loss}")

                # Report intermediate result to Optuna
                trial.report(stats_dict["test"]["losses"][-1], step=epoch)
        
                # Check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    return stats_dict
# ==============================================================================================================

# Function to get value, prioritizing config file > default
def get_param(config_section_dict, config_key, default_val=None, type_func=None):
    # Try to get from config file
    if config_section_dict and config_key in config_section_dict:
        val = config_section_dict[config_key]
        return type_func(val) if type_func else val
    
    # Use default
    else:
        return default_val

class Objective:
    def __init__(
            self,
            seed,
            starting_model,
            pde_name,
            pde_params,
            with_pde_params,
            sys_mode,
            distill_mode,
            ewc_mode,
            ewc_params,
            ewc_fisher_diag,
            input_units,
            layers,
            train_steps,
            epochs,
            eval_every,
            device,
            train_dataset,
            eval_dataset,
            train_bc_dataset,
            train_ic_dataset,
            eval_bc_dataset,
            eval_ic_dataset,
            distill_dataset,
            distill_on_actual_data,
            distill_with_pde_params,
            batch_size_list,
            lr_init_interval,
            phy_weight_interval,
            bc_weight_interval,
            ic_weight_interval,
            distill_weight_interval,
            ewc_weight_interval,
            models_dir):
        self.seed = seed
        self.starting_model = starting_model
        self.pde_name = pde_name
        self.pde_params = pde_params
        self.with_pde_params = with_pde_params
        self.sys_mode = sys_mode
        self.distill_mode = distill_mode
        self.ewc_mode = ewc_mode
        self.ewc_params = ewc_params
        self.ewc_fisher_diag = ewc_fisher_diag
        self.input_units = input_units
        self.layers = layers
        self.train_steps = train_steps
        self.epochs = epochs
        self.eval_every = eval_every
        self.device = device
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_bc_dataset = train_bc_dataset
        self.train_ic_dataset = train_ic_dataset
        self.eval_bc_dataset = eval_bc_dataset
        self.eval_ic_dataset = eval_ic_dataset
        self.distill_dataset = distill_dataset
        self.distill_on_actual_data = distill_on_actual_data
        self.distill_with_pde_params = distill_with_pde_params
        self.batch_size_list = batch_size_list
        self.lr_init_interval = lr_init_interval
        self.phy_weight_interval = phy_weight_interval
        self.bc_weight_interval = bc_weight_interval
        self.ic_weight_interval = ic_weight_interval
        self.distill_weight_interval = distill_weight_interval
        self.ewc_weight_interval = ewc_weight_interval
        self.activation = torch.nn.Tanh()
        self.models_dir = models_dir
        self.models = []
        
        
    def __call__(self, trial):
        # Define hyperparameters to optimize
        phy_weight = trial.suggest_float("phy_weight", self.phy_weight_interval[0], self.phy_weight_interval[1], log=True)

        bc_weight = trial.suggest_float("bc_weight", self.bc_weight_interval[0], self.bc_weight_interval[1], log=True)

        ic_weight = trial.suggest_float("ic_weight", self.ic_weight_interval[0], self.ic_weight_interval[1], log=True)

        distill_weight = trial.suggest_float("distill_weight", self.distill_weight_interval[0], self.distill_weight_interval[1], log=True)
        
        ewc_weight = trial.suggest_float("ewc_weight", self.ewc_weight_interval[0], self.ewc_weight_interval[1], log=True)
        self.batch_size = trial.suggest_categorical("batch_size", self.batch_size_list)
        lr_init = trial.suggest_float("lr_init", self.lr_init_interval[0], self.lr_init_interval[1], log=True)

        # Seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # Load data and build dataloaders
        train_dataloader = DataLoader(self.train_dataset, self.batch_size, generator=gen, shuffle=True)

        eval_dataloader = DataLoader(self.eval_dataset, self.batch_size, generator=gen, shuffle=True)

        if self.train_bc_dataset == None:
            train_bc_dataloader = None
        else:
            train_bc_dataloader = DataLoader(self.train_bc_dataset, self.batch_size, generator=gen, shuffle=True)

        if self.train_ic_dataset == None:
            train_ic_dataloader = None
        else:
            train_ic_dataloader = DataLoader(self.train_ic_dataset, self.batch_size, generator=gen, shuffle=True)

        if self.eval_bc_dataset == None:
            eval_bc_dataloader = None
        else:
            eval_bc_dataloader = DataLoader(self.eval_bc_dataset, self.batch_size, generator=gen, shuffle=True)

        if self.eval_ic_dataset == None:
            eval_ic_dataloader = None
        else:
            eval_ic_dataloader = DataLoader(self.eval_ic_dataset, self.batch_size, generator=gen, shuffle=True)        

        if self.distill_dataset == None:
            distill_dataloader = None
        else:
            distill_dataloader = DataLoader(self.distill_dataset, self.batch_size, generator=gen, shuffle=True)


        # NN activation function

        # Model init
        if self.starting_model is None:
            model = PdeNet(
                pde=Pde(name=self.pde_name, params=self.pde_params),
                bc_weight=bc_weight,
                ic_weight=ic_weight,
                phy_weight=phy_weight,
                distill_weight=distill_weight,
                ewc_weight=ewc_weight,
                input_units=self.input_units,
                hidden_units=self.layers,
                lr_init=lr_init,
                device=self.device,
                activation=self.activation,    
                last_activation=False
            ).to(self.device)
        else:
            model = resume_model(model_path=self.starting_model, device=self.device)

        # Optimizer
        optim = Adam(params=model.parameters(), lr=lr_init)
        #optim = LBFGS(params=model.parameters(), lr=lr_init)

        stats_dict = train_loop(
            model = model,
            with_pde_params = self.with_pde_params,
            train_steps = self.train_steps,
            epochs = self.epochs,
            optim = optim,
            sys_mode = self.sys_mode,
            distill_mode = self.distill_mode,
            ewc_mode = self.ewc_mode,
            ewc_params = self.ewc_params,
            ewc_fisher_diag = self.ewc_fisher_diag,
            train_dataloader = train_dataloader,
            train_bc_dataloader = train_bc_dataloader,
            train_ic_dataloader = train_ic_dataloader,
            eval_dataloader = eval_dataloader,
            eval_bc_dataloader = eval_bc_dataloader,
            eval_ic_dataloader = eval_ic_dataloader,
            distill_dataloader = distill_dataloader,
            distill_on_actual_data = self.distill_on_actual_data,
            distill_with_pde_params = self.distill_with_pde_params,
            device = self.device,
            eval_every = eval_every,
            trial = trial
            )

        torch.cuda.empty_cache()

        # Put the model in evaluation mode
        model.eval()

        name = f"trial{trial.number}"
        os.makedirs(f"{self.models_dir}/{name}", exist_ok=True)

        self.save_model(model=model, name=name)
        self.save_stats(stats_dict=stats_dict, key="train", name=name)
        self.save_stats(stats_dict=stats_dict, key="test", name=name)

        return stats_dict["test"]["losses"][-1]

    def save_model(self, model: PdeNet, name: str):
        # Save the model
        # Create a dictionary to save both the model's state_dict and its architectural parameters
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'pde': self.pde_name,
            'pde_params': self.pde_params,
            'input_units': model.input_units,
            'hidden_units': model.hidden_units,
            'lr_init': model.lr_init,
            'batch_size': self.batch_size,
            'bc_weight': model.bc_weight,
            'ic_weight': model.ic_weight,
            'phy_weight': model.phy_weight,
            'distill_weight': model.distill_weight,
            'ewc_weight': model.ewc_weight,
            'activation': model.activation
        }

        # Save the checkpoint dictionary
        torch.save(checkpoint, f"{self.models_dir}/{name}/model.pth")
    
    def save_stats(self, stats_dict: dict, key: str, name: str):
        curves = []
        
        curves.append(torch.tensor(stats_dict[key]["step_list"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["tot_losses"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["out_losses"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["der_losses"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["hes_losses"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["pde_losses"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["bc_losses"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["ic_losses"]).cpu().numpy())
        curves.append(torch.tensor(stats_dict[key]["losses"]).cpu().numpy())
        if key == "train":
            curves.append(torch.tensor(stats_dict[key]["times"]).cpu().numpy())

        # Stack loss curves
        stacked_curves = np.column_stack(curves)

        # Save loss curves
        with open(f'{self.models_dir}/{name}/{key}_stats.npy', 'wb') as f:
            np.save(f, stacked_curves)


# =================================================================================================================

if __name__ == "__main__":
    # Init parser for command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config.yaml', type=str, help='Path to the configuration file (YAML)')

    # Parse command-line arguments
    cli_args = parser.parse_args()

    # Load configuration from YAML file
    config_params = {}
    if os.path.exists(cli_args.config):
        with open(cli_args.config, 'r') as f:
            config_params = yaml.safe_load(f)
    else:
        print(f"Warning: Config file '{cli_args.config}' not found. Using default arguments.")

    # Extract parameters from config_params

    # Get config sections
    training_config = config_params.get('training', {})
    hyperparams_config = config_params.get('hyperparams', {})
    paths_config = config_params.get('paths', {})

    # PDE considered
    pde_name = get_param(training_config, 'pde', default_val='Allen-Cahn')
    pde_params = [float(p) for p in get_param(training_config, 'pde_params', default_val=[0.0], type_func=list)]

    # Training mode
    sys_mode = get_param(training_config, 'sys_mode', default_val='PINN')
    distill_mode = get_param(training_config, 'distill_mode', default_val='Forgetting')
    ewc_mode = get_param(training_config, 'ewc_mode', default_val='Off')

    # Architecture
    input_units = get_param(hyperparams_config, 'input_units', default_val=2, type_func=int)  
    with_pde_params = get_param(training_config, 'with_pde_params', default_val=False, type_func=bool)
    str_layers = get_param(hyperparams_config, 'layers', default_val=[50, 50, 50, 50], type_func=list)
    layers = [int(layer) for layer in str_layers]

    # Learning
    train_steps = get_param(training_config, 'train_steps', default_val=-1, type_func=int)
    epochs = get_param(training_config, 'epochs', default_val=100, type_func=int)
    eval_every = get_param(training_config, 'eval_every', default_val=100, type_func=int)

    # Seed
    seed = get_param(training_config, 'seed', default_val=42, type_func=int)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Device
    device = get_param(training_config, 'device', default_val='cpu') #cuda:0

    # Experiment name
    experiment_name = get_param(training_config, 'experiment_name', default_val='grid')

    # Models directory
    models_dir = get_param(paths_config, 'models_dir', default_val='./models')
    os.makedirs(models_dir, exist_ok=True) # Ensure models directory exists

    # Copy the config file in the model directory
    shutil.copy(cli_args.config, models_dir)

    # Datasets directory
    datasets_dir = get_param(paths_config, 'datasets_dir', default_val='.')

    # Training dataset
    train_dataset_file = get_param(paths_config, 'train_dataset', default_val='')
    if train_dataset_file == '':
        print('ERROR: Required train_dataset arg not specified. Exiting.', file=sys.stderr)
        sys.exit(1)
    train_dataset = torch.load(os.path.join(datasets_dir, train_dataset_file), weights_only=False)

    # Evaluation dataset
    eval_dataset_file = get_param(paths_config, 'eval_dataset', default_val='')
    if eval_dataset_file == '':
        print('ERROR: Required eval_dataset arg not specified. Exiting.', file=sys.stderr)
        sys.exit(1)
    eval_dataset = torch.load(os.path.join(datasets_dir, eval_dataset_file), weights_only=False)

    # Boundary conditions dataset
    train_bc_dataset_file = get_param(paths_config, 'train_bc_dataset', default_val='')
    if train_bc_dataset_file == '':
        train_bc_dataset = None
    else:
        train_bc_dataset = torch.load(os.path.join(datasets_dir, train_bc_dataset_file), weights_only=False)

    # Initial conditions dataset
    train_ic_dataset_file = get_param(paths_config, 'train_ic_dataset', default_val='')
    if train_ic_dataset_file == '':
        train_ic_dataset = None
    else:
        train_ic_dataset = torch.load(os.path.join(datasets_dir, train_ic_dataset_file), weights_only=False)

    # Boundary conditions dataset
    eval_bc_dataset_file = get_param(paths_config, 'eval_bc_dataset', default_val='')
    if eval_bc_dataset_file == '':
        eval_bc_dataset = None
    else:
        eval_bc_dataset = torch.load(os.path.join(datasets_dir, eval_bc_dataset_file), weights_only=False)

    # Initial conditions dataset
    eval_ic_dataset_file = get_param(paths_config, 'eval_ic_dataset', default_val='')
    if eval_ic_dataset_file == '':
        eval_ic_dataset = None
    else:
        eval_ic_dataset = torch.load(os.path.join(datasets_dir, eval_ic_dataset_file), weights_only=False)
    
    # Starting model (optional)
    model_file = get_param(paths_config, 'starting_model', default_val='')
    if model_file == '':
        starting_model = None
    else:
        starting_model = model_file

    # Distillation dataset and distillation model
    distill_dataset_file = get_param(paths_config, 'distill_dataset', default_val='')
    distill_on_actual_data = False
    if distill_dataset_file == '':
        distill_dataset = None
        distill_model = None
        distill_with_pde_params = False
        distill_mode = 'Forgetting'
    else:
        if distill_dataset_file == train_dataset_file:
            distill_on_actual_data = True
        distill_dataset = torch.load(os.path.join(datasets_dir, distill_dataset_file), weights_only=False)
        distill_model_file = get_param(paths_config, 'distill_model', default_val='')
        distill_with_pde_params = get_param(training_config, 'distill_with_pde_params', default_val=False, type_func=bool)
        # if the distill_dataset_file is passed and the distill_model is not passed,
        # assume that the distill_dataset_file is already labeled by distill_model
        if distill_model_file == '':
            distill_model = None
        # if both are passed, label the distillation dataset whith distill_model and work with it
        else:
            distill_model = resume_model(model_path=distill_model_file, device=device)
            distill_dataset = distill_model.label(dataset=distill_dataset, save=False, with_pde_in_params=distill_with_pde_params)

    # EWC
    if ewc_mode == 'On' and distill_model is not None:
        with torch.no_grad():
            ewc_params = distill_model.get_weights()
        ewc_fisher_diag = distill_model.get_fisher_diag(dataset=distill_dataset, sys_mode=sys_mode, with_pde_params=distill_with_pde_params)
    else:
        ewc_mode = 'Off'
        ewc_params = None
        ewc_fisher_diag = None

    # Check if optimize hyperparameters or fix them
    hyp_mode = get_param(hyperparams_config, 'mode', default_val='Fix')

    if hyp_mode == 'Fix':
        bc_weight_fixed = get_param(hyperparams_config, 'bc_weight', default_val=1.0, type_func=float)
        bc_weight = [bc_weight_fixed, bc_weight_fixed]
        ic_weight_fixed = get_param(hyperparams_config, 'ic_weight', default_val=1.0, type_func=float)
        ic_weight = [ic_weight_fixed, ic_weight_fixed]
        phy_weight_fixed = get_param(hyperparams_config, 'phy_weight', default_val=1.0, type_func=float)
        phy_weight = [phy_weight_fixed, phy_weight_fixed]
        distill_weight_fixed = get_param(hyperparams_config, 'distill_weight', default_val=1.0, type_func=float)
        distill_weight = [distill_weight_fixed, distill_weight_fixed]
        ewc_weight_fixed = get_param(hyperparams_config, 'ewc_weight', default_val=1.0, type_func=float)
        ewc_weight = [ewc_weight_fixed, ewc_weight_fixed]
        lr_init_fixed = get_param(hyperparams_config, 'lr_init', default_val=1e-5, type_func=float)
        lr_init = [lr_init_fixed, lr_init_fixed]
        batch_size = [get_param(hyperparams_config, 'batch_size', default_val=1024, type_func=int)]
    else: # hyp_mode == 'Optimize'
        bc_weight_str = get_param(hyperparams_config, 'bc_weight', default_val=[1.0, 1.0], type_func=list)
        bc_weight = [float(x) for x in bc_weight_str]
        ic_weight_str = get_param(hyperparams_config, 'ic_weight', default_val=[1.0, 1.0], type_func=list)
        ic_weight = [float(x) for x in ic_weight_str]
        phy_weight_str = get_param(hyperparams_config, 'phy_weight', default_val=[1.0, 1.0], type_func=list)
        phy_weight = [float(x) for x in phy_weight_str]
        distill_weight_str = get_param(hyperparams_config, 'distill_weight', default_val=[1.0, 1.0], type_func=list)
        distill_weight = [float(x) for x in distill_weight_str]
        ewc_weight_str = get_param(hyperparams_config, 'ewc_weight', default_val=[1.0, 1.0], type_func=list)
        ewc_weight = [float(x) for x in ewc_weight_str]
        lr_init_str = get_param(hyperparams_config, 'lr_init', default_val=[1e-5, 1e-1], type_func=list)
        lr_init = [float(x) for x in lr_init_str]
        batch_size_str = get_param(hyperparams_config, 'batch_size', default_val=[1024], type_func=list)
        batch_size = [int(x) for x in batch_size_str]

    objective = Objective(
        seed = seed,
        starting_model = starting_model,
        pde_name = pde_name,
        pde_params = pde_params,
        with_pde_params = with_pde_params,
        sys_mode = sys_mode,
        distill_mode = distill_mode,
        ewc_mode = ewc_mode,
        ewc_params = ewc_params,
        ewc_fisher_diag = ewc_fisher_diag,
        input_units=input_units,
        layers = layers,
        train_steps = train_steps,
        epochs = epochs,
        eval_every = eval_every,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        train_bc_dataset = train_bc_dataset,
        train_ic_dataset = train_ic_dataset,
        eval_bc_dataset = eval_bc_dataset,
        eval_ic_dataset = eval_ic_dataset,
        distill_dataset = distill_dataset,
        distill_on_actual_data = distill_on_actual_data,
        distill_with_pde_params = distill_with_pde_params,
        batch_size_list = batch_size,
        lr_init_interval = lr_init,
        phy_weight_interval = phy_weight,
        bc_weight_interval = bc_weight,
        ic_weight_interval = ic_weight,
        distill_weight_interval = distill_weight,
        ewc_weight_interval = ewc_weight,
        device = device,
        models_dir = models_dir
        )

    if starting_model is None:
        if distill_mode == "Output":
            pruner = optuna.pruners.ThresholdPruner(upper=0.005, n_warmup_steps=0)
            n_trials = 20
        elif sys_mode == "Output":
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
            n_trials = 10
        else:
            pruner = optuna.pruners.ThresholdPruner(upper=0.001, n_warmup_steps=0)
            n_trials = 20
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        n_trials = 10
    study = optuna.create_study(direction = "minimize", pruner=pruner)
    study.optimize(objective, n_trials = n_trials)

    print("Best trial params:", study.best_trial.params)
    print("Best trial value:", study.best_trial.value)

    best_trial_filename = f"{models_dir}/trial{study.best_trial.number}"
    best_trial_new_filename = f"{models_dir}/best_trial"
    if os.path.exists(best_trial_new_filename):
        if os.path.isfile(best_trial_new_filename):
            os.remove(best_trial_new_filename)
        elif os.path.isdir(best_trial_new_filename):
            shutil.rmtree(best_trial_new_filename)
    os.rename(best_trial_filename, best_trial_new_filename)