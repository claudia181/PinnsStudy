"""
model.py
===========

This module implements the PINN model class.

Functions:
- get_dictionary: Constructs a dictionary object from a tensor of keys and a tensor of values.

Global lists:
- SYS_MODES     : system loss term modes.
- DISTILL_MODES : distillation loss term modes.
- EWC_MODES     : elastic weight consolidation loss term modes.
- DWA_MODES     : dinamic weight adaptation modes.
- LOSS_TERMS    : loss terms keys.

Classes:
- PdeNet: PINN class.
    - Constructor:
        You have to pass to the constructor
            - the PDE string identifier,
            - the list of hidden units,
            - the device.
        You can pass to the constructor
            - informations about the input units (number, if space/time is part of the input, 
            if the spatio-temporal input is encoded through Fourier features, 
            which of the PDE parameters are given as input to the net, 
            which of the IC informations are given as input), 
            - informations about how to train the network, i.e. about the loss function to use: 
                - which network function informations to use for the system loss term 
                (e.g. the output value, the derivative wrt the spatio-temporal inputs, ...),
                - if to perform distillation and what to distill (e.g. forgetting, output, derivative, ...) 
                - if to perform elastic weight consolidation
                - if and how to perform dynamic weight adaptation (initial weights, moving avg parameter, ...)
                - a priori importances on loss terms (fixed)
    - Public methods:
        - Setters:
            - set_fourier_features
            - set_ewc
            - set_dwa
        - Getters:
            - get_weights
            - get_fisher_diag
        - Model inference methods:
            - forward
            - forward_single
            - derivative
        - Loss computation:
            - loss_fn
            - eval_losses
        - Model usage methods:
            - label
            - evaluate
"""

import torch
from torch import nn
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian
from pde_utils import Pde, key_idx, ic_key_idx, key_str
from torch.utils.data import DataLoader
import os
from torch.utils.data import TensorDataset
from typing import Tuple
from generate import X, U, DU, D2U, PDE_VALUES, IC_VALUES, RESIDUAL_KEYS, RESIDUAL_VALUES

SYS_MODES       = ["Output", "PINN", "Derivative", "Derivative_x", "Derivative_t", "Hessian", "Hessian_x", "Hessian_t"]
DISTILL_MODES   = ["Forgetting", "Output", "PINN", "Derivative", "Derivative_x", "Derivative_t", "Hessian", "Hessian_x", "Hessian_t"]
EWC_MODES       = ["On", "Off"]
DWA_MODES       = ["Off", "Std", "Norm1", "NormK"]
LOSS_TERMS      = ["out_loss", "der_loss", "derx_loss", "dert_loss", "hes_loss", "hesx_loss", "hest_loss", "res_loss", 
                   "nl_loss", "nl_bc_loss", "nl_ic_loss",
                   "distill_out_loss", "distill_der_loss", "distill_derx_loss", "distill_dert_loss", "distill_hes_loss", "distill_hesx_loss", "distill_hest_loss",
                   "ewc_loss", "bc_loss", "ic_loss"]

def get_dictionary(keys: torch.Tensor, values: torch.Tensor, pde_name: str):
    dictionary = {}
    keys = keys[0].tolist()
    values = [values[:, i] for i in range(values.shape[1])]
    for key, value in zip([key_str(k, pde_name) for k in keys], [v for v in values]):
        if len(value) == 1: value = value.item()
        dictionary[key] = value
    return dictionary

# Model
class PdeNet(torch.nn.Module):
    """
    Class representing a PINN model.

    Attributes
    ----------
    pde : str
        String identifying the PDE for which the model is constructed.
    input_units : int
        Total number of input units (space, time, additional parameters).
    time_in_input : bool
        True if the time is provided as input.
    space_in_input : bool
        True if the space is provided as input.
    fourier_features : int
        Number of Fourier features for the encoding of spatio-temporal coordinates.
    frequency_variance : float
        Variance of the 0-centered Gaussian distribution 
        from which the frequency mtx (B) for Fourier features are sampled.
    B : torch.Tensor
        Random matrix for Fourier encoding computation.
    pde_params_in_input : list
        List of the keys of the PDE parameters provided in input to the model.
    ic_params_in_input : list
        List of the keys of the initial conditions parameters provided in input to the model.
    hidden_units : list
        List of the hidden units of the model.
    activation : nn.Module
        Activation function of the network.
    net : OrderedDict
        The NN.
    sys_mode : str
        Identify the loss function system term.
    distill_mode : str
        Identify the loss function distillation term.
    ewc_mode : str
        Elastic weight consolidation on or off.
    dwa_mode : str
        Dynamic weight adaptation mode in DWA_MODES.
    bc_mode : str
        Identify the loss function boundary condition term.
    monitor_conflicts : bool
        If True and dwa is active, the last cosine similarities btw res grad and the other grads are maintained in model.res_conflicts.
    (prev_)bc_weight : float
        Starting weight of the boundary condition term.
    (prev_)ic_weight : float
        Starting weight of the initial condition term.
    (prev_)res_weight : float
        Starting weight of the residual term.
    (prev_)out_weight : float
        Starting weight of the output term.
    (prev_)der_weight : float
        Starting weight of the 1st derivative term.
    (prev_)hes_weight : float
        Starting weight of the 2nd derivative term.
    (prev_)nl_weight : float
        Starting weight of the residual term relative to the unlabeled points (e.g. for residual replay).
    (prev_)distill_out_weight : float
        Starting weight of the distillation output term.
    (prev_)distill_der_weight : float
        Starting weight of the distillation 1st derivative term.
    (prev_)distill_hes_weight : float
        Starting weight of the distillation 2nd derivative term.
    (prev_)ewc_weight : float
        Starting weight of the elastic weight consolidation term.
    ewc_auto_weighting : bool
        Enabling the ewc term auto weightig.
    ewc_warm_up : int
        Number of training steps before setting the ewc weight if ewc_auto_weighting = True.
    ewc_decay : float
        Decay factor for ewc_weight when ewc_auto_weighting = True.
    ewc_model_weights : torch.Tensor
        Optimal params of a previous model.
    ewc_fisher_diag : torch.Tensor
        Diagonal elements of the fisher information matrix relative to ewc_model_weights,
        computed on some data.
    alpha : float
        Moving average weight for dynamic weight adaptation.
    moving_avg_frequency : int
        Moving average frequency for dynamic weight adaptation.
    dwa_warm_up : int
        Warm up steps for dynamic weight adaptation.
    sys_importance : float
        Importance of system term in the loss.
    bc_importance : float
        Importance of boundary condition term in the loss.
    ic_importance : float
        Importance of initial condition term in the loss.
    nl_importance : float
        Importance of unlabeled points term in the loss.
    distill_importance : float
        Importance of distillation term in the loss.
    ewc_importance : float
        Importance of elastic weight consolidation term in the loss.
    device : str
        Device of the model.
    """

    def __init__(
            self,
            pde: str,
            hidden_units: list,
            device: str,
            input_units: int = None,
            time_in_input: bool = False,
            space_in_input: bool = True,
            fourier_features: int = -1,
            frequency_variance: float = 1.0,
            pde_params_in_input: list = [],
            ic_params_in_input: list = [],
            sys_mode: str = "Output",
            distill_mode: str = "Forgetting",
            ewc_mode: str = "Off",
            dwa_mode: str = "Off",
            bc_mode: str = "Dirichlet",
            monitor_conflicts: bool = False,
            bc_weight: float = 1.0,
            ic_weight: float = 1.0,
            res_weight: float = 1.0,
            out_weight: float = 1.0,
            der_weight: float = 1.0,
            derx_weight: float = 1.0,
            dert_weight: float = 1.0,
            hes_weight: float = 1.0,
            hesx_weight: float = 1.0,
            hest_weight: float = 1.0,
            nl_weight: float = 1.0,
            nl_bc_weight: float = 1.0,
            nl_ic_weight: float = 1.0,
            distill_out_weight: float = 1.0,
            distill_der_weight: float = 1.0,
            distill_derx_weight: float = 1.0,
            distill_dert_weight: float = 1.0,
            distill_hes_weight: float = 1.0,
            distill_hesx_weight: float = 1.0,
            distill_hest_weight: float = 1.0,
            ewc_weight: float = 1.0,
            ewc_auto_weighting: bool = False,
            ewc_warm_up: int = 0,
            ewc_decay: float = 1.0,
            ewc_model_weights: torch.Tensor = None,
            ewc_fisher_diag: torch.Tensor = None,
            alpha: float = 0.9,
            moving_avg_frequency: int = 1,
            dwa_warm_up: int = 0,
            sys_importance: float = 1.0,
            bc_importance: float = 1.0,
            ic_importance: float = 1.0,
            nl_importance: float = 1.0,
            nl_bc_importance: float = 1.0,
            nl_ic_importance: float = 1.0,
            distill_importance: float = 1.0,
            ewc_importance: float = 1.0,
            activation: nn.Module = nn.Tanh(),
            last_activation: bool = False,
            *args,
            **kwargs
            ) -> None:
        """
        Initialize a PdeNet.

        Parameters
        ----------
        pde : str
            String identifying the PDE for which the model is constructed.
        device : str
            Device of the model.
        input_units : int
            Total number of input units (space, time, additional parameters).
        time_in_input : bool
            True if the time is provided as input.
        space_in_input : bool
            True if the space is provided as input.
        fourier_features : int
            Number of Fourier features for the encoding of spatio-temporal coordinates.
        frequency_variance : float
            Variance of the 0-centered Gaussian distribution 
            from which the frequency mtx (B) for Fourier features are sampled.
        pde_params_in_input : list
            List of the keys of the PDE parameters provided in input to the model.
        ic_params_in_input : list
            List of the keys of the initial conditions parameters provided in input to the model.
        hidden_units : list
            List of the hidden units of the model.
        activation : nn.Module
            Activation function of the network.
        sys_mode : str
            Identify the loss function system term.
        distill_mode : str
            Identify the loss function distillation term.
        ewc_mode : str
            Elastic weight consolidation on or off.
        dwa_mode : str
            Dynamic weight adaptation mode in DWA_MODES.
        bc_mode : str
            Identify the loss function boundary condition term.
        monitor_conflicts : bool
            If True and dwa is active, the last cosine similarities btw res grad and the other grads are maintained in model.res_conflicts.
        bc_weight : float
            Starting weight of the boundary condition term.
        ic_weight : float
            Starting weight of the initial condition term.
        res_weight : float
            Starting weight of the residual term.
        out_weight : float
            Starting weight of the output term.
        der_weight : float
            Starting weight of the 1st derivative term.
        hes_weight : float
            Starting weight of the 2nd derivative term.
        distill_out_weight : float
            Starting weight of the distillation output term.
        distill_der_weight : float
            Starting weight of the distillation 1st derivative term.
        distill_hes_weight : float
            Starting weight of the distillation 2nd derivative term.
        ewc_weight : float
            Starting weight of the elastic weight consolidation term.
        ewc_auto_weighting : bool
            Enabling the ewc term auto-weighting.
        ewc_warm_up : int
            Number of training steps before setting the ewc weight if ewc_auto_weighting = True.
        ewc_decay : float
            Decay factor for ewc_weight when ewc_auto_weighting = True.
        ewc_model_weights : torch.Tensor
            Optimal params of a previous model.
        ewc_fisher_diag : torch.Tensor
            Diagonal elements of the fisher information matrix relative to ewc_model_weights,
            computed on some data.
        alpha : float
            Moving average weight for dynamic weight adaptation.
        moving_avg_frequency : int
            Moving average frequency for dynamic weight adaptation.
        dwa_warm_up : int
            Warm up steps for dynamic weight adaptation.
        sys_importance : float
            Importance of system term in the loss.
        bc_importance : float
            Importance of boundary condition term in the loss.
        ic_importance : float
            Importance of initial condition term in the loss.
        distill_importance : float
            Importance of distillation term in the loss.
        ewc_importance : float
            Importance of elastic weight consolidation term in the loss.
        """
        super().__init__(*args, **kwargs)
        
        # Set the parameters
        self.device = device
        if pde == "Cahn-Hiliard":
            self.pde = "Advection-Reaction-Diffusion"
            self.cahn_hiliard = True
        else:
            self.pde = pde
            self.cahn_hiliard = False
        self.bc_mode = bc_mode
        self.time_in_input = time_in_input
        self.space_in_input = space_in_input
        self.fourier_features = fourier_features
        self.frequency_variance = frequency_variance
        if fourier_features != -1:
            torch.manual_seed(42)
            self.B = torch.randn(2 * space_in_input + time_in_input, fourier_features) * frequency_variance
            self.B = self.B.to(device)
        else:
            self.B = None
        self.pde_params_in_input = pde_params_in_input
        self.ic_params_in_input = ic_params_in_input
        items = sys_mode.split("+")
        for item in items:
            if item not in SYS_MODES:
                raise ValueError(f"Unrecognized item '{item}' in 'sys_mode' (it must be in {SYS_MODES}).")
        self.sys_mode = sys_mode
        self.bc_weight = bc_weight
        self.ic_weight = ic_weight
        self.res_weight = res_weight
        self.out_weight = out_weight
        self.der_weight = der_weight
        self.derx_weight = derx_weight
        self.dert_weight = dert_weight
        self.hes_weight = hes_weight
        self.hesx_weight = hesx_weight
        self.hest_weight = hest_weight
        self.prev_bc_weight = bc_weight
        self.prev_ic_weight = ic_weight
        self.prev_res_weight = res_weight
        self.prev_out_weight = out_weight
        self.prev_der_weight = der_weight
        self.prev_derx_weight = derx_weight
        self.prev_dert_weight = dert_weight
        self.prev_hes_weight = hes_weight
        self.prev_hesx_weight = hesx_weight
        self.prev_hest_weight = hest_weight
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.last_activation = last_activation
        self.nl_weight = nl_weight
        self.nl_bc_weight = nl_bc_weight
        self.nl_ic_weight = nl_ic_weight
        self.prev_nl_weight = nl_weight
        self.prev_nl_bc_weight = nl_bc_weight
        self.prev_nl_ic_weight = nl_ic_weight
        items = distill_mode.split("+")
        for item in items:
            if item not in DISTILL_MODES:
                raise ValueError(f"Unrecognized item '{item}' in 'sys_mode' (it must be in {DISTILL_MODES}).")
        self.distill_mode = distill_mode
        self.distill_out_weight = distill_out_weight
        self.distill_der_weight = distill_der_weight
        self.distill_derx_weight = distill_derx_weight
        self.distill_dert_weight = distill_dert_weight
        self.distill_hes_weight = distill_hes_weight
        self.distill_hesx_weight = distill_hesx_weight
        self.distill_hest_weight = distill_hest_weight
        self.prev_distill_out_weight = distill_out_weight
        self.prev_distill_der_weight = distill_der_weight
        self.prev_distill_derx_weight = distill_derx_weight
        self.prev_distill_dert_weight = distill_dert_weight
        self.prev_distill_hes_weight = distill_hes_weight
        self.prev_distill_hesx_weight = distill_hesx_weight
        self.prev_distill_hest_weight = distill_hest_weight
        if ewc_mode not in EWC_MODES:
            raise ValueError(f"Parameter 'ewc_mode' must be in {EWC_MODES}, not {ewc_mode}.")
        self.ewc_mode = ewc_mode
        self.ewc_weight = ewc_weight
        self.ewc_auto_weighting = ewc_auto_weighting
        self.ewc_model_weights = ewc_model_weights
        self.ewc_fisher_diag = ewc_fisher_diag
        if self.ewc_fisher_diag is not None and self.ewc_model_weights is not None and len(self.ewc_fisher_diag) != len(self.ewc_model_weights):
            raise ValueError(f"EWC model weights tensor length ({len(self.ewc_model_weights)}) must be equal to the Fisher information tensor length ({len(self.ewc_fisher_diag)}).")
        self.ewc_warm_up = ewc_warm_up
        self.ewc_decay = ewc_decay
        if dwa_mode not in DWA_MODES:
            raise ValueError(f"Parameter 'dwa_mode' must be in {DWA_MODES}, not {dwa_mode}.")
        self.dwa_mode = dwa_mode
        self.monitor_conflicts = monitor_conflicts
        self.res_conflicts = {
            "bc": 1.0, "ic": 1.0, 
            "out": 1.0, "der": 1.0, "derx": 1.0, "dert": 1.0, "hes": 1.0, "hesx": 1.0, "hest": 1.0, 
            "distill_out": 1.0, "distill_der": 1.0, "distill_derx": 1.0, "distill_dert": 1.0, "distill_hes": 1.0, "distill_hesx": 1.0, "distill_hest": 1.0, 
            "nl": 1.0, "nl_bc": 1.0, "nl_ic": 1.0, "ewc": 1.0
            }
        self.alpha = alpha
        self.moving_avg_frequency = moving_avg_frequency
        self.dwa_warm_up = dwa_warm_up
        self.moving_avg_count = 0
        self.bc_last_norm = None
        self.ic_last_norm = None
        self.out_last_norm = None
        self.der_last_norm = None
        self.derx_last_norm = None
        self.dert_last_norm = None
        self.hes_last_norm = None
        self.hesx_last_norm = None
        self.hest_last_norm = None
        self.res_last_norm = None
        self.nl_last_norm = None
        self.nl_bc_last_norm = None
        self.nl_ic_last_norm = None
        self.distill_out_last_norm = None
        self.distill_der_last_norm = None
        self.distill_derx_last_norm = None
        self.distill_dert_last_norm = None
        self.distill_hes_last_norm = None
        self.distill_hesx_last_norm = None
        self.distill_hest_last_norm = None
        self.ewc_last_norm = None
        self.sys_importance = sys_importance
        self.bc_importance = bc_importance
        self.ic_importance = ic_importance
        self.nl_importance = nl_importance
        self.nl_bc_importance = nl_bc_importance
        self.nl_ic_importance = nl_ic_importance
        self.distill_importance = distill_importance
        self.ewc_importance = ewc_importance
        self.activation = activation

        # Build the network
        self._build_net()

        # Define the loss container: average over all elements (it always return a scalar in R)
        self.loss_container = nn.MSELoss(reduction='mean')

    def set_fourier_features(self, fourier_features: int = -1, frequency_variance: float = 1.0) -> None:
        """
        Set the Fourier features state variables.

        Parameters
        ----------
        fourier_features : int
            The number of Fourier features of the encoding or -1 (no encoding).
        
        Returns
        -------
        None
        """
        self.fourier_features = fourier_features
        self.frequency_variance = frequency_variance
        if fourier_features != -1:
            torch.manual_seed(42)
            self.B = torch.randn(2 * self.space_in_input + self.time_in_input, fourier_features) * frequency_variance
            self.B = self.B.to(self.device)
        else:
            self.B = None
        self._build_net()

    def _build_net(self) -> None:
        """
        Build the NN and save it in the 'net' state variable.

        Returns
        -------
        None
        """
        net_dict = OrderedDict()

        # First layer
        if self.fourier_features == -1:
            net_dict['lin0'] = nn.Linear(self.input_units, self.hidden_units[0])
        else:
            n = 2 * self.fourier_features + len(self.pde_params_in_input) + len(self.ic_params_in_input)
            net_dict['lin0'] = nn.Linear(n, self.hidden_units[0])
        net_dict['act0'] = self.activation

        # Hidden layers
        for i in range(1, len(self.hidden_units)):
            net_dict[f'lin{i}'] = nn.Linear(in_features=self.hidden_units[i-1], out_features=self.hidden_units[i])
            net_dict[f'act{i}'] = self.activation

        # Last layer
        net_dict[f'lin{len(self.hidden_units)}'] = nn.Linear(self.hidden_units[-1], 1)
        if self.last_activation:
            net_dict[f'act{len(self.hidden_units)}'] = self.activation
        
        # Glorot initialization
        #for i in range(0, len(hidden_units + 1)):
        #    init.xavier_normal_(net_dict[f"lin{i}"], gain=1.0)

        # Save model
        self.net = nn.Sequential(net_dict).to(self.device)
    
    def set_ewc(
            self,
            ewc_mode: str,
            ewc_model_weights: torch.Tensor = None,
            ewc_fisher_diag: torch.Tensor = None,
            ewc_weight: float = None,
            ewc_auto_weighting: bool = None,
            ewc_warm_up: int = None,
            ewc_decay: float = None
            ) -> None:
        """
        Set the EWC state variables.

        Parameters
        ----------
        ewc_mode : str
            On/Off.
        ewc_model_weights : torch.Tensor
            Optimal params of a previous model.
        ewc_fisher_diag : torch.Tensor
            Diagonal elements of the fisher information matrix relative to ewc_model_weights,
            computed on some data.
        ewc_weight : float
            Starting weight of the elastic weight consolidation term.
        ewc_auto_weighting : bool
            Enabling the ewc term auto-weighting.
        ewc_warm_up : int
            Number of warm up steps before auto weighting when ewc_auto_weighting = True.
        ewc_decay : float
            Decay factor for ewc_weight when ewc_auto_weighting = True.
        
        Returns
        -------
        None
        """
        if ewc_mode not in EWC_MODES:
            raise ValueError(f"Parameter 'ewc_mode' must be in {EWC_MODES}, not {ewc_mode}.")
        self.ewc_mode = ewc_mode
        if ewc_weight is not None:
            self.ewc_weight = ewc_weight
        if ewc_auto_weighting is not None:
            self.ewc_auto_weighting = ewc_auto_weighting
        if ewc_warm_up is not None:
            self.ewc_warm_up = ewc_warm_up
        if ewc_decay is not None:
            self.ewc_decay = ewc_decay 
        if ewc_fisher_diag is None and ewc_model_weights is not None:
            self.ewc_model_weights = ewc_model_weights
            if len(self.ewc_fisher_diag) != len(self.ewc_model_weights):
                raise ValueError(f"EWC model weights tensor length ({len(self.ewc_model_weights)}) must be equal to the Fisher information tensor length ({len(self.ewc_fisher_diag)}).")
        elif ewc_fisher_diag is not None and ewc_model_weights is None:
            self.ewc_fisher_diag = ewc_fisher_diag
            if len(self.ewc_fisher_diag) != len(self.ewc_model_weights):
                raise ValueError(f"EWC model weights tensor length ({len(self.ewc_model_weights)}) must be equal to the Fisher information tensor length ({len(self.ewc_fisher_diag)}).")
        else:
            if ewc_model_weights is not None:
                self.ewc_model_weights = ewc_model_weights
            if ewc_fisher_diag is not None:
                self.ewc_fisher_diag = ewc_fisher_diag
            if len(self.ewc_fisher_diag) != len(self.ewc_model_weights):
                raise ValueError(f"EWC model weights tensor length ({len(self.ewc_model_weights)}) must be equal to the Fisher information tensor length ({len(self.ewc_fisher_diag)}).")
        
    def set_dwa(self,
                dwa_mode: str,
                alpha: float = None,
                moving_avg_frequency: int = None,
                res_weight: float = None,
                out_weight: float = None,
                der_weight: float = None,
                derx_weight: float = None,
                dert_weight: float = None,
                hes_weight: float = None,
                hesx_weight: float = None,
                hest_weight: float = None,
                bc_weight: float = None,
                ic_weight: float = None,
                nl_weight: float = None,
                nl_bc_weight: float = None,
                nl_ic_weight: float = None,
                distill_out_weight: float = None,
                distill_der_weight: float = None,
                distill_derx_weight: float = None,
                distill_dert_weight: float = None,
                distill_hes_weight: float = None,
                distill_hesx_weight: float = None,
                distill_hest_weight: float = None,
                ewc_weight: float = None
                ) -> None:
        """
        Set the DWA state variables.

        Parameters
        ----------
        dwa_mode : str
            Dynamic weight adaptation mode in DWA_MODES.
        alpha : float
            DWA moving average factor.
        moving_avg_frequency : int
            DWA moving average frequency.
        res_weight : float
            Residual term starting weight.
        out_weight : float
            Output term starting weight.
        der_weight : float
            1st derivative term starting weight.
        hes_weight : float
            2nd derivative term starting weight.
        bc_weight : float
            Boundary condition term starting weight.
        ic_weight : float
            Initial condition term starting weight.
        nl_weight : float
            Unlabeled term starting weight.
        distill_out_weight : float
            Distillation output term starting weight.
        distill_der_weight : float
            Distillation 1st derivative term starting weight.
        distill_hes_weight : float
            Distillation 2nd derivative term starting weight.
        ewc_weight : float
            EWC term starting weight.

        Returns
        -------
        None
        """
        if dwa_mode not in DWA_MODES:
            raise ValueError(f"Parameter 'dwa_mode' must be in {DWA_MODES}, not {dwa_mode}.")
        self.dwa_mode = dwa_mode
        if alpha is not None: self.alpha = alpha
        if moving_avg_frequency is not None: self.moving_avg_frequency = moving_avg_frequency
        if res_weight is not None:
            self.res_weight = res_weight
            self.prev_res_weight = res_weight
        if out_weight is not None:
            self.out_weight = out_weight
            self.prev_out_weight = out_weight
        if der_weight is not None:
            self.der_weight = der_weight
            self.prev_der_weight = der_weight
        if derx_weight is not None:
            self.derx_weight = derx_weight
            self.prev_derx_weight = derx_weight
        if dert_weight is not None:
            self.dert_weight = dert_weight
            self.prev_dert_weight = dert_weight
        if hes_weight is not None:
            self.hes_weight = hes_weight
            self.prev_hes_weight = hes_weight
        if hesx_weight is not None:
            self.hesx_weight = hesx_weight
            self.prev_hesx_weight = hesx_weight
        if hest_weight is not None:
            self.hest_weight = hest_weight
            self.prev_hest_weight = hest_weight
        if bc_weight is not None:
            self.bc_weight = bc_weight
            self.prev_bc_weight = bc_weight
        if ic_weight is not None:
            self.ic_weight = ic_weight
            self.prev_ic_weight = ic_weight
        if nl_weight is not None:
            self.nl_weight = nl_weight
            self.prev_nl_weight = nl_weight
        if nl_bc_weight is not None:
            self.nl_bc_weight = nl_bc_weight
            self.prev_nl_bc_weight = nl_bc_weight
        if nl_ic_weight is not None:
            self.nl_ic_weight = nl_ic_weight
            self.prev_nl_ic_weight = nl_ic_weight
        if distill_out_weight is not None:
            self.distill_out_weight = distill_out_weight
            self.prev_distill_out_weight = distill_out_weight
        if distill_der_weight is not None:
            self.distill_der_weight = distill_der_weight
            self.prev_distill_der_weight = distill_der_weight
        if distill_derx_weight is not None:
            self.distill_derx_weight = distill_derx_weight
            self.prev_distill_derx_weight = distill_derx_weight
        if distill_dert_weight is not None:
            self.distill_dert_weight = distill_dert_weight
            self.prev_distill_dert_weight = distill_dert_weight
        if distill_hes_weight is not None:
            self.distill_hes_weight = distill_hes_weight
            self.prev_distill_hes_weight = distill_hes_weight
        if distill_hesx_weight is not None:
            self.distill_hesx_weight = distill_hesx_weight
            self.prev_distill_hesx_weight = distill_hesx_weight
        if distill_hest_weight is not None:
            self.distill_hest_weight = distill_hest_weight
            self.prev_distill_hest_weight = distill_hest_weight
        if ewc_weight is not None:
            self.ewc_weight = ewc_weight

    
    # x = [[m1, m2, ..., md], ..., [M1, M2, ..., Md]] -> net(x) = [^u([m1, ..., md]), ..., ^u([M1, ..., Md])]
    # Forward function
    def forward(self, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor:
        """
        Perform NN inference on a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters input.

        Returns
        -------
        torch.Tensor
            The output of the PINN.
        """
        if self.fourier_features != -1:
            x = 2 * torch.pi * x @ self.B
            x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        if pde_params is not None and pde_params != []:
            x = torch.cat([x, pde_params], dim=-1)
        return self.net(x).squeeze(-1)
    
    # [x1, x2, ..., xd] -> [[x1, x2, ..., xd]] -> net([[x1, x2, ..., xd]]) = [^u([x1, ..., xd])] -> ^u([x1, ..., xd])
    # Forward function for individual samples
    def forward_single(self, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor:
        """
        Perform NN inference on a single input.

        Parameters
        ----------
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters input.

        Returns
        -------
        torch.Tensor
            The output of the PINN.
        """
        if self.fourier_features != -1:
            x = 2 * torch.pi * x @ self.B
            x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        if pde_params is not None:
            x = torch.cat([x, pde_params.detach()], dim=-1)
        return self.net(x.reshape((1,-1))).reshape((-1))
    
    # Concatenates all parameters (weights and biases) of a model into a single 1D tensor.
    def get_weights(self) -> torch.Tensor:
        """
        Return the learnable parameters/weights of the PINN.

        Returns
        -------
        torch.Tensor
            The learnable parameters/weights of the PINN.
        """
        # p.view(-1) flattens each parameter tensor into a 1D tensor
        # torch.cat() concatenates 1D tensors into a single 1D tensor
        return torch.cat([param.view(-1) for param in self.parameters() if param.requires_grad])
    
    def derivative(self, order: int, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor:
        """
        Compute nth order derivative of the PINN wrt the spatio-temporal input at x.

        Parameters
        ----------
        order : int
            The order of the derivative.
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters input.

        Returns
        -------
        torch.Tensor
            The nth order derivative of the PINN at x.
        """
        v = None
        if pde_params is None:
            if order == 0:
                v = self.forward(x)#.reshape(-1) # in R^(n)
            if order == 1:
                v = vmap(jacrev(self.forward_single))(x)[:, 0, :].squeeze() # in R^(nxd) #TODO added .squeeze() also after
            elif order == 2:
                v = vmap(hessian(self.forward_single))(x)[:, 0, :, :].squeeze() # in R^(nxdxd)
            elif order == 4:
                # shape (n, out_dim, d, d, d, d)
                f_4th = hessian(hessian(self.forward_single))
                v = vmap(f_4th)(x)[:, 0, :, :, :, :].squeeze()
        else:
            if order == 0:
                v = self.forward(x, pde_params) # in R^(nx1)
            if order == 1:
                v = vmap(jacrev(self.forward_single, argnums=0), in_dims=(0, 0))(x, pde_params)[:, 0, :].squeeze() # in R^(nxd)
            elif order == 2:
                v = vmap(hessian(self.forward_single, argnums=0), in_dims=(0, 0))(x, pde_params)[:, 0, :, :].squeeze() # in R^(nxdxd)
            elif order == 4:
                f_4th = hessian(hessian(self.forward_single, argnums=0), argnums=0)
                v = vmap(f_4th, in_dims=(0, 0))(x, pde_params)[:, 0, :, :, :, :].squeeze()
        
        if order == 2 and v.dim() < 3:
            v = v.unsqueeze(0)
        return v
    
    def laplacian(self, order: int, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor:

        def lap(x_single):
            if pde_params is None:
                H = hessian(self.forward_single)(x_single)
            else:
                H = hessian(self.forward_single, argnums=0)(x_single, pde_params)

            return H[0, 0] + H[1, 1]
        
        def lap_of_lap(x_single):
            if pde_params is None:
                H2 = hessian(lap)(x_single)
            else:
                H2 = hessian(lap, argnums=0)(x_single, pde_params)
            return H2[0, 0] + H2[1, 1]

        if order == 1:
            v = vmap(lap)(x).squeeze()
        elif order == 2:   
            v = vmap(lap_of_lap)(x).squeeze()
        return v

    
    def _compute_grad_norm(self, loss: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Compute the Euclidean norm of the gradient of the loss (gradient wrt the NN parameters/weights).

        Parameters
        ----------
        loss: torch.Tensor
            Value of the loss function on some input.

        Returns
        -------
        tuple(torch.Tensor, ...), torch.Tensor
            The gradient of the loss at some input and its L2 norm.
        """
        # Compute gradients of loss w.r.t. model parameters
        grads = torch.autograd.grad(
            loss, 
            self.parameters(), 
            create_graph=False, 
            retain_graph=True, 
            allow_unused=True
            )

        # Compute total gradient norm (L2)
        total_norm = torch.norm(torch.stack([g.norm(2) for g in grads if g is not None]))

        detached_grads = tuple(
            g.detach() if g is not None else None
            for g in grads
        )
        return detached_grads, total_norm.detach()
    
    def _compute_cos_sim(self,
        grads_a: tuple[torch.Tensor, ...],
        grads_b: tuple[torch.Tensor, ...],
        norm_a: torch.Tensor, 
        norm_b: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the cosine similarity (the cosine of the angle) btw the gradients of the losses (gradient wrt the NN parameters/weights).

        Parameters
        ----------
        grads_a: tuple[torch.Tensor, ...]
            Value of the loss gradient 'a' on some input.
        grads_b: tuple[torch.Tensor, ...]
            Value of the loss gradient 'b' on some input.
        grads_a: tuple[torch.Tensor, ...]
            Value of the loss gradient norm 'a'.
        grads_b: tuple[torch.Tensor, ...]
            Value of the loss gradient norm 'b'.

        Returns
        -------
        torch.Tensor
            The cosine similarity btw gradient a and gradient b.
        """

        flat_a = []
        flat_b = []

        for ga, gb in zip(grads_a, grads_b):
            if ga is None or gb is None:
                continue
            flat_a.append(ga.view(-1))
            flat_b.append(gb.view(-1))
        
        flat_a = torch.cat(flat_a)
        flat_b = torch.cat(flat_b)
    
        grad_dot = torch.dot(flat_a, flat_b)
        cos = grad_dot / (norm_a * norm_b)

        return cos

    def _compute_loss_terms(
            self,
            task: str,
            x: torch.Tensor = None,
            #x_nl: torch.Tensor = None,
            pde_params: torch.Tensor = None,
            #pde_params_nl: torch.Tensor = None,
            ic_params: torch.Tensor = None,
            #ic_params_nl: torch.Tensor = None,
            u: torch.Tensor = None,
            du: torch.Tensor = None,
            d2u: torch.Tensor = None,
            n: torch.Tensor = None,
            residual_data: dict = None,
            #residual_data_nl: dict = None
            split_space_time: bool = False
            ) -> dict:
        """
        Compute loss terms for a given task on a given input (possibly partially unlabeled).

        Parameters
        ----------
        task : str
            BCs | ICs | EWC | system | system_all | distillation | distillation_all.
        x : torch.Tensor
            Labeled spatio-temporal input.
        x_nl : torch.Tensor
            Unlabeled spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters for labeled input.
        pde_params_nl : torch.Tensor
            PDE parameters for unlabeled input.
        ic_params : torch.Tensor
            Initial condition for labeled input.
        ic_params_nl : torch.Tensor
            Initial condition for unlabeled input.
        u : torch.Tensor
            Output labels.
        du : torch.Tensor
            1st derivative labels.
        d2u : torch.Tensor
            2nd derivative labels.
        n : torch.Tensor
            Outward normal vector at x.
        residual_data : dict
            Residual data (needed to compute the residual) for labeled input.
        residual_data_nl : dict
            Residual data (needed to compute the residual) for unlabeled input.

        Returns
        -------
        dict
            The dictionary of the loss terms values:

            {<distill/nl/''>_<out/der/hes/ic/bc/res/ewc>_loss}.
        """
        der = [None, None, None]
        loss_dict = {}

        if pde_params is not None and ic_params is not None:
            pde_params = torch.cat([pde_params, ic_params], dim=-1)
        elif pde_params is None:
            pde_params = ic_params

        #if pde_params_nl is not None and ic_params_nl is not None:
        #    pde_params_nl = torch.cat([pde_params_nl, ic_params_nl], dim=-1)
        #elif pde_params_nl is None:
        #    pde_params_nl = ic_params_nl

        if task == "BCs" or task == "replay_BCs":
            if self.bc_mode == "Dirichlet":
                bc_loss = self._compute_boundary_cond_loss_term(x=x, pde_params=pde_params, u=u)
            elif self.bc_mode == "Neumann":
                bc_loss = self._compute_boundary_cond_loss_term(x=x, pde_params=pde_params, du=du, n=n)
            else:
                raise ValueError(f"Unrecognized boundary condition mode '{self.bc_mode}'.")
            if bc_loss is None:
                return {}
            elif task == "BCs":
                return {f"bc_loss": bc_loss}
            else:
                return {f"nl_bc_loss": bc_loss}
        elif task == "ICs" or task == "replay_ICs":
            ic_loss = self._compute_initial_cond_loss_term(x=x, pde_params=pde_params, u=u)
            if ic_loss is None:
                return {}
            elif task == "ICs":
                return {"ic_loss": ic_loss}
            else:
                return {"nl_ic_loss": ic_loss}
        elif task == "EWC":
            ewc_loss = self._compute_ewc_loss_term()
            if ewc_loss is None:
                return {}
            else:
                return {"ewc_loss": ewc_loss}
        elif task == "system":
            mode = self.sys_mode
            s = ""
        elif task == "system_all":
            mode = "All"
            s = ""
        elif task == "replay":
            mode = "PINN"
            s = "nl_"
        elif task == "distillation_all":
            mode = "All"
            s = "distill_"
        elif task == "distillation":
            s = "distill_"
            mode = self.distill_mode
        else:
            raise ValueError(f"Unrecognized task '{task}' (allowed values: system | distillation | BCs | ICs | EWC)")

        if mode == "All":
            # Get the prediction on new points
            u_pred = self.forward(x, pde_params) # in R^(nx1)

            # Evaluate the 1st derivative of the network wrt input
            du_pred = self.derivative(order=1, x=x, pde_params=pde_params) # in R^(nxd)

            # Evaluate the 2nd derivative of the network wrt input
            d2u_pred = self.derivative(order=2, x=x, pde_params=pde_params) # in R^(nxdxd)

            if self.cahn_hiliard:
                lap_pred = self.laplacian(order=1, x=x, pde_params=pde_params)

                lap2_pred = self.laplacian(order=2, x=x, pde_params=pde_params)
            else:
                lap_pred = None

                lap2_pred = None
            
            # Compute the loss terms
            out_loss = self.loss_container(u_pred, u) # in R
            der_loss = self.loss_container(du_pred, du) # in R
            hes_loss = self.loss_container(d2u_pred, d2u) # in R

            if self.pde != "Allen-Cahn":
                der_loss_space = self.loss_container(du_pred[:, :2], du[:, :2])
                der_loss_time = self.loss_container(du_pred[:, 2:], du[:, 2:])

                hes_loss_space = self.loss_container(d2u_pred[:, :2, :2], d2u[:, :2, :2])
                hes_loss_time = self.loss_container(d2u_pred[:, 2, 2], d2u[:, 2, 2])
            
            if "distillation" not in task:
                pde_residual = Pde.residual(pde_name=self.pde, u=u_pred, du=du_pred, d2u=d2u_pred, lap=lap_pred, lap2=lap2_pred, **residual_data)

                # Compute the loss term
                res_loss = self.loss_container(pde_residual, torch.zeros_like(pde_residual)) # in R

                loss_dict = {f"{s}out_loss": out_loss, f"{s}der_loss": der_loss, f"{s}hes_loss": hes_loss, f"{s}res_loss": res_loss}
            else:
                loss_dict =  {f"{s}out_loss": out_loss, f"{s}der_loss": der_loss, f"{s}hes_loss": hes_loss}

            if self.pde != "Allen-Cahn":
                loss_dict[f"{s}derx_loss"] = der_loss_space
                loss_dict[f"{s}dert_loss"] = der_loss_time
    
                loss_dict[f"{s}hesx_loss"] = hes_loss_space
                loss_dict[f"{s}hest_loss"] = hes_loss_time

            return loss_dict
            
        
        if mode == "Forgetting":
            # zero distillation loss
            return {}
        
        if "Output" in mode:
            # Get the prediction
            u_pred = self.forward(x, pde_params) # in R^(nx1)

            der[0] = u_pred.reshape(-1)

            # Compute the loss term
            out_loss = self.loss_container(u_pred, u)

            loss_dict[f"{s}out_loss"] = out_loss

        du_pred = None
        d2u_pred = None

        if "Derivative" in mode and "Derivative_x" not in mode and "Derivative_t" not in mode:
            # Evaluate the 1st derivative of the network wrt input
            du_pred = self.derivative(order=1, x=x, pde_params=pde_params) # in R^(nxd)
            der[1] = du_pred

            if split_space_time:
                der_loss_space = self.loss_container(du_pred[:, :2], du[:, :2])
                der_loss_time = self.loss_container(du_pred[:, 2:], du[:, 2:])
                loss_dict[f"{s}derx_loss"] = der_loss_space
                loss_dict[f"{s}dert_loss"] = der_loss_time

            # Compute the loss term
            der_loss = self.loss_container(du_pred, du) # in R

            loss_dict[f"{s}der_loss"] = der_loss
        
        if "Derivative_x" in mode:
            if du_pred is None:
                # Evaluate the 1st derivative of the network wrt input
                du_pred = self.derivative(order=1, x=x, pde_params=pde_params) # in R^(nxd)
                der[1] = du_pred

            der_loss_space = self.loss_container(du_pred[:, :2], du[:, :2])
            loss_dict[f"{s}derx_loss"] = der_loss_space
        
        if "Derivative_t" in mode:
            # Evaluate the 1st derivative of the network wrt input
            if du_pred is None:
                du_pred = self.derivative(order=1, x=x, pde_params=pde_params) # in R^(nxd)
                der[1] = du_pred

            der_loss_time = self.loss_container(du_pred[:, 2:], du[:, 2:])
            loss_dict[f"{s}dert_loss"] = der_loss_time

        if "Hessian" in mode:
            # Evaluate the 2nd derivative of the network wrt input
            d2u_pred = self.derivative(order=2, x=x, pde_params=pde_params) # in R^(nxdxd)
            der[2] = d2u_pred

            if split_space_time:
                hes_loss_space = self.loss_container(d2u_pred[:, :2, :2], d2u[:, :2, :2])
                hes_loss_time = self.loss_container(d2u_pred[:, 2, 2], d2u[:, 2, 2])
                loss_dict[f"{s}hesx_loss"] = hes_loss_space
                loss_dict[f"{s}hest_loss"] = hes_loss_time

            # Compute the loss term
            hes_loss = self.loss_container(d2u_pred, d2u) # in R

            loss_dict[f"{s}hes_loss"] = hes_loss
        
        if "Hessian_x" in mode:
            if d2u_pred is None:
                # Evaluate the 1st derivative of the network wrt input
                d2u_pred = self.derivative(order=2, x=x, pde_params=pde_params) # in R^(nxdxd)
                der[2] = d2u_pred

            hes_loss_space = self.loss_container(d2u_pred[:, :2, :2], d2u[:, :2, :2])
            loss_dict[f"{s}hesx_loss"] = hes_loss_space
        
        if "Hessian_t" in mode:
            # Evaluate the 1st derivative of the network wrt input
            if d2u_pred is None:
                d2u_pred = self.derivative(order=2, x=x, pde_params=pde_params) # in R^(nxdxd)
                der[2] = d2u_pred

            hes_loss_time = self.loss_container(d2u_pred[:, 2, 2], d2u[:, 2, 2])
            loss_dict[f"{s}hest_loss"] = hes_loss_time

        if "PINN" in mode:
            required_derivatives = Pde.residual_required_derivatives(pde_name=self.pde)
            for order in required_derivatives:
                if der[order] is None:
                    der[order] = self.derivative(order=order, x=x, pde_params=pde_params)

            if self.cahn_hiliard:
                lap = self.laplacian(order=1, x=x, pde_params=pde_params)
                lap2 = self.laplacian(order=2, x=x, pde_params=pde_params)
            else:
                lap = None
                lap2 = None

            pde_residual = Pde.residual(pde_name=self.pde, u=der[0], du=der[1], d2u=der[2], lap=lap, lap2=lap2, **residual_data)

            #if x_nl is not None:
            #    for order in required_derivatives:
            #        der[order] = self.derivative(order=order, x=x_nl, pde_params=pde_params_nl)
            #    if self.cahn_hiliard:
            #        lap = self.laplacian(order=1, x=x_nl, pde_params=pde_params)
            #        lap2 = self.laplacian(order=2, x=x_nl, pde_params=pde_params)
            #    else:
            #        lap = None
            #        lap2 = None
            #    pde_residual_nl = Pde.residual(pde_name=self.pde, u=der[0], du=der[1], d2u=der[2], lap=lap, lap2=lap2, **residual_data_nl)
#
            #    pde_residual = torch.cat([pde_residual, pde_residual_nl], dim=0)

            # Compute the loss terms
            res_loss = self.loss_container(pde_residual, torch.zeros_like(pde_residual)) # in R

            if s == "nl_":
                loss_dict["nl_loss"] = res_loss
            else:
                loss_dict[f"{s}res_loss"] = res_loss

        if loss_dict == {}:
            raise ValueError(f"Unrecognized mode '{mode}'.")
        
        return loss_dict

    def _compute_boundary_cond_loss_term(
            self,
            x: torch.Tensor,
            pde_params: torch.Tensor,
            u: torch.Tensor = None,
            du: torch.Tensor = None,
            n: torch.Tensor = None
            ) -> torch.Tensor:
        """
        Compute the boundary condition loss term on the given input.

        Parameters
        ----------
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters for input.
        u : torch.Tensor
            Output labels.
        du : torch.Tensor
            1st derivative labels.
        n : torch.Tensor
            Outward normal vector at x.

        Returns
        -------
        torch.Tensor
            Value of the BC loss term.
        """
        if x is not None:
            if self.bc_mode == "Dirichlet":
                if u is None:
                    raise ValueError("Dirichlet boundary conditions needs u.")
                # Get the prediction on boundary points
                u_pred = self.forward(x, pde_params) # in R^(nx1)
                # boundary loss on boundary points
                bc_loss = self.loss_container(u_pred.reshape((-1)), u.reshape((-1))) # in R
            elif self.bc_mode == "Neumann":
                if du is None:
                    raise ValueError("Neumann boundary conditions needs du.")
                if n is None:
                    raise ValueError("Neumann boundary conditions needs n.")
                du_pred = self.derivative(1, x, pde_params)
                outward_flux = (du[:, :2] * n).sum(dim=1)
                outward_flux_pred = (du_pred[:, :2] * n).sum(dim=1)
                bc_loss = self.loss_container(outward_flux_pred.reshape((-1)), outward_flux.reshape((-1))) # in R
        else:
            bc_loss = None
        
        return bc_loss
    
    def _compute_initial_cond_loss_term(
            self,
            x: torch.Tensor,
            pde_params: torch.Tensor,
            u: torch.Tensor = None,
            du: torch.Tensor = None
            ) -> torch.Tensor:
        """
        Compute the initial condition loss term on the given input.

        Parameters
        ----------
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters for input.
        u : torch.Tensor
            Output labels.
        du : torch.Tensor
            1st derivative labels.

        Returns
        -------
        torch.Tensor
            Value of the IC loss term.
        """
        if x is not None:
            # Get the prediction on boundary points
            u_pred = self.forward(x, pde_params) # in R^(nx1)
            # boundary loss on boundary points
            ic_loss = self.loss_container(u_pred.reshape((-1)), u.reshape((-1))) # in R
        else:
            ic_loss = None
        
        return ic_loss
    
    def _compute_ewc_loss_term(self) -> torch.Tensor:
        """
        Compute the EWC loss term on the given input.

        Returns
        -------
        torch.Tensor
            Value of the EWC loss term.
        """
        if self.ewc_mode == "Off" or self.ewc_model_weights is None or self.ewc_fisher_diag is None:
            # Compute the loss term
            ewc_loss = None

        else:
            # Compute the loss term
            ewc_loss = torch.sum(self.ewc_fisher_diag * ((self.get_weights() - self.ewc_model_weights) ** 2))
            #print(f"ewc_fisher_diag: {torch.mean(self.ewc_fisher_diag)}\nweights: {torch.mean((self.get_weights() - self.ewc_model_weights) ** 2)}")
        return ewc_loss
    
    def _update_task_weights(
            self,
            loss_terms: dict
            ) -> None:
        """
        Update the weight of each task and the model state accordingly.

        Parameters
        ----------
        loss_terms : dict
            Dictionary of the actual loss terms values.

        Returns
        -------
        None
        """
        res_loss = loss_terms.get("res_loss")
        out_loss = loss_terms.get("out_loss")
        der_loss = loss_terms.get("der_loss")
        derx_loss = loss_terms.get("derx_loss")
        dert_loss = loss_terms.get("dert_loss")
        hes_loss = loss_terms.get("hes_loss")
        hesx_loss = loss_terms.get("hesx_loss")
        hest_loss = loss_terms.get("hest_loss")

        if res_loss is not None:
            res_loss = self.sys_importance * res_loss
        if out_loss is not None:
            out_loss = self.sys_importance * out_loss
        if der_loss is not None:
            der_loss = self.sys_importance * der_loss
        if derx_loss is not None:
            derx_loss = self.sys_importance * derx_loss
        if dert_loss is not None:
            dert_loss = self.sys_importance * dert_loss
        if hes_loss is not None:
            hes_loss = self.sys_importance * hes_loss
        if hesx_loss is not None:
            hesx_loss = self.sys_importance * hesx_loss
        if hest_loss is not None:
            hest_loss = self.sys_importance * hest_loss


        bc_loss = loss_terms.get("bc_loss")
        ic_loss = loss_terms.get("ic_loss")

        if bc_loss is not None:
            bc_loss = self.bc_importance * bc_loss
        if ic_loss is not None:
            ic_loss = self.ic_importance * ic_loss

        nl_loss = loss_terms.get("nl_loss")
        nl_bc_loss = loss_terms.get("nl_bc_loss")
        nl_ic_loss = loss_terms.get("nl_ic_loss")

        if nl_loss is not None:
            nl_loss = self.nl_importance * nl_loss
        if nl_bc_loss is not None:
            nl_bc_loss = self.nl_bc_importance * nl_bc_loss
        if nl_ic_loss is not None:
            nl_ic_loss = self.nl_ic_importance * nl_ic_loss

        distill_out_loss = loss_terms.get("distill_out_loss")
        distill_der_loss = loss_terms.get("distill_der_loss")
        distill_derx_loss = loss_terms.get("distill_derx_loss")
        distill_dert_loss = loss_terms.get("distill_dert_loss")
        distill_hes_loss = loss_terms.get("distill_hes_loss")
        distill_hesx_loss = loss_terms.get("distill_hesx_loss")
        distill_hest_loss = loss_terms.get("distill_hest_loss")

        if distill_out_loss is not None:
            distill_out_loss = self.distill_importance * distill_out_loss
        if distill_der_loss is not None:
            distill_der_loss = self.distill_importance * distill_der_loss
        if distill_derx_loss is not None:
            distill_derx_loss = self.distill_importance * distill_derx_loss
        if distill_dert_loss is not None:
            distill_dert_loss = self.distill_importance * distill_dert_loss
        if distill_hes_loss is not None:
            distill_hes_loss = self.distill_importance * distill_hes_loss
        if distill_hesx_loss is not None:
            distill_hesx_loss = self.distill_importance * distill_hesx_loss
        if distill_hest_loss is not None:
            distill_hest_loss = self.distill_importance * distill_hest_loss

        ewc_loss = loss_terms.get("ewc_loss")

        if ewc_loss is not None:
            ewc_loss = self.ewc_importance * ewc_loss
        
        active_terms = []
        
        if res_loss is not None:
            res_grad, res_norm = self._compute_grad_norm(res_loss)
            active_terms.append(res_norm)
        else:
            res_norm = 0.0
        if out_loss is not None:
            out_grad, out_norm = self._compute_grad_norm(out_loss)
            if self.monitor_conflicts:
                self.res_conflicts["out"] = self._compute_cos_sim(res_grad, out_grad, res_norm, out_norm)
            active_terms.append(out_norm)
        else:
            out_norm = 0.0
        if der_loss is not None:
            der_grad, der_norm = self._compute_grad_norm(der_loss)
            if self.monitor_conflicts:
                self.res_conflicts["der"] = self._compute_cos_sim(res_grad, der_grad, res_norm, der_norm)
            active_terms.append(der_norm)
        else:
            der_norm = 0.0
        if derx_loss is not None:
            derx_grad, derx_norm = self._compute_grad_norm(derx_loss)
            if self.monitor_conflicts:
                self.res_conflicts["derx"] = self._compute_cos_sim(res_grad, derx_grad, res_norm, derx_norm)
            active_terms.append(derx_norm)
        else:
            derx_norm = 0.0
        if dert_loss is not None:
            dert_grad, dert_norm = self._compute_grad_norm(dert_loss)
            if self.monitor_conflicts:
                self.res_conflicts["dert"] = self._compute_cos_sim(res_grad, dert_grad, res_norm, dert_norm)
            active_terms.append(dert_norm)
        else:
            dert_norm = 0.0
        if hes_loss is not None:
            hes_grad, hes_norm = self._compute_grad_norm(hes_loss)
            if self.monitor_conflicts:
                self.res_conflicts["hes"] = self._compute_cos_sim(res_grad, hes_grad, res_norm, hes_norm)
            active_terms.append(hes_norm)
        else:
            hes_norm = 0.0
        
        if hesx_loss is not None:
            hesx_grad, hesx_norm = self._compute_grad_norm(hesx_loss)
            if self.monitor_conflicts:
                self.res_conflicts["hesx"] = self._compute_cos_sim(res_grad, hesx_grad, res_norm, hesx_norm)
            active_terms.append(hesx_norm)
        else:
            hesx_norm = 0.0

        if hest_loss is not None:
            hest_grad, hest_norm = self._compute_grad_norm(hest_loss)
            if self.monitor_conflicts:
                self.res_conflicts["hest"] = self._compute_cos_sim(res_grad, hest_grad, res_norm, hest_norm)
            active_terms.append(hest_norm)
        else:
            hest_norm = 0.0

        if bc_loss is not None:
            bc_grad, bc_norm = self._compute_grad_norm(bc_loss)
            if self.monitor_conflicts:
                self.res_conflicts["bc"] = self._compute_cos_sim(res_grad, bc_grad, res_norm, bc_norm)
            active_terms.append(bc_norm)
        else:
            bc_norm = 0.0
        if ic_loss is not None:
            ic_grad, ic_norm = self._compute_grad_norm(ic_loss)
            if self.monitor_conflicts:
                self.res_conflicts["ic"] = self._compute_cos_sim(res_grad, ic_grad, res_norm, ic_norm)
            active_terms.append(ic_norm)
        else:
            ic_norm = 0.0

        if nl_loss is not None:
            nl_grad, nl_norm = self._compute_grad_norm(nl_loss)
            if self.monitor_conflicts:
                self.res_conflicts["nl"] = self._compute_cos_sim(res_grad, nl_grad, res_norm, nl_norm)
            active_terms.append(nl_norm)
        else:
            nl_norm = 0.0
        
        if nl_bc_loss is not None:
            nl_bc_grad, nl_bc_norm = self._compute_grad_norm(nl_bc_loss)
            if self.monitor_conflicts:
                self.res_conflicts["nl_bc"] = self._compute_cos_sim(res_grad, nl_bc_grad, res_norm, nl_bc_norm)
            active_terms.append(nl_bc_norm)
        else:
            nl_bc_norm = 0.0
        
        if nl_ic_loss is not None:
            nl_ic_grad, nl_ic_norm = self._compute_grad_norm(nl_ic_loss)
            if self.monitor_conflicts:
                self.res_conflicts["nl_ic"] = self._compute_cos_sim(res_grad, nl_ic_grad, res_norm, nl_ic_norm)
            active_terms.append(nl_ic_norm)
        else:
            nl_ic_norm = 0.0

        if distill_out_loss is not None:
            distill_out_grad, distill_out_norm = self._compute_grad_norm(distill_out_loss)
            if self.monitor_conflicts:
                self.res_conflicts["distill_out"] = self._compute_cos_sim(res_grad, distill_out_grad, res_norm, distill_out_norm)
            active_terms.append(distill_out_norm)
        else:
            distill_out_norm = 0.0
        if distill_der_loss is not None:
            distill_der_grad, distill_der_norm = self._compute_grad_norm(distill_der_loss)
            if self.monitor_conflicts:
                self.res_conflicts["distill_der"] = self._compute_cos_sim(res_grad, distill_der_grad, res_norm, distill_der_norm)
            active_terms.append(distill_der_norm)
        else:
            distill_der_norm = 0.0

        if distill_derx_loss is not None:
            distill_derx_grad, distill_derx_norm = self._compute_grad_norm(distill_derx_loss)
            if self.monitor_conflicts:
                self.res_conflicts["distill_derx"] = self._compute_cos_sim(res_grad, distill_derx_grad, res_norm, distill_derx_norm)
            active_terms.append(distill_derx_norm)
        else:
            distill_derx_norm = 0.0
        
        if distill_dert_loss is not None:
            distill_dert_grad, distill_dert_norm = self._compute_grad_norm(distill_dert_loss)
            if self.monitor_conflicts:
                self.res_conflicts["distill_dert"] = self._compute_cos_sim(res_grad, distill_dert_grad, res_norm, distill_dert_norm)
            active_terms.append(distill_dert_norm)
        else:
            distill_dert_norm = 0.0

        if distill_hes_loss is not None:
            distill_hes_grad, distill_hes_norm = self._compute_grad_norm(distill_hes_loss)
            if self.monitor_conflicts:
                self.res_conflicts["distill_hes"] = self._compute_cos_sim(res_grad, distill_hes_grad, res_norm, distill_hes_norm)
            active_terms.append(distill_hes_norm)
        else:
            distill_hes_norm = 0.0
        
        if distill_hesx_loss is not None:
            distill_hesx_grad, distill_hesx_norm = self._compute_grad_norm(distill_hesx_loss)
            if self.monitor_conflicts:
                self.res_conflicts["distill_hesx"] = self._compute_cos_sim(res_grad, distill_hesx_grad, res_norm, distill_hesx_norm)
            active_terms.append(distill_hesx_norm)
        else:
            distill_hesx_norm = 0.0

        if distill_hest_loss is not None:
            distill_hest_grad, distill_hest_norm = self._compute_grad_norm(distill_hest_loss)
            if self.monitor_conflicts:
                self.res_conflicts["distill_hest"] = self._compute_cos_sim(res_grad, distill_hest_grad, res_norm, distill_hest_norm)
            active_terms.append(distill_hest_norm)
        else:
            distill_hest_norm = 0.0

        if ewc_loss is not None:
            ewc_grad, ewc_norm = self._compute_grad_norm(ewc_loss)
            if self.monitor_conflicts:
                self.res_conflicts["ewc"] = self._compute_cos_sim(res_grad, ewc_grad, res_norm, ewc_norm)
            #active_terms.append(ewc_norm)
        else:
            ewc_norm = 0.0

        norm_sum = sum(active_terms)

        active_weights = []

        if res_loss is not None:
            self.prev_res_weight = self.res_weight
            if self.res_last_norm is not None:
                res_weight_new = norm_sum / self.res_last_norm
            else:
                res_weight_new = norm_sum / res_norm
            if not res_weight_new.isnan() and not res_weight_new.isinf():
                self.res_last_norm = res_norm
                self.res_weight = self.alpha * self.res_weight + (1 - self.alpha) * res_weight_new
            else:
                self.res_last_norm = torch.tensor(0.0, device=self.device)
                self.res_weight = 0.0
            active_weights.append(self.res_weight)
        
        if out_loss is not None:
            self.prev_out_weight = self.out_weight
            if self.out_last_norm is not None:
                out_weight_new = norm_sum / self.out_last_norm
            else:                
                out_weight_new = norm_sum / out_norm
            if not out_weight_new.isnan() and not out_weight_new.isinf():
                self.out_last_norm = out_norm
                self.out_weight = self.alpha * self.out_weight + (1 - self.alpha) * out_weight_new
            else:
                self.out_last_norm = torch.tensor(0.0, device=self.device)
                self.out_weight = 0.0
            active_weights.append(self.out_weight)

        if der_loss is not None:
            self.prev_der_weight = self.der_weight
            if self.der_last_norm is not None:
                der_weight_new = norm_sum / self.der_last_norm
            else:
                der_weight_new = norm_sum / der_norm
            if not der_weight_new.isnan() and not der_weight_new.isnan():
                self.der_last_norm = der_norm
                self.der_weight = self.alpha * self.der_weight + (1 - self.alpha) * der_weight_new
            else:
                self.der_last_norm = torch.tensor(0.0, device=self.device)
                self.der_weight = 0.0
            active_weights.append(self.der_weight)
        
        if derx_loss is not None:
            self.prev_derx_weight = self.derx_weight
            if self.derx_last_norm is not None:
                derx_weight_new = norm_sum / self.derx_last_norm
            else:
                derx_weight_new = norm_sum / derx_norm
            if not derx_weight_new.isnan() and not derx_weight_new.isinf():
                self.derx_last_norm = derx_norm
                self.derx_weight = self.alpha * self.derx_weight + (1 - self.alpha) * derx_weight_new
            else:
                self.derx_last_norm = torch.tensor(0.0, device=self.device)
                self.derx_weight = 0.0
            active_weights.append(self.derx_weight)

        if dert_loss is not None:
            self.prev_dert_weight = self.dert_weight
            if self.dert_last_norm is not None:
                dert_weight_new = norm_sum / self.dert_last_norm
            else:
                dert_weight_new = norm_sum / dert_norm
            if not dert_weight_new.isnan() and not dert_weight_new.isinf():
                self.dert_last_norm = dert_norm
                self.dert_weight = self.alpha * self.dert_weight + (1 - self.alpha) * dert_weight_new
            else:
                self.dert_last_norm = torch.tensor(0.0, device=self.device)
                self.dert_weight = 0.0
            active_weights.append(self.dert_weight)

        if hes_loss is not None:
            self.prev_hes_weight = self.hes_weight
            if self.hes_last_norm is not None:
                hes_weight_new = norm_sum / self.hes_last_norm
            else:
                hes_weight_new = norm_sum / hes_norm
            if hes_weight_new is not None and not hes_weight_new.isnan() and not hes_weight_new.isinf():
                self.hes_last_norm = hes_norm
                self.hes_weight = self.alpha * self.hes_weight + (1 - self.alpha) * hes_weight_new
            else:
                self.hes_last_norm = torch.tensor(0.0, device=self.device)
                self.hes_weight = 0.0
            active_weights.append(self.hes_weight)
        
        if hesx_loss is not None:
            self.prev_hesx_weight = self.hesx_weight
            if self.hesx_last_norm is not None:
                hesx_weight_new = norm_sum / self.hesx_last_norm
            else:
                hesx_weight_new = norm_sum / hesx_norm
            if hesx_weight_new is not None and not hesx_weight_new.isnan() and not hesx_weight_new.isinf():
                self.hesx_last_norm = hesx_norm
                self.hesx_weight = self.alpha * self.hesx_weight + (1 - self.alpha) * hesx_weight_new
            else:
                self.hesx_last_norm = torch.tensor(0.0, device=self.device)
                self.hesx_weight = 0.0
            active_weights.append(self.hesx_weight)
        
        if hest_loss is not None:
            self.prev_hest_weight = self.hest_weight
            if self.hest_last_norm is not None:
                hest_weight_new = norm_sum / self.hest_last_norm
            else:
                hest_weight_new = norm_sum / hest_norm
            if hest_weight_new is not None and not hest_weight_new.isnan() and not hest_weight_new.isinf():
                self.hest_last_norm = hest_norm
                self.hest_weight = self.alpha * self.hest_weight + (1 - self.alpha) * hest_weight_new
            else:
                self.hest_last_norm = torch.tensor(0.0, device=self.device)
                self.hest_weight = 0.0
            active_weights.append(self.hest_weight)

        if bc_loss is not None:
            self.prev_bc_weight = self.bc_weight
            if self.bc_last_norm is not None:
                bc_weight_new = norm_sum / self.bc_last_norm
            else:
                bc_weight_new = norm_sum / bc_norm
            if bc_weight_new is not None and not bc_weight_new.isnan() and not bc_weight_new.isinf():
                self.bc_last_norm = bc_norm
                self.bc_weight = self.alpha * self.bc_weight + (1 - self.alpha) * bc_weight_new
            else:
                self.bc_last_norm = torch.tensor(0.0, device=self.device)
                self.bc_weight = 0.0
            active_weights.append(self.bc_weight)
        
        if ic_loss is not None:
            self.prev_ic_weight = self.ic_weight
            if self.ic_last_norm is not None:
                ic_weight_new = norm_sum / self.ic_last_norm
            else:
                ic_weight_new = norm_sum / ic_norm
            if ic_weight_new is not None and not ic_weight_new.isnan() and not ic_weight_new.isinf():
                self.ic_last_norm = ic_norm
                self.ic_weight = self.alpha * self.ic_weight + (1 - self.alpha) * ic_weight_new
            else:
                self.ic_last_norm = torch.tensor(0.0, device=self.device)
                self.ic_weight = 0.0
            active_weights.append(self.ic_weight)
        
        if nl_loss is not None:
            self.prev_nl_weight = self.nl_weight
            if self.nl_last_norm is not None:
                nl_weight_new = norm_sum / self.nl_last_norm
            else:
                nl_weight_new = norm_sum / nl_norm
            if nl_weight_new is not None and not nl_weight_new.isnan() and not nl_weight_new.isinf():
                self.nl_last_norm = nl_norm
                self.nl_weight = self.alpha * self.nl_weight + (1 - self.alpha) * nl_weight_new
            else:
                self.nl_last_norm = torch.tensor(0.0, device=self.device)
                self.nl_weight = 0.0
            active_weights.append(self.nl_weight)
        
        if nl_bc_loss is not None:
            self.prev_nl_bc_weight = self.nl_bc_weight
            if self.nl_bc_last_norm is not None:
                nl_bc_weight_new = norm_sum / self.nl_bc_last_norm
            else:
                nl_bc_weight_new = norm_sum / nl_bc_norm
            if nl_bc_weight_new is not None and not nl_bc_weight_new.isnan() and not nl_bc_weight_new.isinf():
                self.nl_bc_last_norm = nl_bc_norm
                self.nl_bc_weight = self.alpha * self.nl_bc_weight + (1 - self.alpha) * nl_bc_weight_new
            else:
                self.nl_bc_last_norm = torch.tensor(0.0, device=self.device)
                self.nl_bc_weight = 0.0
            active_weights.append(self.nl_bc_weight)
        
        if nl_ic_loss is not None:
            self.prev_nl_ic_weight = self.nl_ic_weight
            if self.nl_ic_last_norm is not None:
                nl_ic_weight_new = norm_sum / self.nl_ic_last_norm
            else:
                nl_ic_weight_new = norm_sum / nl_ic_norm
            if nl_ic_weight_new is not None and not nl_ic_weight_new.isnan() and not nl_ic_weight_new.isinf():
                self.nl_ic_last_norm = nl_ic_norm
                self.nl_ic_weight = self.alpha * self.nl_ic_weight + (1 - self.alpha) * nl_ic_weight_new
            else:
                self.nl_ic_last_norm = torch.tensor(0.0, device=self.device)
                self.nl_ic_weight = 0.0
            active_weights.append(self.nl_ic_weight)

        if distill_out_loss is not None:
            self.prev_distill_out_weight = self.distill_out_weight
            if self.distill_out_last_norm is not None:
                distill_out_weight_new = norm_sum / self.distill_out_last_norm
            else:                
                distill_out_weight_new = norm_sum / distill_out_norm
            if distill_out_weight_new is not None and not distill_out_weight_new.isnan() and not distill_out_weight_new.isinf():
                self.distill_out_last_norm = distill_out_norm
                self.distill_out_weight = self.alpha * self.distill_out_weight + (1 - self.alpha) * distill_out_weight_new
            else:
                self.distill_out_last_norm = torch.tensor(0.0, device=self.device)
                self.distill_out_weight = 0.0
            active_weights.append(self.distill_out_weight)

        if distill_der_loss is not None:
            self.prev_distill_der_weight = self.distill_der_weight
            if self.distill_der_last_norm is not None:
                distill_der_weight_new = norm_sum / self.distill_der_last_norm
            else:
                distill_der_weight_new = norm_sum / distill_der_norm
            if distill_der_weight_new is not None and not distill_der_weight_new.isnan() and not distill_der_weight_new.isinf():
                self.distill_der_last_norm = distill_der_norm
                self.distill_der_weight = self.alpha * self.distill_der_weight + (1 - self.alpha) * distill_der_weight_new
            else:
                self.distill_der_last_norm = torch.tensor(0.0, device=self.device)
                self.distill_der_weight = 0.0
            active_weights.append(self.distill_der_weight)
        
        if distill_derx_loss is not None:
            self.prev_distill_derx_weight = self.distill_derx_weight
            if self.distill_derx_last_norm is not None:
                distill_derx_weight_new = norm_sum / self.distill_derx_last_norm
            else:
                distill_derx_weight_new = norm_sum / distill_derx_norm
            if distill_derx_weight_new is not None and not distill_derx_weight_new.isnan() and not distill_derx_weight_new.isinf():
                self.distill_derx_last_norm = distill_derx_norm
                self.distill_derx_weight = self.alpha * self.distill_derx_weight + (1 - self.alpha) * distill_derx_weight_new
            else:
                self.distill_derx_last_norm = torch.tensor(0.0, device=self.device)
                self.distill_derx_weight = 0.0
            active_weights.append(self.distill_derx_weight)
        
        if distill_dert_loss is not None:
            self.prev_distill_dert_weight = self.distill_dert_weight
            if self.distill_dert_last_norm is not None:
                distill_dert_weight_new = norm_sum / self.distill_dert_last_norm
            else:
                distill_dert_weight_new = norm_sum / distill_dert_norm
            if distill_dert_weight_new is not None and not distill_dert_weight_new.isnan() and not distill_dert_weight_new.isinf():
                self.distill_dert_last_norm = distill_dert_norm
                self.distill_dert_weight = self.alpha * self.distill_dert_weight + (1 - self.alpha) * distill_dert_weight_new
            else:
                self.distill_dert_last_norm = torch.tensor(0.0, device=self.device)
                self.distill_dert_weight = 0.0
            active_weights.append(self.distill_dert_weight)

        if distill_hes_loss is not None:
            self.prev_distill_hes_weight = self.distill_hes_weight
            if self.distill_hes_last_norm is not None:
                distill_hes_weight_new = norm_sum / self.distill_hes_last_norm
            else:
                distill_hes_weight_new = norm_sum / distill_hes_norm
            if distill_hes_weight_new is not None and not distill_hes_weight_new.isnan() and not distill_hes_weight_new.isinf():
                self.distill_hes_last_norm = distill_hes_norm
                self.distill_hes_weight = self.alpha * self.distill_hes_weight + (1 - self.alpha) * distill_hes_weight_new
            else:
                self.distill_hes_last_norm = torch.tensor(0.0, device=self.device)
                self.distill_hes_weight = 0.0
            active_weights.append(self.distill_hes_weight)
        
        if distill_hesx_loss is not None:
            self.prev_distill_hesx_weight = self.distill_hesx_weight
            if self.distill_hesx_last_norm is not None:
                distill_hesx_weight_new = norm_sum / self.distill_hesx_last_norm
            else:
                distill_hesx_weight_new = norm_sum / distill_hesx_norm
            if distill_hesx_weight_new is not None and not distill_hesx_weight_new.isnan() and not distill_hesx_weight_new.isinf():
                self.distill_hesx_last_norm = distill_hesx_norm
                self.distill_hesx_weight = self.alpha * self.distill_hesx_weight + (1 - self.alpha) * distill_hesx_weight_new
            else:
                self.distill_hesx_last_norm = torch.tensor(0.0, device=self.device)
                self.distill_hesx_weight = 0.0
            active_weights.append(self.distill_hesx_weight)
        
        if distill_hest_loss is not None:
            self.prev_distill_hest_weight = self.distill_hest_weight
            if self.distill_hest_last_norm is not None:
                distill_hest_weight_new = norm_sum / self.distill_hest_last_norm
            else:
                distill_hest_weight_new = norm_sum / distill_hest_norm
            if distill_hest_weight_new is not None and not distill_hest_weight_new.isnan() and not distill_hest_weight_new.isinf():
                self.distill_hest_last_norm = distill_hest_norm
                self.distill_hest_weight = self.alpha * self.distill_hest_weight + (1 - self.alpha) * distill_hest_weight_new
            else:
                self.distill_hest_last_norm = torch.tensor(0.0, device=self.device)
                self.distill_hest_weight = 0.0
            active_weights.append(self.distill_hest_weight)

        #if ewc_loss is not None:
        #    if self.ewc_last_norm is not None:
        #        ewc_weight_new = norm_sum / self.ewc_last_norm
        #    else:
        #        ewc_weight_new = norm_sum / ewc_norm
        #    self.ewc_last_norm = ewc_norm
        #    self.prev_ewc_weight = self.ewc_weight
        #    self.ewc_weight = self.alpha * self.ewc_weight + (1 - self.alpha) * ewc_weight_new
        #    active_weights.append(self.ewc_weight)

        if self.dwa_mode != "Std": # Weight normalization
            weight_sum = sum(active_weights)
            # Normalize weights in such a way they sum to 1
            if self.dwa_mode == "Norm1":
                k = 1
            else: # self.dwa_mode == "NormK"
                # Normalize weights in such a way they sum to |loss_terms|
                k = len(active_weights)

            if res_loss is not None:
                self.res_weight = self.res_weight * k / weight_sum
            if out_loss is not None:
                self.out_weight = self.out_weight * k / weight_sum
            if der_loss is not None:
                self.der_weight = self.der_weight * k / weight_sum
            if derx_loss is not None:
                self.derx_weight = self.derx_weight * k / weight_sum
            if dert_loss is not None:
                self.dert_weight = self.dert_weight * k / weight_sum
            if hes_loss is not None:
                self.hes_weight = self.hes_weight * k / weight_sum
            if hesx_loss is not None:
                self.hesx_weight = self.hesx_weight * k / weight_sum
            if hest_loss is not None:
                self.hest_weight = self.hest_weight * k / weight_sum
            if bc_loss is not None:
                self.bc_weight = self.bc_weight * k / weight_sum
            if ic_loss is not None:
                self.ic_weight = self.ic_weight * k / weight_sum        
            if nl_loss is not None:
                self.nl_weight = self.nl_weight * k / weight_sum
            if nl_bc_loss is not None:
                self.nl_bc_weight = self.nl_bc_weight * k / weight_sum
            if nl_ic_loss is not None:
                self.nl_ic_weight = self.nl_ic_weight * k / weight_sum
            if distill_out_loss is not None:
                self.distill_out_weight = self.distill_out_weight * k / weight_sum
            if distill_der_loss is not None:
                self.distill_der_weight = self.distill_der_weight * k / weight_sum
            if distill_derx_loss is not None:
                self.distill_derx_weight = self.distill_derx_weight * k / weight_sum
            if distill_dert_loss is not None:
                self.distill_dert_weight = self.distill_dert_weight * k / weight_sum
            if distill_hes_loss is not None:
                self.distill_hes_weight = self.distill_hes_weight * k / weight_sum
            if distill_hesx_loss is not None:
                self.distill_hesx_weight = self.distill_hesx_weight * k / weight_sum
            if distill_hest_loss is not None:
                self.distill_hest_weight = self.distill_hest_weight * k / weight_sum
            #if ewc_loss is not None:
            #    self.ewc_weight = self.ewc_weight * num_terms / weight_sum


    def _update_grad_norms(
            self,
            loss_terms: dict
            ) -> None:
        """
        Update the weight of each task and the model state accordingly.

        Parameters
        ----------
        loss_terms : dict
            Dictionary of the actual loss terms values.

        Returns
        -------
        None
        """
        res_loss = loss_terms.get("res_loss")
        out_loss = loss_terms.get("out_loss")
        der_loss = loss_terms.get("der_loss")
        derx_loss = loss_terms.get("derx_loss")
        dert_loss = loss_terms.get("dert_loss")
        hes_loss = loss_terms.get("hes_loss")
        hesx_loss = loss_terms.get("hesx_loss")
        hest_loss = loss_terms.get("hest_loss")

        if res_loss is not None:
            res_loss = self.sys_importance * res_loss
        if out_loss is not None:
            out_loss = self.sys_importance * out_loss
        if der_loss is not None:
            der_loss = self.sys_importance * der_loss
        if derx_loss is not None:
            derx_loss = self.sys_importance * derx_loss
        if dert_loss is not None:
            dert_loss = self.sys_importance * dert_loss
        if hes_loss is not None:
            hes_loss = self.sys_importance * hes_loss
        if hesx_loss is not None:
            hesx_loss = self.sys_importance * hesx_loss
        if hest_loss is not None:
            hest_loss = self.sys_importance * hest_loss

        bc_loss = loss_terms.get("bc_loss")
        ic_loss = loss_terms.get("ic_loss")

        if bc_loss is not None:
            bc_loss = self.bc_importance * bc_loss
        if ic_loss is not None:
            ic_loss = self.ic_importance * ic_loss

        nl_loss = loss_terms.get("nl_loss")
        nl_bc_loss = loss_terms.get("nl_bc_loss")
        nl_ic_loss = loss_terms.get("nl_ic_loss")

        if nl_loss is not None:
            nl_loss = self.nl_importance * nl_loss
        if nl_bc_loss is not None:
            nl_bc_loss = self.nl_bc_importance * nl_bc_loss
        if nl_ic_loss is not None:
            nl_ic_loss = self.nl_ic_importance * nl_ic_loss

        distill_out_loss = loss_terms.get("distill_out_loss")
        distill_der_loss = loss_terms.get("distill_der_loss")
        distill_derx_loss = loss_terms.get("distill_derx_loss")
        distill_dert_loss = loss_terms.get("distill_dert_loss")
        distill_hes_loss = loss_terms.get("distill_hes_loss")
        distill_hesx_loss = loss_terms.get("distill_hesx_loss")
        distill_hest_loss = loss_terms.get("distill_hest_loss")

        if distill_out_loss is not None:
            distill_out_loss = self.distill_importance * distill_out_loss
        if distill_der_loss is not None:
            distill_der_loss = self.distill_importance * distill_der_loss
        if distill_derx_loss is not None:
            distill_derx_loss = self.distill_importance * distill_derx_loss
        if distill_dert_loss is not None:
            distill_dert_loss = self.distill_importance * distill_dert_loss
        if distill_hes_loss is not None:
            distill_hes_loss = self.distill_importance * distill_hes_loss
        if distill_hesx_loss is not None:
            distill_hesx_loss = self.distill_importance * distill_hesx_loss
        if distill_hest_loss is not None:
            distill_hest_loss = self.distill_importance * distill_hest_loss

        ewc_loss = loss_terms.get("ewc_loss")

        if ewc_loss is not None:
            ewc_loss = self.ewc_importance * ewc_loss
        
        if res_loss is not None:
            res_grad, res_norm = self._compute_grad_norm(res_loss)
            self.res_last_norm = res_norm
        
        if out_loss is not None:
            out_grad, out_norm = self._compute_grad_norm(out_loss)
            self.out_last_norm = out_norm
            if self.monitor_conflicts:
                self.res_conflicts["out"] = self._compute_cos_sim(res_grad, out_grad, res_norm, out_norm)
        
        if der_loss is not None:
            der_grad, der_norm = self._compute_grad_norm(der_loss)
            self.der_last_norm = der_norm
            if self.monitor_conflicts:
                self.res_conflicts["der"] = self._compute_cos_sim(res_grad, der_grad, res_norm, der_norm)
            
        if derx_loss is not None:
            derx_grad, derx_norm = self._compute_grad_norm(derx_loss)
            self.derx_last_norm = derx_norm
            if self.monitor_conflicts:
                self.res_conflicts["derx"] = self._compute_cos_sim(res_grad, derx_grad, res_norm, derx_norm)
        
        if dert_loss is not None:
            dert_grad, dert_norm = self._compute_grad_norm(dert_loss)
            self.dert_last_norm = dert_norm
            if self.monitor_conflicts:
                self.res_conflicts["dert"] = self._compute_cos_sim(res_grad, dert_grad, res_norm, dert_norm)
        
        if hes_loss is not None:
            hes_grad, hes_norm = self._compute_grad_norm(hes_loss)
            self.hes_last_norm = hes_norm
            if self.monitor_conflicts:
                self.res_conflicts["hes"] = self._compute_cos_sim(res_grad, hes_grad, res_norm, hes_norm)
        
        if hesx_loss is not None:
            hesx_grad, hesx_norm = self._compute_grad_norm(hesx_loss)
            self.hesx_last_norm = hesx_norm
            if self.monitor_conflicts:
                self.res_conflicts["hesx"] = self._compute_cos_sim(res_grad, hesx_grad, res_norm, hesx_norm)
            
        if hest_loss is not None:
            hest_grad, hest_norm = self._compute_grad_norm(hest_loss)
            self.hest_last_norm = hest_norm
            if self.monitor_conflicts:
                self.res_conflicts["hest"] = self._compute_cos_sim(res_grad, hest_grad, res_norm, hest_norm)

        if bc_loss is not None:
            bc_grad, bc_norm = self._compute_grad_norm(bc_loss)
            self.bc_last_norm = bc_norm
            if self.monitor_conflicts:
                self.res_conflicts["bc"] = self._compute_cos_sim(res_grad, bc_grad, res_norm, bc_norm)
        else:
            bc_norm = 0.0
        if ic_loss is not None:
            ic_grad, ic_norm = self._compute_grad_norm(ic_loss)
            self.ic_last_norm = ic_norm
            if self.monitor_conflicts:
                self.res_conflicts["ic"] = self._compute_cos_sim(res_grad, ic_grad, res_norm, ic_norm)
        
        if nl_loss is not None:
            nl_grad, nl_norm = self._compute_grad_norm(nl_loss)
            self.nl_last_norm = nl_norm
            if self.monitor_conflicts:
                self.res_conflicts["nl"] = self._compute_cos_sim(res_grad, nl_grad, res_norm, nl_norm)
        
        if nl_bc_loss is not None:
            nl_bc_grad, nl_bc_norm = self._compute_grad_norm(nl_bc_loss)
            self.nl_bc_last_norm = nl_bc_norm
            if self.monitor_conflicts:
                self.res_conflicts["nl_bc"] = self._compute_cos_sim(res_grad, nl_bc_grad, res_norm, nl_bc_norm)
        
        if nl_ic_loss is not None:
            nl_ic_grad, nl_ic_norm = self._compute_grad_norm(nl_ic_loss)
            self.nl_ic_last_norm = nl_ic_norm
            if self.monitor_conflicts:
                self.res_conflicts["nl_ic"] = self._compute_cos_sim(res_grad, nl_ic_grad, res_norm, nl_ic_norm)
        
        if distill_out_loss is not None:
            distill_out_grad, distill_out_norm = self._compute_grad_norm(distill_out_loss)
            self.distill_out_last_norm = distill_out_norm
            if self.monitor_conflicts:
                self.res_conflicts["distill_out"] = self._compute_cos_sim(res_grad, distill_out_grad, res_norm, distill_out_norm)
        
        if distill_der_loss is not None:
            distill_der_grad, distill_der_norm = self._compute_grad_norm(distill_der_loss)
            self.distill_der_last_norm = distill_der_norm
            if self.monitor_conflicts:
                self.res_conflicts["distill_der"] = self._compute_cos_sim(res_grad, distill_der_grad, res_norm, distill_der_norm)
        
        if distill_derx_loss is not None:
            distill_derx_grad, distill_derx_norm = self._compute_grad_norm(distill_derx_loss)
            self.distill_derx_last_norm = distill_derx_norm
            if self.monitor_conflicts:
                self.res_conflicts["distill_derx"] = self._compute_cos_sim(res_grad, distill_derx_grad, res_norm, distill_derx_norm)

        if distill_dert_loss is not None:
            distill_dert_grad, distill_dert_norm = self._compute_grad_norm(distill_dert_loss)
            self.distill_dert_last_norm = distill_dert_norm
            if self.monitor_conflicts:
                self.res_conflicts["distill_dert"] = self._compute_cos_sim(res_grad, distill_dert_grad, res_norm, distill_dert_norm)
        
        if distill_hes_loss is not None:
            distill_hes_grad, distill_hes_norm = self._compute_grad_norm(distill_hes_loss)
            self.distill_hes_last_norm = distill_hes_norm
            if self.monitor_conflicts:
                self.res_conflicts["distill_hes"] = self._compute_cos_sim(res_grad, distill_hes_grad, res_norm, distill_hes_norm)
        
        if distill_hesx_loss is not None:
            distill_hesx_grad, distill_hesx_norm = self._compute_grad_norm(distill_hesx_loss)
            self.distill_hesx_last_norm = distill_hesx_norm
            if self.monitor_conflicts:
                self.res_conflicts["distill_hesx"] = self._compute_cos_sim(res_grad, distill_hesx_grad, res_norm, distill_hesx_norm)
        
        if distill_hest_loss is not None:
            distill_hest_grad, distill_hest_norm = self._compute_grad_norm(distill_hest_loss)
            self.distill_hest_last_norm = distill_hest_norm
            if self.monitor_conflicts:
                self.res_conflicts["distill_hest"] = self._compute_cos_sim(res_grad, distill_hest_grad, res_norm, distill_hest_norm)

        if ewc_loss is not None:
            ewc_grad, ewc_norm = self._compute_grad_norm(ewc_loss)
            self.ewc_last_norm = ewc_norm
            if self.monitor_conflicts:
                self.res_conflicts["ewc"] = self._compute_cos_sim(res_grad, ewc_grad, res_norm, ewc_norm)


    def _compute_weighted_loss(
            self,
            loss_terms: dict,
            delayed_weights: bool = False
            ) -> torch.Tensor:
        """
        Compute the weighted loss given the dictionary of loss terms values.

        Parameters
        ----------
        loss_terms : dict
            Dictionary of the actual loss terms values.
        delayed_weights : bool
            If True, the delayed (previous) weights values are used.

        Returns
        -------
        torch.Tensor
            Value of the actual weighted loss.
        """

        res_loss = loss_terms.get("res_loss", 0.0)
        out_loss = loss_terms.get("out_loss", 0.0)
        der_loss = loss_terms.get("der_loss", 0.0)
        derx_loss = loss_terms.get("derx_loss", 0.0)
        dert_loss = loss_terms.get("dert_loss", 0.0)
        hes_loss = loss_terms.get("hes_loss", 0.0)
        hesx_loss = loss_terms.get("hesx_loss", 0.0)
        hest_loss = loss_terms.get("hest_loss", 0.0)

        bc_loss = loss_terms.get("bc_loss", 0.0)
        ic_loss = loss_terms.get("ic_loss", 0.0)

        nl_loss = loss_terms.get("nl_loss", 0.0)
        nl_bc_loss = loss_terms.get("nl_bc_loss", 0.0)
        nl_ic_loss = loss_terms.get("nl_ic_loss", 0.0)

        distill_out_loss = loss_terms.get("distill_out_loss", 0.0)
        distill_der_loss = loss_terms.get("distill_der_loss", 0.0)
        distill_derx_loss = loss_terms.get("distill_derx_loss", 0.0)
        distill_dert_loss = loss_terms.get("distill_dert_loss", 0.0)
        distill_hes_loss = loss_terms.get("distill_hes_loss", 0.0)
        distill_hesx_loss = loss_terms.get("distill_hesx_loss", 0.0)
        distill_hest_loss = loss_terms.get("distill_hest_loss", 0.0)

        ewc_loss = loss_terms.get("ewc_loss", 0.0)

        if not delayed_weights:
            weighted_sys_loss = \
                self.res_weight * res_loss + \
                self.out_weight * out_loss + \
                self.der_weight * der_loss + \
                self.derx_weight * derx_loss + \
                self.dert_weight * dert_loss + \
                self.hes_weight * hes_loss + \
                self.hesx_weight * hesx_loss + \
                self.hest_weight * hest_loss + \
                self.bc_importance * self.bc_weight * bc_loss + \
                self.ic_importance * self.ic_weight * ic_loss

            weighted_nl_loss = \
                self.nl_weight * nl_loss + \
                self.nl_bc_weight * nl_bc_loss + \
                self.nl_ic_weight * nl_ic_loss

            weighted_distill_loss = \
                self.distill_out_weight * distill_out_loss + \
                self.distill_der_weight * distill_der_loss + \
                self.distill_derx_weight * distill_derx_loss + \
                self.distill_dert_weight * distill_dert_loss + \
                self.distill_hes_weight * distill_hes_loss + \
                self.distill_hesx_weight * distill_hesx_loss + \
                self.distill_hest_weight * distill_hest_loss
        else:
            weighted_sys_loss = \
                self.prev_res_weight * res_loss + \
                self.prev_out_weight * out_loss + \
                self.prev_der_weight * der_loss + \
                self.prev_derx_weight * derx_loss + \
                self.prev_dert_weight * dert_loss + \
                self.prev_hes_weight * hes_loss + \
                self.prev_hesx_weight * hesx_loss + \
                self.prev_hest_weight * hest_loss + \
                self.bc_importance * self.prev_bc_weight * bc_loss + \
                self.ic_importance * self.prev_ic_weight * ic_loss

            weighted_nl_loss = \
                self.prev_nl_weight * nl_loss + \
                self.prev_nl_bc_weight * nl_bc_loss + \
                self.prev_nl_ic_weight * nl_ic_loss

            weighted_distill_loss = \
                self.prev_distill_out_weight * distill_out_loss + \
                self.prev_distill_der_weight * distill_der_loss + \
                self.prev_distill_derx_weight * distill_derx_loss + \
                self.prev_distill_dert_weight * distill_dert_loss + \
                self.prev_distill_hes_weight * distill_hes_loss + \
                self.prev_distill_hesx_weight * distill_hesx_loss + \
                self.prev_distill_hest_weight * distill_hest_loss

        weighted_loss = \
            self.sys_importance * weighted_sys_loss + \
            self.nl_importance * weighted_nl_loss + \
            self.distill_importance * weighted_distill_loss
        
        if ewc_loss is not None and self.ewc_auto_weighting:
            if self.ewc_warm_up == 0:
                self.ewc_weight = (weighted_loss / (self.ewc_importance * ewc_loss)).item()
                print(f"EWC weight: {self.ewc_weight}")
                self.ewc_warm_up -= 1
            elif self.ewc_warm_up > 0:
                self.ewc_warm_up -= 1
            else:
                self.ewc_weight *= self.ewc_decay
        
            weighted_ewc_loss = self.ewc_weight * ewc_loss

            weighted_loss += (self.ewc_importance * weighted_ewc_loss)
        
        return weighted_loss

    def loss_fn(self,
        x: torch.Tensor,
        pde_params: torch.Tensor = None,
        ic_params: torch.Tensor = None,
        u: torch.Tensor = None,
        du: torch.Tensor = None,
        d2u: torch.Tensor = None,
        residual_data: dict = None,
        x_nl: torch.Tensor = None,
        pde_params_nl: torch.Tensor = None,
        ic_params_nl: torch.Tensor = None,
        residual_data_nl: dict = None,
        x_bc: torch.Tensor = None,
        n: torch.Tensor = None,
        pde_params_bc: torch.Tensor = None,
        ic_params_bc: torch.Tensor = None,
        u_bc: torch.Tensor = None,
        du_bc: torch.Tensor = None,
        x_ic: torch.Tensor = None,
        pde_params_ic: torch.Tensor = None,
        ic_params_ic: torch.Tensor = None,
        u_ic: torch.Tensor = None,

        x_nl_bc: torch.Tensor = None,
        n_nl: torch.Tensor = None,
        pde_params_nl_bc: torch.Tensor = None,
        ic_params_nl_bc: torch.Tensor = None,
        u_nl_bc: torch.Tensor = None,
        du_nl_bc: torch.Tensor = None,
        x_nl_ic: torch.Tensor = None,
        pde_params_nl_ic: torch.Tensor = None,
        ic_params_nl_ic: torch.Tensor = None,
        u_nl_ic: torch.Tensor = None,

        x_distill: torch.Tensor = None,
        pde_params_distill: torch.Tensor = None,
        ic_params_distill: torch.Tensor = None,
        u_distill: torch.Tensor = None,
        du_distill: torch.Tensor = None,
        d2u_distill: torch.Tensor = None,
        delayed_weights: bool = False
    ) -> torch.Tensor:
        """
        Evaluate the PINN loss function on the given input.

        Parameters
        ----------
        x : torch.Tensor
            Labeled spatio-temporal points inside the domain D (in int(D) subset of R^(nxd)).
        pde_params : torch.Tensor
            PDE parameters for labeled points.
        ic_params : torch.Tensor
            Initial condition parameters for labeled points in int(D).
        u : torch.Tensor
            Function labels of labeled points in int(D).
        du : torch.Tensor
            Function 1st derivative labels of labeled points in int(D).
        d2u : torch.Tensor
            Function 2nd derivative labels of labeled points in int(D).
        residual_data : dict
            Residual data in labeled points in int(D).
        x_nl : torch.Tensor
            Unlabeled spatio-temporal points in int(D).
        pde_params_nl : torch.Tensor
            PDE parameters for unlabeled points.
        ic_params_nl : torch.Tensor
            Initial condition parameters for unlabeled points.
        residual_data_nl: dict
            Residual data in unlabeled points in int(D).
        x_bc : torch.Tensor
            Spatio-temporal points in bd(D).
        n : torch.Tensor
            Outward normal at x_bc.
        pde_params_bc : torch.Tensor
            PDE parameters for points in bd(D).
        ic_params_bc : torch.Tensor
            Initial condition parameters for points in bd(D).
        u_bc : torch.Tensor
            Function labels of points in bd(D).
        du_bc : torch.Tensor
            Function derivatives of points in bd(D).
        x_ic : torch.Tensor
            Spatio-temporal points in D at t0.
        pde_params_ic : torch.Tensor
            PDE parameters for points D at t0.
        ic_params_ic : torch.Tensor
            Initial condition parameters for points in D at t0.
        u_ic : torch.Tensor
            Function labels of points in D at t0.
        x_distill : torch.Tensor
            Spatio-temporal distillation points in D.
        pde_params_distill : torch.Tensor
            PDE parameters for distillation points in D.
        ic_params_distill : torch.Tensor
            Spatio-temporal distillation points in D at t0.
        u_distill : torch.Tensor
            Function labels of distillation points in D.
        du_distill : torch.Tensor
            Function 1st derivative labels of distillation points in int(D).
        d2u_distill : torch.Tensor
            Function 2nd derivative labels of distillation points in int(D).
        delayed_weights : bool
            If True, delayed weights are used in the loss.

        Returns
        -------
        torch.Tensor
            Value of the PINN loss function.
        """
        sys_loss_terms = self._compute_loss_terms(
            task="system",
            x=x, #x_nl=x_nl,
            pde_params=pde_params, #pde_params_nl=pde_params_nl,
            ic_params=ic_params, #ic_params_nl=ic_params_nl,
            u=u, du=du, d2u=d2u,
            residual_data=residual_data#, residual_data_nl=residual_data_nl
            )
        
        if x_bc is not None:
            bc_loss_term = self._compute_loss_terms(
                task="BCs",
                x=x_bc, pde_params=pde_params_bc, ic_params=ic_params_bc,
                u=u_bc, du=du_bc, n=n
                )
        else:
            bc_loss_term = {}

        if x_ic is not None:
            ic_loss_term = self._compute_loss_terms(
                task="ICs",
                x=x_ic, pde_params=pde_params_ic, ic_params=ic_params_ic,
                u=u_ic
                )
        else:
            ic_loss_term = {}

        if x_nl is not None:
            nl_loss_term = self._compute_loss_terms(
                task="replay",
                x=x_nl,
                pde_params=pde_params_nl,
                ic_params=ic_params_nl,
                residual_data=residual_data_nl
            )
        else:
            nl_loss_term = {}
        if x_nl_bc is not None:
            nl_bc_loss_term = self._compute_loss_terms(
                task="replay_BCs",
                x=x_nl_bc,
                pde_params=pde_params_nl_bc,
                ic_params=ic_params_nl_bc,
                u=u_nl_bc, du=du_nl_bc, n=n_nl
            )
        else:
            nl_bc_loss_term = {}
        if x_nl_ic is not None:
            nl_ic_loss_term = self._compute_loss_terms(
                task="replay_ICs",
                x=x_nl_ic, pde_params=pde_params_nl_ic, ic_params=ic_params_nl_ic,
                u=u_nl_ic
            )
        else:
            nl_ic_loss_term = {}

        if x_distill is not None and self.distill_mode != "Forgetting":    
            distill_loss_terms = self._compute_loss_terms(
                task="distillation",
                x=x_distill, pde_params=pde_params_distill, ic_params=ic_params_distill,
                u=u_distill, du=du_distill, d2u=d2u_distill
                )
        else:
            distill_loss_terms = {}
        
        if self.ewc_mode == "On":
            ewc_loss_term = self._compute_loss_terms(
                task="EWC"
            )
        else:
            ewc_loss_term = {}

        loss_terms = sys_loss_terms | bc_loss_term | ic_loss_term | nl_loss_term | nl_bc_loss_term | nl_ic_loss_term | distill_loss_terms | ewc_loss_term
        if self.dwa_mode != "Off" and self.moving_avg_count % self.moving_avg_frequency == 0 and self.moving_avg_count >= self.dwa_warm_up:
            self._update_task_weights(loss_terms)
        else:
            self._update_grad_norms(loss_terms)
        self.moving_avg_count += 1

        weighted_loss = self._compute_weighted_loss(loss_terms, delayed_weights=delayed_weights)

        return weighted_loss       
        

    # Function that evaluate the various losses (1st order distillation, phase 2)
    def eval_losses(self,
        x: torch.Tensor, # points inside the domain (in R^(nxd))
        pde_params: torch.Tensor = None,
        ic_params: torch.Tensor = None,
        u: torch.Tensor = None, # target values of the solution (in R^(nx1))
        du: torch.Tensor = None, # target values of the 1st derivative of the solution (in R^(nxd))
        d2u: torch.Tensor = None, # target values of the Hessian of the solution (in R^(nxdxd))
        residual_data: dict = None,
        x_bc: torch.Tensor = None, # boundary points (in R^(nxd))
        n: torch.Tensor = None,
        pde_params_bc: torch.Tensor = None,
        ic_params_bc: torch.Tensor = None,
        u_bc: torch.Tensor = None, # boundary values (in R^(nx1))
        du_bc: torch.Tensor = None,
        x_ic: torch.Tensor = None, # initial points (in R^(nxd))
        pde_params_ic: torch.Tensor = None,
        ic_params_ic: torch.Tensor = None,
        u_ic: torch.Tensor = None, # initial values (in R^(nx1))
        x_nl: torch.Tensor = None,
        pde_params_nl: torch.Tensor = None,
        ic_params_nl: torch.Tensor = None,
        residual_data_nl: dict = None,

        x_nl_bc: torch.Tensor = None, # boundary points (in R^(nxd))
        n_nl: torch.Tensor = None,
        pde_params_nl_bc: torch.Tensor = None,
        ic_params_nl_bc: torch.Tensor = None,
        u_nl_bc: torch.Tensor = None, # boundary values (in R^(nx1))
        du_nl_bc: torch.Tensor = None,
        x_nl_ic: torch.Tensor = None, # initial points (in R^(nxd))
        pde_params_nl_ic: torch.Tensor = None,
        ic_params_nl_ic: torch.Tensor = None,
        u_nl_ic: torch.Tensor = None,

        x_distill: torch.Tensor = None, # distillation points inside the domain (in R^(nxd))
        pde_params_distill: torch.Tensor = None,
        ic_params_distill: torch.Tensor = None,
        u_distill: torch.Tensor = None, # distillation target values (in R^(nx1))
        du_distill: torch.Tensor = None, # distillation target values of the 1st derivative (previous model) (in R^(nxd))
        d2u_distill: torch.Tensor = None, # distillation target values of the Hessian (previous model) (in R^(nxdxd))
        delayed_weights: bool = False,
        split_space_time: bool = False
    ) -> dict:
        """
        Compute PINN loss terms on the given input.

        Parameters
        ----------
        x : torch.Tensor
            Spatio-temporal points inside the domain D (in D subset of R^(nxd)).
        pde_params : torch.Tensor
            PDE parameters.
        ic_params : torch.Tensor
            Initial condition parameters for points in D.
        u : torch.Tensor
            Function labels of points in D.
        du : torch.Tensor
            Function 1st derivative labels of points in D.
        d2u : torch.Tensor
            Function 2nd derivative labels of in D.
        residual_data : dict
            Residual data in points in D.
        x_bc : torch.Tensor
            Spatio-temporal points in bd(D).
        n : torch.Tensor
            Outward normal at x_bc.
        pde_params_bc : torch.Tensor
            PDE parameters for points in bd(D).
        ic_params_bc : torch.Tensor
            Initial condition parameters for points in bd(D).
        u_bc : torch.Tensor
            Function labels of points in bd(D).
        du_bc : torch.Tensor
            Function derivatives of points in bd(D).
        x_ic : torch.Tensor
            Spatio-temporal points in D at t0.
        pde_params_ic : torch.Tensor
            PDE parameters for points D at t0.
        ic_params_ic : torch.Tensor
            Initial condition parameters for points in D at t0.
        u_ic : torch.Tensor
            Function labels of points in D at t0.
        x_nl : torch.Tensor
            Spatio-temporal unlabeled points in D.
        pde_params_nl : torch.Tensor
            PDE parameters for unlabeled points in D.
        ic_params_nl : torch.Tensor
            Spatio-temporal unlabeled points in D at t0.
        residual_data_nl : torch.Tensor
            Residual data in unlabeled points in D.
        x_distill : torch.Tensor
            Spatio-temporal distillation points in D.
        pde_params_distill : torch.Tensor
            PDE parameters for distillation points in D.
        ic_params_distill : torch.Tensor
            Spatio-temporal distillation points in D at t0.
        u_distill : torch.Tensor
            Function labels of distillation points in D.
        du_distill : torch.Tensor
            Function 1st derivative labels of distillation points in int(D).
        d2u_distill : torch.Tensor
            Function 2nd derivative labels of distillation points in int(D).
        delayed_weights : bool
            If True, the delayed (previous) weights are used for the weighted loss computation.

        Returns
        -------
        dict
            PINN loss terms values.
        """
        all_loss_terms = self._compute_loss_terms(
            task="system_all",
            x=x, pde_params=pde_params, ic_params=ic_params,
            u=u, du=du, d2u=d2u,
            residual_data=residual_data,
            split_space_time=split_space_time
            )
        
        sys_loss_terms = self._compute_loss_terms(
            task="system",
            x=x, pde_params=pde_params, ic_params=ic_params,
            u=u, du=du, d2u=d2u,
            residual_data=residual_data
            )
        
        if x_bc is not None:
            bc_loss_term = self._compute_loss_terms(
                task="BCs",
                x=x_bc, pde_params=pde_params_bc, ic_params=ic_params_bc,
                u=u_bc, du=du_bc, n=n
                )
        else:
            bc_loss_term = {}

        if x_ic is not None:
            ic_loss_term = self._compute_loss_terms(
                task="ICs",
                x=x_ic, pde_params=pde_params_ic, ic_params=ic_params_ic,
                u=u_ic
                )
        else:
            ic_loss_term = {}
        
        if x_nl is not None:
            nl_loss_term = self._compute_loss_terms(
                task="replay",
                x=x_nl, pde_params=pde_params_nl, ic_params=ic_params_nl,
                residual_data=residual_data_nl
                )
        else:
            nl_loss_term = {}

        if x_nl_bc is not None:
            nl_bc_loss_term = self._compute_loss_terms(
                task="replay_BCs",
                x=x_nl_bc, pde_params=pde_params_nl_bc, ic_params=ic_params_nl_bc,
                u=u_nl_bc, du=du_nl_bc, n=n_nl
            )
        else:
            nl_bc_loss_term = {}

        if x_nl_ic is not None:
            nl_ic_loss_term = self._compute_loss_terms(
                task="replay_ICs",
                x=x_nl_ic, pde_params=pde_params_nl_ic, ic_params=ic_params_nl_ic,
                u=u_nl_ic
            )
        else:
            nl_ic_loss_term = {}

        if x_distill is not None:    
            distill_loss_terms = self._compute_loss_terms(
                task="distillation_all",
                x=x_distill, pde_params=pde_params_distill, ic_params=ic_params_distill,
                u=u_distill, du=du_distill, d2u=d2u_distill,
                split_space_time=split_space_time
                )
            active_distill_loss_terms = {}
            if self.distill_mode != "Forgetting":
                if "Output" in self.distill_mode:
                    active_distill_loss_terms["distill_out_loss"] = distill_loss_terms["distill_out_loss"]
                if "Derivative" in self.distill_mode and "Derivative_x" not in self.distill_mode and "Derivative_t" not in self.distill_mode:
                    active_distill_loss_terms["distill_der_loss"] = distill_loss_terms["distill_der_loss"]
                if "Derivative_x" in self.distill_mode:
                    active_distill_loss_terms["distill_derx_loss"] = distill_loss_terms["distill_derx_loss"]
                if "Derivative_t" in self.distill_mode:
                    active_distill_loss_terms["distill_dert_loss"] = distill_loss_terms["distill_dert_loss"]
                if "Hessian" in self.distill_mode and "Hessian_x" not in self.distill_mode and "Hessian_t" not in self.distill_mode:
                    active_distill_loss_terms["distill_hes_loss"] = distill_loss_terms["distill_hes_loss"]
                if "Hessian_x" in self.distill_mode:
                    active_distill_loss_terms["distill_hesx_loss"] = distill_loss_terms["distill_hesx_loss"]
                if "Hessian_t" in self.distill_mode:
                    active_distill_loss_terms["distill_hest_loss"] = distill_loss_terms["distill_hest_loss"]
        else:
            distill_loss_terms = {}
            active_distill_loss_terms = {}
        
        if self.ewc_fisher_diag is not None and self.ewc_model_weights is not None:
            ewc_loss_term = self._compute_loss_terms(
                task="EWC"
            )
        else:
            ewc_loss_term = {}

        train_loss_terms = sys_loss_terms | bc_loss_term | ic_loss_term | nl_loss_term | nl_bc_loss_term | nl_ic_loss_term | active_distill_loss_terms | ewc_loss_term
        
        train_loss = self._compute_weighted_loss(train_loss_terms, delayed_weights=delayed_weights)

        all_loss_terms = all_loss_terms | bc_loss_term | ic_loss_term | nl_loss_term | nl_bc_loss_term | nl_ic_loss_term | distill_loss_terms | ewc_loss_term | {"weighted_loss": train_loss}
        
        return all_loss_terms

    def label(self, dataset: str|TensorDataset) -> TensorDataset:
        """
        Label the dataset with model predictions.

        Parameters
        ----------
        dataset : str|TensorDataset
            Path to the dataset or the dataset object.

        Returns
        -------
        TensorDataset
            The labeled dataset.
        """
        if type(dataset) is str:
            if not os.path.exists(dataset):
                raise ValueError(f"'File {dataset}' not found.")
            dataset = torch.load(os.path.join(dataset), weights_only=False)
        tensors = [t.clone() for t in dataset.tensors]
        x = tensors[X].float()
        self.eval()
        with torch.no_grad():
            params_values_in_input = None
            if self.pde_params_in_input is not None and self.pde_params_in_input != []:
                param_values = tensors[PDE_VALUES]
                pde_params_in_input_indexes = [key_idx(key, self.pde) for key in self.pde_params_in_input]
                params_values_in_input = param_values[:, pde_params_in_input_indexes]
            
            if self.ic_params_in_input is not None and self.ic_params_in_input != []:
                ic_values = tensors[IC_VALUES]
                ic_params_in_input_indexes = [ic_key_idx(key, self.pde) for key in self.ic_params_in_input]    
                ic_values_in_input = ic_values[:, ic_params_in_input_indexes]
                if params_values_in_input is not None:
                    params_values_in_input = torch.cat([params_values_in_input, ic_values_in_input], dim=-1)
                else:
                    params_values_in_input = ic_values_in_input

            tensors[U] = self.forward(x, params_values_in_input)
            tensors[DU] = self.derivative(order=1, x=x, pde_params=params_values_in_input)
            tensors[D2U] = self.derivative(order=2, x=x, pde_params=params_values_in_input)
            
        return TensorDataset(*tensors)

    def get_fisher_diag(self, dataset: TensorDataset) -> torch.Tensor:
        """
        Return the vector containing the diagonal of \n
        the Fisher information matrix \n
        associated with the model parameters,\n
        computed on the data in dataset.

        Parameters
        ----------
        dataset : TensorDataset
            The dataset object.

        Returns
        -------
        torch.Tensor
            The diagonal Fisher information vector.
        """
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        fisher_diag = {name: torch.zeros_like(param).to(self.device).float() for name, param in self.named_parameters() if param.requires_grad}

        self.eval()
        for z in dataloader:
            x = z[X].to(self.device).float()
            u = z[U].to(self.device).float()
            du = z[DU].to(self.device).float()
            d2u = z[D2U].to(self.device).float()
            pde_param_values = None
            if self.pde_params_in_input is not None and self.pde_params_in_input != []:
                pde_param_values = z[PDE_VALUES].to(self.device).float()
                pde_params_in_input_indexes = [key_idx(key, self.pde) for key in self.pde_params_in_input]
                pde_param_values = pde_param_values[:, pde_params_in_input_indexes]
            
            ic_param_values = None
            if self.ic_params_in_input is not None and self.ic_params_in_input != []:
                ic_param_values = z[IC_VALUES].to(self.device).float()
                ic_params_in_input_indexes = [ic_key_idx(key, self.pde) for key in self.ic_params_in_input]
                ic_param_values = ic_param_values[:, ic_params_in_input_indexes]
                if pde_param_values is not None:
                    pde_param_values = torch.cat([pde_param_values, ic_param_values], dim=-1)
                else:
                    pde_param_values = ic_param_values

            residual_info_keys = z[RESIDUAL_KEYS].to(self.device).int()
            residual_info_values = z[RESIDUAL_VALUES].to(self.device).float()
            residual_info_dict = get_dictionary(residual_info_keys, residual_info_values, self.pde)

            self.zero_grad()
            #prev_dwa_mode = self.dwa_mode
            self.dwa_mode = "Off"
            #prev_ewc_mode = self.ewc_mode
            self.ewc_mode = "Off"
            #prev_distill_mode = self.distill_mode
            self.distill_mode = "Forgetting"

            u_pred = self.forward(x, pde_param_values)
            per_sample_loss = (u_pred - u) ** 2
            loss = per_sample_loss.mean()
            #loss = self.loss_fn(
            #    x=x, pde_params=pde_param_values,
            #    u=u, du=du, d2u=d2u,
            #    residual_data=residual_info_dict
            #)

            #self.dwa_mode = prev_dwa_mode
            #self.ewc_mode = prev_ewc_mode
            #self.distill_mode = prev_distill_mode
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_diag[name] += (param.grad.detach() ** 2) #* len(x))#TODO
            #for name, param in self.named_parameters():
            #    if param.grad is not None:
            #        fisher_diag[name] += (param.grad ** 2).detach()

        # Normalize by total number of samples
        for name in fisher_diag:
            fisher_diag[name] /= (len(dataset))
        ## Average over all samples
        #num_batches = len(dataloader)
        #for name in fisher_diag:
        #    fisher_diag[name] /= num_batches

        # Get diagonal Fisher information vector
        fisher_diag_vector = torch.cat([
            fisher_diag[name].view(-1) 
            for name, param in self.named_parameters() if param.requires_grad
        ])

        #fisher_diag_vector /= torch.mean(fisher_diag_vector)

        return fisher_diag_vector

    def evaluate(self, dataset: str|TensorDataset, split_space_time: bool = False) -> tuple:
        """
        Evaluate the model on the given dataset and return the loss terms values.

        Parameters
        ----------
        dataset : TensorDataset
            The dataset object.
        split_space_time : bool
            If True, also the losses of time and space derivatives are separatedly returned.

        Returns
        -------
        tuple
            <output loss, 1st derivative loss, (1st space derivative loss, 1st time derivative loss,) 2nd derivative loss, residual loss>.
        """
        if type(dataset) is str:
            if not os.path.exists(dataset):
                raise ValueError(f"'File {dataset}' not found.")
            dataset = torch.load(os.path.join(dataset), weights_only=False)
        tensors = dataset.tensors
        x = tensors[X].float()
        u = tensors[U]
        du = tensors[DU]
        d2u = tensors[D2U]
        self.eval()
        with torch.no_grad():
            params_values_in_input = None
            if self.pde_params_in_input is not None and self.pde_params_in_input != []:
                param_values = tensors[PDE_VALUES]
                pde_params_in_input_indexes = [key_idx(key, self.pde) for key in self.pde_params_in_input]
                params_values_in_input = param_values[:, pde_params_in_input_indexes]
            else:
                params_values_in_input = None
            
            if self.ic_params_in_input is not None and self.ic_params_in_input != []:
                ic_values = tensors[IC_VALUES]
                ic_params_in_input_indexes = [ic_key_idx(key, self.pde) for key in self.ic_params_in_input]    
                ic_values_in_input = ic_values[:, ic_params_in_input_indexes]
            else:
                ic_values_in_input = None
            
            residual_info_keys = tensors[RESIDUAL_KEYS].to(self.device).int()
            residual_info_values = tensors[RESIDUAL_VALUES].to(self.device).float()
            residual_info_dict = get_dictionary(residual_info_keys, residual_info_values, self.pde)

            loss_dict = self.eval_losses(
                x=x,
                pde_params=params_values_in_input,
                ic_params=ic_values_in_input,
                residual_data=residual_info_dict,
                u=u, du=du, d2u=d2u,
                split_space_time=split_space_time
            )
            if split_space_time:
                return loss_dict["out_loss"], loss_dict["der_loss"], loss_dict["derx_loss"], loss_dict["dert_loss"], loss_dict["hes_loss"], loss_dict["hesx_loss"], loss_dict["hest_loss"], loss_dict["res_loss"]
            else:
                return loss_dict["out_loss"], loss_dict["der_loss"], loss_dict["hes_loss"], loss_dict["res_loss"]