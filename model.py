"""
model.py
===========

This module implements a PINN model.

Classes:
- Pinn: PINN class.
"""

import torch
from torch import nn
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.utils import parameters_to_vector
from phy_sys_dataset import PhySysDataset
from typing import Tuple, List, Self, Callable
import os
from physics_task import PhysicsTask, NeumannBCTask, DirichletBCTask, ICTask, OutputTask, DerivativeTask, SpatialDerivativeTask, TemporalDerivativeTask, Derivative2Task, SpatialDerivative2Task, TemporalDerivative2Task, AdvectionReactionDiffusionTask, StationaryAllenCahnTask
from AdvectionReactionDiffusion.advection_reaction_diffusion import AdvectionReactionDiffusion
from StationaryAllenCahn.allen_cahn import AllenCahn

EWC_MODES = ["On", "Off"]
DWA_MODES = ["Off", "Std", "Norm1", "NormK"]
ACTIVATION = {
    "tanh": nn.Tanh
}

# PINN class definition ----------------------
class Pinn(torch.nn.Module):
    """
    Class representing a PINN model.

    **State**
    ---------

    General attributes
    ------------------
        device : str 
            Device on which the model is.
        temporal_input : int
            Number of temporal input dimentions (0 or 1).
        spatial_input : int
            Number of spatial input dimentions (0, 1, 2 or 3).
        param_input : int
            Number of system parametrization input dimentions (0, 1, 2, ...).
        input_units : int
            Number of input units.
        hidden_units : List[int]
            List containing the number of units for each hidden layer.
        activation : callable
            The activation function of the network (Tanh).
        activation_str: str
            The string identifying the cativation function of the network in ACTIVATION ("tanh").
        train_task_list : List[PhysicsTask]
            List of tasks on which the model is trained.
        eval_task_list : List[PhysicsTask]
            List of tasks on which the model is evaluated.
        monitor_conflicts : bool
            True if training gradient conflicts are monitored.
        loss_container : callable
            The loss function (nn.MSELoss(reduction='mean')).
        net : the NN.

    Dynamic weight adaptation
    -------------------------
        dwa_mode : str
            Identifier of the training loss weighting schema used (in ["Off", "Std", "Norm1", or "NormK"]).
        dwa_alpha : float
            DWA moving avg factor.
        dwa_moving_avg_frequency : int
            DWA weights updating frequency.
        dwa_warm_up : int
            Number of steps to wait before starting DWA.

    Fourier feature encoding
    ------------------------
        ff_encoding : bool
            True if ff encoding is applied.
        B : torch.Tensor
            Frequency matrix for the random ff encoding.
        fourier_features : int
            Number of Fourier features of the ff encoding.
        frequency_variance : float
            Variance for B items sampling.
        
    Elastic weight consolidation
    ----------------------------
        ewc : bool
            True is EWC regularization is used for training.
        ewc_frictioning_weights : torch.Tensor
            Vector of weights of the EWC model.
        ewc_fisher_diag : torch.Tensor
            Diagonal of the FIM for EWC regularization.
        ewc_weight : float
            Weight of the EWC term in the loss function.
        ewc_auto_weighting : bool
            True -> EWC auto weighting application.
        ewc_warm_up : int
            Number of steps to wait before starting applying EWC regularization in training.
        ewc_decay : float
            Decay factor for the EWC term in the loss function.
    """
    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, strict, assign)
    def __init__(
            self,
            device: str = "cpu",
            activation_function_key: str = "tanh",
            temporal_input: int = 1,
            spatial_input: int = 2,
            param_input: int = 0,
            hidden_units: List[int] = [],

            ff_encoding: bool = False,
            B: torch.Tensor = None,
            fourier_features: int = None,
            B_gen_frequency_variance: float = None,
            B_gen_seed: int = 42,

            ewc: bool = False,
            ewc_weight: float = None,
            ewc_auto_weighting: bool = False,
            ewc_warm_up: int = 0, 
            ewc_decay: float = 1.0,
            ewc_frictioning_weights: torch.Tensor = None, 
            ewc_fisher_diagonal: torch.Tensor = None,

            dwa_mode: str = "Off", 
            dwa_alpha: float = None, 
            dwa_moving_avg_frequency: int = None, 
            dwa_warm_up: int = None,

            train_task_list: List[PhysicsTask] = [],
            eval_task_list: List[PhysicsTask] = [],

            monitor_conflicts: bool = False,
            conflict_reference_task: int = None,

            #loss_container: Callable = None,
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,

            input_params_dict: dict = None,
            
            *args,
            **kwargs
        ) -> None:
        """
        Constructor initializing the PINN.

        Parameters
        ----------
        device : str
        activation_str : str
            String identifying the activation function of the network (according to ACTIVATION dict).
        temporal_input : int
            1 if the time is provided as input, 0 otw.
        spatial_input : int
            Number of spatial dimensions in input.
        param_input : int
            Number of parametrization dimensions in input.
        hidden_units : List[int]
            List of the hidden units of the model.
        
        Returns
        -------
        _None_
        """
        super().__init__(*args, **kwargs)
        
        # Device
        self.device = device

        # Number of temporal inputs (0 or 1)
        self.temporal_input = temporal_input

        # Number of spatial inputs (0, 1, 2 or 3)
        self.spatial_input = spatial_input

        # Number of physical system parameters
        self.param_input = param_input

        # Total number of input units
        self.input_units = spatial_input + temporal_input + param_input

        # List of the number of hidden units of each layer
        self.hidden_units = hidden_units

        # String identifier of the NN activation function
        self.activation_function_key = activation_function_key

        # Fourier encoding
        self.ff_encoding = ff_encoding
        if self.ff_encoding:
            if B is not None:
                ## Set self.B and self.fourier_features
                self.set_ff(B=B)
                ## Variance of sampled frequencies
                self.frequency_variance = None
            elif fourier_features is None:
                raise ValueError("ff_encoding = True but both B = None and fourier_features = None.")
            elif B_gen_frequency_variance is None:
                raise ValueError("ff_encoding = True but both B = None and B_gen_frequency_variance = None.")
            else:
                ## Set self.B, self.fourier_features and self.frequency_variance
                self.sample_B_and_set_ff(n_fourier_features=fourier_features, frequency_variance=B_gen_frequency_variance, seed=B_gen_seed)
        else:
            ## Number of Fourier features
            self.fourier_features = None

            ## Variance of sampled frequencies
            self.frequency_variance = None

            ## Frequency mtx
            self.B = None

        # Build the network
        self._build_net()

        # Elastic weight consolidation
        ## This set self.ewc, self.ewc_weight, self.ewc_auto_weighting, 
        ## self.ewc_warm_up, self.ewc_decay, self.ewc_frictioning_weights, 
        ## self.ewc_fisher_diagonal
        self.set_ewc(
            ewc_on=ewc,

            ## Frictioning weights
            ewc_frictioning_weights=ewc_frictioning_weights,

            ## Fisher diagonal of the frictioning model
            ewc_fisher_diag=ewc_fisher_diagonal,

            ## Balancing weight of the EWC term in the loss function
            ewc_weight=ewc_weight,

            ## Authomatic determination of the balancing weight of the EWC term in the loss
            ewc_auto_weighting=ewc_auto_weighting,

            ## Number of training steps to wait before applying EWC regularization
            ewc_warm_up=ewc_warm_up,

            ## Decay factor for the EWC weight in the loss
            ewc_decay=ewc_decay
        )
        
        # Dynamic weight adaptation
        ## This set self.dwa_mode, self.dwa_alpha, self.dwa_moving_avg_frequency, 
        ## self.dwa_warm_up, self.dwa_moving_avg_count
        self.set_dwa(
            ## Mode
            dwa_mode=dwa_mode,

            ## Running average factor
            dwa_alpha=dwa_alpha,

            ## Frequency of update of weights
            dwa_moving_avg_frequency=dwa_moving_avg_frequency,

            ## Number of steps to wait before applying the DWA method
            dwa_warm_up=dwa_warm_up
        )

        # Set of tasks on which the PINN is trained,
        # a of PhysicsTask objects (the training objectives).
        self.train_task_list = train_task_list

        # Set of tasks on which the PINN is evaluated, 
        # a list of PhysicsTask objects (the ones on which evaluation metrics are collected).
        self.eval_task_list = eval_task_list

        # Conflicts' monitoring
        ## This set self.monitor_conflicts and self.conflict_reference_task
        self.set_conflict_monitoring(
            ## Conflict monitoring activation
            monitor_conflicts=monitor_conflicts,

            ## Index of the PhysicsTask wrt which compute the conflicts
            conflict_reference_task=conflict_reference_task
        )

        # Define the loss container: average over all elements of the loss tensor (it always return a scalar in R)
        self.loss_container = nn.MSELoss(reduction='mean')

        # The optimizer
        self.optimizer = optimizer

        # The learning rate scheduler 
        self.lr_scheduler = lr_scheduler

        self.input_params_dict = input_params_dict

        self.evaluator = Evaluator()

    def _build_net(self) -> None:
        """
        Build the NN and save it in the `net` state variable.
        """
        net_dict = OrderedDict()
        activation = ACTIVATION[self.activation_function_key]()

        # First layer
        net_dict['lin0'] = nn.Linear(self.input_units, self.hidden_units[0])
        net_dict['act0'] = activation

        # Hidden layers
        for i in range(1, len(self.hidden_units)):
            net_dict[f'lin{i}'] = nn.Linear(in_features=self.hidden_units[i-1], out_features=self.hidden_units[i])
            net_dict[f'act{i}'] = activation

        # Last layer
        net_dict[f'lin{len(self.hidden_units)}'] = nn.Linear(self.hidden_units[-1], 1)
        
        # Glorot initialization
        #for i in range(0, len(hidden_units + 1)):
        #    init.xavier_normal_(net_dict[f"lin{i}"], gain=1.0)

        # Final network architecture
        self.net = nn.Sequential(net_dict)#.to(self.device)

    def set_ff(self, B: torch.Tensor) -> None:
        """
        Set the network with the fourier feature encoding of the spatio-temporal input using B as frequency matrix.

        Parameters
        ----------
        B : torch.Tensor
            The the frequency matrix of shape (dim(spacetime), n_features).

        Returns
        -------
        _None_
        """
        # Set the B frequency matrix
        self.B = B

        # Update the model state

        ## Number of Fourier features
        self.fourier_features = B.shape[1]

        ## New total number of input units
        self.input_units = 2 * (self.spatial_input + self.temporal_input) * self.fourier_features + self.param_input

        ## Fourier encoding application
        self.ff_encoding = True

        ## Rebuild the network
        self._build_net()
    
    def sample_B_and_set_ff(self, n_fourier_features: int, frequency_variance: float, seed: int = 42) -> None:
        """
        Sample the frequency matrix B and set the network with the fourier feature encoding of the spatio-temporal input.

        Parameters
        ----------
        n_fourier_features : int
            Number of Fourier features for the encoding of spatio-temporal coordinates.
        frequency_variance : float
            Variance of the 0-centered Gaussian distribution 
            from which the frequency matrix (B) for Fourier features is sampled.
        seed : int = 42
            seed for the random sampling of B.

        Returns
        -------
        _None_
        """
        # Sample the frequency matrix B
        torch.manual_seed(seed)

        #self.B = torch.randn(2 * (self.spatial_input + self.temporal_input), n_fourier_features) * frequency_variance
        #self.B = self.B.to(self.device)

        # Random generation of the frequency mtx B s.t. it convert the spatio-temporal inputs into n_fourier_features features
        B = torch.randn(2 * (self.spatial_input + self.temporal_input), n_fourier_features) * frequency_variance

        # Set the B frequency matrix
        self.B = B

        # Update the model state
        
        ## Number of Fourier features
        self.fourier_features = n_fourier_features

        ## Variance of the 0-centered Gaussian distribution from which B is sampled.
        self.frequency_variance = frequency_variance

        ## Fourier encoding application
        self.ff_encoding = True

        ## New total number of input units
        self.input_units = 2 * (self.spatial_input + self.temporal_input) * self.fourier_features + self.param_input

        # Rebuild the network
        self._build_net()
    
    def set_dwa(
            self,
            dwa_mode: str,
            dwa_alpha: float = None, #0.9,
            dwa_moving_avg_frequency: int = None, #1,
            dwa_warm_up: int = None#0,
    ) -> None:
        """
        Set the dynamic weight adaptation schema.

        Parameters
        ----------
        dwa_mode : str
            Dynamic weight adaptation mode ("Off", "Std", "Norm1", or "NormK").
        dwa_alpha : float
            Moving average weight for dynamic weight adaptation.
        dwa_moving_avg_frequency : int
            Moving average frequency for dynamic weight adaptation.
        dwa_warm_up : int
            Warm up steps for dynamic weight adaptation.
        
        Returns
        -------
        None.
        """
        # Check the available DWA modes
        if dwa_mode not in DWA_MODES:
            raise ValueError(f"Parameter 'dwa_mode' must be in {DWA_MODES}, not {dwa_mode}.")
        if dwa_mode == "Off":
            self.dwa_mode = dwa_mode
            self.dwa_alpha = None
            self.dwa_moving_avg_frequency = None
            self.dwa_warm_up = None
            self.dwa_moving_avg_count = None
        else:
            # Check None values
            if dwa_alpha is None:
                raise ValueError(f"dwa_alpha cannot be None.")
            elif dwa_moving_avg_frequency is None:
                raise ValueError(f"dwa_moving_avg_frequency cannot be None.")
            elif dwa_warm_up is None:
                raise ValueError(f"dwa_warm_up cannot be None.")
        
            # DWA method
            self.dwa_mode = dwa_mode

            # DWA running average factor
            self.dwa_alpha = dwa_alpha

            # Frequency of update of weights
            self.dwa_moving_avg_frequency = dwa_moving_avg_frequency

            # Number of steps to wait before applying the DWA method
            self.dwa_warm_up = dwa_warm_up

            # Counter to account for the weights' update frequency
            self.dwa_moving_avg_count = 0
    
    def set_ewc(
            self,
            ewc_on: bool,
            ewc_frictioning_weights: torch.Tensor = None,
            ewc_fisher_diag: torch.Tensor = None,
            ewc_weight: float = None,
            ewc_auto_weighting: bool = False,
            ewc_warm_up: int = 0,
            ewc_decay: float = 1.0
    ) -> None:
        """
        Set EWC regularization for training.

        Parameters
        ----------
        ewc_on : bool
            True if EWC has to be applied.
        ewc_frictioning_weights : torch.Tensor
            Optimal params of a previous model.
        ewc_fisher_diag : torch.Tensor
            Diagonal elements of the fisher information matrix relative to ewc_frictioning_weights,
            evaluated on some data.
        ewc_weight : float
            Starting weight of the elastic weight consolidation term.
        ewc_auto_weighting : bool
            Enabling the ewc term auto-weighting.
        ewc_warm_up : int
            Number of training steps before setting the ewc weight if ewc_auto_weighting = True.
        ewc_decay : float
            Decay factor for ewc_weight when ewc_auto_weighting = True.

        Returns
        -------
        _None_
        """
        if not ewc_on:
            self.ewc_weight = None
            self.ewc_auto_weighting = None
            self.ewc_frictioning_weights = None
            self.ewc_fisher_diagonal = None
            self.ewc_warm_up = None
            self.ewc_decay = None
            self.ewc = False
        else:
            # Check for None values
            if ewc_frictioning_weights is None:
                raise ValueError(f"ewc_frictioning_weights cannot be None.")
            elif ewc_fisher_diag is None:
                raise ValueError(f"ewc_fisher_diag cannot be None.")
            elif ewc_weight is None:
                raise ValueError(f"ewc_weight cannot be None.")
            elif ewc_auto_weighting is None:
                raise ValueError(f"ewc_auto_weighting cannot be None.")
            elif ewc_warm_up is None:
                raise ValueError(f"ewc_warm_up cannot be None.")
            elif ewc_decay is None:
                raise ValueError(f"ewc_decay cannot be None.")

            # Size of the diagonal of the Fisher information matrix
            diag_len = len(ewc_fisher_diag)

            # Number of optimal weights of the attracting model.
            n_weights_ewc = len(ewc_frictioning_weights)

            # Number of weights of the underlying model
            n_weights = sum(p.numel() for p in self.parameters())

            # Check if the number of items in the Fisher diagonal match the number of weights of the attracting model
            if diag_len != n_weights_ewc:
                raise ValueError(f"The diagonal of the Fisher information ({diag_len} elements) must have many elements as the number parameters of the EWC model ({n_weights_ewc}).")

            # Check if the number of items in the Fisher diagonal match the number of weights of the underlying model
            if diag_len != n_weights:
                raise ValueError(f"The diagonal of the Fisher information ({diag_len} elements) must have many elements as the number parameters of the model ({n_weights}).")

            # Balancing weight of the EWC term in the loss function
            self.ewc_weight = ewc_weight

            # Authomatic determination of the balancing weight of the EWC term in the loss
            self.ewc_auto_weighting = ewc_auto_weighting

            # Set the frictioning model weights
            self.ewc_frictioning_weights = ewc_frictioning_weights

            # Set the Fisher diagonal
            self.ewc_fisher_diagonal = self.ewc_fisher_diagonal

            # Number of training steps to wait before applying EWC regularization
            self.ewc_warm_up = ewc_warm_up

            # Decay factor for the EWC weight in the loss
            self.ewc_decay = ewc_decay

            # Apply EWC
            self.ewc = True

    def set_conflict_monitoring(
            self,
            monitor_conflicts: bool,
            conflict_reference_task: int = None
    ) -> None:
        """
        Set the monitoring of training gradient conflicts.

        Parameters
        ----------
        monitor_conflicts : bool
            If True and DWA is active, updates each task.conflict attribute with the cosine similarity 
            btw the task gradient and the conflict_reference_task gradient.
        conflict_reference_task : int
            The objective wrt which the conflicts are computed.
        
        Returns
        -------
        _None_
        """
        # Apply conflict monitoring
        self.monitor_conflicts = monitor_conflicts

        # Check that conflict_reference_task is not None if conflict monitoring is active
        if monitor_conflicts:
            if conflict_reference_task is None:
                raise ValueError(f"conflict_reference_task cannot be None.")
            elif conflict_reference_task >= len(self.train_task_list):
                raise ValueError(f"conflict_reference_task cannot be >= len(self.train_task_list): {conflict_reference_task} >= {len(self.train_task_list)}.")
        
        # PhysicsTask wrt which compute the conflicts
        self.conflict_reference_task = conflict_reference_task

    # Forward function for batches of data
    def forward(self, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor:
        """
        Perform an inference step on a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters in input.

        Returns
        -------
        _torch.Tensor_
            The output of the PINN.
        """
        # Check if the number of spatio-temporal coordinates in input is correct
        if x.shape[-1] != self.spatial_input + self.temporal_input:
            raise ValueError(f"The NN expect {self.spatial_input + self.temporal_input} spatio-temporal coordinates in input, but got {x.shape[-1]} instead.")
        if self.ff_encoding:
            # Apply the Fourier encoding
            x = 2 * torch.pi * x @ self.B
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        if self.param_input != 0:
            # Check if pde_params argument has been passsed
            if pde_params is None:
                raise ValueError(f"The NN expect {self.param_input} physical system parameters in input, but pde_params = None.")
            
            # Check if the number of physical system parameters in input is correct
            if pde_params.shape[-1] != self.param_input:
                raise ValueError(f"The NN expect {self.param_input} physical system parameters in input, but got {pde_params.shape[-1]} instead.")
            
            # Concatenate the needed parameters of the physical system to the spatio-temporal input
            x = torch.cat([x, pde_params], dim=-1)

        # Apply the network function to the resulting input (batch) tensor and returns
        ## If x.shape == (1, n_inputs), i.e. x = tensor([[x, y, t]])
        if len(x.shape) == 2 and x.shape[0] == 1:
            return self.net(x).flatten() # -> tensor([scalar_pred])
        
        ## If x.shape == (n_inputs,) or x.shape == (batch_size > 1, n_inputs)
        else:
            return self.net(x).squeeze() # -> tensor(scalar_pred) or tensor([scalar_pred_1, ..., scalar_pred_batch_size])
            ## For a NN outputing a vector instead of a scalar, the .squeeze() has to be removed.
 
    # Concatenates all parameters (weights and biases) of a model into a single 1D tensor.
    def get_weights(self) -> torch.Tensor:
        """
        Concatenates an independent copy of all the learnable parameters (weights and biases) 
        of the model into a single 1D tensor and returns it.

        Parameters
        ----------
        None

        Returns
        -------
        _torch.Tensor_
            An independent copy of the learnable parameters/weights of the PINN.
        """
        # Get a tuple of all the learnable parameters of the underlying model
        learnable_params = (p for p in self.parameters() if p.requires_grad)

        # Get an independent copy of the parameters into a 1-dim vector (shape (n_params,))
        param_vector = parameters_to_vector(learnable_params)

        # Return the vector of parameters
        return param_vector
    
    def derivative(self, order: int, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor:
        """
        Compute n^th order derivative of the PINN wrt the spatio-temporal input at x.

        Parameters
        ----------
        order : int
            The order of the derivative.
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters in input.

        Returns
        -------
        _torch.Tensor_
            The nth order derivative of the PINN at x.
        """
        if order == 0:
            return self.forward(x, pde_params)
        if order == 1:
            batch_of_gradients = vmap(jacrev(self.forward, argnums=0))(x, pde_params)
            return batch_of_gradients
        elif order == 2:
            batch_of_hessians = vmap(hessian(self.forward, argnums=0))(x, pde_params)
            Hxx = batch_of_hessians[:, 0, 0]
            Hyy = batch_of_hessians[:, 1, 1]
            Htt = batch_of_hessians[:, 2, 2]
            Hxy = batch_of_hessians[:, 0, 1]
            Hxt = batch_of_hessians[:, 0, 2]
            Hyt = batch_of_hessians[:, 1, 2]
            return torch.stack([Hxx, Hyy, Htt, Hxy, Hxt, Hyt], dim=1)
    
    def _compute_grad_norm(self, loss: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Compute the Euclidean norm of the gradient of the loss (gradient wrt the NN parameters/weights).

        Parameters
        ----------
        loss: torch.Tensor
            Vector of loss function values on some input.

        Returns
        -------
        Tuple(torch.Tensor, ...), torch.Tensor
            The gradient of the loss at some input and its L2 norm.
        """
        # Compute gradients of loss w.r.t. model parameters
        grads = torch.autograd.grad(
            loss, 
            self.parameters(),
            # allow_unused=True,
            retain_graph=True # retain the graph for a successive call to loss.backward
            )

        # Compute total gradient norm (L2)
        gradient_vector = torch.cat([g.detach().flatten() for g in grads])
        gradient_norm = gradient_vector.norm()

        return gradient_vector, gradient_norm
    
    def _update_conflicts(
        self,
        task_list: List[PhysicsTask],
        reference_task: PhysicsTask
        ) -> torch.Tensor:
        """
        Compute the cosine similarity (the cosine of the angle) \n
        btw the gradients of the losses of each task in `task_list` and the gradient of `reference_task`\n
        (gradient wrt the NN parameters/weights).\n
        The variable task.conflict is updated for each task in task_list.

        Parameters
        ----------
        task_list : List[Physicstask]
            List of task for which computing the conflicts.
        reference_task : PhysicsTask
            Task wrt the conflicts are computed.

        Returns
        -------
        None
        """
        for task in task_list:
            task.conflict = torch.dot(task.grad, reference_task.grad) / (task.grad_norm * reference_task.grad_norm)
    
    def _update_task_weights(self) -> None:
        """
        Update the weight of each objective and the model state accordingly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for task in self.train_task_list:
            task.grad, task.grad_norm = self._compute_grad_norm(task.loss_value)

        if self.monitor_conflicts:
            self._update_conflicts(
                task_list=self.train_task_list, 
                reference_task=self.train_task_list[self.conflict_reference_task]
            )
        
        norm_sum = sum([task.grad_norm for task in self.train_task_list])
        
        for task in self.train_task_list:
            weight_new = norm_sum / task.grad_norm
            if weight_new.isnan() or weight_new.isinf():
                task.grad_norm = torch.tensor(0.0, device=self.device)
                task.weight = 0.0
                print("Weight reset.\n")
            else:
                task.weight = self.dwa_alpha * task.weight + (1 - self.dwa_alpha) * weight_new

        if self.dwa_mode != "Std":
            active_weights = [task.weight for task in self.train_task_list]
            weight_sum = sum(active_weights)
            # Normalize weights in such a way they sum to 1
            if self.dwa_mode == "Norm1":
                k = 1
            elif self.dwa_mode == "NormK":
                # Normalize weights in such a way they sum to |loss_terms|
                k = len(active_weights)
            else:
                raise ValueError(f"Unrecognized loss balancing mode '{self.dwa_mode}'.")
            
            for task in self.train_task_list:
                task.weight = task.weight * k / weight_sum

    def _update_train_grad_norms(self) -> None:
        """
        Update the gradient norm of each objective and the model state accordingly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for task in self.train_task_list:
            task.grad, task.grad_norm = self._compute_grad_norm(task.loss_value)

        if self.monitor_conflicts:
            self._update_conflicts(
                task_list=self.train_task_list, 
                reference_task=self.train_task_list[self.conflict_reference_task]
            )
    
    def _update_eval_grad_norms(self) -> None:
        """
        Update the gradient norm of each evaluation task and the model state accordingly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for task in self.eval_task_list:
            task.grad, task.grad_norm = self._compute_grad_norm(task.loss_value)

        if self.monitor_conflicts:
            self._update_conflicts(
                task_list=self.eval_task_list, 
                reference_task=self.train_task_list[self.conflict_reference_task]
            )
    
    def train_loss(
            self,
            x_list: List[torch.Tensor], # spatio-temporal input, for each task
            pde_param_list: List[torch.Tensor] = None, # physics parameters in input, for each task
            labels: dict = None, # true labels, if some task needs (some of) them (dictionary of lists, where each list has one item for each task)
    ) -> torch.Tensor:
        """
        Training loss function.

        Parameters
        ----------
        x_list: List[torch.Tensor]
            List of spatio-temporal inputs, one tensor (batch) for each task.
        input_param_list : List[torch.Tensor]
            List of physics parameters in input, one tensor (batch) for each task.
        labels : dict
            True labels, if some task needs (some of) them; dictionary of lists, where each list has one item (batch) for each task.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        if labels is None:
            labels = {}
        for i, task in enumerate(self.train_task_list):
            x = x_list[i]
            pde_parameters = pde_param_list[i]

            l_dict = {}
            for key in labels.keys():
                if key in task.loss_required_labels():
                    l_dict[key] = labels[key][i]
                else:
                    raise ValueError("Missing input parameters")
            
            task.loss_value = self.evaluator(model=self, task=task, x=x, pde_parameters=pde_parameters, **l_dict) #task.loss(x=x, input_params=input_params, model=self, **l_dict)

        if self.dwa_mode != "Off" and self.dwa_moving_avg_count % self.dwa_moving_avg_frequency == 0 and self.dwa_moving_avg_count >= self.dwa_warm_up:
            self._update_task_weights()
            self.dwa_moving_avg_count += 1
        else:
            self._update_train_grad_norms()

        weighted_loss = sum([task.weight * task.loss_value for task in self.train_task_list])

        if self.ewc:
            # Compute the loss term
            ewc_loss = torch.sum(self.ewc_fisher_diag * ((self.get_weights() - self.ewc_frictioning_weights) ** 2))
            
            if self.ewc_auto_weighting:
                if self.ewc_warm_up == 0:
                    self.ewc_weight = (weighted_loss / ewc_loss)#.item()
                    print(f"EWC weight: {self.ewc_weight}")
                    self.ewc_warm_up -= 1
                elif self.ewc_warm_up > 0:
                    self.ewc_warm_up -= 1
                else:
                    self.ewc_weight *= self.ewc_decay

            weighted_loss += (self.ewc_weight * ewc_loss)

        return weighted_loss
        
    def eval_loss(
            self,
            x_list: List[torch.Tensor],
            pde_param_list: List[torch.Tensor] = None,
            labels: dict = None
    ) -> torch.Tensor:
        """
        Validation loss function.

        Parameters
        ----------
        x_list: List[torch.Tensor]
            List of spatio-temporal inputs, one tensor (batch) for each task.
        input_param_list : List[torch.Tensor]
            List of physics parameters in input, one tensor (batch) for each task.
        labels : dict
            True labels, if some task needs (some of) them; dictionary of lists, where each list has one item (batch) for each task.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        if labels is None:
            labels = {}
        for i, task in enumerate(self.eval_task_list):
            x = x_list[i]
            pde_parameters = pde_param_list[i]

            l_dict = {}
            for key in labels.keys():
                if key in task.loss_required_labels():
                    l_dict[key] = labels[key][i]
                else:
                    raise ValueError("Missing input parameters")
            
            task.loss_value = self.evaluator(model=self, task=task, x=x, pde_parameters=pde_parameters, **l_dict)#task.loss(x=x, input_params=input_params, model=self, **l_dict)
            self._update_eval_grad_norms()

        weighted_loss = sum([task.weight * task.loss_value for task in self.eval_task_list])
        return weighted_loss

    def label(
            self, 
            dataset: PhySysDataset, 
            spacetime_idx: int, 
            param_idx: int, 
            u_idx: int, 
            du_idx: int, 
            d2u_idx: int, 
            param_subidxs: List[int] = None
        ) -> PhySysDataset:
        """
        Label the dataset with model predictions.

        Parameters
        ----------
        dataset : PhySysDataset
            The dataset to label.
        spacetime_idx : int
            Index of the spatio-temporal input.
        param_idx : int
            Index of the system parametrization input.
        u_idx : int
            Index of the output labels.
        du_idx : int
            Index of the 1st derivative labels.
        d2u_idx : int
            Index of the 2nd derivative labels.
        param_subidxs : List[int] = None
            Indexes of the system parameters in input.

        Returns
        -------
        PhySysDataset
            The labeled dataset.
        """
        tensors = [t.clone() for t in dataset.columns()]
        x = tensors[spacetime_idx].float()
        self.eval()
        with torch.no_grad():
            params = None
            if self.param_input != 0:
                if param_subidxs is None:
                    params = tensors[param_idx]
                else:
                    params = tensors[param_idx][:, param_subidxs]
                # params_values_in_input = torch.cat([params_values_in_input, ic_values_in_input], dim=-1)

            tensors[u_idx] = self.forward(x, params)
            tensors[du_idx] = self.derivative(order=1, x=x, pde_params=params)
            tensors[d2u_idx] = self.derivative(order=2, x=x, pde_params=params)
        labeled_dataset = PhySysDataset(cols=([(key, val) for key, val in zip(dataset.cols.keys(), tensors)]))
        labeled_dataset.subkeys = dataset.subkeys
        return labeled_dataset

    def get_fisher_diag(
            self, 
            dataset: PhySysDataset, 
            spacetime_key: str = "spacetime", 
            param_key: str = "param", 
            param_subkeys: List[str] = None, 
            u_key: str = "u"
    ) -> torch.Tensor:
        """
        Return the vector containing the diagonal of \n
        the Fisher information matrix \n
        associated with the model parameters,\n
        computed on the data in dataset.

        Parameters
        ----------
        dataset : PhySysDataset
            The dataset object.
        spacetime_key : str
            Key of the spatio-temporal coordinate column.
        param_key : str 
            Key of the physical system parameter vector column.
        param_subkeys : List[str]
            Subkeys of the entries physical system parameter vector which are given in input to the model.
        u_key : str
            Key of the unknown field column.

        Returns
        -------
        torch.Tensor
            The diagonal Fisher information vector.
        """
        self.eval()
        self.zero_grad()

        weights = self.get_weights()
        fisher_diag = torch.zeros_like(weights).to(self.device).float()

        x = dataset.cols[spacetime_key]
        u = dataset.cols[u_key]
        param_subidxs = [dataset.index(key=param_key, subkey=subkey) for subkey in param_subkeys]
        params = dataset.cols[param_key]
        if param_subkeys != None:
            params = params[:, param_subidxs]

        old_dwa_mode = self.dwa_mode
        self.dwa_mode = "Off"
        old_ewc = self.ewc
        self.ewc = False

        u_pred = self.forward(x, params)
        per_sample_loss = (u_pred - u) ** 2
        #loss = per_sample_loss.mean()

        for loss in per_sample_loss:
            loss.backward()

            # Accumulate squared gradients
            for i, w in enumerate(weights):
                if w.grad is not None:
                    fisher_diag[i] += (w.grad ** 2) / len(dataset)
                    w.grad = None

        self.zero_grad()
        self.dwa_mode = old_dwa_mode
        self.ewc = old_ewc

        return fisher_diag.detach()
    
    def save(
        self,
        filepath: str # .pth
    ) -> None:
        """
        Save the model as a .pth in filepath, that contains model state and training hyperparameters.

        Parameters
        ----------
        filepath : str
            Filepath of the model file.

        Returns
        -------
        None
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),

            "device": self.device,

            "spatial_input": self.spatial_input,
            "temporal_input": self.temporal_input,
            "param_input": self.param_input,

            "input_units": self.input_units,
            "hidden_units": self.hidden_units,
            "activation_function_key": self.activation_function_key,

            "ff_encoding": self.ff_encoding,
            "dwa_mode": self.dwa_mode,
            "ewc": self.ewc,
            #"monitor_conflicts": self.monitor_conflicts,
            #"conflict_reference_task": self.conflict_reference_task,

            "train_tasks": [task.id for task in self.train_task_list],
            "eval_tasks": [task.id for task in self.eval_task_list]
        }

        if self.ff_encoding:
            checkpoint["fourier_features"] = self.fourier_features
            checkpoint["frequency_variance"] = self.frequency_variance
            # self.B is registered in model_state_dict
        if self.dwa_mode != "Off":
            checkpoint["dwa_alpha"] = self.dwa_alpha
            checkpoint["dwa_moving_avg_frequency"] = self.dwa_moving_avg_frequency
            checkpoint["dwa_warm_up"] = self.dwa_warm_up
        if self.ewc:
            checkpoint["ewc_weight"] = self.ewc_weight
            checkpoint["ewc_auto_weighting"] = self.ewc_auto_weighting
            checkpoint["ewc_warm_up"] = self.ewc_warm_up
            checkpoint["ewc_decay"] = self.ewc_decay

        # Save the checkpoint dictionary
        torch.save(checkpoint, filepath)

    def get_extra_state(self) -> dict:
        return {
            "device": self.device,
            "temporal_input": self.temporal_input,
            "spatial_input": self.spatial_input,
            "param_input": self.param_input,
            "ff_encoding": self.ff_encoding,
            "fourier_features": self.fourier_features,
            "input_units": self.input_units,
            "hidden_units": self.hidden_units,
            "activation_function_key": self.activation_function_key,
            "train_task_list": [task.state_dict() for task in self.train_task_list],
            "eval_task_list": [task.state_dict() for task in self.eval_task_list],
            "ewc": self.ewc,
            "ewc_auto_weighting": self.ewc_auto_weighting,
            "ewc_warm_up": self.ewc_warm_up,
            "ewc_decay": self.ewc_decay,
            "B": self.B,
            "ewc_frictioning_weights": self.ewc_frictioning_weights,
            "ewc_fisher_diagonal": self.ewc_fisher_diagonal,
            "dwa_mode": self.dwa_mode,
            "dwa_alpha": self.dwa_alpha,
            "dwa_moving_avg_frequency": self.dwa_moving_avg_frequency,
            "dwa_warm_up": self.dwa_warm_up,
            "dwa_moving_avg_count": self.dwa_moving_avg_count,
            "monitor_conflicts": self.monitor_conflicts,
            "conflict_reference_task": self.conflict_reference_task,
            "loss_container": self.loss_container.__class__.__name__,
            "optimizer": {self.optimizer.__class__.__name__: self.optimizer.state_dict()},
            "lr_scheduler": {self.lr_scheduler.__class__.__name__: self.lr_scheduler.state_dict()}
        }

    def set_extra_state(self, state: dict) -> None:
        self.device = state["device"]
        self.temporal_input = state["temporal_input"]
        self.spatial_input = state["spatial_input"]
        self.param_input = state["param_input"]
        self.ff_encoding = state["ff_encoding"]
        self.fourier_features = state["fourier_features"]
        self.input_units = state["input_units"]
        self.hidden_units = state["hidden_units"]
        self.activation_function_key = state["activation_function_key"]
        #self.train_task_list = 
        #self.eval_task_list =
        self.ewc = state["ewc"]
        self.ewc_auto_weighting = state["ewc_auto_weighting"]
        self.ewc_warm_up = state["ewc_warm_up"]
        self.ewc_decay = state["ewc_decay"]
        self.B = state["B"]
        self.ewc_frictioning_weights = state["ewc_frictioning_weights"]
        self.ewc_fisher_diagonal = state["ewc_fisher_diagonal"]
        self.dwa_mode = state["dwa_mode"]
        self.dwa_alpha = state["dwa_alpha"]
        self.dwa_moving_avg_frequency = state["dwa_moving_avg_frequency"]
        self.dwa_warm_up = state["dwa_warm_up"]
        #self.dwa_moving_avg_count
        self.monitor_conflicts = state["monitor_conflicts"]
        self.conflict_reference_task = state["conflict_reference_task"]
        #self.loss_container

        if self.optimizer is not None:
            optimizer_class = self.optimizer.__class__.__name__
            if optimizer_class == state["optimizer"].keys()[0]:
                self.optimizer.load_state_dict(state["optimizer"][optimizer_class])
            else:
                raise TypeError(f"Different optimizer class: {optimizer_class} != {state['optimizer'].keys()[0]}.")
            
        if self.lr_scheduler is not None:
            lr_scheduler_class = self.lr_scheduler.__class__.__name__
            if lr_scheduler_class == state["lr_scheduler"].keys()[0]:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"][lr_scheduler_class])
            else:
                raise TypeError(f"Different lr_scheduler class: {lr_scheduler_class} != {state['lr_scheduler'].keys()[0]}.")


    @staticmethod
    def load(
        filepath: str,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    ) -> Self:
        # Check the filepath existence
        if not os.path.exists(filepath):
            raise ValueError(f"File '{filepath}' not found.")

        # Load the state dictionary of the checkpoint model
        checkpoint = torch.load(filepath, weights_only=False)

        # Initialize an empty-state PINN model
        model = Pinn(optimizer=optimizer, lr_scheduler=lr_scheduler)

        # Load the checkpoint state into the model
        model.load_state_dict(checkpoint)

        # Move the model into the checkpoint device
        #model.to(checkpoint["device"])

    @staticmethod
    def load(filepath: str) -> Self:
        if not os.path.exists(filepath):
            raise ValueError(f"File '{filepath}' not found.")

        checkpoint = torch.load(filepath, weights_only=False)

        model = Pinn(
            device=checkpoint["device"],
            spatial_input=checkpoint["spatial_input"],
            temporal_input=checkpoint["temporal_input"],
            param_input=checkpoint["param_input"],
            hidden_units=checkpoint["hidden_units"],
            activation_str=checkpoint["activation_str"]
        ).to(checkpoint["device"])

        model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint["ff_encoding"]:
            model.set_ff(model.B)
        if checkpoint["dwa_mode"] != "Off":
            model.set_dwa(
                dwa_mode=checkpoint["dwa_mode"],
                dwa_alpha=checkpoint["dwa_alpha"],
                dwa_moving_avg_frequency=checkpoint["dwa_moving_avg_frequency"],
                dwa_warm_up=checkpoint["dwa_warm_up"]
            )
        if checkpoint["ewc"]:
            model.set_ewc(
                ewc_frictioning_weights=checkpoint["ewc_frictioning_weights"],
                ewc_fisher_diag=checkpoint["ewc_fisher_diag"],
                ewc_weight=checkpoint["ewc_weight"],
                ewc_auto_weighting=checkpoint["ewc_auto_weighting"],
                ewc_warm_up=checkpoint["ewc_warm_up"],
                ewc_decay=checkpoint["ewc_decay"]
            )

class Evaluator:
    """
    """

    def __init__(self):
        pass

    def __call__(self, model: Pinn, task: PhysicsTask, **kwargs) -> List[torch.Tensor]:
        if type(task) is AdvectionReactionDiffusionTask:
            return self._advection_reaction_diffusion_loss(model=model, task=task, **kwargs)
        elif type(task) is StationaryAllenCahnTask:
            return self._stationary_allen_cahn_loss(model=model, task=task, **kwargs)
        elif type(task) is NeumannBCTask:
            return self._neumann_bc_loss(model=model, task=task, **kwargs)
        elif type(task) is DirichletBCTask:
            return self._dirichlet_bc_loss(model=model, task=task, **kwargs)
        elif type(task) is ICTask:
            return self._ic_loss(model=model, task=task, **kwargs)
        elif type(task) is OutputTask:
            return self._output_loss(model=model, task=task, **kwargs)
        elif type(task) is DerivativeTask:
            return self._derivative_loss(model=model, task=task, **kwargs)
        elif type(task) is SpatialDerivativeTask:
            return self._spatial_derivative_loss(model=model, task=task, **kwargs)
        elif type(task) is TemporalDerivativeTask:
            return self._temporal_derivative_loss(model=model, task=task, **kwargs)
        elif type(task) is Derivative2Task:
            return self._derivative2_loss(model=model, task=task, **kwargs)
        elif type(task) is SpatialDerivative2Task:
            return self._spatial_derivative2_loss(model=model, task=task, **kwargs)
        elif type(task) is TemporalDerivative2Task:
            return self._temporal_derivative2_loss(model=model, task=task, **kwargs)
        
    @classmethod
    def _advection_reaction_diffusion_loss(
        cls,
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        task: AdvectionReactionDiffusionTask
    ) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted PDE residual field and the null field.
        """
        if len(task.param_keys) != len(pde_parameters):
            raise ValueError(f"The number of expected PDE parameters is {len(task.param_keys)}, while {len(pde_parameters)} PDE parameters are passed.")

        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        
        u = model.derivative(order=0, x=x, pde_params=input_params)
        du = model.derivative(order=1, x=x, pde_params=input_params)
        d2u = model.derivative(order=2, x=x, pde_params=input_params)
        x_ = x[:, 0]
        y = x[:, 1]
        t = x[:, 2]

        D_values = None
        vx_values = None
        vy_values = None
        source_values = None
        A_values = None
        B_values = None
        for key, value in zip(task.param_keys, pde_parameters):
            if key == "D":
                D_values = value
            elif key == "vx":
                vx_values = value
            elif key == "vy":
                vy_values = value
            elif key == "s":
                source_values = value
            elif key == "A":
                A_values = value
            elif key == "B":
                B_values = value

        if D_values is None:
            D_values = task.D

        if source_values is None:
            source_values = task.source_fn(x=x_, y=y, t=t)
        
        if A_values is not None:
            task.implicit_source_fn.set_A(A_values)
        if B_values is not None:
            task.implicit_source_fn.set_B(B_values)
        implicit_source_values = task.implicit_source_fn(u=u)

        if vx_values is None or vy_values is None:
            velocity_values = task.velocity_fn(x=x_, y=y, t=t)
            if vx_values is not None and vy_values is None:
                velocity_values = torch.stack((vx_values, velocity_values[:, 1]), dim=1)
            elif vx_values is None and vy_values is not None:
                velocity_values = torch.stack((velocity_values[:, 0], vy_values), dim=1)
        else:
            velocity_values = torch.stack((vx_values, vy_values))
        
        mse_loss = torch.nn.MSELoss(reduction='mean')
        
        residual_value = AdvectionReactionDiffusion.residual(
            du=du, 
            d2u=d2u, 
            velocity=velocity_values, 
            source=source_values, 
            implicit_source=implicit_source_values, 
            D=D_values
        )

        return mse_loss(residual_value, torch.zeros_like(residual_value))

    @classmethod
    def _stationary_allen_cahn_loss(
        cls, 
        x: torch.Tensor, 
        pde_parameters: torch.Tensor, 
        model: Pinn,
        task: StationaryAllenCahnTask
    ) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted PDE residual field and the null field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None

        mse_loss = torch.nn.MSELoss(reduction='mean')
    
        u = model.derivative(order=0, x=x, pde_params=input_params)
        d2u = model.derivative(order=2, x=x, pde_params=input_params)

        x_ = x[:, 0]
        y = x[:, 1]

        if task.lam_index is not None:
            lam_values = pde_parameters[:, task.lam_index]
        else:
            lam_values = task.lam
        if task.xi_indexes is not None:
            xi_values = pde_parameters[:, task.xi_indexes]
        else:
            xi_values = task.xi_vector

        residual_value = AllenCahn.residual(u=u, d2u=d2u, x=x_, y=y, lam=lam_values, force_params=xi_values)

        return mse_loss(residual_value, torch.zeros_like(residual_value))

    @classmethod
    def _neumann_bc_loss(
        cls, 
        x: torch.Tensor, 
        pde_parameters: torch.Tensor, 
        model: Pinn, 
        task: NeumannBCTask,
        du: torch.Tensor, 
        n: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted outward flux and the wanted outward flux.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(task.out_flux(du=du_pred, n=n), task.out_flux(du=du, n=n))

    @classmethod
    def _dirichlet_bc_loss(
        cls, 
        x: torch.Tensor, 
        pde_parameters: torch.Tensor, 
        model: Pinn, 
        task: DirichletBCTask,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted boundary value and the wanted boundary value.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        u_pred = model.forward(x=x, pde_params=input_params)
        return mse_loss(u_pred, u)

    @classmethod
    def _ic_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, task: ICTask, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted initial state field and the wanted initial state field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        u_pred = model.forward(x=x, pde_params=input_params)
        return mse_loss(u_pred, u)

    @classmethod
    def _output_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, task: OutputTask, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted u field and the wanted u field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        u_pred = model.forward(x=x, pde_params=input_params)
        return mse_loss(u_pred, u)

    @classmethod
    def _derivative_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, task: DerivativeTask, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du field and the wanted du field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(du_pred, du)

    @classmethod
    def _spatial_derivative_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, task: SpatialDerivativeTask, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du_xy field and the wanted du_xy field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(du_pred[:, :2], du[:, :2])

    @classmethod
    def _temporal_derivative_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, task: TemporalDerivativeTask, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du_t field and the wanted du_t field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(du_pred[:, 2:], du[:, 2:])

    @classmethod
    def _derivative2_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, task: Derivative2Task, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted d2u field and the wanted d2u field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
        return mse_loss(d2u_pred, d2u)

    @classmethod
    def _spatial_derivative2_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, task: SpatialDerivative2Task, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted [d2u_xx, d2u_yy, d2u_xy] field and the wanted [d2u_xx, d2u_yy, d2u_xy] field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
        return mse_loss(d2u_pred[:, :2], d2u[:, :2]) # mse_loss(d2u_pred[:, :2, :2], d2u[:, :2, :2])

    @classmethod
    def _temporal_derivative2_loss(cls, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted d2u_tt field and the wanted d2u_tt field.
        """
        if model.input_param_dict is not None:
            input_params = pde_parameters[:, list(model.input_param_dict.values())]
        else:
            input_params = None
        mse_loss = torch.nn.MSELoss(reduction='mean')
        d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
        return mse_loss(d2u_pred[:, 2], d2u[:, 2]) # mse_loss(d2u_pred[:, 2, 2], d2u[:, 2, 2])