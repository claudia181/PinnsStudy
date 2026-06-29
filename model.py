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
from physics_task import PhysicsTask
from phy_sys_dataset import PhySysDataset
from typing import Tuple, List

EWC_MODES = ["On", "Off"]
DWA_MODES = ["Off", "Std", "Norm1", "NormK"]
ACTIVATION = {
    "tanh": nn.Tanh
}

# PINN class definition ----------------------
class Pinn(torch.nn.Module):
    """
    Class representing a PINN model.
    """
    def __init__(
            self,
            device: str,
            hidden_units: list,
            activation_str: str = "tanh",
            temporal_input: int = 1,
            spatial_input: int = 2,
            param_input: int = 0,
            task_list: List[PhysicsTask] = [],
            eval_task_list: List[PhysicsTask] = [],
            ewc_mode: str = "Off",
            dwa_mode: str = "Off",
            dwa_alpha: float = 0.9,
            dwa_moving_avg_frequency: int = 1,
            dwa_warm_up: int = 0,
            monitor_conflicts: bool = False,
            conflict_reference_task: int = 0,
            ewc_weight: float = 1.0,
            ewc_auto_weighting: bool = False,
            ewc_warm_up: int = 0,
            ewc_decay: float = 1.0,
            ewc_model_weights: torch.Tensor = None,
            ewc_fisher_diag: torch.Tensor = None,
            *args,
            **kwargs
            ) -> None:
        """
        Initialize a PINN.

        Parameters
        ----------
        device : str
        hidden_units : list
            List of the hidden units of the model.
        activation_str : str
            String identifying the activation function of the network (according to ACTIVATION dict).
        temporal_input : int
            1 if the time is provided as input, 0 otw.
        spatial_input : int
            Number of spatial dimensions in input.
        param_input : list
            Number of parametrization dimensions in input.
        task_list : List[PhysicsTask]
            List of PhysicsTask objects (the training objectives).
        eval_task_list : List[PhysicsTask]
            List of PhysicsTask objects (the ones on which evaluation metrics are collected).
        ewc_mode : str
            Elastic weight consolidation mode ("On" or "Off").
        dwa_mode : str
            Dynamic weight adaptation mode ("Off", "Std", "Norm1", or "NormK").
        dwa_alpha : float
            Moving average weight for dynamic weight adaptation.
        dwa_moving_avg_frequency : int
            Moving average frequency for dynamic weight adaptation.
        dwa_warm_up : int
            Warm up steps for dynamic weight adaptation.
        monitor_conflicts : bool
            If True and DWA is active, updates each task.conflict attribute with the cosine similarity 
            btw the task gradient and the conflict_reference_task gradient.
        conflict_reference_task : int
            The objective wrt which the conflicts are computed.
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
            evaluated on some data.
        
        Returns
        -------
        None.
        """
        super().__init__(*args, **kwargs)
        
        # Set the parameters
        self.device = device

        self.temporal_input = temporal_input
        self.spatial_input = spatial_input
        self.param_input = param_input

        self.fourier_features = -1
        self.frequency_variance = None
        self.B = None

        self.input_units = spatial_input + temporal_input + param_input
        self.hidden_units = hidden_units

        self.task_list = task_list
        self.eval_task_list = eval_task_list
        
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
        self.conflict_reference_task = conflict_reference_task
        self.dwa_alpha = dwa_alpha
        self.dwa_moving_avg_frequency = dwa_moving_avg_frequency
        self.dwa_warm_up = dwa_warm_up
        self.dwa_moving_avg_count = 0
        self.activation_str = activation_str
        self.activation = ACTIVATION[activation_str]()

        # Build the network
        self._build_net()

        # Define the loss container: average over all elements (it always return a scalar in R)
        self.loss_container = nn.MSELoss(reduction='mean')

    def _build_net(self) -> None:
        """
        Build the NN and save it in the 'net' state variable.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        net_dict = OrderedDict()

        # First layer
        net_dict['lin0'] = nn.Linear(self.input_units, self.hidden_units[0])
        net_dict['act0'] = self.activation

        # Hidden layers
        for i in range(1, len(self.hidden_units)):
            net_dict[f'lin{i}'] = nn.Linear(in_features=self.hidden_units[i-1], out_features=self.hidden_units[i])
            net_dict[f'act{i}'] = self.activation

        # Last layer
        net_dict[f'lin{len(self.hidden_units)}'] = nn.Linear(self.hidden_units[-1], 1)
        
        # Glorot initialization
        #for i in range(0, len(hidden_units + 1)):
        #    init.xavier_normal_(net_dict[f"lin{i}"], gain=1.0)

        # Set the network architecture
        self.net = nn.Sequential(net_dict).to(self.device)

    def set_ff(self, B: torch.Tensor) -> None:
        """
        Set the network with the fourier feature encoding of the spatio-temporal input using B as frequency matrix.

        Parameters
        ----------
        B : torch.Tensor
            The the frequency matrix of shape (dim(spacetime), n_features).

        Returns
        -------
        None
        """
        # Set the encoding matrix B
        self.B = B

        # Update the model state
        self.fourier_features = B.shape[1]
        self.input_units = 2 * (self.spatial_input + self.temporal_input) * self.fourier_features + self.param_input

        # Rebuild the network
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
        None
        """
        # Sample the frequency matrix B
        torch.manual_seed(seed)
        self.B = torch.randn(2 * (self.spatial_input + self.temporal_input), n_fourier_features) * frequency_variance
        self.B = self.B.to(self.device)

        # Update the model state
        self.fourier_features = n_fourier_features
        self.frequency_variance = frequency_variance
        self.input_units = 2 * (self.spatial_input + self.temporal_input) * self.fourier_features + self.param_input

        # Rebuild the network
        self._build_net()

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
        torch.Tensor
            The output of the PINN.
        """
        if self.fourier_features != -1:
            x = 2 * torch.pi * x @ self.B
            x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        if pde_params is not None and pde_params != []:
            x = torch.cat([x, pde_params], dim=-1)
        return self.net(x).squeeze(-1)
    
    # Forward function for individual samples
    def _forward_single(self, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor:
        """
        Perform an inference step on a single input point.

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
        Concatenates all the learnable parameters (weights and biases) 
        of the model into a single 1D tensor and returns it.

        Parameters
        ----------
        None

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
            PDE parameters in input.

        Returns
        -------
        torch.Tensor
            The nth order derivative of the PINN at x.
        """
        # n = number of samples
        # d = number of input dimensions
        v = None
        if pde_params is None:
            if order == 0:
                v = self.forward(x) # shape (n)
                v = vmap(jacrev(self._forward_single))(x)[:, 0, :].squeeze() # shape (n, d)
                #.squeeze() removes dimensions of size 1: e.g. (1, 3, 1, 5) -> (3, 5)
            elif order == 2:
                v = vmap(hessian(self._forward_single))(x)[:, 0, :, :].squeeze() # shape (n, d, d)
            elif order == 4:
                f_4th = hessian(hessian(self._forward_single))
                v = vmap(f_4th)(x)[:, 0, :, :, :, :].squeeze() # shape (n, d, d, d, d)
        else:
            if order == 0:
                v = self.forward(x, pde_params) # shape (n)
            if order == 1:
                v = vmap(jacrev(self._forward_single, argnums=0), in_dims=(0, 0))(x, pde_params)[:, 0, :].squeeze() # shape (n, d)
            elif order == 2:
                v = vmap(hessian(self._forward_single, argnums=0), in_dims=(0, 0))(x, pde_params)[:, 0, :, :].squeeze() # shape (n, d, d)
            elif order == 4:
                f_4th = hessian(hessian(self._forward_single, argnums=0), argnums=0)
                v = vmap(f_4th, in_dims=(0, 0))(x, pde_params)[:, 0, :, :, :, :].squeeze() # shape (n, d, d, d, d)
        
        if order == 2 and v.dim() < 3:
            v = v.unsqueeze(0)
        return v
    
    def laplacian(self, order: int, x: torch.Tensor, pde_params: torch.Tensor = None) -> torch.Tensor: #TODO: test
        """
        Compute laplacian of the PINN wrt the spatio-temporal input at x.

        Parameters
        ----------
        order : int
            1 -> Laplacian, 2 -> Laplacian of Laplacian.
        x : torch.Tensor
            Spatio-temporal input.
        pde_params : torch.Tensor
            PDE parameters in input.

        Returns
        -------
        torch.Tensor
            The Laplacian of the PINN at x.
        """
        def lap(x_single):
            if pde_params is None:
                H = hessian(self._forward_single)(x_single)
            else:
                H = hessian(self._forward_single, argnums=0)(x_single, pde_params)

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
            create_graph=False, 
            #retain_graph=True, 
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
        grads_a: Tuple[torch.Tensor, ...],
        grads_b: Tuple[torch.Tensor, ...],
        norm_a: torch.Tensor, 
        norm_b: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the cosine similarity (the cosine of the angle) \n
        btw the gradients of the losses (gradient wrt the NN parameters/weights).

        Parameters
        ----------
        grads_a : tuple[torch.Tensor, ...]
            Value of the loss gradient vector 'a' on some input.
        grads_b : tuple[torch.Tensor, ...]
            Value of the loss gradient vector 'b' on some input.
        grads_a : tuple[torch.Tensor, ...]
            Value of the norm of the loss gradient vector 'a'.
        grads_b : tuple[torch.Tensor, ...]
            Value of the norm of the loss gradient vector 'b'.

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
    
    def _compute_ewc_loss_term(self) -> torch.Tensor:
        """
        Compute the EWC loss term on the given input.

        Parameters
        ----------
        None

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
        for task in self.task_list:
            task.grad, task.grad_norm = self._compute_grad_norm(task.loss_value)

        if self.monitor_conflicts:
            reference_task = self.task_list[self.conflict_reference_task]
            for task in self.task_list:
                task.conflict = self._compute_cos_sim(task.grad, reference_task.grad, task.grad_norm, reference_task.grad_norm)
        
        norm_sum = sum([task.grad_norm for task in self.task_list])
        
        for task in self.task_list:
            weight_new = norm_sum / task.grad_norm
            if weight_new.isnan() or weight_new.isinf():
                task.grad_norm = torch.tensor(0.0, device=self.device)
                task.weight = 0.0
                print("Weight reset.\n")
            else:
                task.weight = self.dwa_alpha * task.weight + (1 - self.dwa_alpha) * weight_new

        if self.dwa_mode != "Std":
            active_weights = [task.weight for task in self.task_list]
            weight_sum = sum(active_weights)
            # Normalize weights in such a way they sum to 1
            if self.dwa_mode == "Norm1":
                k = 1
            elif self.dwa_mode == "NormK":
                # Normalize weights in such a way they sum to |loss_terms|
                k = len(active_weights)
            else:
                raise ValueError(f"Unrecognized loss balancing mode '{self.dwa_mode}'.")
            
            for task in self.task_list:
                task.weight = task.weight * k / weight_sum

    def _update_grad_norms(self) -> None:
        """
        Update the gradient norm of each objective and the model state accordingly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for task in self.task_list:
            task.grad, task.grad_norm = self._compute_grad_norm(task.loss_value)

        if self.monitor_conflicts:
            reference_task = self.task_list[self.conflict_reference_task]
            for task in self.task_list:
                task.conflict = self._compute_cos_sim(task.grad, reference_task.grad, task.grad_norm, reference_task.grad_norm)
    
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
            reference_task = self.task_list[self.conflict_reference_task]
            for task in self.eval_task_list:
                task.conflict = self._compute_cos_sim(task.grad, reference_task.grad, task.grad_norm, reference_task.grad_norm)
    
    def train_loss(
            self,
            x_list: List[torch.Tensor], # spatio-temporal input, for each task
            input_param_list: List[torch.Tensor] = None, # physics parameters in input, for each task
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
        for i, task in enumerate(self.task_list):
            x = x_list[i]
            input_params = input_param_list[i]

            l_dict = {}
            for key in labels.keys():
                if key in task.loss_required_labels():
                    l_dict[key] = labels[key][i]
                else:
                    raise ValueError("Missing input parameters")
            
            task.loss_value = task.loss(x=x, input_params=input_params, model=self, **l_dict)

        if self.dwa_mode != "Off" and self.dwa_moving_avg_count % self.dwa_moving_avg_frequency == 0 and self.dwa_moving_avg_count >= self.dwa_warm_up:
            self._update_task_weights()
        else:
            self._update_grad_norms()
        self.dwa_moving_avg_count += 1

        weighted_loss = sum([task.weight * task.loss_value for task in self.task_list])

        if self.ewc_mode == "On":
            # Compute the loss term
            ewc_loss = torch.sum(self.ewc_fisher_diag * ((self.get_weights() - self.ewc_model_weights) ** 2))
            #print(f"ewc_fisher_diag: {torch.mean(self.ewc_fisher_diag)}\nweights: {torch.mean((self.get_weights() - self.ewc_model_weights) ** 2)}")

            if self.ewc_auto_weighting:
                if self.ewc_warm_up == 0:
                    self.ewc_weight = (weighted_loss / ewc_loss).item()
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
            input_param_list: List[torch.Tensor] = None,
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
            input_params = input_param_list[i]

            l_dict = {}
            for key in labels.keys():
                if key in task.loss_required_labels():
                    l_dict[key] = labels[key][i]
                else:
                    raise ValueError("Missing input parameters")
            
            task.loss_value = task.loss(x=x, input_params=input_params, model=self, **l_dict)
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
            dataset: TensorDataset, 
            spacetime_idx: int, 
            param_idx: int, 
            u_idx: int, 
            #du_idx: int, 
            #d2u_idx: int, 
            param_subidxs: List[int] = None
    ) -> torch.Tensor:
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
            x = z[spacetime_idx].to(self.device).float()
            u = z[u_idx].to(self.device).float()

            params = None
            if self.param_input != 0:
                if param_subidxs is None:
                    params = z[param_idx]
                else:
                    params = z[param_idx][:, param_subidxs]

            self.zero_grad()

            old_dwa_mode = self.dwa_mode
            self.dwa_mode = "Off"
            self.ewc_mode = "Off"

            u_pred = self.forward(x, params)
            per_sample_loss = (u_pred - u) ** 2
            loss = per_sample_loss.mean()
            #loss = self.loss_fn(
            #    x=x, pde_params=pde_param_values,
            #    u=u, du=du, d2u=d2u,
            #    ge_data=ge_info_dict
            #)
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

        self.dwa_mode = old_dwa_mode

        return fisher_diag_vector
    
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

            "fourier_features": self.fourier_features,
            "frequency_variance": self.frequency_variance,
            "B": self.B,

            "input_units": self.input_units,
            "hidden_units": self.hidden_units,
            "activation_str": self.activation_str,

            "dwa_mode": self.dwa_mode,
            "dwa_alpha": self.dwa_alpha,
            "dwa_moving_avg_frequency": self.dwa_moving_avg_frequency,
            "dwa_warm_up": self.dwa_warm_up,

            "ewc_mode": self.ewc_mode,
            "ewc_weight": self.ewc_weight,
            "ewc_auto_weighting": self.ewc_auto_weighting,
            "ewc_model_weights": self.ewc_model_weights,
            "ewc_fisher_diag": self.ewc_fisher_diag,

            "monitor_conflicts": self.monitor_conflicts,
            "conflict_reference_task": self.conflict_reference_task,

            "train_tasks": [task.id for task in self.task_list],
            "eval_tasks": [task.id for task in self.eval_task_list]
        }
        # Save the checkpoint dictionary
        torch.save(checkpoint, filepath)