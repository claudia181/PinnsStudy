"""
physics_task.py
===========

This module implements physics tasks, which may relate to 
boundary conditions, initial conditions, output learning, 
derivative learning, physics-informed learning, etc.

A generic PhysycsTask class is defined and all the physics tasks subclass it.

Each specific physics task has to define
    - an associated loss function, which defines how to compute the relative loss term;
    - a loss_required_labels function, which returns the keys of the labels that are
      necessary to compute the loss function of the task.

Moreover any physics task has associated the following attributes:
    - weight, containing the weight for the loss term of the task in the multi-objective loss;
    - loss_value, optionally filled with the last loss value obtained for the task;
    - grad_norm, optionally filled with the last gradient norm of the task loss term;
    - grad, optionally filled with the last gradient of the task loss term;
    - conflict, optionally filled with the last cosine similarity between the gradient of the 
      task loss term and a reference gradient vector;
    - parameters, a dictionary whose elements identify the physical system parameters whose 
      values are fixed for all the training dataset entries (the ones whose value varies across 
      the dataset entries are supposed to be inputed to the model in order to perform predictions;
      hence, for physics-informed tasks, the varying physics parameters are expected to be part of 
      the task loss function inputs).
    - id, an identifier string for the task.
"""

from typing import Callable
import torch
from advection_reaction_diffusion import AdvectionReactionDiffusion
from allen_cahn import AllenCahn
from model import Pinn
from typing import List, Self

# ===================================== PhysicsTask class =====================================
class PhysicsTask:

    def __init__(
            self,
            task_id: str = None,
            parameters: dict = None,
            lhs: Callable[..., torch.Tensor] = None,
            loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            weight: float = None
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        task_id : str
            Task identifier.
        parameters : dict = None
            Set of fixed parameters.
        lhs : Callable[..., torch.Tensor] = None
            Function that computes the left hand side of the equation (in case of physics-informed task).
        loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
            Function that computes the loss term of the task.
        weight : float = None
            Current weight of the task (it weights the loss term of the task in the multi-objective loss function).

        Returns
        -------
        None
        """
        self.id = task_id
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        self._lhs = lhs
        self.loss = loss
        self.weight = weight
        self.grad_norm = None
        self.grad = None
        self.conflict = None
        self.loss_value = None
    
    def lhs(self,
            n: torch.Tensor = None,
            u: torch.Tensor = None,
            du: torch.Tensor = None,
            d2u: torch.Tensor = None,
            input_parameters: dict = None
    ) -> torch.Tensor:
        """
        Left hand side. It calls self._lhs.

        Parameters
        ----------
        n : torch.Tensor = None
            Outward normal vectors to the boundary.
        u : torch.Tensor = None
            Output labels.
        du : torch.Tensor = None
            1st derivative labels (vectors).
        d2u : torch.Tensor = None
            2nd derivative labels (matrices).
        input_parameters : dict = None
            Set of varying parameters, which are given in input to the model.
        
        Returns
        -------
        torch.Tensor
        """
        if input_parameters is None:
            input_parameters = {}
        all_parameters = self.parameters | input_parameters
        return self._lhs(u=u, du=du, d2u=d2u, n=n, **all_parameters)
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
        """
        return []
    
    def copy(self) -> Self:
        return PhysicsTask(
            task_id = self.id,
            parameters = self.parameters,
            lhs = self._lhs,
            loss = self.loss,
            weight = self.weight
        )


class NeumannBCTask(PhysicsTask):
    """
    Task for Neumann boundary condiitons.
    """

    def __init__(self, weight: float = None):

        def lhs(du: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            outward_flux = (du[:, :2] * n).sum(dim=1)
            return outward_flux

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            du_pred = model.derivative(order=1, x=x, pde_params=input_params)
            return mse_loss(lhs(du=du_pred, n=n), lhs(du=du, n=n))
        
        super().__init__(
            task_id="NeumannBC",
            lhs=lhs,
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du", "n"]
    
    def copy(self) -> Self:
        return NeumannBCTask(weight=self.weight)

class DirichletBCTask(PhysicsTask):
    """
    Task for Dirichlet boundary condiitons.
    """

    def __init__(self, weight: float = None):
        def lhs(u: torch.Tensor) -> torch.Tensor:
            return u

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            u_pred = model.forward(x=x, pde_params=input_params)
            return mse_loss(u_pred, u)
        
        super().__init__(
            task_id="DirichletBC",
            lhs=lhs,
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["u"]
    
    def copy(self) -> Self:
        return DirichletBCTask(weight=self.weight)
    
class ICTask(PhysicsTask):
    """
    Task for initial condiitons.
    """

    def __init__(self, weight: float = None):

        def lhs(u: torch.Tensor) -> torch.Tensor:
            return u

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            u_pred = model.forward(x=x, pde_params=input_params)
            return mse_loss(u_pred, u)
        
        super().__init__(
            task_id="IC",
            lhs=lhs,
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["u"]
    
    def copy(self) -> Self:
        return ICTask(weight=self.weight)
    
class OutputTask(PhysicsTask):
    """
    Task for output learning.
    """

    def __init__(self, weight: float = None):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            u_pred = model.forward(x=x, pde_params=input_params)
            return mse_loss(u_pred, u)
        
        super().__init__(
            task_id="Output",
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["u"]
    
    def copy(self) -> Self:
        return OutputTask(weight=self.weight)
    
class DerivativeTask(PhysicsTask):
    """
    Task for 1st derivative learning.
    """

    def __init__(self, weight: float = None):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            du_pred = model.derivative(order=1, x=x, pde_params=input_params)
            return mse_loss(du_pred, du)
        
        super().__init__(
            task_id="Derivative",
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du"]
    
    def copy(self) -> Self:
        return DerivativeTask(weight=self.weight)
    
class SpatialDerivativeTask(PhysicsTask):
    """
    Task for 1st spatial derivative learning.
    """

    def __init__(self, weight: float = None):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            du_pred = model.derivative(order=1, x=x, pde_params=input_params)
            return mse_loss(du_pred[:, :2], du[:, :2])
        
        super().__init__(
            task_id="Derivative_x",
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du"]
    
    def copy(self) -> Self:
        return SpatialDerivativeTask(weight=self.weight)
    
class TemporalDerivativeTask(PhysicsTask):
    """
    Task for 1st tempporal derivative learning.
    """

    def __init__(self, weight: float = None):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            du_pred = model.derivative(order=1, x=x, pde_params=input_params)
            return mse_loss(du_pred[:, 2:], du[:, 2:])
        
        super().__init__(
            task_id="Derivative_t",
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du"]
    
    def copy(self) -> Self:
        return TemporalDerivativeTask(weight=self.weight)

class Derivative2Task(PhysicsTask):
    """
    Task for 2nd derivative learning.
    """

    def __init__(self, weight: float = None):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
            return mse_loss(d2u_pred, d2u)
        
        super().__init__(
            task_id="Derivative2",
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["d2u"]
    
    def copy(self) -> Self:
        return Derivative2Task(weight=self.weight)

class SpatialDerivative2Task(PhysicsTask):
    """
    Task for 2nd spatial derivative learning.
    """

    def __init__(self, weight: float = None):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
            return mse_loss(d2u_pred[:, :2, :2], d2u[:, :2, :2])
        
        super().__init__(
            task_id="Derivative2_x",
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["d2u"]
    
    def copy(self) -> Self:
        return SpatialDerivative2Task(weight=self.weight)

class TemporalDerivative2Task(PhysicsTask):
    """
    Task for 2nd temporal derivative learning.
    """

    def __init__(self, weight: float = None):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
            return mse_loss(d2u_pred[:, 2, 2], d2u[:, 2, 2])
        
        super().__init__(
            task_id="Derivative2_t",
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["d2u"]
    
    def copy(self) -> Self:
        return TemporalDerivative2Task(weight=self.weight)

class AdvectionReactionDiffusionTask(PhysicsTask):
    """
    Task for the advection-reaction-diffusion governing equation.
    """

    def __init__(self, 
            parameters: dict,
            velocity: Callable,
            weight: float = None
    ):
        self.parameters = parameters
        self.velocity = velocity

        def lhs(
                u: torch.Tensor, 
                du: torch.Tensor, 
                d2u: torch.Tensor,
                v: torch.Tensor,
                input_parameters: dict = None
            ) -> torch.Tensor:
            if input_parameters is None:
                input_parameters = {}
            all_parameters = self.parameters | input_parameters
            all_parameters["v"] = v
            return AdvectionReactionDiffusion.residual(u=u, du=du, d2u=d2u, **all_parameters)

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn) -> torch.Tensor:

            u = model.derivative(order=0, x=x, pde_params=input_params)
            du = model.derivative(order=1, x=x, pde_params=input_params)
            d2u = model.derivative(order=2, x=x, pde_params=input_params)

            x_ = x[:, 0]
            y = x[:, 1]
            t = x[:, 2]
            v = self.velocity(x_, y, t)

            mse_loss = torch.nn.MSELoss(reduction='mean')
            input_param_dict = dict(zip(model.pde_params_in_input, input_params.T))
            lhs_value = lhs(u=u, du=du, d2u=d2u, v=v, input_parameters=input_param_dict)
            return mse_loss(lhs_value, torch.zeros_like(lhs_value))
        
        super().__init__(
            task_id="AdvectionReactionDiffusionGE",
            parameters=parameters,
            lhs=lhs,
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return []

    def copy(self) -> Self:
        return AdvectionReactionDiffusionTask(
            parameters = self.parameters,
            velocity = self.velocity,
            weight = self.weight
        )

class StationaryAllenCahnTask(PhysicsTask):
    """
    Task for the stationary Allen-Cahn governing equation.
    """

    def __init__(self, parameters: dict, weight: float = None):

        def lhs(
                u: torch.Tensor,
                d2u: torch.Tensor,
                input_parameters: dict
        ) -> torch.Tensor:
            if input_parameters is None:
                input_parameters = {}
            all_parameters = self.parameters | input_parameters
            return AllenCahn.residual(u=u, d2u=d2u, **all_parameters)

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')

            u = model.derivative(order=0, x=x, pde_params=input_params)
            d2u = model.derivative(order=2, x=x, pde_params=input_params)

            input_param_dict = dict(zip(model.pde_params_in_input, input_params.T))
            lhs_value = lhs(u=u, d2u=d2u, input_parameters=input_param_dict)
            return mse_loss(lhs_value, torch.zeros_like(lhs_value))
        
        super().__init__(
            task_id="StationaryAllenCahnGE",
            parameters=parameters,
            lhs=lhs,
            loss=loss,
            weight=weight
        )
    
    def loss_required_labels(self) -> List[str]:
        return []
    
    def copy(self) -> Self:
        return StationaryAllenCahnTask(
            parameters = self.parameters,
            weight = self.weight
        )