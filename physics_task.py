"""
physics_task.py
===========

This module implements physics tasks, which may relate to 
boundary conditions, initial conditions, output learning, 
derivative learning, physics-informed learning, etc.

A generic PhysycsTask class is defined and all the physics tasks subclass it.

Each specific physics task has to define
    - an associated loss function, which defines how to compute the corresponding loss term;
    - a loss_required_labels function, which returns the keys of the labels that are
      necessary to compute the loss function of the task.

Moreover any physics task has associated the following attributes:
    - `weight`, containing the weight for the loss term of the task in the multi-objective loss;
    - `loss_value`, optionally filled with the last loss value obtained for the task;
    - `grad_norm`, optionally filled with the last gradient norm of the task loss term;
    - `grad`, optionally filled with the last gradient of the task loss term;
    - `conflict`, optionally filled with the last cosine similarity between the gradient of the 
      task loss term and a reference gradient vector;
    - `parameters`, a dictionary whose elements identify the physical system parameters whose 
      values are fixed for all the training dataset entries (the ones whose value varies across 
      the dataset entries are supposed to be inputed to the model in order to perform predictions;
      hence, for physics-informed tasks, the varying physics parameters are expected to be part of 
      the task loss function inputs).
    - `id`, an identifier string for the task.
"""

from typing import Callable
import torch
from AdvectionReactionDiffusion.advection_reaction_diffusion import AdvectionReactionDiffusion
from StationaryAllenCahn.allen_cahn import AllenCahn
from model import Pinn
from typing import List, Self
from AdvectionReactionDiffusion.advection_velocity import Velocity
from AdvectionReactionDiffusion.reaction_source import Source

# ===================================== PhysicsTask =====================================
class PhysicsTask:

    def __init__(
            self,
            task_id: str = None,
            parameters: dict = None,
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
        weight : float = None
            Current weight of the task (it weights the task loss term in the multi-objective loss function).

        Returns
        -------
        _None_
        """
        self.id = task_id
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        self.weight = weight
        self.grad_norm = None
        self.grad = None
        self.conflict = None
        self.loss_value = None

    def loss(self) -> torch.Tensor | None:
        """
        Loss function giving the loss term for the task.
        """
        return None
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return []
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = PhysicsTask()
        task.load_state(self.state_dict())

    def state_dict(self) -> dict:
        """
        Function to get the state dictionary of the object.
        """
        return {
            "id": self.id,
            "parameters": self.parameters,
            "weight": self.weight,
            "grad": self.grad,
            "grad_norm": self.grad_norm,
            "conflict": self.conflict,
            "loss_value": self.loss_value
        }

    def load_state(self, state: dict) -> None:
        """
        Function to load a state into the object.

        Parameters
        ----------
        state : dict
            The dictionary of the state to load.
        """
        if self.id != state["id"]:
            raise TypeError(f"Physics task type mismatch: {self.id} != {state['id']}.")
        self.parameters = state["parameters"]
        self.weight = state["weight"]
        self.grad = state["grad"]
        self.grad_norm = state["grad_norm"]
        self.conflict = state["conflict"]
        self.loss_value = state["loss_value"]

    @staticmethod
    def load(state: dict, **kwargs) -> Self:
        if state["id"] == "NeumannBC":
            return NeumannBCTask().load_state(state)
        elif state["id"] == "DirichletBC":
            return DirichletBCTask().load_state(state)
        elif state["id"] == "IC":
            return ICTask().load_state(state)
        elif state["id"] == "Output":
            return OutputTask().load_state(state)
        elif state["id"] == "Derivative":
            return DerivativeTask().load_state(state)
        elif state["id"] == "Derivative_x":
            return SpatialDerivativeTask().load_state(state)
        elif state["id"] == "Derivative_t":
            return TemporalDerivativeTask().load_state(state)
        elif state["id"] == "Derivative2":
            return Derivative2Task().load_state(state)
        elif state["id"] == "Derivative2_x":
            return SpatialDerivative2Task().load_state(state)
        elif state["id"] == "Derivative2_t":
            return TemporalDerivative2Task().load_state(state)
        elif state["id"] == "AdvectionReactionDiffusionGE":
            return AdvectionReactionDiffusionTask(velocity=kwargs["velocity"]).load_state(state)
        elif state["id"] == "StationaryAllenCahnGE":
            return StationaryAllenCahnTask().load_state(state)

# ===================================== NeumannBCTask =====================================
class NeumannBCTask(PhysicsTask):
    """
    Task for Neumann boundary conditions.
    """

    def __init__(self, weight: float = None):
        super().__init__(
            task_id="NeumannBC",
            weight=weight
        )

    def out_flux(self, du: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Function returning the outward flux through the boundary of the spatial domain: 
        - component of the gradient field du along the outward normal field to the boundary surface.
        """
        outward_flux = (du[:, :2] * n).sum(dim=1)
        return outward_flux

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted outward flux and the wanted outward flux.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(self.out_flux(du=du_pred, n=n), self.out_flux(du=du, n=n))
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["du", "n"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = NeumannBCTask()
        task.load_state(self.state_dict())
        return task

# ===================================== DirichletBCTask =====================================
class DirichletBCTask(PhysicsTask):
    """
    Task for Dirichlet boundary conditions.
    """

    def __init__(self, weight: float = None):
        super().__init__(
            task_id="DirichletBC",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted boundary value and the wanted boundary value.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        u_pred = model.forward(x=x, pde_params=input_params)
        return mse_loss(u_pred, u)
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["u"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = DirichletBCTask()
        task.load_state(self.state_dict())
        return task

# ===================================== ICTask =====================================
class ICTask(PhysicsTask):
    """
    Task for initial conditions.
    """

    def __init__(self, weight: float = None):
        super().__init__(
            task_id="IC",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted initial state field and the wanted initial state field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        u_pred = model.forward(x=x, pde_params=input_params)
        return mse_loss(u_pred, u)
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["u"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = ICTask()
        task.load_state(self.state_dict())
        return task

# ===================================== OutputTask =====================================
class OutputTask(PhysicsTask):
    """
    Task for output learning.
    """

    def __init__(self, weight: float = None):
        
        super().__init__(
            task_id="Output",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted u field and the wanted u field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        u_pred = model.forward(x=x, pde_params=input_params)
        return mse_loss(u_pred, u)
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["u"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = OutputTask()
        task.load_state(self.state_dict())
        return task

# ===================================== DerivativeTask =====================================
class DerivativeTask(PhysicsTask):
    """
    Task for 1st derivative learning.
    """

    def __init__(self, weight: float = None):
        super().__init__(
            task_id="Derivative",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du field and the wanted du field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(du_pred, du)
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["du"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = DerivativeTask()
        task.load_state(self.state_dict())
        return task

# ===================================== SpatialDerivativeTask =====================================
class SpatialDerivativeTask(PhysicsTask):
    """
    Task for 1st spatial derivative learning.
    """

    def __init__(self, weight: float = None):
        super().__init__(
            task_id="Derivative_x",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du_xy field and the wanted du_xy field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(du_pred[:, :2], du[:, :2])
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["du"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = SpatialDerivativeTask()
        task.load_state(self.state_dict())
        return task

# ===================================== TemporalDerivativeTask =====================================
class TemporalDerivativeTask(PhysicsTask):
    """
    Task for 1st tempporal derivative learning.
    """

    def __init__(self, weight: float = None): 
        super().__init__(
            task_id="Derivative_t",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du_t field and the wanted du_t field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        du_pred = model.derivative(order=1, x=x, pde_params=input_params)
        return mse_loss(du_pred[:, 2:], du[:, 2:])
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["du"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = TemporalDerivativeTask()
        task.load_state(self.state_dict())
        return task

# ===================================== Derivative2Task =====================================
class Derivative2Task(PhysicsTask):
    """
    Task for 2nd derivative learning.
    """

    def __init__(self, weight: float = None):
        super().__init__(
            task_id="Derivative2",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted d2u field and the wanted d2u field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
        return mse_loss(d2u_pred, d2u)
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["d2u"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = Derivative2Task()
        task.load_state(self.state_dict())
        return task

# ===================================== SpatialDerivative2Task =====================================
class SpatialDerivative2Task(PhysicsTask):
    """
    Task for 2nd spatial derivative learning.
    """

    def __init__(self, weight: float = None):
        super().__init__(
            task_id="Derivative2_x",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted [d2u_xx, d2u_yy, d2u_xy] field and the wanted [d2u_xx, d2u_yy, d2u_xy] field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
        return mse_loss(d2u_pred[:, :2], d2u[:, :2]) # mse_loss(d2u_pred[:, :2, :2], d2u[:, :2, :2])
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["d2u"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = SpatialDerivative2Task()
        task.load_state(self.state_dict())
        return task

# ===================================== TemporalDerivative2Task =====================================
class TemporalDerivative2Task(PhysicsTask):
    """
    Task for 2nd temporal derivative learning.
    """

    def __init__(self, weight: float = None):
        
        super().__init__(
            task_id="Derivative2_t",
            weight=weight
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted d2u_tt field and the wanted d2u_tt field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
        d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
        return mse_loss(d2u_pred[:, 2], d2u[:, 2]) # mse_loss(d2u_pred[:, 2, 2], d2u[:, 2, 2])
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return ["d2u"]
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = TemporalDerivative2Task()
        task.load_state(self.state_dict())
        return task

# ===================================== AdvectionReactionDiffusionTask =====================================
class AdvectionReactionDiffusionTask(PhysicsTask):
    """
    Task for the advection-reaction-diffusion governing equation.
    """

    def __init__(self,
            velocity: Callable = None,
            source: Callable = None,
            implicit_source: Callable = None,
            D: float = None,
            weight: float = None
    ):
        self.velocity_fn = velocity
        self.source_fn = source
        self.implicit_source_fn = implicit_source
        self.D = D

        super().__init__(
            task_id="AdvectionReactionDiffusionGE",
            weight=weight
        )

    def residual(self,
            u: torch.Tensor = None,
            du: torch.Tensor = None,
            d2u: torch.Tensor = None,
            D_values: torch.Tensor = None,
            velocity_values: torch.Tensor = None,
            source_values: torch.Tensor = None,
            implicit_source_values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Left hand side. It calls self._lhs.
    
        Parameters
        ----------
        u : torch.Tensor = None
            Output labels.
        du : torch.Tensor = None
            1st derivative labels.
        d2u : torch.Tensor = None
            2nd derivative labels.
        input_parameters : dict = None
            Set of varying parameters, which are given in input to the model.
            
        Returns
        -------
        _torch.Tensor_
        """
        if velocity_values is None:
            velocity_values = self.velocity_fn(x=x, y=y, t=t)
        if source_values is None:
            source_values = self.source_fn(x=x, y=y, t=t)
        if implicit_source_values is None:
            implicit_source_values = self.implicit_source_fn(u=u)
        if D_values is None:
            D_values = self.D

        return AdvectionReactionDiffusion.residual(
            du=du, 
            d2u=d2u, 
            velocity=velocity_values, 
            source=source_values, 
            implicit_source=implicit_source_values, 
            D=D_values
        )

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted PDE residual field and the null field.
        """
        u = model.derivative(order=0, x=x, pde_params=input_params)
        du = model.derivative(order=1, x=x, pde_params=input_params)
        d2u = model.derivative(order=2, x=x, pde_params=input_params)
        x_ = x[:, 0]
        y = x[:, 1]
        t = x[:, 2]
        v = self.velocity(x_, y, t)
        mse_loss = torch.nn.MSELoss(reduction='mean')
        input_param_dict = dict(zip(model.pde_params_in_input, input_params.T))
        input_param_dict["velocity"] = v
        residual_value = self.residual(u=u, du=du, d2u=d2u, input_parameters=input_param_dict)
        return mse_loss(residual_value, torch.zeros_like(residual_value))
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return []

    def copy(self) -> Self:
        """
        Copy function.
        """
        task = AdvectionReactionDiffusionTask()
        task.load_state(self.state_dict())
        task.velocity = self.velocity
        return task

    def state_dict(self):
        extra = {}
        if isinstance(self.velocity_fn, Velocity):
            extra ["velocity_fn"] = self.velocity_fn.state_dict()
        if isinstance(self.source_fn, Source):
            extra ["source_fn"] = self.source_fn.state_dict()
        if isinstance(self.implicit_source_fn, Source):
            extra ["implicit_source_fn"] = self.implicit_source_fn.state_dict()
        return super().state_dict() | extra

    def load_state(self, state):
        super().load_state(state)
        if "velocity_fn" in state.keys():
            self.velocity_fn = Velocity.null_velocity()
            self.velocity_fn.load_state(state["velocity_fn"])
        else:
            self.velocity_fn = None
        if "source_fn" in state.keys():
            self.source_fn = Source.null_source()
            self.source_fn.load_state(state["source_fn"])
        else:
            self.source_fn = None
        if "implicit_source_fn" in state.keys():
            self.implicit_source_fn = Source.null_source()
            self.implicit_source_fn.load_state(state["implicit_source_fn"])
        else:
            self.implicit_source_fn = None

# ===================================== StationaryAllenCahnTask =====================================
class StationaryAllenCahnTask(PhysicsTask):
    """
    Task for the stationary Allen-Cahn governing equation.
    """

    def __init__(self, parameters: dict = None, weight: float = None):

        super().__init__(
            task_id="StationaryAllenCahnGE",
            parameters=parameters,
            weight=weight
        )

    def residual(self,
            u: torch.Tensor = None,
            d2u: torch.Tensor = None,
            input_parameters: dict = None
    ) -> torch.Tensor:
        """
        Left hand side. It calls self._lhs.
    
        Parameters
        ----------
        u : torch.Tensor = None
            Output labels.
        d2u : torch.Tensor = None
            2nd derivative labels.
        input_parameters : dict = None
            Set of varying parameters, which are given in input to the model.
            
        Returns
        -------
        _torch.Tensor_
        """
        if input_parameters is None:
            input_parameters = {}
        all_parameters = self.parameters | input_parameters
        return AllenCahn.residual(u=u, d2u=d2u, **all_parameters)

    def loss(self, x: torch.Tensor, input_params: torch.Tensor, model: Pinn) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted PDE residual field and the null field.
        """
        mse_loss = torch.nn.MSELoss(reduction='mean')
    
        u = model.derivative(order=0, x=x, pde_params=input_params)
        d2u = model.derivative(order=2, x=x, pde_params=input_params)
    
        input_param_dict = dict(zip(model.pde_params_in_input, input_params.T))
        residual_value = self.residual(u=u, d2u=d2u, input_parameters=input_param_dict)
        return mse_loss(residual_value, torch.zeros_like(residual_value))
    
    def loss_required_labels(self) -> List[str]:
        """
        Function returning the keys of the set of labels necessary to compute the loss term of the task.
        """
        return []
    
    def copy(self) -> Self:
        """
        Copy function.
        """
        task = StationaryAllenCahnTask()
        task.load_state(self.state_dict())
        return task