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
            input_param_indexes: List[int] = None,
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
        self.input_param_indexes = input_param_indexes
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
            "input_param_indexes": self.input_param_indexes,
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
        self.input_param_indexes = state["input_param_indexes"]
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
            return AdvectionReactionDiffusionTask().load_state(state)
        elif state["id"] == "StationaryAllenCahnGE":
            return StationaryAllenCahnTask().load_state(state)

# ===================================== NeumannBCTask =====================================
class NeumannBCTask(PhysicsTask):
    """
    Task for Neumann boundary conditions.
    """

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        super().__init__(
            task_id="NeumannBC",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def out_flux(self, du: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Function returning the outward flux through the boundary of the spatial domain: 
        - component of the gradient field du along the outward normal field to the boundary surface.
        """
        outward_flux = (du[:, :2] * n).sum(dim=1)
        return outward_flux

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, du: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted outward flux and the wanted outward flux.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        super().__init__(
            task_id="DirichletBC",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted boundary value and the wanted boundary value.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        super().__init__(
            task_id="IC",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted initial state field and the wanted initial state field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        
        super().__init__(
            task_id="Output",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted u field and the wanted u field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        super().__init__(
            task_id="Derivative",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du field and the wanted du field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        super().__init__(
            task_id="Derivative_x",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du_xy field and the wanted du_xy field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None,  weight: float = None): 
        super().__init__(
            task_id="Derivative_t",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted du_t field and the wanted du_t field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        super().__init__(
            task_id="Derivative2",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted d2u field and the wanted d2u field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        super().__init__(
            task_id="Derivative2_x",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted [d2u_xx, d2u_yy, d2u_xy] field and the wanted [d2u_xx, d2u_yy, d2u_xy] field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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

    def __init__(self, input_param_indexes: List[int] = None, weight: float = None):
        
        super().__init__(
            task_id="Derivative2_t",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted d2u_tt field and the wanted d2u_tt field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None
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
            param_keys: List[str] = [],
            input_param_indexes: List[int] = None,
            velocity: Velocity = None,
            source: Source = None,
            implicit_source: Source = None,
            D: float = None,
            weight: float = None
    ):
        self.param_keys = param_keys
        self.input_param_indexes = input_param_indexes

        if velocity is None:
            self.velocity_fn = Velocity.null_velocity()
        else:
            self.velocity_fn = velocity

        if source is None:
            self.source_fn = Source.null_source()
        else:
            self.source_fn = source

        if implicit_source is None:
            self.implicit_source_fn = Source.null_source()
        else:
            self.implicit_source_fn = implicit_source

        if D is None:
            self.D = 0.0
        else:
            self.D = D

        super().__init__(
            task_id="AdvectionReactionDiffusionGE",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(
            self,
            x: torch.Tensor,
            pde_parameters: torch.Tensor,
            model: Pinn
    ) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted PDE residual field and the null field.
        """
        if len(self.param_keys) != len(pde_parameters):
            raise ValueError(f"The number of expected PDE parameters is {len(self.param_keys)}, while {len(pde_parameters)} PDE parameters are passed.")

        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
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
        for key, value in zip(self.param_keys, pde_parameters):
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
            D_values = self.D

        if source_values is None:
            source_values = self.source_fn(x=x_, y=y, t=t)
        
        if A_values is not None:
            self.implicit_source_fn.set_A(A_values)
        if B_values is not None:
            self.implicit_source_fn.set_B(B_values)
        implicit_source_values = self.implicit_source_fn(u=u)

        if vx_values is None or vy_values is None:
            velocity_values = self.velocity_fn(x=x_, y=y, t=t)
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
        extra = {
            "D": self.D,
            "velocity_fn": self.velocity_fn.state_dict(),
            "source_fn": self.source_fn.state_dict(),
            "implicit_source_fn": self.implicit_source_fn.state_dict()
        }
        return super().state_dict() | extra

    def load_state(self, state):
        super().load_state(state)

        self.velocity_fn = Velocity.null_velocity()
        self.source_fn = Source.null_source()
        self.implicit_source_fn = Source.null_source()
        self.D = 0.0

        if "D" in state.keys():
            self.D = state["D"]
        if "velocity_fn" in state.keys():
            self.velocity_fn.load_state(state["velocity_fn"])
        if "source_fn" in state.keys():
            self.source_fn.load_state(state["source_fn"])
        if "implicit_source_fn" in state.keys():
            self.implicit_source_fn.load_state(state["implicit_source_fn"])

# ===================================== StationaryAllenCahnTask =====================================
class StationaryAllenCahnTask(PhysicsTask):
    """
    Task for the stationary Allen-Cahn governing equation.
    """

    def __init__(self, lam: torch.Tensor = None, xi_vector: torch.Tensor = None, lam_index: int = None, xi_indexes: List[int] = None, weight: float = None):
        self.lam = lam
        self.xi_vector = xi_vector
        self.lam_index = lam_index
        self.xi_indexes = xi_indexes
        input_param_indexes = []
        if lam_index is not None:
            input_param_indexes.append(lam_index)
        if xi_indexes is not None:
            input_param_indexes += xi_indexes#[i+1 for i in range(len(xi_vector))]
        if input_param_indexes == []:
            input_param_indexes = None
        super().__init__(
            task_id="StationaryAllenCahnGE",
            input_param_indexes=input_param_indexes,
            weight=weight
        )

    def loss(self, x: torch.Tensor, pde_parameters: torch.Tensor, model: Pinn) -> torch.Tensor:
        """
        Loss function giving the loss term of the task:
        - MSE btw the predicted PDE residual field and the null field.
        """
        if self.input_param_indexes is not None:
            input_params = pde_parameters[:, model.input_param_indexes]
        else:
            input_params = None

        mse_loss = torch.nn.MSELoss(reduction='mean')
    
        u = model.derivative(order=0, x=x, pde_params=input_params)
        d2u = model.derivative(order=2, x=x, pde_params=input_params)

        x_ = x[:, 0]
        y = x[:, 1]

        if self.lam_index is not None:
            lam_values = pde_parameters[:, self.lam_index]
        else:
            lam_values = self.lam
        if self.xi_indexes is not None:
            xi_values = pde_parameters[:, self.xi_indexes]
        else:
            xi_values = self.xi_vector

        residual_value = AllenCahn.residual(u=u, d2u=d2u, x=x_, y=y, lam=lam_values, force_params=xi_values)

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

    def state_dict(self) -> dict:
        extra = {
            "lam": self.lam,
            "lam_index": self.lam_index,
            "xi_vector": self.xi_vector,
            "xi_indexes": self.xi_indexes
        }
        return super().state_dict() | extra

    def load_state(self, state):
        super().load_state(state)

        self.lam = state["lam"]
        self.lam_index = state["lam_index"]
        self.xi_vector = state["xi_vector"]
        self.xi_indexes = state["xi_indexes"]