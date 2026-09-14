"""
physics_task.py
===========

This module implements physics tasks, which may relate to 
boundary conditions, initial conditions, output learning, 
derivative learning, physics-informed learning, etc.

A generic PhysycsTask class is defined and all the physics tasks subclass it.

Each specific physics task has to define
    - a loss_required_labels function, which returns the keys of the labels that are
      necessary to compute the loss function of the task.

Moreover any physics task has associated the following attributes:
    - `weight`, containing the weight for the loss term of the task in the multi-objective loss;
    - `loss_value`, optionally filled with the last loss value obtained for the task;
    - `grad_norm`, optionally filled with the last gradient norm of the task loss term;
    - `grad`, optionally filled with the last gradient of the task loss term;
    - `conflict`, optionally filled with the last cosine similarity between the gradient of the 
      task loss term and a reference gradient vector;
    - `id`, an identifier string for the task.
"""

import torch
from typing import List, Self
from AdvectionReactionDiffusion.advection_velocity import Velocity
from AdvectionReactionDiffusion.reaction_source import Source
from typing import Callable

# ===================================== PhysicsTask =====================================
class PhysicsTask:

    def __init__(
            self,
            task_id: str = None,
            weight: float = None,
            loss_fn: Callable = None
    ):
        """
        Constructor.

        Parameters
        ----------
        task_id : str
            Task identifier.
        weight : float = None
            Current weight of the task (it weights the task loss term in the multi-objective loss function).
        """
        self.id = task_id
        self.weight = weight
        self.loss_fn = loss_fn
        self.grad_norm = None
        self.grad = None
        self.conflict = None
        self.loss_value = None
    
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
        return task

    def state_dict(self) -> dict:
        """
        Function to get the state dictionary of the object.
        """
        return {
            "id": self.id,
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
        self.weight = state["weight"]
        self.grad = state["grad"]
        self.grad_norm = state["grad_norm"]
        self.conflict = state["conflict"]
        self.loss_value = state["loss_value"]

    def __str__(self) -> str:
        string = f"*** General task attributes: ***\n"
        string += f"- ID: {self.id}\n"
        string += f"- Weight: {self.weight}\n"
        string += f"- Gradient: {self.grad}\n"
        string += f"- Gradient norm: {self.grad_norm}\n"
        string += f"- Conflict: {self.conflict}\n"
        string += f"- Loss: {self.loss_value}\n"
        return string
    
    def __repr__(self) -> str:
        return self.__str__()

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

    def __init__(
            self,
            top_flux: float = None,
            right_flux: float = None,
            bottom_flux: float = None,
            left_flux: float = None, 
            circumference_flux: float = None,
            weight: float = None
    ):
        """
        Parameters
        ----------
        top_flux : float
            Flux value for the top side of the rectangular domain.
        right_flux : float
            Flux value for the right side of the rectangular domain.
        bottom_flux : float
            Flux value for the bottom side of the rectangular domain.
        left_flux : float
            Flux value for the left side of the rectangular domain. 
        circumference_flux : float
            Flux value for the circumference of the circular domain.
        weight : float
            Weight of the task in the loss.
        """
        self.top_flux = top_flux
        self.right_flux = right_flux
        self.bottom_flux = bottom_flux
        self.left_flux = left_flux
        self.circumference_flux = circumference_flux
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

    def state_dict(self) -> dict:
        extra = {
            "top_flux": self.top_flux,
            "right_flux": self.right_flux,
            "bottom_flux": self.bottom_flux,
            "left_flux": self.left_flux,
            "circumference_flux": self.circumference_flux
        }
        return super().state_dict() | extra
    
    def load_state(self, state):
        super().load_state(state)
        self.top_flux = state["top_flux"]
        self.right_flux = state["right_flux"]
        self.bottom_flux = state["bottom_flux"]
        self.left_flux = state["left_flux"]
        self.circumference_flux = state["circumference_flux"]

    def __str__(self) -> str:
        string = super().__str__()
        string += "\n*** Task-specific attributes: ***"
        string += (f"\n- top_flux: {self.top_flux}")
        string += (f"\n- right_flux: {self.right_flux}")
        string += (f"\n- bottom_flux: {self.bottom_flux}")
        string += (f"\n- left_flux: {self.left_flux}")
        string += (f"\n- circumference_flux: {self.circumference_flux}")
        return string
    
    def __repr__(self) -> str:
        return self.__str__()

# ===================================== DirichletBCTask =====================================
class DirichletBCTask(PhysicsTask):
    """
    Task for Dirichlet boundary conditions.
    """

    def __init__(
            self, 
            top_u: float | torch.Tensor = None, 
            right_u: float | torch.Tensor = None,
            bottom_u: float | torch.Tensor = None,
            left_u: float | torch.Tensor = None,
            circumference_u: float | torch.Tensor = None,
            weight: float = None
    ):
        """
        Parameters
        ----------
        top_u: float | torch.Tensor
            u value(s) for the top side of the rectangular domain.
        right_u: float | torch.Tensor
            u value(s) for the right side of the rectangular domain.
        bottom_u: float | torch.Tensor
            u value(s) for the bottom side of the rectangular domain.
        left_u: float | torch.Tensor
            u value(s) for the left side of the rectangular domain.
        circumference_u: float | torch.Tensor
            u value(s) for the circumference of the circular domain.
        weight : float
            Weight of the task in the loss.
        """
        super().__init__(
            task_id="DirichletBC",
            weight=weight
        )
        self.top_u = top_u
        self.right_u = right_u
        self.bottom_u = bottom_u
        self.left_u = left_u
        self.circumference_u = circumference_u
    
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

    def state_dict(self) -> dict:
        extra = {
            "top_u": self.top_u,
            "right_u": self.right_u,
            "bottom_u": self.bottom_u,
            "left_u": self.left_u,
            "circumference_u": self.circumference_u
        }
        return super().state_dict() | extra
    
    def load_state(self, state):
        super().load_state(state)
        self.top_u = state["top_u"]
        self.right_u = state["right_u"]
        self.bottom_u = state["bottom_u"]
        self.left_u = state["left_u"]
        self.circumference_u = state["circumference_u"]

    def __str__(self) -> str:
        string = super().__str__()
        string += "\n*** Task-specific attributes: ***"
        string += (f"\n- top_u: {self.top_u}")
        string += (f"\n- right_u: {self.right_u}")
        string += (f"\n- bottom_u: {self.bottom_u}")
        string += (f"\n- left_u: {self.left_u}")
        string += (f"\n- circumference_u: {self.circumference_u}")
        return string
    
    def __repr__(self) -> str:
        return self.__str__()

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
            velocity: Velocity = None,
            source: Source = None,
            implicit_source: Source = None,
            D: float = None,
            weight: float = None
    ):
        """
        Parameters
        ----------
        param_keys : List[str]
            List of string identifiers of the pde parameters.
        velocity : Velocity
            Callable velocity object for the advection process of the system.
        source : Source
            Callable explicit source (depending uniquely on spatio-temporal coordinates) object for the reaction process of the system.
        implicit_source : Source
            Callable implicit source (depending also on the u-field) object for the reaction process of the system.
        D : float
            Diffusion coefficient for the diffusion process of the system.
        weight : float
            Weight of the task in the loss.
        """
        self.param_keys = param_keys

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
            weight=weight
        )
    
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

    def state_dict(self) -> dict:
        extra = {
            "D": self.D,
            "velocity_fn": self.velocity_fn.state_dict(),
            "source_fn": self.source_fn.state_dict(),
            "implicit_source_fn": self.implicit_source_fn.state_dict(),
            "param_keys": self.param_keys
        }
        return super().state_dict() | extra

    def __str__(self) -> str:
        string = super().__str__()
        string += "\n*** Task-specific attributes: ***"
        string += (f"\n- D: {self.D}\n\n")
        string += f"{self.velocity_fn.__str__()}\n"
        string += f"{self.source_fn.__str__()}\n"
        string += f"{self.implicit_source_fn.__str__()}\n"
        string += f"- param_keys: {self.param_keys}\n"
        return string

    def __repr__(self) -> str:
        return self.__str__()

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
        if "param_keys" in state.keys():
            self.param_keys = state["param_keys"]

# ===================================== StationaryAllenCahnTask =====================================
class StationaryAllenCahnTask(PhysicsTask):
    """
    Task for the stationary Allen-Cahn governing equation.
    """

    def __init__(
            self, 
            lam: torch.Tensor = None, 
            xi_vector: torch.Tensor = None, 
            lam_index: int = None, 
            xi_indexes: List[int] = None, 
            weight: float = None
    ):
        """
        Parameters
        ----------
        lam : torch.Tensor
            Thickness parameter value for the stationary Allen-Cahn system.
        xi_vector : torch.Tensor
            Force parameters' values for the stationary Allen-Cahn system.
        lam_index : int
            Thickness parameter index in the parameters column.
        xi_indexes : List[int]
            Force parameters' indexes in the parameters column.
        weight : float
            Weight of the task in the loss.
        """
        self.lam = lam
        self.xi_vector = xi_vector
        self.lam_index = lam_index
        self.xi_indexes = xi_indexes
        
        super().__init__(
            task_id="StationaryAllenCahnGE",
            weight=weight
        )
    
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

    def __str__(self) -> str:
        string = super().__str__()
        string += "\n*** Task-specific attributes: ***\n"
        string += (f"- lam: {self.lam}\n")
        string += (f"- lam_index: {self.lam_index}\n")
        string += (f"- xi_vector: {self.xi_vector}\n")
        string += (f"- xi_indexes: {self.xi_indexes}\n")
        return string
    
    def __repr__(self) -> str:
        return self.__str__()