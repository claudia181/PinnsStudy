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
    ):
        """
        Constructor.

        Parameters
        ----------
        task_id : str
            Task identifier.
        parameters : dict = None
            Dictionary of fixed parameters {..., "param_name": param_value, ...}.
        weight : float = None
            Current weight of the task (it weights the task loss term in the multi-objective loss function).
        """
        self.id = task_id
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
            return AdvectionReactionDiffusionTask().load_state(state)
        elif state["id"] == "StationaryAllenCahnGE":
            return StationaryAllenCahnTask().load_state(state)

# ===================================== NeumannBCTask =====================================
class NeumannBCTask(PhysicsTask):
    """
    Task for Neumann boundary conditions.
    """

    def __init__(self, parameters: dict = None, weight: float = None):
        """
        Parameters
        ----------
        parameters : dict
            - rectangular domain: {"top_flux": top_flux_value, "right_flux": right_flux_value, ...}
            - circular domain: {"flux": flux_value}
        weight : int
            Weight of the task in the loss.
        """
        super().__init__(
            task_id="NeumannBC",
            parameters=parameters,
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

# ===================================== DirichletBCTask =====================================
class DirichletBCTask(PhysicsTask):
    """
    Task for Dirichlet boundary conditions.
    """

    def __init__(self, parameters: dict = None, weight: float = None):
        """
        Parameters
        ----------
        parameters : dict
            - rectangular domain: {"top_u": top_u, "right_u": right_u, ...}
            - circular domain: {"u": u}
        weight : int
            Weight of the task in the loss.
        """
        super().__init__(
            task_id="DirichletBC",
            parameters=parameters,
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
        task = DirichletBCTask()
        task.load_state(self.state_dict())
        return task

# ===================================== ICTask =====================================
class ICTask(PhysicsTask):
    """
    Task for initial conditions.
    """

    def __init__(self, parameters: dict = None, weight: float = None):
        super().__init__(
            task_id="IC",
            parameters=parameters,
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
        parameters = {
            "xi_vector": self.xi_vector,
            "lam": self.lam
        }
        
        super().__init__(
            task_id="StationaryAllenCahnGE",
            parameters=parameters,
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
            #"lam": self.lam,
            "lam_index": self.lam_index,
            #"xi_vector": self.xi_vector,
            "xi_indexes": self.xi_indexes
        }
        return super().state_dict() | extra

    def load_state(self, state):
        super().load_state(state)

        self.lam = state["lam"]
        self.lam_index = state["lam_index"]
        self.xi_vector = state["xi_vector"]
        self.xi_indexes = state["xi_indexes"]