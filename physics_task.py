from typing import Callable
import torch
from advection_reaction_diffusion import AdvectionReactionDiffusion
from allen_cahn import AllenCahn
from model2 import Pinn
from typing import List

TASKS = ["PDE", "Output", "Derivative", "Derivative_x", "Derivative_t", "Hessian", "Hessian_x", "Hessian_t"]

# ===================================== PhysicsTask class =====================================
class PhysicsTask:

    def __init__(
            self,
            task_id: str = None,
            parameters: dict = None,
            lhs: Callable[..., torch.Tensor] = None,
            loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            weight: float = None,
            device: str = "cpu"
    ):
        """
        Constructor.

        Parameters
        ----------
        
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
        self.device = device
    
    def lhs(self,
            n: torch.Tensor = None,
            u: torch.Tensor = None,
            du: torch.Tensor = None,
            d2u: torch.Tensor = None,
            lap: torch.Tensor = None,
            lap2: torch.Tensor = None,
            input_parameters: dict = None
    ) -> torch.Tensor:
        if input_parameters is None:
            input_parameters = {}
        all_parameters = self.parameters | input_parameters
        return self._lhs(u=u, du=du, d2u=d2u, lap=lap, lap2=lap2, n=n, **all_parameters)
    
    def loss_required_labels(self) -> List[str]:
        return []


class NeumannBCTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

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
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du", "n"]

class DirichletBCTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):
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
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["u"]
    
class ICTask(PhysicsTask):

    def __init__(self, indexes: List[int] = [], weight: float = None, device: str = "cpu"):

        self.indexes = indexes

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
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["u"]
    
class OutputTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            u_pred = model.forward(x=x, pde_params=input_params)
            return mse_loss(u_pred, u)
        
        super().__init__(
            task_id="Output",
            loss=loss,
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["u"]
    
class DerivativeTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            du_pred = model.derivative(order=1, x=x, pde_params=input_params)
            return mse_loss(du_pred, du)
        
        super().__init__(
            task_id="Derivative",
            loss=loss,
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du"]
    
class SpatialDerivativeTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            du_pred = model.derivative(order=1, x=x, pde_params=input_params)
            return mse_loss(du_pred[:, :2], du[:, :2])
        
        super().__init__(
            task_id="Derivative_x",
            loss=loss,
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du"]
    
class TemporalDerivativeTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, du: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            du_pred = model.derivative(order=1, x=x, pde_params=input_params)
            return mse_loss(du_pred[:, 2:], du[:, 2:])
        
        super().__init__(
            task_id="Derivative_t",
            loss=loss,
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["du"]

class Derivative2Task(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
            return mse_loss(d2u_pred, d2u)
        
        super().__init__(
            task_id="Derivative2",
            loss=loss,
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["d2u"]

class SpatialDerivative2Task(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
            return mse_loss(d2u_pred[:, :2, :2], d2u[:, :2, :2])
        
        super().__init__(
            task_id="Derivative2_x",
            loss=loss,
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["d2u"]

class TemporalDerivative2Task(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(x: torch.Tensor, input_params: torch.Tensor, model: Pinn, d2u: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
            return mse_loss(d2u_pred[:, 2, 2], d2u[:, 2, 2])
        
        super().__init__(
            task_id="Derivative2_t",
            loss=loss,
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return ["d2u"]

class AdvectionReactionDiffusionTask(PhysicsTask):

    def __init__(self, 
            parameters: dict,
            velocity: Callable,
            weight: float = None, 
            device: str = "cpu"
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
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return []

class StationaryAllenCahnTask(PhysicsTask):

    def __init__(self, parameters: dict, weight: float = None, device: str = "cpu"):

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
            weight=weight,
            device=device
        )
    
    def loss_required_labels(self) -> List[str]:
        return []