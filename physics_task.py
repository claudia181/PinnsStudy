from typing import Callable
import torch

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
        self.parameters = parameters
        self._lhs = lhs
        self.loss = loss
        self.weight = weight
        self.grad_norm = None
        self.loss_value = None
        self.device = device
    
    def lhs(self,
            n: torch.Tensor = None,
            u: torch.Tensor = None,
            du: torch.Tensor = None,
            d2u: torch.Tensor = None,
            lap: torch.Tensor = None,
            lap2: torch.Tensor = None
        ) -> torch.Tensor:
        return self._lhs(u=u, du=du, d2u=d2u, lap=lap, lap2=lap2, n=n, **self.parameters)


class NeumannBCTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def lhs(du: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            outward_flux = (du[:, :2] * n).sum(dim=1)
            return outward_flux

        def loss(du: torch.Tensor, du_pred: torch.tensor, n: torch.Tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(lhs(du=du_pred, n=n), lhs(du=du, n=n))
        
        super().__init__(
            task_id="NeumannBC",
            lhs=lhs,
            loss=loss,
            weight=weight,
            device=device
        )

class DirichletBCTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def lhs(u: torch.Tensor) -> torch.Tensor:
            return u

        def loss(u: torch.Tensor, u_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(u_pred, u)
        
        super().__init__(
            task_id="DirichletBC",
            lhs=lhs,
            loss=loss,
            weight=weight,
            device=device
        )

class ICTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def lhs(u: torch.Tensor) -> torch.Tensor:
            return u

        def loss(u: torch.Tensor, u_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(u_pred, u)
        
        super().__init__(
            task_id="IC",
            lhs=lhs,
            loss=loss,
            weight=weight,
            device=device
        )

class OutputTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(u: torch.Tensor, u_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(u_pred, u)
        
        super().__init__(
            task_id="Output",
            loss=loss,
            weight=weight,
            device=device
        )

class DerivativeTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(du: torch.Tensor, du_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(du_pred, du)
        
        super().__init__(
            task_id="Derivative",
            loss=loss,
            weight=weight,
            device=device
        )

class SpatialDerivativeTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(du: torch.Tensor, du_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(du_pred[:, :2], du[:, :2])
        
        super().__init__(
            task_id="Derivative_x",
            loss=loss,
            weight=weight,
            device=device
        )

class TemporalDerivativeTask(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(du: torch.Tensor, du_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(du_pred[:, 2:], du[:, 2:])
        
        super().__init__(
            task_id="Derivative_t",
            loss=loss,
            weight=weight,
            device=device
        )

class Derivative2Task(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(d2u: torch.Tensor, d2u_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(d2u_pred, d2u)
        
        super().__init__(
            task_id="Derivative",
            loss=loss,
            weight=weight,
            device=device
        )

class SpatialDerivative2Task(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(d2u: torch.Tensor, d2u_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(d2u_pred[:, :2, :2], d2u[:, :2, :2])
        
        super().__init__(
            task_id="Derivative_x",
            loss=loss,
            weight=weight,
            device=device
        )

class TemporalDerivative2Task(PhysicsTask):

    def __init__(self, weight: float = None, device: str = "cpu"):

        def loss(d2u: torch.Tensor, d2u_pred: torch.tensor) -> torch.Tensor:
            mse_loss = torch.nn.MSELoss(reduction='mean')
            return mse_loss(d2u_pred[:, 2, 2], d2u[:, 2, 2])
        
        super().__init__(
            task_id="Derivative_t",
            loss=loss,
            weight=weight,
            device=device
        )