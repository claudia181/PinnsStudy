"""
initial_condition.py
===========

This module implements the `InitialCondition` classes, for the initial conditions of an advection-reaction-diffusion system.

Classes:
- `InitialCondition`
"""

from typing import Tuple, Self, List, Callable
from fipy import CellVariable, Grid2D, Gmsh2D
import numpy as np

# ===================================== Initial class =====================================
class InitialField:
    """
    """
    def __init__(self, id: str):
        self.id = id

    def __call__(self, x: np.ndarray, y: np.ndarray) -> None:
        return
    
    def state_dict(self) -> dict:
        return {"id": self.id}
    
    def load_state(self, state: dict) -> None:
        self.id = state["id"]

# ===================================== Initial class =====================================
class InitialConditions:
    """
    Superclass for the initial conditions.

    Attributes
    ----------
    components : List[InitialField]
    """
    def __init__(
            self,
            scalar_fields: List[InitialField] = None
    ) -> None:
        self.scalar_fields = scalar_fields

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u0 = np.zeros_like(self.x)
        for field in self.scalar_fields:
            u0 += field(x=x, y=y)
        return u0

    def state_dict(self) -> dict:
        state = {}
        for field in self.scalar_fields:
            state["id"] = field.state_dict()
        return state

    def load_state(self, state: dict) -> None:
        for field in self.scalar_fields:
            field.load_state(state[field.id])

# ===================================== Initial class =====================================
class ConstantField(InitialField):
    """
    """
    def __init__(self, value: float):
        super().__init__(id="constant")
        self.value = value

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.value * np.ones_like(x)
    
    def state_dict(self) -> dict:
        return super().state_dict() | {"value": self.value}
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.value = state["value"]

# ===================================== Initial class =====================================
class GaussianBumpField(InitialField):
    """
    """
    def __init__(self, amp: float, xc: float, yc: float, sigma: float):
        super().__init__(id="gaussian_bump")
        self.amp = amp
        self.xc = xc
        self.yc = yc
        self.sigma = sigma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.amp * np.exp(-((x - self.xc) ** 2 + (y - self.yc) ** 2) / (2 * self.sigma ** 2))
    
    def state_dict(self) -> dict:
        return super().state_dict() | {"amp": self.amp, "xc": self.xc, "yc": self.yc, "sigma": self.sigma}
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.amp = state["amp"]
        self.xc = state["xc"]
        self.yc = state["yc"]
        self.sigma = state["sigma"]

# ===================================== Initial class =====================================
class PeriodicCirclesField(InitialField):
    """
    """
    def __init__(self, A: float, B: float, Cx: float, Cy: float, D: float):
        super().__init__(id="periodic_circles")
        self.A = A
        self.B = B
        self.Cx = Cx
        self.Cy = Cy
        self.D = D

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.A * np.sin(self.B * np.sqrt(self.Cx * x ** 2 + self.Cy * y ** 2) + self.D)
    
    def state_dict(self) -> dict:
        return super().state_dict() | {"A": self.A, "B": self.B, "Cx": self.Cx, "Cy": self.Cy, "D": self.D}
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.A = state["A"]
        self.B = state["B"]
        self.Cx = state["Cx"]
        self.Cy = state["Cy"]
        self.D = state["D"]

# ===================================== Initial class =====================================
class PeriodicValleysField(InitialField):
    """
    """
    def __init__(self, A: float, B: float, Cx: float, Cy: float, D: float):
        super().__init__(id="periodic_valleys")
        self.A = A
        self.B = B

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.A * np.sin(self.B * (x * y)) # circle^-1
    
    def state_dict(self) -> dict:
        return super().state_dict() | {"A": self.A, "B": self.B}
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.A = state["A"]
        self.B = state["B"]

# ===================================== Initial class =====================================
class PeriodicStripesField(InitialField):
    """
    """
    def __init__(self, A: float, Bx: float, By: float):
        super().__init__(id="periodic_valleys")
        self.A = A
        self.Bx = Bx
        self.By = By

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.A * np.sin(self.Bx * x + self.By * y) # stripes
    
    def state_dict(self) -> dict:
        return super().state_dict() | {"A": self.A, "Bx": self.Bx, "By": self.By}
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.A = state["A"]
        self.Bx = state["Bx"]
        self.By = state["By"]