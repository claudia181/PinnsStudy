"""
initial_condition.py
===========

This module implements the `InitialCondition` classes, for the initial conditions of an advection-reaction-diffusion system.

Classes:
- `InitialCondition`
"""

from typing import List
import numpy as np

# ===================================== Initial field =====================================
class InitialField:
    """
    Superclass representing a generic initial condition scalar field. 
    
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.

    Attributes
    ----------
    id : str
        Field identifier.
    """
    def __init__(self, id: str):
        self.id = id

    def __call__(self, x: np.ndarray, y: np.ndarray) -> None:
        return
    
    def state_dict(self) -> dict:
        return {"id": self.id}
    
    def load_state(self, state: dict) -> None:
        self.id = state["id"]

# ===================================== Constant field =====================================
class ConstantField(InitialField):
    """
    Class representing a constant scalar field.

    It subclasses the InitialField class.
        
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.
    
    Attributes
    ----------
    value : float
        Field constant value.
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

# ===================================== Gaussian bump field =====================================
class GaussianBumpField(InitialField):
    """
    Class representing a scalar field with a gaussian bump:
    - amp * exp(-((x - xc)^2 + (y - yc)^2) / (2 * sigma^2)).

    It subclasses the InitialField class.
        
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.
    
    Attributes
    ----------
    amp : float
        Amplitude of the bump.
    xc : float
        x-coordinate of the center of the gaussian.
    yc : float
        y-coordinate of the center of the gaussian.
    sigma : float
        Standard deviation of the gaussian.
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

# ===================================== Concentric circles field =====================================
class CirclesField(InitialField):
    """
    Class representing a scalar field with concentric circles:
    - A * sin(B * sqrt(Cx * x^2 + Cy * y^2) + D).

    It subclasses the InitialField class.
        
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.
    
    Attributes
    ----------
    A : float
    B : float
    Cx : float
    Cy : float
    D : float
    """
    def __init__(self, A: float, B: float, Cx: float, Cy: float, D: float):
        super().__init__(id="periodic_circles")
        self.A = A
        self.B = B
        self.Cx = Cx
        self.Cy = Cy
        self.D = D

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.A * np.sin(self.B * np.sqrt(self.Cx * x ** 2 + self.Cy * y ** 2) + self.D) # concentric circles
    
    def state_dict(self) -> dict:
        return super().state_dict() | {"A": self.A, "B": self.B, "Cx": self.Cx, "Cy": self.Cy, "D": self.D}
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.A = state["A"]
        self.B = state["B"]
        self.Cx = state["Cx"]
        self.Cy = state["Cy"]
        self.D = state["D"]

# ===================================== Valleys field =====================================
class ValleysField(InitialField):
    """
    Class representing a scalar field with periodic valleys:
    - A * sin(B * (x * y)).

    It subclasses the InitialField class.
        
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.
    
    Attributes
    ----------
    A : float
    B : float
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

# ===================================== Stripes field =====================================
class StripesField(InitialField):
    """
    Class representing a striped scalar field:
    - A * sin(Bx * x + By * y).

    It subclasses the InitialField class.
        
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.
    
    Attributes
    ----------
    A : float
    Bx : float
    By : float
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

# ===================================== Grid field =====================================
class GridField(InitialField):
    """
    Class representing a grid-like scalar field:
    - Ax * sin(Bx * x^2 + Cx) + Ay * sin(By * y^2 + Cy).

    It subclasses the InitialField class.
        
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.
    
    Attributes
    ----------
    Ax : float
    Ay : float
    Bx : float
    By : float
    Cx : float
    Cy : float
    """
    def __init__(self, Ax: float, Ay: float, Bx: float, By: float, Cx: float, Cy: float) -> None:
        super().__init__(id="periodic_valleys")
        self.Ax = Ax
        self.Ay = Ay
        self.Bx = Bx
        self.By = By
        self.Cx = Cx
        self.Cy = Cy

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.Ax * np.sin(self.Bx * x ** 2 + self.Cx) + self.Ay * np.sin(self.By * y ** 2 + self.Cy) # grid
    
    def state_dict(self) -> dict:
        return super().state_dict() | {
            "Ax": self.Ax,
            "Ay": self.Ay, 
            "Bx": self.Bx,
            "By": self.By,
            "Cx": self.Cx,
            "Cy": self.Cy
            }
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.Ax = state["Ax"]
        self.Ay = state["Ay"]
        self.Bx = state["Bx"]
        self.By = state["By"]
        self.Cx = state["Cx"]
        self.Cy = state["Cy"]

# ===================================== Uniform noise field =====================================
class UniformNoiseField(InitialField):
    """
    Class representing a uniform noise scalar field:
    - U(min_noise, max_noise).

    It subclasses the InitialField class.
        
    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.
    
    Attributes
    ----------
    min_noise : float
    max_noise : float
    """
    def __init__(self, min_noise: float, max_noise: float) -> None:
        super().__init__(id="periodic_valleys")
        self.min_noise = min_noise
        self.max_noise = max_noise

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.random.uniform(low=self.min_noise * np.ones_like(x), high=self.max_noise * np.ones_like(x))
    
    def state_dict(self) -> dict:
        return super().state_dict() | {
            "min_noise": self.min_noise,
            "max_noise": self.max_noise
            }
    
    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.min_noise = state["min_noise"]
        self.max_noise = state["max_noise"]

# ===================================== Initial condition =====================================
class InitialCondition:
    """
    Class for the initial condition.

    It is callable and provides `state_dict` and `load_state` functions for saving and loading the state of an object.

    Attributes
    ----------
    scalar_fields : List[InitialField]
        The components (scalar fields) to sum up to obtain the initial condition.
    """
    def __init__(
            self,
            scalar_fields: List[InitialField] = None
    ) -> None:
        self.scalar_fields = scalar_fields

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u0 = np.zeros_like(x)
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