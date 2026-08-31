"""
boundary.py
===========

This module implements the `RectangularBoundary` and `CircularBoundary` classes, for the boundary conditions of an advection-reaction-diffusion system.

Classes:
- `Boundary`
- `RectangularBoundary`
- `CircularBoundary`
"""

from typing import Tuple
from fipy import CellVariable, Grid2D, Gmsh2D

# ===================================== Boundary class =====================================
class Boundary:
    """
    Class for the boundaries.

    Attributes
    ----------
    shape : str
    """
    def __init__(
            self,
            shape: str = "rectangle"
    ) -> None:
        
        self.shape = shape

    def _check(self, shape: str):
        """
        Check the acceptability of a side description.
        """
        if shape not in ["rectangle", "circle"]:
            raise ValueError(f"Invalid boundary shape '{shape}': Valid conditions are 'rectangle' and 'circle'.")

    def apply_conditions(self, rho: CellVariable, mesh: Grid2D | Gmsh2D) -> None:
        return

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the object.
        """
        return {
            "shape": self.shape
        }

    def load_state(self, state: dict) -> None:
        """
        Loads the given state into the object.
        """
        self._check(state["shape"])
        self.top = state["shape"]

# ===================================== RectangularBoundary class =====================================
class RectangularBoundary(Boundary):
    """
    Class for the boundaries of rectangular shape of advection-reaction-diffusion systems.

    `mode`: "Dirichlet" | "Neumann".

    `value`:
    - u value if `mode` = "Dirichlet",
    - u outward flux if `mode` = "Neumann".

    Attributes
    ----------
    top : Tuple[str, float]
        (top side mode, top side value)
    bottom : Tuple[str, float]
        (bottom side mode, bottom side value)
    left : Tuple[str, float]
        (left side mode, left side value)
    right : Tuple[str, float]
        (right side mode, right side value)
    """
    def __init__(
            self,
            top: Tuple[str, float] = ("Neumann", 0.0),
            bottom: Tuple[str, float] = ("Neumann", 0.0),
            left: Tuple[str, float] = ("Neumann", 0.0),
            right: Tuple[str, float] = ("Neumann", 0.0)
    ) -> None:
        self._check(top)
        self._check(bottom)
        self._check(left)
        self._check(right)

        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        super().__init__(shape="rectangle")

    def _check(self, side: Tuple[str, float]):
        """
        Check the acceptability of a side description.
        """
        if side[0] not in ["Dirichlet", "Neumann"]:
            raise ValueError(f"Invalid boundary condition '{side[0]}': Valid conditions are 'Dirichlet' and 'Neumann'.")

    def apply_conditions(self, rho: CellVariable, mesh: Grid2D | Gmsh2D) -> None:
        meshFaces = [mesh.facesLeft, mesh.facesRight, mesh.facesTop, mesh.facesBottom]
        sides = [self.left, self.right, self.top, self.bottom]
        for side, mesh_faces in zip(sides, meshFaces):
            mode, value = side
            if mode == "Neumann":
                rho.faceGrad.constrain(value, mesh_faces)
            elif mode == "Dirichlet":
                rho.constrain(value, mesh_faces)


    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the object.
        """
        state = {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right
        }
        return super().state_dict() | state

    def load_state(self, state: dict) -> None:
        """
        Loads the given state into the object.
        """
        super().load_state()
        self._check(state["top"])
        self._check(state["bottom"])
        self._check(state["left"])
        self._check(state["right"])

        self.top = state["top"]
        self.bottom = state["bottom"]
        self.left = state["left"]
        self.right = state["right"]

# ===================================== CircularBoundary class =====================================
class CircularBoundary(Boundary):
    """
    Class for the boundaries of circular shape of advection-reaction-diffusion systems.

    `mode`: "Dirichlet" | "Neumann".

    `value`:
    - u value if `mode` = "Dirichlet",
    - u outward flux if `mode` = "Neumann".

    Attributes
    ----------
    circumference : Tuple[str, float]
        (boundary mode, boundary value)
    """
    def __init__(
            self,
            circumference: Tuple[str, float] = ("Dirichlet", 0.0)
    ) -> None:
        self._check(circumference)
        self.circumference = circumference
        super().__init__(shape="circle")

    def _check(self, perimeter: Tuple[str, float]):
        """
        Check the acceptability of a circumference description.
        """
        if perimeter[0] not in ["Dirichlet", "Neumann"]:
            raise ValueError(f"Invalid boundary condition '{perimeter[0]}': Valid conditions are 'Dirichlet' and 'Neumann'.")

    def apply_conditions(self, rho: CellVariable, mesh: Grid2D | Gmsh2D) -> None:
        mode, value = self.circumference
        if mode == "Neumann":
            rho.faceGrad.constrain(value, mesh.exteriorFaces)
        elif mode == "Dirichlet":
            rho.constrain(value, mesh.exteriorFaces)

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the object.
        """
        return super().state_dict() | {"circumference": self.circumference}

    def load_state(self, state: dict) -> None:
        """
        Loads the given state into the object.
        """
        super().load_state()
        self._check(state["circumference"])
        self.circumference = state["circumference"]