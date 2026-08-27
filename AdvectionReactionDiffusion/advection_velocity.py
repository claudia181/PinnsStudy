"""
advection_velocity.py
===========

This module implements the `Velocity` class, representing velocity vector fields taking part in the advection process.

Spatio-temporal domain: xyt

Classes:
- `Velocity` (For creating velocity vector fields)
"""

import torch
import numpy as np
from typing import Callable, Self

# ===================================== Velocity class =====================================
class Velocity:
    """
        Class for velocity functions involved in the advection process.
    
        Maps the scalar time into velocity vector-type objects (producing a vector field):
        - Callable[np.ndarray, [np.ndarray, ..., np.ndarray]],
        - Callable[torch.Tensor, [torch.Tensor, ..., torch.Tensor]].
    
        Attributes
        ----------
        rotation_mode : str
            Rotation mode in {"const", "sin", "exp"}.
        radial_expansion_mode : str
            Expansion mode in {"const", "sin", "exp"}.
        rotation_weight : float
            Rotation weight.
        radial_expansion_weight : float
            Radial expantion weight.
        rotation_frequency : float
            Rotation frequency (for alpha_mode = "sin").
        radial_expansion_frequency : float
            Expansion frequency (for beta_mode = "sin").
        rotation_decay_factor : float
            Rotation decay factor (for alpha_mode = "exp").
        radial_expansion_decay_factor : float
            Expansion decay factor (for beta_mode = "exp").
        fn : Callable
            The velocity function applied:
            - v(x, y, t) = [v_x(x, y, t), v_y(x, y, t)]
            - v_x(x, y, t) = - a(t) * y + b(t) * x
            - v_y(x, y, t) = a(t) * x + b(t) * y
            - a(t) = alpha * f_a(t)
            - b(t) = beta * f_b(t)
            - f_a(t) = 
                - 1 (alpha_mode = "const")
                - sin(omega_a * t) (alpha_mode = "sin")
                - e^(- gamma_a * t) (alpha_mode = "exp")
            - f_b(t) = 
                - 1 (beta_mode = "const")
                - sin(omega_b * t) (beta_mode = "sin")
                - e^(- gamma_b * t) (beta_mode = "exp").
        """
    def __init__(
            self,

            rotation_weight: float,
            rotation_mode: str,

            radial_expansion_weight: float,
            radial_expansion_mode: str,

            rotation_frequency: float = None,
            rotation_decay_factor: float = None,
            
            radial_expansion_frequency: float = None,
            radial_expansion_decay_factor: float = None
    ) -> None:
        """
        Constructor of a velocity vector field object.
        
        Velocity function:\n
        v(x, y, t) = [v_x(x, y, t), v_y(x, y, t)],
            - v_x(x, y, t) = - a(t) * y + b(t) * x
            - v_y(x, y, t) = a(t) * x + b(t) * y
            - a(t) = alpha * f_a(t)
            - b(t) = beta * f_b(t)
            - f_a(t) = 
                - 1 (alpha_mode = "const")
                - sin(omega_a * t) (alpha_mode = "sin")
                - e^(- gamma_a * t) (alpha_mode = "exp")
            - f_b(t) = 
                - 1 (beta_mode = "const")
                - sin(omega_b * t) (beta_mode = "sin")
                - e^(- gamma_b * t) (beta_mode = "exp")
        
        Parameters
        ----------
        rotation_mode : str
            Rotation mode in {"const", "sin", "exp"}.
        radial_expansion_mode : str
            Expansion mode in {"const", "sin", "exp"}.
        rotation_weight : float
            Rotation weight.
        radial_expansion_weight : float
            Radial expantion weight.
        rotation_frequency : float
            Rotation frequency (for alpha_mode = "sin").
        radial_expansion_frequency : float
            Expansion frequency (for beta_mode = "sin").
        rotation_decay_factor : float
            Rotation decay factor (for alpha_mode = "exp").
        radial_expansion_decay_factor : float
            Expansion decay factor (for beta_mode = "exp").
        
        Returns
        -------
        _None_
        """
        self._check_configuration(
            mode=rotation_mode,
            frequency=rotation_frequency,
            decay_factor=rotation_decay_factor
        )
        self._check_configuration(
            mode=radial_expansion_mode,
            frequency=radial_expansion_frequency,
            decay_factor=radial_expansion_decay_factor
        )
        self.rotation_weight = rotation_weight
        self.rotation_mode = rotation_mode # in {"const", "sin", "exp"}
        self.rotation_frequency = rotation_frequency
        self.rotation_decay_factor = rotation_decay_factor
        self.rotation_coeff = self._coefficient_law(
            mode=rotation_mode,
            weight=rotation_weight,
            frequency=rotation_frequency,
            decay_factor=rotation_decay_factor
        )

        self.radial_expansion_weight = radial_expansion_weight
        self.radial_expansion_mode = radial_expansion_mode # in {"const", "sin", "exp"}
        self.radial_expansion_frequency = radial_expansion_frequency
        self.radial_expansion_decay_factor = radial_expansion_decay_factor
        self.radial_expantion_coeff = self._coefficient_law(
            mode=radial_expansion_mode,
            weight=radial_expansion_weight, 
            frequency=radial_expansion_frequency, 
            decay_factor=radial_expansion_decay_factor
        )

        self.fn = self._get_velocity_fn()

    # Law for evolving the coefficients of the velocity vecrtor components
    def _coefficient_law(
            self, 
            mode: str, 
            weight: float, 
            frequency: float = None, 
            decay_factor: float = None
    ) -> Callable:
        """
        Parameters
        ----------
        mode : str
            Scheduling over time ("const", "sin", "cos", "exp").
        weight : float
            Multiplying coefficient.
        frequency : float
            for mode = "sin".
        decay_factor : float
            for mode = "exp".
        """
        if weight is None: weight = 0.0
        if frequency is None: frequency = 0.0
        if decay_factor is None: decay_factor = 0.0

        if mode == "const":
            f  = lambda t: weight
        elif mode == "sin":
            f  = lambda t: weight * np.sin(frequency * t)
        elif mode == "cos":
            f  = lambda t: weight * np.cos(frequency * t)
        elif mode == "exp":
            f  = lambda t: weight * np.exp(- decay_factor * t)
        else:
            raise ValueError(f"mode must be const|sin|exp, not {mode}.")
        return f

    def _check_configuration(self, mode: str, frequency: float = None, decay_factor: float = None):
        if mode not in ["const", "sin", "cos", "exp"]:
            raise ValueError(f"Invalid source mode '{mode}'. Valid values are 'const', 'sin', 'cos', 'exp'.")
        if (mode == "sin" or mode == "cos") and frequency is None:
            raise ValueError(f"Mode '{mode}' requires to specify the frequency argument.")
        if mode == "exp" and decay_factor is None:
            raise ValueError(f"Mode '{mode}' requires to specify the decay_factor argument.")

    def _get_velocity_fn(self) -> Callable:
        # Velocity vector field
        def v(x, y, t):
            a_t = self.rotation_coeff(t)
            b_t = self.radial_expantion_coeff(t)
            vx = - a_t * y + b_t * x
            vy = a_t * x + b_t * y
            return np.array([vx, vy])
        return v

    def __call__(
            self,
            x: np.ndarray | torch.Tensor = None,
            y: np.ndarray | torch.Tensor = None,
            t: np.ndarray | torch.Tensor = None
    ) -> np.ndarray | torch.Tensor:
        """
        Call funcction.
        """
        return self.fn(x=x, y=y, t=t)

    def set_rotation(
            self,
            rotation_weight: float,
            rotation_mode: str,
            rotation_frequency: float,
            rotation_decay_factor: float
    ) -> None:
        self._check_configuration(
            mode=rotation_mode,
            frequency=rotation_frequency,
            decay_factor=rotation_decay_factor
        )
        self.rotation_weight = rotation_weight
        self.rotation_mode = rotation_mode # in {"const", "sin", "exp"}
        self.rotation_frequency = rotation_frequency
        self.rotation_decay_factor = rotation_decay_factor
        self.fn = self._get_velocity_fn()

    def set_radial_expantion(
            self,
            radial_expansion_weight: float,
            radial_expansion_mode: str,
            radial_expansion_frequency: float,
            radial_expansion_decay_factor: float
    ) -> None:
        self._check_configuration(
            mode=radial_expansion_mode,
            frequency=radial_expansion_frequency,
            decay_factor=radial_expansion_decay_factor
        )
        self.radial_expansion_weight = radial_expansion_weight
        self.radial_expansion_mode = radial_expansion_mode # in {"const", "sin", "exp"}
        self.radial_expansion_frequency = radial_expansion_frequency
        self.radial_expansion_decay_factor = radial_expansion_decay_factor
        self.fn = self._get_velocity_fn()

    def state_dict(self) -> dict:
        return {
            "rotation_weight": self.rotation_weight,
            "rotation_mode": self.rotation_mode,
            "rotation_frequency": self.rotation_frequency,
            "rotation_decay_factor": self.rotation_decay_factor,
            "radial_expansion_weight": self.radial_expansion_weight,
            "radial_expansion_mode": self.radial_expansion_mode,
            "radial_expansion_frequency": self.radial_expansion_frequency,
            "radial_expansion_decay_factor": self.radial_expansion_decay_factor
        }

    def load_state(self, state: dict) -> None:
        self._check_configuration(
            mode=state["rotation_mode"],
            frequency=state["rotation_frequency"],
            decay_factor=state["rotation_decay_factor"]
        )
        self._check_configuration(
            mode=state["radial_expansion_mode"],
            frequency=state["radial_expansion_frequency"],
            decay_factor=state["radial_expansion_decay_factor"]
        )
        self.rotation_weight = state["rotation_weight"]
        self.rotation_mode = state["rotation_mode"] # in {"const", "sin", "exp"}
        self.rotation_frequency = state["rotation_frequency"]
        self.rotation_decay_factor = state["rotation_decay_factor"]
        self.rotation_coeff = self._coefficient_law(
            mode=state["rotation_mode"],
            weight=state["rotation_weight"],
            frequency=state["rotation_frequency"],
            decay_factor=state["rotation_decay_factor"]
        )

        self.radial_expansion_weight = state["radial_expansion_weight"]
        self.radial_expansion_mode = state["radial_expansion_mode"] # in {"const", "sin", "exp"}
        self.radial_expansion_frequency = state["radial_expansion_frequency"]
        self.radial_expansion_decay_factor = state["radial_expansion_decay_factor"]
        self.radial_expantion_coeff = self._coefficient_law(
            mode=state["radial_expansion_mode"],
            weight=state["radial_expansion_weight"], 
            frequency=state["radial_expansion_frequency"], 
            decay_factor=state["radial_expansion_decay_factor"]
        )

        self.fn = self._get_velocity_fn()

    def mode_view(self) -> dict:
        state_view = {
            "rotation_weight": self.rotation_weight,
            "rotation_mode": self.rotation_mode,
            "radial_expansion_weight": self.radial_expansion_weight,
            "radial_expansion_mode": self.radial_expansion_mode
        }

        if self.rotation_mode in ["sin", "cos"]:
            state_view["rotation_frequency"] = self.rotation_frequency
        elif self.rotation_mode == "exp":
            state_view["rotation_decay_factor"] = self.rotation_decay_factor
        if self.radial_expansion_mode in ["sin", "cos"]:
            state_view["radial_expansion_frequency"] = self.radial_expansion_frequency
        elif self.radial_expansion_mode == "exp":
            state_view["radial_expansion_decay_factor"] = self.radial_expansion_decay_factor
        
        return state_view

    @classmethod
    def null_velocity(cls) -> Self:
        """
        Returns a null velocity vector field (Velocity object).
        """
        return Velocity(
            rotation_weight=0.0,
            rotation_mode="const",
            radial_expansion_weight=0.0,
            radial_expansion_mode="const"
        )