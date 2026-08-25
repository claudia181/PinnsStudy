"""
advection_reaction_diffusion.py
===========

This module implements the `AdvectionReactionDiffusion` class, representing advection-reaction-diffusion systems, giving methods to define a system and to solve (simulate it).

Spatio-temporal domain:
- Time-dependent
- 2-dimentional spatial domain

Functions:
- `velocity_field`

Classes:
- `Source` (For creating explicit or implicit scalar source fields)
- `AdvectionReactionDiffusion`.
"""

import torch
from fipy import CellVariable, Grid2D, Gmsh2D, TransientTerm, ConvectionTerm, DiffusionTerm, Viewer, FaceVariable
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any, List, Tuple, Set
from trajectory import Trajectory

class Source:
    """
    Class for source functions involved in the reaction process.

    Maps vector-type object into scalars (producing a scalar field):
    - Callable[[np.ndarray, ..., np.ndarray], np.ndarray],
    - Callable[[torch.Tensor, ..., torch.Tensor], torch.Tensor].

    The input of the source function can be the spatio-temporal coordinates (or a subset of them) or the scalar field `u` (implicit sources).

    Attributes
    ----------
    mode : str
        The type of source: 'constant', 'decay', 'periodic', 'temporary', 'logistic', 'AllenCahn', 'Arrhenius'.
    sigma : float
        Std dev of the gaussian bump of the source.
    center : Tuple[float, float]
        Center of the gaussian bump of the source.
    amp : float
        Amplitude of the gaussian bump of the source.
    delta : float
        Decay factor for decaying sources.
    period : float
        Period for periodic sources.
    A : float
        Parameter for implicit sources 'logistic', 'AllenCahn' or 'Arrhenius'.
    B : float
        Parameter for implicit sources 'logistic' or 'Arrhenius'.
    fn : Callable
        The source function applied:
        - 'constant': s(x, y) = amp * G(x, y)
        - 'decay': s(x, y, t) = amp * e^(- delta * t) * G(x, y)
        - 'periodic': s(x, y, t) = amp * sin((2 pi / period) * t) * G(x, y)
        - 'temporary': s(x, y, t) = amp * (t < period) * G(x, y)
        - 'logistic': s(u) = A * u^2 - B * u
        - 'AllenCahn': s(u) = A * (u^3 - u)
        - 'Arrhenius': s(u) = A * e^(- B / u)
    """
    def __init__(
            self,
            mode: str,
            sigma: float = None,
            center: Tuple[float, float] = None,
            amp: float = None,
            delta: float = None,
            period: float = None,
            A: float = None,
            B: float = None
    ) -> None:
        """
        Constructor: build a scalar source function with state s(x, y, t, u) with values in R.
        
        Gaussian sources s(x, y, t) =
            - amp * f(t) * G(x, y)
            - G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) )
            - f(t) =
                - 1 (constant)
                - e^(- delta * t) (decay)
                - sin((2 pi / period) * t) (periodic)
                - (t < period) (temporary)
        Sources s(u) =
            - A * u^2 - B * u (logistic)
            - A * (u^3 - u) (AllenCahn)
            - A * e^(- B / u) (Arrhenius)
        
        Parameters
        ----------
        mode : str
            Source function type identifier:
            - "constant": s(x, y) = amp * G(x, y)
            - "decay": s(x, y, t) = amp * e^(- delta * t) * G(x, y)
            - "oscillate": s(x, y, t) = amp * sin((2 pi / period) * t) * G(x, y)
            - "temporary": s(x, y, t) = amp * (t < period) * G(x, y)
            - "logistic": s(u) = A * u^2 - B * u
            - "AllenCahn": s(u) = A * (u^3 - u)
            - "Arrhenius": s(u) = A * e^(- B / u)
        sigma : float
            Standard deviation of the Gaussian.
        center : tuple
            Center (xc, yc) of the Gaussian.
        amp : float
            For Gaussian-type sources: s(x, y, t) = amp * f(t) * G(x, y).
        delta : float
            For mode = "decay", the decay rate: f(t) = e^(- delta * t).
        period : float
            - For mode = "oscillate": f(t) = sin(2 * pi / period * t).
            - For mode = "temporary": f(t) = (t < period).
        A : float
            - For mode = "logistic": s(u) = A * u^2 - B * u
            - For mode = "AllenCahn": s(u) = A * (u^3 - u)
            - For mode = "Arrhenius": s(u) = A * e^(- B / u)
        B : float
            - For mode = "logistic": s(u) = A * u^2 - B * u
            - For mode = "Arrhenius": s(u) = A * e^(- B / u)
    
        Returns
        -------
        _None_
        """
        self._check_configuration(mode=mode, amp=amp, delta=delta, period=period, A=A, B=B)

        self.sigma = sigma
        self.center = center
        self.mode = mode
        self.amp = amp
        self.delta = delta
        self.period = period
        self.A = A
        self.B = B
        self.fn = self._get_source_fn()

    def _check_configuration(
            self,
            mode: str,
            amp: float,
            delta: float,
            period: float,
            A: float,
            B: float
    ) -> None:
        """
        Check the acceptability of a configuration.
        """
        if mode in ["constant", "decay", "periodic", "temporary"]:
            self._is_implicit = False
        elif mode in ["logistic", "AllenCahn", "Arrhenius"]:
            self._is_implicit = True
        else:
            raise ValueError(f"Argument 'mode' must be 'constant'|'decay'|'periodic'|'temporary'|'logistic'|'AllenCahn'|'Arrhenius', not {mode}.")
        
        if not self._is_implicit:
            if amp is None:
                raise ValueError(f"Explicit source function requires to specify the amp parameter (amplitude).")
            if mode == "decay" and delta is None:
                raise ValueError(f"Decaying source function requires to specify the delta parameter (decay factor).")
            if mode == "periodic" and period is None:
                raise ValueError(f"Periodic source function requires to specify the period parameter (signal period).")
            if mode == "temporary" and period is None:
                raise ValueError(f"Temporary source function requires to specify the period parameter (signal duration).")
        else:
            if A is None:
                raise ValueError(f"Implicit source function requires to specify the A parameter.")
            if B is None and mode != "AllenCahn":
                raise ValueError(f"Implicit source function requires to specify the B parameter.")

    def _get_source_fn(self) -> Callable:
        """
        Returns the source function corresponding to the state of the object.
        """
        if self.mode == "constant":
            # Constant source
            def source_fn(x, y, **kwargs):
                if x is None or y is None:
                    raise ValueError(f"A constant source requires spatial coordinates: x = {x}, y = {y}.")
                return self.amp * self._G(x, y) 
                    
        elif self.mode == "decay":
            def source_fn(x, y, t, **kwargs):
                # Decaying source
                if x is None or y is None or t is None:
                    raise ValueError(f"A decaying source requires spatio-temporal coordinates: x = {x}, y = {y}, t = {t}.")
                return self.amp * np.exp(- self.delta * t) * self._G(x, y)
                    
        elif self.mode == "periodic":
            # Periodic source
            def source_fn(x, y, t, **kwargs):
                if x is None or y is None or t is None:
                    raise ValueError(f"A periodic source requires spatio-temporal coordinates: x = {x}, y = {y}, t = {t}.")       
                w = 2 * np.pi / self.period
                return self.amp * np.sin(w * t) * self._G(x, y)
                
        elif self.mode == "temporary":
            # Temporary source
            def source_fn(x, y, t, **kwargs):
                if x is None or y is None or t is None:
                    raise ValueError(f"A temporary source requires spatio-temporal coordinates: x = {x}, y = {y}, t = {t}.")
                return self.amp * self._G(x, y) * (t < self.period)
            
        elif self.mode == "logistic":
            # Logistic source
            def source_fn(u, x = None, y = None, t = None):    
                if u is None:
                    raise TypeError(f"A logistic source requires 'u'.")
                return self.A * u ** 2 - self.B * u
                
        elif self.mode == "AllenCahn":
            # Allen-Cahn-type source
            def source_fn(u, **kwargs):
                if u is None:
                    raise TypeError(f"An AllenCahn source requires 'u'.")
                return self.A * (u ** 3 - u)
                        
        elif self.mode == "Arrhenius":
            # Arrhenius-type source
            def source_fn(u, **kwargs):
                if u is None:
                    raise TypeError(f"An Arrhenius source requires 'u'.")
                return self.A * np.exp(- self.B / u)

        return source_fn

    def _G(# Gaussian spot
            self,
            x: np.ndarray | torch.Tensor, 
            y: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """
        Gaussian bump at (x, y).
        """
        xc, yc = self.center
        return np.exp(- ((x - xc) ** 2 + (y - yc) ** 2)/(2 * self.sigma ** 2))

    def __call__(
            self,
            x: np.ndarray | torch.Tensor = None,
            y: np.ndarray | torch.Tensor = None,
            t: np.ndarray | torch.Tensor = None,
            u: np.ndarray | torch.Tensor = None
    ) -> np.ndarray | torch.Tensor:
        """
        Call funcction.
        """
        return self.fn(x=x, y=y, t=t, u=u)

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the object.
        """
        return {
            "mode": self.mode,
            "amp": self.amp,
            "center": self.center,
            "sigma": self.sigma,
            "delta": self.delta,
            "period": self.period,
            "A": self.A,
            "B": self.B
        }

    def load_state(self, state: dict):
        """
        Loads the given state into the object.
        """
        self._check_configuration(
            mode=state["mode"], 
            amp=state["amp"], 
            delta=state["delta"], 
            period=state["period"], 
            A=state["A"], 
            B=state["B"]
        )
        self.mode = state["mode"]
        self.amp = state["amp"]
        self.center = state["center"]
        self.sigma = state["sigma"]
        self.delta = state["delta"]
        self.period = state["period"]
        self.A = state["A"]
        self.B = state["B"]

    def mode_view(self) -> dict:
        """
        Returns a dictionary of the object state taking part in the call process (the interesting pieces for the current mode).
        """
        if self.mode == "constant":
            return {
                "mode": self.mode, 
                "amp": self.amp
            }
        
        elif self.mode == "decay":
            return {
                "mode": self.mode, 
                "amp": self.amp,
                "center": self.center,
                "sigma": self.sigma,
                "delta": self.delta
            }
        elif self.mode == "periodic" or self.mode == "temporary":
            return {
                "mode": self.mode, 
                "amp": self.amp,
                "center": self.center,
                "sigma": self.sigma,
                "period": self.period
            }
                    
        elif self.mode == "logistic" or self.mode == "Arrhenius":
            return {
                "mode": self.mode, 
                "A": self.A,
                "B": self.B
            }
                        
        elif self.mode == "AllenCahn":
            return {
                "mode": self.mode, 
                "A": self.A
            }

# A null source
null_source = Source(
    mode="constant",
    center=(0.0, 0.0),
    amp=0.0
)


class Velocity:
    def __init__(
            self,
            rotation_weight: float = None,
            radial_expansion_weight: float = None,
            rotation_mode: str = None,
            radial_expansion_mode: str = None,
            rotation_frequency: float = None,
            radial_expansion_frequency: float = None,
            rotation_decay_factor: float = None,
            radial_expansion_decay_factor: float = None
    ) -> None:
        self.rotation_weight = rotation_weight
        self.radial_expansion_weight = radial_expansion_weight
        self.rotation_mode = rotation_mode # in {"const", "sin", "exp"}
        self.radial_expansion_mode = radial_expansion_mode # in {"const", "sin", "exp"}
        self.rotation_frequency = rotation_frequency
        self.radial_expansion_frequency = radial_expansion_frequency
        self.rotation_decay_factor = rotation_decay_factor
        self.radial_expansion_decay_factor = radial_expansion_decay_factor

        self.a

    # Law for evolving the coefficients of the velocity vecrtor components
    def _coefficient_law(
            self, 
            mode: str, 
            weight: float = None, 
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
        if mode is None: mode = "const"
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


def velocity_field(field: str = "rotation_expansion", **p: Any) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Build a velocity vector field.

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
    field : str
        For now only "rotation_expansion" available.
    **p : Any
        Additional keyword arguments:
        - rotation_mode : (str, default: "const") rotation mode in {"const", "sin", "exp"}
        - radial_expansion_mode : (str, default: "const") expansion mode in {"const", "sin", "exp"}
        - rotation_weight : (float, default: 1.0) rotation weight
        - radial_expansion_weight : (float, default: 0.0) radial expantion weight
        - rotation_frequency : (float, default: 1.0) rotation frequency (for alpha_mode = "sin")
        - radial_expansion_frequency : (float, default: 1.0) expansion frequency (for beta_mode = "sin")
        - rotation_decay_factor : (float, default: 0.5) rotation decay factor (for alpha_mode = "exp")
        - radial_expansion_decay_factor : (float, default: 0.5) expansion decay factor (for beta_mode = "exp").
    """
    # Law for evolving the coefficients of the velocity vecrtor components
    def law(mode, weight, frequency = None, decay_factor = None):
        # mode: scheduling over time (const, sin, cos, exp)
        # weight: multiplying coefficient
        # frequency: for mode = "sin"
        # decay_factor: for mode = "exp"
        if weight is None: weight = 0.0
        if frequency is None: frequency = 0.0
        if decay_factor is None: decay_factor = 0.0
        if mode is None: mode = "const"
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

    if field == "rotation_expansion":
        # Defaults
        rotation_weight = p.get("rotation_weight", 1.0)
        radial_expansion_weight = p.get("radial_expansion_weight", 0.0)
        rotation_mode = p.get("rotation_mode", "const")
        radial_expansion_mode = p.get("radial_expansion_mode",  "const")
        rotation_frequency = p.get("rotation_frequency", 1.0) # (rotation mode = "sin")
        radial_expansion_frequency = p.get("radial_expansion_frequency", 1.0) # (expansion mode = "sin")
        rotation_decay_factor = p.get("rotation_decay_factor", 1.0) # (rotation mode = "exp")
        radial_expansion_decay_factor = p.get("radial_expansion_decay_factor", 1.0) # (expansion mode = "exp")

        # Rotation time law
        a = law(
                mode=rotation_mode,
                weight=rotation_weight,
                frequency=rotation_frequency,
                decay_factor=rotation_decay_factor
            )
        
        # Radial expantion time law
        b = law(
                mode=radial_expansion_mode,
                weight=radial_expansion_weight, 
                frequency=radial_expansion_frequency, 
                decay_factor=radial_expansion_decay_factor
            )

        # Velocity vector field
        def v(x, y, t):
            a_t = a(t)
            b_t = b(t)
            return np.array([- a_t * y + b_t * x, a_t * x + b_t * y])

        return v

    raise ValueError(f"Unknown field '{field}'")


def null_velocity_field() -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Returns a null velocity vector field.
    """
    return velocity_field(field="rotation_expansion", alpha_mode="const", alpha=0.0, beta_mode="const", beta=0.0)

# ===================================== AdvectionReactionDiffusion class =====================================
class AdvectionReactionDiffusion:
    """
    Class representing an advection-reaction-diffusion system.

    Attributes
    ----------
    v : Callable
        Velocity field function.
    s : Callable
        Source field function.
    i_s : str
        Implicit source, i.e. use u, "logistic", "Arrhenius" or "AllenCahn".
    D : float
        Diffusion coefficient.
    x : CellVariable | np.ndarray
        x coordinates of cell centers of the spatial grid.
    y : CellVariable | np.ndarray
        y coordinates of cell centers of the spatial grid.
    x_faces : FaceVariable | np.ndarray
        x coordinates of cell faces of the spatial grid.
    y_faces : FaceVariable | np.ndarray
        y coordinates of cell faces of the spatial grid.
    dx : float
        Spatial resolution of the spatial grid.
    xmin : float
        Minimum x coordinate value.
    ymin : float
        Minimum y coordinate value.
    xmax : float
        Maximum x coordinate value.
    ymax : float
        Maximum y coordinate value.
    shape : str
        Shape of the system domain: "rectangle" | "circle".
    mesh : Grid2D | Gmsh2D
        Mesh of the grid domain of points. 
    u0 : np.ndarray
        Initial conditions, i.e. initial (at t0) u values.
    boundary_mode : str
        Boundary mode for the circle.
    boundary_value :
        Boundary value (out normal or function value) for the circle.
    left_mode : str
        Boundary mode for the left side of the rectangle.
    left_value : float
        Boundary value for the left side of the rectangle.
    right_mode : str
        Boundary mode for the right side of the rectangle.
    right_value : float
        Boundary value for the right side of the rectangle.
    top_mode : str
        Boundary mode for the top side of the rectangle.
    top_value : float
        Boundary value for the top side of the rectangle.
    bottom_mode : str
        Boundary mode for the bottom side of the rectangle.
    bottom_value : float
        Boundary value for the bottom side of the rectangle.
    trajectory : Trajectory
        Simulated trajectory of the system.
    """

    def __init__(
            self,
            velocity: Callable = None,
            source: Callable = None,
            implicit_source: Callable = None,
            diffusion_coeff: float = None
            ):
        """
        Constructor.

        Parameters
        ----------
        velocity : Callable
            Velocity field.
            - Default: 
            `velocity_field(
                rotation_mode="const", 
                radial_expansion_mode="const", 
                rotation_weight=1.0, 
                radial_expansion_weight=0.0
            )`.
        source : Callable
            Source function defined on spatio-temporal coordinates.
            - Default: `constant_source()`.
        implicit_source : Callable
            Source function defined in terms of u.
            - Default: `constant_source()`.
        diffusion_coeff : float
            The diffusion coefficient.
            - Default: `0.0`.
        """
        # Set the velocity field
        if velocity is None:
            self.v = null_velocity_field()
        else:
            self.v = velocity

        # Set the source field
        if source is None:
            self.s = null_source()
        else:
            self.s = source
        if implicit_source is None:
            self.i_s = null_source()
        else:
            self.i_s = implicit_source
        
        if diffusion_coeff is None:
            self.D = 0.0
        else:
            self.D = diffusion_coeff
        
        # Spatial grid
        self.x, self.y = None, None
        self.x_faces, self.y_faces = None, None
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None
        self.mesh = None
        self.dx = None

        # Initial state
        self.u0 = None

        # Rectangular domain: boundary modes and values
        self.left_mode, self.left_value = None, None
        self.right_mode, self.right_value = None, None
        self.top_mode, self.top_value = None, None
        self.bottom_mode, self.bottom_value = None, None

        # Circular domain: boundary mode and value
        self.boundary_mode = None
        self.boundary_value = None

        # Domain shape
        self.shape = None

        # Trajectory
        self.trajectory = None
    
    def set_spatial_points(self, 
            mode: str, 
            x_range: tuple = None, 
            y_range: tuple = None, 
            dx: float = None,
            cell_size: float = None,
            radius: float = None
        ) -> None:
        """
        Set the spatial (2D) domain, filling
        - self.x, self.y,
        - self.x_faces, self.y_faces,
        - self.xmax, self.xmin, self.ymax, self.ymin.

        Parameters
        ----------
        mode : str
            "rectangle" | "circle".
        x_range : tuple
            [x_min, x_max].
        y_range : tuple
            [y_min, y_max].
        dx : float
            x-step.
        cell_size : float
            Cell size for the "circle" mode.
        radius : float
            Radius for the "circle" mode.

        Returns
        -------
        None
        """
        self.shape = mode
        self.dx = dx

        # RECTANGULAR GRID
        if mode == "rectangle":
            # Minimum and maximum horizontal and vertical values (defining the xy plane)
            self.xmin, self.xmax = x_range
            self.ymin, self.ymax = y_range

            # Number of horizontal and vertical cells
            nx = int(round((self.xmax - self.xmin) / dx))
            ny = int(round((self.ymax - self.ymin) / dx))

            # Generate the rectangular mesh
            self.mesh = Grid2D(dx=dx, dy=dx, nx=nx, ny=ny)

            # Store the coordinates of the centers of the cells 
            # of the grid in self.x and self.y
            # (cellCenters returns (2, n_cells))
            xc, yc = self.mesh.cellCenters
            self.x = xc + self.xmin
            self.y = yc + self.ymin

            # Store the coordinates of the centers of the cells 
            # of the grid boundary in self.x_faces and self.y_faces
            # (faceCenters returns (2, n_cells))
            xf, yf = self.mesh.faceCenters
            self.x_faces = xf + self.xmin
            self.y_faces = yf + self.ymin

        # CIRCULAR GRID
        elif mode == "circle":
            # Generate circular mesh
            # (this require gmsh)
            self.mesh = Gmsh2D(f'''
                Point(1) = {{0, 0, 0, {cell_size}}};
                Point(2) = {{{radius}, 0, 0, {cell_size}}};
                Point(3) = {{0, {radius}, 0, {cell_size}}};
                Point(4) = {{-{radius}, 0, 0, {cell_size}}};
                Point(5) = {{0, -{radius}, 0, {cell_size}}};
                Circle(1) = {{2, 1, 3}};
                Circle(2) = {{3, 1, 4}};
                Circle(3) = {{4, 1, 5}};
                Circle(4) = {{5, 1, 2}};
                Curve Loop(1) = {{1, 2, 3, 4}};
                Plane Surface(1) = {{1}};
            ''')

            # Store the coordinates of the centers 
            # of the cells of the grid in self.x and self.y
            # (cellCenters returns (2, n_cells))
            self.x, self.y = self.mesh.cellCenters

            # Store the coordinates of the centers 
            # of the cells of the grid boundary in self.x_faces and self.y_faces
            # (faceCenters returns (2, n_faces))
            self.x_faces, self.y_faces = self.mesh.faceCenters

            # Update domain bounds
            self.xmin, self.xmax = -radius, radius
            self.ymin, self.ymax = -radius, radius

    def set_IC(
            self,
            gaussian: bool, 
            periodic_circles: bool, 
            periodic_valleys: bool, 
            periodic_stripes: bool, 
            periodic_grid: bool, 
            uniform_noise: bool, 
            u0: np.ndarray = None, 
            centers: List[Tuple[float, float]] = None, amps: List[float] = None, sigmas: List[float] = None, 
            A: float = None, Ax: float = None, Ay: float = None, 
            B: float = None, Bx: float = None, By: float = None, 
            Cx: float = None, Cy: float = None, 
            D: float = None, 
            min_noise: float = None, max_noise: float = None
            ) -> None:
        """
        Set the initial conditions.

        Default: zero on all the domain.

        Parameters
        ----------
        gaussian : bool
            If True, a set of Gaussian bumps is added to the scalar field u0.
            - amp * e^(-((x - xc)^2 + (y - yc)^2) / (2 * sigma^2)).
        periodic_circles : bool
            If True, a set of concentric circles is added to the scalar field u0.
            - A * sin(B * sqrt(Cx * x^2 + Cy * y^2) + D).
        periodic_valleys : bool
            If True, a set of concentric valleys is added to the scalar field u0.
            - A * sin(B * (x * y))
        periodic_stripes : bool
            If True, a set of stripes is added to the initial scalar field u0.
            - A * sin(Bx * x + By * y)
        periodic_grid : bool
            If True, add sine waves along the x and y dimentions to the scalar field u0.
            - Ax * sin(Bx * x^2 + Cx) + Ay * sin(By * y^2 + Cy)
        uniform_noise : bool
            If True, add uniform noise (between min_noise and max_noise) to the scalar field u0.
        u0 : np.ndarray
            Initial scalar field.
        centers : list
            Used if gaussian is True; centers of the Gaussians.
        amps : list
            Used if gaussian is True; amplitudes, one for each center, regulate the height of each Gaussian.
        sigmas : list
            Used if gaussian is True; regulate the width of each Gaussian.
        A : float
            Parameter to shape the initial state (see above).
        Ax : float
            Parameter to shape the initial state (see above).
        Ay : float
            Parameter to shape the initial state (see above).
        B : float
            Parameter to shape the initial state (see above).
        Bx : float
            Parameter to shape the initial state (see above).
        By : float
            Parameter to shape the initial state (see above).
        Cx : float
            Parameter to shape the initial state (see above).
        Cy : float
            Parameter to shape the initial state (see above).
        D : float
            Parameter to shape the initial state (see above).
        min_noise : float
            Parameter to shape the initial state (see above).
        max_noise : float
            Parameter to shape the initial state (see above).
        
        Returns
        -------
        None
        """
        # Base scalar field, which can be successively deformed by adding bumps, valleys or uniform noise.
        if u0 is not None:
            self.u0 = u0 * np.ones_like(self.x)
        else:
            self.u0 = np.zeros_like(self.x)

        # Adding gaussian bumps
        if gaussian:
            def normal(x0, y0, sigma = 0.1, amp = 1.0):
                return amp * np.exp(-((self.x - x0)**2 + (self.y - y0)**2) / (2 * sigma**2))

            if centers is None or centers == []:
                centers = []
                amps = []
                sigmas = []
            else:
                if amps is None or amps == []:
                    amps = [1.0 for _ in centers]
                if sigmas is None or sigmas == []:
                    sigmas = [0.1 for _ in centers]

            for center, amp, sigma in zip(centers, amps, sigmas):
                self.u0 += normal(x0=center[0], y0=center[1], sigma=sigma, amp=amp)
        
        # Adding concentric circles
        if periodic_circles:
            self.u0 += A * np.sin(B * np.sqrt(Cx * self.x**2 + Cy * self.y**2) + D) # concentric circles
        if periodic_valleys:
            self.u0 += A * np.sin(B * (self.x * self.y)) # circle^-1
        
        # Adding stripes
        if periodic_stripes:
            self.u0 += A * np.sin(Bx * self.x + By * self.y) # stripes
        
        # Adding grid pattern
        if periodic_grid:
            self.u0 += Ax * np.sin(Bx * self.x**2 + Cx) + Ay * np.sin(By * self.y**2 + Cy)

        # Adding uniform noise
        if uniform_noise:
            self.u0 += np.random.uniform(low=min_noise * np.ones_like(self.u0), high=max_noise * np.ones_like(self.u0))
    
    def set_BC(
            self,
            left: tuple = ["Neumann", 0.0],
            right: tuple = ["Neumann", 0.0],
            top: tuple = ["Neumann", 0.0],
            bottom: tuple = ["Neumann", 0.0],
            mode: str = "Neumann",
            value: float = 0.0
            ) -> None:
        """
        Set the boundary conditions (Neumann or Dirichlet).
        Default: Neumann with 0 value.
        - For rectangular domains each side can have its own condition (Neumann or Dirichlet).
        - For circular domains the BC is specified for all the entire circumference.
        - Neumann condition -> the value is the outward flux.
        - Dirichlet condition -> the value is the u value on the boundary.

        Parameters
        ----------
        left : tuple
            Used for rectangular spatial domains, left side BCs [str mode, float value].
        right : tuple
            Used for rectangular spatial domains, right side BCs [str mode, float value].
        top : tuple
            Used for rectangular spatial domains, top side BCs [str mode, float value].
        bottom : tuple
            Used for rectangular spatial domains, bottom side BCs [str mode, float value].
        mode : str
            Used for circular spatial domains, circumference BCs mode, "Neumann" | "Dirichlet".
        value : float
            Used for circular spatial domains, circumference BCs value (out normal flux for Neumann or function value for Dirichlet).
        
        Returns
        -------
        None
        """
        # Rectangular boundary
        if self.shape == "rectangle":
            self.left_mode, self.left_value = left
            self.right_mode, self.right_value = right
            self.top_mode, self.top_value = top
            self.bottom_mode, self.bottom_value = bottom
            modes = [self.left_mode, self.right_mode, self.top_mode, self.bottom_mode]
            sides = ["left", "right", "top", "bottom"]
            for mode, side in zip(modes, sides):
                if mode not in ["Neumann", "Dirichlet"]:
                    raise ValueError(f"Unrecognized {side} boundary mode '{mode}'.")
        
        # Circular boundary
        elif self.shape == "circle":
            self.boundary_mode = mode
            self.boundary_value = value

    def solve(
            self,
            t0: float,
            tN: float,
            dt: float,
            n_samples: int,
            seed: int,
            snapshot_times: Set[float] = None,
            vmin: float = None,
            vmax: float = None,
            cmap: str = "inferno",
            figsize: tuple = (3.5, 3.5)
            ) -> None:
        """
        Solve the system (numerical simulation) on the domain points and produce a trajectory, updating the object state.

        Parameters
        ----------
        t0 : float
            Initial time value.
        tN : float
            Final time value.
        dt : float
            Time step.
        n_samples : int
            Simulation samples to store along the timeline.
        seed : int
            Seed for the uniform at random sampling of the `n_samples` points along the timeline.
        snapshot_times : set
            Set of time values on which to store the full grid of computed solution values.
        vmin : float
            Minimum value for visualization.
        vmax : float
            Maximum value for visualization.
        cmap : str
            Color map for visualization.
        figsize : tuple
            Figure size for visualization.
        
        Returns
        -------
        None
        """
        # Cell variable for the solution field
        rho = CellVariable(name="rho", mesh=self.mesh)

        # Set the initial field
        rho.setValue(self.u0)

        # Set the BCs (Neumann or Dirichlet)
        ## Rectangular domain
        if self.shape == "rectangle":
            modes = [self.left_mode, self.right_mode, self.top_mode, self.bottom_mode]
            values = [self.left_value, self.right_value, self.top_value, self.bottom_value]
            meshFaces = [self.mesh.facesLeft, self.mesh.facesRight, self.mesh.facesTop, self.mesh.facesBottom]
            for mode, value, meshFaces in zip(modes, values, meshFaces):
                if mode == "Neumann":
                    rho.faceGrad.constrain(value, meshFaces)
                elif mode == "Dirichlet":
                    rho.constrain(value, meshFaces)
        ## Circular domain
        elif self.shape == "circle":
            if self.boundary_mode == "Neumann":
                rho.faceGrad.constrain(self.boundary_value, self.mesh.exteriorFaces)
            elif self.boundary_mode == "Dirichlet":
                rho.constrain(self.boundary_value, self.mesh.exteriorFaces)

        # Time instants to simulate (the timeline)
        timeline = np.arange(start=t0, stop=tN, step=dt)

        # Snapshot times
        if snapshot_times is None:
            snapshot_times = []

        # Simulation visualization options
        if vmin is None or vmax is None:
            u0min = self.u0.min()
            u0max = self.u0.max()
            margin = (u0max - u0min) / 6
            datamin = u0min - margin
            datamax = u0max + margin
        else:
            datamin = vmin
            datamax = vmax
        if self.shape == "rectangle":
            viewer = Viewer(vars=rho, cmap=cmap, datamin=datamin, datamax=datamax)
            fig = plt.gcf()
            fig.set_size_inches(figsize[0], figsize[1])

        # Velocity field
        velocity = FaceVariable(mesh=self.mesh, rank=1)

        # Source field
        source_term = CellVariable(mesh=self.mesh, rank=0)
        implicit_source_term = CellVariable(mesh=self.mesh, rank=0)

        # Initialize a trajectory object
        if self.shape == "rectangle":
            self.trajectory = Trajectory(x_coords=self.x, y_coords=self.y, nt=len(timeline), dt=dt, n_samples=n_samples, snapshot_times=snapshot_times, shape=self.shape, nx=self.mesh.nx, ny=self.mesh.ny, dx=self.dx, seed=seed)
        else: # "circle"
            self.trajectory = Trajectory(x_coords=self.x, y_coords=self.y, nt=len(timeline), dt=dt, n_samples=n_samples, snapshot_times=snapshot_times, shape=self.shape, seed=seed)

        # Run simulation
        for i, t in enumerate(timeline):
            ## Instanciate the velocity field
            v_value = self.v(self.x_faces, self.y_faces, t)

            ## Instanciate the source fields
            s_value = self.s(self.x, self.y, t)
            i_s_value = self.i_s(rho.value.copy())

            ## Update velocity and source variables
            velocity.setValue(v_value)
            source_term.setValue(s_value)
            implicit_source_term.setValue(i_s_value)

            ## Update the trajectory by appending the current frame
            self.trajectory.append(t=t, f_snapshot=rho.value.copy(), velocity_snapshot=self.v(self.x, self.y, t), source_snapshot=s_value)

            ## Simulation visualization
            if self.shape == "rectangle":
                viewer.plot()

            ## Simulation step
            eq = TransientTerm() + ConvectionTerm(coeff=velocity) + implicit_source_term - source_term - DiffusionTerm(coeff=self.D)
            eq.solve(var=rho, dt=dt)

    @classmethod
    def residual(
        cls,
        du: torch.Tensor,
        d2u: torch.Tensor,
        D: torch.Tensor = None,
        velocity: torch.Tensor = None,
        source: torch.Tensor = None,
        implicit_source: str = None,
        u: torch.Tensor = None,
        A: torch.Tensor = None,
        B: torch.Tensor = None
        ) -> torch.Tensor:
        """
        Compute the residual for the governing equation of the system.

        Parameters
        ----------
        u : torch.Tensor
            u field.
        du : torch.Tensor
            u gradient field.
        d2u : torch.Tensor
            u Hessian field.
        vx : torch.Tensor
            x-components of the velocity.
        vy : torch.Tensor
            y-components of the velocity.
        D : float
            Diffusion coefficient.
        source : torch.Tensor
            Source values; if None, it means that the source is implicit.
        implicit_source : str
            AllenCahn | logistic | Arrhenius.
        A : float
            Implicit source parameter.
        B : float
            Implicit source parameter.
        
        if implicit_source == "AllenCahn":\n
            implicit_source_term = A * (u^3 - u)\n
        elif implicit_source == "logistic":\n
            implicit_source_term = A * u^2 - B * u\n
        elif implicit_source == "Arrhenius":\n
            implicit_source_term = A * e^(- B / u)\n
        else:\n
            implicit_source_term = 0.0

        Returns
        -------
        torch.Tensor
            The residual value.
        """
        dux = du[:, 0]
        duy = du[:, 1]
        dut = du[:, 2]

        uxx = d2u[:, 0] # d2u[:, 0, 0]
        uyy = d2u[:, 1] # d2u[:, 1, 1]
        # utt = d2u[:, 2, 2]

        if velocity is None:
            vx = 0.0
            vy = 0.0
        else:
            vx = velocity[:, 0]
            vy = velocity[:, 1]

        if D is None:
            diffusion_term = 0.0
        else:
            diffusion_term = D * (uxx + uyy)

        if source is None:
            source_term = 0.0
        else:
            source_term = source

        if implicit_source == "AllenCahn":
            implicit_source_term = A * (u ** 3 - u)
        elif implicit_source == "logistic":
            implicit_source_term = A * u ** 2 - B * u
        elif implicit_source == "Arrhenius":
            implicit_source_term = A * torch.exp(- B / u)
        else:
            implicit_source_term = 0.0
        return dut + vx * dux + vy * duy + implicit_source_term - source_term - diffusion_term