"""
advection_reaction_diffusion.py
===========

This module implements the logic for the ARD system class.

Spatio-temporal domain:
- Time-dependent
- 2-dimentional spatial domain

Functions:
- make_source: Returns a source function.
- velocity_field: Returns a velocity function.

Classes:
- AdvectionReactionDiffusion: Implements the ARD system logic and methods.
"""

import torch
from fipy import CellVariable, Grid2D, Gmsh2D, TransientTerm, ConvectionTerm, DiffusionTerm, Viewer, FaceVariable
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any, List, Tuple
from derivatives_computation import derivative
    
def make_source(
        sigma: float = 1.0,
        center: tuple = (0.0, 0.0),
        mode: str = "constant",
        amp: float = 0.0,
        delta: float = 0.1,
        period: float = 5.0,
        A: float = 0.0,
        B: float = 0.0
        ) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray], float]:
    """
    Build a scalar source function s(x, y, t, u) with values in R.

    Gaussian sources s(x, y, t) =
        - amp * f(t) * G(x, y)
        - G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) )
        - f(t) =
            - 1 (constant)
            - e^(- delta * t) (decay)
            - sin((2 pi / period) * t) (oscillate)
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
    Callable
        A source function s(x, y, t, u).
    """
    if sigma is None: sigma = 1.0
    if center is None: center = (0.0, 0.0)
    xc, yc = center
    if xc is None: xc = 0.0
    if yc is None: yc = 0.0
    if mode is None: mode = "constant"
    if amp is None: amp = 0.0
    if delta is None: delta = 0.1
    if period is None: period = 5.0
    #if A is None: A = 1.0
    #if B is None: B = 1.0

    def G(x, y): # Gaussian spot
        return np.exp(- ((x - xc) ** 2 + (y - yc) ** 2)/(2 * sigma ** 2))

    if mode == "constant":

        def source(x, y, t = None, u = None):
            return amp * G(x, y)
        
    elif mode == "decay":

        def source(x, y, t, u = None):
            return amp * np.exp(- delta * t) * G(x, y)
        
    elif mode == "oscillate":
        w = 2 * np.pi / period

        def source(x, y, t, u = None):
            return amp * np.sin(w * t) * G(x, y)
    
    elif mode == "temporary":
        def source(x, y, t, u = None):
            return amp * G(x, y) * (t < period)

    elif mode == "logistic":
        def source(x = None, y = None, t = None, u = None):
            if u is None:
                raise TypeError(f"Logistic source missing 1 required argument: 'u'.")
            return A * u**2 - B * u
    
    elif mode == "AllenCahn":
        def source(x = None, y = None, t = None, u = None):
            if u is None:
                raise TypeError(f"AllenCahn source missing 1 required argument: 'u'.")
            return A * (u**3 - u)
            
    elif mode == "Arrhenius":
        def source(x = None, y = None, t = None, u = None):
            if u is None:
                raise TypeError(f"Arrhenius source missing 1 required argument: 'u'.")
            return A * np.exp(- B / u)
        
    else:
        raise ValueError(f"Argument 'mode' must be 'constant'|'decay'|'oscillate'|'temporary'|'logistic'|'AllenCahn'|'Arrhenius', not {mode}.")

    return source


def constant_source(
        sigma: float = 1.0,
        center: tuple = (0.0, 0.0),
        amp: float = 0.0
        ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a constant-in-time source function:
    - s(x, y) = amp * G(x, y),
    - G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) ).

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    center : tuple
        Center (xc, yc) of the Gaussian.
    amp : float
        Constant multiplicative coefficient of the Gaussian.

    Returns
    -------
    Callable
        Constant-in-time source function s(x, y) = amp * G(x, y).
    """
    xc, yc = center

    def G(x, y): # Gaussian spot
        return np.exp(- ((x - xc) ** 2 + (y - yc) ** 2)/(2 * sigma ** 2))
    
    def source(x, y):
        return amp * G(x, y)
    
    return source

def decaying_surce(
        sigma: float = 1.0,
        center: tuple = (0.0, 0.0),
        amp: float = 0.0,
        delta: float = 0.1
        ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns an exponentially-decaying-in-time source function:
    - s(x, y, t) = amp * e^(- delta * t) * G(x, y),
    - G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) ).

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    center : tuple
        Center (xc, yc) of the Gaussian.
    amp : float
        Constant multiplicative coefficient of the Gaussian.
    delta : float
        The decay rate: e^(- delta * t).
    
    Returns
    -------
    Callable
        Exponentially-decaying-in-time source function s(x, y, t) = amp * e^(- delta * t) * G(x, y).
    """
    xc, yc = center
    
    def G(x, y): # Gaussian spot
        return np.exp(- ((x - xc) ** 2 + (y - yc) ** 2)/(2 * sigma ** 2))
        
    def source(x, y, t):
        return amp * np.exp(- delta * t) * G(x, y)

    return source

def oscillating_source(
        sigma: float = 1.0,
        center: tuple = (0.0, 0.0),
        amp: float = 0.0,
        period: float = 5.0
        ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns an oscillating-in-time source function:
    - s(x, y, t) = amp * sin((2 * pi / period) * t) * G(x, y),
    - G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) ).

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    center : tuple
        Center (xc, yc) of the Gaussian.
    amp : float
        Constant multiplicative coefficient of the Gaussian.
    period : float
        - sin(2 * pi / period * t).

    Returns
    -------
    Callable
        Oscillating-in-time source s(x, y, t) = amp * sin((2 * pi / period) * t) * G(x, y).
    """
    xc, yc = center
    
    def G(x, y): # Gaussian spot
        return np.exp(- ((x - xc) ** 2 + (y - yc) ** 2)/(2 * sigma ** 2))
        
    w = 2 * np.pi / period
    
    def source(x, y, t):
        return amp * np.sin(w * t) * G(x, y)

    return source

def temporary_source(
        sigma: float = 1.0,
        center: tuple = (0.0, 0.0),
        amp: float = 0.0,
        period: float = 5.0
        ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a limited-in-time (discontinuous-in-time) source function:
        - s(x, y, t) = amp * (t < period) * G(x, y)
        - G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) )

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    center : tuple
        Center (xc, yc) of the Gaussian.
    amp : float
        Constant multiplicative coefficient of the Gaussian.
    period : float
        Source limit time.

    Returns
    -------
    Callable
        Limited-in-time (discontinuous-in-time) source function s(x, y, t) = amp * (t < period) * G(x, y).
    """
    xc, yc = center
        
    def G(x, y): # Gaussian spot
        return np.exp(- ((x - xc) ** 2 + (y - yc) ** 2)/(2 * sigma ** 2))
        
    def source(x, y, t):
        return amp * G(x, y) * (t < period)
    
    return source

def logistic_source(
        A: float = 0.0,
        B: float = 0.0
        ) -> Callable[[np.ndarray], float]:
    """
    Returns a logistic source function:
        - s(u) = A * u^2 - B * u.

    Parameters
    ----------
    A : float
    B : float

    Returns
    -------
    Callable
        Source function s(u) = A * u^2 - B * u.
    """
    def source(u):
        return A * u ** 2 - B * u

    return source

def allen_cahn_source(
        A: float = 0.0
        ) -> Callable[[np.ndarray], float]:
    """
    Returns an Allen-Cahn-type source function:
        - s(u) = A * (u^3 - u)

    Parameters
    ----------
    A : float

    Returns
    -------
    Callable
        Source function s(u) = A * (u^3 - u).
    """
    def source(u):
        return A * (u ** 3 - u)

    return source

def arrhenius_source(
        A: float = 0.0,
        B: float = 0.0
        ) -> Callable[[np.ndarray], float]:
    """
    Returns an Arrhenius-type source function:
        - s(u) = A * e^(- B / u)

    Parameters
    ----------
    A : float
    B : float

    Returns
    -------
    Callable
        Source function s(u) = A * e^(- B / u).
    """
    def source(u = None):
        return A * np.exp(- B / u)

    return source

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
        # defaults
        rotation_weight = p.get("rotation_weight", 1.0)
        radial_expansion_weight = p.get("radial_expansion_weight", 0.0)
        rotation_mode = p.get("rotation_mode", "const")
        radial_expansion_mode = p.get("radial_expansion_mode",  "const")
        rotation_frequency = p.get("rotation_frequency", 1.0) # (rotation mode = "sin")
        radial_expansion_frequency = p.get("radial_expansion_frequency", 1.0) # (expansion mode = "sin")
        rotation_decay_factor = p.get("rotation_decay_factor", 1.0) # (rotation mode = "exp")
        radial_expansion_decay_factor = p.get("radial_expansion_decay_factor", 1.0) # (expansion mode = "exp")

        # rotation time law
        a = law(
                mode=rotation_mode,
                weight=rotation_weight,
                frequency=rotation_frequency,
                decay_factor=rotation_decay_factor
            )
        
        # radial expantion time law
        b = law(
                mode=radial_expansion_mode,
                weight=radial_expansion_weight, 
                frequency=radial_expansion_frequency, 
                decay_factor=radial_expansion_decay_factor
            )

        def v(x, y, t):
            a_t = a(t)
            b_t = b(t)
            return np.array([- a_t * y + b_t * x, a_t * x + b_t * y])

        return v

    raise ValueError(f"Unknown field '{field}'")

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
    implicit_source : str
        Implicit source, i.e. use u, "logistic", "Arrhenius" or "AllenCahn".
    A : float
        Implicit source parameter.
    B : float
        Implicit source parameter.
    D : float
        Diffusion coefficient.
    x : np.ndarray
        x coordinates of grid points.
    y : np.ndarray
        y coordinates of grid points.
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
    t : list
        Time instants values list.
    u0 : np.ndarray
        Initial conditions, i.e. initial (at t0) u values.
    u : list
        List of u values (one item per time instant).
    du : list
        List of du values (one item per time instant).
    d2u : list
        List of d2u values (one item per time instant).
    velocity : list
        Trajectory of the velocity vector field.
    source : list
        Trajectory of the source scalar field.
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
        source : Callable
            Source function defined on spatio-temporal coordinates.
        implicit_source : Callable
            Source function defined in terms of u.
        diffusion_coeff : float
            The diffusion coefficient (default: 0).
        """

        if velocity is None:
            self.v = velocity_field(
                rotation_mode="const", 
                radial_expansion_mode="const", 
                rotation_weight=1.0, 
                radial_expansion_weight=0.0
                )
        else:
            self.v = velocity

        if source is None:
            self.s = constant_source()
        else:
            self.s = source
        if implicit_source is None:
            self.i_s = constant_source()
        else:
            self.i_s = implicit_source
        
        if diffusion_coeff is None:
            self.D = 0.0
        else:
            self.D = diffusion_coeff
        
        # Spatial grid
        self.x, self.y = None, None
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None
        self.mesh = None
        self.dx, self.dt = None, None

        # Temporal grid
        self.t = None

        # Initial state
        self.u0 = None

        # Solution, 1st and 2nd derivative trajectories
        self.u, self.du, self.d2u = None, None, None

        # Velocity vector field trajectory
        self.velocity = None

        # Source trajectory
        self.source = None

        # Rectangular domain: sides modes and values
        self.left_mode, self.left_value = None, None
        self.right_mode, self.right_value = None, None
        self.top_mode, self.top_value = None, None
        self.bottom_mode, self.bottom_value = None, None

        # Circular domain: mode and value
        self.boundary_mode = None
        self.boundary_value = None

        # Domain shape
        self.shape = None
    
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
        - self.xmax, self.ymax.

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
        dy : float
            y-step.
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

            # Update domain bounds for the Viewer
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
        
        Returns
        -------
        None
        """
        # Base scalar field, which can be successively deformed by adding bumps, valleys
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
        - For rectangular domains each side can have its own condition (Neumann or Dirichlet)
        - For circular domains the BC is specified for all the entire circumference.

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
            t0: float = 0.0,
            tN: float = 0.0,
            dt: float = 1.0,
            snapshots: set = None,
            snapshot_start: float = 0.0,
            all_snapshots: bool = False,
            vmin: float = None,
            vmax: float = None,
            cmap: str = "inferno",
            figsize: tuple = (3.5, 3.5)
            ) -> None:
        """
        Solve the system on the domain points and produce a trajectory, updating the object state.
        Finite differences numerical simulation with backward Euler time discretization.

        Parameters
        ----------
        t0 : float
            Initial time value.
        tN : float
            Final time value.
        dt : float
            Time step.
        snapshots : set
            Set of time values on which to store the computed solution values.
        n_snapshots : int
            Number of snapshots to store.
        snapshot_start : float
            When to start to store snapshots.
        all_snapshots : bool
            If True, a snapshot for each step is taken.
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
        rho = CellVariable(name="rho", mesh=self.mesh)

        # Set the initial field
        rho.setValue(self.u0)

        # Set the BCs (Neumann or Dirichlet)
        # Rectangular domain
        if self.shape == "rectangle":
            modes = [self.left_mode, self.right_mode, self.top_mode, self.bottom_mode]
            values = [self.left_value, self.right_value, self.top_value, self.bottom_value]
            meshFaces = [self.mesh.facesLeft, self.mesh.facesRight, self.mesh.facesTop, self.mesh.facesBottom]
            for mode, value, meshFaces in zip(modes, values, meshFaces):
                if mode == "Neumann":
                    rho.faceGrad.constrain(value, meshFaces)
                elif mode == "Dirichlet":
                    rho.constrain(value, meshFaces)
        # Circular domain
        elif self.shape == "circle":
            if self.boundary_mode == "Neumann":
                rho.faceGrad.constrain(self.boundary_value, self.mesh.exteriorFaces)
            elif self.boundary_mode == "Dirichlet":
                rho.constrain(self.boundary_value, self.mesh.exteriorFaces)

        # Time instants to simulate
        timeline = np.arange(start=t0, stop=tN, step=dt)

        # Snapshots to store
        if snapshots is None:
            snapshots = [t for t in timeline if t >= snapshot_start]
        else:
            snapshots = [t for t in snapshots if t >= snapshot_start]

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
        
        # Trajectory data containers
        self.u = []
        self.du = []
        self.d2u = []
        self.t = []
        self.velocity = []
        self.source = []

        # Run simulation
        for i, t in enumerate(timeline):
            # Compute velocity field
            v = self.v(self.x_faces, self.y_faces, t)

            # Compute source fields
            s = self.s(self.x, self.y, t, rho.value.copy())
            i_s = self.i_s(self.x, self.y, t, rho.value.copy())

            # Update variables
            velocity.setValue(v)
            source_term.setValue(s)
            implicit_source_term.setValue(i_s)

            # Simulation visualization
            if self.shape == "rectangle":
                viewer.plot()
            
            # Store simulation data (snapshots)
            if all_snapshots or (snapshots != [] and abs(t - np.array(list(snapshots))).min() < 1e-2) and t >= snapshot_start:
                self.t.append(t)
                self.u.append(rho.value.copy())

                # Create the gradient vector
                #du = np.stack([du_dx, du_dy, du_dt], axis=-1)  # (n_cells, 3)

                # Create the Hessian matrix
                #d2u = np.stack([
                #    np.stack([d2u_dxdx, d2u_dxdy, d2u_dxdt], axis=-1),
                #    np.stack([d2u_dxdy, d2u_dydy, d2u_dydt], axis=-1),
                #    np.stack([d2u_dxdt, d2u_dydt, d2u_dtdt], axis=-1)
                #], axis=-2)  # (n_cells, 3, 3)

                # Store simulation data
                self.velocity.append(self.v(self.x, self.y, t))
                self.source.append(s)

            # Simulation step
            eq = TransientTerm() + ConvectionTerm(coeff=velocity) + implicit_source_term - source_term - DiffusionTerm(coeff=self.D)
            eq.solve(var=rho, dt=dt)

        self.u = torch.stack([torch.from_numpy(u_snapshot) for u_snapshot in self.u])
        self.du = derivative(f=self.u, dx=self.dx, dt=self.dt, order=1, method="central")
        self.d2u = derivative(f=self.u, dx=self.dx, dt=self.dt, order=2, method="central")

    @classmethod
    def residual(
        cls,
        u: torch.Tensor,
        du: torch.Tensor,
        d2u: torch.Tensor,
        v: torch.Tensor,
        D: float,
        source: torch.Tensor = None,
        implicit_source: str = None,
        A: float = 1.0,
        B: float = 1.0
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

        uxx = d2u[:, 0, 0]
        uyy = d2u[:, 1, 1]
        # utt = d2u[:, 2, 2]

        vx = v[:, 0]
        vy = v[:, 1]

        diffusion_term = D * (uxx + uyy)

        if source is not None:
            source_term = source
        else:
            source_term = 0.0
        if implicit_source == "AllenCahn":
            implicit_source_term = A * (u**3 - u)
        elif implicit_source == "logistic":
            implicit_source_term = A * u**2 - B * u
        elif implicit_source == "Arrhenius":
            implicit_source_term = A * torch.exp(- B / u)
        else:
            implicit_source_term = 0.0
        return dut + vx * dux + vy * duy + implicit_source_term - source_term - diffusion_term