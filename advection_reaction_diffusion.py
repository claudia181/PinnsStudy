"""
advection_reaction_diffusion.py
===========

This module implements the logic for the ARD PDE class.

Spatio-temporal domain:
- Time-dependent
- 2-dimentional spatial domain

Functions:
- make_source: Returns a source function.
- make_velocity: Returns a velocity function.

Classes:
- AdvectionReactionDiffusion: Implements the ARD PDE logic and methods required by the interface module pde_utils.py.
"""

import torch
from fipy import CellVariable, Grid2D, Gmsh2D, TransientTerm, ConvectionTerm, ImplicitSourceTerm, DiffusionTerm, Viewer, FaceVariable
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any, List, Tuple
    
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
    Build a Gaussian source (real-valued) function.

    Consider G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) ).

    Parameters
    ----------
    mode : str
        Source function type identifier:
        - constant: s(x, y) = amp * G(x, y)
        - decay: s(x, y, t)  = amp * e^(-delta t) * G(x, y)
        - oscillate: s(x, y, t)  = amp * sin((2 pi/period) t) * G(x, y)
        - temporary: amp * G(x, y) * (t < period)
    sigma : float
        G parameter.
    center : tuple
        G parameter (xc, yc).
    amp : float
        For Gaussian-type sources: amp * (G(x, y) * something).
    delta : float
        For mode = "decay", the decay rate: something = e^(- delta * t).
    period : float
        For mode = "oscillate": something = sin(2*pi/period * t).

    Returns
    -------
    Callable
        A Gaussian/Logistic/Arrhenius source function s(x, y, t, u).
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

    def G(x, y):  # Gaussian spot
        return np.exp(- ((x - xc)**2 + (y - yc)**2)/(2 * sigma**2))

    if mode == "constant":

        def source(x, y, t = None, u = None):
            return amp * G(x, y)
        
    elif mode == "decay":

        def source(x, y, t, u = None):
            return amp * np.exp(- delta * t) * G(x, y)
        
    elif mode == "oscillate":
        w = 2*np.pi/period

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
        raise ValueError(f"Argument 'mode' must be 'constant'|'decay'|'oscillate'|'temporary', not {mode}.")

    return source

def make_velocity(field: str = "rotation_expansion", **p: Any) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Build a (vector-valued) velocity function.

    v(x,y,t) = [v_x(x,y,t), v_y(x,y,t)],
        - v_x(x,y,t) = -a(t)*y + b(t)*x
        - v_y(x,y,t) = a(t)*x + b(t)*y
        - a(t) = alpha | alpha * sin(omega_a*t) | alpha * e^(-gamma_a*t)
        - b(t) = beta | beta * sin(omega_b*t) | beta * e^(-gamma_b*t)

    Parameters
    ----------
    field : str
        For now only "rotation_expansion" available.
    **p : Any
        Additional keyword arguments:
        - alpha_mode : (str, default: "const") rotation mode in {"const", "sin", "exp"}
        - beta_mode : (str, default: "const") expansion mode in {"const", "sin", "exp"}
        - alpha : (float, default: 1.0) rotation weight
        - beta : (float, default: 0.0) radial expantion weight
        - omega_a : (float, default: 1.0) rotation frequency (only if alpha_mode = "sin")
        - omega_b : (float, default: 1.0) expansion frequency (only if beta_mode = "sin")
        - gamma_a : (float, default: 0.5) rotation decay factor (alpha_mode = "exp")
        - gamma_b : (float, default: 0.5) expansion decay factor (beta_mode = "exp").
    """
    def law(mode = None, a = None, omega = None, gamma = None):
        # mode: scheduling over time
        # a: weight
        # omega: frequency (mode = "sin")
        # gamma: decay factor (mode = "exp")
        if a is None: a = 0.0
        if omega is None: omega = 0.0
        if gamma is None: gamma = 0.0
        if mode is None: mode = "const"
        if mode == "const":
            f  = lambda t: a
        elif mode == "sin":
            f  = lambda t: a * np.sin(omega * t)
        elif mode == "cos":
            f  = lambda t: a * np.cos(omega * t)
        elif mode == "exp":
            f  = lambda t: a * np.exp(- gamma * t)
        else:
            raise ValueError(f"mode must be const|sin|exp, not {mode}.")
        return f

    if field == "rotation_expansion":
        # defaults
        alpha = p.get("alpha", 1.0)                 # rotation weight
        beta = p.get("beta", 0.0)                   # radial expansion weight
        alpha_mode = p.get("alpha_mode", "const")   # rotation mode
        beta_mode = p.get("beta_mode",  "const")    # expansion mode
        omega_a = p.get("omega_a", 1.0)             # rotation frequency (rotation mode = "sin")
        omega_b = p.get("omega_b", 1.0)             # expansion frequency (expansion mode = "sin")
        gamma_a = p.get("gamma_a", 0.5)             # rotation decay factor (rotation mode = "exp")
        gamma_b = p.get("gamma_b", 0.5)             # expansion decay factor (expansion mode = "exp")

        a = law(mode=alpha_mode, a=alpha, omega=omega_a, gamma=gamma_a) # rotation time law
        b = law(mode=beta_mode, a=beta, omega=omega_b, gamma=gamma_b) # radial expantion time law

        def v(x, y, t):
            a_t = a(t); b_t = b(t)
            return np.array([- a_t * y + b_t * x, a_t * x + b_t * y])

        return v

    raise ValueError(f"Unknown field '{field}'")


# ===================================== AdvectionReactionDiffusion equation class =====================================
class AdvectionReactionDiffusion:
    """
    Class representing an advection-reaction-diffusion PDE.

    Attributes
    ----------
    v : Callable
        Vector valued velocity function.
    s : Callable
        Real valued source function.
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
        List of v values (one item per time instant).
    source : list
        List of s values (one item per time instant).
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
            implicit_source: Callable = None,#implicit_source: str = None,
            #A: float = None,
            #B: float = None,
            diffusion_coeff: float = None
            ):
        """
        Constructor.

        Parameters
        ----------
        velocity : Callable
            Velocity function.
        source : Callable
            Source function.
        implicit_source : str
            Logistic, Arrhenius or AllenCahn.
        A : float
            Implicit source parameter.
        B : float
            Implicit source parameter.

            - if implicit_source == "AllenCahn":\n
                implicit_source_term = A * (u^3 - u)\n
            - elif implicit_source == "logistic":\n
                implicit_source_term = A * u^2 - B * u\n
            - elif implicit_source == "Arrhenius":\n
                implicit_source_term = A * np.exp(- B / u)\n
            - else:\n
                implicit_source_term = 0.0
        
        diffusion_coeff : float
            The diffusion coefficient of the PDE (default: 0).
        """

        if velocity is None:
            self.v = make_velocity(alpha_mode="const", beta_mode="const", alpha=1.0, beta=0.0)
        else:
            self.v = velocity

        if source is None:
            self.s = make_source(mode="constant", amp=0.0)
            #self.implicit_source = False
        else:
            self.s = source
            #self.implicit_source = implicit_source
        
        #if A is None:
        #    A = 1.0
        #if B is None:
        #    B = 1.0

        #if implicit_source is None:
        #    implicit_source = ""
        #    
        #self.implicit_source = {
        #    "name": implicit_source,
        #    "A": A,
        #    "B": B
        #}
        if implicit_source is None:
            self.i_s = make_source(mode="constant", amp=0.0)
            #self.implicit_source = False
        else:
            self.i_s = implicit_source
        
        if diffusion_coeff is None:
            self.D = 0.0
        else:
            self.D = diffusion_coeff
        
        self.x, self.y = None, None
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None
        self.mesh = None

        self.t = None
        self.u0 = None
        self.u, self.du, self.d2u = None, None, None
        self.velocity = None
        self.source = None
        self.left_mode, self.left_value = None, None
        self.right_mode, self.right_value = None, None
        self.top_mode, self.top_value = None, None
        self.bottom_mode, self.bottom_value = None, None

        self.boundary_mode = None
        self.boundary_value = None

        self.shape = None
    
    def set_spatial_points(self, 
            mode: str, 
            x_range: tuple = None, 
            y_range: tuple = None, 
            dx: float = None, 
            dy: float = None,
            cell_size: float = None,
            radius: float = None
        ) -> None:
        """
        Set the spatial (2D) domain.

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
        if mode == "rectangle":
            self.xmin, self.xmax = x_range
            self.ymin, self.ymax = y_range

            nx = int(round((self.xmax - self.xmin) / dx))
            ny = int(round((self.ymax - self.ymin) / dy))

            self.mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
            xc, yc = self.mesh.cellCenters
            self.x = xc + self.xmin
            self.y = yc + self.ymin
            xf, yf = self.mesh.faceCenters
            self.x_faces = xf + self.xmin
            self.y_faces = yf + self.ymin
        elif mode == "circle":
            # 1. Generate the unstructured circular mesh
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

            # 2. Update coordinate references
            # cellCenters returns (2, n_cells)
            self.x, self.y = self.mesh.cellCenters
            # faceCenters returns (2, n_faces)
            self.x_faces, self.y_faces = self.mesh.faceCenters

            # 3. Update domain bounds for the Viewer
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
        u0 : np.ndarray
            Initial values.
        centers : list
            Centers of the Gaussians.
        amps : list
            Amplitudes, one for each center, regulate the height of each Gaussian.
        sigmas : list
            Regulate the width of each Gaussian.
        
        Returns
        -------
        None
        """
        if u0 is not None:
            self.u0 = u0 * np.ones_like(self.x)
        else:
            self.u0 = np.zeros_like(self.x)

        if gaussian:
            def normal(x0, y0, sigma = 0.1, amp = 1.0):
                return amp * np.exp(-((self.x - x0)**2 + (self.y - y0)**2) / (2 * sigma**2))

            if centers is None:
                centers = []
                amps = []
                sigmas = []
            else:
                if amps is None or amps == []: amps = [1.0 for _ in centers]
                if sigmas is None or sigmas == []: sigmas = [0.1 for _ in centers]

            for center, amp, sigma in zip(centers, amps, sigmas):
                self.u0 += normal(x0=center[0], y0=center[1], sigma=sigma, amp=amp)
        
        if periodic_circles:
            self.u0 += A * np.sin(B * np.sqrt(Cx * self.x**2 + Cy * self.y**2) + D) # concentric circles
        if periodic_valleys:
            self.u0 += A * np.sin(B * (self.x * self.y)) # circle^-1
        if periodic_stripes:
            self.u0 += A * np.sin(Bx * self.x + By * self.y) # stripes
        if periodic_grid:
            self.u0 += Ax * np.sin(Bx * self.x**2 + Cx) + Ay * np.sin(By * self.y**2 + Cy)
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

        Parameters
        ----------
        left : tuple
            Left side BCs, [str mode, float value].
        right : tuple
            Right side BCs, [str mode, float value].
        top : tuple
            Top side BCs, [str mode, float value].
        bottom : tuple
            Bottom side BCs, [str mode, float value].
        mode : str
            Circumference BCs mode, "Neumann" | "Dirichlet".
        value : float
            Circumference BCs value (out normal value or function value).
        
        Returns
        -------
        None
        """
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
                
        elif self.shape == "circle":
            self.boundary_mode = mode
            self.boundary_value = value

    def solve(
            self,
            t0: float = 0.0,
            tN: float = 0.0,
            dt: float = 1.0,
            snapshots: set = None,
            n_snapshots: int = None,
            snapshot_start: float = 0.0,
            vmin: float = None,
            vmax: float = None,
            cmap: str = "inferno",
            figsize: tuple = (3.5, 3.5)
            ) -> None:
        """
        Solve the PDE on the domain points, updating the object state.

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
        #if self.implicit_source["name"] == "CahnHiliard":
        #    rho = CellVariable(name="rho", mesh=self.mesh, hasOld=True)
        #else:
        rho = CellVariable(name="rho", mesh=self.mesh)  
        mu = CellVariable(name="mu", mesh=self.mesh)

        rho.setValue(self.u0)  # initial density

        if self.shape == "rectangle":
            modes = [self.left_mode, self.right_mode, self.top_mode, self.bottom_mode]
            values = [self.left_value, self.right_value, self.top_value, self.bottom_value]
            meshFaces = [self.mesh.facesLeft, self.mesh.facesRight, self.mesh.facesTop, self.mesh.facesBottom]

            for mode, value, meshFaces in zip(modes, values, meshFaces):
                if mode == "Neumann":
                    rho.faceGrad.constrain(value, meshFaces)
                elif mode == "Dirichlet":
                    rho.constrain(value, meshFaces)
        elif self.shape == "circle":
            if self.boundary_mode == "Neumann":
                rho.faceGrad.constrain(self.boundary_value, self.mesh.exteriorFaces)
            elif self.boundary_mode == "Dirichlet":
                rho.constrain(self.boundary_value, self.mesh.exteriorFaces)

        times = np.arange(start=t0, stop=tN, step=dt)
        times2 = [t for t in times if t >= snapshot_start]

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
            #if self.implicit_source["name"] == "CahnHiliard":
            #    viewer = Viewer(vars=mu, cmap=cmap, datamin=datamin, datamax=datamax)
            #else:
            viewer = Viewer(vars=rho, cmap=cmap, datamin=datamin, datamax=datamax)
            fig = plt.gcf()
            fig.set_size_inches(figsize[0], figsize[1])

        if snapshots is None:
            snapshots = []

        if n_snapshots is None:
            n_snapshots = 1

        snapshot_frequency = max(int(round((len(times2) / n_snapshots))), 1)
    
        u = [None, self.u0.copy(), self.u0.copy()]

        velocity = FaceVariable(mesh=self.mesh, rank=1)
        source_term = CellVariable(mesh=self.mesh, rank=0)
        implicit_source_term = CellVariable(mesh=self.mesh, rank=0)
        
        self.u = []
        self.du = []
        self.d2u = []
        self.t = []
        self.velocity = []
        self.source = []

        for i, t in enumerate(times):
            v = self.v(self.x_faces, self.y_faces, t)
            s = self.s(self.x, self.y, t, rho.value.copy())
            i_s = self.i_s(self.x, self.y, t, rho.value.copy())

            velocity.setValue(v)
            source_term.setValue(s)
            implicit_source_term.setValue(i_s)

            #if self.implicit_source:
            #    source_term = ImplicitSourceTerm(coeff=source)
            #else:
            #    source_term = source
            
            u[-1] = rho.value.copy()
            if self.shape == "rectangle":
                viewer.plot()
            if i % snapshot_frequency == 0 or (snapshots != [] and abs(t - np.array(list(snapshots))).min() < 1e-2):
                if t >= snapshot_start:
                    self.t.append(t)
                    self.u.append(u[-1])

                # Compute derivatives
                du_dt = (u[-1] - u[-2]) / dt
                du_dx = rho.grad[0, :].value
                du_dy = rho.grad[1, :].value
                du = np.stack([du_dx, du_dy, du_dt], axis=-1)  # (n_cells, 3)

                if i > 1:
                    d2u_dtdt = (u[-1] - 2 * u[-2] + u[-3]) / (dt**2)
                else:
                    d2u_dtdt = np.zeros_like(u[-1])

                d2u_dxdx = rho.grad[0, :].faceGrad.divergence.value
                d2u_dydy = rho.grad[1, :].faceGrad.divergence.value
                d2u_dxdy = rho.grad[0, :].grad[1, :].value

                if i > 0:
                    d2u_dxdt = (du_dx - du_dx_pred) / dt
                    d2u_dydt = (du_dy - du_dy_pred) / dt
                else:
                    d2u_dxdt = np.zeros_like(u[-1])
                    d2u_dydt = np.zeros_like(u[-1])
                
                du_dx_pred, du_dy_pred = du_dx, du_dy

                # assemble Hessian per cell
                d2u = np.stack([
                    np.stack([d2u_dxdx, d2u_dxdy, d2u_dxdt], axis=-1),
                    np.stack([d2u_dxdy, d2u_dydy, d2u_dydt], axis=-1),
                    np.stack([d2u_dxdt, d2u_dydt, d2u_dtdt], axis=-1)
                ], axis=-2)  # (n_cells, 3, 3)

                if t >= snapshot_start:
                    self.du.append(du)
                    self.d2u.append(d2u)
                    self.velocity.append(self.v(self.x, self.y, t))
                    self.source.append(s)
            
            #if self.implicit_source["name"] == "AllenCahn":
            #    implicit_source_term = self.implicit_source["A"] * (rho**2 - 1) * rho
            #elif self.implicit_source["name"] == "logistic":
            #    implicit_source_term = self.implicit_source["A"] * rho**2 - self.implicit_source["B"] * rho
            #elif self.implicit_source["name"] == "Arrhenius":
            #    implicit_source_term = self.implicit_source["A"] * np.exp(- self.implicit_source["B"] / rho)
            #else:
            #    implicit_source_term = 0.0

            eq = TransientTerm() + ConvectionTerm(coeff=velocity) + implicit_source_term - source_term - DiffusionTerm(coeff=self.D)
            eq.sweep(var=rho, dt=dt) # eq.solve(var=rho, dt=dt)
        
            u[-3] = u[-2]
            u[-2] = u[-1]

    @classmethod
    def residual(
        cls,
        u: torch.Tensor,
        du: torch.Tensor,
        d2u: torch.Tensor,

        #lap: torch.Tensor,
        #lap2: torch.Tensor,
        
        vx: torch.Tensor, 
        vy: torch.Tensor,
        D: float,
        source: torch.Tensor = None,
        implicit_source: str = None,
        A: float = 1.0,
        B: float = 1.0
        ) -> torch.Tensor:
        """
        Compute the residual.

        Parameters
        ----------
        u : torch.Tensor
            Function values.
        du : torch.Tensor
            1st derivative values.
        d2u : torch.Tensor
            2nd derivative values.
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
            implicit_source_term = A * np.exp(- B / u)\n
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

        #vx = velocity[:, 0]
        #vy = velocity[:, 1]

        diffusion_term = D * (uxx + uyy)

        if source is not None:
            source_term = source
        else:
            source_term = 0.0
        #if implicit_source == "CahnHiliard":
        #    return dut - 6*u*(dux**2 + duy**2) - (3*u**2-1)*lap + D*lap2

        #else:
        if implicit_source == "AllenCahn":
            implicit_source_term = A * (u**3 - u)
        elif implicit_source == "logistic":
            implicit_source_term = A * u**2 - B * u
        elif implicit_source == "Arrhenius":
            implicit_source_term = A * np.exp(- B / u)
        else:
            implicit_source_term = 0.0
        return dut + vx * dux + vy * duy + implicit_source_term - source_term - diffusion_term