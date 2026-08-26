"""
advection_reaction_diffusion.py
===========

This module implements the `AdvectionReactionDiffusion` class, representing advection-reaction-diffusion systems, giving methods to define a system and to solve (simulate it).

Spatio-temporal domain:
- Time-dependent
- 2-dimentional spatial domain

Classes:
- `AdvectionReactionDiffusion`
"""

import torch
from fipy import CellVariable, Grid2D, Gmsh2D, TransientTerm, ConvectionTerm, DiffusionTerm, Viewer, FaceVariable
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Set
from trajectory import Trajectory
from reaction_source import Source
from advection_velocity import Velocity

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
    i_s : Callable
        Implicit source function, i.e. using u.
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
            self.v = Velocity.null_velocity()
        else:
            self.v = velocity

        # Set the source field
        if source is None:
            self.s = Source.null_source()
        else:
            self.s = source
        if implicit_source is None:
            self.i_s = Source.null_source()
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