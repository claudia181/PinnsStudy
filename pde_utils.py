"""
pde_utils.py
===========

This module is an interface for using the various implemented PDEs. 
It provides a uniform way to interact with any PDE class and performs some error checks, 
encapsulating the "burocratic" part of the code, separating this latter from the logic/business code 
(that is specific of and located in each single PDEs class implementation).

Global lists of the keys of the PDE parameters that identify a particular PDE instantiation:
- ALLEN_CAHN_KEYS
- PENDULUM_KEYS
- ARD_KEYS
- PENDULUM_IC_KEYS
- ARD_IC_KEYS

Global list of values that the string type PDE parameters can assume.
- ADD_INFO

Utility functions to manage the various PDEs:
- n_pde_params
- n_ic_params
- key_str
- key_idx
- ic_key_str
- ic_key_idx

Classes:
- PDE: Class thought be instanciated with one of the implemented PDEs. It provides methods to 
set the spatial domain, set the initial conditions and the boundary conditions, 
solve the PDE, compute the residual (all these perform calls to methods of the specific PDE class).
"""

import torch
from allen_cahn import AllenCahn
from pendulum import Pendulum
from advection_reaction_diffusion import AdvectionReactionDiffusion, make_source, make_velocity
from typing import Any, Tuple

MAX_ARD = 100
MAX_AC = 5

ALLEN_CAHN_KEYS = ["lam", "xi"] + [f"xi{i+1}" for i in range(MAX_AC)] + ["force"]
PENDULUM_KEYS = ["m", "l", "g", "b", "A", "w",
                       "force"]
ARD_KEYS = ["alpha", "beta", "alpha_mode", "beta_mode", "omega_a", "omega_b", "gamma_a", "gamma_b",
            "sigma", "xc", "yc", "source_mode", "amp", "delta", "T", "A", "B",
            "D", "vx", "vy", "s", "implicit_source"] + \
                [f"sigma{i}" for i in range(MAX_ARD)] + \
                [f"xc{i}" for i in range(MAX_ARD)] + \
                [f"yc{i}" for i in range(MAX_ARD)] + \
                [f"source_mode{i}" for i in range(MAX_ARD)] + \
                [f"amp{i}" for i in range(MAX_ARD)] + \
                [f"delta{i}" for i in range(MAX_ARD)] + \
                [f"T{i}" for i in range(MAX_ARD)]
PENDULUM_IC_KEYS = ["u0", "du0"]
ARD_IC_KEYS = ["u0"] + \
                [f"sigma{i}" for i in range(MAX_ARD)] + \
                [f"xc{i}" for i in range(MAX_ARD)] + \
                [f"yc{i}" for i in range(MAX_ARD)] + \
                [f"amp{i}" for i in range(MAX_ARD)] + \
                ["A", "B", "D"] + \
                ["Ax", "Bx", "Cx"] + \
                ["Ay", "By", "Cy"] + \
                ["min_noise", "max_noise"] + \
                ["gaussian", "periodic_circles", "periodic_valleys", "periodic_stripes", "periodic_grid", "uniform_noise"]
ADD_INFO = ["constant", "decay", "oscillate", "temporary", "logistic", "AllenCahn", "CahnHiliard", "Arrhenius", "const", "sin", "cos", "exp"
            ]

def n_pde_params(pde: str) -> int:
    """
    Returns the number of keys of pde.
    """
    # ------------------- Allen-Cahn -------------------
    if pde == "Allen-Cahn":
        return len(ALLEN_CAHN_KEYS)
    # ------------------- Pendulum -------------------
    elif pde == "Pendulum":
        return len(PENDULUM_KEYS)
    # ------------------- Advection-Reaction-Diffusion -------------------
    elif pde == "Advection-Reaction-Diffusion":
        return len(ARD_KEYS)
    # --------------------------------------
    raise ValueError(f"PDE name '{pde}' not found.")

def n_ic_params(pde: str):
    """
    Returns the number of IC parameters of pde.
    """
    # ------------------- Allen-Cahn -------------------
    if pde == "Allen-Cahn":
        return 0
    # ------------------- Pendulum -------------------
    elif pde == "Pendulum":
        return len(PENDULUM_IC_KEYS)
    # ------------------- Advection-Reaction-Diffusion -------------------
    elif pde == "Advection-Reaction-Diffusion":
        return len(ARD_IC_KEYS)
    # --------------------------------------
    raise ValueError(f"PDE name '{pde}' not found.")
    
def key_str(key_idx: int, name: str) -> str:
    """
    Returns the key associated with the given index key_idx for the PDE identified by name.
    """
    # ------------------- Allen-Cahn -------------------
    if name == "Allen-Cahn":
        return ALLEN_CAHN_KEYS[key_idx]
    # ------------------- Pendulum -------------------
    elif name == "Pendulum":
        return PENDULUM_KEYS[key_idx]
    # ------------------- Advection-Reaction-Diffusion -------------------
    elif name == "Advection-Reaction-Diffusion":
        return ARD_KEYS[key_idx]
    # ------------------- Additional info -------------------
    elif name == "Additional info":
        return ADD_INFO[key_idx]
    # --------------------------------------
    raise ValueError(f"Name '{name}' not found.")

def key_idx(key_str: str, name: str) -> int:
    """
    Returns the index associated with the given key key_str for the PDE identified by name.
    """
    # ------------------- Allen-Cahn -------------------
    if name == "Allen-Cahn":
        for i in range(len(ALLEN_CAHN_KEYS)):
            if ALLEN_CAHN_KEYS[i] == key_str: return i
        raise ValueError(f"Key index not found for {key_str} ({name}).")
    # ------------------- Pendulum -------------------
    elif name == "Pendulum":
        for i in range(len(PENDULUM_KEYS)):
            if PENDULUM_KEYS[i] == key_str: return i
        raise ValueError(f"Key index not found for {key_str} ({name}).")
    # ------------------- Advection-Reaction-Diffusion -------------------
    elif name == "Advection-Reaction-Diffusion":
        for i in range(len(ARD_KEYS)):
            if ARD_KEYS[i] == key_str: return i
        raise ValueError(f"Key index not found for {key_str} ({name}).")
    # ------------------- Additional info -------------------
    elif name == "Additional info":
        for i in range(len(ADD_INFO)):
            if ADD_INFO[i] == key_str: return i
        raise ValueError(f"Error: pde_utils: key_idx: key_str not found for {key_str} ({name}).")
    # --------------------------------------
    raise ValueError(f"Eq. {name} not found.")

def ic_key_str(key_idx: int, name: str) -> str:
    """
    Returns the IC key associated with the given index key_idx for the PDE identified by name.
    """
    # ------------------- Allen-Cahn -------------------
    if name == "Allen-Cahn":
        raise ValueError(f"Eq. {name} has no IC information.")
    # ------------------- Pendulum -------------------
    elif name == "Pendulum":
        return PENDULUM_IC_KEYS[key_idx]
    # ------------------- Advection-Reaction-Diffusion -------------------
    elif name == "Advection-Reaction-Diffusion":
        return ARD_IC_KEYS[key_idx]
    # --------------------------------------
    raise ValueError(f"Eq. {name} not found.")

def ic_key_idx(key_str: str, name: str) -> int:
    """
    Returns the IC index associated with the given index key_str for the PDE identified by name.
    """
    # ------------------- Allen-Cahn -------------------
    if name == "Allen-Cahn":
        raise ValueError(f"Eq. {name} has no IC information.")
    # ------------------- Pendulum -------------------
    elif name == "Pendulum":
        for i in range(len(PENDULUM_IC_KEYS)):
            if PENDULUM_IC_KEYS[i] == key_str: return i
        raise ValueError(f"Key index not found for {key_str} ({name}).")
    # ------------------- Advection-Reaction-Diffusion -------------------
    elif name == "Advection-Reaction-Diffusion":
        for i in range(len(ARD_IC_KEYS)):
            if ARD_IC_KEYS[i] == key_str: return i
        raise ValueError(f"Key index not found for {key_str} ({name}).")
    # --------------------------------------
    raise ValueError(f"Eq. {name} not found.")

# ==================================== PDE class ====================================
class Pde:
    """
    Class representing one of the implemented PDE classes.

    Attributes
    ----------
    points : torch.Tensor
        Spatial coordinates.
    times : list
        Temporal coordinates (list of float).
    solution : list
        Solution values, one item for each time instant.
    der : list
        Derivatives values [u, du, d2u]; each of u, du and d2u is a list with one item for each time instant.
    name : str
        String identifier of the PDE.
    pde : AllenCahn|Pendulum|AdvectionReactionDiffusion
        Object representing the PDE.
    """
    # +++++++++++++++++++++++++++ CONSTRUCTOR +++++++++++++++++++++++++++
    def __init__(self, name: str, param_keys: list = [], param_values: list = [], device: str = "cpu"): #TODO: update the calls (param_keys, param_values instead of params)
        """
        Constructor.

        Parameters
        ----------
        name : str
            PDE string identifier.
        param_keys : list
            List of PDE parameters keys.
        param_values : list
            List of PDE parameters values.
        device : str
        """
        self.points = None
        self.times = None
        self.solution = None
        self.der = None
        self.name = name
        param_dict = {}
        for key_idx, value in zip(param_keys, param_values):
            key = key_str(key_idx, self.name)
            param_dict[key] = value

        # ------------------- Allen-Cahn -------------------
        if self.name == "Allen-Cahn":
            lam = param_dict.get("lam")
            xi  = param_dict.get("xi")
            if xi is None:
                xi = []
                for i in range(MAX_AC):
                    el = param_dict.get(f"xi{i+1}")
                    if el is not None:
                        xi.append(el)
            else:
                xi = [xi]

            self.pde = AllenCahn(lam=lam, force_params=xi, device=device)

        # ------------------- Pendulum -------------------
        elif self.name == "Pendulum":
            m = param_dict.get("m") # mass
            l = param_dict.get("l") # rode length
            g = param_dict.get("g") # gravity acceleration
            b = param_dict.get("b") # damping coefficient
            A = param_dict.get("A") # drive amplitude
            w = param_dict.get("w") # drive frequency

            self.pde = Pendulum(m=m, l=l, g=g, b=b, A=A, w=w)

        # ------------------- Advection-Reaction-Diffusion -------------------
        elif self.name == "Advection-Reaction-Diffusion":
            # Velocity function parameters
            alpha       = param_dict.get("alpha")       # rotation weight
            beta        = param_dict.get("beta")        # radial expansion weight
            alpha_mode  = param_dict.get("alpha_mode")  # rotation mode
            if alpha_mode is not None:
                alpha_mode = key_str(int(alpha_mode), "Additional info")
            beta_mode   = param_dict.get("beta_mode")   # expansion mode
            if beta_mode is not None:
                beta_mode = key_str(int(beta_mode), "Additional info")   # expansion mode
            omega_a     = param_dict.get("omega_a")     # rotation frequency (rotation mode = "sin")
            omega_b     = param_dict.get("omega_b")     # expansion frequency (expansion mode = "sin")
            gamma_a     = param_dict.get("gamma_a")     # rotation decay factor (rotation mode = "exp")
            gamma_b     = param_dict.get("gamma_b")     # expansion decay factor (expansion mode = "exp")

            velocity = make_velocity(
                alpha=alpha, beta=beta,
                alpha_mode=alpha_mode, beta_mode=beta_mode,
                omega_a=omega_a, omega_b=omega_b,
                gamma_a=gamma_a, gamma_b=gamma_b
                )

            # Source function parameters
            sigma = param_dict.get("sigma")
            xc = param_dict.get("xc")
            yc = param_dict.get("yc")
            period = param_dict.get("T")
            amp = param_dict.get("amp")
            delta = param_dict.get("delta")
            A = param_dict.get("A")
            B = param_dict.get("B")
            mode = param_dict.get("source_mode")
            if mode is not None:
                mode = key_str(int(mode), "Additional info")
                if mode == "logistic" or mode == "AllenCahn" or mode == "Arrhenius" or mode == "CahnHiliard":
                    implicit_source = mode
                    source = None
                else:
                    implicit_source = None
                    source = make_source(sigma=sigma, center=(xc, yc), mode=mode, amp=amp, delta=delta, period=period)
            else:
                implicit_source = None
                source = None
            
            if sigma is None and xc is None and yc is None and period is None and amp is None and delta is None:
                sources = []
                for i in range(MAX_ARD):
                    sigma = param_dict.get(f"sigma{i}")
                    xc = param_dict.get(f"xc{i}")
                    yc = param_dict.get(f"yc{i}")
                    period = param_dict.get(f"T{i}")
                    mode = param_dict.get(f"source_mode{i}")
                    if mode is not None:
                        mode = key_str(int(mode), "Additional info")
                        if mode == "logistic" or mode == "AllenCahn" or mode == "Arrhenius" or mode == "CahnHiliard":
                            raise ValueError(f"Invalid source_mode{i} '{mode}'.")
                    amp = param_dict.get(f"amp{i}")
                    delta = param_dict.get(f"delta{i}")
                    source = make_source(sigma=sigma, center=(xc, yc), mode=mode, amp=amp, delta=delta, period=period)
                    sources.append(source)
                def source_fn(x, y, t, u = None):
                    return sum([s(x, y, t, u) for s in sources])
                source = source_fn

            # Diffusion coefficient
            d = param_dict.get("D")

            self.pde = AdvectionReactionDiffusion(velocity=velocity, source=source, implicit_source=implicit_source, diffusion_coeff=d, A=A, B=B)

    # +++++++++++++++++++++++++++++ SET SPATIAL POINTS +++++++++++++++++++++++++++++
    def set_spatial_points(
            self,
            X: torch.Tensor = None,
            mode: str = None,
            x_range: tuple = None,
            y_range: tuple = None,
            dx: float = None,
            dy: float = None,
            cell_size: float = None,
            radius: float = None
            ) -> None:
        """
        Set the spatial domain points.

        Parameters
        ----------
        X : torch.Tensor
        mode : str
        x_range : tuple
        y_range : tuple
        dx : float
        dy : float
        cell_size : float
        radius : float

        Returns
        -------
        None
        """
        # ------------------- Allen-Cahn -------------------
        if self.name == "Allen-Cahn":
            if X is None:
                raise ValueError("Allen-Cahn requires 'X' argument.")
            self.pde.set_spatial_points(x=X[:, 0], y=X[:, 1])
            self.points = torch.stack([self.pde.x, self.pde.y], dim=1)

        # ------------------- Pendulum -------------------
        if self.name == "Pendulum":
            # Zero-dimentional spatial domain
            self.points = torch.stack([torch.tensor([0.0]), torch.tensor([0.0])], dim=1)

        # ------------------- Advection-Reaction-Diffusion -------------------
        elif self.name == "Advection-Reaction-Diffusion":
            if mode is None:
                raise ValueError("Advection-Reaction-Diffusion equation requires 'mode' information for spatial domain.")
            if mode == "rectangle":
                if x_range is None:
                    raise ValueError("Advection-Reaction-Diffusion equation requires 'x_range' information.")
                if y_range is None:
                    raise ValueError("Advection-Reaction-Diffusion equation requires 'y_range' information.")
                if dx is None:
                    raise ValueError("Advection-Reaction-Diffusion equation requires 'dx' information.")
                if dy is None:
                    raise ValueError("Advection-Reaction-Diffusion equation requires 'dy' information.")
                self.pde.set_spatial_points(mode=mode, x_range=x_range, y_range=y_range, dx=dx, dy=dy)
            elif mode == "circle":
                if cell_size is None:
                    raise ValueError("Advection-Reaction-Diffusion equation requires 'cell_size' information.")
                if radius is None:
                    raise ValueError("Advection-Reaction-Diffusion equation requires 'radius' information.")
                self.pde.set_spatial_points(mode=mode, cell_size=cell_size, radius=radius)
            self.points = torch.stack([torch.from_numpy(self.pde.x), torch.from_numpy(self.pde.y)], dim=1)
    
    def set_BC(self, bc_dict: dict) -> None:
        """
        Set the boundary conditions.

        Parameters
        ----------
        bc_dict : dict
            Boundary conditions dictionary.

        Returns
        -------
        None
        """
        # ------------------- Allen-Cahn -------------------
        # _

        # ------------------- Pendulum -------------------
        # _

        # ------------------- Advection-Reaction-Diffusion -------------------
        if self.name == "Advection-Reaction-Diffusion":
            self.pde.set_BC(**bc_dict)
    
    def set_IC(self, keys: list = [], values: list = []) -> None:
        """
        Set the boundary conditions.

        Parameters
        ----------
        keys : list
            Initial conditions parameters keys.
        values : list
            Initial conditions parameters vaues.

        Returns
        -------
        None
        """
        ic_dict = {}
        for key_idx, value in zip(keys, values):
            key = ic_key_str(key_idx, self.name)
            ic_dict[key] = value

        # ------------------- Allen-Cahn -------------------
        # Zero-dimentional temporal domain

        # ------------------- Pendulum -------------------
        if self.name == "Pendulum":
            u0  = ic_dict.get("u0")
            du0 = ic_dict.get("du0")
            self.pde.set_IC(u0=u0, du0=du0)

        # ------------------- Advection-Reaction-Diffusion -------------------
        elif self.name == "Advection-Reaction-Diffusion":
            if self.pde.x is None or self.pde.y is None:
                raise RuntimeError("Advection-Reaction-Diffusion equation requires spatial points initialization before setting the initial consitions.")
            u0 = ic_dict.get("u0", 0.0)
            gaussian = ic_dict.get("gaussian", False)
            periodic_circles = ic_dict.get("periodic_circles", False)
            periodic_valleys = ic_dict.get("periodic_valleys", False)
            periodic_stripes = ic_dict.get("periodic_stripes", False)
            periodic_grid = ic_dict.get("periodic_grid", False)
            uniform_noise = ic_dict.get("uniform_noise", False)

            centers = None
            amps = None
            sigmas = None
            if gaussian:
                centers = []
                amps = []
                sigmas = []
                for i in range(MAX_ARD):
                    centers.append((ic_dict.get(f"xc{i}"), ic_dict.get(f"yc{i}")))
                centers = [center for center in centers if center is not None and None not in center]
                for i, center in enumerate(centers):
                    amps.append(ic_dict.get(f"amp{i}"))
                    sigmas.append(ic_dict.get(f"sigma{i}"))

                amps = [amp for amp in amps if amp is not None]
                sigmas = [sigma for sigma in sigmas if sigma is not None and sigma]

                if centers == []: centers = None
                if amps == []: amps = None
                if sigmas == []: sigmas = None

            A = None
            B = None
            D = None
            Ax = None
            Bx = None
            Cx = None
            Ay = None
            By = None
            Cy = None
            min_noise = None
            max_noise = None

            if periodic_circles:
                A = ic_dict.get("A", 1.0)
                B = ic_dict.get("B", 1.0)
                Cx = ic_dict.get("Cx", 1.0)
                Cy = ic_dict.get("Cy", 1.0)
                D = ic_dict.get("D", 0.0)
            if periodic_valleys:
                A = ic_dict.get("A", 1.0)
                B = ic_dict.get("B", 1.0)
            if periodic_stripes:
                A = ic_dict.get("A", 1.0)
                Bx = ic_dict.get("Bx", 1.0)
                By = ic_dict.get("By", 1.0)
            if periodic_grid:
                Ax = ic_dict.get("Ax", 1.0)
                Ay = ic_dict.get("Ay", 1.0)
                Bx = ic_dict.get("Bx", 1.0)
                By = ic_dict.get("By", 1.0)
                Cx = ic_dict.get("Cx", 0.0)
                Cy = ic_dict.get("Cy", 0.0)
            if uniform_noise:
                min_noise = ic_dict.get("min_noise", 1.0)
                max_noise = ic_dict.get("max_noise", -1.0)

            self.pde.set_IC(
                u0=u0, 
                gaussian=gaussian, 
                periodic_circles=periodic_circles, 
                periodic_valleys=periodic_valleys, 
                periodic_stripes=periodic_stripes, 
                periodic_grid=periodic_grid, 
                uniform_noise=uniform_noise, 
                centers=centers, amps=amps, sigmas=sigmas, 
                A=A, B=B, D=D, 
                Ax=Ax, Ay=Ay, 
                Bx=Bx, By=By, 
                Cx=Cx, Cy=Cy, 
                min_noise=min_noise, max_noise=max_noise
            )

    # +++++++++++++++++++++++++++++ SOLVE METHOD +++++++++++++++++++++++++++++
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
        Solve the PDE.

        Parameters
        ----------
        t0 : float
            Initial time.
        tN : float
            Final time.
        dt : float
            Time step.
        snapshots : set
            Snapshots instants.
        n_snapshots : int
            Number of snapshots.
        snapshot_start : float
            When to start to store snapshots (only for ARD).
        vmin : float
            For visualization (ARD PDEs).
        vmax : float
            For visualization (ARD PDEs).
        cmap : str
            For visualization (ARD PDEs).
        figsize : tuple
            For visualization (ARD PDEs).

        Returns
        -------
        None
        """
        # ------------------- Allen-Cahn -------------------
        if self.name == "Allen-Cahn":
            if self.pde.x is None:
                raise ValueError("Missing spatial information.")
            self.pde.solve()
            self.times = [0.0]
            self.solution = [self.pde.u]
            self.der = [[self.pde.u], [self.pde.du], [self.pde.d2u]]

        # ------------------- Pendulum -------------------
        elif self.name == "Pendulum":
            if self.pde.u0 is None or self.pde.du0 is None:
                print("Warning: Pendulum system initialized with default initial consitions.")
                self.pde.set_IC()
            self.pde.solve(t0=t0, tN=tN, snapshots=snapshots, n_snapshots=n_snapshots)
            self.times = self.pde.t
            u = [torch.tensor([item]) for item in self.pde.u]
            du = [torch.tensor([item]) for item in self.pde.du]
            d2u = [torch.tensor([item]) for item in self.pde.d2u]
            self.solution = u
            self.der = [u, du, d2u]
        
        # ------------------- Advection-Reaction-Diffusion -------------------
        elif self.name == "Advection-Reaction-Diffusion":
            if self.pde.x is None:
                raise RuntimeError("Missing spatial information for Advection-Reaction-Diffusion equation.")
            if self.pde.u0 is None:
                print("Warning: Advection-Reaction-Diffusion system initialized with default initial consitions.")
                self.pde.set_IC()
            self.pde.solve(t0=t0, tN=tN, dt=dt, snapshots=snapshots, n_snapshots=n_snapshots, snapshot_start=snapshot_start, vmin=vmin, vmax=vmax, cmap=cmap, figsize=figsize)
            self.times = self.pde.t
            u = [torch.from_numpy(u_snapshot) for u_snapshot in self.pde.u]
            du = [torch.from_numpy(du_snapshot) for du_snapshot in self.pde.du]
            d2u = [torch.from_numpy(d2u_snapshot) for d2u_snapshot in self.pde.d2u]
            self.solution = u
            self.der = [u, du, d2u]
    
    # +++++++++++++++++++++++++++++ RESIDUAL COMPUTATION (class method) +++++++++++++++++++++++++++++
    @classmethod    
    def residual(
        cls,
        pde_name: str,
        u: torch.Tensor = None,
        du: torch.Tensor = None,
        d2u: torch.Tensor = None,
        lap: torch.Tensor = None,
        lap2: torch.Tensor = None,
        **kwargs: Any
        ) -> torch.Tensor:
        """
        Compute the PDE residual.

        Parameters
        ----------
        pde_name: str
            PDE string identifier.
        u : torch.Tensor
            Solution values.
        du : torch.Tensor
            1st derivative values.
        d2u : torch.Tensor
            2nd derivative values.
        lap : torch.Tensor
            Laplacian value.
        lap2 : torch.Tensor
            Laplacian of Laplacian value.
        **kwargs : Any
            Additional keywords arguments:
            - Allen-Cahn: kwargs keys = {force, lam};
            - Pendulum: kwargs keys = {force, m, l, g, b};
            - Advection-Reaction-Diffusion: kwargs = {vx, vy, D, s, implicit_source, A, B}.

        Returns
        -------
        torch.Tensor
            The residual.
        """
        # ------------------- Allen-Cahn -------------------
        if pde_name == "Allen-Cahn":
            force = kwargs.get("force")
            lam = kwargs.get("lam")

            if u is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs u.")
            if d2u is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs d2u.")           
            if force is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs force.")
            if lam is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs lam.")   

            return AllenCahn.residual(u=u, d2u=d2u, force=force, lam=lam)
        
        # ------------------- Pendulum -------------------
        elif pde_name == "Pendulum":
            force   = kwargs.get("force")
            m       = kwargs.get("m")
            l       = kwargs.get("l") # rode length
            g       = kwargs.get("g") # gravity acceleration
            b       = kwargs.get("b")

            if u is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs u.")
            if d2u is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs d2u.")
            if force is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs force.")
            if m is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs m.")
            if l is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs l.")
            if g is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs g.")
            if b is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs b.")

            return Pendulum.residual(u=u, du=du, d2u=d2u, force=force, m=m, l=l, g=g, b=b)
        
        # ------------------- Advection-Reaction-Diffusion -------------------
        elif pde_name == "Advection-Reaction-Diffusion":
            vx = kwargs.get("vx")
            vy = kwargs.get("vy")
            D = kwargs.get("D")
            s = kwargs.get("s")
            implicit_source = kwargs.get("implicit_source")
            A = kwargs.get("A")
            B = kwargs.get("B")

            if u is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs 'u'.")
            if du is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs 'du'.")
            if d2u is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs 'd2u'.")
            if vx is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs 'vx'.")
            if vy is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs 'vy'.")
            if D is None:
                raise ValueError(f"Eq. {pde_name} residual computation needs 'D'.")
            if -1 in implicit_source:
                implicit_source = None
            else:
                implicit_source = key_str(int(implicit_source[0].item()), "Additional info")
            if A is None:
                if implicit_source is not None:
                    raise ValueError(f"Eq. {pde_name} residual computation needs 'A'.")
                else:
                    A = 1.0
            if B is None:
                if implicit_source is not None:
                    raise ValueError(f"Eq. {pde_name} residual computation needs 'B'.")
                else:
                    B = 1.0
            return AdvectionReactionDiffusion.residual(u=u, du=du, d2u=d2u, vx=vx, vy=vy, D=D, source=s, implicit_source=implicit_source, A=A, B=B, lap=lap, lap2=lap2)
    
    # +++++++++++++++++++++++++++++ RESIDUAL REQUIRED DERIVATIVES (class method) +++++++++++++++++++++++++++++
    @classmethod
    def residual_required_derivatives(cls, pde_name: str) -> list:
        """
        Return the order of the required derivatives for the residual computation of the PDE.

        Parameters
        ----------
        pde_name : str
            PDE string identifier.
        
        Returns
        -------
        list
            Integer list of the order of the required derivatives for the residual computation (including the 0-order).
        """
        # ------------------- Allen-Cahn -------------------
        if pde_name == "Allen-Cahn":
            return [0, 2]
        
        # ------------------- Pendulum -------------------
        elif pde_name == "Pendulum":
            return [0, 1, 2]
        
        # ------------------- Advection-Reaction-Diffusion -------------------
        elif pde_name == "Advection-Reaction-Diffusion":
            return [0, 1, 2]
    
    # +++++++++++++++++++++++++++++ ADDITIONAL INFO +++++++++++++++++++++++++++++
    def additional_info(self, t: float, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        t : float
            Time.
        **kwargs : Any
            Additional keywords arguments:
            - Allen-Cahn: kwargs keys = {x, y};
            - Pendulum: kwargs keys = {};
            - Advection-Reaction-Diffusion: kwargs keys = {x, y}.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            <keys indexes, values>.
        """
        # ------------------- Allen-Cahn -------------------
        if self.name == "Allen-Cahn":
            force = self.pde.force
            if force is None:
                x = kwargs.get("x")
                y = kwargs.get("y")
                if x is None:
                    raise ValueError(f"Missing 'x' information.")
                if y is None:
                    raise ValueError(f"Missing 'y' information.")
                force = self.pde.compute_force(x, y)
            lam = torch.flatten(self.pde.lam.repeat(len(force), 1)) #TODO: make lam a scalar in AC and adjust
            return torch.tensor([key_idx("force", self.name), key_idx("lam", self.name)]), torch.stack([force, lam], dim=1)
        
        # ------------------- Pendulum -------------------
        elif self.name == "Pendulum":
            force = self.pde.force[t]
            if force is None:
                force = torch.tensor([self.pde.compute_force(t)])
            else:
                force = torch.tensor([force])
            m = torch.tensor([self.pde.m])
            l = torch.tensor([self.pde.l])
            g = torch.tensor([self.pde.g])
            b = torch.tensor([self.pde.b])
            return torch.tensor([key_idx("force", self.name), key_idx("m", self.name), key_idx("l", self.name), key_idx("g", self.name), key_idx("b", self.name)]), torch.stack([force, m, l, g, b], dim=1)
        
        # ------------------- Advection-Reaction-Diffusion -------------------
        elif self.name == "Advection-Reaction-Diffusion":
            if self.pde.velocity is None:
                x = kwargs.get("x")
                y = kwargs.get("y")
                if x is None:
                    raise ValueError(f"Missing 'x' information.")
                if y is None:
                    raise ValueError(f"Missing 'y' information.")
                velocity = self.pde.compute_velocity(x, y, t)
            else:
                velocity = self.pde.velocity[t]
            vx = torch.from_numpy(velocity[0])
            vy = torch.from_numpy(velocity[1])

            if self.pde.source is not None:
                s = torch.tensor(self.pde.source[t])
            else:
                x = kwargs.get("x")
                y = kwargs.get("y")
                if x is None:
                    raise ValueError(f"Missing 'x' information.")
                if y is None:
                    raise ValueError(f"Missing 'y' information.")
                s = self.pde.compute_source(x=x, y=y, t=t)

            D = torch.flatten(torch.tensor(self.pde.D).repeat(len(vx), 1))
            A = torch.flatten(torch.tensor(self.pde.implicit_source["A"]).repeat(len(vx), 1))
            B = torch.flatten(torch.tensor(self.pde.implicit_source["B"]).repeat(len(vx), 1))

            if self.pde.implicit_source["name"] != "":
                implicit_source = torch.flatten(torch.tensor(
                    key_idx(self.pde.implicit_source["name"], "Additional info"), 
                    dtype=torch.int).repeat(len(vx), 1)
                )
            else:
                implicit_source = torch.flatten(torch.tensor(
                    -1, dtype=torch.int).repeat(len(vx), 1)
                )

            keys = torch.tensor([key_idx("vx", self.name), key_idx("vy", self.name), key_idx("s", self.name), key_idx("D", self.name), key_idx("A", self.name), key_idx("B", self.name), key_idx("implicit_source", self.name)])
            values = torch.stack([vx, vy, s, D, A, B, implicit_source], dim=1)

            return keys, values