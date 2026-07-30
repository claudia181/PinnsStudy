"""
pendulum.py
===========

This module implements the logic for the pendulum PDE class.

Spatio-temporal domain:
- Time-dependent
- 0-dimentional spatial domain

Global parameters:
- MASS [float] (m): default mass value.
- RODE_LENGTH [float] (l): default rode length.
- GRAV_ACC [float] (g): gravitatonal acceleration.
- DAMPING_COEFF [float] (b): default damping value.
- DRIVE_AMPLITUDE [float] (A): default drive amplitude.
- DRIVE_FREQUENCY [float] (w): default drive frequency.

Initial conditions default values:
- U0: default initial angle
- DU0: default initial angular velocity

Classes:
- Pendulum: Implements the pendulum ODE logic and methods required by the interface module pde_utils.py.
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp

# Default values for pendulum ODE parameters
MASS = 1.0              # m (mass)
RODE_LENGTH = 1.0       # l (rode length)
GRAV_ACC = 9.80665      # g (gravity acceleration)
DAMPING_COEFF = 0.0     # b (damping coefficient)
DRIVE_AMPLITUDE = 0.0   # A (drive amplitude)
DRIVE_FREQUENCY = 0.0   # w (drive frequency)

# Default initial conditions for pendulum ODE
U0 = np.pi / 3  # initial angle (60 degrees)
DU0 = 0.0       # initial angular velocity

class Pendulum:
    """
    Class representing a pendulum ODE.

    Attributes
    ----------
    m : float
        Mass.
    l : float
        Rode length.
    g : float
        Graviattional acceleration.
    b : float
        Damping coefficient.
    A : float
        Drive amplitude.
    w : float
        Drive frequency.
    u0 : float
        Initial angle.
    du0 : float
        Initial angular velocity.
    t : np.ndarray
        Time values.
    u : np.ndarray
        Solution values (one for each time instant).
    du : np.ndarray
     1st derivative values (one for each time instant).
    d2u : np.ndarray
        2nd derivative values (one for each time instant).
    force : np.ndarray
        Force values (one for each time instant).
    """

    def __init__(
            self,
            m: float = None,
            l: float = None,
            g: float = None,
            b: float = None,
            A: float = None,
            w: float = None
            ):
        """
        Constructor.

        Parameters
        ----------
        m : float
            Mass.
        l : float
            Rode length.
        g : float
            Graviattional acceleration.
        b : float
            Damping coefficient.
        A : float
            Drive amplitude.
        w : float
            Drive frequency.
        """
        if m is None: m = MASS
        if l is None: l = RODE_LENGTH
        if g is None: g = GRAV_ACC
        if b is None: b = DAMPING_COEFF
        if A is None: A = DRIVE_AMPLITUDE
        if w is None: w = DRIVE_FREQUENCY

        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.A = A
        self.w = w

        self.u0, self.du0 = None, None
        self.t = None
        self.u, self.du, self.d2u = None, None, None
        self.force = None
    
    def set_IC(self, u0: float = None, du0: float = None) -> None:
        """
        Set the initial conditions.

        Parameters
        ----------
        u0 : float
            Initial angle.
        du0 : float
            Initial angular velocity.

        Returns
        -------
        None
        """
        # Initial angle
        if u0 == None:
            self.u0 = U0
        else:
            self.u0 = u0
        
        # Initial angular velocity
        if du0 == None:
            self.du0 = DU0
        else:
            self.du0 = du0
    
    def _pendulum(self, t: float|np.ndarray, y: tuple) -> list:
        """
        Parameters
        ----------
        t : float|np.ndarray
            Time instant(s).
        y : tuple
            <u, du>.
        
        Returns
        -------
        list
            [du, d2u]
        """
        u, du = y
        d2u = - (self.b/(self.m*self.l**2)) * du - (self.g/self.l) * np.sin(u) + (self.A/(self.m*self.l**2)) * np.cos(self.w*t)
        return [du, d2u]
    
    def solve(self, t0: float = 0.0, tN: float = 0.0, snapshots: set = None, n_snapshots: int = None) -> None:
        """
        Solve the (possibly damped and driven) pendulum system.

        Fixed-step RK45 over the provided time grid t_ (monotone 1D vector).

        Parameters
        ----------
        t0 : float
            Initial time value.
        tN : float
            Final time value.
        snapshots : set
            Set of time values on which to store the computed solution values.
        n_snapshots : int
            Number of snapshots to store.

        Returns
        -------
        None
        """
        if snapshots == []:
            snapshots = None

        t_eval = snapshots

        if n_snapshots is not None:
            t_eval = np.linspace(start=t0, stop=tN, num=n_snapshots + 1)
        
        sol = solve_ivp(
            fun=self._pendulum,
            t_span=(t0, tN),
            t_eval=t_eval,            # t_eval=None let the solver pick steps adaptively
            y0=(self.u0, self.du0),
            method="RK45",
            rtol=1e-9, atol=1e-12,        # tight tolerances for chaotic regimes
            max_step=0.1                  # cap step size for better resolution
        )
        print(sol.message)

        self.u = sol.y[0] #np.unwrap(sol.y[0])
        self.du = sol.y[1]
        self.t = sol.t
        self.d2u = np.array([self._pendulum(ti, [ui, dui])[1] for ti, ui, dui in zip(self.t, self.u, self.du)])
        self.force = (self.A/(self.m*self.l**2)) * np.cos(self.w*self.t)
    
    def compute_force(self, t: float|np.ndarray) -> float|np.ndarray:
        """
        Compute the force value at time t.

        Parameters
        ----------
        t : float|np.ndarray
            Time.
        
        Returns
        -------
        float|np.ndarray
            Force value at time t.
        """
        return (self.A / (self.m * self.l**2)) * np.cos(self.w * t)
    
    @classmethod
    def residual(
        cls,
        u: torch.Tensor,
        du: torch.Tensor,
        d2u: torch.Tensor, 
        m: float,
        l: float,
        g: float,
        b: float,
        force: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the Pendulum residual.

        Parameters
        ----------
        u : torch.Tensor
            Solution values.
        du : torch.Tensor
            1st derivative values.
        d2u : torch.Tensor
            2nd derivative values.
        m : float
            Mass.
        l : float
            Rode length.
        g : float
            Gravitational acceleration.
        b : float
            Damping coefficient.
        force : torch.Tensor
            Force values.
        """
        #utt = d2u[:, 0]
        #ut = du[:, 0]
        return d2u + (b/(m*l**2)) * du + (g/l) * torch.sin(u) - force #(A/(m*l**2)) * np.cos(w*t)