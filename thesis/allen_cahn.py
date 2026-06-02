"""
allen_cahn.py
===========

This module implements the logic for the 2D time-independent Allen-Cahn PDE class
(time independent --> describing an equilibrium situation of phase separation).

Spatio-temporal domain:
- Time-independent
- 2-dimentional spatial domain

Global parameters:
- THICKNESS_PARAM [float] (lambda): default thickness value of the considered surface.
- FORCE_PARAMS [list] (xi): default force values

Classes:
- AllenCahn: Implements the Allen-Cahn PDE logic and methods required by the interface module pde_utils.py.
"""

import torch

# Default values for Allen-Cahn PDE parameters
THICKNESS_PARAM = 0.01 # lambda
FORCE_PARAMS = [0.0] # xi

class AllenCahn:
    """
    Class representing a time-independent (equilibrium) Allen-Cahn PDE.

    Attributes
    ----------
    lam : float
        PDE parameter that indicates the thickness of the considered surface.
    force_params : list
        PDE parameters describing the forces in the system.
    x : float
        x coordinates of the domain points.
    y : float
        y coordinates of the domain points.
    u : np.ndarray
        Solution values.
    du : np.ndarray
     1st derivative values.
    d2u : np.ndarray
        2nd derivative values.
    force : np.ndarray
        Force values.
    """
    def __init__(self, lam: float = None, force_params: list = None, device: str = "cpu"):
        """
        Constructor.

        Parameters
        ----------
        lam : float
            PDE parameter that indicates the thickness of the considered surface.
        force_params : list
            PDE parameters describing the forces in the system.
        device : str
        """
        self.device = device

        if lam is None: lam = THICKNESS_PARAM
        if force_params is None: force_params = FORCE_PARAMS

        self.lam = torch.tensor(lam, device=device)  
        self.force_params = torch.tensor(force_params, device=device)

        self.x, self.y = None, None
        self.u, self.du, self.d2u = None, None, None
        self.force = None
  
    def set_spatial_points(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Set the spatial domain points.

        Parameters
        ----------
        x : torch.Tensor
            x coordinates.
        y : torch.Tensor
            y coordinates.
        
        Returns
        -------
        None
        """
        self.x, self.y = x, y

    def solve(self) -> None:
        """
        Compute the solution of the Allen-Cahn PDE on the spatial points.

        Parameters
        ----------
        _

        Returns
        -------
        None
        """
        self.u = self._sol()
        self.du = self._der()
        self.d2u = self._hes()
        uxx = self.d2u[:, 0, 0]
        uyy = self.d2u[:, 1, 1]
        self.force = self.lam * (uxx + uyy) + self.u ** 3 - self.u 

    def compute_force(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the force at the given spatial points.

        Parameters
        ----------
        x : torch.Tensor
        y : torch.Tensor

        Returns
        -------
        torch.Tensor
            Force values at the given spatial points.
        """
        self.x = x
        self.y = y
        u = self._sol()
        d2u = self._hes()
        uxx = d2u[:, 0, 0]
        uyy = d2u[:, 1, 1]
        force = self.lam * (uxx + uyy) + u ** 3 - u
        return force
    
    @classmethod
    def residual(cls, u: torch.Tensor, d2u: torch.Tensor, force: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Compute the residual.

        Parameters
        ----------
        u : torch.Tensor
            Solution values.
        du : torch.Tensor
            1st derivative values.
        d2u : torch.Tensor
            2nd derivative values.
        force : torch.Tensor
            Force values.
        lam : float
            Thickness parameter.

        Returns
        -------
        torch.Tensor
            Residual values.
        """
        uxx = d2u[:, 0, 0]
        uyy = d2u[:, 1, 1]
        return lam * (uxx + uyy) - u + u ** 3 - force

    def _sol(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Solution values.
        """
        if len(self.force_params) == 1:
            return torch.exp(- self.force_params[0] * (self.x + 0.7)) * torch.sin(torch.pi * self.x) * torch.sin(torch.pi * self.y)
        else:
            u = 0.0
            for j in range(1, len(self.force_params) + 1):
                xi_j = self.force_params[j-1].item()
                u += (xi_j * torch.sin(j * torch.pi * self.x) * torch.sin(j * torch.pi * self.y) / (j ** 2))
            return u / self.force_params.shape[0] # normalize

    def _der(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Gradient vector relative to the solution u (analytically computed).
        """
        if len(self.force_params) == 1:
            ux = torch.exp(- self.force_params[0] * (self.x + 0.7)) * torch.sin(torch.pi * self.y) * (- self.force_params[0] * torch.sin(torch.pi * self.x) + torch.pi * torch.cos(torch.pi * self.y))
            uy = torch.exp(- self.force_params[0] * (self.x + 0.7)) * torch.pi * torch.sin(torch.pi * self.x) * torch.cos(torch.pi * self.y)
            return torch.column_stack((ux, uy))
        else:
            ux = 0.0
            uy = 0.0
            for j in range(1, len(self.force_params) + 1):
                xi_j = self.force_params[j-1].item()
                ux += (xi_j * torch.pi * torch.cos(j * torch.pi * self.x) * torch.sin(j * torch.pi * self.y) / j)
                uy += (xi_j * torch.pi * torch.sin(j * torch.pi * self.x) * torch.cos(j * torch.pi * self.y) / j)
            pdv = torch.stack((ux, uy), dim=1)
            return pdv / self.force_params.shape[0] # normalize

    def _hes(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Hessian of the solution u (analytically computed).
        """
        if len(self.force_params) == 1:
            uxx = torch.sin(torch.pi * self.y) * torch.exp(- self.force_params[0] * (self.x + 0.7)) * (torch.sin(torch.pi * self.x * (self.force_params[0] ** 2 - torch.pi ** 2) - 2 * self.force_params[0] * torch.pi * torch.cos(torch.pi * self.x)))
            uxy = torch.cos(torch.pi * self.y) * torch.exp(- self.force_params[0] * (self.x + 0.7)) * torch.pi * (torch.pi * torch.cos(torch.pi * self.x) - self.force_params[0] * torch.sin(torch.pi * self.x))
            uyy = - torch.exp(- self.force_params[0] * (self.x + 0.7)) * torch.pi ** 2 * torch.sin(torch.pi * self.x) * torch.sin(torch.pi * self.y)
            return torch.column_stack((uxx, uxy, uxy, uyy)).reshape((-1, 2, 2))
        else:
            uxx = 0.0
            uyy = 0.0
            uxy = 0.0
            for j in range(1, len(self.force_params) + 1):
                xi_j = self.force_params[j-1].item()
                uxx += (- xi_j * (torch.pi ** 2) * torch.sin(j * torch.pi * self.x) * torch.sin(j * torch.pi * self.y))
                uyy += (- xi_j * (torch.pi ** 2) * torch.sin(j * torch.pi * self.x) * torch.sin(j * torch.pi * self.y))
                uxy += (xi_j * (torch.pi ** 2) * torch.cos(j * torch.pi * self.x) * torch.cos(j * torch.pi * self.y))
            hes = torch.stack((uxx, uxy, uxy, uyy), dim=-1).reshape((-1, 2, 2))
            return hes / self.force_params.shape[0] # normalize