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
- AllenCahn: Implements the Allen-Cahn PDE logic and methods.
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
    x : torch.Tensor
        x coordinates of the domain points.
    y : torch.Tensor
        y coordinates of the domain points.
    u : torch.Tensor
        Solution values.
    du : torch.Tensor
     1st derivative values.
    d2u : torch.Tensor
        2nd derivative values.
    force : torch.Tensor
        Force values.
    """
    def __init__(self, lam: float = None, force_params: list = None):
        """
        Constructor.

        Parameters
        ----------
        lam : float
            PDE parameter that indicates the thickness of the considered surface.
        force_params : list
            PDE parameters describing the forces in the system.
        """

        if lam is None: lam = THICKNESS_PARAM
        if force_params is None: force_params = FORCE_PARAMS

        self.lam = torch.tensor(lam)  
        self.force_params = torch.tensor(force_params)

        self.x, self.y = None, None
        self.u, self.du, self.d2u = None, None, None
        self.force = None
  
    def set_spatial_points(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Set the spatial domain points.

        Parameters
        ----------
        x : torch.Tensor
            x-coordinates.
        y : torch.Tensor
            y-coordinates.
        """
        self.x, self.y = x, y

    def solve(self) -> None:
        """
        Compute the solution of the Allen-Cahn PDE on the domain points.
        """
        self.u = self._sol()
        self.du = self._der()
        self.d2u = self._hes()
        uxx = self.d2u[:, 0]
        uyy = self.d2u[:, 1]
        self.force = self.lam * (uxx + uyy) + self.u ** 3 - self.u
    
    @classmethod
    def residual(cls, u: torch.Tensor, d2u: torch.Tensor, x: torch.Tensor, y: torch.Tensor, lam: float, force_params: torch.Tensor) -> torch.Tensor:
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
        force_params : torch.Tensor
            Force parameters.
        lam : float
            Thickness parameter.

        Returns
        -------
        torch.Tensor
            Residual values.
        """
        uxx = d2u[:, 0]
        uyy = d2u[:, 1]
        force = AllenCahn.force(x=x, y=y, lam=lam, force_params=force_params)
        return lam * (uxx + uyy) - u + u ** 3 - force

    @classmethod
    def force(cls, x: torch.Tensor, y: torch.Tensor, lam: float, force_params: torch.Tensor) -> torch.Tensor:
        """
        Compute the force.

        Parameters
        ----------
        x : torch.Tensor
            x-coordinates.
        y : torch.Tensor
            y-coordinates.
        lam : float
            Thickness parameter.
        force_params : torch.Tensor
            Force parameters.

        Returns
        -------
        torch.Tensor
            Force values.
        """
        if len(force_params) == 1:
            u = torch.exp(- force_params[0] * (x + 0.7)) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

            uxx = torch.sin(torch.pi * y) * torch.exp(- force_params[0] * (x + 0.7)) * (torch.sin(torch.pi * x * (force_params[0] ** 2 - torch.pi ** 2) - 2 * force_params[0] * torch.pi * torch.cos(torch.pi * x)))
            uyy = - torch.exp(- force_params[0] * (x + 0.7)) * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
            
        else:
            u = 0.0
            for j in range(1, len(force_params) + 1):
                xi_j = force_params[j-1].item()
                u += (xi_j * torch.sin(j * torch.pi * x) * torch.sin(j * torch.pi * y) / (j ** 2))
            u = u / force_params.shape[0] # normalize

            uxx = 0.0
            uyy = 0.0
            for j in range(1, len(force_params) + 1):
                xi_j = force_params[j-1].item()
                uxx += (- xi_j * (torch.pi ** 2) * torch.sin(j * torch.pi * x) * torch.sin(j * torch.pi * y)) / force_params.shape[0]
                uyy += (- xi_j * (torch.pi ** 2) * torch.sin(j * torch.pi * x) * torch.sin(j * torch.pi * y)) / force_params.shape[0]

        force = lam * (uxx + uyy) + u ** 3 - u
        return force

    def _sol(self) -> torch.Tensor:
        """
        Computes the solution function scalar field.

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
        Computes the solution spatial gradient vector field.

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
        Computes the solution 2nd spatial derivatives.
        
        Returns
        -------
        torch.Tensor
            Hessian of the solution u (analytically computed).
        """
        if len(self.force_params) == 1:
            uxx = torch.sin(torch.pi * self.y) * torch.exp(- self.force_params[0] * (self.x + 0.7)) * (torch.sin(torch.pi * self.x * (self.force_params[0] ** 2 - torch.pi ** 2) - 2 * self.force_params[0] * torch.pi * torch.cos(torch.pi * self.x)))
            uxy = torch.cos(torch.pi * self.y) * torch.exp(- self.force_params[0] * (self.x + 0.7)) * torch.pi * (torch.pi * torch.cos(torch.pi * self.x) - self.force_params[0] * torch.sin(torch.pi * self.x))
            uyy = - torch.exp(- self.force_params[0] * (self.x + 0.7)) * torch.pi ** 2 * torch.sin(torch.pi * self.x) * torch.sin(torch.pi * self.y)
            return torch.column_stack((uxx, uyy, uxy), dim=1)
        else:
            uxx = 0.0
            uyy = 0.0
            uxy = 0.0
            for j in range(1, len(self.force_params) + 1):
                xi_j = self.force_params[j-1].item()
                uxx += (- xi_j * (torch.pi ** 2) * torch.sin(j * torch.pi * self.x) * torch.sin(j * torch.pi * self.y))
                uyy += (- xi_j * (torch.pi ** 2) * torch.sin(j * torch.pi * self.x) * torch.sin(j * torch.pi * self.y))
                uxy += (xi_j * (torch.pi ** 2) * torch.cos(j * torch.pi * self.x) * torch.cos(j * torch.pi * self.y))
            #hes = torch.stack((uxx, uxy, uxy, uyy), dim=-1).reshape((-1, 2, 2))
            hes = torch.stack((uxx, uyy, uxy), dim=1)
            return hes / self.force_params.shape[0] # normalize