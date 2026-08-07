import numpy as np
import torch
import random

def derivative(f: torch.Tensor, dx: float, dt: float, order: int, method: str = "central") -> torch.Tensor:
    """
    Compute partial derivatives of the specified order using a finite difference approximation method among
    - central differences (method = 'central'), 
    - forward differences (method = 'forward'),
    - backward differences (method = 'backward').

    Parameters
    ----------
    f : torch.Tensor
        The grid of function values (a scalar field discretization).
    dx : float
        The spatial resolution of the grid (the spatial step of the finite difference approximation).
    dt : float
        The temporal resolution of the grid (the temporal step of the finite difference approximation).
    order : int
        The order of the derivatives to approximate.
    method : str
        The finite difference approximation schema. 
        - Values: 'central', 'forward', 'backward'.
        - Default: 'central'.
    
    Returns
    -------
    - order == 0 -> f,
    - order == 1 -> [df/dx, df/dy, df/dt],
    - order == 2 -> [d2f/dxdx, d2f/dydy, d2f/dtdt, d2f/dxdy, d2f/dxdt, d2f/dydt].
    """
    if order == 0:
        return f
    elif order == 1:
        if method == "central":
            df_x = (f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1]) / (2 * dx)
            df_y = (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2]) / (2 * dx)
            df_t = (f[2:, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (2 * dt)
        elif method == "forward":
            df_x = (f[1:-1, 2:, 1:-1] - f[1:-1, 1:-1, 1:-1]) / (dx)
            df_y = (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, 1:-1]) / (dx)
            df_t = (f[2:, 1:-1, 1:-1] - f[1:-1, 1:-1, 1:-1]) / (dt)
        elif method == "backward":
            df_x = (f[1:-1, 1:-1, 1:-1] - f[1:-1, :-2, 1:-1]) / (dx)
            df_y = (f[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, :-2]) / (dx)
            df_t = (f[1:-1, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (dt)
        else:
            raise ValueError(f"Invalid difference method {method}. Valid values are 'central', 'forward' or 'backward'.")
        return torch.stack([df_x, df_y, df_t], dim=1)
    elif order == 2:
        if method == "central":
            d2f_xx = (f[1:-1, 2:, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / (dx ** 2)
            d2f_yy = (f[1:-1, 1:-1, 2:] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / (dx ** 2)
            d2f_tt = (f[2:, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / (dt ** 2)
            d2f_xt = (f[2:, 2:, 1:-1] - f[2:, :-2, 1:-1] - f[:-2, 2:, 1:-1] + f[:-2, :-2, 1:-1]) / (4 * dx * dt)
            d2f_yt = (f[2:, 1:-1, 2:] - f[2:, 1:-1, :-2] - f[:-2, 1:-1, 2:] + f[:-2, 1:-1, :-2]) / (4 * dx * dt)
            d2f_xy = (f[1:-1, 2:, 2:] - f[1:-1, :-2, 2:] - f[1:-1, 2:, :-2] + f[1:-1, :-2, :-2]) / (4 * dx ** 2)
        elif method == "forward":
            d2f_xx = (f[1:-1, 3:, 1:-1] - 2 * f[1:-1, 2:, 1:-1] + f[1:-1, 1:-1, 1:-1]) / (dx ** 2)
            d2f_yy = (f[1:-1, 1:-1, 3:] - 2 * f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, 1:-1]) / (dx ** 2)
            d2f_tt = (f[3:, 1:-1] - 2 * f[2:, 1:-1, 1:-1] + f[1:-1, 1:-1, 1:-1]) / (dt ** 2)
            d2f_xt = (f[2:, 2:, 1:-1] - f[2:, 1:-1, 1:-1] - f[1:-1, 2:, 1:-1] + f[1:-1, 1:-1, 1:-1]) / (dx * dt)
            d2f_yt = (f[2:, 1:-1, 2:] - f[2:, 1:-1, 1:-1] - f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, 1:-1]) / (dx * dt)
            d2f_xy = (f[1:-1, 2:, 2:] - f[1:-1, 1:-1, 2:] - f[1:-1, 2:, 1:-1] + f[1:-1, 1:-1, 1:-1]) / (dx ** 2)
        elif method == "backward":
            d2f_xx = (f[1:-1, 1:-1, 1:-1] - 2 * f[1:-1, :-2, 1:-1] + f[1:-1, :-3, 1:-1]) / (dx ** 2)
            d2f_yy = (f[1:-1, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, :-2] + f[1:-1, 1:-1, :-3]) / (dx ** 2)
            d2f_tt = (f[1:-1, 1:-1, 1:-1] - 2 * f[:-2, 1:-1, 1:-1] + f[:-3, 1:-1, 1:-1]) / (dt ** 2)
            d2f_xt = (f[1:-1, 1:-1, 1:-1] - f[1:-1, :-2, 1:-1] - f[:-2, 1:-1, 1:-1] + f[:-2, :-2, 1:-1]) / (dx * dt)
            d2f_yt = (f[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, :-2] - f[:-2, 1:-1, 1:-1] + f[:-2, 1:-1, :-2]) / (dx * dt)
            d2f_xy = (f[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, :-2] - f[1:-1, 1:-1, :-2] + f[1:-1, :-2, :-2]) / (dx ** 2)
        else:
            raise ValueError(f"Invalid difference method {method}. Valid values are 'central', 'forward' or 'backward'.")
        return torch.stack([d2f_xx, d2f_yy, d2f_tt, d2f_xy, d2f_xt, d2f_yt], dim=1)




def compute_d_dt(history: list, dt: float) -> float:
    """
    Approximates time derivative based on available history length.
    u_history: list of arrays [u_{n-2}, u_{n-1}, u_n]
    """
    n = len(history)

    if n < 2:
        # At t0, derivative is zero (or unknown)
        return np.zeros_like(history[-1])

    elif n == 2:
        # Step 1: 1st-order backward difference
        return (history[1] - history[0]) / dt

    else:
        # Step 2+: 2nd-order 3-point BDF2 formula
        return (3 * history[-1] - 4 * history[-2] + history[-3]) / (2 * dt)
        
def compute_d2_dt2(history: list, dt: float) -> float:
    """
    Approximates 2nd time derivative.
    """
    n = len(history)

    if n < 3:
        # Need at least 3 points to compute a second time derivative
        return np.zeros_like(history[-1])

    else:
        # Standard 2nd-order central/backward 3-point stencil
        return (history[-1] - 2 * history[-2] + history[-3]) / (dt**2)
    
def insert(buf: list, item: Any, capacity: int = 3) -> None:
    """
    Insert the new item in the FIFO limited capacity buffer.
    """
    if len(buf) >= capacity:
        buf.pop(0)
    buf.append(item)