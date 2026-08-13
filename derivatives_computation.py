import torch
from typing import List, Collection

def derivative(f: Collection[torch.Tensor], nx: int, ny: int, dx: float, dt: float, order: int) -> List[torch.Tensor]:
    """
    Approximate the partial derivatives of the specified order using finite differences:
    - central differences for the interior points, 
    - forward differences for the left boundary points,
    - backward differences for the right boundary points.

    Parameters
    ----------
    f : Collection[torch.Tensor]
        Collection of nt tensors, each containing the f values of a frame, each of shape (ny * nx,).
    nx: int
        number of cells of the horizontal side.
    ny: int
        number of cells of the vertical side.
    dx : float
        The spatial resolution of the grid (the spatial step of the finite difference approximation), assumed to be the same for both the x- and y-direction.
    dt : float
        The temporal resolution of the grid (the temporal step of the finite difference approximation).
    order : int
        The order of the derivatives to approximate.
    
    Returns
    -------
    _torch.Tensor_ \n
    Derivative vector field discretization.
    - order == 1 -> [df/dx, df/dy, df/dt], shape (nt, ny * nx, 3),
    - order == 2 -> [d2f/dxdx, d2f/dydy, d2f/dtdt, d2f/dxdy, d2f/dxdt, d2f/dydt], shape (nt, ny * nx, 6).
    """
    # Convert each tensor of f from shape (ny * nx,) to shape (ny, nx)
    f = [ft.reshape(ny, nx) for ft in f]

    # Convert the collection f into a torch tensor of shape (nt, ny, nx)
    f = torch.stack(list(f))

    # Check if there are enough points to approximate the derivatives with finite differences
    if f.shape[0] < 2:
        raise ValueError(f"Not enough values to approximate the t-derivative component: {f.shape[0]} < 2.")
    if f.shape[1] < 2:
            raise ValueError(f"Not enough values to approximate the y-derivative component: {f.shape[0]} < 2.")
    if f.shape[2] < 2:
            raise ValueError(f"Not enough values to approximate the x-derivative component: {f.shape[0]} < 2.")
    
    # 1st order derivative
    if order == 1:
        ## Initialize empty derivative tensors
        df_x = torch.empty_like(f)
        df_y = torch.empty_like(f)
        df_t = torch.empty_like(f)

        ## Central differences for the interior points
        if f.shape[1] > 2:
            df_y[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dx)
        if f.shape[2] > 2:
            df_x[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dx)
        if f.shape[0] > 2:
            df_t[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * dt)

        ## Forward differences for the left boundary points
        df_y[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dx
        df_x[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dx
        df_t[0, :, :] = (f[1, :, :] - f[0, :, :]) / dt

        ## Backward differences for the right boundary points
        df_y[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dx
        df_x[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dx
        df_t[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dt

        ## Returns a (nt * ny * nx, 3)-shaped tensor of gradient vectors [df/dx, df/dy, df/dt]
        df_x = df_x.reshape(f.shape[0], nx * ny) # reshape df_x to (nt, nx * ny)
        df_y = df_y.reshape(f.shape[0], nx * ny) # reshape df_y to (nt, nx * ny)
        df_t = df_t.reshape(f.shape[0], nx * ny) # reshape df_t to (nt, nx * ny)
        gradient_field = torch.stack([df_x, df_y, df_t], dim=-1) # stack a (nt * ny * nx, 3)-shaped tensor
        return gradient_field

    # 2nd order derivative
    elif order == 2:
        ## Initialize empty 2nd derivative tensors
        d2f_xx = torch.empty_like(f)
        d2f_yy = torch.empty_like(f)
        d2f_tt = torch.empty_like(f)
        d2f_xt = torch.empty_like(f)
        d2f_yt = torch.empty_like(f)
        d2f_xy = torch.empty_like(f)

        ## Diagonal 2nd derivatives xx, yy, tt
        ### (Central)^2 differences on the interior points
        if f.shape[1] > 2:
            d2f_yy[:, 1:-1, :] = (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / (dx ** 2)
        if f.shape[2] > 2:
            d2f_xx[:, :, 1:-1] = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / (dx ** 2)
        if f.shape[0] > 2:
            d2f_tt[1:-1, :, :] = (f[2:, :, :] - 2 * f[1:-1, :, :] + f[:-2, :, :]) / (dt ** 2)

        ### (Forward)^2 differences on the left boundary points
        d2f_yy[:, 0, :] = (f[:, 2, :] - 2 * f[:, 1, :] + f[:, 0, :]) / (dx ** 2)
        d2f_xx[:, :, 0] = (f[:, :, 2] - 2 * f[:, :, 1] + f[:, :, 0]) / (dx ** 2)
        d2f_tt[0, :, :] = (f[2, :, :] - 2 * f[1, :, :] + f[0, :, :]) / (dt ** 2)

        ### (Backward)^2 differences on the right boundary points
        d2f_yy[:, -1, :] = (f[:, -1, :] - 2 * f[:, -2, :] + f[:, -3, :]) / (dx ** 2)
        d2f_xx[:, :, -1] = (f[:, :, -1] - 2 * f[:, :, -2] + f[:, :, -3]) / (dx ** 2)
        d2f_tt[-1, :, :] = (f[-1, :, :] - 2 * f[-2, :, :] + f[-3, :, :]) / (dt ** 2)

        ## Cross 2nd derivatives xt, yt, xy
        ### Initialize 1st derivative tensors
        df_x = torch.empty_like(f)
        df_y = torch.empty_like(f)
        
        ### Central differences on the interior points
        if f.shape[1] > 2:
            df_y[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dx)
        if f.shape[2] > 2:
            df_x[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dx)
        
        ### Forward differences on the left boundary points
        df_y[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dx
        df_x[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dx
        
        ### Backward differences on the right boundary points
        df_y[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dx
        df_x[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dx

        ### (Forward|Central|Backward) * (Central) differences on the t-interior points
        if f.shape[0] > 2:
            d2f_xt[1:-1, :, :] = (df_x[2:, :, :] - df_x[:-2, :, :]) / (2 * dt)
            d2f_yt[1:-1, :, :] = (df_y[2:, :, :] - df_y[:-2, :, :]) / (2 * dt)
        ### (Forward|Central|Backward) * (Central) differences on the y-interior points
            d2f_xy[:, :, 1:-1] = (df_x[:, :, 2:] - df_x[:, :, :-2]) / (2 * dx)

        ### (Forward|Central|Backward) * (Forward) differences on the t-left boundary points
        d2f_xt[0, :, :] = (df_x[1, :, :] - df_x[0, :, :]) / dt
        d2f_yt[0, :, :] = (df_y[1, :, :] - df_y[0, :, :]) / dt
        ### (Forward|Central|Backward) * (Forward) differences on the y-left boundary points
        d2f_xy[:, :, 0] = (df_x[:, :, 1] - df_x[:, :, 0]) / dx
        
        ### (Forward|Central|Backward) * (Backward) differences on the t-right boundary points
        d2f_xt[-1, :, :] = (df_x[-1, :, :] - df_x[-2, :, :]) / dt
        d2f_yt[-1, :, :] = (df_y[-1, :, :] - df_y[-2, :, :]) / dt
        ### (Forward|Central|Backward) * (Backward) differences on the y-right boundary points
        d2f_xy[:, -1, :] = (df_x[:, -1, :] - df_x[:, -2, :]) / dx

        ## Returns a (nt * ny * nx, 6)-shaped tensor of vectors [d2f/dxdx, d2f/dydy, d2f/dtdt, d2f/dxdy, d2f/dxdt, d2f/dydt]
        d2f_xx = d2f_xx.reshape(f.shape[0], nx * ny) # reshape df_xx to (nt, nx * ny)
        d2f_yy = d2f_yy.reshape(f.shape[0], nx * ny) # reshape df_yy to (nt, nx * ny)
        d2f_tt = d2f_tt.reshape(f.shape[0], nx * ny) # reshape df_tt to (nt, nx * ny)
        d2f_xy = d2f_xy.reshape(f.shape[0], nx * ny) # reshape df_xy to (nt, nx * ny)
        d2f_xt = d2f_xt.reshape(f.shape[0], nx * ny) # reshape df_xt to (nt, nx * ny)
        d2f_yt = d2f_yt.reshape(f.shape[0], nx * ny) # reshape df_yt to (nt, nx * ny)
        der2_field = torch.stack([d2f_xx, d2f_yy, d2f_tt, d2f_xy, d2f_xt, d2f_yt], dim=-1) # stack a (nt * ny * nx, 6)-shaped tensor
        return der2_field