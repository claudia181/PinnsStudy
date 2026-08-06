import numpy as np
import torch
import random

def derivative(f: np.ndarray|torch.Tensor, order: int, dx: float, dt: float, method: str = "central") -> np.ndarray|torch.Tensor:
    if order == 0:
        return f
    elif order == 1:
        if method == "central":
            df_x = (f[2:, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (2 * dx)
            df_y = (f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1]) / (2 * dx)
            df_t = (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2]) / (2 * dt)
        elif method == "forward":
            df_x = (f[2:, 1:-1, 1:-1] - f[1:-1, 1:-1, 1:-1]) / (dx)
            df_y = (f[1:-1, 2:, 1:-1] - f[1:-1, 1:-1, 1:-1]) / (dx)
            df_t = (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, 1:-1]) / (dt)
        elif method == "backward":
            df_x = (f[1:-1, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (dx)
            df_y = (f[1:-1, 1:-1, 1:-1] - f[1:-1, :-2, 1:-1]) / (dx)
            df_t = (f[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, :-2]) / (dt)
        else:
            raise ValueError(f"Invalid difference method {method}. Valid values are 'central', 'forward' or 'backward'.")
    elif order == 2:
        if method == "central":
            d2f_xx = (f[2:, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / (dx ** 2)
            d2f_yy = (f[1:-1, 2:, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / (dx ** 2)
            d2f_tt = (f[1:-1, 1:-1, 2:] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / (dt ** 2)
            d2f_xt = (f[2:, 1:-1, 2:] - f[:-2, 1:-1, 2:] - f[:2, 1:-1, :-2] + f[:-2, 1:-1, :-2]) / (4 * dx * dt)
        elif method == "forward":
            d2f_xx = (f[3:, 1:-1, 1:-1] - 2 * f[2:, 1:-1, 1:-1] + f[1:-1, 1:-1, 1:-1]) / (dx ** 2)
            d2f_yy = (f[1:-1, 3:, 1:-1] - 2 * f[1:-1, 2:, 1:-1] + f[1:-1, 1:-1, 1:-1]) / (dx ** 2)
            d2f_tt = (f[1:-1, 1:-1, 3:] - 2 * f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, 1:-1]) / (dt ** 2)
            d2f_xt = (f[2:, 1:-1, 2:] - f[1:-1, 1:-1, 2:] - f[:2, 1:-1, 1:-1] + f[1:-1, 1:-1, 1:-1]) / (dx * dt)
        elif method == "backward":
            d2f_xx = (f[1:-1, 1:-1, 1:-1] - 2 * f[:-2, 1:-1, 1:-1] + f[:-3, 1:-1, 1:-1]) / (dx ** 2)
            d2f_yy = (f[1:-1, 1:-1, 1:-1] - 2 * f[1:-1, :-2, 1:-1] + f[1:-1, :-3, 1:-1]) / (dx ** 2)
            d2f_tt = (f[1:-1, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, :-2] + f[1:-1, 1:-1, :-3]) / (dt ** 2)
            


    pdv_save = torch.from_numpy(np.column_stack((pdv_t_grid[::downsample,::downsample,::downsample].reshape((-1,1)), pdv_x_grid[::downsample,::downsample,  ::downsample].reshape((-1,1)), pdv_y_grid[::downsample,::downsample,::downsample].reshape((-1,1)))))