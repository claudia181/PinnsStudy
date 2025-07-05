import torch
from functools import partial

SYS_MODES = ['Output', 'PINN', 'Output+PINN', 'Derivative', 'Hessian', 'Derivative+Hessian', 'Sobolev', 'Sobolev+Hessian']
DISTILL_MODES = ['Forgetting', 'Output', 'PINN', 'Derivative', 'Hessian', 'Derivative+Hessian', 'Sobolev', 'Sobolev+Hessian']

# Given a 0-centered dim-dimentional ball of radius radius,
# sample n dim-dimentional vectors uniformly distributed in the ball boundary (ball surface)
# ~ Uniform({x in R^dim | ||x|| = radius})
def ball_boundary_uniform(n: int, radius: float, dim: int):
    # sample n dim-dimentional vectors from N(0,1)
    angle = torch.distributions.Normal(0., 1.).sample((n, dim))
    #angle = torch.distributions.Uniform(0., 1.).sample((n, dim))*2 - 1 # [?] non ho capito

    # normalize the sampled vectors obtaining unitary vectors (of Euclidean norm 1)
    # normalizing gives a uniform distribution on the surface of the unit sphere in R^dim (rotational symmetry)
    norms = torch.norm(angle, p=2., dim=1).reshape((-1,1))
    angle = angle/norms

    # multiply by the radius of the ball, obtaining vectors of length (Euclidean norm) radius
    pts = radius*angle
    return pts

# Allen-Cahn --------------------------------------------------------------------------------

LAMBDA = 0.01 # lambda value for the (single parameter) Allen-Cahn PDE

# Solution u to the (single parameter) Allen-Cahn PDE
def allen_cahn_true(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    if len(params) == 1:
        return torch.exp(- params[0] * (x[:, 0] + 0.7)) * torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])
    else:
        i = torch.arange(1, params.shape[1] + 1, dtype=torch.float32).to(params.device)
        u = torch.sum(params * torch.sin(i * torch.pi * x[:, 0]) * torch.sin(i * torch.pi * x[:, 1]) / (i ** 2), dim=1)
        return u / params.shape[1]

# External force f relative to the solution u
def allen_cahn_forcing(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    u = allen_cahn_true(x, params)
    hes = allen_cahn_hes(x, params)
    uxx = hes[:, 0, 0]
    uyy = hes[:, 1, 1]
    return - LAMBDA * (uxx + uyy) - u + u ** 3

def allen_cahn_residual(x: torch.Tensor, u_pred: torch.Tensor, Hu_pred: torch.Tensor, params: list, force: torch.Tensor = None) -> torch.Tensor:
    if force is None:
        force = allen_cahn_forcing(x, params)
    return LAMBDA * (Hu_pred[:, 0, 0] + Hu_pred[:, 1, 1]) - u_pred + u_pred ** 3 - force

# Gradient vector relative to the solution u (pdv -> partial derivative)
# Analytically computed derivative
def allen_cahn_pdv(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    if len(params) == 1:
        ux = torch.exp(- params[0] * (x[:, 0] + 0.7)) * torch.sin(torch.pi * x[:, 1]) * (- params[0] * torch.sin(torch.pi * x[:, 0]) + torch.pi * torch.cos(torch.pi * x[:, 1]))
        uy = torch.exp(- params[0] * (x[:, 0] + 0.7)) * torch.pi * torch.sin(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1])
        return torch.column_stack((ux, uy))
    else:
        i = torch.arange(1, params.shape[1] + 1, dtype=torch.float32).to(x.device)
        ux = torch.sum(params * i * torch.pi * torch.cos(i * torch.pi * x[:, 0]) * torch.sin(i * torch.pi * x[:, 1]) / (i ** 2), dim=1)
        uy = torch.sum(params * i * torch.pi * torch.sin(i * torch.pi * x[:, 0]) * torch.cos(i * torch.pi * x[:, 1]) / (i ** 2), dim=1)
        pdv = torch.stack((ux, uy), dim=1)
        return pdv / params.shape[1]

# Hessian of the solution u
def allen_cahn_hes(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    if len(params) == 1:
        uxx = torch.sin(torch.pi * x[:, 1]) * torch.exp(- params[0] * (x[:, 0] + 0.7)) * (torch.sin(torch.pi * x[:, 0] * (params[0] ** 2 - torch.pi ** 2) - 2 * params[0] * torch.pi * torch.cos(torch.pi * x[:, 0])))
        uxy = torch.cos(torch.pi * x[:, 1]) * torch.exp(- params[0] * (x[:, 0] + 0.7)) * torch.pi * (torch.pi * torch.cos(torch.pi * x[:, 0]) - params[0] * torch.sin(torch.pi * x[:, 0]))
        uyy = - torch.exp(- params[0] * (x[:, 0] + 0.7)) * torch.pi ** 2 * torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])
        return torch.column_stack((uxx, uxy, uxy, uyy)).reshape((-1, 2, 2))
    else:
        i = torch.arange(1, params.shape[1] + 1, dtype=torch.float32).to(x.device)
        uxx = torch.sum(- params * (i ** 2) * (torch.pi ** 2) * torch.sin(i * torch.pi * x[:, 0]) * torch.sin(i * torch.pi * x[:, 1]) / (i ** 2), dim=1)
        uxy = torch.sum(params * (i ** 2) * (torch.pi ** 2) * torch.cos(i * torch.pi * x[:, 0]) * torch.cos(i * torch.pi * x[:, 1]) / (i ** 2), dim=1)
        uyy = torch.sum(- params * (i ** 2) * (torch.pi ** 2) * torch.sin(i * torch.pi * x[:, 0]) * torch.sin(i * torch.pi * x[:, 1]) / (i ** 2), dim=1)
        hes = torch.stack((uxx, uxy, uxy, uyy), dim=-1).reshape((-1, 2, 2))
        return hes / params.shape[1]

class Pde:
    def __init__(self, name: str, params: torch.Tensor):
        self.name = name
        self.params = params
        if name == 'Allen-Cahn':
            f = partial(allen_cahn_true, params = params)
            df = partial(allen_cahn_pdv, params = params)
            d2f = partial(allen_cahn_hes, params = params)
            residual = partial(allen_cahn_residual, params = params)
            force = partial(allen_cahn_forcing, params = params)
            self.solution = f
            self.der = [f, df, d2f]
            self.residual = residual
            self.force = force

            self.psolution = allen_cahn_true
            self.pder = [allen_cahn_true, allen_cahn_pdv, allen_cahn_hes]
            self.presidual = allen_cahn_residual
            self.pforce = allen_cahn_forcing