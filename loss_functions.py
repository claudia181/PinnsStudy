"""
loss_functions.py
===========

This module implements the loss functions for any PhysicsTask and a function to attach the appropriate loss to any PhysicsTask.
"""

import torch
from physics_task import PhysicsTask, NeumannBCTask, DirichletBCTask, ICTask, OutputTask, DerivativeTask, SpatialDerivativeTask, TemporalDerivativeTask, Derivative2Task, SpatialDerivative2Task, TemporalDerivative2Task, AdvectionReactionDiffusionTask, StationaryAllenCahnTask
from AdvectionReactionDiffusion.advection_reaction_diffusion import AdvectionReactionDiffusion
from StationaryAllenCahn.allen_cahn import AllenCahn
from model import Pinn

def attach_loss_function(task: PhysicsTask) -> None:
    if type(task) is AdvectionReactionDiffusionTask:
        task.loss_fn = advection_reaction_diffusion_loss #(model=model, task=task, **kwargs)
    elif type(task) is StationaryAllenCahnTask:
        task.loss_fn = stationary_allen_cahn_loss
    elif type(task) is NeumannBCTask:
        task.loss_fn = neumann_bc_loss
    elif type(task) is DirichletBCTask:
        task.loss_fn = dirichlet_bc_loss
    elif type(task) is ICTask:
        task.loss_fn = ic_loss
    elif type(task) is OutputTask:
        task.loss_fn = output_loss
    elif type(task) is DerivativeTask:
        task.loss_fn = derivative_loss
    elif type(task) is SpatialDerivativeTask:
        task.loss_fn = spatial_derivative_loss
    elif type(task) is TemporalDerivativeTask:
        task.loss_fn = temporal_derivative_loss
    elif type(task) is Derivative2Task:
        task.loss_fn = derivative2_loss
    elif type(task) is SpatialDerivative2Task:
        task.loss_fn = spatial_derivative2_loss
    elif type(task) is TemporalDerivative2Task:
        task.loss_fn = temporal_derivative2_loss

def advection_reaction_diffusion_loss(
    x: torch.Tensor,
    pde_parameters: torch.Tensor,
    model: Pinn,
    task: AdvectionReactionDiffusionTask
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted PDE residual field and the null field.
    """
    if len(task.param_keys) != len(pde_parameters):
        raise ValueError(f"The number of expected PDE parameters is {len(task.param_keys)}, while {len(pde_parameters)} PDE parameters are passed.")
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    
    u = model.derivative(order=0, x=x, pde_params=input_params)
    du = model.derivative(order=1, x=x, pde_params=input_params)
    d2u = model.derivative(order=2, x=x, pde_params=input_params)
    x_ = x[:, 0]
    y = x[:, 1]
    t = x[:, 2]
    D_values = None
    vx_values = None
    vy_values = None
    source_values = None
    A_values = None
    B_values = None
    for key, value in zip(task.param_keys, pde_parameters):
        if key == "D":
            D_values = value
        elif key == "vx":
            vx_values = value
        elif key == "vy":
            vy_values = value
        elif key == "s":
            source_values = value
        elif key == "A":
            A_values = value
        elif key == "B":
            B_values = value
    if D_values is None:
        D_values = task.D
    if source_values is None:
        source_values = task.source_fn(x=x_, y=y, t=t)
    
    if A_values is not None:
        task.implicit_source_fn.set_A(A_values)
    if B_values is not None:
        task.implicit_source_fn.set_B(B_values)
    implicit_source_values = task.implicit_source_fn(u=u)
    if vx_values is None or vy_values is None:
        velocity_values = task.velocity_fn(x=x_, y=y, t=t)
        if vx_values is not None and vy_values is None:
            velocity_values = torch.stack((vx_values, velocity_values[:, 1]), dim=1)
        elif vx_values is None and vy_values is not None:
            velocity_values = torch.stack((velocity_values[:, 0], vy_values), dim=1)
    else:
        velocity_values = torch.stack((vx_values, vy_values))
    
    mse_loss = torch.nn.MSELoss(reduction='mean')
    
    residual_value = AdvectionReactionDiffusion.residual(
        du=du, 
        d2u=d2u, 
        velocity=velocity_values, 
        source=source_values, 
        implicit_source=implicit_source_values, 
        D=D_values
    )
    return mse_loss(residual_value, torch.zeros_like(residual_value))

def stationary_allen_cahn_loss(
    x: torch.Tensor, 
    pde_parameters: torch.Tensor, 
    model: Pinn,
    task: StationaryAllenCahnTask
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted PDE residual field and the null field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')

    u = model.derivative(order=0, x=x, pde_params=input_params)
    d2u = model.derivative(order=2, x=x, pde_params=input_params)
    x_ = x[:, 0]
    y = x[:, 1]
    if task.lam_index is not None:
        lam_values = pde_parameters[:, task.lam_index]
    else:
        lam_values = task.lam
    if task.xi_indexes is not None:
        xi_values = pde_parameters[:, task.xi_indexes]
    else:
        xi_values = task.xi_vector
    residual_value = AllenCahn.residual(u=u, d2u=d2u, x=x_, y=y, lam=lam_values, force_params=xi_values)
    return mse_loss(residual_value, torch.zeros_like(residual_value))

def neumann_bc_loss(
    x: torch.Tensor, 
    pde_parameters: torch.Tensor, 
    model: Pinn, 
    task: NeumannBCTask,
    du: torch.Tensor, 
    n: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted outward flux and the wanted outward flux.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    du_pred = model.derivative(order=1, x=x, pde_params=input_params)
    return mse_loss(task.out_flux(du=du_pred, n=n), task.out_flux(du=du, n=n))

def dirichlet_bc_loss(
    x: torch.Tensor, 
    pde_parameters: torch.Tensor, 
    model: Pinn, 
    task: DirichletBCTask,
    u: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted boundary value and the wanted boundary value.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    u_pred = model.forward(x=x, pde_params=input_params)
    return mse_loss(u_pred, u)

def ic_loss(
        x: torch.Tensor, 
        pde_parameters: torch.Tensor, 
        model: Pinn, 
        task: ICTask, 
        u: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted initial state field and the wanted initial state field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    u_pred = model.forward(x=x, pde_params=input_params)
    return mse_loss(u_pred, u)

def output_loss(
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        task: OutputTask,
        u: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted u field and the wanted u field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    u_pred = model.forward(x=x, pde_params=input_params)
    return mse_loss(u_pred, u)

def derivative_loss(
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        task: DerivativeTask,
        du: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted du field and the wanted du field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    du_pred = model.derivative(order=1, x=x, pde_params=input_params)
    return mse_loss(du_pred, du)

def spatial_derivative_loss(
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        task: SpatialDerivativeTask,
        du: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted du_xy field and the wanted du_xy field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    du_pred = model.derivative(order=1, x=x, pde_params=input_params)
    return mse_loss(du_pred[:, :2], du[:, :2])

def temporal_derivative_loss(
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        task: TemporalDerivativeTask,
        du: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted du_t field and the wanted du_t field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    du_pred = model.derivative(order=1, x=x, pde_params=input_params)
    return mse_loss(du_pred[:, 2:], du[:, 2:])

def derivative2_loss(
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        task: Derivative2Task,
        d2u: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted d2u field and the wanted d2u field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
    return mse_loss(d2u_pred, d2u)

def spatial_derivative2_loss(
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        task: SpatialDerivative2Task,
        d2u: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted [d2u_xx, d2u_yy, d2u_xy] field and the wanted [d2u_xx, d2u_yy, d2u_xy] field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
    return mse_loss(d2u_pred[:, :2], d2u[:, :2]) # mse_loss(d2u_pred[:, :2, :2], d2u[:, :2, :2])

def temporal_derivative2_loss(
        x: torch.Tensor,
        pde_parameters: torch.Tensor,
        model: Pinn,
        d2u: torch.Tensor
) -> torch.Tensor:
    """
    Loss function giving the loss term of the task:
    - MSE btw the predicted d2u_tt field and the wanted d2u_tt field.
    """
    if model.input_param_dict is not None:
        input_params = pde_parameters[:, list(model.input_param_dict.values())]
    else:
        input_params = None
    mse_loss = torch.nn.MSELoss(reduction='mean')
    d2u_pred = model.derivative(order=2, x=x, pde_params=input_params)
    return mse_loss(d2u_pred[:, 2], d2u[:, 2]) # mse_loss(d2u_pred[:, 2, 2], d2u[:, 2, 2])