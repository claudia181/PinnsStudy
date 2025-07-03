import torch
from torch import nn
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian
from utils import ball_boundary_uniform, Pde, SYS_MODES, DISTILL_MODES
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import sys
from torch.utils.data import TensorDataset

# Model
class PdeNet(torch.nn.Module):
    def __init__(
            self,
            pde: Pde,
            hidden_units: list,
            lr_init: float,
            device: str,
            bc_weight: float=1.,
            ic_weight: float=1.,
            phy_weight: float=1.,
            distill_weight: float=1.,
            ewc_weight: float=1.,
            activation: nn.Module=nn.Tanh(),
            last_activation: bool=True,
            *args,
            **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)
        
        # Save the parameters
        self.pde = pde
        self.bc_weight = bc_weight
        self.ic_weight = ic_weight
        self.phy_weight = phy_weight
        self.hidden_units = hidden_units
        self.distill_weight = distill_weight
        self.ewc_weight = ewc_weight
        self.lr_init = lr_init
        self.activation = activation
        self.device = device

        # Define the net, first layer
        net_dict = OrderedDict(
            {'lin0': nn.Linear(2, hidden_units[0]),
            'act0': activation}
        )

        # Define the net, hidden layers
        for i in range(1, len(hidden_units)):
            net_dict.update({f'lin{i}': nn.Linear(in_features=hidden_units[i-1], out_features=hidden_units[i])})
            net_dict.update({f'act{i}': activation})

        # Define the net, last layer
        net_dict.update({f'lin{len(hidden_units)}': nn.Linear(in_features=hidden_units[-1], out_features=1)})
        
        # Define the last activation
        if last_activation:
            net_dict.update({f'act{len(hidden_units)}': activation})

        # Save the network
        self.net = nn.Sequential(net_dict).to(self.device)

        # Define the optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr_init)

        # define the loss container: average over all elements (it always return a scalar in R)
        self.loss_container = nn.MSELoss(reduction='mean')
    
    # x = [[m1, m2, ..., md], ..., [M1, M2, ..., Md]] -> net(x) = [^u([m1, ..., md]), ..., ^u([M1, ..., Md])]
    # Forward function
    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        return self.net(x)
    
    # [x1, x2, ..., xd] -> [[x1, x2, ..., xd]] -> net([[x1, x2, ..., xd]]) = [^u([x1, ..., xd])] -> ^u([x1, ..., xd])
    # Forward function for individual samples
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape((1,-1))).reshape((-1))
    
    # Concatenates all parameters (weights and biases) of a model into a single 1D tensor.
    def get_weights(self) -> torch.Tensor:
        # p.view(-1) flattens each parameter tensor into a 1D tensor
        # torch.cat() concatenates 1D tensors into a single 1D tensor
        return torch.cat([param.view(-1) for param in self.parameters() if param.requires_grad])
    

    # loss function (1st order distillation, phase 2)
    def loss_fn(self,
        x_new: torch.Tensor, # new points inside the domain (in R^(nxd))
        y_new: torch.Tensor = None, # labels of points inside the domain (typically not available) (in R^(nx1))
        Dy_new: torch.Tensor = None,
        Hy_new: torch.Tensor = None,
        force_new: torch.Tensor = None,
        x_bc: torch.Tensor = None, # boundary points (in R^(nxd))
        y_bc: torch.Tensor = None, # boundary values (in R^(nx1))
        x_ic: torch.Tensor = None, # initial points (in R^(nxd))
        y_ic: torch.Tensor = None, # initial values (in R^(nx1))
        x_distill: torch.Tensor = None, # distillation points inside the domain (in R^(nxd))
        y_distill: torch.Tensor = None, # distillation target values (in R^(nx1))
        Dy_distill: torch.Tensor = None, # distillation target values of the 1st derivative (previous model) (in R^(nxd))
        Hy_distill: torch.Tensor = None, # distillation target values of the Hessian (previous model) (in R^(nxdxd))
        force_distill: torch.Tensor = None,
        distill_on_actual_data: bool = False,
        sys_mode: str = 'Output', # defines the system loss:
                    # Output: 0th derivative,
                    # PINN: PDE residual,
                    # Output+PINN: 0th derivative and PDE residual,
                    # Derivative: 1st derivative,
                    # Hessian: 2nd derivative,
                    # Derivative+Hessian: 1st and 2nd derivatives,
                    # Sobolev: 0th and 1st derivatives,
                    # Sobolev+Hessian: 0th, 1st and 2nd derivatives
        distill_mode: str = 'Forgetting', # defines the distillation loss:
                    # Forgetting: no distillation,
                    # Output: distill the 0th derivative,
                    # PINN: distill the residual,
                    # Derivative: distill the 1st derivative,
                    # Hessian: distill the 2nd derivative,
                    # Derivative+Hessian: distill the 1st and 2nd derivatives,
                    # Sobolev: 0th and 1st derivatives,
                    # Sobolev+Hessian: distill the 0th, the 1st and 2nd derivatives
        ewc_mode: str = 'Off', # Off -> no elastic weight consolidation performed,
                               # On ->  elastic weight consolidation performed
        ewc_params: torch.Tensor = None, # optimal params of a previous model
        ewc_fisher_diag: torch.Tensor = None # diagonal elements of the fisher information mtx relative to ewc_params,
                                             # computed on some data
    ) -> torch.Tensor:
        # Check that the sys_mode and distill_mode parameters are correct
        if sys_mode not in SYS_MODES:
            raise ValueError(f'sys_mode should be in {SYS_MODES}, but found {sys_mode}')
        if distill_mode not in DISTILL_MODES:
            raise ValueError(f'sys_mode should be in {DISTILL_MODES}, but found {distill_mode}')
        
        y_new_pred = None
        Dy_new_pred = None
        Hy_new_pred = None

        # ---------------------- SYSTEM LEARNING ----------------------------
        #
        # Note: The only unsupervised system learning modality is the PINN one;
        #       all the others use a lable (the solution value for Output,
        #       the 1st derivative value for Derivative, the 2nd derivative for Hessian, ...)

        if sys_mode == 'Output':
            # Get the prediction on the new points
            y_new_pred = self.forward(x_new) # in R^(nx1)

            phy_loss = self.loss_container(y_new_pred, y_new)
        
        elif sys_mode == 'PINN':
            # Get the prediction on the new points
            y_new_pred = self.forward(x_new) # in R^(nx1)

            # Evaluate 2nd derivative of the network wrt the input at the new points
            Hy_new_pred = vmap(hessian(self.forward_single))(x_new)[:,0,:,:] # in R^(nxdxd)

            # lambda*(uxx + uyy) - u + u^3 = 0
            # pde residual loss on the new points
            pde_residual = self.pde.residual(x=x_new, u_pred=y_new_pred.reshape((-1)), Hu_pred=Hy_new_pred, force=force_new)
            phy_loss = self.loss_container(pde_residual, torch.zeros_like(pde_residual)) # in R
            # phy_loss = self.loss_container(pde_pred, allen_cahn_forcing(x_new)) # in R
        
        elif sys_mode == 'Output+PINN':
            # Get the prediction on the new points
            y_new_pred = self.forward(x_new) # in R^(nx1)

            out_loss = self.loss_container(y_new_pred, y_new)

            # Evaluate 2nd derivative of the network wrt the input at the new points
            Hy_new_pred = vmap(hessian(self.forward_single))(x_new)[:,0,:,:] # in R^(nxdxd)

            # pde residual loss on the new points
            pde_residual = self.pde.residual(x=x_new, u_pred=y_new_pred.reshape((-1)), Hu_pred=Hy_new_pred, force=force_new)
            pinn_loss = self.loss_container(pde_residual, torch.zeros_like(pde_residual)) # in R

            phy_loss = out_loss + pinn_loss

        elif sys_mode == 'Derivative':
            # Evaluate the 1st derivative of the network wrt input at the new points
            Dy_new_pred = vmap(jacrev(self.forward_single))(x_new)[:, 0, :] # in R^(nxd)

            # 1st derivative loss on new points
            phy_loss = self.loss_container(Dy_new_pred, Dy_new) # in R

        elif sys_mode == 'Hessian':
            # Evaluate the 2nd derivative of the network wrt input at the new points
            Hy_new_pred = vmap(hessian(self.forward_single))(x_new)[:,0,:,:] # in R^(nxdxd)

            # 2nd derivative loss on new points
            phy_loss = self.loss_container(Hy_new_pred, Hy_new) # in R

        elif sys_mode == 'Derivative+Hessian':
            # Evaluate the 1st derivative of the network wrt input at the new points
            Dy_new_pred = vmap(jacrev(self.forward_single))(x_new)[:,0,:] # in R^(nxd)

            # Evaluate the 2nd derivative of the network wrt input at the new points
            Hy_new_pred = vmap(hessian(self.forward_single))(x_new)[:,0,:,:] # in R^(nxdxd)

            # 1st derivative loss on new points
            der_loss = self.loss_container(Dy_new_pred, Dy_new) # in R

            # 2nd derivative loss on new points
            hes_loss = self.loss_container(Hy_new_pred, Hy_new) # in R

            # Derivative+Hessian loss on new points
            phy_loss = der_loss + hes_loss # in R

        elif sys_mode == 'Sobolev':
            # Get the prediction on new points
            y_new_pred = self.forward(x_new) # in R^(nx1)

            # Evaluate the 1st derivative of the network wrt input at the new points
            Dy_new_pred = vmap(jacrev(self.forward_single))(x_new)[:,0,:] # in R^(nxd)

            # 1st derivative loss on new points
            der_loss = self.loss_container(Dy_new_pred, Dy_new) # in R

            # 0th derivative loss on new points
            out_loss = self.loss_container(y_new_pred, y_new) # in R

            # Sobolev loss on new points
            phy_loss = der_loss + out_loss # in R

        elif sys_mode == 'Sobolev+Hessian':
            # Get the prediction on new points
            y_new_pred = self.forward(x_new) # in R^(nx1)

            # Evaluate the 1st derivative of the network wrt input at the new points
            Dy_new_pred = vmap(jacrev(self.forward_single))(x_new)[:, 0, :] # in R^(nxd)

            # Evaluate the 2nd derivative of the network wrt input at the new points
            Hy_new_pred = vmap(hessian(self.forward_single))(x_new)[:,0,:,:] # in R^(nxdxd)
            
            # 0th derivative loss on new points
            out_loss = self.loss_container(y_new_pred, y_new) # in R

            # 1st derivative loss on distillation points
            der_loss = self.loss_container(Dy_new_pred, Dy_new) # in R

            # Sobolev loss on new points
            sob_loss = der_loss + out_loss # in R

            # 2nd derivative loss on distillation points
            hes_loss = self.loss_container(Hy_new_pred, Hy_new) # in R

            # Sobolev+Hessian loss on new points
            phy_loss = sob_loss + hes_loss # in R

        if x_bc is not None:
            # Get the prediction on boundary points
            y_bc_pred = self.forward(x_bc) # in R^(nx1)

            # boundary loss on boundary points
            bc_loss = self.loss_container(y_bc_pred.reshape((-1)), y_bc.reshape((-1))) # in R
        else:
            bc_loss = torch.tensor([0.], device=self.device) # in R
        
        if x_ic is not None:
            # Get the prediction on initial points
            y_ic_pred = self.forward(x_ic) # in R^(nx1)

            # loss on initial points
            ic_loss = self.loss_container(y_ic_pred.reshape((-1)), y_ic.reshape((-1))) # in R
        else:
            ic_loss = torch.tensor([0.], device=self.device) # in R

        # system loss on the new points
        sys_loss = self.phy_weight * phy_loss + self.bc_weight * bc_loss + self.ic_weight * ic_loss # in R

        # -------------------------- DISTILLATION ------------------------------
        #
        # Note: In a more general way than in the classic setting, here
        #       distillation is intended as applicable not only to distill knowledge
        #       from another model, but also from some data. This motivates the PINN
        #       distillation modality, that is actually not supervised.
        #
        # Note: If distill_on_actual_data is True, use x_new as distillation points,
        #       possibly avoiding repeated computations of model output and derivatives.
        #       In this case, the assumption is that y, Dy and Hy contain the outputs of
        #       the previous model on the points x_new.

        y_pred = None
        Dy_pred = None
        Hy_pred = None

        if distill_on_actual_data:
            y_pred = y_new_pred
            Dy_pred = Dy_new_pred
            Hy_pred = Hy_new_pred

        if distill_mode == 'Forgetting':
            # zero distillation loss
            distill_loss = torch.tensor([0.], device=self.device) # in R
        
        elif distill_mode == 'Output':
            if y_pred is None:
                # Get the prediction on distillation points
                y_pred = self.forward(x_distill) # in R^(nx1)

            # 0th derivative loss on distillation points
            distill_loss = self.loss_container(y_pred, y_distill) # in R

        elif distill_mode == 'PINN':
            if y_pred is None:
                # Get the prediction on the distillation points
                y_pred = self.forward(x_distill) # in R^(nx1)

            if Dy_pred is None:
                # Evaluate 1st derivative of the network wrt the input at the distillation points
                Dy_pred = vmap(jacrev(self.forward_single))(x_distill)[:,0,:] # in R^(nxd)

            if Hy_pred is None:
                # Evaluate 2nd derivative of the network wrt the input at the distillation points
                Hy_pred = vmap(hessian(self.forward_single))(x_distill)[:,0,:,:] # in R^(nxdxd)
            
            # lambda*(uxx + uyy) - u + u^3 = 0
            # pde residual loss on the distillation points
            pde_residual = self.pde.residual(x=x_distill, u_pred=y_pred.reshape((-1)), Hu_pred=Hy_pred, force=force_distill)
            distill_loss = self.loss_container(pde_residual, torch.zeros_like(pde_residual)) # in R                
            # distill_loss = self.loss_container(pde_pred, allen_cahn_forcing(x)) # in R
        
        elif distill_mode == 'Derivative':
            if Dy_pred is None:
                # Evaluate the 1st derivative of the network wrt input at the distillation points
                Dy_pred = vmap(jacrev(self.forward_single))(x_distill)[:, 0, :] # in R^(nxd)

            # 1st derivative loss on distillation points
            distill_loss = self.loss_container(Dy_pred, Dy_distill) # in R
            
        elif distill_mode == 'Hessian':
            if Hy_pred is None:
                # Evaluate the 2nd derivative of the network wrt input at the distillation points
                Hy_pred = vmap(hessian(self.forward_single))(x_distill)[:,0,:,:] # in R^(nxdxd)

            # 2nd derivative loss on distillation points
            distill_loss = self.loss_container(Hy_pred, Hy_distill) # in R

        elif distill_mode == 'Derivative+Hessian':
            if Dy_pred is None:
                # Evaluate the 1st derivative of the network wrt input at the distillation points
                Dy_pred = vmap(jacrev(self.forward_single))(x_distill)[:,0,:] # in R^(nxd)

            if Hy_pred is None:
                # Evaluate the 2nd derivative of the network wrt input at the distillation points
                Hy_pred = vmap(hessian(self.forward_single))(x_distill)[:,0,:,:] # in R^(nxdxd)

            # 1st derivative loss on distillation points
            der_loss = self.loss_container(Dy_pred, Dy_distill) # in R

            # 2nd derivative loss on distillation points
            hes_loss = self.loss_container(Hy_pred, Hy_distill) # in R

            # Derivative+Hessian loss on distillation points
            distill_loss = der_loss + hes_loss # in R
        
        elif distill_mode == 'Sobolev':
            if y_pred is None:
                # Get the prediction on distillation points
                y_pred = self.forward(x_distill) # in R^(nx1)

            if Dy_pred is None:
                # Evaluate the 1st derivative of the network wrt input at the distillation points
                Dy_pred = vmap(jacrev(self.forward_single))(x_distill)[:,0,:] # in R^(nxd)

            # 1st derivative loss on distillation points
            der_loss = self.loss_container(Dy_pred, Dy_distill) # in R

            # 0th derivative loss on distillation points
            out_loss = self.loss_container(y_pred, y_distill) # in R

            # Sobolev loss on distillation points
            distill_loss = der_loss + out_loss # in R
        
        elif distill_mode == 'Sobolev+Hessian':
            if y_pred is None:
                # Get the prediction on distillation points
                y_pred = self.forward(x_distill) # in R^(nx1)

            if Dy_pred is None:
                # Evaluate the 1st derivative of the network wrt input at the distillation points
                Dy_pred = vmap(jacrev(self.forward_single))(x_distill)[:,0,:] # in R^(nxd)

            if Hy_pred is None:
                # Evaluate the 2nd derivative of the network wrt input at the distillation points
                Hy_pred = vmap(hessian(self.forward_single))(x_distill)[:,0,:,:] # in R^(nxdxd)
            
            # 0th derivative loss on distillation points
            out_loss = self.loss_container(y_pred, y_distill) # in R

            # 1st derivative loss on distillation points
            der_loss = self.loss_container(Dy_pred, Dy_distill) # in R

            # Sobolev loss on distillation points
            sob_loss = der_loss + out_loss # in R

            # 2nd derivative loss on distillation points
            hes_loss = self.loss_container(Hy_pred, Hy_distill) # in R

            # Sobolev+Hessian loss on distillation points
            distill_loss = sob_loss + hes_loss # in R
        
        # -------------------------------- EWC --------------------------------

        if ewc_mode == 'Off' or ewc_params is None or ewc_fisher_diag is None:
            ewc_loss = torch.tensor([0.], device=self.device) # in R
        else:
            ewc_loss = torch.mean(ewc_fisher_diag * ((self.get_weights() - ewc_params) ** 2))
    
        # -------------------------------- Total loss --------------------------------
        tot_loss = sys_loss + self.distill_weight * distill_loss + self.ewc_weight * ewc_loss # in R

        return tot_loss

    # Function that evaluate the various losses (1st order distillation, phase 2)
    def eval_losses(self,
        x: torch.Tensor, # points inside the domain (in R^(nxd))
        y: torch.Tensor, # target values of the solution (in R^(nx1))
        Dy: torch.Tensor, # target values of the 1st derivative of the solution (in R^(nxd))
        Hy: torch.Tensor, # target values of the Hessian of the solution (in R^(nxdxd))
        force: torch.Tensor,
        sys_mode: str = 'Output', # defines the system loss:
                    # Output: 0th derivative,
                    # PINN: PDE residual,
                    # Output+PINN: 0th derivative and PDE residual,
                    # Derivative: 1st derivative,
                    # Hessian: 2nd derivative,
                    # Derivative+Hessian: 1st and 2nd derivatives,
                    # Sobolev: 0th and 1st derivatives,
                    # Sobolev+Hessian: 0th, 1st and 2nd derivatives
        x_bc: torch.Tensor = None, # boundary points (in R^(nxd))
        y_bc: torch.Tensor = None, # boundary values (in R^(nx1))
        x_ic: torch.Tensor = None, # initial points (in R^(nxd))
        y_ic: torch.Tensor = None # initial values (in R^(nx1))
        #print_to_screen: bool = False,    
    ):
        # Check that the mode parameter is correct
        if sys_mode not in SYS_MODES:
            raise ValueError(f'mode should be in {SYS_MODES}, but found {sys_mode}')    
        
        # Get the prediction
        y_pred = self.forward(x)

        # Evaluate the 1st derivative of the network at the points
        Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]

        # Evaluate the 1st derivative of the network at the points
        Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:,:]
            
        # lambda*(uxx + uyy) - u + u^3 = 0
        # Apply the differential operator without external forcing
        # pde_pred = LAMBDA * (Hy_pred[:, 0, 0] + Hy_pred[:, 1, 1]) - y_pred.reshape((-1)) + y_pred.reshape((-1))**3

        # pde residual loss
        pde_residual = self.pde.residual(x=x, u_pred=y_pred.reshape((-1)), Hu_pred=Hy_pred, force=force)
        pde_loss = self.loss_container(pde_residual, torch.zeros_like(pde_residual)) # in R
        # pde_loss = self.loss_container(pde_pred, allen_cahn_forcing(x))
        
        # 1st derivative loss
        der_loss = self.loss_container(Dy_pred, Dy)
        
        # 0th derivative loss
        out_loss = self.loss_container(y_pred, y)
        
        # 2nd derivative loss
        hes_loss = self.loss_container(Hy_pred, Hy)        
        
        # Sobolev loss
        sob_loss = out_loss + der_loss
        
        # weight for the total loss
        if sys_mode == 'PINN':
            phy_loss = pde_loss

        elif sys_mode == 'Derivative':
            phy_loss = der_loss

        elif sys_mode == 'Output':
            phy_loss = out_loss

        elif sys_mode == 'Sobolev':
            phy_loss = sob_loss

        elif sys_mode == 'Hessian':
            phy_loss = hes_loss

        elif sys_mode == 'Derivative+Hessian':
            phy_loss = der_loss + hes_loss

        elif sys_mode == 'Sobolev+Hessian':
            phy_loss = sob_loss + hes_loss

        elif sys_mode == 'Forgetting':
            phy_loss = torch.tensor([0.], device=self.device)

        else:
            phy_loss = sob_loss

        if x_bc is not None:
            # Get the prediction on boundary points
            y_bc_pred = self.forward(x_bc)

            # boundary loss on boundary points
            bc_loss = self.loss_container(y_bc_pred.reshape((-1)), y_bc.reshape((-1)))
        else:
            bc_loss = torch.tensor([0.], device=self.device)
        
        if x_ic is not None:
            # Get the prediction on initial points
            y_ic_pred = self.forward(x_ic)

            # loss on initial points
            ic_loss = self.loss_container(y_ic_pred.reshape((-1)), y_ic.reshape((-1)))
        else:
            ic_loss = torch.tensor([0.], device=self.device)
        
        # Total loss
        sys_loss = self.phy_weight * phy_loss + self.bc_weight * bc_loss + self.ic_weight * ic_loss
    
        return out_loss, der_loss, hes_loss, pde_loss, bc_loss, ic_loss, sys_loss
    

    # compute the 2nd derivative loss term of Sobolev+Hessian loss, using finite differences approximation
    #   - compute the 1st derivative of the model,
    #   - evaluate it on x_pde,
    #   - compute the finite differences to approximate the 2nd derivative in x_pde
    #   - compute the error btw this approximation and the provided H_pde 
    def hessian_sobolev_error(self, x_pde: torch.Tensor, H_pde: torch.Tensor):
        # x_pde in R^(nxd), H_pde in R^(nxdxd) (value of the hessian in n different points)

        # sample n=100 d=2-dimentional points uniformly on the unit ball boundary
        # rand_vec is a batch of n d-dimentional vectors that are the directions
        # used for the computation (approximation) of the directional second derivatives H*rand_vec_i
        rand_vec = ball_boundary_uniform(n=100, radius=1., dim=2).to(self.device) # in R^(nxd) = R^(100x2)

        # Noise
        sigma = 0.01
        noise = rand_vec * sigma # in R^(nxd)

        # Finite difference approximation of the directional second derivatives
        # ~ [H*rand_vec_1, ..., H*rand_vec_n]
        def rand_proj(x):
            # x in R^d

            # compute the gradient wrt the input and apply it to x
            der_pred = jacrev(self.forward_single)(x) # in R^d

            # compute the gradient wrt the input and apply it to each x+noisei, i in {1, ..., n}
            # x+noise = [x+noise1, ..., x+noisen] in R^(nxd) (x+noisei in R^d)
            der_eps_pred = vmap(jacrev(self.forward_single))(x+noise) # in R^(nx1xd)

            # finite difference approximation of the directional second derivatives
            # ~ [H*rand_vec_1, ..., H*rand_vec_n]
            # der_eps_pred[:,0,:]-der_pred = [der_eps_pred[:,0,:]_1-der_pred, ..., der_eps_pred[:,0,:]_n-der_pred] in R^(nxd)
            hess_pred = (der_eps_pred[:,0,:]-der_pred)/sigma # in R^(nxd)

            return hess_pred
        
        # Finite difference approximations of the 2nd directional derivatives H*rand_vec_i, i in {1, ..., n}
        hess_proj_pred = vmap(rand_proj)(x_pde) # in  R^(nx1xd)

        # The true directional 2nd derivatives H*rand_vec_i, i in {1, ..., n}
        hess_proj_true = torch.einsum('bij,pj->bpi', H_pde, rand_vec)
        # equivalent to
        # hess_proj_true = torch.matmul(H_pde, rand_vec.T)  # in R^(nxdxd); b=n(#Hessians), i=j=d (Hessian dimentions, directions dimention), p=n(#directions); (b, i, j) @ (j, p) -> (b, i, p)
        # hess_proj_true = hess_proj_true.permute(0, 2, 1)  # -> (b, p, i)

        error = torch.norm(hess_proj_pred - hess_proj_true, p=2, dim=2).mean()
        return error
    
    # Label the dataset specified by data_path using the model specified by model_path
    def label(self, data_path=None, dataset=None, save=False):
        # Seed
        seed = 30
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed)

        if dataset is None:
            if not os.path.exists(data_path):
                print(f'ERROR: {data_path} file not found.', file=sys.stderr)
                sys.exit(1)

            dataset = torch.load(os.path.join(data_path), weights_only=False)

        dataloader = DataLoader(dataset, batch_size=1024, generator=gen, shuffle=False)

        x_d_list = []
        y_d_list = []
        Dy_d_list = []
        Hy_d_list = []
        force_d_list = []
        params_d_list = []

        # Set the model to evaluation mode.
        self.eval()
        with torch.no_grad():
            for x in dataloader:
                x_d = x[0].to(self.device).float().requires_grad_(True)
                y_d = self.forward(x_d)
                Dy_d = vmap(jacrev(self.forward_single))(x_d)[:,0,:]
                Hy_d = vmap(hessian(self.forward_single))(x_d)[:,0,:,:]
                force_d = x[4].to(self.device).float().requires_grad_(True)
                params_d = x[5].to(self.device).float().requires_grad_(True)

                x_d_list.append(x_d)
                y_d_list.append(y_d)
                Dy_d_list.append(Dy_d)
                Hy_d_list.append(Hy_d)
                force_d_list.append(force_d)
                params_d_list.append(params_d)

            x_d_full = torch.cat(x_d_list, dim=0)
            y_d_full = torch.cat(y_d_list, dim=0)
            Dy_d_full = torch.cat(Dy_d_list, dim=0)
            Hy_d_full = torch.cat(Hy_d_list, dim=0)
            force_d_full = torch.cat(force_d_list, dim=0)
            params_d_full = torch.cat(params_d_list, dim=0)

            labeled_dataset = TensorDataset(x_d_full, y_d_full, Dy_d_full, Hy_d_full, force_d_full, params_d_full)

            # save the full dataset
            if save:
                labeled_data_path = data_path.rsplit('.', 1)[0] if '.' in data_path else data_path
                labeled_data_path += '_labeled.pth'
                torch.save(labeled_dataset, f'{labeled_data_path}')

            return labeled_dataset

    # Return the diagonal of the Fisher information mtx associated with the model parameters, computed on the data specified by the dataloader
    def get_fisher_diag(self, dataset: TensorDataset, sys_mode: str) -> torch.Tensor:
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

        fisher_diag = {name: torch.zeros_like(param).to(self.device).float() for name, param in self.named_parameters() if param.requires_grad}

        self.eval()
        for z in dataloader:
            x = z[0].to(self.device).float()
            y = z[1].to(self.device).float()
            self.zero_grad()
            loss = self.loss_fn(
                x_new = x,
                y_new = y,
                sys_mode = sys_mode,
                distill_mode = 'Forgetting',
                ewc_mode = 'Off'
            )
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_diag[name] += (param.grad ** 2).detach()

        # Average over all samples
        num_batches = len(dataloader)
        for name in fisher_diag:
            fisher_diag[name] /= num_batches

        # Get diagonal Fisher information vector
        fisher_diag_vector = torch.cat([fisher_diag[name].view(-1) for name, param in self.named_parameters() if param.requires_grad])

        return fisher_diag_vector


    def evaluate(self, data_path: str):
        # Seed
        seed = 30
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed)

        if not os.path.exists(data_path):
            print(f'ERROR: {data_path} file not found.', file=sys.stderr)
            sys.exit(1)
        dataset = torch.load(os.path.join(data_path), weights_only=False)
        dataloader = DataLoader(dataset, batch_size=1024, generator=gen, shuffle=False)

        out_tot_loss = 0.0
        der_tot_loss = 0.0
        hes_tot_loss = 0.0
        pde_tot_loss = 0.0

        # Set the model to evaluation mode.
        self.eval()
        with torch.no_grad():
            for data in dataloader:
                x = data[0].to(self.device).float().requires_grad_(True)

                y_true = data[1].to(self.device).float().requires_grad_(True)
                Dy_true = data[2].to(self.device).float().requires_grad_(True)
                Hy_true = data[3].to(self.device).float().requires_grad_(True)
                force_true = data[4].to(self.device).float().requires_grad_(True)

                out_loss, der_loss, hes_loss, pde_loss, _, _, _ = self.eval_losses(x=x, y=y_true, Dy=Dy_true, Hy=Hy_true, force=force_true)

                out_tot_loss += out_loss
                der_tot_loss += der_loss
                hes_tot_loss += hes_loss
                pde_tot_loss += pde_loss
            
            out_tot_loss /= len(dataloader)
            der_tot_loss /= len(dataloader)
            hes_tot_loss /= len(dataloader)
            pde_tot_loss /= len(dataloader)

        return out_tot_loss, der_tot_loss, hes_tot_loss, pde_tot_loss

# Load a saved model from a .pth file
def resume_model(model_path: str, device='cpu') -> PdeNet:
    if not os.path.exists(model_path):
            print(f'ERROR: {model_path} file not found.', file=sys.stderr)
            sys.exit(1)

    # Load the entire checkpoint dictionary
    checkpoint = torch.load(model_path, weights_only=False)

    # Extract the architectural parameters
    pde_name = checkpoint['pde']
    pde_params = checkpoint['pde_params']
    hidden_units = checkpoint['hidden_units']
    lr_init = checkpoint['lr_init']
    bc_weight = checkpoint['bc_weight']
    ic_weight = checkpoint['ic_weight']
    phy_weight = checkpoint['phy_weight']
    distill_weight = checkpoint['distill_weight']
    activation = checkpoint['activation']

    # Create a new instance of the distillation model architecture using the loaded parameters
    model = PdeNet(
        pde=Pde(pde_name, pde_params),
        bc_weight=bc_weight,
        ic_weight=ic_weight,
        phy_weight=phy_weight,
        distill_weight=distill_weight,
        hidden_units=hidden_units,
        lr_init=lr_init,
        device=device,
        activation=activation
        ).to(device)

    # Load the distillation model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    return model