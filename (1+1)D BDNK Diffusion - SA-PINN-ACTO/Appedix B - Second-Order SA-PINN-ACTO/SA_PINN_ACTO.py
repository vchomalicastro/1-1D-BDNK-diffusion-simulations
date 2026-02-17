import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.autograd

from BDNK_Functions import *

class PINN_BDNK_1D(nn.Module):
    def __init__(self, Nl, Nn, lb, ub):
        super().__init__()
        # Scaling from physical units to [-1,1]
        self.register_buffer('lb', torch.as_tensor(lb, dtype=torch.get_default_dtype()))
        self.register_buffer('ub', torch.as_tensor(ub, dtype=torch.get_default_dtype()))
        self.register_buffer('sA',  torch.tensor(1.0, dtype=torch.get_default_dtype()))

        # Base network (produces raw candidates)
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, Nl):
            self.net.add_module(f'Linear_layer_{num}', nn.Linear(Nn, Nn))
            self.net.add_module(f'Tanh_layer_{num}', nn.Tanh())
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 1))  # returns only alpha_raw

        # IC algebraic enforcement toggles
        self.alpha_ic_func = None

    # Helpers
    def _scale(self, X):  # (t,x) in physical -> [-1,1]
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    # Autograd utilities
    def gradients(self, y, x, retain_graph=True):
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=retain_graph
        )[0]

    # Forward with hard IC
    def forward(self, X_phys):
        t_phys = X_phys[:, 0:1]
        x_phys = X_phys[:, 1:2]
        
        L = float(self.ub[1].item())
        xL1 = torch.full((1, 1), -L, dtype=X_phys.dtype, device=X_phys.device)
        xR1 = torch.full((1, 1),  L, dtype=X_phys.dtype, device=X_phys.device)
        X_L  = torch.cat([t_phys, xL1.expand_as(x_phys)], dim=1)
        X_R  = torch.cat([t_phys, xR1.expand_as(x_phys)], dim=1)
    
        # Raw network outputs
        X_cat = torch.cat([X_phys, X_L, X_R], dim=0)
        Z_cat = self._scale(X_cat)
        raw_cat = self.net(Z_cat)
    
        N = X_phys.shape[0]
        raw    = raw_cat[:N]
        raw_L  = raw_cat[N:2*N]
        raw_R  = raw_cat[2*N:]

        if self.alpha_ic_func is None:
            raise RuntimeError("IC function must be set before forward pass.")
    
        alpha_ic_x  = self.alpha_ic_func(x_phys)
        alpha_ic_L1 = self.alpha_ic_func(xL1)
        alpha_ic_R1 = self.alpha_ic_func(xR1)
    
        g = torch.exp(-1.0 * t_phys)  # g(0)=1
        h = 1.0 - g                   # h(0)=0

        alpha_enf_scaled    = g * alpha_ic_x    + h * raw
        alpha_enf_scaled_L  = g * alpha_ic_L1   + h * raw_L
        alpha_enf_scaled_R  = g * alpha_ic_R1   + h * raw_R

        # Enforce periodic boundary conditions on raw outputs
        xi = (x_phys + L) / (2.0 * L)
        alpha_periodic_scaled = alpha_enf_scaled - (alpha_enf_scaled_R - alpha_enf_scaled_L) * xi
        
        # Scale back to physical units
        alpha_phys = self.sA * alpha_periodic_scaled
        return alpha_phys

    # Residuals
    def pde_residual(self, X):
        X = X.requires_grad_(True)
        t = X[:, 0:1]
        x = X[:, 1:2]

        alpha = self.forward(X)

        T     = T_func(t, x)
        n     = n_from_alpha_func(alpha, T)
        sigma = sigma_func(alpha, T)
        lambd = lambd_func(sigma)

        g_n  = self.gradients(n, X)
        n_t  = g_n[:, 0:1]

        g_a     = self.gradients(alpha, X)
        alpha_t = g_a[:, 0:1]
        alpha_x = g_a[:, 1:2]

        q   = lambd * T * alpha_t
        g_q = self.gradients(q, X)
        q_t = g_q[:, 0:1]

        p   = sigma * T * alpha_x
        g_p = self.gradients(p, X)
        p_x = g_p[:, 1:2]

        R = n_t + q_t - p_x
        return R / self.sA

    def loss_ic(self, X_ic_t, alpha1_ic_t):
        X_ic_t = X_ic_t.clone().detach().requires_grad_(True)
        alpha_ic_pred = self.forward(X_ic_t)
        g_alpha = self.gradients(alpha_ic_pred, X_ic_t)
        alpha_t_pred = g_alpha[:, 0:1]

        return ((alpha_t_pred - alpha1_ic_t)**2).mean() / self.sA

    def loss_pde(self, X_colloc):
        return (self.pde_residual(X_colloc)**2).mean()

    def ic_residual(self, *args, **kwargs):
        return (torch.zeros(1,1, device=self.lb.device), torch.zeros(1,1, device=self.lb.device))

    def loss_bc(self, xL, xR):
        outL = self.forward(xL); outR = self.forward(xR)
        return ((outL - outR)**2).mean()

    def loss_mass(self, t_grid, x_grid):
        z = torch.zeros(1, device=self.lb.device, dtype=self.lb.dtype)
        return z
