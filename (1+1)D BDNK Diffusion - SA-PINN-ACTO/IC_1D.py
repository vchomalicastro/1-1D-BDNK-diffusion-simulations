import numpy as np
import torch
from BDNK_Functions import *

def IC_BDNK(x, L):
    N_c, N_f = 3, 3
    N = x.shape[0]
    t_vals = x[:, 0:1]
    x_vals = x[:, 1:2]

    # Initial condition n(t=0,x)
    p, q, r = 0.2, 7.0, 1.0      # Initial condition parameters
    
    # First setup:
    n = p * torch.exp(- (q * x_vals / L)**2) + r
    
    # Second setup:
    # sharpness = 60
    # n = (1.1 - 0.1*torch.tanh(sharpness*((4*x_vals/L)**2-1)))
    
    # Third setup:
    #n = 1e-3*(p * torch.exp(- (q * x_vals / L)**2) + r)
    
    T = T_func(t_vals, x_vals)
    
    alpha = alpha_from_n_func(n, T)
    
    dx = x_vals[1] - x_vals[0]
    alpha_padded = torch.cat([alpha[:1], alpha, alpha[-1:]], dim=0)
    N_x = -(alpha_padded[2:] - alpha_padded[:-2]) / (2.0 * dx)

    
    # Initial condition J^0(t=0,x)
    d, f, g = 0.05, 10.0, 1.05      # Initial condition parameters

    # First setup:
    J0 = d * torch.exp(- (f * x_vals / L)**2) + g

    # Second setup:
    #J0 = torch.ones_like(x_vals)*g

    # Third setup:
    #J0 = 1e-3 * torch.ones_like(x_vals)*g

    return J0, alpha, N_x