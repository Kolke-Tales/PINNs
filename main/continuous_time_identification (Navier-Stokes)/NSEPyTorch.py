"""
@author: Sayan Chowdhury
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.insert(0, 'E:/RESEARCH_SHIP_RESISTANCE_AI_ML/codes/NSE/PINNs/Utilities')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from tqdm import tqdm

torch.manual_seed(1234)

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the RANS equations
def nse(u, v, p, x, y, t, Re):
    # Define the constants
    rho = 1.0
    mu = rho / Re
    
    # Compute the gradients
    u_x = torch.autograd.grad(u, x, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, create_graph=True)[0]    
    u_t = torch.autograd.grad(u, t, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, create_graph=True)[0]
    
    # Compute the residuals
    f_u = rho * u_x + rho * v_y - torch.autograd.grad(p, x, create_graph=True)[0] + mu * (u_xx + u_yy) - rho * u_t
    f_v = rho * u_x + rho * v_y - torch.autograd.grad(p, y, create_graph=True)[0] + mu * (v_xx + v_yy) - rho * v_t
    f_p = u_x + v_y
    
    return f_u, f_v, f_p

