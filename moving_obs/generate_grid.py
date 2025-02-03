import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch.nn.functional as F
import collections
import numpy as np

UPPER_X = 0.5 #torch.max(traj_states[:,:,0])
LOWER_X = -0.5 # -UPPER_X
UPPER_Y = 0.1 #torch.max(traj_states[:,:,0])
LOWER_Y = -0.1 # -UPPER_X
UPPER_THETA = 0.1 #torch.max(traj_states[:,:,1])
LOWER_THETA = -0.1 #-UPPER_THETA

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

def split_grid_init(bound_upper = [UPPER_X, UPPER_Y, UPPER_THETA], bound_lower = [LOWER_X, LOWER_Y, LOWER_THETA], grid_size = 16):
    
    x_max = bound_upper[0]
    y_max = bound_upper[1]
    theta_max = bound_upper[2]

    x_min = bound_lower[0]
    y_min = bound_lower[1]
    theta_min = bound_lower[2]

    # Calculate grid cell size
    x_step = (x_max - x_min) / grid_size
    y_step = (y_max - y_min) / grid_size
    theta_step = (theta_max - theta_min) / grid_size

    # Generate grid coordinates
    x_coords = torch.linspace(x_min, x_max, grid_size + 1)
    y_coords = torch.linspace(y_min, y_max, grid_size + 1)
    theta_coords = torch.linspace(theta_min, theta_max, grid_size + 1)

    x_midpoints = (x_coords[:-1] + x_coords[1:]) / 2
    y_midpoints = (y_coords[:-1] + y_coords[1:]) / 2
    theta_midpoints = (theta_coords[:-1] + theta_coords[1:]) / 2

    # Create grid tensor
    X_mid, Y_mid, THETA_mid = torch.meshgrid(x_midpoints, y_midpoints, theta_midpoints, indexing='ij')
    grid = torch.stack((X_mid, Y_mid, THETA_mid), dim=-1)
    grid_all = grid.view(-1, 3).to(device)
    grid_all = torch.cat((grid_all, torch.zeros_like(grid_all)), dim=1)

    return grid_all, x_step, y_step, theta_step

grid_all, x_step, y_step, theta_step = split_grid_init(bound_upper = [UPPER_X, UPPER_Y, UPPER_THETA], bound_lower = [LOWER_X, LOWER_Y, LOWER_THETA], grid_size = 16)

offset = torch.FloatTensor([[x_step, y_step, theta_step, 0.2, 0.2, 0.2]]).to(device)

lower = grid_all - offset / 2
upper = grid_all + offset / 2

torch.save(grid_all, 'grid_all.pt')
torch.save(lower, 'lower.pt')
torch.save(upper, 'upper.pt')
