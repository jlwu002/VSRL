target_steps = 15 # Check for collisions with obstacles from step 1 to target_steps

obs1_collide = set()
obs2_collide = set()
obs3_collide = set()
obs4_collide = set()
obs5_collide = set()

import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool

from PPO import PPO
import gym_examples

import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import matplotlib.patches as patches

device = 'cuda'
class MultiStepDynamicsNet(nn.Module):
    def __init__(self, dynamics_model, policy_model, steps=1):
        super(MultiStepDynamicsNet, self).__init__()
        self.dynamics_model = dynamics_model
        self.policy_model = policy_model
        self.steps = steps
        self.u_low = 0.0
        self.u_high = 8.0

    def forward(self, state):
        for _ in range(self.steps):
            action = self.policy_model(state)
            action = F.relu(action - self.u_low) + self.u_low  
            action = -F.relu(-action + self.u_high) + self.u_high
            state = self.dynamics_model(state, action)
        return state


class DynamicsNet(nn.Module):
    def __init__(self):
        super(DynamicsNet, self).__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )
        self.dt = 0.02
        
    def forward(self, state, action):
        model_input = torch.cat((state[:, 2:3], action), dim = 1)
        model_output = self.dynamics(model_input)
        increment = torch.cat((state[:, 3:] * self.dt, model_output), dim = 1) #dt is already integrated in model dynamics output 
        state_next = state + increment
        return state_next

def get_obstacles(step):
    mat = np.array([
        [0.6, 0.0, 0.6, 0.1],
        [-0.5, 0.2, -0.4, 0.3],
        [-0.3, 0.4, -0.4, 0.5],
        [-0.1, 0.3, 0.0, 0.4],
        [-0.7, 0.5, -0.4, 0.6]
    ])
            
    obs1_x1 = mat[0][0] + (mat[0][2] - mat[0][0])/500.0 * step
    obs1_y1 = mat[0][1] + (mat[0][3] - mat[0][1])/500.0 * step
    obs1_x2 = obs1_x1 + 0.1
    obs1_y2 = obs1_y1 + 0.1
    
    obs2_x1 = mat[1][0] + (mat[1][2] - mat[1][0])/500.0 * step
    obs2_y1 = mat[1][1] + (mat[1][3] - mat[1][1])/500.0 * step
    obs2_x2 = obs2_x1 + 0.1
    obs2_y2 = obs2_y1 + 0.1
    
    obs3_x1 = mat[2][0] + (mat[2][2] - mat[2][0])/500.0 * step
    obs3_y1 = mat[2][1] + (mat[2][3] - mat[2][1])/500.0 * step
    obs3_x2 = obs3_x1 + 0.1
    obs3_y2 = obs3_y1 + 0.1
    
    obs4_x1 = mat[3][0] + (mat[3][2] - mat[3][0])/500.0 * step
    obs4_y1 = mat[3][1] + (mat[3][3] - mat[3][1])/500.0 * step
    obs4_x2 = obs4_x1 + 0.1
    obs4_y2 = obs4_y1 + 0.1
    
    obs5_x1 = mat[4][0] + (mat[4][2] - mat[4][0])/500.0 * step
    obs5_y1 = mat[4][1] + (mat[4][3] - mat[4][1])/500.0 * step
    obs5_x2 = obs5_x1 + 0.1
    obs5_y2 = obs5_y1 + 0.1
    
    obs1 = np.array([obs1_x1, obs1_y1, obs1_x2, obs1_y2])
    obs2 = np.array([obs2_x1, obs2_y1, obs2_x2, obs2_y2])
    obs3 = np.array([obs3_x1, obs3_y1, obs3_x2, obs3_y2])
    obs4 = np.array([obs4_x1, obs4_y1, obs4_x2, obs4_y2])
    obs5 = np.array([obs5_x1, obs5_y1, obs5_x2, obs5_y2])
    
    obs_all = np.array([obs1, obs2, obs3, obs4, obs5])   
    obs_all = obs_all.reshape(1, 5, 4)
    
    obs_all = torch.FloatTensor(obs_all).to(device)
    return obs_all

def reachability_loss(lb_selected, ub_selected, obstacles):
    x_lb = lb_selected[:, 0]
    x_ub = ub_selected[:, 0]
    y_lb = lb_selected[:, 1]
    y_ub = ub_selected[:, 1]
    theta_lb = lb_selected[:, 2]
    theta_ub = ub_selected[:, 2]

    obj0 = obstacles[:, 0, :]
    obj1 = obstacles[:, 1, :]
    obj2 = obstacles[:, 2, :]
    obj3 = obstacles[:, 3, :]
    obj4 = obstacles[:, 4, :]
    
    loss1 = F.relu(x_ub - obj0[:, 0]) * F.relu(obj0[:, 2] - x_lb) * F.relu(y_ub - obj0[:, 1]) * F.relu(obj0[:, 3] - y_lb)
    loss2 = F.relu(x_ub - obj1[:, 0]) * F.relu(obj1[:, 2] - x_lb) * F.relu(y_ub - obj1[:, 1]) * F.relu(obj1[:, 3] - y_lb)
    loss3 = F.relu(x_ub - obj2[:, 0]) * F.relu(obj2[:, 2] - x_lb) * F.relu(y_ub - obj2[:, 1]) * F.relu(obj2[:, 3] - y_lb)
    loss4 = F.relu(x_ub - obj3[:, 0]) * F.relu(obj3[:, 2] - x_lb) * F.relu(y_ub - obj3[:, 1]) * F.relu(obj3[:, 3] - y_lb)
    loss5 = F.relu(x_ub - obj4[:, 0]) * F.relu(obj4[:, 2] - x_lb) * F.relu(y_ub - obj4[:, 1]) * F.relu(obj4[:, 3] - y_lb)
    
    loss_obj_others = F.relu(theta_ub - 60 * torch.pi / 180) + F.relu(-60 * torch.pi / 180 - theta_lb) + F.relu(-0.2 - y_lb) 
    
    return loss1, loss2, loss3, loss4, loss5, loss_obj_others
    
#################################### Testing ###################################
def test():
    print("============================================================================================")
    has_continuous_action_space = True
    action_std = 0.0            # set same std for action distribution which was used while saving

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    device = 'cuda'
    dynamics_model = DynamicsNet()
    dynamics_model.load_state_dict(torch.load(f'quad2d_dynamics.pth'))
    dynamics_model = dynamics_model.to(device)

    # initialize a PPO agent
    ppo_agent = PPO(6, 2, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    checkpoint_path = "path_to_model.pth"
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    norm = float("inf")
    lower = torch.load('lower.pt')
    upper = torch.load('upper.pt')
    grid_all = torch.load('grid_all.pt')

    with torch.no_grad():
        for step in range(1, target_steps + 1):
            multi_step_dynamics_model = MultiStepDynamicsNet(dynamics_model, ppo_agent.policy.actor, steps = step).to(device)
            multi_step_dynamics_model_bound = BoundedModule(multi_step_dynamics_model, torch.empty_like(upper).to(device))
            ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
            bounded_x = BoundedTensor(torch.empty_like(upper).to(device), ptb)
            lb, ub = multi_step_dynamics_model_bound.compute_bounds(x=(bounded_x,), method='backward')

            obs = get_obstacles(step)
            print(step)

            loss1, loss2, loss3, loss4, loss5, loss_obj_others = reachability_loss(lb, ub, obs)

            # get idx where loss1 > 0
            idx_loss1 = torch.where(loss1 > 0)[0].cpu().numpy().tolist()
            idx_loss2 = torch.where(loss2 > 0)[0].cpu().numpy().tolist()
            idx_loss3 = torch.where(loss3 > 0)[0].cpu().numpy().tolist()
            idx_loss4 = torch.where(loss4 > 0)[0].cpu().numpy().tolist()
            idx_loss5 = torch.where(loss5 > 0)[0].cpu().numpy().tolist()
            idx_loss_obj_others = torch.where(loss_obj_others > 0)[0].cpu().numpy().tolist()

            obs1_collide.update(idx_loss1)
            obs2_collide.update(idx_loss2)
            obs3_collide.update(idx_loss3)
            obs4_collide.update(idx_loss4)
            obs5_collide.update(idx_loss5)

    
    
    # obs1_collide_lst = list(obs1_collide)
    # obs2_collide_lst = list(obs2_collide)
    # obs3_collide_lst = list(obs3_collide)
    # obs4_collide_lst = list(obs4_collide)
    # obs5_collide_lst = list(obs5_collide)

    # Splitting based on the collide_idx above works, but splitting along a specific axis (e.g., the 1st axis) may also be effective (as shown below).

    idx1 = grid_all[:, 0] < 0.0 # e.g., threshold for split is 0.0
    idx2 = ~idx1

    grid_all_1 = grid_all[idx1]
    grid_all_2 = grid_all[idx2]
    lower1 = lower[idx1]
    upper1 = upper[idx1]
    lower2 = lower[idx2]
    upper2 = upper[idx2]

    torch.save(grid_all_1, 'grid_all_1.pt')
    torch.save(grid_all_2, 'grid_all_2.pt')
    torch.save(lower1, 'lower1.pt')
    torch.save(upper1, 'upper1.pt')
    torch.save(lower2, 'lower2.pt')
    torch.save(upper2, 'upper2.pt')

if __name__ == '__main__':
    test()
