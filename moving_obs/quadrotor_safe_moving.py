"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np
from gym.utils import seeding
import gym
from gym import logger, spaces
# from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from torch import nn
import torch
import os

# input
# state: [x, y, theta, x_dot, y_dot, theta_dot]
# action: [u1, u2]

class DynamicsModel(nn.Module):
    def __init__(self):
        super(DynamicsModel, self).__init__()
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
    

class QuadrotorSafeMovingEnv(gym.Env):
    def __init__(self):
        self.seed()

        self.obs_high = np.array([5.0, 5.0, np.pi/3.0, 2.0, 2.0, 2.0], dtype=np.float32)   
        self.obs_low = np.array([-5.0, -5.0, -np.pi/3.0, -2.0, -2.0, -2.0], dtype=np.float32)   

        self.action_high = np.array([8.0, 8.0], dtype=np.float32)
        self.action_low = np.array([0.0, 0.0], dtype=np.float32)

        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype=np.float32)

        self.state = None
        self.num_steps = 500
        self.reward_all = 0

        self.dynamics_model = DynamicsModel()
        self.dynamics_model.load_state_dict(torch.load(f'{os.path.dirname(os.path.abspath(__file__))}/quad2d_dynamics.pth'))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        self.steps += 1
        force = np.clip(action, self.action_low, self.action_high)
    
        input_state = torch.FloatTensor(self.state).unsqueeze(0)
        input_action = torch.FloatTensor(force).unsqueeze(0)
        state_tensor = self.dynamics_model(input_state, input_action)        

        self.state = state_tensor.detach().cpu().numpy()[0]

        x, y, theta, x_dot, y_dot, theta_dot = self.state
        goal = np.array([0.6, 0.6, 0.0], dtype=np.float32)
        
        safe_func = 0.0
        
        rectangles = self.get_obs_loc()
        
        for xl, yl, xu, yu in rectangles:
            if x >= xl and x <= xu and y >= yl and y <= yu:
                safe_func = safe_func + 1.0
        
        if (abs(theta)) > np.pi / 3.0:
            safe_func = safe_func + 1.0
        
        reward = -np.sum(np.abs(self.state[:3] - goal))

        reward = np.exp(reward)

        self.reward_all += reward

        if self.steps >= self.num_steps or abs(theta) > np.pi / 3.0 or abs(x) > 5.0 or abs(y) > 5.0 or y < -0.15:
            terminated = True
        else:
            terminated = False

        return np.array(self.state, dtype=np.float32), reward, terminated, {'cost': safe_func, 'reward_all': self.reward_all}

    def reset(self):
        self.reward_all = 0
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        self.steps = 0
        init_region_high = np.array([0.5, 0.1, 0.1 , 0.1, 0.1, 0.1], dtype=np.float32)
        init_region_low = np.array([-0.5, -0.1, -0.1, -0.1, -0.1, -0.1], dtype=np.float32)
        self.state = self.np_random.uniform(low=init_region_low, high=init_region_high, size=(6,))
        return np.array(self.state, dtype=np.float32)
    
    def get_obs_loc(self):
        
        mat = np.array([
            [0.6, 0.0, 0.6, 0.1],
            [-0.5, 0.2, -0.4, 0.3],
            [-0.3, 0.4, -0.4, 0.5],
            [-0.1, 0.3, 0.0, 0.4],
            [-0.7, 0.5, -0.4, 0.6]
        ])
        
        step = self.steps
        
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
        return obs_all
        
        

    def close(self):
        return
