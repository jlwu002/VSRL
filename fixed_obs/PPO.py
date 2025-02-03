import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch.nn.functional as F
import collections
import numpy as np

HIDDEN_SIZES = 16

MAX_INCREASE_COUNT = 300 # decrease if you want faster training, but this may result in more controllers

print("============================================================================================")
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

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
        increment = torch.cat((state[:, 3:], model_output), dim = 1)
        state_next = state + increment * self.dt
        return state_next
    

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, HIDDEN_SIZES),
                            nn.ReLU(),
                            nn.Linear(HIDDEN_SIZES, HIDDEN_SIZES),
                            nn.ReLU(),
                            nn.Linear(HIDDEN_SIZES, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, HIDDEN_SIZES),
                            nn.ReLU(),
                            nn.Linear(HIDDEN_SIZES, HIDDEN_SIZES),
                            nn.ReLU(),
                            nn.Linear(HIDDEN_SIZES, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, HIDDEN_SIZES),
                        nn.Tanh(),
                        nn.Linear(HIDDEN_SIZES, HIDDEN_SIZES),
                        nn.Tanh(),
                        nn.Linear(HIDDEN_SIZES, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.use_reachability_loss = False
        self.verify_steps = 1

        dynamics_model = DynamicsModel().to(device)
        dynamics_model.load_state_dict(torch.load(f"quad2d_cpu.pth"))
        self.dynamics_model = dynamics_model.to(device)
        self.target_verify_steps = 50

        self.checkpoint_loaded = False
        self.grid_all = None

        self.verify_step_offset = 0
        self.violation_set_lower = torch.FloatTensor([]).to(device)
        self.violation_set_upper = torch.FloatTensor([]).to(device)
        self.violation_set_input = torch.FloatTensor([]).to(device)
        self.potential_unsafe_regions = collections.defaultdict(list)
        self.buffer_input = torch.FloatTensor([]).to(device)
        self.buffer_lower = torch.FloatTensor([]).to(device)
        self.buffer_upper = torch.FloatTensor([]).to(device)
        self.count = 0
        self.count_increase = 0
        self.input_selected_size = 100000

        self.exclude_idx = set()

        # load initial grid
        self.grid_all = torch.load('grid_all.pt')
        self.grid_bound_lower = torch.load('lower.pt')
        self.grid_bound_upper = torch.load('upper.pt')

    def get_obstacles(self, step):
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
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()
    
    def find_reachability_grid(self):
        reach_continue = True
        
        grid_all, grid_bound_lower, grid_bound_upper = self.grid_all, self.grid_bound_lower, self.grid_bound_upper

        norm = float("inf")

        lower_masked = grid_bound_lower
        upper_masked = grid_bound_upper
        grid_all_masked = grid_all

        self.verify_steps_prev = self.verify_steps

        while reach_continue:
            if self.verify_steps > self.target_verify_steps:
                print("verify_steps > target_verify_steps")
                return None, None, None
            print("self.count", self.count)
            print("self.count_increase", self.count_increase)
            with torch.no_grad():
                multi_step_dynamics_model = MultiStepDynamicsNet(self.dynamics_model, self.policy.actor, steps = self.verify_steps).to(device)
                multi_step_dynamics_model_bound = BoundedModule(multi_step_dynamics_model, torch.empty_like(upper_masked).to(device))
                ptb = PerturbationLpNorm(norm = norm, x_L=lower_masked, x_U=upper_masked)
                bounded_x = BoundedTensor(torch.empty_like(upper_masked).to(device), ptb)
                lb, ub = multi_step_dynamics_model_bound.compute_bounds(x=(bounded_x,), method='backward')

                reachability_loss = self.reachability_loss(lb, ub)
                
                unsafe_mask = reachability_loss > 0
                input_selected = grid_all_masked[unsafe_mask]
                lower_selected = lower_masked[unsafe_mask]
                upper_selected = upper_masked[unsafe_mask]
                
                input_safe = grid_all_masked[~unsafe_mask]
                lower_safe = lower_masked[~unsafe_mask]
                upper_safe = upper_masked[~unsafe_mask]
                
                # further split the grid, and calculate bounds (the bounds will be tighter), filter out safe/usnafe regions, limit 0.0125
                # input_selected, lower_selected, upper_selected, input_safe_add, lower_safe_add, upper_safe_add = \
                #     self.split_grid(input_selected, lower_selected, upper_selected)
                # input_safe = torch.cat((input_safe, input_safe_add), 0)
                # lower_safe = torch.cat((lower_safe, lower_safe_add), 0)
                # upper_safe = torch.cat((upper_safe, upper_safe_add), 0)
                
                if input_selected.size(0) >= self.input_selected_size:
                    self.count_increase += 1
                
                self.input_selected_size = input_selected.size(0)
                print('self.input_selected_size', self.input_selected_size)

                if self.count > 5: # record unsafe regions
                    self.violation_set_lower = torch.cat((self.violation_set_lower, lower_selected), 0)
                    self.violation_set_upper = torch.cat((self.violation_set_upper, upper_selected), 0)
                    self.violation_set_input = torch.cat((self.violation_set_input, input_selected), 0)

                if len(lower_selected) == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("Saving checkpoint, step:", self.verify_steps)
                    print("--------------------------------------------------------------------------------------------")
                    if self.verify_steps % 5 == 0:
                        self.potential_unsafe_regions[self.verify_steps] = (self.violation_set_input, self.violation_set_lower, self.violation_set_upper)
                        self.violation_set_lower = torch.FloatTensor([]).to(device)
                        self.violation_set_upper = torch.FloatTensor([]).to(device)
                        self.violation_set_input = torch.FloatTensor([]).to(device)
                        
                        self.buffer_input = torch.FloatTensor([]).to(device)
                        self.buffer_lower = torch.FloatTensor([]).to(device)
                        self.buffer_upper = torch.FloatTensor([]).to(device)
                        self.buffer_obstacle = torch.FloatTensor([]).to(device)
                        
                        for key, value in self.potential_unsafe_regions.items():
                            if len(value[0]) == 0:
                                continue                            
                            grid_all_sub, lb_sub, ub_sub = value

                            with torch.no_grad():
                                multi_step_dynamics_model = MultiStepDynamicsNet(self.dynamics_model, self.policy.actor, steps = key - 5).to(device)
                                multi_step_dynamics_model_bound = BoundedModule(multi_step_dynamics_model, torch.empty_like(lb_sub).to(device))
                                ptb = PerturbationLpNorm(norm = norm, x_L=lb_sub, x_U=ub_sub)
                                bounded_x = BoundedTensor(torch.empty_like(lb_sub).to(device), ptb)
                                lower_sub, upper_sub = multi_step_dynamics_model_bound.compute_bounds(x=(bounded_x,), method='backward')
                            self.buffer_input = torch.cat((self.buffer_input, grid_all_sub), 0)
                            self.buffer_lower = torch.cat((self.buffer_lower, lower_sub), 0)
                            self.buffer_upper = torch.cat((self.buffer_upper, upper_sub), 0)

                            obs_curr = self.get_obstacles(key)
                            self.buffer_obstacle = torch.cat((self.buffer_obstacle, obs_curr.repeat(self.buffer_lower.shape[0], 1, 1)), 0)

                    torch.save(self.policy.state_dict(), f"./outputs/quad2d_moving_obs_checkpoint_{self.verify_steps}.pth")

                    self.verify_steps += 1
                    self.count = 0
                    self.input_selected_size = 100000
                    self.count_increase = 0

                elif self.count_increase >= MAX_INCREASE_COUNT:
                    self.violation_set_lower = torch.cat((self.violation_set_lower, lower_selected), 0)
                    self.violation_set_upper = torch.cat((self.violation_set_upper, upper_selected), 0)
                    self.violation_set_input = torch.cat((self.violation_set_input, input_selected), 0)
                    
                    self.grid_all = input_safe
                    self.grid_bound_lower = lower_safe
                    self.grid_bound_upper = upper_safe

                    torch.save(self.policy.state_dict(), f"./outputs/quad2d_moving_obs_checkpoint_{self.verify_steps}_not_verified.pth")
                    self.verify_steps += 1
                    
                    if self.verify_steps > self.target_verify_steps:
                        print("--------------------------------------------------------------------------------------------")
                        print("verify_steps >= target_verify_steps")
                        print("--------------------------------------------------------------------------------------------")
                        return None, None, None                        
                    self.count = 0
                    self.count_increase = 0
                    self.input_selected_size = 100000
                else:
                    self.count += 1
                    break

        print("--------------------------------------------------------------------------------------------")
        print("Increasing steps to : ", self.verify_steps)
        print("--------------------------------------------------------------------------------------------")
        self.verify_step_offset = max(self.verify_steps - 5, 0)

        if self.verify_step_offset != 0:
            with torch.no_grad():
                multi_step_dynamics_model = MultiStepDynamicsNet(self.dynamics_model, self.policy.actor, steps = self.verify_step_offset).to(device)
                multi_step_dynamics_model_bound = BoundedModule(multi_step_dynamics_model, torch.empty_like(input_selected).to(device))
                ptb = PerturbationLpNorm(norm = norm, x_L=lower_selected, x_U=upper_selected)
                bounded_x = BoundedTensor(torch.empty_like(input_selected).to(device), ptb)
                lower_selected, upper_selected = multi_step_dynamics_model_bound.compute_bounds(x=(bounded_x,), method='backward')

        return input_selected, lower_selected, upper_selected

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        if self.use_reachability_loss:
            input_selected, lower_selected, upper_selected = self.find_reachability_grid()
            if input_selected is not None:
                self.multi_step_dynamics_model = MultiStepDynamicsNet(self.dynamics_model, self.policy.actor, steps = self.verify_steps - self.verify_step_offset).to(device)
                self.multi_step_dynamics_model_bound = BoundedModule(self.multi_step_dynamics_model, torch.empty_like(input_selected).to(device))
                norm = float("inf")
                ptb = PerturbationLpNorm(norm = norm, x_L=lower_selected, x_U=upper_selected)
                bounded_x = BoundedTensor(torch.empty_like(input_selected).to(device), ptb)
                
            if len(self.buffer_input) != 0:
                self.five_step_dynamics_model = MultiStepDynamicsNet(self.dynamics_model, self.policy.actor, steps = 5).to(device)
                self.five_step_dynamics_model = BoundedModule(self.five_step_dynamics_model, torch.empty_like(self.buffer_input).to(device))
                norm = float("inf")
                ptb = PerturbationLpNorm(norm = norm, x_L=self.buffer_lower, x_U=self.buffer_upper)
                bounded_x_buffer = BoundedTensor(torch.empty_like(self.buffer_input).to(device), ptb)  

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            if self.use_reachability_loss and len(self.buffer_input) != 0:
                lb_buffer, ub_buffer = self.multi_step_dynamics_model_bound.compute_bounds(x=(bounded_x_buffer,), method='backward')
                reachability_loss_prev = self.reachability_loss(lb_buffer, ub_buffer)
                self.bound = 1000.0 
                reachability_loss_prev_clamped = torch.clamp(reachability_loss_prev, 0, self.bound).mean()
                coeff = (loss.detach().mean()/reachability_loss_prev_clamped.detach())   
                coeff = min(coeff, 100.0)

                loss = loss + coeff * reachability_loss_prev_clamped 

            if self.use_reachability_loss and input_selected is not None:
                lb_selected, ub_selected = self.multi_step_dynamics_model_bound.compute_bounds(x=(bounded_x,), method='backward')
                self.bound = 1000.0 
                reachability_loss = self.reachability_loss(lb_selected, ub_selected)
                reachability_loss_clamped = torch.clamp(reachability_loss, 0, self.bound).mean()
                coeff = (loss.detach().mean()/reachability_loss_clamped.detach())  
                coeff = min(coeff, 100.0)
                loss = loss + coeff * reachability_loss_clamped 

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def reachability_loss(self, lb_selected, ub_selected):
        x_lb = lb_selected[:, 0]
        x_ub = ub_selected[:, 0]
        y_lb = lb_selected[:, 1]
        y_ub = ub_selected[:, 1]
        theta_lb = lb_selected[:, 2]
        theta_ub = ub_selected[:, 2]

        rectangles = [
            [-0.3, 0.4, 0.2, 0.2],
            [-1.2, 0.2, 0.4, 0.2],
            [0.0, 0.5, 0.1, 0.5],
            [0.6, 0.0, 0.1, 0.2],
            [-0.8, 0.7, 0.1, 0.2]
        ]
        rectangles = np.array(rectangles)

        rectangles[:, 2:4] = rectangles[:, 0:2] + rectangles[:, 2:4]
        
        obj1_xl, obj1_yl, obj1_xu, obj1_yu = rectangles[0]
        obj2_xl, obj2_yl, obj2_xu, obj2_yu = rectangles[1]
        obj3_xl, obj3_yl, obj3_xu, obj3_yu = rectangles[2]
        obj4_xl, obj4_yl, obj4_xu, obj4_yu = rectangles[3]
        obj5_xl, obj5_yl, obj5_xu, obj5_yu = rectangles[4]

        loss_obj1 = F.relu(x_ub - obj1_xl) * F.relu(obj1_xu - x_lb) * F.relu(y_ub - obj1_yl) * F.relu(obj1_yu - y_lb)
        loss_obj2 = F.relu(x_ub - obj2_xl) * F.relu(obj2_xu - x_lb) * F.relu(y_ub - obj2_yl) * F.relu(obj2_yu - y_lb)
        loss_obj3 = F.relu(x_ub - obj3_xl) * F.relu(obj3_xu - x_lb) * F.relu(y_ub - obj3_yl) * F.relu(obj3_yu - y_lb)
        loss_obj4 = F.relu(x_ub - obj4_xl) * F.relu(obj4_xu - x_lb) * F.relu(y_ub - obj4_yl) * F.relu(obj4_yu - y_lb)
        loss_obj5 = F.relu(x_ub - obj5_xl) * F.relu(obj5_xu - x_lb) * F.relu(y_ub - obj5_yl) * F.relu(obj5_yu - y_lb)
        
        loss_obj_others = F.relu(theta_ub - 0.785) + F.relu(-0.785 - theta_lb) + F.relu(-0.2 - y_lb) # train 0.2, verify 0.15
        reachability_loss = loss_obj1 + loss_obj2 + loss_obj3 + loss_obj4 + loss_obj5 + loss_obj_others

        return reachability_loss

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


