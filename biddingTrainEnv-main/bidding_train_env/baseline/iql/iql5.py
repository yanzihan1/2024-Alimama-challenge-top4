import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import torch
from copy import deepcopy
import os
from torch.distributions import Normal
from torch.optim.lr_scheduler import StepLR, ExponentialLR


class Q(nn.Module):
    '''
    IQL-Q网络
    '''

    def __init__(self, dim_observation, dim_action):
        super(Q, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dropout_rate=0.1

        self.obs_FC = nn.Linear(self.dim_observation, 64)
        self.action_FC = nn.Linear(dim_action, 64)
        self.FC1 = nn.Linear(128, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)  # Dropout layer


    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs_embedding = self.obs_FC(obs)
        action_embedding = self.action_FC(acts)
        embedding = torch.cat([obs_embedding, action_embedding], dim=-1)
        q = self.FC3(F.relu(self.FC2(F.relu(self.FC1(embedding)))))
        return q


class V(nn.Module):
    '''
        IQL-V网络
        '''

    def __init__(self, dim_observation):
        super(V, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        return self.FC4(result)


class Actor(nn.Module):
    '''
    IQL-动作网络
    '''

    def __init__(self, dim_observation, dim_action, log_std_min=-10, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC_mu = nn.Linear(64, dim_action)
        self.FC_std = nn.Linear(64, dim_action)

    def forward(self, obs):
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mu = self.FC_mu(x)
        log_std = self.FC_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, obs, epsilon=1e-6):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action, dist

    def get_action(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action.detach().cpu()

    def get_det_action(self, obs):
        mu, _ = self.forward(obs)
        return mu.detach().cpu()


class IQL:
    '''
    IQL网络
    '''

    def __init__(self, dim_obs=3, dim_actions=1, gamma=0.99, tau=0.01, V_lr=1e-4, critic_lr=1e-4, actor_lr=1e-4,
                 network_random_seed=1, expectile=0.7, temperature=3.0):
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.V_lr = V_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.network_random_seed = network_random_seed
        self.expectile = expectile
        self.temperature = temperature
        torch.random.manual_seed(self.network_random_seed)
        self.value_net = V(self.num_of_states)
        self.critic1 = Q(self.num_of_states, self.num_of_actions)
        self.critic2 = Q(self.num_of_states, self.num_of_actions)
        self.critic3 = Q(self.num_of_states, self.num_of_actions)
        self.critic4 = Q(self.num_of_states, self.num_of_actions)
        self.critic5 = Q(self.num_of_states, self.num_of_actions)

        self.critic1_target = Q(self.num_of_states, self.num_of_actions)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Q(self.num_of_states, self.num_of_actions)
        self.critic3_target.load_state_dict(self.critic3.state_dict())
        self.critic3_target = Q(self.num_of_states, self.num_of_actions)
        self.critic4_target.load_state_dict(self.critic4.state_dict())
        self.critic4_target = Q(self.num_of_states, self.num_of_actions)
        self.critic5_target.load_state_dict(self.critic5.state_dict())
        self.critic5_target = Q(self.num_of_states, self.num_of_actions)

        self.actors = Actor(self.num_of_states, self.num_of_actions)
        self.GAMMA = gamma
        self.tau = tau
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.critic3_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.critic4_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.critic5_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)

        self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)
        self.deterministic_action = True
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.critic1.cuda()
            self.critic2.cuda()
            self.critic3.cuda()
            self.critic4.cuda()
            self.critic5.cuda()
            self.critic1_target.cuda()
            self.critic2_target.cuda()
            self.critic3_target.cuda()
            self.critic4_target.cuda()
            self.critic5_target.cuda()

            self.value_net.cuda()
            self.actors.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

    def step(self, states, actions, rewards, next_states, dones,step_num):
        '''
        训练网络
        '''

        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions)
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = self.calc_policy_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic1_loss, critic2_loss,critic3_loss,critic4_loss,critic5_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.critic3_optimizer.zero_grad()
        critic3_loss.backward()
        self.critic3_optimizer.step()
        self.critic4_optimizer.zero_grad()
        critic4_loss.backward()
        self.critic4_optimizer.step()
        self.critic5_optimizer.zero_grad()
        critic5_loss.backward()
        self.critic5_optimizer.step()

        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)
        self.update_target(self.critic3, self.critic3_target)
        self.update_target(self.critic4, self.critic4_target)
        self.update_target(self.critic5, self.critic5_target)


        return critic1_loss.cpu().data.numpy(), value_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    def take_actions(self, states):
        '''
        输出动作
        '''
        states = torch.Tensor(states).type(self.FloatTensor)
        if self.deterministic_action:
            actions = self.actors.get_det_action(states)
        else:
            actions = self.actors.get_action(states)
        actions = torch.clamp(actions, 0)
        actions = actions.cpu().data.numpy()
        return actions

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            q3 = self.critic3_target(states, actions)
            q4 = self.critic4_target(states, actions)
            q5 = self.critic5_target(states, actions)

            # min_Q = torch.min(q1, q2, q3, q4, q5)
            stacked_tensors = torch.stack((q1, q2, q3, q4, q5))
            min_Q, min_indices = torch.min(stacked_tensors, dim=0)

        exp_a = torch.exp(min_Q - v) * self.temperature
        number_100=torch.FloatTensor([100.0]).to("cuda:0")
        exp_a = torch.min(exp_a, number_100)

        _, dist = self.actors.evaluate(states)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    def calc_value_loss(self, states, actions):

        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            q3 = self.critic3_target(states, actions)
            q4 = self.critic4_target(states, actions)
            q5 = self.critic5_target(states, actions)

            # min_Q = torch.min(q1, q2, q3, q4, q5)
            stacked_tensors = torch.stack((q1, q2, q3, q4, q5))
            min_Q, min_indices = torch.min(stacked_tensors, dim=0)

        value = self.value_net(states)
        value_loss = self.l2_loss(min_Q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q3 = self.critic3(states, actions)
        q4 = self.critic4(states, actions)
        q5 = self.critic5(states, actions)
        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        critic3_loss = ((q3 - q_target) ** 2).mean()
        critic4_loss = ((q4 - q_target) ** 2).mean()
        critic5_loss = ((q5 - q_target) ** 2).mean()

        return critic1_loss, critic2_loss, critic3_loss, critic4_loss, critic5_loss

    def update_target(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1. - self.tau) * target_param.data + self.tau * local_param.data)

    def save_net(self, save_path):
        '''
        存储模型
        '''
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.critic1, save_path + "/critic1" + ".pkl")
        torch.save(self.critic2, save_path + "/critic2" + ".pkl")
        torch.save(self.critic3, save_path + "/critic3" + ".pkl")
        torch.save(self.critic4, save_path + "/critic4" + ".pkl")
        torch.save(self.critic5, save_path + "/critic5" + ".pkl")

        torch.save(self.value_net, save_path + "/value_net" + ".pkl")
        torch.save(self.actors, save_path + "/actor" + ".pkl")

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cuda:0'):
        '''
        加载模型
        '''
        if os.path.isfile(load_path + "/critic.pt"):
            self.critic1.load_state_dict(torch.load(load_path + "/critic1.pt", map_location='cpu'))
            self.critic2.load_state_dict(torch.load(load_path + "/critic2.pt", map_location='cpu'))
            self.critic3.load_state_dict(torch.load(load_path + "/critic3.pt", map_location='cpu'))
            self.critic4.load_state_dict(torch.load(load_path + "/critic4.pt", map_location='cpu'))
            self.critic5.load_state_dict(torch.load(load_path + "/critic5.pt", map_location='cpu'))

            self.actors.load_state_dict(torch.load(load_path + "/actor.pt", map_location='cpu'))
        else:
            self.critic1 = torch.load(load_path + "/critic1.pkl", map_location='cpu')
            self.critic2 = torch.load(load_path + "/critic2.pkl", map_location='cpu')
            self.critic3 = torch.load(load_path + "/critic3.pkl", map_location='cpu')
            self.critic4 = torch.load(load_path + "/critic4.pkl", map_location='cpu')
            self.critic5 = torch.load(load_path + "/critic5.pkl", map_location='cpu')

            self.actors = torch.load(load_path + "/actor.pkl", map_location='cpu')
        self.value_net = torch.load(load_path + "/value_net.pkl", map_location='cpu')
        print("model stored path " + next(self.critic1.parameters()).device.type)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.critic3_target = deepcopy(self.critic3)
        self.critic4_target = deepcopy(self.critic4)
        self.critic5_target = deepcopy(self.critic5)

        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.critic3_optimizer = Adam(self.critic3.parameters(), lr=self.critic_lr)
        self.critic4_optimizer = Adam(self.critic4.parameters(), lr=self.critic_lr)
        self.critic5_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)

        self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)

        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.critic1.cuda()
            self.critic2.cuda()
            self.value_net.cuda()
            self.actors.cuda()
            self.critic1_target.cuda()
            self.critic2_target.cuda()
        print("model stored path " + next(self.critic1.parameters()).device.type)

    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)