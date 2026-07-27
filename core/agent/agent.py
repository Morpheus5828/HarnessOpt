import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent.config import *
from core.agent.actor import *
from core.agent.critic import *


class RecurrentTD3Agent:
    def __init__(self, seq_batch=16):
        self.actor = ActorBiGRU().to(device)
        # Remplacement de copy.deepcopy
        self.actor_target = ActorBiGRU().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)

        self.critic = CriticBiGRU().to(device)
        # Remplacement de copy.deepcopy
        self.critic_target = CriticBiGRU().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.max_action = 1.0
        self.discount = 0.99
        self.tau = TAU
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0
        self.seq_batch = seq_batch

    def select_action(self, state_np, noise=0.1):
        state = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(
                -self.max_action, self.max_action)
        return action

    def train(self, replay_buffer, batch_size=256):
        if replay_buffer.n_seq < self.seq_batch:
            return
        self.total_it += 1
        state, action, next_state, reward, mask, lengths = replay_buffer.sample(self.seq_batch)
        denom = mask.sum() + 1e-6

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, lengths) + noise).clamp(
                -self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, lengths)
            target_Q = reward + self.discount * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action, lengths)
        critic_loss = (((current_Q1 - target_Q) ** 2) * mask).sum() / denom + \
                      (((current_Q2 - target_Q) ** 2) * mask).sum() / denom
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            q1_pred, _ = self.critic(state, self.actor(state, lengths), lengths)
            actor_loss = -(q1_pred * mask).sum() / denom
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class RLAgent:
    def __init__(self, use_td3=True):
        self.use_td3 = use_td3

        self.actor = Actor().to(device)
        # Remplacement de copy.deepcopy
        self.actor_target = Actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic().to(device)
        # Remplacement de copy.deepcopy
        self.critic_target = Critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.max_action = 1.0
        self.discount = 0.99
        self.tau = TAU
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2 if use_td3 else 1
        self.total_it = 0

    def select_action(self, state_np, noise=0.1):
        state = torch.FloatTensor(state_np).to(device)
        action = self.actor(state).cpu().data.numpy()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(-self.max_action, self.max_action)
        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward = replay_buffer.sample(batch_size)
        with torch.no_grad():
            if self.use_td3:
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = reward + (self.discount * torch.min(target_Q1, target_Q2))
            else:
                next_action = self.actor_target(next_state)
                target_Q1, _ = self.critic_target(next_state, next_action)
                target_Q = reward + (self.discount * target_Q1)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q)
        if self.use_td3:
            critic_loss += F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            q1_pred, _ = self.critic(state, self.actor(state))
            actor_loss = -q1_pred.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class SACAgent:
    def __init__(self):
        self.actor = ActorSAC().to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic().to(device)
        # Remplacement de copy.deepcopy
        self.critic_target = Critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.alpha = 0.2
        self.discount = 0.99
        self.tau = TAU

    def select_action(self, state_np, noise=0.0):
        state = torch.FloatTensor(state_np).to(device)
        action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + (self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        actor_loss = ((self.alpha * log_pi) - torch.min(qf1_pi, qf2_pi)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
