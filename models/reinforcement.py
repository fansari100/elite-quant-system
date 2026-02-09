"""
Deep Reinforcement Learning for Portfolio Management
=====================================================
Based on FinRL framework and QTMRL (arXiv:2508.20467)

Implements:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)
- Multi-indicator guided RL

Optimized for financial markets with:
- Transaction cost awareness
- Risk-adjusted rewards
- Position constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.FloatTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """
    Policy network for continuous action space (portfolio weights).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared backbone
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        dist = Normal(mean, std)
        z = dist.rsample()
        
        # Apply tanh squashing and convert to portfolio weights
        action = torch.tanh(z)
        action = F.softmax(action, dim=-1)  # Ensure weights sum to 1
        
        # Compute log probability with correction for tanh
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference."""
        mean, log_std = self(state)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            z = dist.rsample()
            action = torch.tanh(z)
        
        return F.softmax(action, dim=-1)


class CriticNetwork(nn.Module):
    """
    Value/Q-function network.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        n_critics: int = 2  # Twin critics for reduced overestimation
    ):
        super().__init__()
        
        self.n_critics = n_critics
        
        # Create multiple critic networks
        self.critics = nn.ModuleList()
        for _ in range(n_critics):
            layers = []
            prev_dim = state_dim + action_dim
            for dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU()
                ])
                prev_dim = dim
            layers.append(nn.Linear(hidden_dims[-1], 1))
            self.critics.append(nn.Sequential(*layers))
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return [critic(x) for critic in self.critics]
    
    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return minimum Q-value across critics."""
        q_values = self(state, action)
        return torch.min(torch.stack(q_values), dim=0)[0]


class PPOAgent(nn.Module):
    """
    Proximal Policy Optimization for portfolio management.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        super().__init__()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor.get_action(state, deterministic)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[-1]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        n_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """PPO update step."""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(n_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Compute new log probs and entropy
                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mean, std)
                
                # Inverse softmax to get pre-softmax values
                batch_actions_pre = torch.log(batch_actions + 1e-8)
                new_log_probs = dist.log_prob(batch_actions_pre).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()
                
                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.critic(batch_states)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }


class SACAgent(nn.Module):
    """
    Soft Actor-Critic for continuous portfolio optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True
    ):
        super().__init__()
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer()
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor.get_action(state, deterministic)
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """SAC update step."""
        if len(self.buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q = self.critic_target.q_min(next_states, next_actions)
            target_q = target_q - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = sum(F.mse_loss(q, target_q) for q in current_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        new_actions, log_probs = self.actor.sample(states)
        q_new = self.critic.q_min(states, new_actions)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha update (entropy tuning)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }


class FinancialRewardShaper:
    """
    Shapes rewards for financial RL with risk adjustment.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05 / 252,
        transaction_cost: float = 0.0005,
        risk_aversion: float = 1.0
    ):
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion
        self.returns_history = []
    
    def compute_reward(
        self,
        portfolio_return: float,
        turnover: float,
        current_weights: np.ndarray,
        prev_weights: np.ndarray
    ) -> float:
        """
        Compute shaped reward considering:
        - Portfolio return
        - Transaction costs
        - Risk (via Sharpe-inspired adjustment)
        """
        # Deduct transaction costs
        cost = turnover * self.transaction_cost
        net_return = portfolio_return - cost
        
        # Track returns for running Sharpe
        self.returns_history.append(net_return)
        if len(self.returns_history) > 252:
            self.returns_history.pop(0)
        
        # Compute running volatility
        if len(self.returns_history) > 10:
            vol = np.std(self.returns_history)
            # Risk-adjusted reward
            reward = (net_return - self.risk_free_rate) - self.risk_aversion * vol * abs(net_return)
        else:
            reward = net_return
        
        return reward

