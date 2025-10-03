import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    '''
    An actor-critic network for PPO
    The actor learns the policy
    The critic learns the value
    '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        base = self.shared(state)
        action_probs = self.actor(base)
        state_value = self.critic(base)
        return action_probs, state_value 
    
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        clip_ratio,
        lr,
        gamme,
        lam
    ):
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim=64)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.gamma = gamme
        self.lam = lam

    