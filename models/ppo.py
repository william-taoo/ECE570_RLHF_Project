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
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        base = self.shared(state)
        action_logits = self.actor(base)
        state_value = self.critic(base)
        return action_logits, state_value 
    
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        clip_ratio, # How much to constrain policy updates
        lr,
        gamma, # Discount factor for rewards
        lam # GAE
    ):
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam

    # Choose an action given a state
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) # Make tensor and add dimension [1, state_dim]
        logits, _ = self.policy(state)
        
        # Guard against NaNs/Infs in logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def compute_returns(self, rewards, values, masks):
        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            # Calculate temporal difference error
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            
            # Calculate generalized advantage estimation (GAE)
            gae = delta + self.gamma * self.lam * masks[step] * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)
    
    def update(self, states, actions, old_log_probs, returns, advantages, epochs, batch_size):
        dataset_size = len(states)

        for _ in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            losses = []

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_index = indices[start:end]
                
                state_batch = torch.FloatTensor(states[batch_index])
                action_batch = torch.LongTensor(actions[batch_index])
                old_log_prob_batch = torch.FloatTensor(old_log_probs[batch_index])
                returns_batch = returns[batch_index]
                advantages_batch = advantages[batch_index]

                # Normalize for stability
                if advantages_batch.numel() > 1:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                else:
                    advantages_batch = advantages_batch * 0

                # Forward pass (use logits for numeric stability)
                logits, values = self.policy(state_batch)

                # Replace any NaN/Inf logits with large finite numbers
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(action_batch)

                ratio = torch.exp(new_log_probs - old_log_prob_batch)
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                loss_policy = -(torch.min(ratio * advantages_batch, clipped)).mean()
                loss_value = (returns_batch - values.squeeze()).pow(2).mean()

                entropy_bonus = 0.005 * dist.entropy().mean()
                loss = loss_policy + 0.5 * loss_value - entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradient
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

                self.optimizer.step()
                losses.append(loss.item())

        return losses
