import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class AtariCNN(nn.Module):
    """CNN feature extractor for Atari games."""
    
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Output size: 64 * 7 * 7 = 3136 for 84x84 input
        self.feature_dim = 3136
    
    def forward(self, x):
        # Normalize pixel values
        x = x.float() / 255.0
        return self.conv(x)


class PPOActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.features = AtariCNN(in_channels)
        
        self.actor = nn.Sequential(
            nn.Linear(self.features.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.features.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.actor(features), self.critic(features)
    
    def get_action(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
    
    def get_value(self, x):
        features = self.features(x)
        return self.critic(features).squeeze(-1)


class RolloutBuffer:
    """Storage for rollout data."""
    
    def __init__(self, buffer_size, num_envs, obs_shape, device):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.ptr = 0
        
        self.obs = torch.zeros((buffer_size, num_envs, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros((buffer_size, num_envs), dtype=torch.long, device=device)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
    
    def add(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, device=self.device)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1
    
    def reset(self):
        self.ptr = 0
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + self.values
        return returns, advantages


class PPOTrainer:
    """Standard PPO training algorithm."""
    
    def __init__(self, env, device="cuda", lr=2.5e-4, n_steps=128, 
                 n_epochs=4, batch_size=256, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.1, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        
        self.env = env
        self.device = device
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        self.n_actions = env.action_space.n
        self.obs_shape = env.observation_space.shape
        self.num_envs = env.num_envs
        
        self.model = PPOActorCritic(self.n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.buffer = RolloutBuffer(n_steps, self.num_envs, self.obs_shape, device)
        
        self.total_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def collect_rollouts(self, obs):
        """Collect experience from environment."""
        self.buffer.reset()
        episode_infos = []
        
        for _ in range(self.n_steps):
            obs_tensor = torch.as_tensor(obs, device=self.device)
            
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(action_np)
            dones = np.logical_or(terminateds, truncateds)
            
            self.buffer.add(obs, action_np, rewards, dones, log_prob, value)
            
            for i, info in enumerate(infos):
                if 'terminal_observation' in info:
                    if 'episode' in info:
                        episode_infos.append(info['episode'])
            
            obs = next_obs
            self.total_timesteps += self.num_envs
        
        # Compute last value for GAE
        with torch.no_grad():
            last_value = self.model.get_value(torch.as_tensor(obs, device=self.device))
        
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        return obs, returns, advantages, episode_infos
    
    def train_epoch(self, returns, advantages):
        """Perform one epoch of PPO updates."""
        # Flatten batch
        b_obs = self.buffer.obs.reshape(-1, *self.obs_shape)
        b_actions = self.buffer.actions.reshape(-1)
        b_log_probs = self.buffer.log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        total_samples = self.n_steps * self.num_envs
        indices = np.arange(total_samples)
        
        pg_losses, vf_losses, entropy_losses = [], [], []
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_samples, self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                _, new_log_prob, entropy, new_value = self.model.get_action(
                    b_obs[mb_indices], b_actions[mb_indices]
                )
                
                # Policy loss with clipping
                log_ratio = new_log_prob - b_log_probs[mb_indices]
                ratio = torch.exp(log_ratio)
                
                pg_loss1 = -b_advantages[mb_indices] * ratio
                pg_loss2 = -b_advantages[mb_indices] * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                vf_loss = 0.5 * ((new_value - b_returns[mb_indices]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        return np.mean(pg_losses), np.mean(vf_losses), np.mean(entropy_losses)
    
    def train(self, total_timesteps, callback=None):
        """Main training loop."""
        obs, _ = self.env.reset()
        
        while self.total_timesteps < total_timesteps:
            obs, returns, advantages, ep_infos = self.collect_rollouts(obs)
            pg_loss, vf_loss, ent_loss = self.train_epoch(returns, advantages)
            
            # Track episode rewards
            for info in ep_infos:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
            
            if callback:
                callback({
                    'timesteps': self.total_timesteps,
                    'pg_loss': pg_loss,
                    'vf_loss': vf_loss,
                    'entropy': ent_loss,
                    'episode_rewards': self.episode_rewards[-100:] if self.episode_rewards else []
                })
        
        return self.model
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'episode_rewards': self.episode_rewards
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.episode_rewards = checkpoint['episode_rewards']