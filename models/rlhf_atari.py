import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from models.atari_ppo import AtariCNN, RolloutBuffer

class RewardModel(nn.Module):
    """
    Learned reward model for RLHF.
    This class predicts scalar rewards from 
    observations and actions.
    """
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.features = AtariCNN(in_channels)
        
        self.action_embed = nn.Embedding(n_actions, 64)
        
        self.reward_head = nn.Sequential(
            nn.Linear(self.features.feature_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs, action):
        features = self.features(obs)
        action_emb = self.action_embed(action)
        combined = torch.cat([features, action_emb], dim=-1)
        return self.reward_head(combined).squeeze(-1)
    
    def predict_trajectory_reward(self, obs_seq, action_seq):
        total = 0
        for obs, action in zip(obs_seq, action_seq):
            total += self.forward(obs.unsqueeze(0), action.unsqueeze(0))
        return total

class PreferenceDataset:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.comparisons = []
    
    def add(self, traj_a, traj_b, preference):
        '''
        0 = A preferred
        1 = B preferred
        '''
        if len(self.comparisons) >= self.max_size:
            self.comparisons.pop(0)
        self.comparisons.append((traj_a, traj_b, preference))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.comparisons), 
                                   min(batch_size, len(self.comparisons)), 
                                   replace=False)
        return [self.comparisons[i] for i in indices]
    
    def __len__(self):
        return len(self.comparisons)

class SyntheticPreferenceOracle:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
    
    def get_preference(self, traj_a_reward, traj_b_reward):
        """Return preference based on true cumulative rewards."""
        diff = traj_a_reward - traj_b_reward
        
        # Add noise
        if np.random.random() < self.noise_level:
            return np.random.choice([0, 1, 0.5])
        
        if diff > 0.5:
            return 0 # Prefer trajectory A
        elif diff < -0.5:
            return 1 # Prefer trajectory B
        else:
            return 0.5 # Tie


class RLHFActorCritic(nn.Module):
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
    
    def get_log_probs(self, x, action):
        logits, _ = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)

class RLHFPPOTrainer:
    def __init__(self, env, device="cuda", lr=2.5e-4, reward_lr=1e-4,
                 n_steps=128, n_epochs=4, batch_size=256, gamma=0.99, 
                 gae_lambda=0.95, clip_range=0.1, vf_coef=0.5, ent_coef=0.01,
                 max_grad_norm=0.5, kl_coef=0.1, reward_model_updates=5,
                 segment_length=25, n_initial_comparisons=500):
        
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
        self.kl_coef = kl_coef
        self.reward_model_updates = reward_model_updates
        self.segment_length = segment_length
        
        self.n_actions = env.action_space.n
        self.obs_shape = env.observation_space.shape
        self.num_envs = env.num_envs
        
        # Policy model
        self.model = RLHFActorCritic(self.n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        # Reference model
        self.ref_model = RLHFActorCritic(self.n_actions).to(device)
        self.ref_model.load_state_dict(self.model.state_dict())
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Reward model
        self.reward_model = RewardModel(self.n_actions).to(device)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=reward_lr)
        
        # Preference dataset
        self.preference_data = PreferenceDataset()
        self.preference_oracle = SyntheticPreferenceOracle()
        
        # Buffer with modified reward storage
        self.buffer = RolloutBuffer(n_steps, self.num_envs, self.obs_shape, device)
        
        # Trajectory buffer for preference collection
        self.trajectory_buffer = []
        
        self.total_timesteps = 0
        self.episode_rewards = []
        self.learned_rewards = []
        self.n_initial_comparisons = n_initial_comparisons
    
    def collect_initial_preferences(self, obs):
        while len(self.preference_data) < self.n_initial_comparisons:
            # Collect random trajectories
            traj_obs, traj_actions, traj_rewards = [], [], []
            
            for _ in range(self.segment_length * 2):
                action = np.array([self.env.envs[0].action_space.sample() 
                                   for _ in range(self.num_envs)])
                next_obs, rewards, _, _, _ = self.env.step(action)
                
                traj_obs.append(obs[0])
                traj_actions.append(action[0])
                traj_rewards.append(rewards[0])
                obs = next_obs
            
            # Split into two segments and get preference
            mid = self.segment_length
            traj_a = (traj_obs[:mid], traj_actions[:mid])
            traj_b = (traj_obs[mid:], traj_actions[mid:])
            
            reward_a = sum(traj_rewards[:mid])
            reward_b = sum(traj_rewards[mid:])
            
            preference = self.preference_oracle.get_preference(reward_a, reward_b)
            self.preference_data.add(traj_a, traj_b, preference)
        
        self.train_reward_model(epochs=10)
        return obs
    
    def train_reward_model(self, epochs=1):
        if len(self.preference_data) < 10:
            return 0.0
        
        total_loss = 0
        for _ in range(epochs):
            batch = self.preference_data.sample(min(64, len(self.preference_data)))
            
            loss = 0
            for traj_a, traj_b, pref in batch:
                obs_a = torch.tensor(np.array(traj_a[0]), device=self.device, dtype=torch.uint8)
                act_a = torch.tensor(np.array(traj_a[1]), device=self.device)
                obs_b = torch.tensor(np.array(traj_b[0]), device=self.device, dtype=torch.uint8)
                act_b = torch.tensor(np.array(traj_b[1]), device=self.device)
                
                # Predict rewards for both trajectories
                reward_a = self.reward_model(obs_a, act_a).sum()
                reward_b = self.reward_model(obs_b, act_b).sum()
                
                # Bradley-Terry model loss
                log_prob_a = torch.log(torch.sigmoid(reward_a - reward_b) + 1e-8)
                log_prob_b = torch.log(torch.sigmoid(reward_b - reward_a) + 1e-8)
                
                if pref == 0: # A preferred
                    loss -= log_prob_a
                elif pref == 1: # B preferred
                    loss -= log_prob_b
                else: # Tie
                    loss -= 0.5 * (log_prob_a + log_prob_b)
            
            loss = loss / len(batch)
            
            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / epochs
    
    def compute_learned_rewards(self, obs, actions):
        with torch.no_grad():
            obs_flat = obs.reshape(-1, *self.obs_shape)
            act_flat = actions.reshape(-1)
            rewards = self.reward_model(obs_flat, act_flat)
            return rewards.reshape(obs.shape[0], -1)
    
    def compute_kl_penalty(self, obs, actions):
        with torch.no_grad():
            ref_log_probs = self.ref_model.get_log_probs(obs, actions)
        
        curr_log_probs = self.model.get_log_probs(obs, actions)
        kl = curr_log_probs - ref_log_probs
        return kl
    
    def collect_rollouts(self, obs):
        self.buffer.reset()
        episode_infos = []
        traj_segment = {'obs': [], 'actions': [], 'true_rewards': []}
        
        for step in range(self.n_steps):
            obs_tensor = torch.as_tensor(obs, device=self.device)
            
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()
            next_obs, true_rewards, terminateds, truncateds, infos = self.env.step(action_np)
            dones = np.logical_or(terminateds, truncateds)
            
            # Get learned rewards
            with torch.no_grad():
                learned_reward = self.reward_model(
                    obs_tensor, 
                    torch.as_tensor(action_np, device=self.device)
                ).cpu().numpy()
            
            # Store in buffer with learned rewards
            self.buffer.add(obs, action_np, learned_reward, dones, log_prob, value)
            
            # Collect trajectory segments for preference learning
            traj_segment['obs'].append(obs[0])
            traj_segment['actions'].append(action_np[0])
            traj_segment['true_rewards'].append(true_rewards[0])
            
            if len(traj_segment['obs']) >= self.segment_length:
                self.trajectory_buffer.append(traj_segment)
                traj_segment = {'obs': [], 'actions': [], 'true_rewards': []}
            
            for i, info in enumerate(infos):
                if 'terminal_observation' in info:
                    if 'episode' in info:
                        episode_infos.append(info['episode'])
            
            obs = next_obs
            self.total_timesteps += self.num_envs
        
        self.generate_preference_comparisons()
        
        with torch.no_grad():
            last_value = self.model.get_value(torch.as_tensor(obs, device=self.device))
        
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        return obs, returns, advantages, episode_infos
    
    def generate_preference_comparisons(self):
        if len(self.trajectory_buffer) < 2:
            return
        
        # Sample pairs and get preferences
        n_pairs = min(5, len(self.trajectory_buffer) // 2)
        indices = np.random.choice(len(self.trajectory_buffer), n_pairs * 2, replace=False)
        
        for i in range(0, len(indices), 2):
            traj_a = self.trajectory_buffer[indices[i]]
            traj_b = self.trajectory_buffer[indices[i + 1]]
            
            reward_a = sum(traj_a['true_rewards'])
            reward_b = sum(traj_b['true_rewards'])
            
            preference = self.preference_oracle.get_preference(reward_a, reward_b)
            
            self.preference_data.add(
                (traj_a['obs'], traj_a['actions']),
                (traj_b['obs'], traj_b['actions']),
                preference
            )
        
        # Clear old trajectories
        self.trajectory_buffer = self.trajectory_buffer[-50:]
    
    def train_epoch(self, returns, advantages):
        b_obs = self.buffer.obs.reshape(-1, *self.obs_shape)
        b_actions = self.buffer.actions.reshape(-1)
        b_log_probs = self.buffer.log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        total_samples = self.n_steps * self.num_envs
        indices = np.arange(total_samples)
        
        pg_losses, vf_losses, entropy_losses, kl_losses = [], [], [], []
        
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
                
                # KL penalty
                kl_penalty = self.compute_kl_penalty(
                    b_obs[mb_indices], b_actions[mb_indices]
                ).mean()
                
                # Value loss
                vf_loss = 0.5 * ((new_value - b_returns[mb_indices]) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = entropy.mean()
                
                # Total loss with KL penalty
                loss = (pg_loss + self.vf_coef * vf_loss 
                        - self.ent_coef * entropy_loss 
                        + self.kl_coef * kl_penalty)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_losses.append(kl_penalty.item())
        
        return (np.mean(pg_losses), np.mean(vf_losses), 
                np.mean(entropy_losses), np.mean(kl_losses))
    
    def train(self, total_timesteps, callback=None):
        obs, _ = self.env.reset()
        
        # Collect initial preferences and train reward model
        obs = self.collect_initial_preferences(obs)
        
        while self.total_timesteps < total_timesteps:
            obs, returns, advantages, ep_infos = self.collect_rollouts(obs)
            
            # Train reward model periodically
            rm_loss = self.train_reward_model(epochs=self.reward_model_updates)
            
            # PPO update
            pg_loss, vf_loss, ent_loss, kl_loss = self.train_epoch(returns, advantages)
            
            for info in ep_infos:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
            
            if callback:
                callback({
                    'timesteps': self.total_timesteps,
                    'pg_loss': pg_loss,
                    'vf_loss': vf_loss,
                    'entropy': ent_loss,
                    'kl_loss': kl_loss,
                    'reward_model_loss': rm_loss,
                    'n_preferences': len(self.preference_data),
                    'episode_rewards': self.episode_rewards[-100:] if self.episode_rewards else []
                })
        
        return self.model
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ref_model_state_dict': self.ref_model.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'episode_rewards': self.episode_rewards
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ref_model.load_state_dict(checkpoint['ref_model_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.episode_rewards = checkpoint['episode_rewards']
