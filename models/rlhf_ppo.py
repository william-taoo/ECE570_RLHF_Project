import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque, namedtuple

SEGMENT_LENGTH = 20 # Length of trajectory segment
PREF_PAIRS = 2000 # Number of preference pairs used to train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardModel(nn.Module):
    def __init__(self, observation_dim, segment_len=SEGMENT_LENGTH):
        super(RewardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(observation_dim * segment_len, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output a single reward for the segment
        )

    def forward(self, segment_observations):
        B, L, D = segment_observations.size() # Batch, Length, Dimension
        x = segment_observations.view(B, L * D) # Flatten the segment
        return self.fc(x).squeeze(-1) # Return shape (B,)

# Wrapper env that replaces env reward with learned reward model
class LearnedRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, segment_len=SEGMENT_LENGTH):
        super(LearnedRewardWrapper, self).__init__(env)
        self.model = reward_model
        self.seg_len = segment_len
        self.observation_buffer = deque(maxlen=segment_len)
        self.r_mean = 0.0
        self.r_var = 1.0
        self.count = 1e-4
        
    def reset(self, seed=None, options=None):
        self.observation_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self.observation_buffer.append(obs)
        return obs, info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.observation_buffer.append(observation)

        # Create segment
        segment = list(self.observation_buffer)
        if len(segment) < self.seg_len:
            segment = [segment[-1]] * (self.seg_len - len(segment)) + segment
        segment_arr = np.array(segment, dtype=np.float32)[None,...] # (1, L, D)

        with torch.no_grad():
            t = torch.tensor(segment_arr, device=device)
            r = self.model(t).item()
        
        self.count += 1
        old_mean = self.r_mean
        self.r_mean += (r - self.r_mean) / self.count
        self.r_var += (r - old_mean) * (r - self.r_mean)
        std = np.sqrt(self.r_var / max(1.0, self.count - 1))

        if std < 1e-6:
            std = 1.0

        r_norm = (r - self.r_mean) / std
        return observation, float(r_norm), terminated, truncated, info

# Collect trajectories with a policy     
def collect_trajectories(env, epochs=150, max_len=500):
    trajectories = []

    for epoch in range(epochs):
        observation = env.reset()
        observation_seq = []
        done = False
        total = 0

        for t in range(max_len):
            action = env.action_space.sample() # Random action
            next_observation, reward, done, info = env.step(action)
            observation_seq.append(observation)
            observation = next_observation
            total += reward

            if done:
                break

        trajectories.append({
            "observations": np.array(observation_seq),
            "length": len(observation_seq),
            "return": total
        })
    
    return trajectories

# Turn trajectories into segments
def trajectory_to_segments(trajectories, segment_length=SEGMENT_LENGTH):
    segments = []

    for traj in trajectories:
        observations = traj["observations"]
        T = len(observations)

        # Skip short trajectories
        if T < 2:
            continue

        for start in range(0, max(1, T - segment_length + 1), max(1, segment_length // 2)):
            seg = observations[start:start + segment_length]

            # If segment shorter than segment_length, pad with last observation
            if seg.shape[0] < segment_length:
                pad = np.repeat(seg[-1:,...], segment_length - seg.shape[0], axis=0)
                seg = np.concatenate([seg, pad], axis=0)
            segments.append({"observations": seg, "traj_length": T})
    
    return segments

def create_pref_pairs(segments, num_pairs=PREF_PAIRS, noise=0.05):
    pairs = []
    num_segments = len(segments)

    for _ in range(num_pairs):
        a, b = random.sample(range(num_segments), 2)
        seg_a = segments[a]
        seg_b = segments[b]

        # Preference based on trajectory return - prefer longer trajectory
        if seg_a["traj_length"] > seg_b["traj_length"]:
            label = 0
        elif seg_b["traj_length"] > seg_a["traj_length"]:
            label = 1
        else:
            label = random.choice([0, 1]) # Pick randomly

        # Add noise 
        if random.random() < noise:
            label = 1 - label

        pairs.append((seg_a["observations"], seg_b["observations"], label))
    
    return pairs

def train_reward_model(model, pairs, epochs, batch_size, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss() # Binary cross-entropy loss
    model.to(device)
    dataset = pairs
    dataset_len = len(dataset)
    
    for epoch in range(epochs):
        random.shuffle(pairs)
        losses = []
        for i in range(0, dataset_len, batch_size):
            batch = dataset[i:i + batch_size]
            a_obs = np.stack([b[0] for b in batch]).astype(np.float32) # (B, L, D)
            b_obs = np.stack([b[1] for b in batch]).astype(np.float32)
            
            # 0 -> A preferred, 1 -> B preferred
            labels = np.array([0 if b[2] == 0 else 1 for b in batch], dtype=np.float32)
            a_tensor = torch.tensor(a_obs, device=device)
            b_tensor = torch.tensor(b_obs, device=device)
            rA = model(a_tensor)
            rB = model(b_tensor)
            logits = torch.stack([rA, rB], dim=1) # (B, 2)

            # Compute probabilities
            prob_b = torch.softmax(logits, dim=1)[:, 1]
            target = torch.tensor(labels, device=device)
            
            loss = criterion(prob_b, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")
