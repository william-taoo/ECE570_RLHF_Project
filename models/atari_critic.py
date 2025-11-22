import torch.nn as nn


class AtariEncoder(nn.Module):
    '''
    Encoder for Atari frames - uses CNN 
    '''
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x / 255.0
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class ActorCriticCNN(nn.Module):
    '''
    An actor-critic network for PPO
    '''
    def __init__(self, num_actions, in_channels=4, feature_dim=512):
        super().__init__()
        self.encoder = AtariEncoder(in_channels, feature_dim)
        self.policy_head = nn.Linear(feature_dim, num_actions)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, obs):
        feat = self.encoder(obs)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value.squeeze(-1)

class RewardModelCNN(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256, seg_len=20):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, feature_dim), nn.ReLU()
        )
        self.post = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, segments):
        B, L, H, W = segments.shape
        x = segments.view(B * L, 1, H, W)
        x = x / 255.0
        enc = self.frame_encoder(x)
        enc = enc.view(B, L, -1)
        pooled = enc.mean(dim=1)
        score = self.post(pooled).squeeze(-1)
        return score
