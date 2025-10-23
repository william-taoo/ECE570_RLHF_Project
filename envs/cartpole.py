import gymnasium as gym
import numpy as np


class CartPoleEnv():
    '''
    A wrapper for the CartPole-v1 environment
    '''
    def __init__(self, render_mode):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        observation, info = self.env.reset()
        return np.array(observation, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return np.array(obs, dtype=np.float32), float(reward), done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
