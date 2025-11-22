import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class PongEnvWrapper(gym.Wrapper):
    # Wrapper for Pong with standard Atari preprocessing
    def __init__(self, env_name="ALE/Pong-v5", frame_stack=4, frame_skip=4, 
                 resize_shape=(84, 84), clip_rewards=True):
        env = gym.make(env_name, render_mode=None)
        super().__init__(env)
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.resize_shape = resize_shape
        self.clip_rewards = clip_rewards
        self.frames = deque(maxlen=frame_stack)
        
        # Update observation space for stacked grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *resize_shape),
            dtype=np.uint8
        )
    
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.resize_shape, interpolation=cv2.INTER_AREA)
        return resized
    
    def get_stacked_frames(self):
        return np.array(self.frames, dtype=np.uint8)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self.preprocess_frame(obs)
        
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self.get_stacked_frames(), info
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        frames_for_max = []
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if i >= self.frame_skip - 2:
                frames_for_max.append(obs)
            if terminated or truncated:
                break
        
        # Max pool over last 2 frames to handle flickering
        if len(frames_for_max) == 2:
            obs = np.maximum(frames_for_max[0], frames_for_max[1])
        elif len(frames_for_max) == 1:
            obs = frames_for_max[0]
        
        processed = self.preprocess_frame(obs)
        self.frames.append(processed)
        
        if self.clip_rewards:
            total_reward = np.sign(total_reward)
        
        return self.get_stacked_frames(), total_reward, terminated, truncated, info
    
class PongEnvWithRender(PongEnvWrapper):    
    def __init__(self, **kwargs):
        kwargs['env_name'] = "ALE/Pong-v5"
        super().__init__(**kwargs)
        self.env.close()
        self.env = gym.make("ALE/Pong-v5", render_mode="human")

def make_pong_env(render=False, **kwargs):
    if render:
        return PongEnvWithRender(**kwargs)
    return PongEnvWrapper(**kwargs)


def make_vectorized_env(num_envs=8, **kwargs):
    def make_env():
        return PongEnvWrapper(**kwargs)
    
    envs = [make_env() for _ in range(num_envs)]
    return VectorizedEnv(envs)


class VectorizedEnv:
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
    
    def reset(self):
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list
    
    def step(self, actions):
        obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                final_obs = obs
                obs, _ = env.reset()
                info['terminal_observation'] = final_obs
            
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (np.stack(obs_list), np.array(rewards), 
                np.array(terminateds), np.array(truncateds), infos)
    
    def close(self):
        for env in self.envs:
            env.close()