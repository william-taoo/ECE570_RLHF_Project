import gymnasium as gym


class CartPoleEnv:
    '''
    A wrapper for the CartPole-v1 environment
    '''
    def __init__(self, render_mode):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
