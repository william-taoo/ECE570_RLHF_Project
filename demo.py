import numpy as np
from envs.cartpole import CartPoleEnv
from models.ppo import PPO


def run_demo(num_episodes=25):
    # 1. Initialize environment
    env = CartPoleEnv(render_mode=None)
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.n
    clip_ratio = 0.2
    lr = 0.01
    gamma = 0.99
    lam = 0.95

    # 2. Initialize PPO Agent
    agent = PPO(state_dim=state_dim,
                action_dim=action_dim,
                clip_ratio=clip_ratio,
                lr=lr,
                gamme=gamma,
                lam=lam
                )

    # 3. Run a few episodes (with UNTRAINED policy)
    returns = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = agent.select_action(state) # agent chooses action
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if done or truncated:
                break

        returns.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")

    env.close()

    # 4. Print average reward (prototype result)
    avg_reward = np.mean(returns)
    print("\n=== Prototype Result ===")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    run_demo()
