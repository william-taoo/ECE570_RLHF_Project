import numpy as np
import torch
from envs.cartpole import CartPoleEnv
from models.ppo import PPO


def print_results(type, num_episodes, avg_reward, max_reward, max_episode):
    print(f"\n=== {type} Results ===")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Max reward of {max_reward} at episode {max_episode}")

def run_ppo_demo(hidden_dim,num_episodes, clip_ratio, lr, gamma, lam):
    # 1. Initialize environment
    env = CartPoleEnv(render_mode="human")
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.n

    # 2. Initialize PPO Agent
    agent = PPO(state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                clip_ratio=clip_ratio,
                lr=lr,
                gamma=gamma,
                lam=lam
                )

    # 3. Run episodes
    episode_rewards = []
    max_reward = 0
    max_episode = 0
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        # Storage for rollout data
        all_states, all_actions, all_log_probs, all_rewards, all_values, all_masks = [], [], [], [], [], []

        while not done:
            action, log_prob = agent.select_action(state) # agent chooses action
            next_state, reward, done, truncated, _ = env.step(action)
            # env.render() # Show gameplay

            # Store data
            all_states.append(state)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(reward)
            all_masks.append(1 - float(done or truncated))

            with torch.no_grad():
                _, value = agent.policy(torch.FloatTensor(state).unsqueeze(0))
                all_values.append(value.item())

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        if total_reward > max_reward:
            max_reward = total_reward
            max_episode = ep

        # Compute returns and advantages
        returns, advantages = agent.compute_returns(all_rewards, all_values, all_masks)
        np_states = np.array(all_states)
        np_actions = np.array(all_actions)
        np_log_probs = np.array(all_log_probs)

        # Update policy
        agent.update(np_states, np_actions, np_log_probs, returns, advantages, epochs=10, batch_size=64)

        # Clear
        all_states.clear()
        all_actions.clear()
        all_log_probs.clear()
        all_rewards.clear()
        all_values.clear()
        all_masks.clear()

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")

    env.close()

    # 4. Print average reward
    avg_reward = np.mean(episode_rewards)
    print_results("Standard PPO", num_episodes, avg_reward, max_reward, max_episode)

def run_rlhf_ppo_demo():
    pass

if __name__ == "__main__":
    hidden_dim = 128
    num_episodes = 150
    clip_ratio = 0.12
    lr = 0.0008
    gamma = 0.99
    lam = 0.95

    run_ppo_demo(hidden_dim, num_episodes, clip_ratio, lr, gamma, lam)
