import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from envs.atari import make_vectorized_env, make_pong_env
from models.atari_ppo import PPOTrainer
from models.rlhf_atari import RLHFPPOTrainer


def smooth(data, window=10):
    """Smooth data with moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def run_ppo_experiment(total_timesteps, num_envs=8, device="cuda", seed=42):
    """Run standard PPO experiment."""
    print("\n" + "="*60)
    print("Starting Standard PPO Experiment")
    print("="*60)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = make_vectorized_env(num_envs=num_envs)
    trainer = PPOTrainer(env, device=device)
    
    logs = {'timesteps': [], 'rewards': [], 'pg_loss': [], 'vf_loss': [], 'entropy': []}
    
    def callback(info):
        if info['timesteps'] % 10000 < 1000:
            avg_reward = np.mean(info['episode_rewards']) if info['episode_rewards'] else 0
            print(f"PPO | Steps: {info['timesteps']:,} | Avg Reward: {avg_reward:.2f}")
            
            logs['timesteps'].append(info['timesteps'])
            logs['rewards'].append(avg_reward)
            logs['pg_loss'].append(info['pg_loss'])
            logs['vf_loss'].append(info['vf_loss'])
            logs['entropy'].append(info['entropy'])
    
    model = trainer.train(total_timesteps, callback=callback)
    
    env.close()
    return model, trainer.episode_rewards, logs

def run_rlhf_experiment(total_timesteps, num_envs=8, device="cuda", seed=42):
    """Run RLHF PPO experiment."""
    print("\n" + "="*60)
    print("Starting RLHF PPO Experiment")
    print("="*60)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = make_vectorized_env(num_envs=num_envs)
    trainer = RLHFPPOTrainer(env, device=device)
    
    logs = {
        'timesteps': [], 'rewards': [], 'pg_loss': [], 'vf_loss': [], 
        'entropy': [], 'kl_loss': [], 'rm_loss': [], 'n_preferences': []
    }
    
    def callback(info):
        if info['timesteps'] % 10000 < 1000:
            avg_reward = np.mean(info['episode_rewards']) if info['episode_rewards'] else 0
            print(f"RLHF | Steps: {info['timesteps']:,} | Avg Reward: {avg_reward:.2f} | "
                  f"KL: {info['kl_loss']:.4f} | Prefs: {info['n_preferences']}")
            
            logs['timesteps'].append(info['timesteps'])
            logs['rewards'].append(avg_reward)
            logs['pg_loss'].append(info['pg_loss'])
            logs['vf_loss'].append(info['vf_loss'])
            logs['entropy'].append(info['entropy'])
            logs['kl_loss'].append(info['kl_loss'])
            logs['rm_loss'].append(info['reward_model_loss'])
            logs['n_preferences'].append(info['n_preferences'])
    
    model = trainer.train(total_timesteps, callback=callback)
    
    env.close()
    return model, trainer.episode_rewards, logs

def evaluate_model(model, n_episodes=10, device="cuda", render=False):
    """Evaluate a trained model."""
    env = make_pong_env(render=render)
    
    model.eval()
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_tensor)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")
    
    env.close()
    return episode_rewards

def plot_comparison(ppo_rewards, rlhf_rewards, ppo_logs, rlhf_logs, save_path="results"):
    """Create comparison plots."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards over time
    ax1 = axes[0, 0]
    if ppo_rewards:
        ppo_smooth = smooth(ppo_rewards, window=50)
        ax1.plot(ppo_smooth, label='PPO', alpha=0.8, color='blue')
    if rlhf_rewards:
        rlhf_smooth = smooth(rlhf_rewards, window=50)
        ax1.plot(rlhf_smooth, label='RLHF PPO', alpha=0.8, color='orange')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards (Smoothed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average reward over timesteps
    ax2 = axes[0, 1]
    if ppo_logs['timesteps']:
        ax2.plot(ppo_logs['timesteps'], ppo_logs['rewards'], 
                 label='PPO', marker='o', markersize=3, color='blue')
    if rlhf_logs['timesteps']:
        ax2.plot(rlhf_logs['timesteps'], rlhf_logs['rewards'], 
                 label='RLHF PPO', marker='o', markersize=3, color='orange')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Learning Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Policy loss comparison
    ax3 = axes[1, 0]
    if ppo_logs['pg_loss']:
        ax3.plot(ppo_logs['timesteps'], ppo_logs['pg_loss'], 
                 label='PPO', alpha=0.8, color='blue')
    if rlhf_logs['pg_loss']:
        ax3.plot(rlhf_logs['timesteps'], rlhf_logs['pg_loss'], 
                 label='RLHF PPO', alpha=0.8, color='orange')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Policy Loss')
    ax3.set_title('Policy Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Entropy comparison
    ax4 = axes[1, 1]
    if ppo_logs['entropy']:
        ax4.plot(ppo_logs['timesteps'], ppo_logs['entropy'], 
                 label='PPO', alpha=0.8, color='blue')
    if rlhf_logs['entropy']:
        ax4.plot(rlhf_logs['timesteps'], rlhf_logs['entropy'], 
                 label='RLHF PPO', alpha=0.8, color='orange')
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/comparison_plots.png", dpi=150)
    plt.show()
    
    # RLHF-specific plots
    if rlhf_logs['kl_loss']:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        
        axes2[0].plot(rlhf_logs['timesteps'], rlhf_logs['kl_loss'], color='green')
        axes2[0].set_xlabel('Timesteps')
        axes2[0].set_ylabel('KL Divergence')
        axes2[0].set_title('RLHF: KL Penalty from Reference Policy')
        axes2[0].grid(True, alpha=0.3)
        
        axes2[1].plot(rlhf_logs['timesteps'], rlhf_logs['n_preferences'], color='purple')
        axes2[1].set_xlabel('Timesteps')
        axes2[1].set_ylabel('Number of Preferences')
        axes2[1].set_title('RLHF: Accumulated Preference Comparisons')
        axes2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/rlhf_specific_plots.png", dpi=150)
        plt.show()

def print_summary(ppo_rewards, rlhf_rewards, ppo_eval, rlhf_eval):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print("\nTraining Performance (last 100 episodes):")
    if ppo_rewards:
        print(f"  PPO:      Mean={np.mean(ppo_rewards[-100:]):.2f}, "
              f"Std={np.std(ppo_rewards[-100:]):.2f}, "
              f"Max={np.max(ppo_rewards[-100:]):.2f}")
    if rlhf_rewards:
        print(f"  RLHF PPO: Mean={np.mean(rlhf_rewards[-100:]):.2f}, "
              f"Std={np.std(rlhf_rewards[-100:]):.2f}, "
              f"Max={np.max(rlhf_rewards[-100:]):.2f}")
    
    print("\nEvaluation Performance:")
    if ppo_eval:
        print(f"  PPO:      Mean={np.mean(ppo_eval):.2f}, "
              f"Std={np.std(ppo_eval):.2f}")
    if rlhf_eval:
        print(f"  RLHF PPO: Mean={np.mean(rlhf_eval):.2f}, "
              f"Std={np.std(rlhf_eval):.2f}")
    
    print("\n" + "="*60)

def main():
    # Configuration
    TOTAL_TIMESTEPS = 1000000
    NUM_ENVS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    EVAL_EPISODES = 10
    
    print(f"Device: {DEVICE}")
    print(f"Total timesteps per experiment: {TOTAL_TIMESTEPS:,}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    ppo_model, ppo_rewards, ppo_logs = run_ppo_experiment(
        TOTAL_TIMESTEPS, NUM_ENVS, DEVICE, SEED
    )
    
    rlhf_model, rlhf_rewards, rlhf_logs = run_rlhf_experiment(
        TOTAL_TIMESTEPS, NUM_ENVS, DEVICE, SEED
    )
    
    # Evaluate models
    print("\n" + "="*60)
    print("Evaluating PPO Model")
    print("="*60)
    ppo_eval = evaluate_model(ppo_model, EVAL_EPISODES, DEVICE)
    
    print("\n" + "="*60)
    print("Evaluating RLHF PPO Model")
    print("="*60)
    rlhf_eval = evaluate_model(rlhf_model, EVAL_EPISODES, DEVICE)
    
    # Plot and summarize
    # plot_comparison(ppo_rewards, rlhf_rewards, ppo_logs, rlhf_logs, results_dir)
    print_summary(ppo_rewards, rlhf_rewards, ppo_eval, rlhf_eval)
    
    # Save results
    # results = {
    #     'config': {
    #         'total_timesteps': TOTAL_TIMESTEPS,
    #         'num_envs': NUM_ENVS,
    #         'seed': SEED
    #     },
    #     'ppo': {
    #         'training_rewards': ppo_rewards,
    #         'eval_rewards': ppo_eval,
    #         'logs': {k: [float(x) for x in v] for k, v in ppo_logs.items()}
    #     },
    #     'rlhf': {
    #         'training_rewards': rlhf_rewards,
    #         'eval_rewards': rlhf_eval,
    #         'logs': {k: [float(x) for x in v] for k, v in rlhf_logs.items()}
    #     }
    # }
    
    # with open(f"{results_dir}/results.json", 'w') as f:
    #     json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()