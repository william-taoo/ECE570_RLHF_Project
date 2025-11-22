# ECE 570 Project

## Intro
In this project, I'm evaluating the performance differences between a preference-based PPO model and a standard PPO model with CartPole. This is a reinforcement learning (RL) task that uses OpenAI's Gymnasium to support the CartPole and Atari (pong) environment. There are no datasets required for this project, as everything is containerized within Gymnasium. In CartPole, the goal is to balance a pole on the cart for as long as possible. In the preference-based model, we are preferring the episode with the longer trajectory. In pong, you control a paddle and compete against a computer, trying to deflect the opponents ball while trying to score on theirs.

## Experiment
For each model, we execute a run 5 times, saving the average reward each time, and take the average again of those 5 runs. Each model is trained for 150 epochs, and have a benchmark score of 195.

## How to run
First, download all of the requirements in `requirements.txt` by running `pip install -r requirements.txt` in the terminal. Also run the command `autorom --accept-license` to accept the license. Then, run `python demo.py` to run the standard PPO vs. preference-based PPO model experiment. The hyperparameters of the models can be adjusted in the `if __name__ == "__main__"`.