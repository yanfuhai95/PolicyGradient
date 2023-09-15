import time
from itertools import count

import torch
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from policygradient.model import REINFORCE


def make_env():
    env = gym.make("CartPole-v1")
    return env


def plot_result(episode_durations, episode_rewards):
    fig, axs = plt.subplots(ncols=2, figsize=(24, 10))
    fig.suptitle("Result")

    # Duration
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Duration')
    axs[0].plot(durations_t.numpy())

    # Reward
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Reward')
    axs[1].plot(rewards_t.numpy())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_times = 100

    env = make_env()
    state, _ = env.reset()

    state_dim = len(state)
    hidden_dim = 128
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = REINFORCE(state_dim, hidden_dim, action_dim, device=device)
    agent.load('model/cart_pole.pth')

    episode_durations = []
    episode_rewards = []

    with tqdm(total=n_times) as pbar:
        for i_times in range(n_times):
            total_reward = 0

            duration = 0
            state, _ = env.reset()
            for t in count():
                action = agent.select_action(state)
                observation, reward, terminated, truncated, info = env.step(
                    action)
                total_reward += reward

                if terminated or truncated:
                    duration = t + 1
                    break

                state = observation  # move to next state

            episode_durations.append(duration)
            episode_rewards.append(total_reward)

            pbar.set_postfix({
                'duration':  '%d' % duration,
                'reward': '%d' % total_reward,
            })
            pbar.update(1)

    plot_result(episode_durations, episode_rewards)
    env.close()
