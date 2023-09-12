import time
from itertools import count

import torch
import gymnasium as gym

from policygradient.model import REINFORCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env():
    env = gym.make("CartPole-v1", render_mode="human")
    return env

if __name__ == "__main__":
    env = make_env()
    state, _ = env.reset()

    state_dim = len(state)
    hidden_dim = 128
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, hidden_dim, action_dim, device=device)
    agent.load('model/cart_pole.pth')

    for i in count():
        action = agent.select_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
         
        if terminated:
            print("Terminated")
            break
        
        if truncated:
            print("Truncated at", i+1)
            break

        state = observation

    env.close()
