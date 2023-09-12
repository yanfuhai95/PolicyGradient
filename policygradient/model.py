from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class PolicyNet(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return F.softmax(self.linear_stack(x), dim=1)


class REINFORCE:
    
    def __init__(self, 
                 state_dim, hidden_dim, action_dim, 
                 device="cpu", 
                 lr=1e-4, 
                 gamma=0.99,
                 optimizer=optim.AdamW):
        self.device = device
        self.gamma = gamma
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr, amsgrad=True)


    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


    def optimize(self, transitions):
        G = 0
        self.optimizer.zero_grad()
        for t in reversed(transitions):
            state = torch.tensor([t.state], dtype=torch.float).to(self.device)
            action = torch.tensor([t.action]).view(-1, 1).to(self.device)
            
            prob_dist = self.policy_net(state)
            log_prob = torch.log(prob_dist.gather(1, action))
            G = self.gamma * G + t.reward
            loss = -log_prob * G
            loss.backward()

        self.optimizer.step()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))
