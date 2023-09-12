from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim


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
        return nn.Softmax(self.linear_stack(x), dim=1)


class REINFORCE:
    
    def __init__(self, 
                 state_dim, hidden_dim, action_dim, 
                 device="cpu", 
                 batch_size=100,
                 lr=1e-4, 
                 gamma=0.99,
                 optimizer=optim.AdamW):
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr, amsgrad=True)


    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action.item()


    def optimize(self, transitions):
        self.optimizer.zero_grad()
        
        G = 0
        for t in transitions:
            state = torch.tensor([t.state], dtype=torch.float).to(self.device)
            action = torch.tensor([t.action]).view(-1, -1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + t.reward
            loss = -log_prob * G
            loss.backward()

        self.optimizer.step()
