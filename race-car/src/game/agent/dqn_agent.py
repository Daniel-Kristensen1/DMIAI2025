import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.game.agent.base_agent import BaseAgent



# Definer QNetwork her (eller importer den fra en anden fil)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.9,
                 epsilon_min=0.05, epsilon_decay = 0.99998, buffer_size=10000, batch_size=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_net = QNetwork(obs_dim, action_dim)
        self.target_net = QNetwork(obs_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        if isinstance(observation, torch.Tensor):
            obs = observation.unsqueeze(0).to(self.device)
        else:
            obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in obs]
        next_obs = [no.cpu().numpy() if isinstance(no, torch.Tensor) else no for no in next_obs]

        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(obs).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
