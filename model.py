import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Deque
from tqdm import tqdm

# from environment import RiskSensitiveGridWorld


# DRQN MODEL

class DRQN(nn.Module):
    def __init__(self, obs_dim=75, hidden_dim=128, n_actions=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_q = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs, hidden):
        """
        obs: (B, T, obs_dim)
        hidden: (h0, c0) each shape (1, B, hidden_dim)
        """
        x = F.relu(self.fc1(obs))
        x, hidden = self.lstm(x, hidden)
        q = self.fc_q(x)  # (B, T, n_actions)
        return q, hidden

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )


# SEQUENCE REPLAY BUFFER

class SequenceReplayBuffer:
    def __init__(self, capacity=150_000, seq_len=20):
        self.buffer: Deque = deque(maxlen=capacity)
        self.seq_len = seq_len

    def push(self, episode):
        if len(episode) >= self.seq_len:
            self.buffer.append(episode)

    def sample(self, batch_size):
        episodes = random.sample(self.buffer, batch_size)
        batch = []
        for ep in episodes:
            start = random.randint(0, len(ep) - self.seq_len)
            batch.append(ep[start:start + self.seq_len])
        return batch

    def __len__(self):
        return len(self.buffer)


