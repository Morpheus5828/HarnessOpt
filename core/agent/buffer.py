import numpy as np
import torch
from config import *

from core.agent.config import CONFIG, device, STATE_DIM, ACTION_DIM

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000, use_cer=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.use_cer = use_cer
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward):
        batch_size = state.shape[0]
        end_idx = min(self.ptr + batch_size, self.max_size)
        actual_batch_size = end_idx - self.ptr
        self.state[self.ptr:end_idx] = state[:actual_batch_size]
        self.action[self.ptr:end_idx] = action[:actual_batch_size]
        self.next_state[self.ptr:end_idx] = next_state[:actual_batch_size]
        self.reward[self.ptr:end_idx] = reward[:actual_batch_size].reshape(-1, 1)
        self.ptr = (self.ptr + actual_batch_size) % self.max_size
        self.size = min(self.size + actual_batch_size, self.max_size)

    def sample(self, batch_size):
        if self.use_cer:
            ind = np.random.randint(0, self.size, size=batch_size - 1)
            ind = np.append(ind, (self.ptr - 1) % self.max_size)
        else:
            ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device)
        )

class SequenceReplayBuffer:
    def __init__(self, state_dim, action_dim, max_len=150, max_size=4000):
        self.max_len = max_len
        self.max_size = max_size
        self.ptr = 0
        self.n_seq = 0
        self.state = np.zeros((max_size, max_len, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, max_len, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, max_len, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, max_len), dtype=np.float32)
        self.mask = np.zeros((max_size, max_len), dtype=np.float32)
        self.lengths = np.zeros(max_size, dtype=np.int64)

    @property
    def size(self):
        return int(self.lengths[:self.n_seq].sum())

    def add(self, state, action, next_state, reward):
        L = min(len(state), self.max_len)
        if L < 1:
            return
        i = self.ptr
        self.state[i].fill(0.0)
        self.action[i].fill(0.0)
        self.next_state[i].fill(0.0)
        self.reward[i].fill(0.0)
        self.mask[i].fill(0.0)
        self.state[i, :L] = state[:L]
        self.action[i, :L] = action[:L]
        self.next_state[i, :L] = next_state[:L]
        self.reward[i, :L] = np.asarray(reward).reshape(-1)[:L]
        self.mask[i, :L] = 1.0
        self.lengths[i] = L
        self.ptr = (self.ptr + 1) % self.max_size
        self.n_seq = min(self.n_seq + 1, self.max_size)

    def sample(self, n_sequences):
        ind = np.random.randint(0, self.n_seq, size=min(n_sequences, self.n_seq))
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.mask[ind]).to(device),
            torch.LongTensor(self.lengths[ind]),
        )
