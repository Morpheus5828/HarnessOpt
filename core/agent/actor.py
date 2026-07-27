import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent.config import *



class Actor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.ln1(self.l1(state)))
        a = F.relu(self.ln2(self.l2(a)))
        return torch.tanh(self.l3(a))


class ActorSAC(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super(ActorSAC, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.ln1(self.l1(state)))
        a = F.relu(self.ln2(self.l2(a)))
        mean = self.mean_linear(a)
        log_std = torch.clamp(self.log_std_linear(a), min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(1, keepdim=True), torch.tanh(mean)


class ActorBiGRU(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=64):
        super(ActorBiGRU, self).__init__()
        self.inp = nn.Linear(state_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, action_dim)

    def forward(self, state_seq, lengths=None):
        self.rnn.flatten_parameters()
        h = F.relu(self.inp(state_seq))
        out = _run_bigru(self.rnn, h, lengths, state_seq.shape[1])
        return torch.tanh(self.head(out))


def _run_bigru(rnn, h, lengths, total_length):
    if lengths is None:
        out, _ = rnn(h)
        return out
    packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                               enforce_sorted=False)
    out, _ = rnn(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)
    return out
