import torch
import torch.nn as nn
import torch.nn.functional as F
from core.agent.config import *
from core.agent.actor import _run_bigru
from core.agent.tool import *

class Critic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.ln4 = nn.LayerNorm(256)
        self.l5 = nn.Linear(256, 256)
        self.ln5 = nn.LayerNorm(256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l3(F.relu(self.ln2(self.l2(F.relu(self.ln1(self.l1(sa)))))))
        q2 = self.l6(F.relu(self.ln5(self.l5(F.relu(self.ln4(self.l4(sa)))))))
        return q1, q2

class CriticBiGRU(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=64):
        super(CriticBiGRU, self).__init__()
        self.inp1 = nn.Linear(state_dim + action_dim, hidden)
        self.rnn1 = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head1 = nn.Linear(2 * hidden, 1)
        self.inp2 = nn.Linear(state_dim + action_dim, hidden)
        self.rnn2 = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head2 = nn.Linear(2 * hidden, 1)

    def forward(self, state_seq, action_seq, lengths=None):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        sa = torch.cat([state_seq, action_seq], dim=-1)
        L = state_seq.shape[1]
        q1 = self.head1(_run_bigru(self.rnn1, F.relu(self.inp1(sa)), lengths, L))
        q2 = self.head2(_run_bigru(self.rnn2, F.relu(self.inp2(sa)), lengths, L))
        return q1.squeeze(-1), q2.squeeze(-1)
