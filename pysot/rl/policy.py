"""Actor-critic policy that sets HiFT's post-processing parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningNorm(nn.Module):
    """Running mean/std of observations, saved with the policy."""

    def __init__(self, dim, clip=5.0):
        super(RunningNorm, self).__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean += delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class ActorCritic(nn.Module):
    """Gaussian policy over pre-squash actions u; the action in [0,1] is sigmoid(u).

    The mean starts at init_action01 (HiFT's hand-tuned constants), so
    training begins from the baseline tracker's behaviour.
    """

    def __init__(self, obs_dim, init_action01, hidden=256, init_log_std=-1.0):
        super(ActorCritic, self).__init__()
        act_dim = len(init_action01)
        self.norm = RunningNorm(obs_dim)
        self.pi = mlp(obs_dim, hidden, act_dim)
        self.v = mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        p = np.clip(np.asarray(init_action01, dtype=np.float64), 1e-3, 1 - 1e-3)
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.copy_(torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32))

    def dist(self, x):
        """x: observations already passed through self.norm."""
        return Normal(self.pi(x), self.log_std.exp()), self.v(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False, normed=False):
        d, v = self.dist(obs if normed else self.norm(obs))
        u = d.mean if deterministic else d.sample()
        return u, d.log_prob(u).sum(-1), v

    def evaluate(self, x, u):
        d, v = self.dist(x)
        return d.log_prob(u).sum(-1), d.entropy().sum(-1), v

    @staticmethod
    def squash(u):
        return torch.sigmoid(u)
