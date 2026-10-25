"""Checkpoint 6.2's wrong version for scripts/threshold_protocol.py (through
scripts/protocol_m09.py): random play in place of the lab's provided train_dqn.

6.2 guards the platform, not the participant (OPEN_QUESTIONS C38): it compares the live seed's
greedy return, already computed by Exercise 6's run cell, with a threshold the protocol sets by
training the lab's train_dqn and CONFIG on many seeds against random play. labs/mutants/m09.py
holds only mutants of functions participants write, so this one lives here; verify mode never
reads this folder (6.2 is stochastic, so verify skips it).
"""

import numpy as np
import torch
from torch import nn

from prl.lab import mutant


@mutant(ex=6, replaces="train_dqn", id="random", why="never learns: uniform random play")
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):
    class UniformQ(nn.Module):
        """Values drawn afresh at every call: its greedy action is uniform random play."""

        def __init__(self, n_actions):
            super().__init__()
            self.n_actions = n_actions

        def forward(self, obs):
            return torch.rand(obs.shape[0], self.n_actions)

    torch.manual_seed(seed)
    env = make_env()
    q_net = UniformQ(env.action_space.n)
    env.close()
    # Its greedy evaluations, reported as the logger would (the provided evaluate_greedy).
    every, episodes = log_options.get("eval_every"), log_options.get("eval_episodes", 10)
    grid = list(range(every, steps + 1, every)) if every else []
    returns = [evaluate_greedy(q_net, make_env, episodes)["returns"].mean() for _ in grid]  # noqa: F821
    return q_net, {"eval_step": np.array(grid), "eval_return": np.array(returns, np.float32)}
