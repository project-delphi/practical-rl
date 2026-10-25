"""Checkpoint 6.2's wrong versions, for scripts/threshold_protocol.py only (it reads
labs/mutants/protocol/<module>.py next to labs/mutants/<module>.py; day-3 decision 16).

6.2 guards the platform, not the participant (OPEN_QUESTIONS C38): it compares the live seed's
greedy evaluations, already computed by Exercise 6's run cell, with a threshold the protocol sets
by training the lab's provided train_dqn and CONFIG on many seeds. Its wrong versions replace
that provided code, never a function participants write, so they live here, not in
labs/mutants/m09.py (tests/test_mutants.py); verify mode never reads this folder and skips 6.2.

- random: uniform random play.
- untrained: the initial network (seeded as dqn_parts seeds it), never updated, evaluated as the
  logger would. An untrained network sometimes balances for a while, so it is the stronger test.
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


@mutant(ex=6, replaces="train_dqn", id="untrained", why="never learns: the initial network")
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):  # noqa: F811
    torch.manual_seed(seed)  # dqn_parts seeds torch first, so this is the run's initial network
    env = make_env()
    q_net = make_q_net(env.observation_space.shape[0], env.action_space.n, config)  # noqa: F821
    env.close()
    # Its greedy evaluations, as the logger would make them on the grid: the network never
    # changes and evaluate_greedy starts from fixed states, so every evaluation is the same.
    every, episodes = log_options.get("eval_every"), log_options.get("eval_episodes", 10)
    grid = list(range(every, steps + 1, every)) if every else []
    value = float(evaluate_greedy(q_net, make_env, episodes)["returns"].mean())  # noqa: F821
    return q_net, {
        "eval_step": np.array(grid),
        "eval_return": np.full(len(grid), value, np.float32),
    }
