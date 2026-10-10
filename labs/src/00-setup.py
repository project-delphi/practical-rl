# ---
# jupyter:
#   prl:
#     module: m00
#     pins:
#       gymnasium: "1.4.0"
# ---

# %% [markdown] role="ex0-head"
# First, find out what you are running on. Every lab this week prints this at the top, and
# every run record stores it, so a result always says where it came from.

# %% [markdown] role="ex0-predict"
# Before you run the next cell, write down: does this runtime have a GPU? How many CPU cores
# do you expect it to have?

# %% role="ex0-run"
import prl.runtime

rt = prl.runtime.detect()
print(rt.summary())
print("CPU model:", rt.cpu)

# %% [markdown] role="ex0-explain"
# Compare with your prediction. Almost every lab this week is designed for a **CPU** runtime:
# the networks are small, so speed is limited by stepping the environment, not by matrix
# arithmetic. Only the Module 12 stretch and one capstone option are designed for a T4 GPU.

# %% role="ex0-chk"
from prl.checks import m00

lab.check(0, m00.check_runtime, rt)

# %% [markdown]
# ### How fast is this runtime?
#
# The next cell steps CartPole with random actions, then (if PyTorch is installed) trains a
# small network, for a few seconds each. The numbers go into your run record; we use records
# like yours to size the lab budgets. You do not need to do anything with them.

# %%
import time

import gymnasium as gym

budget = lab.use_settings(
    prl.runtime.settings({"bench_seconds": 5.0}, {"bench_seconds": 1.0}, designed="colab-cpu")
)

bench_env = gym.make("CartPole-v1")
bench_env.reset(seed=0)
steps, start = 0, time.perf_counter()
while time.perf_counter() - start < budget.bench_seconds:
    _, _, terminated, truncated, _ = bench_env.step(bench_env.action_space.sample())
    steps += 1
    if terminated or truncated:
        bench_env.reset()
env_rate = steps / (time.perf_counter() - start)
lab.metric("cartpole_env_steps_per_s", round(env_rate))
print(f"CartPole: {env_rate:,.0f} environment steps per second")

try:
    import torch

    net = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x, y = torch.randn(64, 4), torch.randn(64, 2)
    updates, start = 0, time.perf_counter()
    while time.perf_counter() - start < budget.bench_seconds:
        loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        updates += 1
    mlp_rate = updates / (time.perf_counter() - start)
    lab.metric("mlp_updates_per_s", round(mlp_rate))
    lab.metric("torch_threads", torch.get_num_threads())
    print(f"Small MLP: {mlp_rate:,.0f} gradient updates per second (batch 64)")
except ImportError:
    print("PyTorch is not installed here; skipping the network benchmark.")

# %% [markdown] role="ex1-head"
# A random agent picks every action uniformly at random. It is the floor every method this
# week must beat, and checkpoints often ask for "at least 3x random".

# %% [markdown] role="ex1-predict"
# CartPole-v1 pays +1 for every step the pole stays up, and stops at 500 steps. What average
# return do you expect from a random agent? Write a number.


# %% role="ex1-stub"
def run_episode(env, seed):
    """Play one episode with uniformly random actions and return the total reward."""
    # TODO 1: reset the environment with `seed`, seed its action space too, then step with
    # env.action_space.sample() until the episode is terminated or truncated.
    # Add up the rewards and return the total.
    raise NotImplementedError(
        "TODO 1: write run_episode, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# `obs, info = env.reset(seed=seed)` starts an episode. `env.step(action)` returns
# `obs, reward, terminated, truncated, info`. The episode is over when either flag is true.


# %% role="ex1-sol"
@lab.solution(1)
def run_episode(env, seed):
    """Play one episode with uniformly random actions and return the total reward."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    total, done = 0.0, False
    while not done:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total += float(reward)
        done = terminated or truncated
    return total


# %% role="ex1-run"
import numpy as np

env = gym.make("CartPole-v1")
returns = [run_episode(env, seed=SEED * 1000 + i) for i in range(20)]
print(
    f"Mean return over 20 episodes: {np.mean(returns):.1f} "
    f"(min {min(returns):.0f}, max {max(returns):.0f})"
)

# %% [markdown] role="ex1-explain"
# How close was your guess? Why does random play end so quickly, and why do some episodes
# last much longer than others?

# %% role="ex1-chk"
lab.check(1, m00.check_run_episode, run_episode)

# %% [markdown] role="ex2-head"
# One run is one draw from a random process. To say anything about a method you need several
# seeds, and you show their spread.

# %% [markdown] role="ex2-predict"
# You will run 5 seeds of 20 episodes each. Will the five seeds' mean returns agree to within
# 1 point? Within 5? Write your guess.


# %% role="ex2-stub"
def returns_by_seed(n_seeds, n_episodes, base_seed):
    """Return an array of shape (n_seeds, n_episodes) of random-agent returns.

    Seed s uses episode seeds base_seed + 100 * s + i for i in range(n_episodes).
    """
    # TODO 2: call run_episode for each seed and episode; return a NumPy array.
    raise NotImplementedError(
        "TODO 2: write returns_by_seed, or run lab.use_reference(2) to continue with the solution"
    )


# %% [markdown] role="ex2-hint"
# A nested list comprehension works: `[[run_episode(env, ...) for i in ...] for s in ...]`,
# then `np.array(...)`.


# %% role="ex2-sol"
@lab.solution(2)
def returns_by_seed(n_seeds, n_episodes, base_seed):
    """Return an array of shape (n_seeds, n_episodes) of random-agent returns."""
    return np.array(
        [
            [run_episode(env, base_seed + 100 * s + i) for i in range(n_episodes)]
            for s in range(n_seeds)
        ]
    )


# %% role="ex2-run"
import matplotlib.pyplot as plt

by_seed = returns_by_seed(5, 20, SEED * 1000)
running_mean = np.cumsum(by_seed, axis=1) / np.arange(1, 21)
prl.plot.curves(
    np.arange(1, 21),
    {"random agent": running_mean},
    xlabel="episodes",
    ylabel="running mean return",
    title="Random agent on CartPole-v1, 5 seeds",
    budget="5 seeds x 20 episodes",
)
plt.show()
seed_means = by_seed.mean(axis=1)
print("Per-seed mean returns:", np.round(seed_means, 1))
lab.seeds([SEED * 1000 + 100 * s for s in range(5)])
lab.metric("random_mean_return", round(float(by_seed.mean()), 2))

# %% [markdown] role="ex2-explain"
# How far apart are the per-seed means? Suppose someone reported "my agent scores 25, the
# random agent scores 22" from one seed each. What would you ask them?

# %% role="ex2-chk"
lab.check(2, m00.check_returns_by_seed, returns_by_seed)

# %% [markdown] role="ex3-head"
# Each lab ends by printing a **run record**: what ran, where, with which versions, and which
# checkpoints passed. The site's readiness page is built only from records like this one.

# %% [markdown] role="ex3-predict"
# Which fields of the record would tell a reader whether this ran on Colab, and on a GPU?

# %% role="ex3-run"
rec = lab.build_record()
print(
    "platform:", rec["platform"], "| env:", rec["env"], "| accelerator:", rec["hardware"]["accel"]
)
print("checkpoints so far:", [(c["label"], c["pass"]) for c in rec["checkpoints"]])

# %% [markdown] role="ex3-explain"
# The record never includes your name, paths or keys. On Colab the last cell offers it as a
# download; send it to your instructor if they ask for setup evidence.

# %% role="ex3-chk"
lab.check(3, m00.check_record, rec)
