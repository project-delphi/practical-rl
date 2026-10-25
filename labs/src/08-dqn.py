# ---
# jupyter:
#   prl:
#     module: m08
#     pins: {}
# ---

# %% [markdown]
# In this lab you build DQN from its parts and train it on CartPole: a replay buffer, the
# masked TD target and the training loop with a target network. You find a bug in a
# colleague's buffer, swap in the Double DQN target, and check what your network believes,
# Q(s₀), against what its greedy policy earns, the discounted Monte Carlo return. Last, you
# place your one seed beside recorded bands of the reference DQN and of Stable-Baselines3's
# DQN, run at the same budget.
#
# The briefing's equations are named in brackets, for example (masked target) or (Huber). Its
# table "Equations and where you implement them" gives each name's equation number.

# %%
import dataclasses
import textwrap

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch import nn

import prl.data
from prl.checks import m08  # the checkpoints of this lab
from prl.evaluate import Budget, compare
from prl.plot import NEUTRAL_LIGHT

# Budgets. Lab code reads them from here and never branches on QUICK.
budget = lab.use_settings(
    prl.runtime.settings(
        dict(steps=50_000, log_every=1_000, eval_every=5_000, eval_episodes=10),  # live
        dict(steps=10_000, log_every=500, eval_every=1_000, eval_episodes=10),  # QUICK
        designed="colab-cpu",
    )
)
# Your seat's seed, clear of the briefing's, the figures', the bands' and the checks' seeds
# (0-99), so every seat in the room adds new evidence.
RUN_SEED = (SEED + 1) * 1000
print(f"Budget: {budget.steps:,} environment steps per run; greedy evaluation every")
print(f"{budget.eval_every:,} steps ({budget.eval_episodes} episodes). Your seed: {RUN_SEED}.")

# %% [markdown]
# **Provided code.** The next cell defines the configuration (`CONFIG`) and the plumbing that
# every DQN in the workshop shares. Run it; you need not read it now. In Exercise 3 you call
# `dqn_parts`, `epsilon_at`, `act`, `gather_q`, `huber` and the logger that `dqn_parts` makes;
# the provided `dqn_target` calls your `td_target` from Exercise 2. Each has a short docstring.

# %% include labs/shared/dqn_setup.py


# %%
def make_cartpole():
    """Provided. A fresh CartPole-v1: +1 per step, terminated when the pole falls or the cart
    leaves the track, truncated at 500 steps."""
    return gym.make("CartPole-v1")


fields = [f"{k}={v}" for k, v in dataclasses.asdict(CONFIG).items()]
print(prl.runtime.wrap_items("CONFIG: ", fields, sep=", "))


# %% [markdown] role="ex0-head"
# Welcome back. The briefing built DQN in five steps, and the lab follows them: replay
# (Exercise 1), the masked target (2), the loop with a target network and the Huber loss (3),
# Double DQN and Q against Monte Carlo returns (4 and 5), and equal-budget bands (6). If the
# runtime restarted over lunch, rerun the setup cell at the top and every cell above this one,
# then run the cell below. This notebook depends on no other notebook.

# %% role="ex0-run"
env = make_cartpole()
obs, _ = env.reset(seed=RUN_SEED)
q_net = make_q_net(env.observation_space.shape[0], env.action_space.n, CONFIG)
print("CartPole observation (cart position, cart velocity, pole angle, pole angular velocity):")
print("  ", np.round(obs, 3))
q_untrained = q_net(torch.as_tensor(obs)[None]).detach().numpy()[0]
print(f"Q(s, ·; w) of an untrained network: {np.round(q_untrained, 3).tolist()}")
print(f"epsilon at steps 0, 5,000 and 10,000 of a {budget.steps:,}-step run:", end=" ")
print([round(epsilon_at(t, budget.steps, CONFIG), 3) for t in (0, 5_000, 10_000)])

# %% [markdown] role="ex1-head"
# **A replay buffer, and a colleague's.** DQN stores every transition (s, a, r, s',
# terminated, truncated) and learns from minibatches drawn uniformly from the stored ones
# (replay sampling). The buffer holds at most `capacity` transitions in preallocated arrays,
# one row per slot. When it is full, each new transition overwrites the oldest: the write
# pointer `ptr` wraps around to 0. You write `add` and `sample`; the constructor is given.
# Below your buffer is a colleague's, `ColleagueBuffer`, which runs without an error. The run
# cell checks your buffer, then fills a buffer of 1,000 slots with 2,500 numbered transitions
# and draws from both. Read the histogram of how old the drawn transitions are, answer
# `buffer_diagnosis`, and fix the colleague's `sample`.

# %% [markdown] role="ex1-predict"
# A buffer of 1,000 slots has received 2,500 transitions. What share of a correct buffer's draws
# is older than 500 steps? The colleague's `sample` draws `rng.integers(0, self.ptr)`: what is
# `ptr` now, and what share of its draws is older than 500 steps?


# %% role="ex1-stub"
class ReplayBuffer:
    """A ring buffer of transitions in preallocated NumPy arrays.

    add(...) writes one transition at slot ptr, advances ptr (wrapping to 0 at capacity) and
    counts size up to capacity. sample(batch_size, rng) draws batch_size slots uniformly, with
    replacement, from the size slots that hold a transition, and returns
    to_batch(self, idx): a Batch of tensors (obs, act, rew, next_obs, terminated, truncated).
    """

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0  # the slot the next transition goes to
        self.size = 0  # how many slots hold a transition

    def add(self, obs, act, rew, next_obs, terminated, truncated):
        """Write one transition at slot ptr; then advance ptr and size."""
        # TODO 1 (part 1): six array writes, then ptr and size.
        raise NotImplementedError(
            "TODO 1: write ReplayBuffer.add, or run lab.use_reference(1) to continue"
        )

    def sample(self, batch_size, rng):
        """batch_size slots drawn uniformly from the filled ones, as a Batch (to_batch)."""
        # TODO 1 (part 2): one line of rng, one return.
        raise NotImplementedError(
            "TODO 1: write ReplayBuffer.sample, or run lab.use_reference(1) to continue"
        )


def buffer_diagnosis():
    """Which sentence describes ColleagueBuffer? Return "a", "b", "c" or "d".

    a: it draws every stored transition equally often;
    b: it draws only the transitions written since its pointer last wrapped: the newest;
    c: it never overwrites old transitions, so it draws stale ones;
    d: it draws the oldest transitions more often than the newest.
    """
    # TODO 1 (part 3): return one letter, after reading the histogram below.
    raise NotImplementedError(
        "TODO 1: answer buffer_diagnosis, or run lab.use_reference(1) to continue"
    )


class ColleagueBuffer:
    """A colleague's ring buffer, written apart from yours. Is it right?"""

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, terminated, truncated):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i] = next_obs
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # TODO 1 (part 4): once the histogram shows the problem, fix sample. Then rerun this cell
    # and the run cell.
    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.ptr, size=batch_size)  # the slots written so far
        return to_batch(self, idx)


# %% [markdown] role="ex1-hint"
# In `add`, the next slot is `(self.ptr + 1) % self.capacity`, and `size` stops growing at
# `capacity`. In `sample`, `rng.integers(0, n, size=k)` draws k integers from 0 to n − 1. Which
# n: the capacity, the pointer, or the number of filled slots? Ask the same question of the
# colleague's `sample`, after 2,500 writes into 1,000 slots.


# %% role="ex1-sol"
@lab.solution(1)
class ReplayBuffer:
    """A ring buffer of transitions in preallocated NumPy arrays."""

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0  # the slot the next transition goes to
        self.size = 0  # how many slots hold a transition

    def add(self, obs, act, rew, next_obs, terminated, truncated):
        """Write one transition at slot ptr; advance ptr and size."""
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i] = next_obs
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.ptr = (i + 1) % self.capacity  # the oldest goes next
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Uniform over the filled slots (replay sampling)."""
        idx = rng.integers(0, self.size, size=batch_size)
        return to_batch(self, idx)


@lab.solution(1)
def buffer_diagnosis():
    """b: after the pointer wraps, only the newest ptr slots are drawn."""
    return "b"


@lab.solution(1)
class ColleagueBuffer:
    """The colleague's buffer, fixed: draw from every filled slot."""

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, terminated, truncated):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i] = next_obs
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)  # not ptr
        return to_batch(self, idx)


# %% role="ex1-run"
def play_random(buffer, n, seed):
    """Provided. n steps of uniform random play on CartPole, every transition stored."""
    env_r, rng = make_cartpole(), np.random.default_rng(seed)
    obs, _ = env_r.reset(seed=seed)
    for _ in range(n):
        a = int(rng.integers(2))
        nxt, r, terminated, truncated, _ = env_r.step(a)
        buffer.add(obs, a, r, nxt, terminated, truncated)
        obs = nxt
        if terminated or truncated:
            obs, _ = env_r.reset()


def numbered(buffer_cls, capacity, n):
    """Provided. A buffer that received n transitions; obs[:, 0] holds the write step."""
    b = buffer_cls(capacity, 4)
    for t in range(n):
        b.add([t, 0, 0, 0], 0, 1.0, [t + 1, 0, 0, 0], False, False)
    return b


def draw_ages(b, n_written, rng):
    """Provided. The age (n_written - 1 - write step) of 100,000 draws."""
    steps = np.concatenate([b.sample(1_000, rng).obs[:, 0].numpy() for _ in range(100)])
    return n_written - 1 - steps.astype(int)


rng = np.random.default_rng(RUN_SEED)
buf = ReplayBuffer(1_000, 4)
play_random(buf, 1_500, RUN_SEED)
print(f"Your buffer after 1,500 steps into 1,000 slots: size {buf.size}, ptr {buf.ptr}")
batch = buf.sample(64, rng)
print("batch of 64:", ", ".join(f"{k} {tuple(x.shape)}" for k, x in batch._asdict().items()))
nb = numbered(ReplayBuffer, 1_000, 1_500)
written = np.concatenate([nb.sample(1_000, rng).obs[:, 0].numpy() for _ in range(100)])
counts = np.bincount(written.astype(int) % 1_000, minlength=1_000)  # draws per slot
chi2 = stats.chisquare(counts)
print(f"100,000 draws over 1,000 slots: chi-square {chi2.statistic:.0f}, p = {chi2.pvalue:.2f}")
angle = buf.obs[:, 2]
stream = np.corrcoef(angle[:-1], angle[1:])[0, 1]
rows = buf.sample(1_000, rng).obs[:, 2].numpy()  # 1,000 draws, in the order drawn
mixed = np.corrcoef(rows[:-1], rows[1:])[0, 1]
print("pole angle, one transition against the next:")
print(f"  in the order stored {stream:5.2f}   in the order drawn {mixed:5.2f}")
ages = {}
for name, cls in (("your ReplayBuffer", ReplayBuffer), ("ColleagueBuffer", ColleagueBuffer)):
    b = numbered(cls, 1_000, 2_500)
    ages[name] = draw_ages(b, 2_500, np.random.default_rng(RUN_SEED))
    print(
        f"{name:>17}: size {b.size}, ptr {b.ptr}; draws older than 500 steps: "
        f"{np.mean(ages[name] > 500):.1%}; oldest draw {ages[name].max()} steps"
    )
fig, ax = plt.subplots()
bins = np.arange(0, 1_001, 25)
ax.hist(ages["your ReplayBuffer"], bins=bins, histtype="step", lw=2, label="your ReplayBuffer")
ax.hist(ages["ColleagueBuffer"], bins=bins, histtype="step", lw=2, ls="--", label="ColleagueBuffer")
ax.set(
    xlabel="age of the drawn transition (steps)",
    ylabel="draws (of 100,000)",
    title="2,500 transitions into 1,000 slots: which does each buffer draw?",
)
ax.legend()
plt.show()

# %% [markdown] role="ex1-explain"
# The chi-square p-value is not tiny, so there is no evidence against uniform sampling. A
# gradient step on 64 consecutive transitions would see 64 nearly identical states: what do the
# two correlations say a drawn batch sees instead? Then, which of the two problems replay fixes
# does the colleague's buffer bring back, and why did its bug stay hidden for the first
# `capacity` steps of every run? (Gotcha: "Replay is only for data efficiency.")

# %% role="ex1-chk1"
lab.check(1, m08.check_replay_buffer, ReplayBuffer, label="1.1")

# %% role="ex1-chk2"
lab.check(1, m08.check_buffer_diagnosis, buffer_diagnosis, label="1.2")

# %% role="ex1-chk3"
lab.check(1, m08.check_colleague_buffer, ColleagueBuffer, label="1.3")

# %% [markdown] role="ex2-head"
# **The masked target.** For a batch, the TD target is y = r + γ (1 − mask) q_next, where
# q_next is the bootstrap value at s′, for example max_a′ Q(s′, a′; w⁻), and the mask is 1
# where the bootstrap must stop (masked target). Write `td_target` on tensors. The loss on a TD
# error δ = y − Q(s, a; w) is the Huber loss with threshold ψ, δ²/2 when |δ| ≤ ψ and
# ψ(|δ| − ψ/2) beyond (Huber); `huber(x, threshold)` is provided, and the run plots it.

# %% [markdown] role="ex2-predict"
# γ = 0.99, r = 1 and q_next = 50 for three transitions: one terminated, one truncated, one
# neither. What are the three targets? With ψ = 1, what are the Huber loss and its derivative
# at δ = 3?


# %% role="ex2-stub"
def td_target(batch, q_next, gamma):
    """The masked TD target y, shape (B,), float32.

    batch.rew, batch.terminated and batch.truncated are float32 tensors of shape (B,);
    q_next (B,) is the bootstrap value at s'. Bootstrap unless the transition terminated:
    a truncated transition keeps gamma * q_next.
    """
    # TODO 2: one line.
    raise NotImplementedError("TODO 2: write td_target, or run lab.use_reference(2) to continue")


# %% [markdown] role="ex2-hint"
# `batch.terminated` is 1.0 or 0.0, so `(1.0 - batch.terminated)` switches the bootstrap off
# exactly where the episode really ended. `batch.truncated` is not in the formula.


# %% role="ex2-sol"
@lab.solution(2)
def td_target(batch, q_next, gamma):
    """r + gamma (1 - terminated) q_next: truncation still bootstraps."""
    return batch.rew + gamma * (1.0 - batch.terminated) * q_next


# %% role="ex2-run"
three = Batch(
    obs=torch.zeros(3, 4),
    act=torch.zeros(3, dtype=torch.int64),
    rew=torch.ones(3),
    next_obs=torch.zeros(3, 4),
    terminated=torch.tensor([1.0, 0.0, 0.0]),
    truncated=torch.tensor([0.0, 1.0, 0.0]),
)
y3 = td_target(three, torch.full((3,), 50.0), 0.99)
for label, y in zip(("terminated", "truncated", "neither"), y3.tolist(), strict=True):
    print(f"  {label:>10}: y = {y:.2f}")
x = torch.linspace(-3, 3, 601, requires_grad=True)
curves = {
    "Huber, ψ = 1 (provided)": huber(x, 1.0),
    "squared error δ²/2": 0.5 * x**2,
    "squared error clipped at 1/2": torch.clamp(0.5 * x**2, max=0.5),
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
for (label, loss), ls in zip(curves.items(), ("-", "--", ":"), strict=True):
    (grad,) = torch.autograd.grad(loss.sum(), x)
    ax1.plot(x.detach(), loss.detach(), ls=ls, label=label)
    ax2.plot(x.detach(), grad, ls=ls, label=label)
ax1.set(xlabel="TD error δ", ylabel="loss", ylim=(0, 3), title="Loss")
ax2.set(xlabel="TD error δ", ylabel="derivative", ylim=(-3.2, 3.2), title="Its derivative")
ax2.legend(fontsize=8)
plt.show()
xd = torch.tensor(3.0, requires_grad=True)
(g,) = torch.autograd.grad(huber(xd), xd)
print(f"  δ = 3: Huber {huber(xd).item():.3f}, derivative {g.item():.3f}")

# %% [markdown] role="ex2-explain"
# The truncated transition kept its bootstrap. In CartPole every step pays +1, so what would a
# mask on `terminated or truncated` teach the network about the state where the time limit
# struck? (Gotcha: "done = terminated or truncated.") Then compare the right-hand panel's
# three derivatives beyond |δ| = 1: which equals Mnih et al.'s clipped error term, and which
# loss stops learning from large errors altogether? (Huber)

# %% role="ex2-chk"
lab.check(2, m08.check_td_target, td_target)

# %% [markdown] role="ex3-head"
# **The training loop.** `train_dqn` is DQN (Mnih et al. 2015, Algorithm 1) with `CONFIG`:
# minibatches from your buffer, the Huber loss on targets from the target network (DQN loss),
# and a copy of w into w⁻ every 500 steps. The setup is provided:
# `p = dqn_parts(make_env, config, seed, steps, **log_options)` holds the environment, the
# first observation, both networks, the optimizer, an empty `ReplayBuffer` (yours), a NumPy
# generator and the logger, seeded in a fixed order. You write the loop: acting, storing, the
# gradient step, the copy and three calls to the logger. Module 9 reuses your loop (the
# reference copy), so call the provided pieces by their names.

# %% [markdown] role="ex3-predict"
# The target network is copied every 500 steps and gradient steps start at step 1,004, so the
# copies at steps 1,500, 2,000, ..., 50,000 follow a gradient step: 98 of them in a live run.
# If every fit were exact, what would (sync lag) give for Q at a start state after 98 copies?
# Will your run's max Q at the probe states end within 5 of that value, more than 5 below it,
# more than 5 above it, or near 100?


# %% role="ex3-stub"
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):
    """DQN for `steps` environment steps. Returns (q_net, log).

    target_fn(batch, q_net, target_net, gamma) computes the targets y; None means
    dqn_target. Start with p = dqn_parts(make_env, config, seed, steps, **log_options) and
    obs = p.obs. Then, for t = 1, ..., steps:
    1. eps = epsilon_at(t, steps, config); a = act(p.q_net, obs, eps, p.n_actions, p.rng);
       step p.env; p.buffer.add(obs, a, r, next_obs, terminated, truncated);
       obs = next_obs; reset p.env when either flag is set.
    2. If t > config.learning_starts and t % config.train_freq == 0: sample a batch with
       p.rng; q_values = p.q_net(batch.obs); q_sa = gather_q(q_values, batch.act);
       y = target_fn(...); loss = huber(y - q_sa, config.huber_threshold).mean();
       zero_grad, backward, norm = nn.utils.clip_grad_norm_(..., config.max_grad_norm),
       step; p.log.update(t, batch, q_values, q_sa, y, loss, norm).
    3. If config.target_sync and t % config.target_sync == 0: copy p.q_net's weights into
       p.target_net; p.log.sync(t).
    4. Last: p.log.step(t, eps, r, terminated, truncated).
    After the loop: p.env.close(); return p.q_net, p.log.result().
    """
    # TODO 3: about 25 lines.
    raise NotImplementedError("TODO 3: write train_dqn, or run lab.use_reference(3) to continue")


# %% [markdown] role="ex3-hint"
# Copy the weights with `p.target_net.load_state_dict(p.q_net.state_dict())`. `huber` returns
# one loss per transition, so take `.mean()` before `backward()`. Set `obs = next_obs` before
# a reset: the buffer stores the state the step returned, and the next action starts from the
# reset state. `p.log.step` comes last in the loop, after the gradient step and the copy.


# %% role="ex3-sol"
@lab.solution(3)
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):
    """DQN (Mnih et al. 2015, Algorithm 1). Returns (q_net, log)."""
    target_fn = target_fn or dqn_target
    p = dqn_parts(make_env, config, seed, steps, **log_options)
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)
            y = target_fn(batch, p.q_net, p.target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # (DQN loss)
            p.opt.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.opt.step()
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        if config.target_sync and t % config.target_sync == 0:
            p.target_net.load_state_dict(p.q_net.state_dict())  # w- <- w
            p.log.sync(t)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


# %% role="ex3-run"
def syncs_by(t, config):
    """Provided. How many target copies follow at least one gradient step by step t (exact)."""
    if not config.target_sync:
        return 0
    f, k = config.train_freq, config.target_sync
    first = (config.learning_starts // f + 1) * f  # the first gradient step
    return max(0, t // k - (first - 1) // k)


def sync_lag(t, config):
    """Provided. (sync lag): (1 - gamma^k) / (1 - gamma) after k = syncs_by(t) copies."""
    k = syncs_by(t, config)
    return (1 - config.gamma**k) / (1 - config.gamma)


def steps_in_k(ax):
    """Provided. Environment-step ticks as 0, 10k, 20k, ..."""
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}k" if v else "0")


lab.seeds([RUN_SEED])
SNAPSHOTS = (CONFIG.learning_starts + 1, budget.steps // 2, budget.steps)
q_dqn, log_dqn = train_dqn(
    make_cartpole,
    CONFIG,
    RUN_SEED,
    budget.steps,
    log_every=budget.log_every,
    eval_every=budget.eval_every,
    eval_episodes=budget.eval_episodes,
    snapshot_steps=SNAPSHOTS,
)
k = syncs_by(budget.steps, CONFIG)
q_end = log_dqn["q_probe"][-1].max(axis=1).mean()
print(f"Seed {RUN_SEED}: {log_dqn['seconds']:.0f} s for {budget.steps:,} steps.")
print(
    f"greedy return at the end (mean of {budget.eval_episodes} episodes): "
    f"{log_dqn['eval_return'][-1]:.1f}"
)
print(
    f"max Q at the probe states at the end: {q_end:.2f}; "
    f"(sync lag) after {k} copies: {sync_lag(budget.steps, CONFIG):.2f}"
)
print("Single seed: a fact to put beside the band and the room's seeds, not a ranking.")
lab.metric("dqn_return", round(float(log_dqn["eval_return"][-1]), 1))  # room code
lab.metric("train_seconds", round(float(log_dqn["seconds"]), 1))  # room code
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
band = prl.data.band("m08_dqn_return", budget)
prl.plot.with_recorded(
    band,
    log_dqn["eval_return"],
    x=log_dqn["eval_step"],
    your_budget=budget,
    ax=ax1,
    label=f"your seed ({RUN_SEED})",
    xlabel="environment steps",
    ylabel="greedy return (mean of 10 episodes)",
    title="Return: your seed and the band",
)
ax1.legend(fontsize=8)
steps_axis = log_dqn["step"]
ax2.plot(steps_axis, log_dqn["q_probe"].max(axis=2).mean(axis=1), label="max Q, probe states")
ax2.plot(
    steps_axis,
    [sync_lag(t, CONFIG) for t in steps_axis],
    ls="--",
    color=NEUTRAL_LIGHT,
    label="(sync lag)",
)
ax2.set(xlabel="environment steps", ylabel="value", title="Q at the probe states")
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    steps_in_k(ax)
plt.tight_layout()
plt.show()
seeds = f"seeds {band['seeds'][0]}-{band['seeds'][-1]}"
print(
    textwrap.fill(
        f"The band: the reference DQN (this lab's solution), {seeds} at your budget "
        f"({band['record']}).",
        98,
    )
)

# %% [markdown] role="ex3-explain"
# Compare the printed max Q at the probe states with (sync lag) for your number of copies:
# which of your Predict's four answers was it? The return says the policy got better. What
# does it not say about Q? (Gotcha: "My returns went up, so it works.")

# %% role="ex3-chk1"
lab.check(3, m08.check_train_dqn_masks, train_dqn, CONFIG, label="3.1")

# %% role="ex3-chk2"
# Trains your loop again from its own seed for 20,000 steps, with a greedy evaluation every
# 1,000 steps; it passes if the best evaluation beats the first (before any gradient step)
# by at least 91.0, a threshold set by the threshold protocol over 200 seeds.
lab.check(3, m08.check_train_dqn, train_dqn, make_cartpole, CONFIG, label="3.2")

# %% [markdown]
# **Diagnostics for this lab.** Read the logger's arrays, not only the return. Healthy here:
# the loss and the mean |δ| stay of order 1; the steps since the last copy read 0 at every log
# point, because each log point (every 1,000 steps; 500 at QUICK) lands on a copy (every 500);
# the gradient norm sits mostly under the clip. Broken: Q above 100 (a missing mask, or
# divergence), steps since the last copy that climb with the step count (no copy), a mean |δ|
# that grows by orders of magnitude (steps too large). Module 9 reads exactly these arrays.

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True)
axes[0].plot(steps_axis, log_dqn["loss"], label="Huber loss")
axes[0].plot(steps_axis, log_dqn["td_abs_mean"], ls="--", label="mean |δ|")
axes[0].set(title="Loss and TD error", xlabel="environment steps")
axes[0].legend(fontsize=8)
axes[1].plot(steps_axis, log_dqn["steps_since_sync"])
axes[1].set(
    title="Steps since the last copy",
    xlabel="environment steps",
    ylim=(-25, CONFIG.target_sync + 25),  # on the scale of the copy interval
)
axes[2].plot(steps_axis, log_dqn["grad_norm"])
axes[2].axhline(CONFIG.max_grad_norm, ls=":", color=NEUTRAL_LIGHT)
axes[2].text(steps_axis[0], CONFIG.max_grad_norm, " clip", va="top", color=NEUTRAL_LIGHT)
axes[2].set(title="Gradient norm before clipping", xlabel="environment steps")
for ax in axes:
    steps_in_k(ax)
plt.tight_layout()
plt.show()
snap = len(log_dqn["snap_step"]) - 1
same = np.array_equal(
    log_dqn["snap_gathered"][snap],
    log_dqn["snap_q"][snap][np.arange(CONFIG.batch_size), log_dqn["snap_actions"][snap]],
)
print(
    f"snapshot at step {log_dqn['snap_step'][snap]}: the gathered values equal "
    f"Q[row, action] on every row: {same}"
)


# %% [markdown] role="ex4-head"
# **The Double DQN target.** DQN's target uses the target network both to pick the best next
# action and to value it. Double DQN picks with the online network and values with the
# target network: q_next = Q(s′, argmax_a′ Q(s′, a′; w); w⁻) (Double DQN target). It has the
# same signature as `dqn_target`, so your `train_dqn` runs it with `target_fn=`.

# %% [markdown] role="ex4-predict"
# One transition: r = 1, γ = 0.9, not terminated. The online network gives Q(s′, ·) = (2, 5);
# the target network gives (4, 3). What is DQN's target, and what is Double DQN's?


# %% role="ex4-stub"
def double_dqn_target(batch, q_net, target_net, gamma):
    """Double DQN's targets, shape (B,), with no gradient attached.

    q_net and target_net map next observations (B, obs_dim) to values (B, |A|). The online
    network selects a' = argmax Q(s', .; w); the target network evaluates Q(s', a'; w-).
    Finish with your td_target, so the terminal mask stays in one place.
    """
    # TODO 4: three lines inside torch.no_grad().
    raise NotImplementedError(
        "TODO 4: write double_dqn_target, or run lab.use_reference(4) to continue"
    )


# %% [markdown] role="ex4-hint"
# Start from `dqn_target` in the provided cell. Replace its `max` with two steps: the argmax
# of `q_net(batch.next_obs)` over dimension 1, then `gather_q` of
# `target_net(batch.next_obs)` at those actions.


# %% role="ex4-sol"
@lab.solution(4)
def double_dqn_target(batch, q_net, target_net, gamma):
    """Select with the online network, evaluate with the target."""
    with torch.no_grad():
        a_next = q_net(batch.next_obs).argmax(dim=1)  # w selects
        q_next = gather_q(target_net(batch.next_obs), a_next)  # w- values
        return td_target(batch, q_next, gamma)  # (Double DQN target)


# %% role="ex4-run"
one = Batch(
    obs=torch.zeros(1, 1),
    act=torch.zeros(1, dtype=torch.int64),
    rew=torch.ones(1),
    next_obs=torch.zeros(1, 1),
    terminated=torch.zeros(1),
    truncated=torch.zeros(1),
)


def online(next_obs):
    """Provided. A stand-in online network: Q(s', .) = (2, 5) for every row."""
    return torch.tensor([[2.0, 5.0]]).expand(len(next_obs), 2)


def target(next_obs):
    """Provided. A stand-in target network: Q(s', .) = (4, 3) for every row."""
    return torch.tensor([[4.0, 3.0]]).expand(len(next_obs), 2)


print(
    f"DQN target {dqn_target(one, online, target, 0.9).item():.2f}; "
    f"Double DQN target {double_dqn_target(one, online, target, 0.9).item():.2f}"
)
q_ddqn, log_ddqn = train_dqn(
    make_cartpole,
    CONFIG,
    RUN_SEED,
    budget.steps,
    target_fn=double_dqn_target,
    log_every=budget.log_every,
    eval_every=budget.eval_every,
    eval_episodes=budget.eval_episodes,
)
print(f"Double DQN, seed {RUN_SEED}, trained in {log_ddqn['seconds']:.0f} s. Greedy returns:")
print(
    f"  Double DQN {log_ddqn['eval_return'][-1]:.1f}, DQN {log_dqn['eval_return'][-1]:.1f} "
    "(single seed each, not a ranking)"
)
lab.metric("ddqn_return", round(float(log_ddqn["eval_return"][-1]), 1))  # room code

# %% [markdown] role="ex4-explain"
# On the stand-in networks the two targets differ because the networks disagree about the
# best next action. When do the two targets agree exactly? Why does picking with one network
# and valuing with another make it less likely that a lucky overestimate is both picked and
# used? (Double DQN target)

# %% role="ex4-chk"
lab.check(4, m08.check_double_dqn_target, double_dqn_target)

# %% [markdown] role="ex5-head"
# **Q(s₀) against the Monte Carlo return.** What does your network believe a start state
# is worth, and what does its greedy policy earn from there? `evaluate_greedy` runs the greedy
# policy from 10 fixed start states and returns each episode's rewards and
# Q(s₀) = max_a Q(s₀, a; w). You write `mc_return`: the discounted return from every step to the
# end of an episode, computed backward, G_t = r_t + γ G_{t+1} (MC return). G₀ is the Monte Carlo
# estimate of the greedy policy's value at s₀.

# %% [markdown] role="ex5-predict"
# At the end of your run, is Q(s₀) above or below G₀, the discounted MC return its greedy
# policy earns? And at the same seed and budget, is Double DQN's Q(s₀) higher or lower than
# DQN's?


# %% role="ex5-stub"
def mc_return(rewards, gamma):
    """Discounted returns-to-go of one episode: G_t = r_t + gamma G_{t+1}, with G = 0 after
    the last reward. rewards is a 1-D array of length T; returns a float array (T,)."""
    # TODO 5: one backward loop.
    raise NotImplementedError("TODO 5: write mc_return, or run lab.use_reference(5) to continue")


# %% [markdown] role="ex5-hint"
# Loop `t` from `len(rewards) - 1` down to 0 with a running total: `total = rewards[t] +
# gamma * total`, then store it at index t. The last entry is the last reward alone.


# %% role="ex5-sol"
@lab.solution(5)
def mc_return(rewards, gamma):
    """Returns-to-go, computed backward (MC return)."""
    g = np.zeros(len(rewards))
    total = 0.0
    for t in reversed(range(len(rewards))):
        total = rewards[t] + gamma * total
        g[t] = total
    return g


# %% role="ex5-run"
lag = sync_lag(budget.steps, CONFIG)
rows = {}
for name, net in (("DQN", q_dqn), ("Double DQN", q_ddqn)):
    ev = evaluate_greedy(net, make_cartpole, budget.eval_episodes, CONFIG.gamma)
    g0 = np.array([mc_return(r, CONFIG.gamma)[0] for r in ev["rewards"]])
    rows[name] = (ev["q0"].mean(), g0.mean(), ev["returns"].mean())
print(f"After {budget.steps:,} steps, seed {RUN_SEED}, {budget.eval_episodes} greedy episodes;")
print(f"(sync lag) after {syncs_by(budget.steps, CONFIG)} copies: {lag:.2f}")
print("                Q(s₀)       G₀    Q(s₀) − G₀   Q(s₀) − (sync lag)   greedy return")
for name, (q0, g0, ret) in rows.items():
    print(
        f"  {name:>10}   {q0:7.2f}  {g0:7.2f}   {q0 - g0:+11.2f}   {q0 - lag:+18.2f}   {ret:13.1f}"
    )
d = rows["Double DQN"][0] - rows["DQN"][0]
print(f"Double DQN minus DQN at seed {RUN_SEED}: Q(s₀) {d:+.2f}")
lab.metric("dqn_q0", round(float(rows["DQN"][0]), 2))  # room code
lab.metric("ddqn_q0", round(float(rows["Double DQN"][0]), 2))  # room code
lab.metric("dqn_q0_gap", round(float(rows["DQN"][0] - rows["DQN"][1]), 2))  # room code
STYLE = {  # as the briefing's figure: Q(s0) solid with circles, G0 dotted with squares
    "q0": ("eval_q0", "-", "o", r"$Q(s_0)$"),
    "mc": ("eval_disc_return", ":", "s", r"$G_0$"),
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
grid = np.arange(0, budget.steps + 1, budget.steps // 100)
for ax, (name, log) in zip(axes, (("dqn", log_dqn), ("ddqn", log_ddqn)), strict=True):
    for i, (what, (key, ls, marker, sym)) in enumerate(STYLE.items()):
        b = prl.data.band(f"m08_{name}_{what}", budget)
        color = prl.plot.SERIES[i]
        ax.fill_between(
            b["x"],
            b["runs"].min(axis=0),
            b["runs"].max(axis=0),
            color=color,
            alpha=0.2,
            lw=0,
            label=f"{sym}, recorded 5 seeds (min to max)",
        )
        ax.plot(
            log["eval_step"],
            log[key],
            ls=ls,
            marker=marker,
            ms=3,
            color=color,
            label=f"{sym}, your seed",
        )
    ax.plot(
        grid, [sync_lag(t, CONFIG) for t in grid], color=NEUTRAL_LIGHT, ls="--", label="(sync lag)"
    )
    ax.set(xlabel="environment steps", title="DQN" if name == "dqn" else "Double DQN")
    steps_in_k(ax)
axes[0].set(ylabel="value at the start states")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1))
plt.show()
bands = [prl.data.band(f"m08_{n}_q0", budget) for n in ("dqn", "ddqn")]
seeds = f"seeds {bands[0]['seeds'][0]}-{bands[0]['seeds'][-1]}"
records = " and ".join(b["record"] for b in bands)
print(
    textwrap.fill(
        f"The bands: the reference DQN and Double DQN (this lab's solutions), {seeds} at your "
        f"budget ({records}).",
        98,
    )
)


# %% [markdown] role="ex5-explain"
# Is Q(s₀) on the side of G₀ you predicted? After k copies even a perfect fit sits near
# (sync lag), whatever the policy earns: which force dominates at this budget, and what would a
# much longer run show? Then compare Double DQN's Q(s₀) with DQN's at your seed, and each with
# (sync lag): where does the max over noisy estimates (Jensen) show? (sync lag)

# %% role="ex5-chk"
lab.check(5, m08.check_mc_return, mc_return)

# %% [markdown] role="ex6-head"
# **SB3 at the same budget.** Is your implementation right? A matched library run is
# one way to find out. The workshop's records hold Stable-Baselines3 2.9's DQN, with every one
# of the lab's knobs that it has matched, and the reference DQN (this lab's solution code),
# 5 seeds each at your budget. The cell below prints each side's interquartile mean (IQM) of
# the final greedy return with its 95% bootstrap interval, the budget, and what was not
# matched, then draws both bands with your seed. Apply the workshop's rule (overlapping
# intervals are a tie) and answer `sb3_call`.

# %% [markdown] role="ex6-predict"
# Both bands run the same configuration on 5 seeds each, one through the lab's code and one
# through SB3's. Do you expect the rule to call a tie or a winner at this budget? And if it
# calls a tie, does that prove the two implementations are the same?


# %% role="ex6-stub"
def sb3_call():
    """The workshop rule's verdict on the reference DQN against SB3's DQN at this budget:
    return "tie", "reference" (the reference is better) or "sb3"."""
    # TODO 6: one word, after reading the printout below.
    raise NotImplementedError("TODO 6: answer sb3_call, or run lab.use_reference(6) to continue")


# %% [markdown] role="ex6-hint"
# The printout gives each side's IQM and its 95% interval. The rule calls a tie when the two
# intervals overlap; otherwise the side with the higher IQM is better at this budget. Your
# own seed does not enter the call.


# %% role="ex6-sol"
@lab.solution(6)
def sb3_call():
    """The rule's verdict on the printed intervals (verdict.winner)."""
    return {"A": "reference", "B": "sb3"}.get(verdict.winner, verdict.winner)


# %% role="ex6-run"
ref = prl.data.band("m08_dqn_return", budget)
sb3 = prl.data.band("m08_sb3_return", budget)
yours = float(log_dqn["eval_return"][-1])
hw = ref["hardware"]  # where the bands were recorded: a laptop is not Colab
spent = Budget(
    steps=budget.steps,
    seeds=len(ref["seeds"]),
    wall_clock_s=sum(sum(b["seconds"]["per_seed"]) for b in (ref, sb3)),  # all 10 runs
    hardware=f"{hw['cpu']} ({hw['env']}, {hw['torch_threads']} torch threads)",
)
verdict = compare(
    ref["runs"][:, -1], sb3["runs"][:, -1], names=("reference DQN", "SB3 DQN"), budget=spent
)
for name, (center, lo, hi) in (
    ("reference DQN", verdict.summary_a),
    ("SB3 DQN", verdict.summary_b),
):
    print(f"{name:>13}: IQM {center:6.1f}, 95% bootstrap interval [{lo:.1f}, {hi:.1f}]")
print(textwrap.fill(spent.sentence(), 98))
print("Not matched (from the SB3 record's notes):")
for item in (
    "SB3 acts uniformly at random until learning starts; the lab acts epsilon-greedily",
    "SB3 copies the target before that step's gradient step; the lab copies after it",
    "SB3 updates epsilon after each step; the lab computes it before acting",
    "the random draws after the initial weights (the initial weights are the same)",
    "SB3 wraps the environment in a Monitor and a DummyVecEnv",
):
    print(f"  - {item}")
for name, b in (("reference DQN", ref), ("SB3 DQN", sb3)):
    lo, hi = b["runs"][:, -1].min(), b["runs"][:, -1].max()
    where = "inside" if lo <= yours <= hi else ("above" if yours > hi else "below")
    print(f"your seed {yours:.1f} is {where} the {name} range [{lo:.1f}, {hi:.1f}]")
SIDES = (  # as the briefing's figure: reference blue circles, SB3 green triangles
    (ref, "reference DQN", prl.plot.SERIES[0], "-", "o"),
    (sb3, "SB3 2.9 DQN", prl.plot.SERIES[2], "--", "^"),
)
YOU = prl.plot.SERIES[1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), width_ratios=(3, 1), sharey=True)
for b, label, color, ls, _ in SIDES:
    for i, run in enumerate(b["runs"]):
        lab_i = f"{label} (5 seeds)" if i == 0 else None
        ax1.plot(b["x"], run, ls=ls, lw=0.8, color=color, label=lab_i)
ax1.plot(
    log_dqn["eval_step"],
    log_dqn["eval_return"],
    lw=2.2,
    color=YOU,
    label=f"your seed ({RUN_SEED})",
)
ax1.set(xlabel="environment steps", ylabel="greedy return", title="Every seed, same budget")
steps_in_k(ax1)
ax1.legend(fontsize=8)
for i, (b, _, color, _, marker) in enumerate(SIDES):
    ax2.plot(np.full(len(b["seeds"]), i), b["runs"][:, -1], marker, color=color, mfc="none")
ax2.axhline(yours, color=YOU, lw=2.2)  # your seed, against both ranges
ax2.text(0.5, yours, "your seed", ha="center", va="bottom", fontsize=8)
ax2.set(xticks=[0, 1], xticklabels=["reference", "SB3"], xlim=(-0.6, 1.6), title="At the budget")
plt.tight_layout()
plt.show()


# %% [markdown] role="ex6-explain"
# Where your seed landed is a fact about one draw, not a grade. What does the
# reference-against-SB3 call say about your code, and what would it take to say more?
# (Gotcha: "My seed landed above SB3's band, so my DQN is better.")

# %% role="ex6-chk"
lab.check(6, m08.check_sb3_call, sb3_call)

# %%
print(textwrap.fill(verdict.sentence, 98))  # the rule's own sentence, after your answer

# %% [markdown]
# **Your room code.** The finish cell below, `lab.finish()`, ends with one line that starts
# with `ROOM m08`. It holds your seat, the checkpoints you passed, whether QUICK budgets ran, and
# the numbers the debrief pools: your DQN's and Double DQN's greedy return at the budget
# (`dqn_return`, `ddqn_return`), their Q(s₀) at the same seed (`dqn_q0`, `ddqn_q0`), your DQN's
# Q(s₀) − G₀ (`dqn_q0_gap`) and your training time (`train_seconds`). Paste it where your
# instructor says.

# %% [markdown] role="stretch"
# **LunarLander-v3.** Your `train_dqn` assumes nothing about CartPole: it reads the
# observation size and the number of actions from the environment. LunarLander has 8
# observations, 4 actions, rewards of order ±100 when it lands or crashes, and needs the Box2D
# physics engine and pygame, which are not installed with the lab. On Colab (pygame is
# preinstalled), run `!pip install -q box2d==2.3.10` in a new cell, then this cell; locally,
# `uv pip install box2d==2.3.10 pygame`. The run below is untuned (the lab's configuration
# with a larger buffer) and short for this task: read where its greedy return goes, not
# whether it lands. Which part of the configuration would you change first, and why?

# %% role="stretch-run"
stretch = prl.runtime.settings(
    dict(lunar_steps=100_000), dict(lunar_steps=5_000), designed="colab-cpu"
)
try:
    import Box2D  # noqa: F401
    import pygame  # noqa: F401
except ImportError:
    print("Box2D or pygame is not installed, so the stretch is skipped. See the cell above.")
else:
    lunar_cfg = dataclasses.replace(CONFIG, buffer_size=50_000)
    q_lunar, log_lunar = train_dqn(
        lambda: gym.make("LunarLander-v3"),
        lunar_cfg,
        RUN_SEED,
        stretch.lunar_steps,
        log_every=stretch.lunar_steps // 20,
        eval_every=stretch.lunar_steps // 5,
        eval_episodes=5,
    )
    returns = [round(float(v), 1) for v in log_lunar["eval_return"]]
    print("greedy return at each evaluation:", returns)
    print(
        f"{log_lunar['seconds']:.0f} s. A run that lands reliably typically needs far more "
        "steps and tuning."
    )
