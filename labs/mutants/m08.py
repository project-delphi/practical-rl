"""Plausible wrong versions of Module 8's functions (academic director's draft, completed by
the technical expert).

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires every
exact checkpoint of its exercise that calls the function to reject it; stochastic checkpoints
are skipped there, because scripts/threshold_protocol.py has already required every mutant of
a stochastic checkpoint's functions to fail its threshold at every seed, with a stated gap.
Each mutant is a mistake a participant could plausibly make; `why` says which.

Checkpoint 3.2 (how far train_dqn raises CartPole's greedy return above its untrained
network's) is the only stochastic one. Its protocol mutants are the train_dqn mutants below
without exact_only: `target-is-online`, `zero-grad-late`, `no-step` and `random`.
Checkpoint 3.1 is exact and also calls train_dqn, so verify mode needs it to reject every
train_dqn mutant. It does: its chain runs check the log's keys, the snapshots' steps, a target
copy every target_sync steps, the terminal mask and the bootstrap on cut episodes, that Q at
the probe states moves (`no-step`), and, with config.lr = 0, that the gradient norm does not
grow (which rejects `no-zero-grad`) and is not zero (`zero-grad-late`).

exact_only=True marks the mutants that only 3.1 can see: `done-in-buffer` (CartPole rarely
reaches the time limit at the check's budget); `no-zero-grad`, which still learns CartPole
(greedy returns 73-272 at 20,000 steps on seeds 0-3 in a scratch scan); and `no-target-copy`,
whose targets from the initial network sometimes make a fair policy by chance (up to 175
above its first evaluation over seeds 0-99 in a scratch scan), so no return threshold rejects
them all. Exercise 6's mutants depend on what the recorded bands say (see their note).
"""

import numpy as np
import torch
from torch import nn

from prl.lab import mutant


# ------------------------------------------------------------------ 1: ReplayBuffer
@mutant(
    ex=1,
    replaces="ReplayBuffer",
    id="sample-capacity",
    why="draws from all capacity slots, including empty ones before the buffer fills",
)
class ReplayBuffer:
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
        return to_batch(self, rng.integers(0, self.capacity, size=batch_size))  # noqa: F821


@mutant(
    ex=1,
    replaces="ReplayBuffer",
    id="no-wrap",
    why="stops at the last slot when full and keeps overwriting it, so old data never leaves",
)
class ReplayBuffer:  # noqa: F811
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
        self.ptr = min(i + 1, self.capacity - 1)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        return to_batch(self, rng.integers(0, self.size, size=batch_size))  # noqa: F821


@mutant(
    ex=1,
    replaces="ReplayBuffer",
    id="contiguous",
    why="takes batch_size consecutive slots from a random start: correlated batches",
)
class ReplayBuffer:  # noqa: F811
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
        start = int(rng.integers(0, self.size))
        return to_batch(self, (start + np.arange(batch_size)) % self.size)  # noqa: F821


@mutant(
    ex=1,
    replaces="ReplayBuffer",
    id="size-minus-one",
    why="rng.integers(0, self.size - 1): the high end is already exclusive, so one filled slot "
    "is never drawn",
)
class ReplayBuffer:  # noqa: F811
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
        return to_batch(self, rng.integers(0, self.size - 1, size=batch_size))  # noqa: F821


# ------------------------------------------------------------------ 1: the colleague's buffer
@mutant(ex=1, replaces="buffer_diagnosis", id="answer-a", why="reads the histograms as uniform")
def buffer_diagnosis():
    return "a"


@mutant(ex=1, replaces="buffer_diagnosis", id="answer-c", why="blames overwriting, not sampling")
def buffer_diagnosis():  # noqa: F811
    return "c"


@mutant(
    ex=1,
    replaces="ColleagueBuffer",
    id="capacity-not-size",
    why="replaces ptr with capacity: right once full, draws empty slots before",
)
class ColleagueBuffer:
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
        return to_batch(self, rng.integers(0, self.capacity, size=batch_size))  # noqa: F821


# ------------------------------------------------------------------ 2: td_target
@mutant(ex=2, replaces="td_target", id="done-mask", why="masks on terminated or truncated")
def td_target(batch, q_next, gamma):
    done = torch.clamp(batch.terminated + batch.truncated, max=1.0)
    return batch.rew + gamma * (1.0 - done) * q_next


@mutant(ex=2, replaces="td_target", id="no-mask", why="always bootstraps (Module 9's first bug)")
def td_target(batch, q_next, gamma):  # noqa: F811
    return batch.rew + gamma * q_next


@mutant(ex=2, replaces="td_target", id="mask-inverted", why="multiplies by terminated, not 1 - it")
def td_target(batch, q_next, gamma):  # noqa: F811
    return batch.rew + gamma * batch.terminated * q_next


@mutant(ex=2, replaces="td_target", id="discount-reward", why="discounts the reward too")
def td_target(batch, q_next, gamma):  # noqa: F811
    return gamma * (batch.rew + (1.0 - batch.terminated) * q_next)


# ------------------------------------------------------------------ 3: train_dqn
# The solution's body (on the provided dqn_parts) with one change each. Shared pieces
# (dqn_parts, act, gather_q, huber, dqn_target, ...) come from the notebook.
@mutant(
    ex=3,
    replaces="train_dqn",
    id="done-in-buffer",
    why="stores terminated or truncated as terminated, so truncation stops bootstrapping",
    exact_only=True,
)
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):
    target_fn = target_fn or dqn_target  # noqa: F821
    p = dqn_parts(make_env, config, seed, steps, **log_options)  # noqa: F821
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)  # noqa: F821
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)  # noqa: F821
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated or truncated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)  # noqa: F821
            y = target_fn(batch, p.q_net, p.target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # noqa: F821
            p.opt.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.opt.step()
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        if config.target_sync and t % config.target_sync == 0:
            p.target_net.load_state_dict(p.q_net.state_dict())
            p.log.sync(t)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


@mutant(
    ex=3,
    replaces="train_dqn",
    id="no-zero-grad",
    why="never calls opt.zero_grad(): every step's gradient adds to all the earlier ones",
    exact_only=True,
)
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):  # noqa: F811
    target_fn = target_fn or dqn_target  # noqa: F821
    p = dqn_parts(make_env, config, seed, steps, **log_options)  # noqa: F821
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)  # noqa: F821
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)  # noqa: F821
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)  # noqa: F821
            y = target_fn(batch, p.q_net, p.target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # noqa: F821
            loss.backward()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.opt.step()
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        if config.target_sync and t % config.target_sync == 0:
            p.target_net.load_state_dict(p.q_net.state_dict())
            p.log.sync(t)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


@mutant(
    ex=3,
    replaces="train_dqn",
    id="zero-grad-late",
    why="calls opt.zero_grad() after loss.backward(), so every step applies a zero gradient",
)
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):  # noqa: F811
    target_fn = target_fn or dqn_target  # noqa: F821
    p = dqn_parts(make_env, config, seed, steps, **log_options)  # noqa: F821
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)  # noqa: F821
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)  # noqa: F821
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)  # noqa: F821
            y = target_fn(batch, p.q_net, p.target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # noqa: F821
            loss.backward()
            p.opt.zero_grad()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.opt.step()
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        if config.target_sync and t % config.target_sync == 0:
            p.target_net.load_state_dict(p.q_net.state_dict())
            p.log.sync(t)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


@mutant(
    ex=3,
    replaces="train_dqn",
    id="no-step",
    why="computes and clips the gradient but never calls opt.step(): the network never learns",
)
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):  # noqa: F811
    target_fn = target_fn or dqn_target  # noqa: F821
    p = dqn_parts(make_env, config, seed, steps, **log_options)  # noqa: F821
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)  # noqa: F821
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)  # noqa: F821
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)  # noqa: F821
            y = target_fn(batch, p.q_net, p.target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # noqa: F821
            p.opt.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        if config.target_sync and t % config.target_sync == 0:
            p.target_net.load_state_dict(p.q_net.state_dict())
            p.log.sync(t)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


@mutant(
    ex=3,
    replaces="train_dqn",
    id="target-is-online",
    why="computes targets with the online network: no target network (expected to diverge)",
)
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):  # noqa: F811
    target_fn = target_fn or dqn_target  # noqa: F821
    p = dqn_parts(make_env, config, seed, steps, **log_options)  # noqa: F821
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)  # noqa: F821
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)  # noqa: F821
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)  # noqa: F821
            y = target_fn(batch, p.q_net, p.q_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # noqa: F821
            p.opt.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.opt.step()
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


@mutant(
    ex=3,
    replaces="train_dqn",
    id="no-target-copy",
    why="never copies w into w-: the targets come from the initial network (Module 9's bug)",
    exact_only=True,
)
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):  # noqa: F811
    target_fn = target_fn or dqn_target  # noqa: F821
    p = dqn_parts(make_env, config, seed, steps, **log_options)  # noqa: F821
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)  # noqa: F821
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)  # noqa: F821
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)  # noqa: F821
            y = target_fn(batch, p.q_net, p.target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # noqa: F821
            p.opt.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.opt.step()
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


@mutant(ex=3, replaces="train_dqn", id="random", why="never learns: uniform random play")
def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):  # noqa: F811
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


# ------------------------------------------------------------------ 4: double_dqn_target
@mutant(ex=4, replaces="double_dqn_target", id="target-selects", why="DQN's target: w- picks too")
def double_dqn_target(batch, q_net, target_net, gamma):
    with torch.no_grad():
        q_t = target_net(batch.next_obs)
        return td_target(batch, gather_q(q_t, q_t.argmax(dim=1)), gamma)  # noqa: F821


@mutant(
    ex=4,
    replaces="double_dqn_target",
    id="online-evaluates",
    why="the online network both selects and evaluates",
)
def double_dqn_target(batch, q_net, target_net, gamma):  # noqa: F811
    with torch.no_grad():
        q_o = q_net(batch.next_obs)
        return td_target(batch, gather_q(q_o, q_o.argmax(dim=1)), gamma)  # noqa: F821


@mutant(
    ex=4,
    replaces="double_dqn_target",
    id="with-gradient",
    why="no torch.no_grad(): the targets carry the online network's gradient",
)
def double_dqn_target(batch, q_net, target_net, gamma):  # noqa: F811
    a_next = q_net(batch.next_obs).argmax(dim=1)
    q_next = gather_q(target_net(batch.next_obs), a_next) + 0 * q_net(batch.next_obs)[:, 0]  # noqa: F821
    return td_target(batch, q_next, gamma)  # noqa: F821


@mutant(ex=4, replaces="double_dqn_target", id="selects-at-s", why="takes the argmax at s, not s'")
def double_dqn_target(batch, q_net, target_net, gamma):  # noqa: F811
    with torch.no_grad():
        a_next = q_net(batch.obs).argmax(dim=1)
        q_next = gather_q(target_net(batch.next_obs), a_next)  # noqa: F821
        return td_target(batch, q_next, gamma)  # noqa: F821


# ------------------------------------------------------------------ 5: mc_return
@mutant(ex=5, replaces="mc_return", id="undiscounted", why="forgets gamma")
def mc_return(rewards, gamma):
    return np.cumsum(np.asarray(rewards, dtype=float)[::-1])[::-1]


@mutant(ex=5, replaces="mc_return", id="off-by-one", why="discounts the first reward too")
def mc_return(rewards, gamma):  # noqa: F811
    g, total = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        total = gamma * (rewards[t] + total)
        g[t] = total
    return g


@mutant(ex=5, replaces="mc_return", id="forward", why="accumulates from the start, not the end")
def mc_return(rewards, gamma):  # noqa: F811
    g, total = np.zeros(len(rewards)), 0.0
    for t in range(len(rewards)):
        total = rewards[t] + gamma * total
        g[t] = total
    return g


@mutant(ex=5, replaces="mc_return", id="scalar", why="returns G_0 alone, not every G_t")
def mc_return(rewards, gamma):  # noqa: F811
    return float(np.sum(gamma ** np.arange(len(rewards)) * np.asarray(rewards)))


# ------------------------------------------------------------------ 6: sb3_call
# The recorded bands call a tie at both budgets (m08_sb3_call), so both mutants answer with a
# ranking. If a future record ranks the two, `higher-iqm` may become right at that budget:
# replace it with an "always tie" mutant then.
@mutant(ex=6, replaces="sb3_call", id="higher-iqm", why="ranks by IQM and ignores the intervals")
def sb3_call():
    from prl.evaluate import iqm

    ref = prl.data.band("m08_dqn_return", budget)["runs"][:, -1]  # noqa: F821
    sb3 = prl.data.band("m08_sb3_return", budget)["runs"][:, -1]  # noqa: F821
    return "reference" if iqm(ref) >= iqm(sb3) else "sb3"


@mutant(
    ex=6,
    replaces="sb3_call",
    id="your-seed",
    why="lets the participant's own seed decide (the Gotcha: landing above SB3's IQM)",
)
def sb3_call():  # noqa: F811
    from prl.evaluate import iqm

    sb3 = prl.data.band("m08_sb3_return", budget)["runs"][:, -1]  # noqa: F821
    return "reference" if log_dqn["eval_return"][-1] > iqm(sb3) else "sb3"  # noqa: F821
