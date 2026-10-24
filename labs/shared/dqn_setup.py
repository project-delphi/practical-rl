# origin: labs/src/08-dqn.py
# Provided plumbing for the workshop's one DQN (Lab 8): the configuration, the network,
# epsilon-greedy acting, the gather, the DQN target, the logger and greedy evaluation.
# Lab 8 includes this file, and so do Modules 9, 14 and 15b, so every DQN in the workshop
# runs the same configuration. dqn_target reads td_target from the notebook (Lab 8,
# Exercise 3), by its name, each time it runs.
import dataclasses
import time
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

PROBE_SEED = 1_000_000  # probe states: reset(seed=PROBE_SEED + i)
EVAL_SEED = 1_100_000  # evaluation episodes: reset(seed=EVAL_SEED + i)


@dataclasses.dataclass(frozen=True)
class DQNConfig:
    """The DQN configuration every lab uses. Change a copy with dataclasses.replace."""

    hidden: tuple = (64, 64)  # two hidden layers, ReLU
    lr: float = 1e-3  # Adam's step size
    buffer_size: int = 10_000  # replay capacity, in transitions
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4  # environment steps per gradient step
    target_sync: int | None = 500  # steps between copies w -> w-; None: never
    learning_starts: int = 1_000  # gradient steps start after this step
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2  # epsilon falls linearly over this share of the run
    max_grad_norm: float = 10.0  # clip the gradient's total norm to this
    huber_threshold: float = 1.0  # psi in the Huber loss


CONFIG = DQNConfig()


class Batch(NamedTuple):
    """A minibatch of B transitions as tensors."""

    obs: torch.Tensor  # (B, obs_dim) float32
    act: torch.Tensor  # (B,) int64
    rew: torch.Tensor  # (B,) float32
    next_obs: torch.Tensor  # (B, obs_dim) float32
    terminated: torch.Tensor  # (B,) float32: 1.0 if s' is a terminal state
    truncated: torch.Tensor  # (B,) float32: 1.0 if the episode was cut at s'


def to_batch(buffer, idx):
    """The transitions in slots idx of a replay buffer, as a Batch of tensors."""
    cols = (buffer.obs, buffer.act, buffer.rew, buffer.next_obs, buffer.terminated)
    return Batch(*(torch.as_tensor(c[idx]) for c in (*cols, buffer.truncated)))


def make_q_net(obs_dim, n_actions, config):
    """Q(s, .; w): obs_dim inputs, config.hidden ReLU layers, one output per action."""
    layers, width = [], obs_dim
    for h in config.hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, n_actions))
    return nn.Sequential(*layers)


def epsilon_at(t, steps, config):
    """Epsilon at step t of a run of `steps`: linear from eps_start to eps_end, then flat."""
    frac = min(1.0, t / max(1.0, config.eps_fraction * steps))
    return config.eps_start + frac * (config.eps_end - config.eps_start)


def act(q_net, obs, epsilon, n_actions, rng):
    """Epsilon-greedy: a uniform action with probability epsilon, else argmax_a Q(s, a; w)."""
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        q = q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(q.argmax(dim=1)[0])


def gather_q(q_values, actions):
    """Q(s_i, a_i; w) for each row i: q_values (B, |A|) and actions (B,) give (B,)."""
    return q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)


def dqn_target(batch, q_net, target_net, gamma):
    """DQN's target: td_target with the target network's max_a' Q(s', a'; w-)."""
    with torch.no_grad():
        q_next = target_net(batch.next_obs).max(dim=1).values
        return td_target(batch, q_next, gamma)  # (masked target)


def probe_states(make_env, n=32):
    """n fixed start states, the same for every run: reset(seed=PROBE_SEED + i)."""
    env = make_env()
    obs = [env.reset(seed=PROBE_SEED + i)[0] for i in range(n)]
    env.close()
    return np.asarray(obs, dtype=np.float32)


def evaluate_greedy(q_net, make_env, episodes=10, gamma=0.99):
    """Greedy episodes (epsilon = 0) from fixed start states, reset(seed=EVAL_SEED + i).

    Returns a dict: "returns" (undiscounted), "disc_returns" (sum of gamma^t r_t from the
    start state), "q0" (max_a Q(s0, a; w)), "terminated" (bool), each of shape (episodes,),
    and "rewards", a list with one float array per episode.
    """
    env = make_env()
    out = {k: [] for k in ("returns", "disc_returns", "q0", "terminated", "rewards")}
    for i in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + i)
        rewards, done, first = [], False, True
        while not done:
            with torch.no_grad():
                q = q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
            if first:
                out["q0"].append(float(q.max()))
                first = False
            obs, r, terminated, truncated, _ = env.step(int(q.argmax()))
            rewards.append(float(r))
            done = terminated or truncated
        rewards = np.asarray(rewards)
        out["returns"].append(float(rewards.sum()))
        out["disc_returns"].append(float(np.sum(gamma ** np.arange(len(rewards)) * rewards)))
        out["terminated"].append(bool(terminated))
        out["rewards"].append(rewards)
    env.close()
    res = {k: np.asarray(v) for k, v in out.items() if k != "rewards"}
    res["rewards"] = out["rewards"]
    return res


class DQNLogger:
    """Provided. Collects what train_dqn reports, as NumPy arrays (Lab 8, Module 9).

    train_dqn calls step() once per environment step (last), update() once per gradient
    step and sync() at every target copy, then returns result(). Every log_every steps it
    records a log point; every eval_every steps (if set) a greedy evaluation; at the first
    gradient step at or after each of snapshot_steps, one batch snapshot.
    """

    def __init__(
        self,
        config,
        steps,
        q_net,
        target_net,
        make_env,
        *,
        log_every=1000,
        eval_every=None,
        eval_episodes=10,
        probe_obs=None,
        snapshot_steps=(),
        verbose=True,
    ):
        self.cfg, self.steps, self.make_env = config, steps, make_env
        self.q_net, self.target_net = q_net, target_net
        self.log_every, self.eval_every, self.eval_episodes = log_every, eval_every, eval_episodes
        if probe_obs is None:
            probe_obs = probe_states(make_env)
        self.probe = torch.as_tensor(np.asarray(probe_obs, dtype=np.float32))
        with torch.no_grad():
            self.n_actions = int(q_net(self.probe[:1]).shape[1])
        self.pending = sorted(int(s) for s in snapshot_steps)
        self.verbose = verbose
        self.print_every = max(log_every, steps // 10)
        self.start = time.perf_counter()
        self.last_sync = 0
        self.ep_ret = self.ep_disc = 0.0
        self.ep_len = 0
        self.e = {k: [] for k in ("return", "disc_return", "length", "end_step", "terminated")}
        keys = ("step", "epsilon", "loss", "td_abs_mean", "td_abs_max", "q_max_mean")
        keys += ("grad_norm", "q_probe", "q_probe_target", "q_terminal", "steps_since_sync")
        self.p = {k: [] for k in keys}
        keys = ("step", "return", "disc_return", "q0")
        self.ev = {k: [] for k in keys}
        keys = ("step", "q", "actions", "gathered", "target", "reward", "terminated")
        self.s = {k: [] for k in keys}
        self._reset_acc()

    def _reset_acc(self):
        self.acc = dict(n=0, loss=0.0, td=0.0, td_n=0, td_max=0.0, qmax=0.0, gn=0.0)
        self.acc.update(qt=0.0, qt_n=0)

    def update(self, t, batch, q_values, q_sa, y, loss, grad_norm):
        """One gradient step: q_values (B, |A|) and q_sa (B,) as the loss used them."""
        with torch.no_grad():
            td = (y - q_sa).abs()
            a = self.acc
            a["n"] += 1
            a["loss"] += float(loss)
            a["td"] += float(td.sum())
            a["td_n"] += int(td.numel())
            a["td_max"] = max(a["td_max"], float(td.max()))
            a["qmax"] += float(q_values.max(dim=1).values.mean())
            a["gn"] += float(grad_norm)
            term = batch.terminated > 0.5
            a["qt"] += float(q_sa[term].sum())
            a["qt_n"] += int(term.sum())
            while self.pending and t >= self.pending[0]:
                self.pending.pop(0)
                self.s["step"].append(t)
                self.s["q"].append(q_values.detach().numpy().copy())
                self.s["actions"].append(batch.act.numpy().copy())
                self.s["gathered"].append(q_sa.detach().numpy().copy())
                self.s["target"].append(y.detach().numpy().copy())
                self.s["reward"].append(batch.rew.numpy().copy())
                self.s["terminated"].append(batch.terminated.numpy() > 0.5)

    def sync(self, t):
        """The target network was just copied from the online network."""
        self.last_sync = t

    def step(self, t, epsilon, reward, terminated, truncated):
        """The end of environment step t (1, 2, ...), after any update and sync."""
        self.ep_disc += self.cfg.gamma**self.ep_len * float(reward)
        self.ep_ret += float(reward)
        self.ep_len += 1
        if terminated or truncated:
            for k, v in zip(
                self.e,
                (self.ep_ret, self.ep_disc, self.ep_len, t, bool(terminated)),
                strict=True,
            ):
                self.e[k].append(v)
            self.ep_ret = self.ep_disc = 0.0
            self.ep_len = 0
        if t % self.log_every == 0:
            self._log_point(t, epsilon)
        if self.eval_every and t % self.eval_every == 0:
            ev = evaluate_greedy(self.q_net, self.make_env, self.eval_episodes, self.cfg.gamma)
            for k, v in zip(self.ev, (t, ev["returns"], ev["disc_returns"], ev["q0"]), strict=True):
                self.ev[k].append(float(np.mean(v)) if k != "step" else v)
        if self.verbose and t % self.print_every == 0:
            recent = self.e["return"][-10:]
            mean = f"{np.mean(recent):6.1f}" if recent else "     -"
            q = self.p["q_probe"][-1].max(axis=1).mean() if self.p["q_probe"] else np.nan
            print(
                f"step {t:>7,} · eps {epsilon:4.2f} · last 10 returns {mean} · "
                f"max Q at probe states {q:7.2f} · {time.perf_counter() - self.start:5.1f} s"
            )

    def _log_point(self, t, epsilon):
        a, nan = self.acc, float("nan")
        n = a["n"]
        with torch.no_grad():
            q_on = self.q_net(self.probe).numpy().copy()
            q_tg = self.target_net(self.probe).numpy().copy()
        values = (
            t,
            epsilon,
            a["loss"] / n if n else nan,
            a["td"] / a["td_n"] if n else nan,
            a["td_max"] if n else nan,
            a["qmax"] / n if n else nan,
            a["gn"] / n if n else nan,
            q_on,
            q_tg,
            a["qt"] / a["qt_n"] if a["qt_n"] else nan,
            t - self.last_sync,
        )
        for k, v in zip(self.p, values, strict=True):
            self.p[k].append(v)
        self._reset_acc()

    def result(self):
        """The log: a dict of NumPy arrays, plus "config" (a dict) and "seconds"."""
        f4, i8 = np.float32, np.int64
        B, A, P = self.cfg.batch_size, self.n_actions, self.probe.shape[0]
        out = {
            "ep_return": np.asarray(self.e["return"], f4),
            "ep_disc_return": np.asarray(self.e["disc_return"], f4),
            "ep_length": np.asarray(self.e["length"], np.int32),
            "ep_end_step": np.asarray(self.e["end_step"], i8),
            "ep_terminated": np.asarray(self.e["terminated"], bool),
            "probe_obs": self.probe.numpy().copy(),
        }
        for k in ("step", "steps_since_sync"):
            out[k] = np.asarray(self.p[k], i8)
        for k in ("epsilon", "loss", "td_abs_mean", "td_abs_max", "q_max_mean", "grad_norm"):
            out[k] = np.asarray(self.p[k], f4)
        out["q_terminal"] = np.asarray(self.p["q_terminal"], f4)
        for k in ("q_probe", "q_probe_target"):
            out[k] = np.asarray(self.p[k], f4).reshape(-1, P, A)
        out["eval_step"] = np.asarray(self.ev["step"], i8)
        for k in ("return", "disc_return", "q0"):
            out[f"eval_{k}"] = np.asarray(self.ev[k], f4)
        out["snap_step"] = np.asarray(self.s["step"], i8)
        shapes = dict(q=(B, A), actions=(B,), gathered=(B,), target=(B,), reward=(B,))
        types = dict(q=f4, actions=i8, gathered=f4, target=f4, reward=f4)
        for k, shape in shapes.items():
            out[f"snap_{k}"] = np.asarray(self.s[k], types[k]).reshape(-1, *shape)
        out["snap_terminated"] = np.asarray(self.s["terminated"], bool).reshape(-1, B)
        out["config"] = dataclasses.asdict(self.cfg)
        out["seconds"] = round(time.perf_counter() - self.start, 2)
        return out
