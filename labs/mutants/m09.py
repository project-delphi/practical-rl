"""Plausible wrong versions of Module 9's functions (second draft by the academic director;
the technical expert completes it).

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires every
exact checkpoint of its exercise that calls the function to reject it. Checkpoint 6.2 (the live
run against the protocol's threshold) is the only stochastic checkpoint; it calls no participant
function, so it has no mutant here, and its `random` version belongs to the threshold protocol.

Checkpoints 3a-3d all call `diagnoses`, so a `diagnoses` mutant must be wrong on every run to be
rejected by all four: "distractors" picks, for each run, the cause a quick reading would pick.
"""

import gymnasium as gym
import numpy as np

from prl.lab import mutant


# ------------------------------------------------------------------ 1.1: gather_check
@mutant(ex=1, replaces="gather_check", id="argmax", why="compares with max_a Q, not Q(s, a)")
def gather_check(log):
    want = log["snap_q"].max(axis=-1)
    wrong = ~np.isclose(log["snap_gathered"], want, rtol=1e-5, atol=1e-6)
    return float(wrong.mean())


@mutant(ex=1, replaces="gather_check", id="count", why="returns the number of rows, not a share")
def gather_check(log):  # noqa: F811
    q, a = log["snap_q"], log["snap_actions"]
    want = np.take_along_axis(q, a[..., None], axis=-1)[..., 0]
    return float(np.sum(~np.isclose(log["snap_gathered"], want, rtol=1e-5, atol=1e-6)))


@mutant(ex=1, replaces="gather_check", id="exact-equal", why="== instead of a tolerance")
def gather_check(log):  # noqa: F811
    q, a = log["snap_q"], log["snap_actions"]
    want = np.take_along_axis(q, a[..., None], axis=-1)[..., 0]
    return float(np.mean(log["snap_gathered"] != want))


# ------------------------------------------------------------------ 1.2: target_lag
@mutant(ex=1, replaces="target_lag", id="online", why="watches the online network's values")
def target_lag(log):
    q, step = log["q_probe"], log["step"]
    lag = np.zeros(len(step), dtype=np.int64)
    last = step[0]
    for i in range(1, len(step)):
        if np.any(q[i] != q[i - 1]):
            last = step[i]
        lag[i] = step[i] - last
    return lag


@mutant(ex=1, replaces="target_lag", id="log-points", why="counts log points, not env steps")
def target_lag(log):  # noqa: F811
    q_tgt = log["q_probe_target"]
    lag = np.zeros(len(q_tgt), dtype=np.int64)
    last = 0
    for i in range(1, len(q_tgt)):
        if np.any(q_tgt[i] != q_tgt[i - 1]):
            last = i
        lag[i] = i - last
    return lag


# ------------------------------------------------------------------ 2.1: q_drift
@mutant(ex=2, replaces="q_drift", id="undiscounted", why="compares with ep_return, not G_0")
def q_drift(log, window=5000):
    q_bar = log["q_probe"].max(axis=-1).mean(axis=-1)
    end, g = log["ep_end_step"], log["ep_return"]
    drift = np.full(len(log["step"]), np.nan)
    for i, t in enumerate(log["step"]):
        recent = (end > t - window) & (end <= t)
        if recent.any():
            drift[i] = q_bar[i] - g[recent].mean()
    return drift


@mutant(ex=2, replaces="q_drift", id="mean-over-actions", why="averages Q over actions, no max")
def q_drift(log, window=5000):  # noqa: F811
    q_bar = log["q_probe"].mean(axis=(-1, -2))
    end, g0 = log["ep_end_step"], log["ep_disc_return"]
    drift = np.full(len(log["step"]), np.nan)
    for i, t in enumerate(log["step"]):
        recent = (end > t - window) & (end <= t)
        if recent.any():
            drift[i] = q_bar[i] - g0[recent].mean()
    return drift


@mutant(ex=2, replaces="q_drift", id="all-past", why="averages every episode so far")
def q_drift(log, window=5000):  # noqa: F811
    q_bar = log["q_probe"].max(axis=-1).mean(axis=-1)
    end, g0 = log["ep_end_step"], log["ep_disc_return"]
    drift = np.full(len(log["step"]), np.nan)
    for i, t in enumerate(log["step"]):
        recent = end <= t
        if recent.any():
            drift[i] = q_bar[i] - g0[recent].mean()
    return drift


# ------------------------------------------------------------------ 2.2: td_error_scale
@mutant(ex=2, replaces="td_error_scale", id="absolute", why="mean |delta| without the Q scale")
def td_error_scale(log):
    return log["td_abs_mean"]


@mutant(ex=2, replaces="td_error_scale", id="signed-q", why="divides by Q without abs")
def td_error_scale(log):  # noqa: F811
    return log["td_abs_mean"] / log["q_max_mean"]


@mutant(ex=2, replaces="td_error_scale", id="max-delta", why="uses the largest |delta|")
def td_error_scale(log):  # noqa: F811
    return log["td_abs_max"] / np.abs(log["q_max_mean"])


# ------------------------------------------------------------------ 3: diagnoses
@mutant(ex=3, replaces="diagnoses", id="distractors", why="each run's quick-reading cause")
def diagnoses():
    return {"A": "reward", "B": "lr", "C": "reward", "D": "seed"}


@mutant(ex=3, replaces="diagnoses", id="rotated", why="the right causes on the wrong runs")
def diagnoses():  # noqa: F811
    return {"A": "gather", "B": "mask", "C": "sync", "D": "lr"}


# ------------------------------------------------------------------ 4: probe environments
@mutant(ex=4, replaces="ConstantRewardEnv", id="never-ends", why="terminated=False every step")
class ConstantRewardEnv(gym.Env):
    def __init__(self):
        box = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = box
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 1.0, False, False, {}


@mutant(ex=4, replaces="ConstantRewardEnv", id="truncated", why="ends with truncated=True")
class ConstantRewardEnv(gym.Env):  # noqa: F811
    def __init__(self):
        box = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = box
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 1.0, False, True, {}


@mutant(ex=4, replaces="ObsRewardEnv", id="global-rng", why="np.random, so seeds do nothing")
class ObsRewardEnv(gym.Env):
    def __init__(self):
        box = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = box
        self.action_space = gym.spaces.Discrete(1)
        self.obs = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.obs = 1.0 if np.random.random() < 0.5 else -1.0
        return np.array([self.obs], np.float32), {}

    def step(self, action):
        return np.array([self.obs], np.float32), self.obs, True, False, {}


@mutant(ex=4, replaces="ObsRewardEnv", id="fresh-obs", why="step draws a new observation")
class ObsRewardEnv(gym.Env):  # noqa: F811
    def __init__(self):
        box = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = box
        self.action_space = gym.spaces.Discrete(1)
        self.obs = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.obs = 1.0 if self.np_random.random() < 0.5 else -1.0
        return np.array([self.obs], np.float32), {}

    def step(self, action):
        reward = self.obs
        self.obs = 1.0 if self.np_random.random() < 0.5 else -1.0
        return np.array([self.obs], np.float32), reward, True, False, {}


# ------------------------------------------------------------------ 5: iqm, bootstrap_ci
@mutant(ex=5, replaces="iqm", id="round-quarter", why="drops round(n/4), not floor(n/4)")
def iqm(x):
    x = np.sort(np.asarray(x, dtype=float).ravel())
    k = round(len(x) / 4)
    return float(x[k : len(x) - k].mean())


@mutant(ex=5, replaces="iqm", id="median", why="returns the median")
def iqm(x):  # noqa: F811
    return float(np.median(x))


@mutant(ex=5, replaces="iqm", id="unsorted", why="trims without sorting")
def iqm(x):  # noqa: F811
    x = np.asarray(x, dtype=float).ravel()
    k = len(x) // 4
    return float(x[k : len(x) - k].mean())


@mutant(ex=5, replaces="bootstrap_ci", id="level-as-tail", why="quantiles 1 - level and level")
def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([statistic(x[row]) for row in idx])
    lo, hi = np.quantile(boots, [1 - level, level])
    return float(lo), float(hi)


@mutant(ex=5, replaces="bootstrap_ci", id="legacy-rng", why="np.random.seed and randint")
def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):  # noqa: F811
    x = np.asarray(x, dtype=float).ravel()
    np.random.seed(seed)
    idx = np.random.randint(0, len(x), size=(n_boot, len(x)))
    boots = np.array([statistic(x[row]) for row in idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


@mutant(ex=5, replaces="bootstrap_ci", id="no-replacement", why="permutes instead of resampling")
def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):  # noqa: F811
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(rng.permutation(x)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


@mutant(ex=5, replaces="bootstrap_ci", id="normal", why="mean ± 1.96 standard errors")
def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):  # noqa: F811
    x = np.asarray(x, dtype=float).ravel()
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return float(statistic(x) - half), float(statistic(x) + half)


# ------------------------------------------------------------------ 6: band_position
@mutant(ex=6, replaces="band_position", id="top-is-outside", why=">= for above: a tie is out")
def band_position(value, recorded):
    if value >= np.max(recorded):
        return 1
    if value <= np.min(recorded):
        return -1
    return 0


@mutant(ex=6, replaces="band_position", id="sign", why="+1 below and -1 above")
def band_position(value, recorded):  # noqa: F811
    if value > np.max(recorded):
        return -1
    if value < np.min(recorded):
        return 1
    return 0


@mutant(ex=6, replaces="band_position", id="mean", why="compares with the band's mean")
def band_position(value, recorded):  # noqa: F811
    return int(np.sign(value - np.mean(recorded)))
