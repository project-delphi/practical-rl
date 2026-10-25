"""Checkpoints for Module 8 (DQN from scratch).

Exact checks compare the participant's code with fixtures written by
scripts/make_expected.py (`m08_*`): a ring buffer's slots after a known sequence of writes,
targets on fixed tensors, Double DQN's targets from two stand-in networks
that look values up in a table, Monte Carlo returns, and the verdict of the recorded bands.
Values that pass through torch kernels are compared with tolerances, not bit equality: CPU
kernels differ between macOS arm64 and Linux x86-64 (day-3 decision 8).

Checkpoint 3.1 is exact in the sense that matters: a seeded run of the participant's
train_dqn on a five-step chain, whose log must show the structure DQN needs (episodes that
end the way the chain ends, a target copy every `target_sync` steps, terminal rows masked and
truncated rows bootstrapped, and, with the step size set to zero, gradients that do not pile
up from one step to the next). It never compares learned values with numbers, so it does not
depend on torch's kernels. Checkpoint 3.2 is the one StochasticCheck: how far a run of the
participant's train_dqn on CartPole raises the greedy return above its untrained network's
(the lab's provided evaluate_greedy, through the participant's logger), with thresholds from
scripts/threshold_protocol.py.

prl does not depend on torch: every function that needs it imports it inside. Nothing here
trains, plays or computes a target of its own; the checks call the participant's functions
and compare.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import numpy as np

from .. import runtime
from . import CheckFailed, StochasticCheck, assert_close, assert_true, expected

EVAL_EPISODES = 10  # episodes per greedy evaluation in checkpoint 4.2, as in the lab
#: Lab 8's two budgets (the keys its bands match on), to pick the verdict of checkpoint 7.
LAB_BUDGETS = (dict(steps=50_000, eval_every=5_000), dict(steps=10_000, eval_every=1_000))
CALLS = ("tie", "reference", "sb3")


class _Batch(NamedTuple):
    """The lab's Batch: the participant's functions read it by attribute name only."""

    obs: Any
    act: Any
    rew: Any
    next_obs: Any
    terminated: Any
    truncated: Any


def _numpy(x: Any) -> np.ndarray:
    """A tensor (detached) or an array-like, as a float NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _batch(rew, terminated, truncated, *, obs=None, next_obs=None) -> _Batch:
    import torch

    n = len(rew)
    f32 = torch.float32
    return _Batch(
        obs=torch.zeros(n, 1) if obs is None else torch.as_tensor(obs, dtype=f32),
        act=torch.zeros(n, dtype=torch.int64),
        rew=torch.as_tensor(rew, dtype=f32),
        next_obs=torch.zeros(n, 1) if next_obs is None else torch.as_tensor(next_obs, dtype=f32),
        terminated=torch.as_tensor(terminated, dtype=f32),
        truncated=torch.as_tensor(truncated, dtype=f32),
    )


# ------------------------------------------------------------------ 1.1 the replay buffer
def _uniform(counts: np.ndarray, what: str) -> None:
    from scipy import stats

    p = float(stats.chisquare(counts).pvalue)
    assert_true(
        p > 1e-4,
        f"{what}: the draws per slot are not uniform (chi-square p = {p:.1e}). sample should "
        "draw every filled slot with the same probability.",
    )


def check_replay_buffer(ReplayBuffer: type) -> None:
    """add writes at ptr and wraps; sample draws uniformly from the filled slots."""
    import torch

    fx = expected("m08_buffer")
    cap, dim = int(fx["capacity"]), int(fx["obs_dim"])
    b = ReplayBuffer(cap, dim)
    for t in range(int(fx["n_adds"])):
        b.add([t, -t], t % 2, float(t), [t + 1, -t - 1], t == fx["terminal_at"], t == fx["cut_at"])
    n = int(fx["n_adds"])
    assert_true(
        b.size == fx["size"],
        f"After {n} adds into {cap} slots, size should be {int(fx['size'])} (it stops growing "
        f"at capacity), not {b.size}.",
    )
    assert_true(
        b.ptr == fx["ptr"],
        f"After {n} adds into {cap} slots, ptr should be {int(fx['ptr'])} (it wraps to 0 at "
        f"capacity, so the oldest transition is overwritten next), not {b.ptr}.",
    )
    assert_close(
        np.asarray(b.obs)[:, 0],
        fx["obs0"],
        what="obs[:, 0] (the step each slot was written at)",
        hint="slot ptr gets the new transition, then ptr moves on by one, wrapping at capacity",
    )
    for key in ("act", "rew"):
        assert_close(np.asarray(getattr(b, key)), fx[key], what=f"{key} per slot")
    assert_close(np.asarray(b.next_obs)[:, 0], fx["next_obs0"], what="next_obs[:, 0] per slot")
    assert_close(np.asarray(b.terminated), fx["terminated"], what="terminated per slot")
    assert_close(
        np.asarray(b.truncated),
        fx["truncated"],
        what="truncated per slot",
        hint="store truncated as its own flag; it is not terminated",
    )
    batch = b.sample(64, np.random.default_rng(0))
    shapes = dict(obs=(64, dim), act=(64,), rew=(64,), next_obs=(64, dim))
    shapes.update(terminated=(64,), truncated=(64,))
    for key, shape in shapes.items():
        x = getattr(batch, key, None)
        assert_true(
            isinstance(x, torch.Tensor),
            f"sample should return a Batch of tensors (to_batch(self, idx)); batch.{key} is "
            f"{type(x).__name__}.",
        )
        assert_true(
            tuple(x.shape) == shape, f"batch.{key} has shape {tuple(x.shape)}, not {shape}."
        )
    assert_true(batch.act.dtype == torch.int64, f"batch.act is {batch.act.dtype}, not int64.")
    assert_true(batch.obs.dtype == torch.float32, f"batch.obs is {batch.obs.dtype}, not float32.")

    part = ReplayBuffer(int(fx["part_capacity"]), 1)
    for t in range(int(fx["part_adds"])):
        part.add([t + 1], 0, 0.0, [0], False, False)
    drawn = set(_numpy(part.sample(1000, np.random.default_rng(1)).obs[:, 0]).tolist())
    filled = {float(t + 1) for t in range(int(fx["part_adds"]))}
    assert_true(
        drawn <= filled,
        f"A buffer of {int(fx['part_capacity'])} slots holding {int(fx['part_adds'])} "
        "transitions drew an empty slot: draw from the size filled slots, not from capacity.",
    )
    assert_true(
        drawn == filled,
        f"A buffer holding {int(fx['part_adds'])} transitions drew only "
        f"{len(drawn)} of them in 1,000 draws: every filled slot, the newest included, must "
        "be possible (rng.integers(0, n) never returns n).",
    )

    full = ReplayBuffer(int(fx["full_capacity"]), 1)
    for t in range(int(fx["full_adds"])):
        full.add([t], 0, 0.0, [0], False, False)
    rng = np.random.default_rng(2)
    steps = np.stack([_numpy(full.sample(32, rng).obs[:, 0]) for _ in range(2000)]).astype(int)
    cap_full = int(fx["full_capacity"])
    _uniform(np.bincount(steps.ravel() % cap_full, minlength=cap_full), "A full buffer")
    adjacent = float(np.mean(np.diff(steps, axis=1) == 1))
    assert_true(
        adjacent < 0.05,
        f"{adjacent:.0%} of neighbours in a batch were written one step apart: the batch is a "
        "run of consecutive transitions, not independent draws (replay sampling).",
    )


# ------------------------------------------------------------------ 1.2, 1.3 the colleague's buffer
_DIAGNOSIS_HINTS = {
    "a": "the two histograms are not the same: compare the oldest draw of each buffer.",
    "c": "both buffers share your add, so both overwrite old slots. Compare the oldest draw "
    "of each, and ptr with size after 2,500 writes.",
    "d": "look at the colleague's histogram again: which ages does it never draw?",
}


def check_buffer_diagnosis(buffer_diagnosis) -> None:
    """The letter that describes ColleagueBuffer (a multiple-choice answer)."""
    got = buffer_diagnosis()
    want = str(expected("m08_diagnosis")["answer"])
    assert_true(
        isinstance(got, str) and got.strip().lower() in "abcd" and len(got.strip()) == 1,
        f'buffer_diagnosis() should return one letter, "a", "b", "c" or "d"; it returned {got!r}.',
    )
    letter = got.strip().lower()
    assert_true(
        letter == want,
        f"buffer_diagnosis() returned {letter!r}: "
        + _DIAGNOSIS_HINTS.get(letter, "read the ages again.")
        + " Then look at which slots rng.integers(0, self.ptr) can return.",
    )


def check_colleague_buffer(ColleagueBuffer: type) -> None:
    """The fixed sample draws every filled slot uniformly, before and after the first wrap."""
    fx = expected("m08_colleague")
    cap = int(fx["capacity"])
    b = ColleagueBuffer(cap, 4)
    for t in range(int(fx["full_adds"])):
        b.add([t, 0, 0, 0], 0, 1.0, [t + 1, 0, 0, 0], False, False)
    rng = np.random.default_rng(3)
    steps = np.concatenate([_numpy(b.sample(1000, rng).obs[:, 0]) for _ in range(100)])
    counts = np.bincount(steps.astype(int) % cap, minlength=cap)
    never = int(np.sum(counts == 0))
    assert_true(
        never == 0,
        f"After {int(fx['full_adds']):,} writes into {cap:,} slots (ptr {b.ptr}, size {b.size}), "
        f"{never} of the {cap:,} stored transitions were never drawn in 100,000 draws. ptr is "
        "where the next write goes, not how many slots hold a transition.",
    )
    _uniform(counts, "After the pointer wrapped")
    part = ColleagueBuffer(cap, 4)
    n = int(fx["part_adds"])
    for t in range(n):
        part.add([t + 1, 0, 0, 0], 0, 1.0, [t + 2, 0, 0, 0], False, False)
    steps = _numpy(part.sample(30_000, np.random.default_rng(4)).obs[:, 0])
    assert_true(
        steps.min() >= 1,
        f"A buffer of {cap:,} slots holding {n} transitions drew an empty slot: before the "
        "buffer fills, capacity counts slots that hold nothing.",
    )
    _uniform(np.bincount(steps.astype(int) - 1, minlength=n), "Before the buffer filled")


# ------------------------------------------------------------------ 2 the masked target
def check_td_target(td_target) -> None:
    """r + gamma (1 - terminated) q_next on fixed rows, truncated rows included."""
    import torch

    fx = expected("m08_td_target")
    b = _batch(fx["rew"], fx["terminated"], fx["truncated"])
    q_next = torch.as_tensor(fx["q_next"], dtype=torch.float32)
    for gamma, want in zip(fx["gammas"], fx["y"], strict=True):
        got = _numpy(td_target(b, q_next, float(gamma)))
        if got.shape == want.shape and not np.allclose(got, want, atol=1e-5, rtol=1e-6):
            wrong = ~np.isclose(got, want, atol=1e-5, rtol=1e-6)
            cut = (fx["truncated"] > 0) & (fx["terminated"] < 1)
            end = fx["terminated"] > 0
            if np.any(wrong & cut) and not np.any(wrong & ~cut):
                hint = (
                    "only the truncated rows are wrong: a truncated transition (the time limit "
                    "cut the episode) still bootstraps. Mask on terminated alone."
                )
            elif np.any(wrong & end) and not np.any(wrong & ~end):
                hint = "only the terminated rows are wrong: their target is the reward alone."
            else:
                hint = "y = r + gamma * (1 - terminated) * q_next, row by row."
        else:
            hint = "y = r + gamma * (1 - terminated) * q_next, row by row."
        assert_close(got, want, atol=1e-5, what=f"td_target with gamma = {gamma:g}", hint=hint)


# ------------------------------------------------------------------ 3.1 train_dqn's structure
class _Chain:
    """A five-step chain: 1-D observation, reward 1 per step, two actions that do the same.

    Every episode lasts exactly `end` steps and then terminates (terminate=True) or is cut
    by a time limit (terminate=False).
    """

    def __init__(self, end: int, terminate: bool):
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.end, self.terminate, self.t = end, terminate, 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.end
        obs = np.full(1, self.t / self.end, np.float32)
        return obs, 1.0, done and self.terminate, done and not self.terminate, {}

    def close(self):
        pass


_LOG_KEYS = (
    "ep_length",
    "ep_terminated",
    "step",
    "steps_since_sync",
    "q_probe",
    "q_probe_target",
    "grad_norm",
    "snap_step",
    "snap_terminated",
    "snap_target",
    "snap_reward",
)


def _chain_log(train_dqn, config, terminate: bool, **changes) -> dict:
    fx = expected("m08_chain")
    cfg = dataclasses.replace(
        config,
        learning_starts=int(fx["learning_starts"]),
        buffer_size=int(fx["buffer_size"]),
        batch_size=int(fx["batch_size"]),
        target_sync=int(fx["target_sync"]),
        train_freq=1,
        **changes,
    )
    end = int(fx["episode_length"])
    out = train_dqn(
        lambda: _Chain(end, terminate),
        cfg,
        int(fx["seed"]),
        int(fx["steps"]),
        log_every=int(fx["log_every"]),
        probe_obs=np.zeros((2, 1), np.float32),
        snapshot_steps=tuple(int(s) for s in fx["snapshot_steps"]),
        verbose=False,
    )
    log = out[1] if isinstance(out, tuple) and len(out) == 2 else None
    assert_true(
        isinstance(log, dict),
        "train_dqn should return (p.q_net, p.log.result()): the network and the logger's dict.",
    )
    missing = [k for k in _LOG_KEYS if k not in log]
    assert_true(
        not missing,
        f"train_dqn's log has no {', '.join(missing)}: return p.q_net, p.log.result() after "
        "the loop, having called p.log.step, p.log.update and p.log.sync as the docstring "
        "says.",
    )
    return log


def check_train_dqn_masks(train_dqn, config) -> None:
    """A seeded run on a five-step chain that terminates, then on one that is cut: the log's
    episodes, target copies, terminal mask and bootstraps; then gradients at step size 0."""
    fx = expected("m08_chain")
    sync, steps = int(fx["target_sync"]), int(fx["steps"])
    end = int(fx["episode_length"])
    for terminate in (True, False):
        kind = "terminates" if terminate else "is cut by a time limit"
        where = f"On a chain whose every episode {kind} after {end} steps"
        log = _chain_log(train_dqn, config, terminate)
        lengths = np.asarray(log["ep_length"])
        assert_true(
            len(lengths) == steps // end and set(lengths.tolist()) == {end},
            f"{where}, {steps} steps should give {steps // end} episodes of {end} steps; the "
            f"log has {len(lengths)} with lengths {sorted(set(lengths.tolist()))}. Reset p.env "
            "when either flag is set (docstring step 1), and call p.log.step(t, ...) once per "
            "step (step 4).",
        )
        assert_true(
            bool(np.all(np.asarray(log["ep_terminated"]) == terminate)),
            f"{where}, ep_terminated should be {terminate} for every episode: pass "
            "terminated and truncated to p.log.step as p.env.step returned them (step 4).",
        )
        want_snaps = fx["snap_at"].tolist()
        got_snaps = np.asarray(log["snap_step"]).tolist()
        assert_true(
            got_snaps == want_snaps,
            f"{where}, with learning_starts = {int(fx['learning_starts'])} and train_freq = 1 "
            f"the log's batch snapshots should be at steps {want_snaps}, not {got_snaps}: take "
            "a gradient step at every t > config.learning_starts with t % config.train_freq "
            "== 0, read from config, and call p.log.update after each (docstring step 2).",
        )
        q = np.asarray(log["q_probe"], dtype=float)
        assert_true(
            bool(np.abs(q[-1] - q[0]).max() > 1e-6),
            f"{where}, Q at the probe states did not change in {steps} steps of training: "
            "call p.opt.step() after loss.backward() and clip_grad_norm_ (docstring step 2).",
        )
        since = np.asarray(log["steps_since_sync"]).tolist()
        assert_true(
            since == [0] * (steps // int(fx["log_every"])),
            f"{where}, with target_sync = log_every = {sync} every log point should follow a "
            f"target copy, but the steps since the last copy are {since}. Copy w into w- "
            "(p.target_net.load_state_dict(p.q_net.state_dict())) whenever t % "
            "config.target_sync == 0, then call p.log.sync(t) (docstring step 3).",
        )
        assert_close(
            log["q_probe_target"],
            log["q_probe"],
            atol=1e-6,
            what=f"{where}, the target network's values right after a copy",
            hint="the copy must load the online network's current parameters into the target "
            "network that the targets come from",
        )
        stored_end = np.asarray(log["snap_terminated"], dtype=bool)
        target = np.asarray(log["snap_target"], dtype=float)
        reward = np.asarray(log["snap_reward"], dtype=float)
        if terminate:
            assert_true(
                bool(stored_end.any()),
                f"{where}, no batch held a terminal transition: store terminated as p.env.step "
                "returned it (docstring step 1).",
            )
            assert_close(
                target[stored_end],
                reward[stored_end],
                atol=1e-6,
                what=f"{where}, the targets of terminal rows",
                hint="a terminal transition's target is its reward alone",
            )
        else:
            assert_true(
                not stored_end.any(),
                f"{where}, a cut transition was stored as terminated: p.buffer.add takes "
                "terminated and truncated separately (docstring step 1), and only terminated "
                "stops the bootstrap.",
            )
            assert_true(
                bool(np.all(np.abs(target - reward) > 1e-6)),
                f"{where}, a target equals its reward: a truncated transition still bootstraps.",
            )
    # Step size 0: the network never changes, so each step's gradient is that of one batch at
    # the same parameters. Gradients left over from earlier steps make the norm grow.
    log = _chain_log(train_dqn, config, False, lr=0.0, max_grad_norm=float(fx["no_clip"]))
    assert_close(
        log["q_probe"],
        np.broadcast_to(np.asarray(log["q_probe"])[:1], np.shape(log["q_probe"])),
        atol=1e-6,
        what="With config.lr = 0, Q at the probe states over the run",
        hint="the network should not move at step size 0: use p.opt from dqn_parts, which "
        "reads config.lr",
    )
    norms = np.asarray(log["grad_norm"], dtype=float)
    assert_true(
        bool(np.all(np.isfinite(norms)) and norms[0] > 0),
        f"With config.lr = 0 the gradient norms per log point are {np.round(norms, 3).tolist()}: "
        "no gradient reached the parameters. Call loss.backward() after p.opt.zero_grad() and "
        "before clip_grad_norm_ and p.opt.step() (docstring step 2).",
    )
    ratio = float(norms[-1] / norms[0])
    assert_true(
        ratio < float(fx["max_growth"]),
        f"With config.lr = 0 the network never changes, so every gradient is one batch's, yet "
        f"the gradient norm grew {ratio:.0f}-fold over {steps} steps "
        f"({np.round(norms, 2).tolist()}). Gradients are piling up: call p.opt.zero_grad() "
        "before each loss.backward() (docstring step 2).",
    )


# ------------------------------------------------------------------ 3.2 train_dqn learns (stochastic)
def _greedy_improvement(train_dqn, make_env, config, *, seed: int, steps: int) -> float:
    """How far training raised the greedy return above the untrained network's.

    The run evaluates the greedy policy every config.learning_starts steps (10 episodes from
    the lab's start states, by the provided evaluate_greedy through the participant's
    logger). The first evaluation comes before the first gradient step, so it scores the
    untrained network; the metric is the best evaluation minus that one. Code that never
    learns scores exactly 0, however well its initial network happens to balance; the best
    evaluation alone could not tell it apart, since an untrained network balanced for more
    than 130 steps on one seed in 200.
    """
    every = int(config.learning_starts)
    out = train_dqn(
        make_env,
        config,
        seed,
        steps,
        log_every=steps,
        eval_every=every,
        eval_episodes=EVAL_EPISODES,
        verbose=False,
    )
    log = out[1] if isinstance(out, tuple) and len(out) == 2 else None
    if not isinstance(log, dict) or "eval_return" not in log or "eval_step" not in log:
        raise CheckFailed(
            "train_dqn should return (q_net, log.result()), and its log should hold the greedy "
            "evaluations: pass log_options to DQNLogger and call log.step(t, ...) every step."
        )
    grid = list(range(every, steps + 1, every))
    if np.asarray(log["eval_step"]).tolist() != grid:
        raise CheckFailed(
            f"With eval_every = {every:,}, the log's greedy evaluations should be at steps "
            f"{grid[0]:,}, {grid[1]:,}, ..., {grid[-1]:,}; they are at "
            f"{np.asarray(log['eval_step']).tolist()}. Call log.step(t, ...) once per step."
        )
    returns = np.asarray(log["eval_return"], dtype=float)
    return float(returns.max() - returns[0])


check_train_dqn = StochasticCheck(
    ex=3,
    functions=("train_dqn", "make_cartpole", "CONFIG"),
    metric=_greedy_improvement,
    # C33: 20,000 steps, not the lab's 50,000, at both budgets. With the last evaluation as the
    # metric, the gap at 20,000 (68.7 on seeds 0-19) was twice that at 50,000 (33.3), and
    # 10,000 gave none (12.4 over 100 seeds). Over more seeds the last evaluation, and then the
    # best one, let a never-learning network through on some seed, hence the improvement.
    live={"steps": 20_000},
    quick={"steps": 20_000},
    higher_is_better=True,
    min_gap=25.0,  # about one uniform-random policy's own mean return (22.3 over 1,000 episodes)
    what=(
        "how far your train_dqn raised the greedy return on CartPole at the checkpoint's seed: "
        "the best of the run's evaluations (every 1,000 steps, 10 episodes each) minus the "
        "first, taken before any gradient step"
    ),
    thresholds={
        # threshold_protocol.py, live budget {"steps": 20000}, N = 200 seeds (0-199),
        # 2026-10-25, local (Apple M1 Pro, arm64). Solution p1 165.2; strongest wrong version
        # mutant 'random' p99 16.6; gap 148.6 (minimum 25); threshold: the midpoint 90.9035,
        # rounded to 91.0. 0 of 200 solution seeds fail: flake rate at most about 3/N = 1.5%
        # (rule of three).
        "live": (
            91.0,
            "threshold protocol, 200 seeds: runs/2026-10-25-local-threshold-m08-check_train_dqn-live-n200-exp-3.json",
        ),
        # threshold_protocol.py, quick budget {"steps": 20000}, N = 200 seeds (0-199),
        # 2026-10-25, local (Apple M1 Pro, arm64). Solution p1 165.2; strongest wrong version
        # mutant 'random' p99 16.6; gap 148.6 (minimum 25); threshold: the midpoint 90.9035,
        # rounded to 91.0. 0 of 200 solution seeds fail: flake rate at most about 3/N = 1.5%
        # (rule of three).
        "quick": (
            91.0,
            "threshold protocol, 200 seeds: runs/2026-10-25-local-threshold-m08-check_train_dqn-quick-n200-exp-3.json",
        ),
    },
)


# ------------------------------------------------------------------ 4 the Double DQN target
def check_double_dqn_target(double_dqn_target) -> None:
    """Online network selects at s', target network evaluates; masked; no gradient."""
    import torch

    fx = expected("m08_double_dqn")
    w = torch.zeros(1, requires_grad=True)  # so that a target built with a gradient shows it
    online = torch.as_tensor(fx["online"], dtype=torch.float32)
    target = torch.as_tensor(fx["target"], dtype=torch.float32)

    def q_net(s):
        return online[s[:, 0].long()] + 0 * w

    def target_net(s):
        return target[s[:, 0].long()] + 0 * w

    b = _batch(fx["rew"], fx["terminated"], fx["truncated"], obs=fx["obs"], next_obs=fx["next_obs"])
    y = double_dqn_target(b, q_net, target_net, float(fx["gamma"]))
    got = _numpy(y)
    hint = "select a' with q_net at s' (batch.next_obs), evaluate it with target_net"
    for key, words in (
        ("y_dqn", "that is DQN's target: the target network picked a' as well as valuing it."),
        ("y_online", "the online network both picked a' and valued it."),
        ("y_at_s", "a' was picked at s (batch.obs); pick it at s' (batch.next_obs)."),
    ):
        if got.shape == fx[key].shape and np.allclose(got, fx[key], atol=1e-5):
            hint = f"{words} Select with q_net, evaluate with target_net"
    assert_close(got, fx["y"], atol=1e-5, what="double_dqn_target", hint=hint)
    assert_true(
        not getattr(y, "requires_grad", False),
        "The targets carry a gradient: compute them inside torch.no_grad(), as dqn_target does.",
    )


# ------------------------------------------------------------------ 5 Monte Carlo returns
def check_mc_return(mc_return) -> None:
    """Discounted returns-to-go on fixed episodes."""
    fx = expected("m08_mc_return")
    starts = np.concatenate([[0], np.cumsum(fx["lengths"])])
    for i, gamma in enumerate(fx["gammas"]):
        rewards = fx["rewards"][starts[i] : starts[i + 1]]
        want = fx["returns"][starts[i] : starts[i + 1]]
        assert_close(
            mc_return(rewards.copy(), float(gamma)),
            want,
            what=f"mc_return({rewards.tolist() if len(rewards) <= 8 else 'rewards'}, {gamma:g})",
            hint="one G_t per step, from the end: G_t = r_t + gamma * G_{t+1}, with G = 0 after "
            "the last reward (so the last G is the last reward)",
        )


# ------------------------------------------------------------------ 6 the recorded call
def check_sb3_call(sb3_call) -> None:
    """The rule's verdict on the reference DQN against SB3's DQN at the lab's budget."""
    budget = runtime.settings(*LAB_BUDGETS, designed="colab-cpu")
    want = str(expected("m08_sb3_call")[budget.source])
    got = sb3_call()
    assert_true(
        got in CALLS,
        f'sb3_call() should return "tie", "reference" or "sb3"; it returned {got!r}.',
    )
    assert_true(
        got == want,
        f"sb3_call() returned {got!r}, which is not what the rule says at this budget. "
        "Read the printed IQMs and 95% intervals: overlapping intervals are a tie; otherwise "
        "the side with the higher IQM is better. Your own seed does not enter the call.",
    )
