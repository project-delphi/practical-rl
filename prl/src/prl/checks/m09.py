"""Checkpoints for Module 9 (debugging and evaluating RL).

Exact checks compare the participant's code with fixtures written by
scripts/make_expected.py (`m09_*`): the four signals on small hand-made logs (each log carries
every key a plausible wrong version reads, so a wrong version fails a comparison instead of
crashing) and on the recorded runs of the dataset `m09_broken_runs`; the four diagnoses; the
probe environments' spaces, flags, seeding and the values a correct agent must learn on them,
computed from the environment's own samples; the IQM and the percentile bootstrap against
`prl.evaluate` on fixed inputs; and where a value lands against a band.

Checkpoint 6.2 is the one StochasticCheck. It guards the platform, not the participant: the
live seed's greedy return at the end of training, the value Exercise 6's run cell already
computed (no second training run, OPEN_QUESTIONS C38), against a threshold that
scripts/threshold_protocol.py set by training the reference configuration (the lab's
train_dqn and CONFIG) on many seeds, against random play.

prl does not depend on torch, and nothing here trains, plays or computes a signal of its own:
the checks call the participant's functions and compare with fixtures.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from . import CheckFailed, StochasticCheck, assert_true, expected

RUNS = ("healthy", "A", "B", "C", "D")
EXACT_HINT = "start from the two exact checks, then the rewards and the seeds"


# ------------------------------------------------------------------ helpers
def _broken_runs() -> dict[str, np.ndarray]:
    from .. import data

    return data.load("m09_broken_runs")


def _run_log(d: dict[str, np.ndarray], run: str, seed: int) -> dict[str, np.ndarray]:
    """One recorded run's log, as the lab's provided run_log builds it."""
    s = int(np.flatnonzero(d["seeds"] == seed)[0])
    log = {"step": d["step"].copy(), "probe_obs": d["probe_obs"].copy()}
    mine = d[f"{run}__ep_seed"] == seed
    for key, arr in d.items():
        if key.startswith(f"{run}__") and key != f"{run}__ep_seed":
            name = key.split("__", 1)[1]
            log[name] = (arr[mine] if name.startswith("ep_") else arr[s]).copy()
    return log


def _call(fn: Any, *args: Any, what: str, **kwargs: Any) -> Any:
    out = fn(*args, **kwargs)
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget `return`?")
    return out


def _scalar(out: Any, what: str) -> float:
    if isinstance(out, bool) or not isinstance(out, numbers.Real) and np.ndim(out) != 0:
        raise CheckFailed(f"{what} should return one number, not {type(out).__name__}.")
    try:
        return float(out)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should return one number, not {out!r}.") from None


def _curve(out: Any, n: int, what: str) -> np.ndarray:
    try:
        arr = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should return an array of shape ({n},).") from None
    if arr.shape != (n,):
        raise CheckFailed(
            f"{what} should return shape ({n},), one value per log point; got {arr.shape}."
        )
    return arr


def _show(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x, dtype=float), precision=4, suppress_small=True)


def _same(
    got: np.ndarray,
    want: np.ndarray,
    what: str,
    hint: str,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Equal within the tolerance, NaN where NaN is expected."""
    nan_got, nan_want = np.isnan(got), np.isnan(want)
    if not np.array_equal(nan_got, nan_want):
        where = int(np.flatnonzero(nan_got != nan_want)[0])
        raise CheckFailed(
            f"{what}: entry {where} should be {'NaN' if nan_want[where] else f'{want[where]:.6g}'}, "
            f"got {'NaN' if nan_got[where] else f'{got[where]:.6g}'}. Hint: {hint}"
        )
    ok = np.isclose(got, want, rtol=rtol, atol=atol) | nan_want
    if not ok.all():
        if got.size <= 8:
            detail = f"expected {_show(want)}, got {_show(got)}"
        else:
            i = int(np.flatnonzero(~ok)[0])
            detail = f"entry {i} should be {want[i]:.6g}, got {got[i]:.6g}"
        raise CheckFailed(f"{what}: {detail}. Hint: {hint}")


# ------------------------------------------------------------------ 1 two exact checks
def check_gather_check(gather_check) -> None:
    """1/8 on a fixed batch with one wrong row and one row off by 1e-9; then the record."""
    fx = expected("m09_gather")
    log = {k: fx[k].copy() for k in ("snap_q", "snap_actions", "snap_gathered")}
    got = _scalar(_call(gather_check, log, what="gather_check"), "gather_check")
    want = float(fx["answer"])
    if not np.isclose(got, want, rtol=1e-9, atol=1e-12):
        n = fx["snap_actions"].size
        if np.isclose(got, round(want * n)):
            hint = "return the share of rows (between 0 and 1), not their number."
        elif np.isclose(got, 2 / n):
            hint = (
                "a row off by 1e-9 is not a wrong row: compare with np.isclose(..., rtol=1e-5, "
                "atol=1e-6), not ==."
            )
        else:
            hint = (
                "compare each gathered value with Q(s_i, a_i), the entry of snap_q for that "
                "row's own action (np.take_along_axis), not with max_a Q."
            )
        raise CheckFailed(
            f"gather_check on a fixed batch of {n} rows (2 snapshots of 4) with one wrong row "
            f"returned {got:.6g}; expected {want:.6g}. Hint: {hint}"
        )
    fx = expected("m09_record")
    d = _broken_runs()
    for i, run in enumerate(fx["runs"]):
        for j, seed in enumerate(fx["seeds"]):
            what = f"gather_check on the recorded {'healthy run' if run == 'healthy' else f'run {run}'}, seed {seed}"
            got = _scalar(_call(gather_check, _run_log(d, str(run), int(seed)), what=what), what)
            want = float(fx["gather_check"][i, j])
            assert_true(
                np.isclose(got, want, rtol=1e-9, atol=1e-12),
                f"{what} returned {got:.6g}; expected {want:.6g}. It passed the fixed batch: "
                "does it read the log's own keys snap_q, snap_actions and snap_gathered?",
            )


def check_target_lag(target_lag) -> None:
    """Steps since the target's probe values changed: a fixed log, then the record."""
    fx = expected("m09_lag")
    log = {k: fx[k].copy() for k in ("step", "q_probe_target", "q_probe")}
    out = _call(target_lag, log, what="target_lag")
    got = _curve(out, len(fx["step"]), "target_lag")
    want = fx["answer"].astype(float)
    if not np.array_equal(got, want):
        if np.array_equal(got, np.zeros_like(want)):
            hint = (
                "read q_probe_target, the target network's values; the online values in "
                "q_probe change at every log point."
            )
        elif np.array_equal(got, want / 500):
            hint = "count env steps (step[i] - step[j]), not log points."
        else:
            hint = (
                "lag[i] = step[i] - step[j], where j is the latest log point at which "
                "q_probe_target[j] differs from q_probe_target[j - 1] in any entry; lag[0] = 0."
            )
        raise CheckFailed(
            f"target_lag on a fixed log (steps {fx['step'].tolist()}, the target changing at "
            f"log points 2 and 5): expected {want.astype(int).tolist()}, got {_show(got)}. "
            f"Hint: {hint}"
        )
    assert_true(
        np.issubdtype(np.asarray(out).dtype, np.integer),
        f"target_lag should return an integer array (env steps); got dtype {np.asarray(out).dtype}.",
    )
    fx = expected("m09_record")
    d = _broken_runs()
    for i, run in enumerate(fx["runs"]):
        for j, seed in enumerate(fx["seeds"]):
            what = f"target_lag on the recorded {'healthy run' if run == 'healthy' else f'run {run}'}, seed {seed}"
            got = _curve(
                _call(target_lag, _run_log(d, str(run), int(seed)), what=what), d["step"].size, what
            )
            want = fx["target_lag"][i, j].astype(float)
            if not np.array_equal(got, want):
                k = int(np.flatnonzero(got != want)[0])
                raise CheckFailed(
                    f"{what}: at step {int(d['step'][k]):,} expected {int(want[k])}, got "
                    f"{got[k]:.6g}. It passed the fixed log: does it read log['step'] and "
                    "log['q_probe_target'] of the log it is given?"
                )


# ------------------------------------------------------------------ 2 two scale signals
def check_q_drift(q_drift) -> None:
    """Q at the probe states minus the recent mean G_0: two windows of a fixed log, the record."""
    fx = expected("m09_drift")
    keys = ("step", "q_probe", "ep_end_step", "ep_disc_return", "ep_return")
    n = len(fx["step"])
    for window, want in zip(fx["windows"], fx["answers"], strict=True):
        log = {k: fx[k].copy() for k in keys}
        what = f"q_drift on a fixed log with window={int(window)}"
        got = _curve(_call(q_drift, log, window=int(window), what=what), n, what)
        _same(
            got,
            want,
            what,
            "q_bar[i] is the mean over the probe states of max_a Q (max over the last axis, "
            "then the mean); subtract the mean ep_disc_return (discounted, not ep_return) of the "
            "episodes with t - window < ep_end_step <= t; NaN where none ended.",
        )
    fx = expected("m09_record")
    d = _broken_runs()
    for j, seed in enumerate(fx["seeds"]):
        what = f"q_drift on the recorded run C, seed {seed} (window 5,000)"
        got = _curve(_call(q_drift, _run_log(d, "C", int(seed)), what=what), d["step"].size, what)
        _same(
            got,
            fx["q_drift_C"][j],
            what,
            "it passed the fixed log; is the default window 5000?",
            rtol=1e-6,
            atol=1e-6,
        )


def check_td_error_scale(td_error_scale) -> None:
    """Mean |delta| over |mean max_a Q|, on a fixed log and the recorded run A."""
    fx = expected("m09_td_scale")
    log = {k: fx[k].copy() for k in ("td_abs_mean", "td_abs_max", "q_max_mean")}
    n = len(fx["answer"])
    got = _curve(_call(td_error_scale, log, what="td_error_scale"), n, "td_error_scale")
    _same(
        got,
        fx["answer"],
        "td_error_scale on a fixed log (one row with a negative mean max Q)",
        "divide td_abs_mean (the mean |delta|, not td_abs_max) by the absolute value of "
        "q_max_mean, element by element.",
    )
    fx = expected("m09_record")
    d = _broken_runs()
    for j, seed in enumerate(fx["seeds"]):
        what = f"td_error_scale on the recorded run A, seed {seed}"
        got = _curve(
            _call(td_error_scale, _run_log(d, "A", int(seed)), what=what), d["step"].size, what
        )
        _same(
            got,
            fx["td_scale_A"][j],
            what,
            "it passed the fixed log; read the log it is given.",
            rtol=1e-6,
            atol=0,
        )


# ------------------------------------------------------------------ 3 the diagnoses
def check_diagnosis(diagnoses, run: str) -> None:
    """One run's diagnosis (checkpoints 3.1-3.4). A wrong key is named, the answer never."""
    fx = expected("m09_diagnoses")
    runs = [str(r) for r in fx["runs"]]
    causes = [str(c) for c in fx["causes"]]
    out = _call(diagnoses, what="diagnoses()")
    assert_true(
        isinstance(out, dict) and set(out) == set(runs),
        f'diagnoses() should return a dict with exactly the keys "A", "B", "C" and "D"; got '
        f"{sorted(out) if isinstance(out, dict) else type(out).__name__}.",
    )
    got = out[run]
    assert_true(
        isinstance(got, str) and got in causes,
        f"Run {run}: {got!r} is not one of the table's keys: {', '.join(map(repr, causes))}.",
    )
    want = str(fx["answers"][runs.index(run)])
    assert_true(
        got == want,
        f"Run {run}: not {got!r}. Read run {run}'s row of the table against the healthy row: "
        f"{EXACT_HINT}.",
    )


# ------------------------------------------------------------------ 4 probe environments
def _probe_spaces(cls: type, name: str) -> None:
    import gymnasium as gym

    fx = expected("m09_probes")
    env = cls()
    box = gym.spaces.Box(float(fx["low"]), float(fx["high"]), shape=(1,), dtype=np.float32)
    sp = env.observation_space
    assert_true(
        isinstance(sp, gym.spaces.Box) and sp == box,
        f"{name}.observation_space should be gym.spaces.Box(-1.0, 1.0, shape=(1,), "
        f"dtype=np.float32); got {sp!r}.",
    )
    act = env.action_space
    assert_true(
        isinstance(act, gym.spaces.Discrete) and int(act.n) == 1,
        f"{name}.action_space should be gym.spaces.Discrete(1): one action; got {act!r}.",
    )


def _obs(x: Any, name: str, where: str) -> float:
    assert_true(
        isinstance(x, np.ndarray) and x.shape == (1,) and x.dtype == np.float32,
        f"{name}.{where} should return a float32 array of shape (1,) as the observation; got "
        f"{type(x).__name__}"
        + (f" {x.dtype} {x.shape}" if isinstance(x, np.ndarray) else "")
        + ". Hint: np.zeros(1, np.float32) or np.array([obs], np.float32).",
    )
    return float(x[0])


def _one_step(cls: type, name: str, seed: int) -> tuple[float, float]:
    """(observation, reward) of one episode at `seed`, checking the flags and the API."""
    env = cls()
    out = env.reset(seed=seed)
    assert_true(
        isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict),
        f"{name}.reset(seed=...) should return (observation, info), info a dict.",
    )
    o = _obs(out[0], name, "reset")
    res = env.step(0)
    assert_true(
        isinstance(res, tuple) and len(res) == 5,
        f"{name}.step should return five things: observation, reward, terminated, truncated, info.",
    )
    obs2, r, term, trunc, info = res
    o2 = _obs(obs2, name, "step")
    assert_true(
        term is True or (isinstance(term, (bool, np.bool_)) and bool(term)),
        f"{name}.step should end the episode with terminated=True (one step long, a terminal "
        f"state); got terminated={term!r}.",
    )
    assert_true(
        not bool(trunc),
        f"{name}.step should return truncated=False: the episode ends at a terminal state, it is "
        "not cut by a time limit. A truncated step bootstraps, so the agent's Q would not be r.",
    )
    assert_true(isinstance(info, dict), f"{name}.step should return a dict as info.")
    assert_true(
        o2 == o,
        f"{name}.step should return the same observation as reset ({o:+g} at seed {seed}); got {o2:+g}.",
    )
    return o, float(r)


def check_constant_reward_env(ConstantRewardEnv: type) -> None:
    """Probe 1: spaces, flags and seeding; Q = 1 with the mask and 100 without, from its samples."""
    name = "ConstantRewardEnv"
    fx = expected("m09_probes")
    _probe_spaces(ConstantRewardEnv, name)
    gamma = float(fx["gamma"])
    rewards: dict[float, set[float]] = {}
    for s in range(int(fx["seeds_constant"])):
        o, r = _one_step(ConstantRewardEnv, name, s)
        rewards.setdefault(o, set()).add(r)
    assert_true(
        set(rewards) == {0.0},
        f"{name}: the observation should always be 0; saw {sorted(rewards)}.",
    )
    assert_true(
        rewards[0.0] == {float(fx["q1_masked"])},
        f"{name}: every episode should pay +1; saw rewards {sorted(rewards[0.0])}.",
    )
    q_masked = float(fx["q1_masked"])
    q_unmasked = q_masked / (1 - gamma)
    assert_true(
        np.isclose(q_unmasked, float(fx["q1_unmasked"])),
        f"{name}: an agent that ignores terminated should head for r/(1 - gamma) = "
        f"{float(fx['q1_unmasked']):g}; your rewards give {q_unmasked:g}.",
    )


def check_obs_reward_env(ObsRewardEnv: type) -> None:
    """Probe 2: spaces and flags; reward = observation in {+1, -1}; about half +1; seeded."""
    name = "ObsRewardEnv"
    fx = expected("m09_probes")
    _probe_spaces(ObsRewardEnv, name)
    seen: dict[float, set[float]] = {}
    for s in range(int(fx["seeds_obs"])):
        o, r = _one_step(ObsRewardEnv, name, s)
        seen.setdefault(o, set()).add(r)
    assert_true(
        set(seen) <= {1.0, -1.0} and len(seen) == 2,
        f"{name}: the observation should be +1 or -1, each sometimes; saw {sorted(seen)}.",
    )
    for o, rs in seen.items():
        want = float(fx["q2_plus"]) if o > 0 else float(fx["q2_minus"])
        assert_true(
            rs == {want},
            f"{name}: the reward should equal the observation, so a masked agent learns "
            f"Q({o:+g}) = {want:+g}; at observation {o:+g} it paid {sorted(rs)}.",
        )
    n = int(fx["seeds_share"])
    ups = [_obs(ObsRewardEnv().reset(seed=s)[0], name, "reset") > 0 for s in range(n)]
    share = float(np.mean(ups))
    lo, hi = float(fx["share_low"]), float(fx["share_high"])
    assert_true(
        lo < share < hi,
        f"{name}: +1 should come up with probability 1/2; over seeds 0-{n - 1} it came up "
        f"{share:.3f} of the time (expected between {lo} and {hi}).",
    )
    k = int(fx["seeds_repeat"])
    first = [float(ObsRewardEnv().reset(seed=s)[0][0]) for s in range(k)]
    again = [float(ObsRewardEnv().reset(seed=s)[0][0]) for s in range(k)]
    assert_true(
        first == again,
        f"{name}: reset(seed=s) should give the same observation every time for the same s; "
        f"seeds 0-{k - 1} gave different lists on two passes. Hint: draw from self.np_random, "
        "which super().reset(seed=seed) seeds, not from np.random.",
    )


def check_two_step_env(TwoStepEnv: type) -> None:
    """Probe 3 (stretch): obs 0 then 1, rewards 0 then 1, terminated on the second step only."""
    name = "TwoStepEnv"
    fx = expected("m09_probes")
    _probe_spaces(TwoStepEnv, name)
    env = TwoStepEnv()
    out = env.reset(seed=0)
    assert_true(
        isinstance(out, tuple) and len(out) == 2, f"{name}.reset should return (obs, info)."
    )
    o0 = _obs(out[0], name, "reset")
    o1, r1, t1, c1, _ = env.step(0)
    o2, r2, t2, c2, _ = env.step(0)
    got = (
        o0,
        _obs(o1, name, "step"),
        float(r1),
        bool(t1),
        _obs(o2, name, "step"),
        float(r2),
        bool(t2),
    )
    want = (0.0, 1.0, 0.0, False, 1.0, 1.0, True)
    assert_true(
        got == want and not bool(c1) and not bool(c2),
        f"{name}: reset gives observation 0; the first step returns observation 1, reward 0, "
        "terminated=False; the second returns observation 1, reward 1, terminated=True; "
        f"truncated is always False. Got (obs, obs, r, terminated, obs, r, terminated) = {got}.",
    )
    gamma = float(fx["gamma"])
    q1 = float(r2)
    q0 = float(r1) + gamma * q1
    assert_true(
        np.isclose(q0, float(fx["q3_first"])) and np.isclose(q1, float(fx["q3_second"])),
        f"{name}: a correct agent should learn Q(0) = gamma * 1 = {float(fx['q3_first']):g} and "
        f"Q(1) = {float(fx['q3_second']):g}; your rewards give {q0:g} and {q1:g}.",
    )


# ------------------------------------------------------------------ 5 IQM and the bootstrap
def check_iqm(iqm) -> None:
    """Equal to prl.evaluate.iqm (from the fixture) for n = 4, 5, 10, 11, 12, 37 and an outlier."""
    fx = expected("m09_iqm")
    start = 0
    for n, want in zip(fx["lengths"], fx["answers"], strict=True):
        x = fx["x"][start : start + int(n)].copy()
        start += int(n)
        got = _scalar(_call(iqm, x, what=f"iqm of {int(n)} values"), f"iqm of {int(n)} values")
        if not np.isclose(got, want, rtol=1e-9, atol=1e-12):
            k = int(n) // 4
            raise CheckFailed(
                f"iqm of {int(n)} values returned {got:.6g}; expected {want:.6g}. Hint: sort, "
                f"drop k = floor(n / 4) = {k} from each end (here {int(n) - 2 * k} remain), and "
                "return the mean of the rest."
            )


def check_bootstrap_ci(bootstrap_ci) -> None:
    """Equal to prl.evaluate.bootstrap_ci (from the fixture) on three fixed calls."""
    from ..evaluate import iqm as prl_iqm

    fx = expected("m09_bootstrap")
    x = fx["x"]
    calls = (
        ("bootstrap_ci(x, iqm)", (x.copy(), prl_iqm), {}),
        ("bootstrap_ci(x)", (x.copy(),), {}),
        (
            "bootstrap_ci(x[:7], np.median, level=0.9, n_boot=1000, seed=3)",
            (x[:7].copy(), np.median),
            dict(level=0.9, n_boot=1000, seed=3),
        ),
    )
    for (what, args, kwargs), want in zip(calls, fx["answers"], strict=True):
        out = _call(bootstrap_ci, *args, what=what, **kwargs)
        try:
            got = np.asarray([float(v) for v in out])
        except (TypeError, ValueError):
            raise CheckFailed(f"{what} should return two floats (low, high).") from None
        assert_true(
            got.shape == (2,), f"{what} should return two floats (low, high); got {len(got)}."
        )
        assert_true(
            bool(np.allclose(got, want, rtol=1e-9, atol=1e-12)),
            f"{what} returned ({got[0]:.6g}, {got[1]:.6g}); expected ({want[0]:.6g}, "
            f"{want[1]:.6g}). Hint: rng = np.random.default_rng(seed); n_boot resamples of "
            "len(x) indices drawn with replacement in one call, rng.integers(0, n, size=(n_boot, "
            "n)); then np.quantile of the statistics at (1 - level) / 2 and 1 - (1 - level) / 2.",
        )


# ------------------------------------------------------------------ 6 one live seed
def check_band_position(band_position) -> None:
    """+1 above every recorded value, -1 below every one, 0 otherwise; the ends are inside."""
    fx = expected("m09_band_position")
    for value, rec, want in zip(fx["values"], fx["recorded"], fx["answers"], strict=True):
        what = f"band_position({value:g}, {rec.tolist()})"
        got = _call(band_position, float(value), rec.copy(), what=what)
        assert_true(
            isinstance(got, (int, np.integer)) and not isinstance(got, bool),
            f"{what} should return an int (+1, 0 or -1); got {type(got).__name__}.",
        )
        if int(got) != int(want):
            hint = (
                "a value equal to the band's top or bottom is inside: only a strict inequality "
                "puts it outside."
                if value in (rec.max(), rec.min())
                else "+1 above the largest recorded value, -1 below the smallest, 0 otherwise."
            )
            raise CheckFailed(f"{what} returned {int(got)}; expected {int(want)}. Hint: {hint}")


def _live_return(*objs: Any, seed: int, steps: int, eval_every: int, eval_episodes: int) -> float:
    """Checkpoint 6.2's metric.

    Called with one value (the checkpoint, `lab.check(6, check_live_return, live_return)`), it
    is that value: the live seed's greedy return, already computed by the run cell, so the
    checkpoint never trains a second time (C38). Called with the lab's train_dqn and CONFIG
    (scripts/threshold_protocol.py, and its `random` mutant of train_dqn), it trains the
    configuration on CartPole-v1 at `seed` and returns the last greedy evaluation, the same
    quantity at the same budget.
    """
    if len(objs) == 1:
        value = objs[0]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise CheckFailed(
                f"live_return should be one number, the last value of log['eval_return']; got "
                f"{type(value).__name__}."
            )
        if not np.isfinite(float(value)):
            raise CheckFailed(f"live_return is {value}: the run's greedy return should be finite.")
        return float(value)
    train_dqn, config = objs
    import gymnasium as gym

    _, log = train_dqn(
        lambda: gym.make("CartPole-v1"),
        config,
        seed,
        steps,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        verbose=False,
    )
    return float(log["eval_return"][-1])


check_live_return = StochasticCheck(
    ex=6,
    functions=("train_dqn", "CONFIG"),
    metric=_live_return,
    live={"steps": 50_000, "eval_every": 5_000, "eval_episodes": 10},
    quick={"steps": 10_000, "eval_every": 1_000, "eval_episodes": 10},
    higher_is_better=True,
    min_gap=25.0,  # about one uniform-random policy's own mean return on CartPole
    what=(
        "your live seed's greedy return at the end of training (mean of 10 episodes): a guard "
        "on the platform, not on your code"
    ),
    thresholds={
        # threshold_protocol.py, quick budget {"eval_episodes": 10, "eval_every": 1000, "steps":
        # 10000}, N = 100 seeds (0-99), 2026-10-25, local (Apple M1 Pro, arm64). Solution p1
        # 92.54; strongest wrong version mutant 'random' p99 30.32; gap 62.22 (minimum 25);
        # threshold: the midpoint 61.433, rounded to 61.0. 0 of 100 solution seeds fail: flake
        # rate at most about 3/N = 3.0% (rule of three).
        "quick": (
            61.0,
            "threshold protocol, 100 seeds: runs/2026-10-25-local-threshold-m09-check_live_return-quick-n100-exp.json",
        ),
        # No live threshold: at the live budget (50,000 steps) the protocol refused over seeds
        # 0-99 (solution p1 26.75, below random play's p99 30.32; seeds 78 and 94 end below
        # 28.5): the last greedy evaluation of one run at 50,000 steps can collapse late. Over
        # seeds 0-19 alone it gave 69.0, which seeds 78 and 94 contradict, so it is not used.
        # Logs: _notes/m09-te/protocol_live*.log. OPEN_QUESTIONS C38.
    },
)
