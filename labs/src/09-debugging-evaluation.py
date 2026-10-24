# ---
# jupyter:
#   prl:
#     module: m09
#     pins:
#       gymnasium: "1.4.0"
# ---

# %% [markdown]
# A teammate's DQN on CartPole has stopped learning. In this lab you read the recorded logs of
# five runs of Module 8's reference DQN at the same budget: one healthy run and four broken
# ones, A to D, each recorded on 3 seeds. You write four signals and try them on the healthy
# run, then diagnose the four broken runs from all four signals at once. You write two probe
# environments and run every agent on them, to test the diagnoses. Last, you report a recorded
# learning-rate sweep with the interquartile mean (IQM) and bootstrap intervals, and place one
# live seed of your own against its band.
#
# The briefing's equations are named in brackets, for example (TD error) or (IQM). Its table
# "Equations and where you implement them" gives each name's equation number.

# %%
import collections
import dataclasses
import textwrap
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from prl.evaluate import Budget, compare

# Budgets (live, then QUICK). Lab code reads them from here and never branches on QUICK.
# `steps`, `log_every`, `eval_every` and `eval_episodes` are Exercise 8's live run and pick its
# recorded band; the recorded runs of Exercises 1-5 and 7 used the live values. `probe_steps`
# is each probe run of Exercise 6.
budget = lab.use_settings(
    prl.runtime.settings(
        dict(steps=50_000, log_every=500, eval_every=5_000, eval_episodes=10, probe_steps=2_000),
        dict(steps=10_000, log_every=500, eval_every=1_000, eval_episodes=10, probe_steps=2_000),
        designed="colab-cpu",
    )
)
# Your seat's seed, clear of the recorded seeds, the figures' and the checkpoints' (0-99).
RUN_SEED = (SEED + 1) * 1000
_rt = prl.runtime.detect()
HARDWARE = f"{_rt.platform}, {_rt.cpu}, {_rt.n_cpu} CPU"
torch.set_num_threads(min(2, _rt.n_cpu))
print(f"Budget: {budget.steps:,} env steps for your live run; {budget.probe_steps:,} per probe.")

# %% [markdown]
# ### The reference version of your Lab 8 code
#
# The cell below is Module 8's DQN: its configuration `CONFIG`, the masked target `td_target`,
# `gather_q`, which reads Q(s, a; w) for the action taken, and the training loop `train_dqn`,
# which returns the network and a log. Every run in this lab, recorded or live, uses this code
# and this configuration.

# %%
# PROVISIONAL (Module 9, first pass). Replaced by Module 8's reference files, included with
# "# %% include labs/shared/<name>.py", once they exist. Written to the names of Module 8's
# draft (2026-10-24) and _notes/m09-needs.md; Module 9 relies only on train_dqn's signature,
# CONFIG's fields lr and target_sync, the global names td_target and gather_q, and the log keys.
Batch = collections.namedtuple("Batch", "obs act rew next_obs terminated truncated")
PROBE_SEED = 1_000_000  # probe states: reset(seed=PROBE_SEED + i)
EVAL_SEED = 1_100_000  # evaluation episodes: reset(seed=EVAL_SEED + i)


@dataclasses.dataclass(frozen=True)
class DQNConfig:
    """Provided: the DQN configuration every lab uses."""

    hidden: tuple = (64, 64)
    lr: float = 1e-3
    buffer_size: int = 10_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    target_sync: int | None = 500  # env steps between copies w -> w-; None: never
    learning_starts: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2  # epsilon falls linearly over this share of the run
    max_grad_norm: float = 10.0
    huber_threshold: float = 1.0


CONFIG = DQNConfig()


def td_target(batch, q_next, gamma):
    """Provided: r + gamma (1 - terminated) q_next (masked target)."""
    return batch.rew + gamma * (1.0 - batch.terminated) * q_next


def gather_q(q_values, actions):
    """Provided: Q(s_i, a_i; w) for every row i of the batch."""
    return q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)


def dqn_target(batch, q_net, target_net, gamma):
    """Provided: td_target with the target network's max_a' Q(s', a'; w-)."""
    with torch.no_grad():
        q_next = target_net(batch.next_obs).max(dim=1).values
        return td_target(batch, q_next, gamma)


def make_q_net(obs_dim, n_actions, config):
    """Provided: the Q-network, ReLU layers of config.hidden widths."""
    layers, width = [], obs_dim
    for h in config.hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.ReLU()]
        width = h
    return torch.nn.Sequential(*layers, torch.nn.Linear(width, n_actions))


def probe_states(make_env, n=32):
    """Provided: n fixed start states, the same for every run: reset(seed=PROBE_SEED + i)."""
    env = make_env()
    return np.asarray([env.reset(seed=PROBE_SEED + i)[0] for i in range(n)], np.float32)


def greedy_return(q_net, make_env, episodes):
    """Provided: the mean undiscounted return of greedy episodes from fixed start states."""
    env, total = make_env(), 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + i)
        done = False
        while not done:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax())
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
    return total / episodes


def train_dqn(
    make_env,
    config,
    seed,
    steps,
    *,
    target_fn=None,
    log_every=1000,
    eval_every=None,
    eval_episodes=10,
    probe_obs=None,
    snapshot_steps=(),
    verbose=True,
):
    """Provided: DQN with replay, a target network and the Huber loss, for `steps` env steps.
    Returns (q_net, log); the log holds NumPy arrays (Exercise 1 lists them)."""
    target_fn = target_fn or dqn_target
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env()
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    probe = torch.as_tensor(probe_states(make_env) if probe_obs is None else probe_obs)
    q_net = make_q_net(obs_dim, n_act, config)
    target_net = make_q_net(obs_dim, n_act, config)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    cap = min(config.buffer_size, steps)
    buf = {k: np.zeros((cap, obs_dim), np.float32) for k in ("obs", "next_obs")}
    buf.update({k: np.zeros(cap, np.float32) for k in ("rew", "terminated", "truncated")})
    buf["act"] = np.zeros(cap, np.int64)
    log = {k: [] for k in _LOG_KEYS}
    acc = {k: [] for k in ("loss", "td", "td_max", "q_max", "gn", "q_term")}
    pending, last_sync, ep = sorted(snapshot_steps), 0, [0.0, 0.0, 0]
    obs, _ = env.reset(seed=seed)
    start = shown = time.perf_counter()
    for t in range(1, steps + 1):
        frac = min(1.0, (t - 1) / max(1.0, config.eps_fraction * steps))
        eps = config.eps_start + frac * (config.eps_end - config.eps_start)
        if rng.random() < eps:
            a = int(rng.integers(n_act))
        else:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax())
        obs2, r, term, trunc, _ = env.step(a)
        slot = (t - 1) % cap
        for k, x in zip(_BUF_KEYS, (obs, a, r, obs2, float(term), float(trunc)), strict=True):
            buf[k][slot] = x
        ep = [ep[0] + r, ep[1] + config.gamma ** ep[2] * r, ep[2] + 1]
        obs = obs2
        if term or trunc:
            for k, x in zip(_EP_KEYS, (ep[0], ep[1], ep[2], t, bool(term)), strict=True):
                log[k].append(x)
            obs, _ = env.reset()
            ep = [0.0, 0.0, 0]
        if t > config.learning_starts and t % config.train_freq == 0:
            idx = rng.integers(0, min(t, cap), config.batch_size)
            batch = Batch(*(torch.as_tensor(buf[k][idx]) for k in _BUF_KEYS))
            q_values = q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)
            y = target_fn(batch, q_net, target_net, config.gamma)
            loss = torch.nn.functional.huber_loss(q_sa, y, delta=config.huber_threshold)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(q_net.parameters(), config.max_grad_norm)
            opt.step()
            td = (y - q_sa).detach().abs()
            done = batch.terminated > 0.5
            q_max = q_values.detach().max(1).values.mean()
            for k, x in (("loss", loss.detach()), ("td", td.mean()), ("td_max", td.max())):
                acc[k].append(float(x))
            acc["q_max"].append(float(q_max))
            acc["gn"].append(float(gn))
            if done.any():
                acc["q_term"].append(float(q_sa.detach()[done].mean()))
            while pending and t >= pending[0]:
                pending.pop(0)
                values = (t, q_values, batch.act, q_sa, y, batch.rew, done)
                for k, x in zip(_SNAP_KEYS, values, strict=True):
                    log[k].append(x.detach().numpy().copy() if torch.is_tensor(x) else x)
        if config.target_sync and t % config.target_sync == 0:
            target_net.load_state_dict(q_net.state_dict())
            last_sync = t
        if t % log_every == 0:
            with torch.no_grad():
                log["q_probe"].append(q_net(probe).numpy())
                log["q_probe_target"].append(target_net(probe).numpy())
            for k, src, f in _LOG_REDUCE:
                log[k].append(float(f(acc[src])) if acc[src] else np.nan)
            for k, x in (("step", t), ("epsilon", eps), ("steps_since_sync", t - last_sync)):
                log[k].append(x)
            for v in acc.values():
                v.clear()
        if eval_every and t % eval_every == 0:
            log["eval_step"].append(t)
            log["eval_return"].append(greedy_return(q_net, make_env, eval_episodes))
        if verbose and time.perf_counter() - shown > 10:
            shown = time.perf_counter()
            print(f"  step {t:,} of {steps:,}", flush=True)
    out = {k: np.asarray(v) for k, v in log.items()}
    out["probe_obs"] = probe.numpy()
    out["seconds"] = time.perf_counter() - start
    out["config"] = dataclasses.asdict(config)
    return q_net, out


_BUF_KEYS = ("obs", "act", "rew", "next_obs", "terminated", "truncated")
_EP_KEYS = ("ep_return", "ep_disc_return", "ep_length", "ep_end_step", "ep_terminated")
_SNAP_KEYS = (
    "snap_step",
    "snap_q",
    "snap_actions",
    "snap_gathered",
    "snap_target",
    "snap_reward",
    "snap_terminated",
)
_LOG_REDUCE = (
    ("loss", "loss", np.mean),
    ("td_abs_mean", "td", np.mean),
    ("td_abs_max", "td_max", np.max),
    ("q_max_mean", "q_max", np.mean),
    ("grad_norm", "gn", np.mean),
    ("q_terminal", "q_term", np.mean),
)
_LOG_KEYS = (
    ("step", "epsilon", "steps_since_sync", "q_probe", "q_probe_target")
    + tuple(k for k, _, _ in _LOG_REDUCE)
    + _EP_KEYS
    + _SNAP_KEYS
    + ("eval_step", "eval_return")
)


# %% [markdown]
# ### Provided: the recorded runs, one run's log, and the plots
#
# `prl.data.load("m09_broken_runs")` holds the healthy run and runs A to D, each on seeds 0, 1
# and 2, at the live budget. `run_log(run, seed)` returns one run's log as a dict, with the
# keys `train_dqn` writes. `dashboard(runs)` draws the return, the loss, the mean |δ| and Q at
# the start states for the runs you name, every seed as its own line.


# %%
RUNS = prl.data.load("m09_broken_runs")
GAMMA = float(RUNS["gamma"])
G_MAX = sum(GAMMA**t for t in range(500))  # the most a 500-step CartPole episode can earn


def run_log(run, seed):
    """Provided: the log of one recorded run ("healthy", "A" ... "D") and seed (0, 1, 2)."""
    s = int(np.flatnonzero(RUNS["seeds"] == seed)[0])
    log = {"step": RUNS["step"], "probe_obs": RUNS["probe_obs"]}
    for key in RUNS:
        if key.startswith(f"{run}/"):
            name = key.split("/", 1)[1]
            arr = RUNS[key]
            if name.startswith("ep_") and name != "ep_seed":
                log[name] = arr[RUNS[f"{run}/ep_seed"] == seed]
            elif name != "ep_seed":
                log[name] = arr[s]
    return log


def smooth_returns(log, every=2500):
    """Provided: the mean return of the episodes that ended in each `every`-step window."""
    edges = np.arange(every, log["step"][-1] + 1, every)
    end, ret = log["ep_end_step"], log["ep_return"]
    means = [ret[(end > e - every) & (end <= e)].mean() for e in edges]
    return edges, np.array(means)


def q_start(log):
    """Provided: Q at the start states at every log point, the mean over them of max_a Q."""
    return log["q_probe"].max(axis=-1).mean(axis=-1).astype(float)


# The healthy run is the reference, drawn as reference lines are; A-D differ by line style too.
RUN_STYLE = {
    "healthy": dict(color=prl.plot.NEUTRAL_LIGHT, ls="-"),
    "A": dict(color=prl.plot.SERIES[0], ls=(0, (4.5, 1.8))),
    "B": dict(color=prl.plot.SERIES[1], ls=(0, (1, 1.6))),
    "C": dict(color=prl.plot.SERIES[2], ls=(0, (6, 1.8, 1.4, 1.8))),
    "D": dict(color=prl.plot.SERIES[3], ls=(0, (3, 1.2, 1, 1.2))),
}


def dashboard(runs, title, panels=("return", "loss", "td", "q")):
    """Provided: the chosen panels for the runs you name, every seed as its own line."""
    draw = {
        "return": ("return (episodes per 2,500 steps)", "linear", smooth_returns),
        "loss": ("mean loss", "log", lambda g: (g["step"], g["loss"])),
        "td": ("mean |δ|", "log", lambda g: (g["step"], g["td_abs_mean"])),
        "q": ("Q at the start states", "symlog", lambda g: (g["step"], q_start(g))),
    }
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, panel in zip(np.atleast_1d(axes), panels, strict=True):
        ylabel, scale, xy = draw[panel]
        for run in runs:
            for k, seed in enumerate(RUNS["seeds"]):
                name = "healthy" if run == "healthy" else f"run {run}"
                kw = dict(lw=1.2, alpha=0.85, label=name if k == 0 else None)
                ax.plot(*xy(run_log(run, int(seed))), **kw, **RUN_STYLE[run])
        if panel == "q":
            ax.axhline(G_MAX, color=prl.plot.NEUTRAL_LIGHT, ls=":", lw=1, label="99.34")
        ax.set(ylabel=ylabel, yscale=scale, xlabel="env step")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle(title)
    plt.show()


def second_half_median(x):
    """Provided: the median of a signal over the second half of the run."""
    return float(np.nanmedian(np.asarray(x)[len(x) // 2 :]))


def last_five(x):
    """Provided: the mean of a signal over its last five log points."""
    return float(np.nanmean(np.asarray(x)[-5:]))


# %% [markdown] role="ex1-head"
# **Two exact checks.** Some properties of a correct DQN hold exactly, so a log can check them.
# In each logged batch, the value the loss used for row i is Q(s_i, a_i; w), read from the same
# forward pass (TD error). And the target network's values change at every sync. At three
# steps the log kept one batch: `snap_q`, Q(s, ·; w) for every row (shape (S, B, |𝒜|), here
# S = 3 snapshots of B = 32 rows and |𝒜| = 2 actions), `snap_actions` (S, B) and
# `snap_gathered` (S, B), the value the loss used. Every 500 steps, when the target is also
# synced, the log kept the target network's Q at 8 fixed start states, `q_probe_target`
# (shape (L, 8, 2)). Write `gather_check` and `target_lag`. The run cell tries them on the
# healthy run only, every seed.

# %% [markdown] role="ex1-predict"
# On the healthy run, will `gather_check` be about 0.01, exactly 0.0, or a value that changes
# with the seed? What will `target_lag` be at the last log point of the healthy run, and of a
# run whose target never syncs (log points from step 500 to 50,000)?


# %% role="ex1-stub"
def gather_check(log):
    """The share of snapshot rows whose log["snap_gathered"] value is not Q(s_i, a_i), the
    entry of log["snap_q"] for the row's action log["snap_actions"]. Compare with
    np.isclose(..., rtol=1e-5, atol=1e-6). Returns one float in [0, 1]."""
    # TODO 1 (part 1): pick Q(s_i, a_i) for every row, then compare.
    raise NotImplementedError("TODO 1: write gather_check, or run lab.use_reference(1) to continue")


def target_lag(log):
    """Env steps since the target network's probe values last changed, at every log point:
    lag[0] = 0; for i >= 1, lag[i] = step[i] - step[j], where j <= i is the latest log point
    at which log["q_probe_target"][j] differs from [j - 1] in any entry (j = 0 if none).
    Returns an int array of shape (L,)."""
    # TODO 1 (part 2): one loop over the log points, remembering the step of the last change.
    raise NotImplementedError("TODO 1: write target_lag, or run lab.use_reference(1) to continue")


# %% [markdown] role="ex1-hint"
# `np.take_along_axis` picks one entry per row along an axis; its index array needs as many
# dimensions as `snap_q`. The mean of a boolean array is the share of True. In `target_lag`,
# keep `last`, the step of the latest change, starting at `step[0]`; `np.any(a != b)` is True
# if any entry differs.


# %% role="ex1-sol"
@lab.solution(1)
def gather_check(log):
    """Share of batch rows whose gathered value is not Q(s_i, a_i)."""
    q, a = log["snap_q"], log["snap_actions"]
    want = np.take_along_axis(q, a[..., None], axis=-1)[..., 0]
    wrong = ~np.isclose(log["snap_gathered"], want, rtol=1e-5, atol=1e-6)
    return float(wrong.mean())


@lab.solution(1)
def target_lag(log):
    """Env steps since the target network's values last changed."""
    q_tgt, step = log["q_probe_target"], log["step"]
    lag = np.zeros(len(step), dtype=np.int64)
    last = step[0]
    for i in range(1, len(step)):
        if np.any(q_tgt[i] != q_tgt[i - 1]):
            last = step[i]
        lag[i] = step[i] - last
    return lag


# %% role="ex1-run"
dashboard(["healthy"], "The healthy run, 3 seeds")
for seed in RUNS["seeds"]:
    log = run_log("healthy", int(seed))
    lag = target_lag(log)
    print(
        f"healthy, seed {seed}: gather_check {gather_check(log)}; target_lag at the last log "
        f"point {lag[-1]}, largest {lag.max()} (at step {log['step'][np.argmax(lag)]:,})"
    )

# %% [markdown] role="ex1-explain"
# Why is the healthy gather check exactly 0.0 and not merely small, and what would a value near
# 1 say about the code (TD error)? The largest lag comes early in the run: before the first
# gradient step, what does a sync copy? What would a `target_lag` that grows with the step say?

# %% role="ex1-chk1"
lab.check(1, m09.check_gather_check, gather_check, label="1.1")

# %% role="ex1-chk2"
lab.check(1, m09.check_target_lag, target_lag, label="1.2")

# %% [markdown] role="ex2-head"
# **Two scale signals.** Compare what the network believes with what the agent earns: at every
# log point, Q at the start states (the mean over the 8 probe states of max_a Q(s, a; w)) minus
# the mean discounted return of the episodes that ended in the last `window` env steps
# (Q drift). And the TD error against the scale of Q: the mean |δ| divided by the mean over the
# batch of max_a Q(s, a; w) (relative TD error). Write `q_drift` and `td_error_scale`. The run
# cell tries them on the healthy run, every seed.

# %% [markdown] role="ex2-predict"
# CartPole pays 1 per step and stops at 500 steps. At γ = 0.99, what is the largest discounted
# return any episode can earn from its first state? On the healthy run, will the relative TD
# error in the second half be about 0.001, 0.01 or 0.1?


# %% role="ex2-stub"
def q_drift(log, window=5000):
    """Q drift at every log point i: q_bar[i] minus the mean of log["ep_disc_return"] over the
    episodes with t - window < log["ep_end_step"] <= t, where t = log["step"][i] and q_bar =
    log["q_probe"].max(axis=-1).mean(axis=-1). NaN where no episode ended in the window.
    Returns an array of shape (L,)."""
    # TODO 2 (part 1): q_bar first; then a loop over the log points.
    raise NotImplementedError("TODO 2: write q_drift, or run lab.use_reference(2) to continue")


def td_error_scale(log):
    """The relative TD error at every log point: log["td_abs_mean"] / |log["q_max_mean"]|,
    both of shape (L,). Returns an array of shape (L,)."""
    # TODO 2 (part 2): one line.
    raise NotImplementedError(
        "TODO 2: write td_error_scale, or run lab.use_reference(2) to continue"
    )


# %% [markdown] role="ex2-hint"
# `log["q_probe"]` has shape (L, 8, 2): a max over one axis and a mean over the next leave one
# number per log point. Start the output with `np.full(L, np.nan)`, and pick the recent
# episodes with a boolean mask on `log["ep_end_step"]`. NumPy divides arrays element by element.


# %% role="ex2-sol"
@lab.solution(2)
def q_drift(log, window=5000):
    """Q at the probe states minus recent discounted returns."""
    q_bar = log["q_probe"].max(axis=-1).mean(axis=-1)  # (L,)
    end, g0 = log["ep_end_step"], log["ep_disc_return"]
    drift = np.full(len(log["step"]), np.nan)
    for i, t in enumerate(log["step"]):
        recent = (end > t - window) & (end <= t)
        if recent.any():
            drift[i] = q_bar[i] - g0[recent].mean()  # (Q drift)
    return drift


@lab.solution(2)
def td_error_scale(log):
    """Mean |delta| over the batch Q scale, per log point."""
    q_scale = np.abs(log["q_max_mean"])
    return log["td_abs_mean"] / q_scale  # (relative TD error)


# %% role="ex2-run"
print(f"The most any CartPole episode can earn from its first state at γ = {GAMMA}: {G_MAX:.2f}")
fig, (ax_q, ax_r) = plt.subplots(1, 2, figsize=(10, 3.6))
for k, seed in enumerate(RUNS["seeds"]):
    log = run_log("healthy", int(seed))
    drift, scale = q_drift(log), td_error_scale(log)
    print(
        f"healthy, seed {seed}: Q drift (last 5 log points) {last_five(drift):6.2f}; "
        f"relative TD error (second half, median) {second_half_median(scale):.4f}"
    )
    label = "healthy" if k == 0 else None
    ax_q.plot(log["step"], q_start(log), label=label, **RUN_STYLE["healthy"])
    ax_q.plot(
        log["step"],
        q_start(log) - drift,
        ls="--",
        color=prl.plot.SERIES[0],
        label="discounted return earned" if k == 0 else None,
    )
    ax_r.plot(log["step"], scale, **RUN_STYLE["healthy"])
ax_q.axhline(G_MAX, color=prl.plot.NEUTRAL_LIGHT, ls=":", lw=1, label=f"{G_MAX:.2f}")
ax_q.set(xlabel="env step", ylabel="Q at the start states", title="Q against the return earned")
ax_r.set(xlabel="env step", ylabel="mean |δ| / mean max Q", yscale="log", title="Relative TD error")
ax_q.legend()
plt.show()

# %% [markdown] role="ex2-explain"
# On the healthy run, is Q at the start states above or below the discounted return earned,
# and how close does either come to 99.34 (Q drift)? How much does the relative TD error vary
# from seed to seed? Why is it unchanged when every reward is 10 times larger?

# %% role="ex2-chk1"
lab.check(2, m09.check_q_drift, q_drift, label="2.1")

# %% role="ex2-chk2"
lab.check(2, m09.check_td_error_scale, td_error_scale, label="2.2")

# %% [markdown] role="ex3-head"
# **Broken runs A to D.** The teammate's four runs. Each has exactly one of these causes:
#
# | Key | Cause |
# |---|---|
# | `"mask"` | The terminal mask is missing: targets bootstrap past the end of an episode. |
# | `"sync"` | The target network is never updated from the online network. |
# | `"lr"` | The learning rate is far too large. |
# | `"gather"` | The loss reads the wrong entries of Q(s, ·), not Q(s, a) for the action taken. |
# | `"reward"` | The rewards are scaled up: every reward is many times CartPole's 1. |
# | `"seed"` | Nothing is wrong with the code or the configuration: an unlucky seed. |
#
# The run cell computes your four signals on every run and seed, prints the rewards the logged
# batches held, and draws the return and Q panels of all five runs. Write your diagnoses in
# `diagnoses`, with the evidence for each as a comment.

# %% [markdown] role="ex3-predict"
# Before you run the cell: for a run whose terminal mask is missing, which of your four signals
# should leave the healthy range, and which way? For a run whose learning rate is far too
# large?


# %% role="ex3-stub"
def diagnoses():
    """Your diagnosis of each broken run: a dict with the keys "A", "B", "C" and "D", each
    one key of the table above ("mask", "sync", "lr", "gather", "reward" or "seed")."""
    # TODO 3: return {"A": ..., "B": ..., "C": ..., "D": ...}, the evidence as comments.
    raise NotImplementedError("TODO 3: write diagnoses, or run lab.use_reference(3) to continue")


# %% [markdown] role="ex3-hint"
# Start with the exact checks: a check that fails names its cause. For a run that passes both,
# is its Q above any return an episode can earn? 'seed': how many seeds failed? 'reward': which
# rewards did the logged batches hold? What is left is a setting, not a bug.


# %% role="ex3-sol"
@lab.solution(3)
def diagnoses():
    """What made runs A to D (see How runs A to D were made)."""
    return {
        "A": "lr",  # exact checks pass, Q in range, rewards 1, 3 of 3 fail
        "B": "gather",  # the gather check fails
        "C": "mask",  # Q far above 99.34 with rewards of 1
        "D": "sync",  # the target never changes
    }


# %% role="ex3-run"
def short(v):
    """Provided: a number in at most 7 characters."""
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.3g}"


SUMMARY = {  # column -> (your signal, how one seed's curve is summarized)
    "gather_check": (gather_check, float),
    "target_lag, last": (target_lag, lambda lag: float(lag[-1])),
    "Q drift, last 5": (q_drift, last_five),
    "rel. TD error, 2nd half": (td_error_scale, second_half_median),
}
values = {
    name: {run: [summ(fn(run_log(run, int(s)))) for s in RUNS["seeds"]] for run in RUN_STYLE}
    for name, (fn, summ) in SUMMARY.items()
}
healthy_td = np.median(values["rel. TD error, 2nd half"]["healthy"])
print(f"Each cell: the range over seeds {RUNS['seeds'].tolist()}. The relative TD error also as")
print("a multiple of the healthy median.")
WIDTH = {"gather_check": 15, "target_lag, last": 17, "Q drift, last 5": 17}
print(f"{'run':8s}" + "".join(f"{name:>{WIDTH.get(name, 26)}s}" for name in SUMMARY))
for run in RUN_STYLE:
    row = []
    for name in SUMMARY:
        lo, hi = min(values[name][run]), max(values[name][run])
        cell = short(lo) if short(lo) == short(hi) else f"{short(lo)} to {short(hi)}"
        if name == "rel. TD error, 2nd half":
            cell += f" ({hi / healthy_td:.1f}x)"
        row.append(f"{cell:>{WIDTH.get(name, 26)}s}")
    print(f"{run:8s}" + "".join(row))
print(f"Other evidence (Q at the start states can reach about {G_MAX:.2f} at most):")
for run in RUN_STYLE:
    logs = [run_log(run, int(s)) for s in RUNS["seeds"]]
    seen = np.unique(np.concatenate([g["snap_reward"].ravel() for g in logs]))
    q_end = [q_start(g)[-1] for g in logs]
    final = [smooth_returns(g)[1][-1] for g in logs]
    print(
        f"  {run:8s} rewards in the batches {seen.tolist()}; Q at start, end "
        f"{short(min(q_end))} to {short(max(q_end))}; return, end {[round(x) for x in final]}"
    )
dashboard(list(RUN_STYLE), "All five runs, 3 seeds each", panels=("return", "q"))
print(f"Your diagnoses: {diagnoses()}")

# %% [markdown] role="ex3-explain"
# Run B fails the gather check and is out of range on two more signals: why does one wrong
# gather move them too? Which signals single out one cause, and which flag several? For run A,
# which evidence rules out each of the other five causes? (Gotcha: "An out-of-range signal
# names the cause.")

# %% role="ex3-chk1"
lab.check(3, m09.check_diagnosis, diagnoses, "A", label="3.1")

# %% role="ex3-chk2"
lab.check(3, m09.check_diagnosis, diagnoses, "B", label="3.2")

# %% role="ex3-chk3"
lab.check(3, m09.check_diagnosis, diagnoses, "C", label="3.3")

# %% role="ex3-chk4"
lab.check(3, m09.check_diagnosis, diagnoses, "D", label="3.4")

# %% [markdown]
# ### How runs A to D were made
#
# This cell shows how the teammate made runs A to D: a configuration change, or a function
# swapped into `train_dqn`, never an edit of the shared code. Run it after checkpoints 3.1-3.4.


# %%
def td_target_no_mask(batch, q_next, gamma):
    """Run C's td_target: it ignores terminated."""
    return batch.rew + gamma * q_next


def gather_q_dim0(q_values, actions):
    """Run B's gather_q: dimension 0 instead of 1."""
    return q_values.gather(0, actions.long().unsqueeze(1)).squeeze(1)


AGENTS = {  # run -> (configuration changes, functions swapped in)
    "healthy": ({}, {}),
    "A": (dict(lr=CONFIG.lr * 100), {}),
    "B": ({}, dict(gather_q=gather_q_dim0)),
    "C": ({}, dict(td_target=td_target_no_mask)),
    "D": (dict(target_sync=None), {}),
}


def train_agent(run, make_env, config, seed, steps, **kwargs):
    """Provided: train_dqn as run `run` was made; the reference functions come back after."""
    changes, swaps = AGENTS[run]
    saved = {name: globals()[name] for name in swaps}
    globals().update(swaps)
    try:
        return train_dqn(make_env, dataclasses.replace(config, **changes), seed, steps, **kwargs)
    finally:
        globals().update(saved)


# %% [markdown] role="ex4-head"
# **Probe environments.** A diagnosis from logs is a hypothesis. A probe environment tests one
# piece of an agent where the right answer is known (Jones 2021). Write probe 1,
# `ConstantRewardEnv` (reward +1 every episode), and probe 2, `ObsRewardEnv` (reward equal to
# a random observation of +1 or −1). The run cell trains the healthy agent and the agents of
# runs A to D on both.

# %% [markdown] role="ex4-predict"
# Probe 1 pays 1 and ends after one step. At γ = 0.99, what Q should a correct agent learn, and
# what Q would an agent that ignores `terminated` converge to? The probe run syncs the target
# every 100 steps for 2,000 steps: about what Q will run C show at the end? Which runs will
# fail probe 2?


# %% role="ex4-stub"
class ConstantRewardEnv(gym.Env):
    """Probe 1 (Jones): one action; the observation is always 0; every episode is one step
    long, pays +1 and ends with terminated=True. Spaces: a float32 Box of shape (1,) on
    [-1, 1], and Discrete(1). reset(*, seed=None, options=None) passes seed to super().reset()
    and returns (observation, info)."""

    # TODO 4 (part 1): __init__ sets the two spaces; then reset and step.
    def __init__(self):
        raise NotImplementedError(
            "TODO 4: write ConstantRewardEnv, or run lab.use_reference(4) to continue"
        )


class ObsRewardEnv(gym.Env):
    """Probe 2 (Jones): the same spaces; the observation is +1 or -1 with equal probability,
    drawn from self.np_random at reset; the reward equals it; every episode is one step long
    and ends with terminated=True; step returns the same observation."""

    # TODO 4 (part 2): keep the observation on self between reset and step.
    def __init__(self):
        raise NotImplementedError(
            "TODO 4: write ObsRewardEnv, or run lab.use_reference(4) to continue"
        )


# %% [markdown] role="ex4-hint"
# `super().reset(seed=seed)` seeds `self.np_random`, Gymnasium's generator for the
# environment, so the same seed gives the same observations. Return float32 arrays of shape
# (1,), not Python floats: the agent stacks them into a batch. `step` returns five things:
# observation, reward, terminated, truncated, info.


# %% role="ex4-sol"
@lab.solution(4)
class ConstantRewardEnv(gym.Env):
    """Probe 1: observation 0, one step, reward +1."""

    def __init__(self):
        box = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = box
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 1.0, True, False, {}


@lab.solution(4)
class ObsRewardEnv(gym.Env):
    """Probe 2: observation +1 or -1, one step, reward = observation."""

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
        return np.array([self.obs], np.float32), self.obs, True, False, {}


# %% role="ex4-run"
PROBE_CONFIG = dataclasses.replace(
    CONFIG, learning_starts=64, target_sync=100, train_freq=1, eps_fraction=0.0
)
probes = {
    "probe 1": (ConstantRewardEnv, np.array([[0.0]], np.float32)),
    "probe 2": (ObsRewardEnv, np.array([[-1.0], [1.0]], np.float32)),
}
probe_q = {}
start = time.perf_counter()
print(f"Each agent on each probe: {budget.probe_steps:,} steps, target sync every 100.")
for run in AGENTS:
    cells = []
    for name, (env_cls, obs) in probes.items():
        q_net, _ = train_agent(
            run,
            env_cls,
            PROBE_CONFIG,
            RUN_SEED,
            budget.probe_steps,
            log_every=100,
            probe_obs=obs,
            verbose=False,
        )
        with torch.no_grad():
            got = q_net(torch.as_tensor(obs))[:, 0].numpy()
        probe_q[(run, name)] = got
        shown = ", ".join(f"Q({o[0]:+.0f}) = {g:8.3f}" for o, g in zip(obs, got, strict=True))
        cells.append(f"{name}: {shown}")
    print(f"  {run:8s} " + " | ".join(cells), flush=True)
print("Exact answers: probe 1, Q(+0) = 1; probe 2, Q(-1) = -1 and Q(+1) = +1.")
print(f"{len(AGENTS) * len(probes)} probe runs in {time.perf_counter() - start:.0f} s.")
lab.metric("probe_q_c", round(float(probe_q[("C", "probe 1")][0]), 2))  # room code

# %% [markdown] role="ex4-explain"
# Run C's Q on probe 1 is far from both 1 and 100: count the target syncs in the run and use
# (masked target) to explain its value. Which runs passed both probes, and what does a pass
# rule out, and what not? (Gotcha: "The probes pass, so the agent is correct.")
#
# **Diagnostics for this lab.** Log more than the return and the loss: the mean and largest
# |δ|, Q at fixed start states from both networks, the discounted return of every episode,
# and now and then one whole batch. Check the exact invariants first: the gathered value is
# Q(s_i, a_i) on every row, and the target's values change at every sync. Then compare Q at the
# start states with the return earned, and the relative TD error with the healthy seeds'
# range. Then test what is left on probes: a correct agent gets them right to a few decimals.

# %% role="ex4-chk1"
lab.check(4, m09.check_constant_reward_env, ConstantRewardEnv, label="4.1")

# %% role="ex4-chk2"
lab.check(4, m09.check_obs_reward_env, ObsRewardEnv, label="4.2")

# %% [markdown] role="ex5-head"
# **A recorded sweep.** The teammate's next question: is the reference learning rate a good
# one? `prl.data.load("m09_sweep")` holds 5 learning rates, 0.1, 0.3, 1, 3 and 10 times the
# reference, each run on seeds 0 to 9 at the live budget, and each run's greedy return at the
# end (`final_return`, shape (5, 10)). Write `iqm(x)`, the interquartile mean (IQM), and
# `bootstrap_ci(x, statistic, ...)`, the percentile bootstrap interval (bootstrap CI), exactly
# as described: the checkpoints compare them with `prl.evaluate.iqm` and
# `prl.evaluate.bootstrap_ci` on the same inputs.

# %% [markdown] role="ex5-predict"
# Will the reference (×1) have the highest IQM of the five? Will the tie rule separate it from
# ×0.3 and ×3? If you had run only seed 0 of each learning rate, would you rank the five the
# same way as by their IQMs over 10 seeds?


# %% role="ex5-stub"
def iqm(x):
    """The interquartile mean (IQM) of the values in x: sort them, drop k = floor(n / 4) from
    each end (n = len(x)), and return the mean of the rest as a float. n = 10 drops 2 from each
    end and averages 6."""
    # TODO 5 (part 1): three lines.
    raise NotImplementedError("TODO 5: write iqm, or run lab.use_reference(5) to continue")


def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):
    """Percentile bootstrap interval for statistic(x) (bootstrap CI): n_boot resamples of x,
    each n values drawn with replacement using np.random.default_rng(seed); statistic of each;
    returns the (1 - level) / 2 and 1 - (1 - level) / 2 quantiles (np.quantile) as two
    floats."""
    # TODO 5 (part 2): about five lines.
    raise NotImplementedError("TODO 5: write bootstrap_ci, or run lab.use_reference(5) to continue")


# %% [markdown] role="ex5-hint"
# `np.sort(x)[k : n - k]` keeps the middle values. In `bootstrap_ci`, the indices must come
# from `np.random.default_rng(seed)`, the generator `prl.evaluate` uses, so that the same seed
# gives the same resamples and the checkpoint can compare the exact interval. `x[row]` with an
# array of indices is one resample: n values drawn with replacement.


# %% role="ex5-sol"
@lab.solution(5)
def iqm(x):
    """Mean of the middle values after dropping floor(n/4) per end."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    k = len(x) // 4
    return float(x[k : len(x) - k].mean())  # (IQM)


@lab.solution(5)
def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):
    """Percentile bootstrap interval for statistic(x)."""
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([statistic(x[row]) for row in idx])
    tails = [(1 - level) / 2, 1 - (1 - level) / 2]
    lo, hi = np.quantile(boots, tails)  # (bootstrap CI)
    return float(lo), float(hi)


# %% role="ex5-run"
SWEEP = prl.data.load("m09_sweep")
factors, final = SWEEP["lr_factor"], SWEEP["final_return"]
ref = int(np.flatnonzero(factors == 1.0)[0])
summary = []
print(f"Greedy return at {int(SWEEP['budget_steps']):,} steps, {final.shape[1]} seeds per rate:")
for f, x in zip(factors, final, strict=True):
    lo, hi = bootstrap_ci(x, iqm)
    summary.append((iqm(x), lo, hi))
    print(f"  lr x{f:<4g} IQM {iqm(x):6.1f} [{lo:6.1f}, {hi:6.1f}]  mean {x.mean():6.1f}")
    print(f"           seeds: {' '.join(f'{v:4.0f}' for v in x)}")
for i, f in enumerate(factors):
    if i != ref:
        v = compare(final[i], final[ref], names=(f"lr x{f:g}", "lr x1"))
        print(textwrap.fill(v.sentence.split(" (IQM")[0], 98, initial_indent="  "))
sweep_budget = Budget(
    steps=int(SWEEP["budget_steps"]),
    seeds=final.shape[1],
    wall_clock_s=float(SWEEP["seconds"].sum()),
    hardware=str(SWEEP["hardware"]),
)
print(textwrap.fill(sweep_budget.sentence(), 98))
by_seed0 = [f"x{factors[i]:g}" for i in np.argsort(-final[:, 0], kind="stable")]
by_iqm = [f"x{factors[i]:g}" for i in np.argsort([-s[0] for s in summary], kind="stable")]
print(f"Ranked by IQM:        {' > '.join(by_iqm)}")
print(f"Ranked by seed 0 only: {' > '.join(by_seed0)}")
fig, ax = plt.subplots()
xs = np.arange(len(factors))
spread = np.linspace(-0.2, 0.2, final.shape[1])  # one x per seed, so equal values stay apart
for i in xs:
    ax.plot(i + spread, final[i], ".", color=prl.plot.NEUTRAL_LIGHT)
centers = np.array([s[0] for s in summary])
err = np.array([[s[0] - s[1], s[2] - s[0]] for s in summary]).T
ax.errorbar(
    xs,
    centers,
    yerr=err,
    marker="o",
    capsize=3,
    color=prl.plot.SERIES[0],
    label="IQM [95% bootstrap CI]",
)
ax.plot(xs, final.mean(axis=1), "x", color=prl.plot.SERIES[1], label="mean")
ax.set(
    xticks=xs,
    xticklabels=[f"x{f:g}" for f in factors],
    xlabel="learning rate / reference",
    ylabel="greedy return at the budget",
    title="Learning-rate sweep, every seed",
)
ax.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# Which rates does the tie rule separate from the reference, and which tie? On which rates do
# the mean and the IQM differ most, and why (IQM)? The highest IQM of five configurations is
# the largest of five noisy estimates: is it likely to be too high or too low as an estimate of
# that configuration's true performance? (Gotcha: "The best configuration's score is its
# expected score.")

# %% role="ex5-chk1"
lab.check(5, m09.check_iqm, iqm, label="5.1")

# %% role="ex5-chk2"
lab.check(5, m09.check_bootstrap_ci, bootstrap_ci, label="5.2")

# %% [markdown] role="ex6-head"
# **One live seed.** Train the reference configuration yourself, on your seat's seed, at the
# lab's budget, and place your run against the recorded band of the reference configuration at
# the same budget (`prl.data.band`). Your seed is one more draw from the same distribution as
# the recorded seeds. Write `band_position`, which says where a value lands against a band. The
# cell reports it, the IQM and interval of the recorded seeds with and without yours, your four
# signals on your own log, and the budget sentence of your run.

# %% [markdown] role="ex6-predict"
# About 20 seats each run one seed against this 10-seed band. About how many seats will land
# outside it? Look at the band's top value first: can any seat land above a band that reaches
# 500? Will adding your seed move the IQM by more or by less than half the width of its
# interval?


# %% role="ex6-stub"
def band_position(value, recorded):
    """+1 if value is above every recorded value, -1 if below every one, 0 otherwise. A value
    equal to the band's top or bottom is inside (band)."""
    # TODO 6: two comparisons with the band's ends.
    raise NotImplementedError(
        "TODO 6: write band_position, or run lab.use_reference(6) to continue"
    )


# %% [markdown] role="ex6-hint"
# The band runs from `np.min(recorded)` to `np.max(recorded)`. Only a strict inequality puts a
# value outside.


# %% role="ex6-sol"
@lab.solution(6)
def band_position(value, recorded):
    """Above (+1), below (-1) or inside (0) the band's range."""
    if value > np.max(recorded):
        return 1
    if value < np.min(recorded):
        return -1
    return 0  # (band)


# %% role="ex6-run"
band = prl.data.band("m09_reference", budget)  # the recording at your budget
start = time.perf_counter()
print(f"Training the reference DQN on seed {RUN_SEED} for {budget.steps:,} steps:", flush=True)
q_live, log_live = train_dqn(
    lambda: gym.make("CartPole-v1"),
    CONFIG,
    RUN_SEED,
    budget.steps,
    log_every=budget.log_every,
    eval_every=budget.eval_every,
    eval_episodes=budget.eval_episodes,
    snapshot_steps=(CONFIG.learning_starts + 1, budget.steps // 2, budget.steps),
)
seconds_live = time.perf_counter() - start
live_return = float(log_live["eval_return"][-1])  # the band's metric at the last grid point
recorded = np.asarray(band["runs"], dtype=float)[:, -1]
lab.metric("live_return", round(live_return, 1))  # room code
print(f"Your greedy return: {live_return:.1f}; band [{recorded.min():.1f}, {recorded.max():.1f}]")
print(f"Band: {band['metric']}; {len(band['seeds'])} seeds; record {band['record']}.")
live_budget = Budget(steps=budget.steps, seeds=1, wall_clock_s=seconds_live, hardware=HARDWARE)
print(textwrap.fill(live_budget.sentence() + " Single seed, not a ranking.", 98))
fig, ax = plt.subplots()
for k, curve in enumerate(band["runs"]):
    label = f"recorded seeds ({len(band['seeds'])})" if k == 0 else None
    ax.plot(band["x"], curve, "o-", ms=3, color=prl.plot.NEUTRAL_LIGHT, lw=1, label=label)
ax.plot(
    log_live["eval_step"],
    log_live["eval_return"],
    "s-",
    color=prl.plot.SERIES[0],
    lw=2.5,
    label=f"your seed {RUN_SEED}",
)
ax.set(
    xlabel="env step",
    ylabel="greedy return (10 episodes)",
    title=f"Reference DQN, {budget.steps:,} steps: your seed over the recorded band",
)
ax.legend()
plt.show()
# Your functions last, so a missing one cannot cost you the training run above.
position = band_position(live_return, recorded)
lab.metric("live_position", position)  # room code: -1 below, 0 inside, +1 above the band
print(f"Your seed is {({1: 'above', 0: 'inside', -1: 'below'})[position]} the band.")
with_you = np.append(recorded, live_return)
for label, x in (("recorded seeds", recorded), ("with yours", with_you)):
    lo, hi = bootstrap_ci(x, iqm)
    print(f"  IQM of the {len(x)} {label}: {iqm(x):.1f} [{lo:.1f}, {hi:.1f}]")
print("Your four signals on your own run:")
print(f"  gather_check {gather_check(log_live):.3f}; target_lag (last) {target_lag(log_live)[-1]}")
print(
    f"  Q drift (last 5) {last_five(q_drift(log_live)):.2f}; relative TD error (second half) "
    f"{second_half_median(td_error_scale(log_live)):.4f}"
)

# %% [markdown] role="ex6-explain"
# Inside the band is not a pass, and outside is not a fail (band). How many of 20 seats would
# you expect outside, and what does the band's top value change? Do your four signals look like
# the healthy run's? Write the sentence you would send the teammate about the reference
# configuration, with its budget. The debrief pools the room's seeds.

# %% role="ex6-chk1"
lab.check(6, m09.check_band_position, band_position, label="6.1")

# %% role="ex6-chk2"
lab.check(6, m09.check_live_return, live_return, label="6.2")

# %% [markdown]
# **Your room code.** The last cell, `lab.finish()`, prints one line that starts with
# `ROOM m09`. It holds your seat, the checkpoints you passed (3.1 to 3.4 are your four
# diagnoses), whether QUICK budgets ran, and the numbers the debrief pools: run C's Q on probe 1
# after your probe run (`probe_q_c`, Exercise 4), your live seed's greedy return
# (`live_return`, Exercise 6) and where it landed against the band (`live_position`: −1 below,
# 0 inside, +1 above). Paste the line where your instructor says.

# %% [markdown] role="stretch"
# **Probe 3: discounting.** Jones's third probe has one action, the observation 0 and then 1,
# two steps, and a reward of +1 at the end only. Write it as `TwoStepEnv` (the first step
# returns observation 1, reward 0, not terminated; the second returns observation 1, reward 1,
# terminated). Work out Q at both observations for a correct agent at γ = 0.99, then run the
# five agents on it. Which run fails this probe that passed the first two, and why?


# %% role="stretch-stub"
class TwoStepEnv(gym.Env):
    """Probe 3: observation 0, then 1; two steps; reward +1 at the end only. The same spaces
    as the other probes."""

    def __init__(self):
        raise NotImplementedError(
            "Stretch: write TwoStepEnv, or run lab.use_reference('stretch') to see the solution"
        )


# %% role="stretch-sol"
@lab.solution("stretch")
class TwoStepEnv(gym.Env):
    """Probe 3: two steps, discounting."""

    def __init__(self):
        box = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = box
        self.action_space = gym.spaces.Discrete(1)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t == 2
        return np.ones(1, np.float32), float(done), done, False, {}


# %% role="stretch-run"
lab.check("stretch", m09.check_two_step_env, TwoStepEnv, label="stretch")
obs3 = np.array([[0.0], [1.0]], np.float32)
for run in AGENTS:
    q_net, _ = train_agent(
        run,
        TwoStepEnv,
        PROBE_CONFIG,
        RUN_SEED,
        budget.probe_steps,
        log_every=100,
        probe_obs=obs3,
        verbose=False,
    )
    with torch.no_grad():
        got = q_net(torch.as_tensor(obs3))[:, 0].numpy()
    print(f"  {run:8s} probe 3: Q(0) = {got[0]:7.3f}, Q(1) = {got[1]:7.3f}")
print("Exact answers: Q(0) = γ · 1 = 0.99, Q(1) = 1.")
