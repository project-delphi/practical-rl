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
# ones, A to D, each recorded on 3 seeds. For each broken run you write the signal that exposes
# it and choose a diagnosis. Then you write two probe environments and run every agent on
# them, to confirm the diagnoses. Last, you report a recorded learning-rate sweep with the
# interquartile mean (IQM) and bootstrap intervals, and add one live seed of your own.
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
            q_max = q_values.max(1).values.mean()
            for k, x in (("loss", loss.detach()), ("td", td.mean()), ("td_max", td.max())):
                acc[k].append(float(x))
            acc["q_max"].append(float(q_max))
            acc["gn"].append(float(gn))
            if done.any():
                acc["q_term"].append(float(q_sa[done].mean()))
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
# ### Provided: the recorded runs, one run's log, and the dashboard
#
# `prl.data.load("m09_broken_runs")` holds the healthy run and runs A to D, each on seeds 0, 1
# and 2, at the live budget. `run_log(run, seed)` returns one run's log as a dict, with the
# keys `train_dqn` writes. `dashboard(runs)` draws four panels for the runs you name, every
# seed as its own line.


# %%
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


# The healthy run is the reference, drawn as reference lines are; A-D differ by line style too.
RUN_STYLE = {
    "healthy": dict(color=prl.plot.NEUTRAL_LIGHT, ls="-"),
    "A": dict(color=prl.plot.SERIES[0], ls=(0, (4.5, 1.8))),
    "B": dict(color=prl.plot.SERIES[1], ls=(0, (1, 1.6))),
    "C": dict(color=prl.plot.SERIES[2], ls=(0, (6, 1.8, 1.4, 1.8))),
    "D": dict(color=prl.plot.SERIES[3], ls=(0, (3, 1.2, 1, 1.2))),
}


def dashboard(runs, title):
    """Provided: return, loss, mean |delta| and Q at the start states, every seed."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for run in runs:
        for k, seed in enumerate(RUNS["seeds"]):
            log = run_log(run, int(seed))
            label = f"run {run}" if (k == 0 and run != "healthy") else None
            label = "healthy" if (k == 0 and run == "healthy") else label
            kw = dict(lw=1.2, alpha=0.85, label=label, **RUN_STYLE[run])
            axes[0, 0].plot(*smooth_returns(log), **kw)
            axes[0, 1].plot(log["step"], log["loss"], **kw)
            axes[1, 0].plot(log["step"], log["td_abs_mean"], **kw)
            q_start = log["q_probe"].max(axis=-1).mean(axis=-1)
            axes[1, 1].plot(log["step"], q_start, **kw)
    axes[0, 0].set(ylabel="return (episodes per 2,500 steps)")
    axes[0, 1].set(ylabel="mean loss", yscale="log")
    axes[1, 0].set(ylabel="mean |δ|", yscale="log", xlabel="env step")
    axes[1, 1].set(ylabel="Q at the start states", yscale="symlog", xlabel="env step")
    axes[0, 0].legend()
    fig.suptitle(title)
    plt.show()


SIGNALS = {}  # signal name -> {run: summary per seed}; printed as a table as you go


def signal_table(name, fn, summary):
    """Provided: apply your signal to every run and seed; print one summary per seed."""
    SIGNALS[name] = {
        run: [summary(fn(run_log(run, int(s)))) for s in RUNS["seeds"]] for run in RUN_STYLE
    }
    print(f"{name}, seeds {RUNS['seeds'].tolist()}:")
    for run, values in SIGNALS[name].items():
        print(f"  {run:8s}" + "".join(f"{v:>12.4g}" for v in values))


# The causes to choose from in Exercises 2-5. Several runs can share a symptom; one cause each.
CAUSES = {
    "mask": "The terminal mask is missing: targets bootstrap past the end of an episode.",
    "sync": "The target network is never updated from the online network.",
    "lr": "The learning rate is far too large.",
    "gather": "The loss reads the wrong entries of Q(s, .), not Q(s, a) for the action taken.",
    "reward": "The rewards are scaled up: every reward is many times CartPole's 1.",
    "seed": "Nothing is wrong with the code or the configuration: an unlucky seed.",
}


# %% [markdown] role="ex1-head"
# **Healthy logs.** Every `log_every` = 500 env steps, `train_dqn` writes the mean loss and the
# mean and largest |δ| over the batches since the last write, δ = y − Q(s, a; w) (TD error);
# the mean over the batch of max_a Q(s, a; w); and Q at 8 fixed start states, from both the
# online and the target network. At each episode's end it writes the return, the discounted
# return from the first state and the end step. At three steps it keeps one whole batch: Q(s, ·)
# for every row, the actions, the values the loss used and the targets. Before you look at the
# healthy run, write down what you expect three of its curves to do, in `healthy_predictions`.
#
# 1. `"loss"`: the mean loss over the run (a) falls steadily toward 0; (b) drops in the first
#    few thousand steps, then stays level or rises slowly while Q grows; (c) rises without
#    bound.
# 2. `"q_start"`: Q at the start states (a) stays near 1, CartPole's reward per step;
#    (b) climbs over the run; (c) jumps to its final value within the first 5,000 steps.
# 3. `"td_max"`: the largest |δ| in a batch (a) is about the mean |δ|; (b) is about 10 times
#    the mean; (c) is close to Q itself, about 100 times the mean.

# %% [markdown] role="ex1-predict"
# Answer the three questions above in `healthy_predictions`, with one sentence of reason for
# each as a comment, before you run the next cell. The checkpoint records your answers; it does
# not grade them.


# %% role="ex1-stub"
def healthy_predictions():
    """Your predictions for the healthy run: a dict with the keys "loss", "q_start" and
    "td_max", each "a", "b" or "c" (the questions above)."""
    # TODO 1: return {"loss": ..., "q_start": ..., "td_max": ...}, with a reason as a comment.
    raise NotImplementedError(
        "TODO 1: write healthy_predictions, or run lab.use_reference(1) to continue"
    )


# %% [markdown] role="ex1-hint"
# The target y moves as Q moves: what happens to the size of δ while Q climbs from 0 toward
# the return the agent earns? For the largest |δ|, think of the transitions where the pole
# falls: their target is y = r, while the network's Q there looks like Q at nearby states that
# did not end.


# %% role="ex1-sol"
@lab.solution(1)
def healthy_predictions():
    """What the healthy record shows (pending the record)."""
    return {
        "loss": "b",  # the targets grow with Q, so the loss need not fall
        "q_start": "b",  # Q climbs toward the discounted return
        "td_max": "c",  # terminal rows: Q near the others', y = r
    }


# %% role="ex1-run"
predicted = healthy_predictions()
RUNS = prl.data.load("m09_broken_runs")
dashboard(["healthy"], "The healthy run, 3 seeds")
for seed in RUNS["seeds"]:
    log = run_log("healthy", int(seed))
    at = np.searchsorted(log["step"], [5_000, 25_000, log["step"][-1]])
    q_start = log["q_probe"].max(axis=-1).mean(axis=-1).astype(float)
    ratio = np.nanmedian(log["td_abs_max"] / log["td_abs_mean"])
    print(
        f"seed {seed}: at steps {log['step'][at].tolist()}, mean loss "
        f"{np.round(log['loss'][at].astype(float), 3).tolist()}, Q at the start states "
        f"{np.round(q_start[at], 1).tolist()}; largest |δ| / mean |δ|, median {ratio:.0f}"
    )
print(f"You predicted {predicted}.")
lab.metric("healthy_predictions", predicted)

# %% [markdown] role="ex1-explain"
# Q at the start states climbs for the whole run, so every target y = r + γ max Q(s′; w⁻)
# climbs with it, and a loss computed against a moving target need not fall. Which of your
# three predictions held? Why is the largest |δ| so much larger than the mean, and on which
# transitions do you expect it? (Gotcha: "The loss should go down.")

# %% role="ex1-chk"
lab.check(1, m09.check_healthy_predictions, healthy_predictions)

# %% [markdown] role="ex2-head"
# **Broken run A.** The teammate's first run. Its return stays far below the healthy run's on
# all 3 seeds. The dashboard puts it next to the healthy run. Then write `td_error_scale`: at
# every log point, the mean |δ| divided by the mean over the batch of max_a Q(s, a; w)
# (relative TD error). Dividing by Q makes the signal free of the reward's units: a run whose
# rewards were all 10 times larger would have 10 times larger δ and 10 times larger Q. Then
# choose run A's cause from `CAUSES` in `diagnose_a`.

# %% [markdown] role="ex2-predict"
# Run A's loss and mean |δ| end several times higher than the healthy run's. Which causes in
# `CAUSES` could do that? On the healthy run, what do you expect the relative TD error to be in
# the second half, roughly: 0.001, 0.01 or 0.1?


# %% role="ex2-stub"
def td_error_scale(log):
    """The relative TD error at every log point: log["td_abs_mean"] / |log["q_max_mean"]|,
    both of shape (L,). Returns an array of shape (L,)."""
    # TODO 2 (part 1): one line.
    raise NotImplementedError(
        "TODO 2: write td_error_scale, or run lab.use_reference(2) to continue"
    )


def diagnose_a():
    """Your diagnosis of run A: one key of CAUSES."""
    # TODO 2 (part 2): return "mask", "sync", "lr", "gather", "reward" or "seed".
    raise NotImplementedError("TODO 2: write diagnose_a, or run lab.use_reference(2) to continue")


# %% [markdown] role="ex2-hint"
# NumPy divides two arrays element by element; `np.abs` takes care of a negative Q. For the
# diagnosis: the signal divides out the reward's scale, so which causes can it still see? The
# last cell of this exercise prints your signal for every run, not only A.


# %% role="ex2-sol"
@lab.solution(2)
def td_error_scale(log):
    """Mean |delta| over the batch Q scale, per log point."""
    q_scale = np.abs(log["q_max_mean"])
    return log["td_abs_mean"] / q_scale  # (relative TD error)


@lab.solution(2)
def diagnose_a():
    """Run A: the learning rate is 100 times the reference."""
    return "lr"


# %% role="ex2-run"
def second_half_median(x):
    """Provided: the median of a signal over the second half of the run."""
    return float(np.nanmedian(np.asarray(x)[len(x) // 2 :]))


dashboard(["healthy", "A"], "Run A against the healthy run")
signal_table("td_error_scale", td_error_scale, second_half_median)
print(f"Your diagnosis of run A: {CAUSES.get(diagnose_a(), diagnose_a())}")

# %% [markdown] role="ex2-explain"
# The healthy run's relative TD error is small and steady, run A's several times higher on
# every seed: each update moves Q further than the targets justify (relative TD error). Which
# other run shows a high value, and why does that make this signal alone too weak to decide?
# Why can "reward" not be A's cause, given how the signal is built? Exercise 6 runs A on the
# probe environments.

# %% role="ex2-chk1"
lab.check(2, m09.check_td_error_scale, td_error_scale, label="2.1")

# %% role="ex2-chk2"
lab.check(2, m09.check_diagnose_a, diagnose_a, label="2.2")

# %% [markdown] role="ex3-head"
# **Broken run B.** Its return stays at the level of a policy that pushes the cart the same
# way every step. At three steps the log kept a whole batch: `snap_q`, Q(s, ·; w) for every
# row (shape (S, B, |𝒜|): S = 3 snapshots of B = 32 rows, |𝒜| = 2 actions), `snap_actions`,
# the action of every row (S, B), and `snap_gathered`, the value the loss used (S, B).
# Write `gather_check`: the share of rows whose gathered value is not Q(s_i, a_i). Then choose
# run B's cause in `diagnose_b`.

# %% [markdown] role="ex3-predict"
# In a correct DQN, what is the value the loss uses for row i, in terms of `snap_q` and
# `snap_actions`? What will `gather_check` return on the healthy run? On run B?


# %% role="ex3-stub"
def gather_check(log):
    """The share of snapshot rows whose log["snap_gathered"] value differs from
    log["snap_q"][s, i, a] with a = log["snap_actions"][s, i]. Use np.isclose(..., rtol=1e-5,
    atol=1e-6) to compare. Returns one float in [0, 1]."""
    # TODO 3 (part 1): pick Q(s_i, a_i) out of snap_q for every row, then compare.
    raise NotImplementedError("TODO 3: write gather_check, or run lab.use_reference(3) to continue")


def diagnose_b():
    """Your diagnosis of run B: one key of CAUSES."""
    # TODO 3 (part 2): return one key of CAUSES.
    raise NotImplementedError("TODO 3: write diagnose_b, or run lab.use_reference(3) to continue")


# %% [markdown] role="ex3-hint"
# `np.take_along_axis(q, a[..., None], axis=-1)` picks one entry of the last axis of `q` per
# row, with `a` holding the index; drop the extra axis with `[..., 0]`. `~np.isclose(...)` marks
# the rows that differ, and the mean of a boolean array is the share of True.


# %% role="ex3-sol"
@lab.solution(3)
def gather_check(log):
    """Share of batch rows whose gathered value is not Q(s_i, a_i)."""
    q, a = log["snap_q"], log["snap_actions"]
    want = np.take_along_axis(q, a[..., None], axis=-1)[..., 0]
    wrong = ~np.isclose(log["snap_gathered"], want, rtol=1e-5, atol=1e-6)
    return float(wrong.mean())


@lab.solution(3)
def diagnose_b():
    """Run B: gather_q reads the wrong entries of Q(s, .)."""
    return "gather"


# %% role="ex3-run"
dashboard(["healthy", "B"], "Run B against the healthy run")
signal_table("gather_check", gather_check, float)
log_b = run_log("B", 0)
q_spread = np.abs(log_b["q_probe"][-1, :, 0] - log_b["q_probe"][-1, :, 1]).mean()
print(f"Run B, seed 0, end of run: mean |Q(s, 0) - Q(s, 1)| at the start states {q_spread:.1f}")
print(f"Your diagnosis of run B: {CAUSES.get(diagnose_b(), diagnose_b())}")

# %% [markdown] role="ex3-explain"
# The TD error compares the target with Q(s, a; w) for the action a that was taken (TD error).
# On the healthy run the gathered value is that entry on every row; on run B it almost never
# is. Run B's two actions end far apart at the start states: which action's values does its
# loss train, and why does that make the greedy policy push the same way every step?

# %% role="ex3-chk1"
lab.check(3, m09.check_gather_check, gather_check, label="3.1")

# %% role="ex3-chk2"
lab.check(3, m09.check_diagnose_b, diagnose_b, label="3.2")

# %% [markdown] role="ex4-head"
# **Broken run C.** Its return never leaves the minimum. Compare what the network believes
# with what the agent earns: at every log point, Q at the start states (the mean over the 8
# probe states of max_a Q(s, a; w)) minus the mean discounted return of the episodes that ended
# in the last `window` env steps (Q drift). Both are estimates of the value of the start
# states, so a healthy run keeps them close. Write `q_drift`, then choose run C's cause.

# %% [markdown] role="ex4-predict"
# CartPole pays 1 per step and stops at 500 steps. At γ = 0.99, what is the largest discounted
# return any episode can earn from its first state? If run C's Q at the start states ends far
# above that number, which causes in `CAUSES` could explain it?


# %% role="ex4-stub"
def q_drift(log, window=5000):
    """Q drift at every log point i: q_bar[i] minus the mean of log["ep_disc_return"] over the
    episodes with t - window < log["ep_end_step"] <= t, where t = log["step"][i] and q_bar =
    log["q_probe"].max(axis=-1).mean(axis=-1). NaN where no episode ended in the window.
    Returns an array of shape (L,)."""
    # TODO 4 (part 1): q_bar first; then a loop over the log points.
    raise NotImplementedError("TODO 4: write q_drift, or run lab.use_reference(4) to continue")


def diagnose_c():
    """Your diagnosis of run C: one key of CAUSES."""
    # TODO 4 (part 2): return one key of CAUSES.
    raise NotImplementedError("TODO 4: write diagnose_c, or run lab.use_reference(4) to continue")


# %% [markdown] role="ex4-hint"
# `log["q_probe"]` has shape (L, 8, 2): the max over the last axis, then the mean over the
# next, leaves one number per log point. Start the output with `np.full(L, np.nan)`, and select
# the recent episodes with a boolean mask on `log["ep_end_step"]`.


# %% role="ex4-sol"
@lab.solution(4)
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


@lab.solution(4)
def diagnose_c():
    """Run C: the terminal mask is missing."""
    return "mask"


# %% role="ex4-run"
def last_five(x):
    """Provided: the mean of a signal over its last five log points."""
    return float(np.nanmean(np.asarray(x)[-5:]))


dashboard(["healthy", "C"], "Run C against the healthy run")
signal_table("q_drift", q_drift, last_five)
gamma = float(RUNS["gamma"])
print(f"The most any CartPole episode can earn from its first state at γ = {gamma}:")
print(f"  {sum(gamma**t for t in range(500)):.2f} (500 rewards of 1)")
print(f"Your diagnosis of run C: {CAUSES.get(diagnose_c(), diagnose_c())}")

# %% [markdown] role="ex4-explain"
# Run C's Q at the start states is far above anything an episode can earn, and keeps rising.
# The target of a transition that ends the episode is r alone, because of the factor
# (1 − terminated) (masked target): what is it without that factor, and why does that let Q
# grow past every return? Which other runs show a positive drift, and how large is it next to
# run C's (Q drift)? Exercise 6 confirms the diagnosis on a probe.

# %% role="ex4-chk1"
lab.check(4, m09.check_q_drift, q_drift, label="4.1")

# %% role="ex4-chk2"
lab.check(4, m09.check_diagnose_c, diagnose_c, label="4.2")

# %% [markdown] role="ex5-head"
# **Broken run D.** Its loss is the smallest of all five runs, and its return stays at the
# minimum. The log keeps Q at the start states from the target network as well,
# `q_probe_target` (shape (L, 8, 2)). Write `target_lag`: at every log point, the number of env
# steps since the target network's values at the probe states last changed. Then choose run
# D's cause.

# %% [markdown] role="ex5-predict"
# Run D's loss falls below one hundredth of the healthy run's within a few thousand steps and
# stays there. Is that good news? In the healthy run the target network is synced every 500
# env steps and the log is written every 500 steps: what is the largest `target_lag` you expect
# there, after learning starts?


# %% role="ex5-stub"
def target_lag(log):
    """Env steps since the target network's probe values last changed, at every log point:
    lag[0] = 0; for i >= 1, lag[i] = step[i] - step[j], where j <= i is the latest log point
    at which log["q_probe_target"][j] differs from [j - 1] in any entry (j = 0 if none).
    Returns an int array of shape (L,)."""
    # TODO 5 (part 1): one loop over the log points, remembering the step of the last change.
    raise NotImplementedError("TODO 5: write target_lag, or run lab.use_reference(5) to continue")


def diagnose_d():
    """Your diagnosis of run D: one key of CAUSES."""
    # TODO 5 (part 2): return one key of CAUSES.
    raise NotImplementedError("TODO 5: write diagnose_d, or run lab.use_reference(5) to continue")


# %% [markdown] role="ex5-hint"
# `np.any(a != b)` is True if any entry differs. Keep `last`, the step of the latest change,
# starting at `step[0]`, and write `step[i] - last` at every log point.


# %% role="ex5-sol"
@lab.solution(5)
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


@lab.solution(5)
def diagnose_d():
    """Run D: the target network is never synced."""
    return "sync"


# %% role="ex5-run"
dashboard(["healthy", "D"], "Run D against the healthy run")
signal_table("target_lag", target_lag, lambda lag: float(np.max(lag)))
q_d = run_log("D", 0)["q_probe"][-1].max(axis=-1).mean()
print(f"Run D, seed 0, end of run: Q at the start states {q_d:.3f}")
print(f"Your diagnosis of run D: {CAUSES.get(diagnose_d(), diagnose_d())}")
diagnoses = {
    run: f()
    for run, f in zip("ABCD", (diagnose_a, diagnose_b, diagnose_c, diagnose_d), strict=True)
}
lab.metric("diagnoses", diagnoses)
print("Your four signals on every run (one value per seed) are in the tables above.")

# %% [markdown] role="ex5-explain"
# The target y = r + γ(1 − terminated) max Q(s′; w⁻) uses the target network's parameters w⁻
# (masked target). If w⁻ never changes, what fixed regression problem is the online network
# solving, and what value of Q solves it when w⁻ is the network's random start? Why is a tiny
# loss here a sign of trouble rather than success? Look back at your four tables: which signals
# single out one run, and which flag more than one?

# %% role="ex5-chk1"
lab.check(5, m09.check_target_lag, target_lag, label="5.1")

# %% role="ex5-chk2"
lab.check(5, m09.check_diagnose_d, diagnose_d, label="5.2")

# %% [markdown] role="ex6-head"
# **Probe environments.** A diagnosis from logs is a hypothesis. A probe environment tests one
# piece of an agent in an environment so small that the right answer is known (Jones 2021).
# Write two, as Gymnasium environments with a one-dimensional observation in [−1, 1] and one
# action, where every episode is one step long and ends with `terminated=True`:
# `ConstantRewardEnv` (probe 1: observation 0, reward +1) and `ObsRewardEnv` (probe 2:
# observation +1 or −1 with equal probability, reward equal to the observation; the step
# returns the same observation). The run cell then trains the healthy agent and the agents of
# runs A to D on both probes, with the configurations and functions that made runs A to D.

# %% [markdown] role="ex6-predict"
# Probe 1 pays 1 and ends after one step. At γ = 0.99, what Q should a correct agent learn?
# What Q would an agent that ignores `terminated` converge to, if it trained long enough? Which
# of runs A to D will fail probe 1, which will fail probe 2, and which will pass both?


# %% role="ex6-stub"
class ConstantRewardEnv(gym.Env):
    """Probe 1. observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32);
    action_space = gym.spaces.Discrete(1). reset(seed=None, options=None) calls
    super().reset(seed=seed) and returns (np.zeros(1, np.float32), {}); step(action) returns
    (np.zeros(1, np.float32), 1.0, True, False, {})."""

    # TODO 6 (part 1): __init__ sets the two spaces; then reset and step.
    def __init__(self):
        raise NotImplementedError(
            "TODO 6: write ConstantRewardEnv, or run lab.use_reference(6) to continue"
        )


class ObsRewardEnv(gym.Env):
    """Probe 2. The same spaces. reset draws the observation o = +1.0 or -1.0 with
    self.np_random.random() < 0.5 giving +1.0, and returns (np.array([o], np.float32), {});
    step returns (np.array([o], np.float32), o, True, False, {}): the same observation, and
    the reward o."""

    # TODO 6 (part 2): keep o on self between reset and step.
    def __init__(self):
        raise NotImplementedError(
            "TODO 6: write ObsRewardEnv, or run lab.use_reference(6) to continue"
        )


# %% [markdown] role="ex6-hint"
# `super().reset(seed=seed)` seeds `self.np_random`, Gymnasium's generator for the
# environment, so the same seed gives the same observations. Return float32 arrays of shape
# (1,), not Python floats: the agent stacks them into a batch.


# %% role="ex6-sol"
@lab.solution(6)
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


@lab.solution(6)
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


# %% role="ex6-run"
def td_target_no_mask(batch, q_next, gamma):
    """Run C's td_target: it ignores terminated."""
    return batch.rew + gamma * q_next


def gather_q_dim0(q_values, actions):
    """Run B's gather_q: dimension 0 instead of 1."""
    return q_values.gather(0, actions.long().unsqueeze(1)).squeeze(1)


# How runs A to D were made, now that you have diagnosed them: a configuration change or a
# function swapped into train_dqn. labs/shared is never edited.
AGENTS = {
    "healthy": ({}, {}),
    "A": (dict(lr=CONFIG.lr * 100), {}),
    "B": ({}, dict(gather_q=gather_q_dim0)),
    "C": ({}, dict(td_target=td_target_no_mask)),
    "D": (dict(target_sync=None), {}),
}
PROBE_CONFIG = dataclasses.replace(
    CONFIG, learning_starts=64, target_sync=100, train_freq=1, eps_fraction=0.0
)


def train_agent(run, make_env, config, seed, **kwargs):
    """Provided: train_dqn as run `run` was made; the reference functions come back after."""
    changes, swaps = AGENTS[run]
    saved = {name: globals()[name] for name in swaps}
    globals().update(swaps)
    try:
        changed = dataclasses.replace(config, **changes)
        return train_dqn(make_env, changed, seed, budget.probe_steps, **kwargs)
    finally:
        globals().update(saved)


probes = {
    "probe 1": (ConstantRewardEnv, np.array([[0.0]], np.float32), [1.0]),
    "probe 2": (ObsRewardEnv, np.array([[-1.0], [1.0]], np.float32), [-1.0, 1.0]),
}
probe_q = {}
start = time.perf_counter()
print(f"Each agent on each probe: {budget.probe_steps:,} steps, target sync every 100.")
for run in AGENTS:
    cells = []
    for name, (env_cls, obs, want) in probes.items():
        q_net, _ = train_agent(
            run, env_cls, PROBE_CONFIG, RUN_SEED, log_every=100, probe_obs=obs, verbose=False
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

# %% [markdown] role="ex6-explain"
# Run C's Q on probe 1 is far above 1 but still far below where it is heading: count the
# target syncs in the run and use (masked target) to explain the number. Why does run B pass
# probe 1 and fail probe 2? Runs A and D pass both probes: what does a passing probe rule out,
# and what does it not? Which probe of Jones's list would catch run D? (Gotcha: "The probes
# pass, so the agent is correct.")
#
# **Diagnostics for this lab.** Log more than the return and the loss: the mean and largest
# |δ|, Q at fixed start states from both networks, the discounted return of every episode, and
# now and then one whole batch. Healthy looks like this: Q at the start states climbs toward the
# discounted return the agent earns and stays near it; the relative TD error is small and
# steady; the gathered value is Q(s_i, a_i) on every row; the target's values change at every
# sync. Broken looks like this: Q above every possible return (a missing mask); a loss that
# vanishes while Q stays at the reward (a frozen target); gathered values that are not
# Q(s_i, a_i) (a wrong index); a relative TD error several times the healthy one (too large a
# step). Then confirm with probes: they are quick, and a correct agent gets them right to a few
# decimals.

# %% role="ex6-chk1"
lab.check(6, m09.check_constant_reward_env, ConstantRewardEnv, label="6.1")

# %% role="ex6-chk2"
lab.check(6, m09.check_obs_reward_env, ObsRewardEnv, label="6.2")

# %% [markdown] role="ex7-head"
# **A recorded sweep.** The teammate's next question: is the reference learning rate a good
# one? `prl.data.load("m09_sweep")` holds 5 learning rates, 0.1, 0.3, 1, 3 and 10 times the
# reference, each run on seeds 0 to 9 at the live budget, and each run's greedy return at the
# end (`final_return`, shape (5, 10)). Write the two tools the workshop compares with:
# `iqm(x)`, the interquartile mean (IQM), and `bootstrap_ci(x, statistic, ...)`, the percentile
# bootstrap interval (bootstrap CI). Write them exactly as described: the checkpoints compare
# them with `prl.evaluate.iqm` and `prl.evaluate.bootstrap_ci` on the same inputs.

# %% [markdown] role="ex7-predict"
# Will the reference (×1) have the highest IQM of the five? Will the tie rule separate it from
# ×0.3 and ×3? If you had run only seed 0 of each learning rate, would you rank the five the
# same way as by their IQMs over 10 seeds?


# %% role="ex7-stub"
def iqm(x):
    """The interquartile mean (IQM) of the values in x: sort them, drop k = floor(n / 4) from
    each end (n = len(x)), and return the mean of the rest as a float. n = 10 drops 2 from each
    end and averages 6."""
    # TODO 7 (part 1): three lines.
    raise NotImplementedError("TODO 7: write iqm, or run lab.use_reference(7) to continue")


def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):
    """Percentile bootstrap interval for statistic(x) (bootstrap CI). x = np.asarray(x,
    dtype=float).ravel(); rng = np.random.default_rng(seed); draw every index at once,
    idx = rng.integers(0, n, size=(n_boot, n)); compute statistic(x[row]) for each row of idx;
    return the two quantiles (1 - level) / 2 and 1 - (1 - level) / 2 of those n_boot values
    (np.quantile) as a tuple of two floats."""
    # TODO 7 (part 2): five lines.
    raise NotImplementedError("TODO 7: write bootstrap_ci, or run lab.use_reference(7) to continue")


# %% [markdown] role="ex7-hint"
# `np.sort(x)[k : n - k]` keeps the middle values. In `bootstrap_ci`, the indices must come
# from `np.random.default_rng(seed)`, the generator `prl.evaluate` uses, so that the same seed
# gives the same resamples and the checkpoint can compare the exact interval. `x[row]` with an
# array of indices is one resample: n values drawn with replacement.


# %% role="ex7-sol"
@lab.solution(7)
def iqm(x):
    """Mean of the middle values after dropping floor(n/4) per end."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    k = len(x) // 4
    return float(x[k : len(x) - k].mean())  # (IQM)


@lab.solution(7)
def bootstrap_ci(x, statistic=np.mean, *, level=0.95, n_boot=5000, seed=0):
    """Percentile bootstrap interval for statistic(x)."""
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([statistic(x[row]) for row in idx])
    tails = [(1 - level) / 2, 1 - (1 - level) / 2]
    lo, hi = np.quantile(boots, tails)  # (bootstrap CI)
    return float(lo), float(hi)


# %% role="ex7-run"
SWEEP = prl.data.load("m09_sweep")
factors, final = SWEEP["lr_factor"], SWEEP["final_return"]
ref = int(np.flatnonzero(factors == 1.0)[0])
summary = []
print(f"Greedy return at {int(SWEEP['budget_steps']):,} steps, {final.shape[1]} seeds per rate:")
for f, x in zip(factors, final, strict=True):
    lo, hi = bootstrap_ci(x, iqm)
    summary.append((iqm(x), lo, hi))
    seeds = " ".join(f"{v:5.0f}" for v in x)
    print(f"  lr x{f:<4g} IQM {iqm(x):6.1f} [{lo:6.1f}, {hi:6.1f}]   seeds: {seeds}")
sweep_budget = Budget(
    steps=int(SWEEP["budget_steps"]),
    seeds=final.shape[1],
    wall_clock_s=float(SWEEP["seconds"].sum()),
    hardware=str(SWEEP["hardware"]),
)
for i, f in enumerate(factors):
    if i != ref:
        v = compare(final[i], final[ref], names=(f"lr x{f:g}", "lr x1"))
        print(textwrap.fill(v.sentence.split(" (IQM")[0], 98, initial_indent="  "))
print(textwrap.fill(sweep_budget.sentence(), 98))
by_seed0 = [f"x{factors[i]:g}" for i in np.argsort(-final[:, 0], kind="stable")]
by_iqm = [f"x{factors[i]:g}" for i in np.argsort([-s[0] for s in summary], kind="stable")]
print(f"Ranked by IQM:    {by_iqm}")
print(f"Ranked by seed 0: {by_seed0}")
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
ax.set(
    xticks=xs,
    xticklabels=[f"x{f:g}" for f in factors],
    xlabel="learning rate / reference",
    ylabel="greedy return at the budget",
    title="Learning-rate sweep, every seed",
)
ax.legend()
plt.show()

# %% [markdown] role="ex7-explain"
# Ten seeds is few, so the IQM keeps 6 values (IQM) and the interval is wide. Which rates does
# the tie rule separate from the reference, and which tie? What does a flat top say about how
# carefully the learning rate needs tuning, and what does a cliff on one side say? The highest
# IQM of five configurations is the largest of five noisy estimates: is it likely to be too
# high or too low as an estimate of that configuration's true performance? (Gotcha: "The best
# configuration's score is its expected score.")

# %% role="ex7-chk1"
lab.check(7, m09.check_iqm, iqm, label="7.1")

# %% role="ex7-chk2"
lab.check(7, m09.check_bootstrap_ci, bootstrap_ci, label="7.2")

# %% [markdown] role="ex8-head"
# **One live seed.** Train the reference configuration yourself, on your seat's seed, at the
# lab's budget, and place your run against the recorded band of the reference configuration at
# the same budget (`prl.data.band`). Your seed is one more draw from the same distribution as
# the recorded seeds. The cell reports where it lands, the IQM and interval of the recorded
# seeds with and without yours (your `iqm` and `bootstrap_ci`), your four signals on your own
# log, and the budget sentence of your run.

# %% [markdown] role="ex8-predict"
# The band has 10 recorded seeds. If your seed comes from the same distribution, what is the
# chance that its return lands above the band's largest or below its smallest? Will adding your
# seed move the IQM by more or by less than half the width of the interval?


# %% role="ex8-run"
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
position = int(live_return > recorded.max()) - int(live_return < recorded.min())
where = {1: "above the band", 0: "inside the band", -1: "below the band"}[position]
print(
    f"Your greedy return: {live_return:.1f}, {where} [{recorded.min():.1f}, {recorded.max():.1f}]"
)
print(f"Band: {band['metric']}; {len(band['seeds'])} seeds; record {band['record']}.")
with_you = np.append(recorded, live_return)
for label, x in (("recorded seeds", recorded), ("with yours", with_you)):
    lo, hi = bootstrap_ci(x, iqm)
    print(f"  IQM of the {len(x)} {label}: {iqm(x):.1f} [{lo:.1f}, {hi:.1f}]")
print("Your four signals on your own run:")
print(f"  td_error_scale (second-half median) {second_half_median(td_error_scale(log_live)):.4f}")
print(f"  gather_check {gather_check(log_live):.3f}")
print(f"  q_drift (last five log points) {last_five(q_drift(log_live)):.2f}")
print(f"  target_lag (largest) {int(np.max(target_lag(log_live)))}")
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
lab.metric("live_return", round(live_return, 1))  # room code
lab.metric("live_position", position)  # room code: -1 below, 0 inside, +1 above the band

# %% [markdown] role="ex8-explain"
# Your seed landed where it landed: inside the band is not a pass, and outside is not a fail
# (band). How often should a correct seed land outside a band of 10? Your four signals on your
# own run: do they look like the healthy run's? Write the sentence you would send the teammate
# about the reference configuration, with its budget. The debrief pools the room's seeds.

# %% role="ex8-chk"
lab.check(8, m09.check_live_return, live_return)

# %% [markdown]
# **Your room code.** The last cell, `lab.finish()`, prints one line that starts with
# `ROOM m09`. It holds your seat, the checkpoints you passed, whether QUICK budgets ran, and
# the numbers the debrief pools: run C's Q on probe 1 after your probe run (`probe_q_c`,
# Exercise 6), your live seed's greedy return (`live_return`, Exercise 8) and where it landed
# against the band (`live_position`: −1 below, 0 inside, +1 above). Your run record keeps your
# predictions and your four diagnoses as well. Paste the line where your instructor says.

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
        run, TwoStepEnv, PROBE_CONFIG, RUN_SEED, log_every=100, probe_obs=obs3, verbose=False
    )
    with torch.no_grad():
        got = q_net(torch.as_tensor(obs3))[:, 0].numpy()
    print(f"  {run:8s} probe 3: Q(0) = {got[0]:7.3f}, Q(1) = {got[1]:7.3f}")
print("Exact answers: Q(0) = γ · 1 = 0.99, Q(1) = 1.")
