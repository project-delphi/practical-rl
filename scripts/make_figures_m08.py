"""Module 8: the recorded bands (experiment records) and light and dark SVGs drawn from them.

  uv run python scripts/make_figures_m08.py --run m08-dqn-live ...  # records go to runs/inbox/
  uv run python scripts/add_run_record.py                           # file them in runs/
  uv run python scripts/export_bands.py                             # copy the bands into prl
  uv run python scripts/make_figures_m08.py             # draw images/08-*.svg and print the report
  uv run python scripts/make_figures_m08.py --check     # fail if a committed SVG differs
  uv run python scripts/make_figures_m08.py --reproduce m08-dqn-quick [--seeds 0,1]
                                                        # rerun and compare with the filed record

Experiments (seeds 0-4, which no seat uses; seats train with (SEED + 1) * 1000):

  m08-dqn-{live,quick}   Lab 8's train_dqn with CONFIG at the lab's budget (live: 50,000
  m08-ddqn-{live,quick}  steps, greedy evaluation every 5,000; QUICK: 10,000 and 1,000), and
                         the same with target_fn=double_dqn_target. The code is
                         labs/shared/dqn_setup.py and labs/shared/dqn.py, exec'd into one
                         namespace: the lab's provided cell and its reference solutions
                         (tests/test_shared.py holds them equal), as Module 9 includes them.
  m08-sb3-{live,quick}   Stable-Baselines3 2.9.0's DQN at the matched configuration
                         (SB3_KWARGS), greedy-evaluated with the lab's evaluate_greedy on the
                         same grid. The lab never imports SB3; only this script does.

  m08-seats              Seats 0-19 at the live budget, as their lab runs: DQN (Exercise 3)
                         and Double DQN (Exercise 4) at RUN_SEED = (seat + 1) * 1000.

Each band record holds per seed the evaluation curves (greedy return, the discounted return G_0 and
Q(s0) = max_a Q(s0, a), means of 10 episodes from reset(seed=EVAL_SEED + i)), the wall-clock
seconds of the run, and max Q at the lab's 32 probe states at the end (what Exercise 4 prints).
The reference runs call train_dqn as the lab's run cells do. Logging options never touch the
training RNGs: `--reproduce NAME --m09-logging` reruns with Module 9's logging (log_every 500,
8 probe states, three snapshots) and requires the same curves, so Module 9's healthy runs at
seeds 0-2 are these runs. The SB3 records also check that SB3's buffer stores the 500-step
truncations as not done (B74). torch runs with TORCH_THREADS threads (Colab's CPU runtime has
2 vCPUs); laptop seconds are never Colab seconds.

Figures (live budget, from the records runs/bands.yml names):

  08-q-vs-mc-dqn         Q(s0) and G_0 of the greedy policy against steps, every seed, for DQN
  08-q-vs-mc-ddqn        and for Double DQN (two subfigure panels), with (sync lag) exact.
  08-final-returns       The reference DQN's and SB3's final greedy returns, every seed, with
                         IQM and 95% bootstrap intervals and the rule's call.

Every build runs quoted_numbers(): each number the director's handoff lists for the page, the
predict_answers, the knowledge checks and the lab text is asserted (exact computations), and
report() prints what the records say for the numbers still pending (B70, C34). --check never
reruns anything: it reads the filed records only, and imports neither torch nor SB3.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_protocol as tp  # noqa: E402
from make_figures_m01 import MIN_FONT, WIDTH_IN, Theme, rc, svg_bytes, themes  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.evaluate import compare  # noqa: E402
from prl.lab import _default_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
RUNS = ROOT / "runs"
SHARED = ROOT / "labs" / "shared"
LAB = ROOT / "labs" / "src" / "08-dqn.py"
NOTEBOOK = ROOT / "notebooks" / "08-dqn.ipynb"
SCRIPT = "scripts/make_figures_m08.py"
PX = 1 / 96

SEEDS = [0, 1, 2, 3, 4]
TORCH_THREADS = 2
GAMMA = 0.99
# Lab 8's settings (its first code cell); the bands match on steps and eval_every.
BUDGETS = {
    "live": dict(steps=50_000, log_every=1_000, eval_every=5_000, eval_episodes=10),
    "quick": dict(steps=10_000, log_every=500, eval_every=1_000, eval_episodes=10),
}
ALGOS = ("dqn", "ddqn", "sb3")
EXPERIMENTS = [f"m08-{a}-{b}" for a in ALGOS for b in BUDGETS]
# m08-seats: what seats 0-19 compute at the live budget (their training seed is
# (seat + 1) * 1000, Lab 8's RUN_SEED), DQN and Double DQN.
SEATS = list(range(20))
SEATS_EXPERIMENT = "m08-seats"
EXPERIMENTS.append(SEATS_EXPERIMENT)
# Module 9's logging (its handoff, section 3), for --reproduce --m09-logging only.
M09_LOG_EVERY, M09_PROBES = 500, 8
# SB3 2.9.0's DQN with the lab's CONFIG on every knob SB3 has (day-3 decision 5).
SB3_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=10_000,
    learning_starts=1_000,
    batch_size=64,
    tau=1.0,
    gamma=GAMMA,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10.0,
    policy_kwargs=dict(net_arch=[64, 64]),  # ReLU and Adam with torch's defaults: SB3's own
    device="cpu",
)
SB3_UNMATCHED = [
    "SB3 acts uniformly at random for the first learning_starts = 1,000 steps; the lab's "
    "epsilon-greedy uses the linear schedule from step 1 (epsilon 0.905 at step 1,000 live, "
    "0.525 QUICK), acting greedily on the untrained network the rest of the time",
    "SB3 copies the target network in _on_step, before that rollout's gradient step; the lab "
    "copies after the gradient step of the same step (both copy every 500 steps)",
    "SB3 updates epsilon after each step from the progress so far; the lab computes it before "
    "acting (one step apart)",
    "random streams differ: SB3 seeds Python, NumPy's global generator and torch, samples the "
    "buffer with np.random.randint and explores with np.random.rand and the action space's "
    "sample; the lab uses one np.random.default_rng(seed) for both. The initial weights and "
    "the first start state are the same for the same seed (torch.manual_seed(seed) before "
    "the same layers; per_seed.same_initial_weights), the draws after that are not",
    "SB3 wraps the environment in a Monitor and a DummyVecEnv (auto-reset, the terminal "
    "observation kept as s'); the loss is SB3's smooth_l1_loss (Huber, psi = 1, the batch "
    "mean), as in the lab",
]


# ------------------------------------------------------------------ the experiments
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def lab_namespace() -> dict:
    """Lab 8's provided code and reference solutions (labs/shared), as Module 9 includes them."""
    import gymnasium as gym

    ns: dict = {}
    for name in ("dqn_setup.py", "dqn.py"):
        exec(compile((SHARED / name).read_text(), f"labs/shared/{name}", "exec"), ns)  # noqa: S102
    ns["make_cartpole"] = lambda: gym.make("CartPole-v1")  # the lab's make_cartpole
    src = LAB.read_text()
    for label, b in BUDGETS.items():  # the budgets are the lab's
        words = ", ".join(f"{k}={v:_}" for k, v in b.items())
        assert f"dict({words})" in src, f"the lab's {label} settings are not {b}"
    assert 'return gym.make("CartPole-v1")' in src
    return ns


def _f32(x) -> float:
    """A float at float32 precision, written short (str(np.float32) round-trips)."""
    return float(str(np.float32(x)))


def _f32_list(a) -> list:
    return [_f32(v) if math.isfinite(v) else None for v in np.asarray(a, dtype=float).ravel()]


def _probe_max(ns: dict, q_net) -> float:
    """Mean over the lab's 32 probe states of max_a Q: what Exercise 4 prints at the end."""
    import torch

    with torch.no_grad():
        q = q_net(torch.as_tensor(ns["probe_states"](ns["make_cartpole"], 32)))
    return _f32(q.max(dim=1).values.mean())


def run_reference(ns: dict, algo: str, label: str, seeds=SEEDS, **log_options) -> tuple:
    """(per_seed, budget) of the lab's DQN (algo "dqn") or Double DQN ("ddqn"), called as
    Exercise 4's and 5's run cells call train_dqn (only the seed differs).

    log_options override the logger's (Module 9's thinned logging, say): they never touch the
    training RNGs, so the curves stay the same (--reproduce checks it with --m09-logging).
    """
    import torch

    torch.set_num_threads(TORCH_THREADS)
    b = BUDGETS[label]
    target_fn = ns["double_dqn_target"] if algo == "ddqn" else None
    per: dict[str, list] = {k: [] for k in ("return", "disc_return", "q0", "seconds")}
    per["q_probe_end"], per["episodes"] = [], []
    for seed in seeds:
        options = dict(
            log_every=b["log_every"],
            eval_every=b["eval_every"],
            eval_episodes=b["eval_episodes"],
            verbose=False,
        )
        options.update(log_options)
        start = time.perf_counter()
        _, log = ns["train_dqn"](
            ns["make_cartpole"], ns["CONFIG"], seed, b["steps"], target_fn=target_fn, **options
        )
        per["seconds"].append(round(time.perf_counter() - start, 2))
        grid = list(range(b["eval_every"], b["steps"] + 1, b["eval_every"]))
        assert log["eval_step"].tolist() == grid
        for key in ("return", "disc_return", "q0"):
            per[key].append(_f32_list(log[f"eval_{key}"]))
        if "probe_obs" not in log_options:  # the lab's 32 probe states, as Exercise 4 prints
            per["q_probe_end"].append(_f32(log["q_probe"][-1].max(axis=1).mean()))
        per["episodes"].append(len(log["ep_length"]))
    budget = {
        "label": label,
        **b,
        "grid": list(range(b["eval_every"], b["steps"] + 1, b["eval_every"])),
        "config": dataclasses.asdict(ns["CONFIG"]),
    }
    return per, budget


def _sb3_truncation(model) -> dict | None:
    """B74: if the transition SB3 stored last was cut by the time limit, what train() sees.

    Called from a callback's _on_step, which SB3 runs before it stores the current step, so
    slot pos - 1 holds the previous step and model._last_obs is still the observation after
    it (the reset one, if the episode ended). SB3's _get_samples draws from NumPy's global
    generator; its state is saved and restored, so the run is unchanged.
    """
    buf = model.replay_buffer
    if buf.pos == 0 and not buf.full:
        return None
    i = (buf.pos - 1) % buf.buffer_size
    if buf.timeouts[i, 0] < 0.5:
        return None
    state = np.random.get_state()
    sample = buf._get_samples(np.array([i]))
    np.random.set_state(state)
    return {
        "step": int(model.num_timesteps) - 1,
        "stored_done": float(buf.dones[i, 0]),
        "sampled_done": float(sample.dones.numpy()[0, 0]),
        "next_obs_is_reset_obs": bool(np.allclose(buf.next_observations[i, 0], model._last_obs[0])),
    }


def run_sb3(ns: dict, label: str, seeds=SEEDS) -> tuple[dict, dict]:
    """(per_seed, budget) of SB3's DQN at the matched configuration."""
    import stable_baselines3
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback

    assert stable_baselines3.__version__ == "2.9.0", stable_baselines3.__version__
    torch.set_num_threads(TORCH_THREADS)
    b = BUDGETS[label]
    cfg = ns["CONFIG"]
    matched = dict(
        learning_rate=cfg.lr,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        target_update_interval=cfg.target_sync,
        exploration_fraction=cfg.eps_fraction,
        exploration_initial_eps=cfg.eps_start,
        exploration_final_eps=cfg.eps_end,
        max_grad_norm=cfg.max_grad_norm,
    )
    assert all(SB3_KWARGS[k] == v for k, v in matched.items()), "SB3_KWARGS drifted from CONFIG"
    assert SB3_KWARGS["policy_kwargs"]["net_arch"] == list(cfg.hidden)
    assert cfg.huber_threshold == 1.0  # SB3's smooth_l1_loss has beta = 1

    per: dict[str, list] = {k: [] for k in ("return", "disc_return", "q0", "seconds")}
    per["q_probe_end"], per["truncations"], per["same_initial_weights"] = [], [], []

    def evaluate(model, out):
        ev = ns["evaluate_greedy"](model.q_net, ns["make_cartpole"], b["eval_episodes"], GAMMA)
        out["step"].append(model.num_timesteps)
        for key, src in (("return", "returns"), ("disc_return", "disc_returns"), ("q0", "q0")):
            out[key].append(float(np.mean(ev[src])))

    class Evaluate(BaseCallback):
        """Greedy evaluation every eval_every steps, after that step's gradient step: SB3
        trains between a rollout's end and the next rollout's start."""

        def __init__(self, out):
            super().__init__()
            self.out = out

        def _on_rollout_start(self) -> None:
            t = self.model.num_timesteps
            if t and t % b["eval_every"] == 0 and t not in self.out["step"]:
                evaluate(self.model, self.out)

        def _on_step(self) -> bool:
            cut = _sb3_truncation(self.model)
            if cut and cut["step"] not in [c["step"] for c in self.out["cut"]]:
                self.out["cut"].append(cut)
            return True

    for seed in seeds:
        out = {k: [] for k in ("step", "return", "disc_return", "q0", "cut")}
        torch.manual_seed(seed)  # as train_dqn does before building its networks
        ours = ns["make_q_net"](4, 2, cfg)
        start = time.perf_counter()
        model = DQN("MlpPolicy", ns["make_cartpole"](), seed=seed, **SB3_KWARGS)
        theirs = list(model.q_net.parameters())
        per["same_initial_weights"].append(
            len(theirs) == len(list(ours.parameters()))
            and all(torch.equal(a, b) for a, b in zip(ours.parameters(), theirs, strict=False))
        )
        model.learn(total_timesteps=b["steps"], callback=Evaluate(out))
        evaluate(model, out)  # the last point, after the last gradient step
        per["seconds"].append(round(time.perf_counter() - start, 2))
        grid = list(range(b["eval_every"], b["steps"] + 1, b["eval_every"]))
        assert out["step"] == grid, out["step"]
        for key in ("return", "disc_return", "q0"):
            per[key].append(_f32_list(out[key]))
        per["q_probe_end"].append(_probe_max(ns, model.q_net))
        per["truncations"].append(out["cut"])
    budget = {
        "label": label,
        **b,
        "grid": list(range(b["eval_every"], b["steps"] + 1, b["eval_every"])),
        "sb3": {k: v for k, v in SB3_KWARGS.items()},
    }
    return per, budget


def run_seats(ns: dict, seats=SEATS) -> tuple[dict, dict]:
    """(per_seed, budget) of every seat's two live runs, as Exercises 3-5 compute them.

    Exercise 3 trains DQN and Exercise 4 Double DQN at the seat's RUN_SEED, with the lab's
    live settings; Exercise 5 evaluates both final networks greedily from the same 10 start
    states, which gives the log's last evaluation again (both are deterministic). The room
    code's scalars follow: dqn_return and ddqn_return (the last greedy return), dqn_q0 and
    ddqn_q0 (Q(s0)), dqn_q0_gap (Q(s0) - G_0).
    """
    import torch

    torch.set_num_threads(TORCH_THREADS)
    b = BUDGETS["live"]
    per: dict[str, list] = {}
    for seat in seats:
        seed = (seat + 1) * 1000
        for algo, target_fn in (("dqn", None), ("ddqn", ns["double_dqn_target"])):
            start = time.perf_counter()
            _, log = ns["train_dqn"](
                ns["make_cartpole"],
                ns["CONFIG"],
                seed,
                b["steps"],
                target_fn=target_fn,
                log_every=b["log_every"],
                eval_every=b["eval_every"],
                eval_episodes=b["eval_episodes"],
                verbose=False,
            )
            per.setdefault(f"{algo}_seconds", []).append(round(time.perf_counter() - start, 2))
            for key in ("return", "disc_return", "q0"):
                per.setdefault(f"{algo}_{key}", []).append(_f32_list(log[f"eval_{key}"]))
            q_end = _f32(log["q_probe"][-1].max(axis=1).mean())
            per.setdefault(f"{algo}_q_probe_end", []).append(q_end)
    budget = {
        "label": "live",
        **b,
        "grid": list(range(b["eval_every"], b["steps"] + 1, b["eval_every"])),
        "seats": list(seats),
        "seed_rule": "(seat + 1) * 1000",
        "config": dataclasses.asdict(ns["CONFIG"]),
    }
    return per, budget


NOTES = {
    "dqn": (
        "Lab 8's train_dqn with CONFIG (labs/shared/dqn_setup.py and labs/shared/dqn.py, the "
        "lab's provided cell and reference solutions), seeds 0-4"
    ),
    "ddqn": (
        "Lab 8's train_dqn with CONFIG and target_fn=double_dqn_target (labs/shared), seeds 0-4"
    ),
    "sb3": (
        "Stable-Baselines3 2.9.0 DQN at the lab's configuration on every knob it has "
        "(budget.sb3), greedy-evaluated with the lab's evaluate_greedy, seeds 0-4"
    ),
}
SEATS_NOTE = (
    "What seats 0-19 compute at the live budget: Exercise 3's DQN and Exercise 4's Double DQN "
    "(Lab 8's train_dqn with CONFIG, labs/shared), each seat at its RUN_SEED (seat + 1) * "
    "1000, so `seeds` lists 1000, 2000, ..., 20000. per_seed lists are indexed like `seeds`: "
    "for dqn and ddqn, the greedy return, G_0 (disc_return) and Q(s0) on budget.grid (the last "
    "point is what Exercise 5 evaluates again and the room code reports), max Q at the lab's 32 "
    "probe states at the end, and the run's wall-clock seconds."
)


def experiment_record(
    name: str, per: dict, budget: dict, seconds: float, git, load: float, seeds=SEEDS
) -> dict:
    import nbformat

    algo, label = ("seats", "live") if name == SEATS_EXPERIMENT else name.split("-")[1:]
    rt = runtime.detect()
    git_sha, dirty = git
    meta = nbformat.read(NOTEBOOK, as_version=4).metadata.get("prl", {})
    b = BUDGETS[label]
    tail = (
        f"Wall-clock on {where()}, {TORCH_THREADS} torch threads, runs one after another "
        f"(load average {load:.1f} at the start: other jobs shared the machine): laptop "
        "seconds, not Colab seconds."
    )
    note = f"{SEATS_NOTE} {tail}" if algo == "seats" else None
    note = note or (
        f"{NOTES[algo]}, at the {label} budget ({b['steps']:,} steps, greedy evaluation of "
        f"{b['eval_episodes']} episodes every {b['eval_every']:,}). per_seed lists are indexed "
        "like `seeds`: return, disc_return (G_0) and q0 (max_a Q(s0, a)) on budget.grid, the "
        "run's wall-clock seconds, q_probe_end (mean over the lab's 32 probe states of max_a Q "
        "at the end) "
        + (
            "and truncations (B74: every transition the 500-step limit cut that SB3 stored "
            "during training, with its done flag as stored and as train() samples it, and "
            "whether its s' is the reset observation). "
            if algo == "sb3"
            else "and episodes. "
        )
        + tail
    )
    args = {
        "notebook": "notebooks/08-dqn.ipynb",
        "notebook_content_sha": meta.get("content_sha"),
        "code": {
            "labs/shared/dqn_setup.py": _sha(SHARED / "dqn_setup.py"),
            "labs/shared/dqn.py": _sha(SHARED / "dqn.py"),
        },
        "algo": algo,
        "repo_dirty": dirty,
        "test_doubles": runtime.test_doubles_used(),
    }
    if algo == "sb3":
        args["unmatched"] = SB3_UNMATCHED
    return {
        "schema": record.SCHEMA,
        "kind": "experiment",
        "experiment": name,
        "script": SCRIPT,
        "args": args,
        "prl_version": tp._prl_version(),
        "git_sha": git_sha,
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "platform": rt.platform,
        "env": _default_env(rt, "colab-cpu"),
        "hardware": {
            "cpu": rt.cpu,
            "n_cpu": rt.n_cpu,
            "ram_gb": rt.ram_gb,
            "accel": rt.accel,
            "gpu": rt.gpu,
            "arch": platform.machine(),
        },
        "torch_threads": TORCH_THREADS,
        "python": rt.python,
        "packages": runtime.package_versions(
            [*runtime.DEFAULT_PACKAGES, "stable-baselines3", "gymnasium"]
        ),
        "budget": budget,
        "seeds": list(seeds),
        "per_seed": per,
        "seconds": round(seconds, 1),
        "note": note,
    }


def where() -> str:
    rt = runtime.detect()
    return f"{_default_env(rt, 'colab-cpu')} ({rt.cpu}, {platform.machine()})"


def measure(name: str, ns: dict, seeds=SEEDS, **log_options) -> tuple[dict, dict]:
    if name == SEATS_EXPERIMENT:
        seats = SEATS if seeds is SEEDS else [s // 1000 - 1 for s in seeds]
        with contextlib.redirect_stdout(io.StringIO()):
            return run_seats(ns, seats)
    algo, label = name.split("-")[1:]
    with contextlib.redirect_stdout(io.StringIO()):
        if algo == "sb3":
            return run_sb3(ns, label, seeds)
        return run_reference(ns, algo, label, seeds, **log_options)


def warm_up(ns: dict) -> None:
    """Pay the one-time costs (torch's optimizer import, SB3's first model) before timing."""
    import torch

    torch.set_num_threads(TORCH_THREADS)
    cfg = ns["CONFIG"]
    with contextlib.redirect_stdout(io.StringIO()):
        ns["train_dqn"](ns["make_cartpole"], cfg, 99, cfg.learning_starts + 8, verbose=False)
        from stable_baselines3 import DQN

        model = DQN("MlpPolicy", ns["make_cartpole"](), seed=99, **SB3_KWARGS)
        model.learn(total_timesteps=SB3_KWARGS["learning_starts"] + 8)


def run(names: list[str], inbox: Path = tp.INBOX) -> list[Path]:
    git = tp._git_state()  # the repo as it is when the runs start
    ns = lab_namespace()
    warm_up(ns)
    paths = []
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; known: {', '.join(EXPERIMENTS)}")
        load = os.getloadavg()[0]
        start = time.perf_counter()
        per, budget = measure(name, ns)
        secs = time.perf_counter() - start
        seeds = [(i + 1) * 1000 for i in SEATS] if name == SEATS_EXPERIMENT else SEEDS
        rec = experiment_record(name, per, budget, secs, git, load, seeds)
        paths.append(tp.write_record(rec, inbox))
        times = {k: v for k, v in per.items() if k.endswith("seconds")}
        print(f"{name}: {secs:.0f} s, per seed {times}", flush=True)
    return paths


def _diff(old, new, path: str, out: list[str], floats: list[float]) -> None:
    """Exact comparison of JSON-like values, except floats (collected as |differences|)."""
    if isinstance(old, float) or isinstance(new, float):
        if old is None or new is None:
            out.append(f"{path}: {old!r} vs {new!r}")
        else:
            floats.append(abs(float(old) - float(new)))
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                out.append(
                    f"{path}[{key!r}]: only in {'the record' if key in old else 'the rerun'}"
                )
            else:
                _diff(old[key], new[key], f"{path}[{key!r}]", out, floats)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append(f"{path}: length {len(old)} vs {len(new)}")
            return
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            _diff(a, b, f"{path}[{i}]", out, floats)
        return
    if old != new:
        out.append(f"{path}: {old!r} vs {new!r}")


def reproduce(
    name: str, seeds: list[int] | None = None, tol: float = 1e-9, m09: bool = False
) -> bool:
    """Rerun one experiment (or some of its seeds) in memory; compare with the filed record.

    Wall-clock seconds are left out. On another platform or thread count, pass a tolerance:
    torch's CPU kernels may round differently there.
    """
    rec = record_for(name)
    seeds = list(rec["seeds"]) if seeds is None else seeds
    idx = [rec["seeds"].index(s) for s in seeds]
    ns = lab_namespace()
    options = {}
    if m09:
        if name.split("-")[1] == "sb3":
            raise SystemExit("--m09-logging applies to the lab's own runs only")
        steps = BUDGETS[name.split("-")[2]]["steps"]
        options = dict(
            log_every=M09_LOG_EVERY,
            probe_obs=ns["probe_states"](ns["make_cartpole"], M09_PROBES),
            snapshot_steps=(ns["CONFIG"].learning_starts + 1, steps // 2, steps),
        )
    per, _ = measure(name, ns, seeds, **options)
    new = json.loads(json.dumps(per))
    old = {k: [v[i] for i in idx] for k, v in rec["per_seed"].items()}
    for key in [k for k in old if k.endswith("seconds")] + (["q_probe_end"] if m09 else []):
        old.pop(key)
        new.pop(key, None)
    exact: list[str] = []
    floats: list[float] = []
    _diff(old, new, "per_seed", exact, floats)
    worst = max(floats, default=0.0)
    good = not exact and worst <= tol
    print(
        f"{name} seeds {seeds}: {'REPRODUCED' if good else 'DIFFERS'} against the record "
        f"({len(floats)} floats, max |diff| {worst:.3g}; {len(exact)} exact mismatches)"
    )
    for line in exact[:10]:
        print(f"  {line}")
    return good


# ------------------------------------------------------------------ the records
def band_records() -> dict[str, dict[str, str]]:
    """experiment name -> record id, as runs/bands.yml names them (one record per band)."""
    spec = yaml.safe_load((RUNS / "bands.yml").read_text()) or {}
    out: dict[str, str] = {}
    for name, entry in spec.items():
        if not name.startswith("m08_"):
            continue
        algo = name.split("_")[1]
        for label, rid in entry["records"].items():
            key = f"m08-{algo}-{label}"
            assert out.setdefault(key, rid) == rid, f"{name}: {key} names two records"
    return out


def _order(path: Path) -> tuple[str, int]:
    m = re.search(r"-exp(?:-(\d+))?\.json$", path.name)
    return (path.name[:10], int(m.group(1) or 1) if m else 0)


def record_for(name: str) -> dict:
    """The record runs/bands.yml names for a band experiment; for m08-seats, the newest."""
    if name == SEATS_EXPERIMENT:
        paths = sorted(RUNS.glob(f"*-{name}-exp*.json"), key=_order)
        if not paths:
            raise SystemExit(f"No filed record of {name}: run --run {name}, then file it.")
        rec = json.loads(paths[-1].read_text())
        rec["_id"] = paths[-1].stem
        assert rec["seeds"] == [(i + 1) * 1000 for i in SEATS], rec["_id"]
        return rec
    rid = band_records().get(name)
    if rid is None:
        raise SystemExit(f"runs/bands.yml names no record for {name}")
    rec = json.loads((RUNS / f"{rid}.json").read_text())
    assert rec["experiment"] == name and rec["seeds"] == SEEDS, rid
    rec["_id"] = rid
    return rec


def curves(name: str) -> dict:
    """The record's per-seed curves as arrays (seeds x grid), with its id and budget."""
    rec = record_for(name)
    out = {k: np.array(rec["per_seed"][k], dtype=float) for k in ("return", "disc_return", "q0")}
    out.update(
        x=np.array(rec["budget"]["grid"]),
        seconds=np.array(rec["per_seed"]["seconds"]),
        q_probe_end=np.array(rec["per_seed"]["q_probe_end"], dtype=float),
        record=rec["_id"],
        rec=rec,
    )
    return out


# ------------------------------------------------------------------ exact computations
def syncs_by(t: int, sync: int = 500, starts: int = 1_000, freq: int = 4) -> int:
    """Target copies at or before step t that follow at least one gradient step."""
    first = (starts // freq + 1) * freq  # the first gradient step
    return max(0, t // sync - (first - 1) // sync)


def sync_lag(t: int) -> float:
    """(sync lag): (1 - gamma^k) / (1 - gamma) after k = syncs_by(t) copies."""
    return (1 - GAMMA ** syncs_by(t)) / (1 - GAMMA)


def _brute_syncs(t: int, sync=500, starts=1_000, freq=4) -> int:
    grads = [s for s in range(1, t + 1) if s > starts and s % freq == 0]
    return sum(1 for s in range(sync, t + 1, sync) if grads and grads[0] <= s)


def e_max_gauss(m: int) -> float:
    """E[max of m iid N(0, 1)] by numerical integration."""
    from scipy import integrate, stats

    def f(x):
        return x * m * stats.norm.pdf(x) * stats.norm.cdf(x) ** (m - 1)

    return integrate.quad(f, -12, 12, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


def _r(x: float, d: int) -> float:
    """Round half up at d decimals, robust to float noise near the half."""
    return math.floor(x * 10**d + 0.5 + 1e-9) / 10**d


def _ring(capacity: int, n: int) -> tuple[list[int], int, int]:
    slots, ptr, size = [None] * capacity, 0, 0
    for t in range(n):
        slots[ptr] = t
        ptr, size = (ptr + 1) % capacity, min(size + 1, capacity)
    return slots, ptr, size


def _huber(x: float, psi: float = 1.0) -> tuple[float, float]:
    a = abs(x)
    return (0.5 * x * x, x) if a <= psi else (psi * (a - 0.5 * psi), psi * math.copysign(1, x))


def quoted_numbers() -> dict:
    """Every exact number the page, its folds, the predict_answers, the knowledge checks and
    the lab text quote (the director's handoff, section 6 and second pass S6), asserted."""
    g = GAMMA
    # Page section 3: the value after k copies.
    assert _r(100 * (1 - g**50), 4) == 39.4994 and _r(100 * (1 - g**50), 1) == 39.5
    assert _r(100 * (1 - g**200), 4) == 86.6020 and _r(100 * (1 - g**200), 1) == 86.6
    assert _r(1 / (1 - g), 9) == 100
    # Page section 4 (Check yourself 4.1) and the Gotchas.
    assert _r(1 + g * 99, 9) == 99.01
    g500 = (1 - g**500) / (1 - g)
    assert _r(g500, 3) == 99.343 and _r(g500, 2) == 99.34
    assert _r(g**500 / (1 - g), 5) == 0.65705 and _r(g**500 / (1 - g), 3) == 0.657
    # Page section 2 (Check yourself 2.1): five slots after seven writes; 1/5 per draw.
    assert _ring(5, 7) == ([5, 6, 2, 3, 4], 2, 5)
    # Page section 5: the demo table, and Theorem 1's floor at C = 1 after the theorem.
    gauss = {m: e_max_gauss(m) for m in (2, 4, 10)}
    assert [_r(gauss[m], 4) for m in (2, 4, 10)] == [0.5642, 1.0294, 1.5388]
    assert [_r(gauss[m], 3) for m in (2, 4, 10)] == [0.564, 1.029, 1.539]
    assert abs(gauss[2] - 1 / math.sqrt(math.pi)) < 1e-9
    assert abs(gauss[10] - 1.5) < abs(gauss[10] - 0.5) and abs(gauss[10] - 1.5) < 1  # "about 1.5"
    uniform = {m: (m - 1) / (m + 1) for m in (2, 4, 10)}
    assert [_r(uniform[m], 3) for m in (2, 4, 10)] == [0.333, 0.6, 0.818]
    floor = {m: math.sqrt(1 / (m - 1)) for m in (2, 4, 10)}
    assert [_r(floor[m], 4) for m in (2, 4, 10)] == [1.0, 0.5774, 0.3333]
    assert [_r(floor[m], 2) for m in (2, 4, 10)] == [1.0, 0.58, 0.33]
    # Page section 6's Predict and knowledge check 8.7.
    assert _r(2 / (5 + 1), 9) == _r(1 / 3, 9)
    assert round(20 * 2 / 6) == 7 and round(20 * 2 / 11) == 4
    # Exercise 1's predict_answer: 1,000 slots after 2,500 writes.
    slots, ptr, size = _ring(1000, 2500)
    written = np.array(slots)
    assert written.min() == 1500 and written.max() == 2499  # "written at steps 1,500 to 2,499"
    ages = 2499 - written
    assert ages.min() == 0 and ages.max() == 999 and len(set(ages.tolist())) == 1000
    assert int(np.sum(ages > 500)) == 499 and _r(100 * np.mean(ages > 500), 1) == 49.9
    assert (ptr, size) == (500, 1000) and sorted(slots[:ptr]) == list(range(2000, 2500))
    assert int(np.sum(2499 - np.array(slots[:ptr]) > 500)) == 0  # 0% of the colleague's draws
    # Exercise 2's predict_answer and run cell (delta = 3); knowledge check 8.3's table.
    assert [1.0, _r(1 + g * 50, 9), _r(1 + g * 50, 9)] == [1.0, 50.5, 50.5]
    assert _huber(3.0) == (2.5, 1.0)
    want = {-4.0: (3.5, -1.0), -0.5: (0.125, -0.5), 0.2: (0.02, 0.2), 3.0: (2.5, 1.0)}
    for x, (loss, grad) in want.items():
        h = _huber(x)
        assert (_r(h[0], 9), _r(h[1], 9)) == (loss, grad), x
        clipped = (min(0.5 * x * x, 0.5), x if abs(x) < 1 else 0.0)
        assert (_r(clipped[0], 9), _r(clipped[1], 9)) == {
            -4.0: (0.5, 0.0),
            -0.5: (0.125, -0.5),
            0.2: (0.02, 0.2),
            3.0: (0.5, 0.0),
        }[x]
    # Exercise 3's Predict and predict_answer, and section 5's fold: copies after the first
    # gradient step (1,004).
    for t in (10_000, 30_000, 50_000):
        assert syncs_by(t) == _brute_syncs(t)
    assert syncs_by(50_000) == 98 and _r(sync_lag(50_000), 3) == 62.654
    assert _r(sync_lag(50_000), 2) == 62.65
    assert syncs_by(10_000) == 18 and _r(sync_lag(10_000), 3) == 16.549
    assert _r(sync_lag(10_000), 2) == 16.55
    # Exercise 4's predict_answer and run cell.
    assert _r(1 + 0.9 * max(4, 3), 9) == 4.6 and _r(1 + 0.9 * 3, 9) == 3.7
    # Exercise 5's predict_answer: G_0 of a policy that survives 150 steps.
    assert _r(100 * (1 - g**150), 2) == 77.85 and _r(100 * (1 - g**150), 1) == 77.9
    # Knowledge checks 8.1, 8.2, 8.4, 8.5, 8.6.
    assert _ring(4, 10) == ([8, 9, 6, 7], 2, 4)
    assert _ring(4, 11)[1] == 3 and _ring(4, 12)[1] == 0
    assert 50_000 * 32 / 50_000 == 32
    assert (1.0, _r(1 + 0.9 * 20, 9), 1.0) == (1.0, 19.0, 1.0)
    k = syncs_by(30_000, sync=1_000)
    assert k == _brute_syncs(30_000, sync=1_000) == 29
    assert _r((1 - g**29) / (1 - g), 2) == 25.28 and _r((1 - g**29) / (1 - g), 1) == 25.3
    err = np.array([0.5, 0.5, -1.0])
    c = float(np.mean(err**2))
    assert err.sum() == 0 and c == 0.5 and 2 + math.sqrt(c / (3 - 1)) == 2.5
    online, target = np.array([3.0, 1.0, 2.0]), np.array([2.0, 4.0, 0.0])
    assert _r(0.5 + 0.9 * target.max(), 9) == 4.1
    assert _r(0.5 + 0.9 * target[online.argmax()], 9) == 2.3
    assert _r(64 - _r(sync_lag(50_000), 2), 9) == 1.35  # 64, 78 and 63 are hypothetical
    return {"gauss": gauss, "uniform": uniform, "floor": floor, "g500": g500}


# ------------------------------------------------------------------ figures
STYLE = {"q0": "solid", "disc_return": (0, (1.2, 1.6))}


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def _thousands(v: float, _pos=None) -> str:
    return f"{v / 1000:,.0f}k" if v else "0"


def q_panel(t: Theme, data: dict, algo: str) -> Figure:
    """Q(s0) and G_0 against steps, every seed, with (sync lag)."""
    fig = Figure(figsize=(WIDTH_IN / 2, 360 * PX))  # one of two panels, 340 x 360 px
    ax = fig.add_axes((0.2, 0.15, 0.76, 0.76))
    c = data[algo]
    color = t.series[0] if algo == "dqn" else t.series[1]
    for key in ("q0", "disc_return"):
        for run in c[key]:
            ax.plot(
                c["x"],
                np.round(run, 3),
                color=color,
                lw=1.3,
                ls=STYLE[key],
                marker="o" if key == "q0" else "s",
                ms=3,
            )
    grid = np.arange(0, BUDGETS["live"]["steps"] + 1, 250)
    ax.plot(
        grid,
        np.round([sync_lag(int(s)) for s in grid], 4),
        color=t.neutral,
        lw=1.6,
        ls=(0, (5, 2)),
        drawstyle="steps-post",
    )
    ax.set_xlim(0, BUDGETS["live"]["steps"])
    ax.set_ylim(0, 110)
    ax.xaxis.set_major_locator(FixedLocator([0, 25_000, 50_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Value at the start states")
    ax.set_title("DQN" if algo == "dqn" else "Double DQN", fontsize=13, color=t.text)
    # Direct labels (both panels): Q(s0) above its five lines, (sync lag) below them, G_0
    # above its cluster (record_facts asserts that the records leave that room).
    ax.text(2_000, 40, "Q(s₀), solid", color=color)
    ax.text(21_000, 10, "(sync lag), dashed", color=t.text)
    ax.text(27_000, 101, "G₀, dotted", color=color)
    return fig


def q_dqn_panel(t: Theme, data: dict) -> Figure:
    return q_panel(t, data, "dqn")


def q_ddqn_panel(t: Theme, data: dict) -> Figure:
    return q_panel(t, data, "ddqn")


def final_figure(t: Theme, data: dict) -> Figure:
    """Final greedy returns at the live budget: every seed, IQM and 95% bootstrap interval."""
    fig = Figure(figsize=(WIDTH_IN, 360 * PX))
    ax = fig.add_axes((0.12, 0.16, 0.85, 0.74))
    v = data["verdict"]
    for i, (key, summary) in enumerate((("ref", v.summary_a), ("sb3", v.summary_b))):
        vals = np.sort(np.round(data[key], 1))
        offsets = np.linspace(-0.12, 0.12, len(vals))
        color = t.series[0] if key == "ref" else t.series[2]
        marker = "o" if key == "ref" else "^"
        ax.plot(i + offsets, vals, marker, color=color, mfc="none", mew=1.6, ms=8, ls="none")
        center, lo, hi = (round(float(s), 1) for s in summary)
        ax.errorbar(
            i + 0.32,
            center,
            yerr=[[center - lo], [hi - center]],
            fmt="D",
            color=color,
            ms=6,
            capsize=6,
            lw=1.8,
        )
    ax.set_xlim(-0.6, 1.8)
    ax.set_ylim(0, 520)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1.1]))
    ax.set_xticklabels(["Reference DQN", "SB3 2.9 DQN"])
    ax.yaxis.set_major_locator(FixedLocator([0, 100, 200, 300, 400, 500]))
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_ylabel("Greedy return at 50,000 steps")
    words = {
        "tie": "Tie: the intervals overlap",
        "A": "Reference DQN better at this budget",
        "B": "SB3 DQN better at this budget",
    }[v.winner]
    ax.set_title(words, fontsize=13, color=t.text)
    handles = [
        Line2D([], [], marker="o", color=t.text, mfc="none", ls="none", ms=8, mew=1.6),
        Line2D([], [], marker="D", color=t.text, ls="-", ms=6, lw=1.8),
    ]
    labels = ["one seed (5 per side)", "IQM and 95% bootstrap interval"]
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=MIN_FONT)
    return fig


def record_facts(data: dict) -> None:
    """What the records say that the figures draw or the reviews will quote (B70, C34)."""
    for algo in ("dqn", "ddqn"):
        c = data[algo]
        x, q0, g0 = c["x"], c["q0"], c["disc_return"]
        lag = np.array([sync_lag(int(t)) for t in x])
        # C34: Q(s0) is below the greedy policy's G_0 on every seed at the end (and at every
        # point), and Q(s0) follows (sync lag) within 3 from 10,000 steps on.
        assert np.all(q0 < g0) and np.all(q0[:, -1] < g0[:, -1])
        assert np.all(np.abs(q0[:, x >= 10_000] - lag[x >= 10_000]) < 3.0)
        # Room for the direct labels: Q(s0) < 39 and G_0 > 52 up to 20,000 steps; Q(s0) > 25
        # from 21,000; G_0 < 100 everywhere.
        early = x <= 20_000
        assert q0[:, early].max() < 39 and g0[:, early].min() > 52
        assert q0[:, x >= 21_000].min() > 25 and g0.max() < 100
    want = str(np.load(ROOT / "prl/src/prl/_expected/m08_sb3_call.npz")["live"])
    v = data["final"]["verdict"]
    assert {"A": "reference", "B": "sb3"}.get(v.winner, v.winner) == want
    quoted_from_records(data)
    seat_facts()


def seat_facts() -> dict:
    """What m08-seats says (seats 0-19, live), for the predict_answers and the debrief.

    Nothing on the page quotes it yet (B70); these are the facts reported to the director,
    asserted so that a quote of them stays true.
    """
    rec = record_for(SEATS_EXPERIMENT)
    d = {k: np.array(v, dtype=float) for k, v in rec["per_seed"].items()}
    lag = sync_lag(50_000)
    q_d, g_d = d["dqn_q0"][:, -1], d["dqn_disc_return"][:, -1]
    q_dd, g_dd = d["ddqn_q0"][:, -1], d["ddqn_disc_return"][:, -1]
    out = {
        "below_dqn": int(np.sum(q_d < g_d)),
        "below_ddqn": int(np.sum(q_dd < g_dd)),
        "ddqn_lower": int(np.sum(q_dd < q_d)),
        "lower_median": _r(float(np.median(q_d - q_dd)), 2),
        "lag_dqn": _span(q_d - lag),
        "lag_ddqn": _span(q_dd - lag),
    }
    assert (out["below_dqn"], out["below_ddqn"], out["ddqn_lower"]) == (17, 18, 19), out
    assert out["lower_median"] == 0.78 and _span(q_d - q_dd) == (-0.75, 2.22)
    assert out["lag_dqn"] == (1.15, 3.10) and out["lag_ddqn"] == (0.53, 2.20)
    # Where Q(s0) is above G_0, the greedy policy plays poorly: return below 91, G_0 below 60.
    for q, g, ret in ((q_d, g_d, d["dqn_return"][:, -1]), (q_dd, g_dd, d["ddqn_return"][:, -1])):
        assert np.all(ret[q > g] < 91) and np.all(g[q > g] < 60)
    # Exercise 3's predict_answer: max Q at the probe states within 5 above (sync lag), 20 of 20.
    assert np.all((d["dqn_q_probe_end"] - lag > 0) & (d["dqn_q_probe_end"] - lag < 5))
    return out


def _span(a, d: int = 2) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    return (_r(a.min(), d), _r(a.max(), d))


def quoted_from_records(data: dict) -> None:
    """Every number the page, its folds and alt text, and the predict_answers quote from the
    band records (second pass S6), asserted."""
    live = {a: data[a] for a in ALGOS}
    quick = {a: curves(f"m08-{a}-quick") for a in ALGOS}
    lag50, lag10 = sync_lag(50_000), sync_lag(10_000)
    # Exercise 3's predict_answer: max Q at the 32 probe states, within 5 of (sync lag), above.
    assert _span(live["dqn"]["q_probe_end"]) == (64.22, 65.19)
    assert _span(quick["dqn"]["q_probe_end"]) == (17.12, 17.57)
    assert all(0 < q - lag50 < 5 for q in live["dqn"]["q_probe_end"])
    assert all(0 < q - lag10 < 5 for q in quick["dqn"]["q_probe_end"])
    # Exercise 5's predict_answer and section 5's fold.
    q_d, g_d = live["dqn"]["q0"][:, -1], live["dqn"]["disc_return"][:, -1]
    q_dd, g_dd = live["ddqn"]["q0"][:, -1], live["ddqn"]["disc_return"][:, -1]
    assert _span(q_d) == (64.20, 65.17) and _span(g_d) == (68.52, 84.40)
    assert _span(q_dd) == (63.48, 64.31) and _span(g_dd) == (70.35, 88.89)
    assert _span(q_d - lag50) == (1.55, 2.52) and _span(q_dd - lag50) == (0.83, 1.65)
    lower = q_d - q_dd  # Double DQN lower at the same seed: 5 of 5 live
    assert np.all(lower > 0) and _span(lower) == (0.05, 1.60)
    lower_q = quick["dqn"]["q0"][:, -1] - quick["ddqn"]["q0"][:, -1]
    assert int(np.sum(lower_q > 0)) == 4 and _span(lower_q[lower_q > 0]) == (0.23, 0.63)
    assert lower_q[3] < 0 and _r(-lower_q[3], 2) == 0.01  # above by 0.01 at seed 3
    gaps = np.concatenate([g_d - q_d, g_dd - q_dd])
    assert _span(gaps, 1) == (3.4, 25.2)
    for runs in (live, quick):  # below G_0 on every seed and evaluation, SB3 included
        assert all(np.all(runs[a]["q0"] < runs[a]["disc_return"]) for a in ALGOS)
    # Section 5's figures, alt text: Q(s0) from about 8 (7.6-8.1) to about 64; G_0 about
    # 53-97 (DQN) and 67-99 (Double DQN), every evaluation.
    first = np.concatenate([live["dqn"]["q0"][:, 0], live["ddqn"]["q0"][:, 0]])
    assert _span(first, 1) == (7.6, 8.1)
    # "to about 64": the ends are 64.20-65.17 (DQN) and 63.48-64.31 (Double DQN), within 1.2.
    assert np.all(np.abs(np.concatenate([q_d, q_dd]) - 64) < 1.2)
    assert (round(live["dqn"]["disc_return"].min()), round(live["dqn"]["disc_return"].max())) == (
        53,
        97,
    )
    assert (round(live["ddqn"]["disc_return"].min()), round(live["ddqn"]["disc_return"].max())) == (
        67,
        99,
    )
    # Exercise 6's predict_answer and section 6's fold.
    for runs, want in (
        (live, ((145.0, 120.7, 173.4), (149.8, 136.0, 187.2))),
        (quick, ((216.9, 148.0, 243.0), (172.9, 158.2, 242.1))),
    ):
        v = compare(runs["dqn"]["return"][:, -1], runs["sb3"]["return"][:, -1])
        got = (tuple(_r(x, 1) for x in v.summary_a), tuple(_r(x, 1) for x in v.summary_b))
        assert v.winner == "tie" and got == want, got
    ref_s, sb3_s = live["dqn"]["seconds"].sum(), live["sb3"]["seconds"].sum()
    assert (_r(ref_s, 1), _r(sb3_s, 1), _r(ref_s + sb3_s, 1)) == (34.5, 48.3, 82.7)
    ref_f, sb3_f = live["dqn"]["return"][:, -1], live["sb3"]["return"][:, -1]
    assert _span(ref_f, 1) == (115.1, 185.8) and _span(sb3_f, 1) == (131.2, 206.8)
    spreads = (_r(np.ptp(ref_f), 1), _r(np.ptp(sb3_f), 1))
    assert spreads == (70.7, 75.6)  # "about 70 to 76"
    v = data["final"]["verdict"]
    assert [round(x) for x in v.summary_a] == [145, 121, 173]  # alt text, "about"
    assert [round(x) for x in v.summary_b] == [150, 136, 187]


def figure_data() -> dict:
    data = {algo: curves(f"m08-{algo}-live") for algo in ALGOS}
    ref, sb3 = data["dqn"]["return"][:, -1], data["sb3"]["return"][:, -1]
    data["final"] = {"ref": ref, "sb3": sb3, "verdict": compare(ref, sb3)}
    return data


FIGURES = {
    "08-q-vs-mc-dqn": (q_dqn_panel, True),
    "08-q-vs-mc-ddqn": (q_ddqn_panel, True),
    "08-final-returns": (final_figure, False),
}


def panel_svg_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Description": "subfigure panel"})
    return buf.getvalue()


def build() -> tuple[dict[Path, bytes], dict]:
    quoted = quoted_numbers()
    data = figure_data()
    record_facts(data)
    out: dict[Path, bytes] = {}
    for name, (draw, panel) in FIGURES.items():
        for t in themes():
            with mpl.rc_context(rc(name, t)):
                arg = data["final"] if name == "08-final-returns" else data
                fig = draw(t, arg)
                out[IMAGES / f"{name}{t.suffix}.svg"] = (panel_svg_bytes if panel else svg_bytes)(
                    fig
                )
    return out, {"quoted": quoted, "data": data}


# ------------------------------------------------------------------ the report (B70, C34)
def report() -> None:
    for label in BUDGETS:
        b = BUDGETS[label]
        c = {algo: curves(f"m08-{algo}-{label}") for algo in ALGOS}
        lag = sync_lag(b["steps"])
        print(
            f"\n{label} budget: {b['steps']:,} steps; (sync lag) after {syncs_by(b['steps'])} "
            f"copies {lag:.2f}"
        )
        for algo in ALGOS:
            d = c[algo]
            q0, g0, ret = d["q0"][:, -1], d["disc_return"][:, -1], d["return"][:, -1]
            print(f"  {algo:4s} ({d['record']})")
            print(f"       final return {np.round(ret, 1).tolist()}")
            print(f"       Q(s0) {np.round(q0, 2).tolist()}")
            print(f"       G_0   {np.round(g0, 2).tolist()}")
            print(
                f"       Q(s0) - G_0 {np.round(q0 - g0, 2).tolist()} "
                f"(below on {int(np.sum(q0 < g0))} of 5)"
            )
            print(f"       max Q at 32 probe states, end {np.round(d['q_probe_end'], 2).tolist()}")
            print(
                f"       seconds per run {d['seconds'].tolist()} (mean {d['seconds'].mean():.1f})"
            )
            if algo == "sb3":
                print(f"       truncations (B74) {d['rec']['per_seed']['truncations']}")
        ref, ddqn, sb3 = (c[a]["return"][:, -1] for a in ALGOS)
        for a, bb, names in (
            (ref, sb3, ("reference DQN", "SB3 DQN")),
            (ddqn, ref, ("Double DQN", "DQN")),
        ):
            print("  " + compare(a, bb, names=names).sentence)
        dq = compare(
            c["dqn"]["q0"][:, -1], c["ddqn"]["q0"][:, -1], names=("DQN Q(s0)", "Double DQN Q(s0)")
        )
        print("  " + dq.sentence)


def main(argv: list[str]) -> int:
    if "--run" in argv:
        names = [a for a in argv[argv.index("--run") + 1 :] if not a.startswith("--")]
        for p in run(names or EXPERIMENTS):
            print(f"Saved {p.name} in runs/inbox/ (file it with scripts/add_run_record.py).")
        return 0
    if "--reproduce" in argv:
        names = [a for a in argv[argv.index("--reproduce") + 1 :] if not a.startswith("--")]
        seeds = None
        if "--seeds" in argv:
            seeds = [int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
            names = [n for n in names if n != argv[argv.index("--seeds") + 1]]
        tol = float(argv[argv.index("--tol") + 1]) if "--tol" in argv else 1e-9
        names = [n for n in names if n in EXPERIMENTS]
        m09 = "--m09-logging" in argv
        return 0 if all([reproduce(n, seeds, tol, m09) for n in names]) else 1
    files, _ = build()
    if "--check" in argv:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m08.py")
        return 1 if stale else 0
    IMAGES.mkdir(exist_ok=True)
    for p, data in files.items():
        p.write_bytes(data)
        print(f"wrote {p.relative_to(ROOT)} ({len(data) / 1024:.0f} KiB)")
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
