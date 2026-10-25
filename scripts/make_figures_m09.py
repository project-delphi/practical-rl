"""Module 9: the recorded runs (experiment records), the figures drawn from them, and every
number the page, the predict_answers and the knowledge checks quote.

  uv run python scripts/make_figures_m09.py --run m09-broken-runs ...  # records -> runs/inbox/
  uv run python scripts/add_run_record.py                              # file them in runs/
  uv run python scripts/make_datasets.py && uv run python scripts/export_bands.py
  uv run python scripts/make_figures_m09.py            # draw images/09-*.svg, print the report
  uv run python scripts/make_figures_m09.py --check    # fail if a committed SVG differs
  uv run python scripts/make_figures_m09.py --reproduce m09-broken-runs [--seeds 0]
                                                       # rerun and compare with the filed record

Experiments. The code is labs/shared/dqn_setup.py and labs/shared/dqn.py exec'd into one
namespace, as the lab includes them, plus the lab's own provided cell "How runs A to D were
made" (its `train_agent` swaps a function or a configuration value into that namespace, never
an edit of the shared code; day-3 decision 3), the lab's probe environments (Exercise 4's and
the stretch's solution cells) and its PROBE_CONFIG. Seeds 0-9 are recording seeds, which no
seat uses (seats train with (SEED + 1) * 1000), except where a record says otherwise.

  m09-broken-runs      The healthy run and runs A-D, seeds 0-2, at the lab's live budget, with
                       Module 9's logging (log every 500 steps, 8 probe states, snapshots at the
                       first gradient step and at 25,000 and 50,000), thinned for JSON (the first
                       32 rows of each snapshot; per episode only its length and whether the time
                       limit cut it). Before the record is written, every margin of the director's
                       handoff (section 4) is asserted on every seed (scripts/m09_signals.py); a
                       failure stops the script and nothing is written. The healthy seeds must
                       reproduce Module 8's band record m08-dqn-live exactly.
  m09-sweep            CONFIG with lr = CONFIG.lr x f, f in 0.1, 0.3, 1, 3, 10, seeds 0-9, live
                       budget, greedy evaluation every 5,000 steps (10 episodes).
  m09-reference-live   The band m09_reference: CONFIG, seeds 0-9, at each of the lab's budgets.
  m09-reference-quick  The live record must equal the sweep's x1 row, and seeds 0-4 at both
                       budgets Module 8's band records, exactly (three reproductions).
  m09-probes           The five agents on probes 1, 2 and 3 (Exercise 4's run cell and the
                       stretch), 2,000 steps each with PROBE_CONFIG, on the seeds of seats 0-9
                       (what those seats see) and on recording seeds 0-9.

torch runs with TORCH_THREADS threads (the lab sets 2, as on Colab's 2 vCPUs), one run after
another after a warm-up. Seconds are laptop seconds, never Colab seconds.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m09_signals  # noqa: E402
import threshold_protocol as tp  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.lab import _default_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
RUNS_DIR = ROOT / "runs"
SHARED = ROOT / "labs" / "shared"
LAB = ROOT / "labs" / "src" / "09-debugging-evaluation.py"
NOTEBOOK = ROOT / "notebooks" / "09-debugging-evaluation.ipynb"
SCRIPT = "scripts/make_figures_m09.py"

TORCH_THREADS = 2
GAMMA = 0.99
# Lab 9's settings (its first code cell). The band m09_reference matches on steps, eval_every.
BUDGETS = {
    "live": dict(
        steps=50_000, log_every=500, eval_every=5_000, eval_episodes=10, probe_steps=2_000
    ),
    "quick": dict(
        steps=10_000, log_every=500, eval_every=1_000, eval_episodes=10, probe_steps=2_000
    ),
}
RUNS = m09_signals.RUNS
BROKEN_SEEDS = [0, 1, 2]
BAND_SEEDS = list(range(10))
LR_FACTORS = [0.1, 0.3, 1.0, 3.0, 10.0]
N_PROBE_STATES = 8
SNAPSHOT_ROWS = 32
SEATS = list(range(10))
PROBE_SEEDS = {"seats": [(s + 1) * 1000 for s in SEATS], "recording": list(range(10))}
EXPERIMENTS = [
    "m09-broken-runs",
    "m09-sweep",
    "m09-reference-live",
    "m09-reference-quick",
    "m09-probes",
]
M08_RECORDS = {
    "live": "2026-10-24-local-m08-dqn-live-exp",
    "quick": "2026-10-24-local-m08-dqn-quick-exp",
}
SCALARS = ("epsilon", "loss", "td_abs_mean", "td_abs_max", "q_max_mean", "grad_norm", "q_terminal")


# ------------------------------------------------------------------ the lab's code
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _assignments(source: str, names: tuple[str, ...]) -> str:
    """Top-level assignments to `names` in a cell, as source (PROBE_CONFIG, probes)."""
    tree = ast.parse(source)
    keep = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets)
    ]
    assert {t.id for n in keep for t in n.targets} == set(names), names
    return "\n".join(ast.get_source_segment(source, n) for n in keep)


def lab_namespace() -> dict:
    """The shared DQN as Lab 9 includes it, with the lab's bug seams and probe environments."""
    import gymnasium as gym

    ns: dict = {}
    for name in ("dqn_setup.py", "dqn.py"):
        exec(compile((SHARED / name).read_text(), f"labs/shared/{name}", "exec"), ns)  # noqa: S102
    src = LAB.read_text()
    for label, b in BUDGETS.items():  # the budgets are the lab's
        words = ", ".join(f"{k}={v:_}" for k, v in b.items())
        assert f"dict({words})" in src, f"the lab's {label} settings are not {b}"
    assert "# %% include labs/shared/dqn_setup.py\n\n# %% include labs/shared/dqn.py" in src
    ns.update(gym=gym, dataclasses=dataclasses, np=np)
    ns["lab"] = types.SimpleNamespace(solution=lambda n: lambda f: f)
    ns["make_cartpole"] = lambda: gym.make("CartPole-v1")  # as the lab's live run makes it
    assert 'lambda: gym.make("CartPole-v1")' in src
    how = m09_signals.code_cell_with("def train_agent(")
    exec(compile(how, "<lab: How runs A to D were made>", "exec"), ns)  # noqa: S102
    for role in ("ex4-sol", "stretch-sol"):
        exec(compile(m09_signals.cell_by_role(role), f"<lab {role}>", "exec"), ns)  # noqa: S102
    probe_cell = m09_signals.cell_by_role("ex4-run")
    exec(compile(_assignments(probe_cell, ("PROBE_CONFIG", "probes")), "<lab ex4-run>", "exec"), ns)  # noqa: S102
    assert list(ns["AGENTS"]) == list(RUNS)
    return ns


def _f32(x) -> float | None:
    """A float at float32 precision, written short (str(np.float32) round-trips); NaN -> None."""
    x = float(x)
    return float(str(np.float32(x))) if math.isfinite(x) else None


def _f32_nested(a) -> list:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        return _f32(arr)
    return [_f32_nested(row) for row in arr]


def _load_avg() -> float:
    return os.getloadavg()[0]


def warm_up(ns: dict) -> None:
    """Pay torch's one-time costs (the optimizer's first step) before anything is timed."""
    import torch

    torch.set_num_threads(TORCH_THREADS)
    cfg = ns["CONFIG"]
    with contextlib.redirect_stdout(io.StringIO()):
        ns["train_dqn"](ns["make_cartpole"], cfg, 99, cfg.learning_starts + 8, verbose=False)


# ------------------------------------------------------------------ m09-broken-runs
def broken_options(ns: dict) -> dict:
    b = BUDGETS["live"]
    return dict(
        log_every=b["log_every"],
        eval_every=b["eval_every"],
        eval_episodes=b["eval_episodes"],
        probe_obs=ns["probe_states"](ns["make_cartpole"], N_PROBE_STATES),
        snapshot_steps=(ns["CONFIG"].learning_starts + 1, b["steps"] // 2, b["steps"]),
        verbose=False,
    )


def thin(log: dict, steps: int) -> dict:
    """One run's log as JSON: Module 9's keys, the snapshots' first 32 rows, and per episode
    its length and whether the time limit cut it (the rest is derived exactly, and checked)."""
    length = log["ep_length"].astype(np.int64)
    end = np.cumsum(length)
    assert np.array_equal(end, log["ep_end_step"]), "episodes must tile the steps"
    assert np.array_equal(log["ep_return"], length.astype(np.float32)), "CartPole pays 1 a step"
    disc = ((1 - GAMMA ** length.astype(float)) / (1 - GAMMA)).astype(np.float32)
    assert np.allclose(log["ep_disc_return"], disc, rtol=1e-6, atol=0), "G_0 of 1-per-step"
    truncated = np.flatnonzero(~log["ep_terminated"])
    assert np.all(length[truncated] == 500), "only the 500-step limit truncates CartPole"
    b = BUDGETS["live"]
    assert log["step"].tolist() == list(range(b["log_every"], steps + 1, b["log_every"]))
    assert log["eval_step"].tolist() == list(range(b["eval_every"], steps + 1, b["eval_every"]))
    r = slice(0, SNAPSHOT_ROWS)
    out = {k: _f32_nested(log[k]) for k in SCALARS}
    out.update(
        q_probe=_f32_nested(log["q_probe"]),
        q_probe_target=_f32_nested(log["q_probe_target"]),
        snap_step=log["snap_step"].tolist(),
        snap_q=_f32_nested(log["snap_q"][:, r]),
        snap_actions=log["snap_actions"][:, r].tolist(),
        snap_gathered=_f32_nested(log["snap_gathered"][:, r]),
        snap_target=_f32_nested(log["snap_target"][:, r]),
        snap_reward=_f32_nested(log["snap_reward"][:, r]),
        snap_terminated=log["snap_terminated"][:, r].tolist(),
        ep_length=length.tolist(),
        ep_truncated=truncated.tolist(),
        eval_return=_f32_nested(log["eval_return"]),
    )
    return out


def run_broken(ns: dict) -> tuple[dict, dict, dict]:
    """(per_seed, budget, extra) of the five runs on seeds 0-2."""
    b = BUDGETS["live"]
    options = broken_options(ns)
    per: dict = {}
    for run in RUNS:
        per[run] = {"seconds": []}
        for seed in BROKEN_SEEDS:
            start = time.perf_counter()
            _, log = ns["train_agent"](
                run, ns["make_cartpole"], ns["CONFIG"], seed, b["steps"], **options
            )
            per[run]["seconds"].append(round(time.perf_counter() - start, 2))
            for key, value in thin(log, b["steps"]).items():
                per[run].setdefault(key, []).append(value)
            print(f"  {run:8s} seed {seed}: {per[run]['seconds'][-1]:.1f} s", flush=True)
    budget = {
        "label": "live",
        "steps": b["steps"],
        "log_every": b["log_every"],
        "eval_every": b["eval_every"],
        "eval_episodes": b["eval_episodes"],
        "grid": list(range(b["log_every"], b["steps"] + 1, b["log_every"])),
        "eval_grid": list(range(b["eval_every"], b["steps"] + 1, b["eval_every"])),
        "snapshot_steps": [int(s) for s in options["snapshot_steps"]],
        "snapshot_rows": SNAPSHOT_ROWS,
        "probe_states": {
            "n": N_PROBE_STATES,
            "seeds": f"reset(seed=PROBE_SEED + i), PROBE_SEED = {ns['PROBE_SEED']:_}",
            "obs": _f32_nested(options["probe_obs"]),
        },
        "gamma": GAMMA,
        "target_sync": ns["CONFIG"].target_sync,
        "config": dataclasses.asdict(ns["CONFIG"]),
    }
    return per, budget, {}


def _m08_curves(label: str) -> dict:
    rec = json.loads((RUNS_DIR / f"{M08_RECORDS[label]}.json").read_text())
    return {s: rec["per_seed"]["return"][i] for i, s in enumerate(rec["seeds"])}


def check_healthy_reproduces_m08(per: dict) -> str:
    """The healthy seeds are Module 8's band runs: their greedy curves must match exactly."""
    m08 = _m08_curves("live")
    for i, seed in enumerate(BROKEN_SEEDS):
        got = per["healthy"]["eval_return"][i]
        assert got == m08[seed], f"healthy seed {seed}: {got} != m08-dqn-live {m08[seed]}"
    return f"healthy seeds {BROKEN_SEEDS} equal {M08_RECORDS['live']} exactly"


# ------------------------------------------------------------------ sweep and band
def _curves(ns: dict, config, seeds: list[int], label: str) -> dict:
    """Greedy-evaluation curves of train_dqn with `config`, as the lab's live run calls it."""
    b = BUDGETS[label]
    per: dict = {k: [] for k in ("return", "disc_return", "q0", "seconds", "episodes")}
    for seed in seeds:
        start = time.perf_counter()
        _, log = ns["train_dqn"](
            ns["make_cartpole"],
            config,
            seed,
            b["steps"],
            log_every=b["log_every"],
            eval_every=b["eval_every"],
            eval_episodes=b["eval_episodes"],
            verbose=False,
        )
        per["seconds"].append(round(time.perf_counter() - start, 2))
        assert log["eval_step"].tolist() == list(
            range(b["eval_every"], b["steps"] + 1, b["eval_every"])
        )
        for key in ("return", "disc_return", "q0"):
            per[key].append(_f32_nested(log[f"eval_{key}"]))
        per["episodes"].append(len(log["ep_length"]))
    return per


def _band_budget(ns: dict, label: str) -> dict:
    b = BUDGETS[label]
    return {
        "label": label,
        "steps": b["steps"],
        "log_every": b["log_every"],
        "eval_every": b["eval_every"],
        "eval_episodes": b["eval_episodes"],
        "grid": list(range(b["eval_every"], b["steps"] + 1, b["eval_every"])),
        "config": dataclasses.asdict(ns["CONFIG"]),
    }


def run_sweep(ns: dict) -> tuple[dict, dict, dict]:
    per = {}
    for f in LR_FACTORS:
        cfg = dataclasses.replace(ns["CONFIG"], lr=ns["CONFIG"].lr * f)
        per[f"x{f:g}"] = _curves(ns, cfg, BAND_SEEDS, "live")
        print(f"  lr x{f:g}: {sum(per[f'x{f:g}']['seconds']):.0f} s", flush=True)
    budget = _band_budget(ns, "live")
    budget.update(
        lr_factors=LR_FACTORS,
        lr=[ns["CONFIG"].lr * f for f in LR_FACTORS],
        configs=[f"x{f:g}" for f in LR_FACTORS],
    )
    return per, budget, {}


def run_reference(ns: dict, label: str) -> tuple[dict, dict, dict]:
    per = _curves(ns, ns["CONFIG"], BAND_SEEDS, label)
    m08 = _m08_curves(label)
    for i, seed in enumerate(BAND_SEEDS):
        if seed in m08:
            assert per["return"][i] == m08[seed], f"{label} seed {seed} differs from Module 8's"
    return per, _band_budget(ns, label), {}


# ------------------------------------------------------------------ probes
PROBES = {
    "probe 1": ("ConstantRewardEnv", [[0.0]]),
    "probe 2": ("ObsRewardEnv", [[-1.0], [1.0]]),
    "probe 3": ("TwoStepEnv", [[0.0], [1.0]]),
}


def run_probes(ns: dict) -> tuple[dict, dict, dict]:
    """Every agent on probes 1-3 as Exercise 4's run cell (and the stretch's) trains it."""
    import torch

    steps = BUDGETS["live"]["probe_steps"]
    assert BUDGETS["quick"]["probe_steps"] == steps
    lab_probes = {k: (cls.__name__, obs.tolist()) for k, (cls, obs) in ns["probes"].items()}
    assert all(lab_probes[k] == PROBES[k] for k in lab_probes), lab_probes
    per: dict = {}
    for group, seeds in PROBE_SEEDS.items():
        per[group] = {}
        for run in RUNS:
            per[group][run] = {"seconds": []}
            for probe, (cls_name, obs) in PROBES.items():
                vals = []
                obs_arr = np.array(obs, np.float32)
                for seed in seeds:
                    start = time.perf_counter()
                    q_net, _ = ns["train_agent"](
                        run,
                        ns[cls_name],
                        ns["PROBE_CONFIG"],
                        seed,
                        steps,
                        log_every=100,
                        probe_obs=obs_arr,
                        verbose=False,
                    )
                    per[group][run]["seconds"].append(round(time.perf_counter() - start, 3))
                    with torch.no_grad():
                        got = q_net(torch.as_tensor(obs_arr))[:, 0].numpy()
                    vals.append(_f32_nested(got))
                per[group][run][probe] = vals
            print(f"  {group} {run}: {sum(per[group][run]['seconds']):.0f} s", flush=True)
    budget = {
        "label": "live and quick (the same)",
        "probe_steps": steps,
        "log_every": 100,
        "probe_config": dataclasses.asdict(ns["PROBE_CONFIG"]),
        "probes": {k: {"env": v[0], "obs": v[1]} for k, v in PROBES.items()},
        "seeds": PROBE_SEEDS,
    }
    return per, budget, {}


# ------------------------------------------------------------------ records
NOTES = {
    "m09-broken-runs": (
        "Module 9's recorded runs: the healthy run and runs A-D (budget/args.runs), seeds 0-2, at "
        "the lab's live budget, made by the lab's own train_agent (labs/shared exec'd into one "
        "namespace; a configuration change or one function swapped in, never an edit of the "
        "shared code). per_seed.<run>.<key> lists are indexed like `seeds`: the logger's scalars "
        "every 500 steps (None where no gradient step fell in the interval), Q at the 8 probe "
        "states from both networks, the snapshots' first 32 rows, per episode its length "
        "(ep_end_step is their running sum, ep_return the length, ep_disc_return "
        "(1 - 0.99^L)/0.01, ep_terminated false only at ep_truncated; each checked against the "
        "log before thinning), the greedy return every 5,000 steps (10 episodes) and the run's "
        "seconds. Every margin of the director's handoff, section 4, held on every seed before "
        "this record was written (scripts/m09_signals.py). "
    ),
    "m09-sweep": (
        "Module 8's reference DQN (labs/shared, CONFIG) with only the learning rate changed, "
        "lr = CONFIG.lr x f for f in budget.lr_factors, seeds 0-9, at Lab 9's live budget. "
        "per_seed.x<f> lists are indexed like `seeds`: greedy return, disc_return (G_0) and q0 "
        "(max_a Q(s0, a)) on budget.grid (means of 10 episodes from reset(seed=EVAL_SEED + i)), "
        "the run's wall-clock seconds and episodes. The x1 row equals m09-reference-live. "
    ),
    "m09-reference-live": (
        "The band m09_reference at the live budget: Module 8's reference DQN (labs/shared, "
        "CONFIG), seeds 0-9. per_seed lists are indexed like `seeds`: greedy return, "
        "disc_return (G_0) and q0 on budget.grid (10 episodes each), seconds and episodes. Seeds "
        "0-4 equal Module 8's m08-dqn-live record exactly. "
    ),
    "m09-reference-quick": (
        "The band m09_reference at the QUICK budget: Module 8's reference DQN (labs/shared, "
        "CONFIG), seeds 0-9. per_seed lists are indexed like `seeds`: greedy return, "
        "disc_return (G_0) and q0 on budget.grid (10 episodes each), seconds and episodes. Seeds "
        "0-4 equal Module 8's m08-dqn-quick record exactly. "
    ),
    "m09-probes": (
        "Exercise 4's probe runs and the stretch's: each of the five agents (the lab's AGENTS and "
        "train_agent) on probes 1-3 (the lab's solution environments), 2,000 steps each with "
        "the lab's PROBE_CONFIG, then Q at the probe observations (budget.probes). "
        "per_seed.<group>.<run>.<probe> lists one entry per seed of budget.seeds.<group>: "
        "'seats' are the seeds seats 0-9 train with ((SEED + 1) * 1000, what those seats see on "
        "this platform), 'recording' seeds 0-9. `seeds` lists both groups. "
    ),
}


def where() -> str:
    rt = runtime.detect()
    return f"{_default_env(rt, 'colab-cpu')} ({rt.cpu}, {platform.machine()})"


def experiment_record(
    name: str, per: dict, budget: dict, seeds: list[int], seconds: float, git, load: float
) -> dict:
    import nbformat

    rt = runtime.detect()
    git_sha, dirty = git
    meta = nbformat.read(NOTEBOOK, as_version=4).metadata.get("prl", {})
    note = (
        NOTES[name]
        + f"Wall-clock on {where()}, {TORCH_THREADS} torch threads, runs one after another "
        f"after a warm-up (load average {load:.1f} at the start: other jobs shared the "
        "machine): laptop seconds, not Colab seconds."
    )
    args = {
        "notebook": "notebooks/09-debugging-evaluation.ipynb",
        "notebook_content_sha": meta.get("content_sha"),
        "code": {
            "labs/shared/dqn_setup.py": _sha(SHARED / "dqn_setup.py"),
            "labs/shared/dqn.py": _sha(SHARED / "dqn.py"),
            "labs/src/09-debugging-evaluation.py": _sha(LAB),
        },
        "repo_dirty": dirty,
        "test_doubles": runtime.test_doubles_used(),
    }
    if name in ("m09-broken-runs", "m09-probes"):
        args["runs"] = {
            "healthy": "CONFIG and the shared code, unchanged",
            "A": "dataclasses.replace(CONFIG, lr=CONFIG.lr * 100)",
            "B": "gather_q rebound to gather_q_dim0 (q_values.gather(0, ...)): wrong dimension",
            "C": "td_target rebound to td_target_no_mask (r + gamma * q_next): no terminal mask",
            "D": "dataclasses.replace(CONFIG, target_sync=None): the target is never synced",
        }
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
        "packages": runtime.package_versions([*runtime.DEFAULT_PACKAGES, "gymnasium"]),
        "budget": budget,
        "seeds": seeds,
        "per_seed": per,
        "seconds": round(seconds, 1),
        "note": note,
    }


def measure(name: str, ns: dict) -> tuple[dict, dict, list[int]]:
    if name == "m09-broken-runs":
        per, budget, _ = run_broken(ns)
        return per, budget, BROKEN_SEEDS
    if name == "m09-sweep":
        per, budget, _ = run_sweep(ns)
        return per, budget, BAND_SEEDS
    if name.startswith("m09-reference-"):
        per, budget, _ = run_reference(ns, name.rsplit("-", 1)[1])
        return per, budget, BAND_SEEDS
    if name == "m09-probes":
        per, budget, _ = run_probes(ns)
        return per, budget, PROBE_SEEDS["seats"] + PROBE_SEEDS["recording"]
    raise SystemExit(f"unknown experiment {name!r}; known: {', '.join(EXPERIMENTS)}")


def broken_arrays(rec: dict) -> dict:
    """The dataset's arrays from a record, by the NumPy builder, without its final assertion
    (so every margin can be printed before anything stops)."""
    import make_datasets

    real = m09_signals.assert_margins
    m09_signals.assert_margins = lambda data: None
    try:
        return make_datasets.m09_broken_runs({make_datasets.M09_BROKEN: rec})
    finally:
        m09_signals.assert_margins = real


def verify_broken(rec: dict) -> m09_signals.Margins:
    """Every margin of the handoff on every seed, printed; the caller stops on a failure."""
    m = m09_signals.margins(broken_arrays(rec))
    print("\n".join(m.lines))
    return m


def run(names: list[str], inbox: Path = tp.INBOX, hold: Path | None = None) -> list[Path]:
    git = tp._git_state()  # the repo as it is when the runs start
    ns = lab_namespace()
    warm_up(ns)
    paths = []
    done: dict[str, dict] = {}
    for name in names:
        load = _load_avg()
        start = time.perf_counter()
        print(f"{name}: running", flush=True)
        per, budget, seeds = measure(name, ns)
        secs = time.perf_counter() - start
        rec = experiment_record(name, per, budget, seeds, secs, git, load)
        if name == "m09-broken-runs":
            print(check_healthy_reproduces_m08(per))
            m = verify_broken(rec)
            if not m.ok():
                if hold is not None:  # kept outside runs/: no dataset may be built from it
                    hold.mkdir(parents=True, exist_ok=True)
                    path = hold / f"{name}-FAILED-margins.json"
                    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
                    print(f"held (not a filed record): {path}")
                raise SystemExit(
                    "A margin failed; no record written. Report it, do not change seeds:\n  "
                    + "\n  ".join(m.hard)
                )
        if name == "m09-reference-live" and "m09-sweep" in done:
            assert done["m09-sweep"]["per_seed"]["x1"]["return"] == per["return"], (
                "the sweep's x1 row must equal m09-reference-live"
            )
            print("m09-reference-live equals the sweep's x1 row exactly")
        done[name] = rec
        paths.append(tp.write_record(rec, inbox))
        print(f"{name}: {secs:.0f} s -> {paths[-1].name}", flush=True)
    return paths


# ------------------------------------------------------------------ main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", nargs="+", metavar="EXPERIMENT", help="record experiments")
    ap.add_argument("--hold", type=Path, help="where a record whose margins fail is kept")
    args = ap.parse_args(argv)
    if args.run:
        run(args.run, hold=args.hold)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
