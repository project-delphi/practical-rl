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
                       limit cut it). Every margin of the director's handoff (section 4) is
                       asserted on every seed (scripts/m09_signals.py) and stated in the record's
                       note; a failure stops the script with the record held outside the
                       inbox. The healthy seeds must reproduce Module 8's band record
                       m08-dqn-live exactly.
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
        "seconds. "
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


def margin_note(m: m09_signals.Margins) -> str:
    """The record's own statement of the margins (scripts/m09_signals.py), first in its note."""
    if m.ok():
        return (
            "Every margin of the director's handoff (section 4) held on every seed when this "
            "record was made (scripts/m09_signals.py). "
        )
    return (
        "MARGINS: every margin of the director's handoff (section 4) held on every seed except "
        + "; ".join(m.hard)
        + ". scripts/make_datasets.py builds no dataset from this record (OPEN_QUESTIONS B81). "
    )


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
            rec["note"] = margin_note(m) + rec["note"]
            if not m.ok():
                if hold is not None:  # not in the inbox: no dataset may be built from it
                    hold.mkdir(parents=True, exist_ok=True)
                    path = hold / f"{name}-FAILED-margins.json"
                    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
                    print(f"held (not in the inbox): {path}")
                raise SystemExit(
                    "A margin failed; nothing in the inbox. Report it, do not change seeds:\n  "
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


# ------------------------------------------------------------------ reproducing a record
def record_for(name: str) -> dict:
    """The filed record of an experiment (the newest, if refiled)."""
    found = sorted(RUNS_DIR.glob(f"*-{name}-exp*.json"))
    if not found:
        raise SystemExit(f"no record of {name} in runs/")
    rec = json.loads(found[-1].read_text())
    assert rec["experiment"] == name
    rec["_id"] = found[-1].stem
    return rec


def _strip_seconds(x):
    if isinstance(x, dict):
        return {k: _strip_seconds(v) for k, v in x.items() if k != "seconds"}
    return x


def reproduce(name: str, seeds: list[int] | None = None) -> bool:
    """Rerun an experiment (or some seeds of the band records) and compare it with the record,
    wall-clock seconds aside. Use after labs/shared or the lab's provided code changes."""
    rec = record_for(name)
    ns = lab_namespace()
    warm_up(ns)
    if seeds is not None and name.startswith("m09-reference-"):
        label = name.rsplit("-", 1)[1]
        per = _curves(ns, ns["CONFIG"], seeds, label)
        idx = [rec["seeds"].index(s) for s in seeds]
        old = {k: [rec["per_seed"][k][i] for i in idx] for k in rec["per_seed"]}
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            per, _, _ = measure(name, ns)
        old = rec["per_seed"]
    same = _strip_seconds(json.loads(json.dumps(per))) == _strip_seconds(old)
    print(
        f"{name}{'' if seeds is None else f' seeds {seeds}'}: "
        f"{'REPRODUCED exactly' if same else 'DIFFERS'} against {rec['_id']}"
    )
    return same


# ------------------------------------------------------------------ exact computations
def _r(x: float, d: int) -> float:
    """Round half up at d decimals, robust to float noise near the half."""
    return math.floor(x * 10**d + 0.5 + 1e-9) / 10**d


def first_gradient_step(learning_starts: int, train_freq: int) -> int:
    return next(t for t in range(1, 10**6) if t > learning_starts and t % train_freq == 0)


def copies_after_learning(steps: int, sync: int, learning_starts: int, train_freq: int) -> int:
    """Target copies at or before `steps` that follow at least one gradient step."""
    first = first_gradient_step(learning_starts, train_freq)
    return sum(1 for s in range(sync, steps + 1, sync) if s >= first)


def e_max_gauss(m: int) -> float:
    """E[max of m iid N(0, 1)] by numerical integration."""
    from scipy import integrate, stats

    def f(x):
        return x * m * stats.norm.pdf(x) * stats.norm.cdf(x) ** (m - 1)

    return integrate.quad(f, -12, 12, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


def _iqm_page(z) -> float:
    s, k = np.sort(np.asarray(z, dtype=float)), len(z) // 4
    return float(s[k : len(s) - k].mean())


def quoted_numbers() -> dict:
    """Every exact number the director's handoff (section 7) lists for the page, the
    predict_answers and the knowledge checks, asserted against the lab and the configuration."""
    import inspect

    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl import evaluate

    g = GAMMA
    src = LAB.read_text()
    setup = (SHARED / "dqn_setup.py").read_text()
    # Configuration constants the page and the lab state.
    assert "target_sync: int | None = 500" in setup and "gamma: float = 0.99" in setup
    assert "learning_starts: int = 1_000" in setup and "train_freq: int = 4" in setup
    assert "lr: float = 1e-3" in setup
    assert "def q_drift(log, window=5000):" in src  # the 5,000-step window
    assert 'A": (dict(lr=CONFIG.lr * 100), {})' in src  # run A: 100 times the reference
    assert 100 / LR_FACTORS[-1] == 10  # "ten times past the sweep's right end"
    assert LR_FACTORS == [0.1, 0.3, 1.0, 3.0, 10.0] and len(BAND_SEEDS) == 10
    assert len(BROKEN_SEEDS) == 3 and BUDGETS["live"]["probe_steps"] == 2_000
    assert inspect.signature(evaluate.bootstrap_ci).parameters["n_boot"].default == 5000
    # Exercise 1's predict_answer: the last log point of a target that never syncs, and the
    # healthy run's largest lag before the first gradient step.
    assert 50_000 - 500 == 49_500
    assert first_gradient_step(1_000, 4) == 1_004
    # Exercise 2's predict_answer and knowledge check 9.1.
    g500 = (1 - g**500) / (1 - g)
    assert _r(g500, 2) == 99.34 and abs(g500 - m09_signals.G_MAX) < 1e-9
    g10 = (1 - g**10) / (1 - g)
    assert _r(g10, 2) == 9.56 and round(4000 - _r(g10, 1)) == 3990 and round(4000 / g500) == 40
    # Exercise 4's predict_answer: 1 and 100; 20 copies after the first gradient step (65)
    # with PROBE_CONFIG (learning_starts 64, train_freq 1, target_sync 100) by 2,000 steps.
    assert "learning_starts=64, target_sync=100, train_freq=1" in src
    assert first_gradient_step(64, 1) == 65 and copies_after_learning(2_000, 100, 64, 1) == 20
    assert _r(1 / (1 - g), 9) == 100 and _r(sum(g**j for j in range(20)), 1) == 18.2
    # Exercise 6's predict_answer: 20 seats against a 10-seed band.
    assert round(20 * 2 / 11) == 4 and round(20 * 1 / 11) == 2
    # Page section 4 (Predict) and the Summary: 2/6 = 1/3 for a 5-seed band.
    assert _r(2 / (5 + 1), 12) == _r(1 / 3, 12)
    # Check yourself 3.1: Q(0) = 0 + gamma * 1.
    assert _r(0 + g * 1, 9) == 0.99
    # Check yourself 5.1.
    z = [12, 500, 480, 35, 500, 210, 9, 495, 500, 260]
    assert sum(z) / 10 == 300.1 and float(np.median(z)) == 370.0
    assert len(z) // 4 == 2 and _iqm_page(z) == 330.0 and len(z) - 2 * (len(z) // 4) == 6
    z0 = [0 if v == 9 else v for v in z]
    assert sum(z0) / 10 == 299.2 and float(np.median(z0)) == 370 and _iqm_page(z0) == 330.0
    assert evaluate.iqm(z) == 330.0  # SciPy's trim_mean(z, 0.25), as the page says
    # Knowledge check 9.2.
    q = np.array([[1.0, 2.0], [3.0, 0.5], [0.2, 0.1], [4.0, 4.5]])
    a, gathered = np.array([1, 0, 0, 1]), np.array([1.0, 0.5, 0.1, 4.0])
    want = q[np.arange(4), a]
    assert want.tolist() == [2.0, 3.0, 0.2, 4.5] and float(np.mean(gathered != want)) == 1.0
    assert np.array_equal(gathered, q[np.arange(4), 1 - a])  # the other action's entry
    # Knowledge check 9.3.
    assert _r(2 / (1 - 0.9), 9) == 20 and _r(2 * (1 - 0.9**10) / (1 - 0.9), 2) == 13.03
    # Knowledge check 9.5.
    z8 = [0, 10, 200, 210, 220, 230, 500, 500]
    assert _iqm_page(z8) == 215 and sum(z8) / 8 == 233.75
    z9 = sorted([*z8, 205])
    assert len(z9) // 4 == 2 and _iqm_page(z9) == 213 and len(z9) - 4 == 5
    # Knowledge check 9.6: [150, 260] and [120, 170] overlap from 150 to 170.
    assert max(150, 120) == 150 and min(260, 170) == 170
    # Knowledge check 9.8.
    e5 = e_max_gauss(5)
    assert _r(e5, 3) == 1.163 and abs(e5 - 1.1629644736) < 1e-9 and _r(10 * e5, 1) == 11.6
    return {"g500": g500, "e5": e5}


# ------------------------------------------------------------------ what the records say
def have_broken_runs() -> bool:
    """m09_broken_runs ships only if every margin held (B81): when make_datasets registers it."""
    import make_datasets

    return "m09_broken_runs" in make_datasets.REGISTRY


def _load(name: str) -> dict:
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    import prl.data

    return prl.data.load(name)


def _band(label: str) -> dict:
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    import prl.data

    return prl.data.band("m09_reference", BUDGETS[label])


def sweep_facts() -> dict:
    """The sweep as Exercise 5 reports it: IQM [CI], mean, the tie calls, the rankings."""
    from prl.evaluate import bootstrap_ci, compare, iqm

    d = _load("m09_sweep")
    final, factors = d["final_return"], d["lr_factor"]
    ref = int(np.flatnonzero(factors == 1.0)[0])
    rows = []
    for f, x in zip(factors, final, strict=True):
        lo, hi = bootstrap_ci(x, iqm)
        rows.append({"f": float(f), "iqm": iqm(x), "lo": lo, "hi": hi, "mean": float(x.mean())})
    calls = {}
    for i, f in enumerate(factors):
        if i != ref:
            v = compare(final[i], final[ref], names=(f"lr x{f:g}", "lr x1"))
            calls[f"x{f:g}"] = {"A": "better", "B": "worse", "tie": "tie"}.get(v.winner, v.winner)
    by_iqm = [f"x{factors[i]:g}" for i in np.argsort([-r["iqm"] for r in rows], kind="stable")]
    by_seed0 = [f"x{factors[i]:g}" for i in np.argsort(-final[:, 0], kind="stable")]
    return {
        "rows": rows,
        "calls": calls,
        "by_iqm": by_iqm,
        "by_seed0": by_seed0,
        "final": final,
        "factors": factors,
        "seconds": d["seconds"],
        "hardware": str(d["hardware"]),
    }


def band_facts() -> dict:
    """Exercise 6: the band's per-seed final values at each budget, and its top."""
    from prl.evaluate import bootstrap_ci, iqm

    out = {}
    for label in BUDGETS:
        b = _band(label)
        final = np.asarray(b["runs"], dtype=float)[:, -1]
        lo, hi = bootstrap_ci(final, iqm)
        # Adding one seed v moves the IQM by an amount monotone in v, so its extremes are at
        # v below every recorded value and above every one (exact, any seat's return).
        shifts = [iqm(np.append(final, v)) - iqm(final) for v in (-1e9, 1e9)]
        out[label] = {
            "final": final,
            "top": float(final.max()),
            "record": b["record"],
            "iqm": iqm(final),
            "ci": (lo, hi),
            "shift": (min(shifts), max(shifts)),
        }
    return out


def probe_facts() -> dict:
    """Exercise 4 and the stretch: each agent's values on each probe, per seed group."""
    rec = record_for("m09-probes")
    out: dict = {}
    for group in PROBE_SEEDS:
        out[group] = {}
        for run in RUNS:
            r = rec["per_seed"][group][run]
            out[group][run] = {p: np.array(r[p], dtype=float) for p in PROBES}
    return out


# ------------------------------------------------------------------ figures
PX = 1 / 96


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def _thousands(v: float, _pos=None) -> str:
    return f"{v / 1000:,.0f}k" if v else "0"


def sweep_figure(t, data: dict):
    """Final greedy return per learning rate: every seed, the IQM with its 95% bootstrap
    interval, and the mean."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator

    from make_figures_m01 import MIN_FONT, WIDTH_IN

    s = data["sweep"]
    fig = Figure(figsize=(WIDTH_IN, 380 * PX))
    ax = fig.add_axes((0.11, 0.16, 0.86, 0.72))
    xs = np.arange(len(s["factors"]))
    for i, (row, vals) in enumerate(zip(s["rows"], s["final"], strict=True)):
        v = np.sort(np.round(vals, 1))
        ax.plot(
            i - 0.12 + np.linspace(-0.1, 0.1, len(v)),
            v,
            "o",
            color=t.series[0],
            mfc="none",
            mew=1.4,
            ms=6,
            ls="none",
        )
        c, lo, hi = (round(row[k], 1) for k in ("iqm", "lo", "hi"))
        ax.errorbar(
            i + 0.16,
            c,
            yerr=[[c - lo], [hi - c]],
            fmt="D",
            color=t.series[0],
            ms=6,
            capsize=5,
            lw=1.8,
        )
        ax.plot(i + 0.3, round(row["mean"], 1), "x", color=t.series[1], ms=8, mew=2)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_ylim(0, 520)
    ax.xaxis.set_major_locator(FixedLocator(list(xs)))
    ax.set_xticklabels([f"×{f:g}" for f in s["factors"]])
    ax.yaxis.set_major_locator(FixedLocator([0, 100, 200, 300, 400, 500]))
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_xlabel("Learning rate, times the reference (1e-3)")
    ax.set_ylabel("Greedy return at 50,000 steps")
    handles = [
        Line2D([], [], marker="o", color=t.series[0], mfc="none", ls="none", ms=6, mew=1.4),
        Line2D([], [], marker="D", color=t.series[0], ls="-", ms=6, lw=1.8),
        Line2D([], [], marker="x", color=t.series[1], ls="none", ms=8, mew=2),
    ]
    labels = ["one seed (10 per rate)", "IQM, 95% bootstrap CI", "mean"]
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.0),
        ncol=3,
        frameon=False,
        fontsize=MIN_FONT,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    return fig


def _dashboard_panel(t, data: dict, panel: str):
    """One panel of the healthy run's dashboard, seeds 0-2 each as its own line."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, FuncFormatter

    from make_figures_m01 import WIDTH_IN

    f, d = data["lab"], data["broken"]
    fig = Figure(figsize=(WIDTH_IN / 2, 300 * PX))
    ax = fig.add_axes((0.2, 0.17, 0.75, 0.72))
    styles = ["solid", (0, (5, 2)), (0, (1.2, 1.6))]
    for k, seed in enumerate(d["seeds"]):
        g = f.run_log("healthy", int(seed))
        x, y = {
            "return": lambda g: f.smooth_returns(g),
            "loss": lambda g: (g["step"], g["loss"]),
            "td": lambda g: (g["step"], g["td_abs_mean"]),
            "q": lambda g: (g["step"], f.q_start(g)),
        }[panel](g)
        ax.plot(
            x,
            np.round(np.asarray(y, dtype=float), 4),
            color=t.series[0],
            ls=styles[k],
            lw=1.4,
            label=f"seed {int(seed)}",
        )
    title, ylabel = {
        "return": ("Return", "Mean return, per 2,500 steps"),
        "loss": ("Loss", "Mean loss (log scale)"),
        "td": ("Mean |δ|", "Mean |δ| (log scale)"),
        "q": ("Q at the start states", "Mean of max_a Q"),
    }[panel]
    if panel in ("loss", "td"):
        ax.set_yscale("log")
    if panel == "q":
        ax.axhline(_r(m09_signals.G_MAX, 2), color=t.neutral, ls=(0, (5, 2)), lw=1.2)
        ax.text(2_000, 96, "99.34, the most\nan episode earns", color=t.text, fontsize=12, va="top")
        ax.set_ylim(0, 110)
    if panel == "return":
        ax.set_ylim(0, 520)
        ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.set_xlim(0, 50_000)
    ax.xaxis.set_major_locator(FixedLocator([0, 25_000, 50_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, color=t.text)
    return fig


def _table_panel(t, data: dict, signal: str):
    """Exercise 3's table as a figure: one signal, every run, every seed."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator

    from make_figures_m01 import WIDTH_IN

    tab = data["margins"].table
    key, title, scale = {
        "gather": ("gather_check", "gather_check", "linear"),
        "lag": ("target_lag_last", "target_lag, last log point", "linear"),
        "drift": ("q_drift_last5", "Q drift, last 5 log points", "symlog"),
        "td": ("td_scale_2nd_half", "Relative TD error, 2nd half", "log"),
    }[signal]
    fig = Figure(figsize=(WIDTH_IN / 2, 300 * PX))
    ax = fig.add_axes((0.22, 0.15, 0.73, 0.74))
    for i, run in enumerate(RUNS):
        v = np.round(np.asarray(tab[run][key], dtype=float), 6)
        color = t.neutral if run == "healthy" else t.series[0]
        ax.plot(i + np.array([-0.15, 0.0, 0.15]), v, "o", color=color, mfc="none", mew=1.5, ms=7)
    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=10)
        ax.yaxis.set_major_locator(FixedLocator([-10, 0, 10, 100, 1000]))
        ax.set_yticklabels(["−10", "0", "10", "100", "1,000"])
    else:
        ax.set_yscale(scale)
    ax.xaxis.set_major_locator(FixedLocator(list(range(len(RUNS)))))
    ax.set_xticklabels(["healthy", "A", "B", "C", "D"])
    ax.set_xlim(-0.6, len(RUNS) - 0.4)
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_title(title, fontsize=13, color=t.text)
    return fig


FIGURES = {"09-sweep": ("sweep", None)}
PENDING_FIGURES = {  # drawn from m09_broken_runs, which ships only when every margin holds
    **{f"09-healthy-dashboard-{p}": ("dashboard", p) for p in ("return", "loss", "td", "q")},
    **{f"09-signal-table-{s}": ("table", s) for s in ("gather", "lag", "drift", "td")},
}


def figure_data() -> dict:
    data = {"sweep": sweep_facts(), "band": band_facts(), "probes": probe_facts()}
    if have_broken_runs():
        d = _load("m09_broken_runs")
        data.update(
            broken=d, lab=m09_signals.lab_functions(d), margins=m09_signals.assert_margins(d)
        )
    return data


def record_facts(data: dict) -> None:
    """What the records must show for the lab's design (asserted on every build)."""
    s = data["sweep"]
    assert s["final"].shape == (5, 10) and np.all(s["seconds"] > 0)
    for label, b in data["band"].items():
        assert len(b["final"]) == 10, label
        # Exercise 6: no seat can move the band's IQM by half its interval's width, and the
        # band's top is below 500, so 20 x 2/11 (not 1/11) seats are expected outside.
        assert max(abs(v) for v in b["shift"]) < (b["ci"][1] - b["ci"][0]) / 2, label
        assert b["top"] < 500, label
    p = data["probes"]
    for group in p:
        # Probe 1 with the mask learns 1 to two decimals on every seed for every agent but C,
        # whose Q heads for 100 (no mask) but stays near 18 after 20 copies.
        for run in ("healthy", "A", "B", "D"):
            assert np.all(np.abs(p[group][run]["probe 1"] - 1.0) < 0.05), (group, run)
        assert np.all(p[group]["C"]["probe 1"] > 5.0), group


def build() -> tuple[dict[Path, bytes], dict]:
    import matplotlib as mpl

    from make_figures_m01 import rc, svg_bytes, themes

    quoted = quoted_numbers()
    data = figure_data()
    record_facts(data)
    figures = dict(FIGURES)
    if "broken" in data:
        figures.update(PENDING_FIGURES)
    out: dict[Path, bytes] = {}
    for name, (kind, arg) in figures.items():
        for t in themes():
            with mpl.rc_context(rc(name, t)):
                if kind == "sweep":
                    fig, panel = sweep_figure(t, data), False
                elif kind == "dashboard":
                    fig, panel = _dashboard_panel(t, data, arg), True
                else:
                    fig, panel = _table_panel(t, data, arg), True
                if panel:
                    buf = io.BytesIO()
                    meta = {"Date": None, "Description": "subfigure panel"}
                    fig.savefig(buf, format="svg", metadata=meta)
                    out[IMAGES / f"{name}{t.suffix}.svg"] = buf.getvalue()
                else:
                    out[IMAGES / f"{name}{t.suffix}.svg"] = svg_bytes(fig)
    return out, {"quoted": quoted, "data": data}


def report(data: dict) -> None:
    s = data["sweep"]
    print(f"\nm09_sweep ({s['hardware']}): greedy return at 50,000 steps, 10 seeds per rate")
    for row, vals in zip(s["rows"], s["final"], strict=True):
        print(
            f"  x{row['f']:<4g} IQM {row['iqm']:6.1f} [{row['lo']:6.1f}, {row['hi']:6.1f}]  mean "
            f"{row['mean']:6.1f}  seeds {np.round(vals, 1).tolist()}"
        )
    print(f"  against x1: {s['calls']}")
    print(f"  ranked by IQM: {' > '.join(s['by_iqm'])}; by seed 0: {' > '.join(s['by_seed0'])}")
    print(
        f"  seconds per run {s['seconds'].min():.1f}-{s['seconds'].max():.1f}, total "
        f"{s['seconds'].sum():.0f}"
    )
    for label, b in data["band"].items():
        print(
            f"m09_reference {label} ({b['record']}): finals {np.round(b['final'], 1).tolist()}; "
            f"top {b['top']:.1f}; IQM {b['iqm']:.1f} [{b['ci'][0]:.1f}, {b['ci'][1]:.1f}]; one "
            f"more seed moves it by {b['shift'][0]:+.1f} to {b['shift'][1]:+.1f} (half the "
            f"interval: {(b['ci'][1] - b['ci'][0]) / 2:.1f})"
        )
    for group, by_run in data["probes"].items():
        print(f"m09-probes, {group} seeds:")
        for run, probes in by_run.items():
            cells = []
            for name, v in probes.items():
                lo, hi = v.min(axis=0), v.max(axis=0)
                cells.append(
                    f"{name} " + ", ".join(f"{a:.3f}..{b:.3f}" for a, b in zip(lo, hi, strict=True))
                )
            print(f"  {run:8s} " + " | ".join(cells))
    if "margins" in data:
        print("\nm09_broken_runs margins:")
        print("\n".join(f"  {line}" for line in data["margins"].lines))
    else:
        print("\nm09_broken_runs is not built (B81: a margin failed); its figures are pending.")


# ------------------------------------------------------------------ main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", nargs="+", metavar="EXPERIMENT", help="record experiments")
    ap.add_argument("--hold", type=Path, help="where a record whose margins fail is kept")
    ap.add_argument("--reproduce", nargs="+", metavar="EXPERIMENT")
    ap.add_argument("--seeds", help="comma-separated seeds (band records only)")
    ap.add_argument("--check", action="store_true", help="fail if a committed SVG differs")
    args = ap.parse_args(argv)
    if args.run:
        run(args.run, hold=args.hold)
        return 0
    if args.reproduce:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        return 0 if all([reproduce(n, seeds) for n in args.reproduce]) else 1
    files, info = build()
    if args.check:
        stale = [p for p, b in files.items() if not p.exists() or p.read_bytes() != b]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m09.py")
        return 1 if stale else 0
    IMAGES.mkdir(exist_ok=True)
    for p, b in files.items():
        p.write_bytes(b)
        print(f"wrote {p.relative_to(ROOT)} ({len(b) / 1024:.0f} KiB)")
    report(info["data"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
