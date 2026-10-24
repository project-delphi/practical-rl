"""Module 5 records and figures: light and dark SVGs (PLAN.md §8).

  uv run python scripts/make_figures_m05.py --run [--only NAME] [--jobs N]  # to runs/inbox/
  uv run python scripts/add_run_record.py              # file them in runs/
  uv run python scripts/make_figures_m05.py            # draw images/05-*.svg, print the numbers
  uv run python scripts/make_figures_m05.py --check    # fail if a committed SVG is out of date
  uv run python scripts/make_figures_m05.py --reproduce [--only NAME] [--jobs N]  # rerun in
                                                       # memory, compare with the filed records

The experiments use the lab's own code: the worked notebook (solutions bound, live budget)
runs in-process up to the stretch's run cell, as scripts/threshold_protocol.py does, for
seat 0. Exercise 5 then runs once per seat from the notebook's own cells: its sweep cell (the
unroled code cell that starts `REGIMES = {`) and its ex5-run cell, with RUN_SEED = (seat + 1)
* 1000, so a seat's numbers are exactly what that seat's notebook prints. Nothing here
reimplements a learner. Each run counts every argmax_first decision's distance from its tie
threshold (the smallest is `min_margin`), so a record says how far the discrete choices are
from the rounding that differs between platforms.

  m05-seats    Exercise 5 for seats 0-29 (seeds (seat + 1) * 1000 + 100..104, T = 3000), with
               the lab's conditions and launches: three regimes after a one-pass launch and
               the warm launch (well-specified, PASSES = {"warm launch": 3}). Plus seat 0's
               Exercise 1-4, 6 and 7 cells (Exercise 7 also with the planted-bug stub). Backs
               the predict_answer entries and the page's After-the-lab fold.
  m05-launch   The same sweep with the launch forced to `passes` for every condition (passes
               1, 3, 10; the "warm launch" key then repeats well-specified at that launch).
  m05-regimes  Seeds 0-9, T = 3000, the three regimes x three methods, one-pass launch,
               cumulative pseudo-regret every 25 rounds. Draws 05-regimes-{spec,misspec,drift}.

  05-ellipse                 The confidence ellipse of the demo (an exact computation),
                             460 x 400 px for its fold; its key is in the page's caption.
  05-regimes-{spec,misspec,drift}  Three subfigure panels: the mean of seeds 0-9 with the
                             min-max band (edges dashed like the line); greedy dashed,
                             LinUCB solid, LinTS dash-dot; a legend in every panel.

The launch is the lab's: run(env, choose, T, seed, passes) shows item t % K at rounds
t < passes * K, then lets the method choose. m05-launch forces `passes` by wrapping run.

Every build runs quoted_numbers(): each number modules/05-contextual-bandits.qmd, its
captions and alt text, knowledge-checks.qmd's Module 5 answers and the m05 predict_answer
entries quote is asserted against an exact computation or a filed record, and every record
those pages cite must exist in runs/. The figures read only the newest filed record of each
experiment, so `--check` never reruns anything. Drawing is deterministic (svg.hashsalt, no
date, text as paths, values rounded before plotting).
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import FixedLocator, FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_protocol as tp  # noqa: E402
from make_figures_m01 import MIN_FONT, Theme, rc, svg_bytes, themes  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.lab import _default_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
RUNS = ROOT / "runs"
NOTEBOOK = ROOT / "notebooks" / "05-contextual-bandits.ipynb"
SCRIPT = "scripts/make_figures_m05.py"

T = 3000
SEATS = list(range(30))
LAUNCH_PASSES = [1, 3, 10]
REGIME_SEEDS = list(range(10))
EVERY = 25  # m05-regimes keeps the cumulative regret after rounds 25, 50, ..., 3000
REGIMES = ("well-specified", "misspecified", "drifting")  # one-pass launch
WARM = "warm launch"  # well-specified, after the lab's longer launch
LAB_PASSES = {WARM: 3}  # the lab's PASSES; every other condition has one pass
METHODS = ("greedy", "LinUCB", "LinTS")
SEATS_BUDGET = {"T": T, "seats": len(SEATS), "seeds_per_seat": 5, "passes": "the lab's PASSES"}
SEATS_BUDGET["lab_passes"] = LAB_PASSES
LAUNCH_BUDGET = {"T": T, "seats": len(SEATS), "seeds_per_seat": 5, "passes": LAUNCH_PASSES}
LAUNCH_BUDGET["forced"] = "every condition"
REGIMES_BUDGET = {"T": T, "every": EVERY, "passes": 1}
ASSERT_EXERCISE_5 = True  # the director's second pass (warm launch) is in


# ------------------------------------------------------------------ the lab's namespace
def lab_namespace() -> dict:
    """The worked notebook's namespace (live budget, solutions bound, seat 0) before the stretch."""
    tp.ensure_current(NOTEBOOK)
    with tp._environment(PRL_QUICK="0", PRL_WORKED="1"):
        ns, _ = tp.solution_namespace(NOTEBOOK, "stretch")
    assert ns["budget"].T == T and ns["RUN_SEED"] == 1000, "seat 0 at the live budget"
    return ns


@functools.cache
def cell_source(cell_id: str) -> str:
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    (src,) = [c.source for c in nb.cells if c.get("id") == cell_id]
    return src


@functools.cache
def sweep_source() -> str:
    """Exercise 5's sweep: the unroled code cell before ex5-predict that starts `REGIMES = {`."""
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    (src,) = [
        c.source for c in nb.cells if c.cell_type == "code" and c.source.startswith("REGIMES = {")
    ]
    return src


class _Recorder:
    """Stands in for `lab` while a run cell is replayed for another seat."""

    def __init__(self) -> None:
        self.metrics: dict = {}
        self.seen: list[int] = []

    def metric(self, key: str, value) -> None:
        self.metrics[key] = value

    def seeds(self, seeds) -> None:
        self.seen += [int(s) for s in seeds]


class Margins:
    """Every argmax_first decision, and how far it is from depending on rounding.

    The tie rule picks the first index within tol = 1e-9 (1 + max|v|) of the max v. A choice
    can change between platforms only if an item's value sits within rounding (about
    1e-16 |v|) of the threshold max v - tol. `min_margin` is the smallest |v_j - threshold|
    over items j whose value is not the max itself: far above 1e-12 means no choice depends
    on rounding. Items exactly equal to the max (`exact_ties`; zeros, say, from b = 0) come
    from the same arithmetic on equal inputs and tie on every platform. `tie_rule` counts the
    decisions where another item was within tol of the max without equalling it.
    """

    def __init__(self) -> None:
        self.decisions = 0
        self.min_margin = math.inf
        self.exact_ties = 0
        self.tie_rule = 0

    def note(self, values) -> None:
        v = np.asarray(values, dtype=float)
        top = v.max()
        threshold = top - 1e-9 * (1.0 + np.abs(v).max())
        others = v[v != top]
        self.decisions += 1
        if others.size:
            self.min_margin = min(self.min_margin, float(np.abs(others - threshold).min()))
        self.exact_ties += int(np.count_nonzero(v == top) > 1)
        self.tie_rule += int(np.count_nonzero(others >= threshold) > 0)

    def as_dict(self) -> dict:
        return {
            "decisions": self.decisions,
            "min_margin": self.min_margin,
            "exact_ties": self.exact_ties,
            "tie_rule": self.tie_rule,
        }


@contextlib.contextmanager
def tracked(ns: dict, margins: Margins):
    """Count every argmax_first decision the lab's functions make (they look it up in ns)."""
    original = ns["argmax_first"]

    def argmax_first(values):
        margins.note(values)
        return original(values)

    ns["argmax_first"] = argmax_first
    try:
        yield
    finally:
        ns["argmax_first"] = original


def forced(ns: dict, passes: int):
    """The lab's run with the launch forced to `passes`, whatever the cell asks for."""
    run = ns["run"]

    def run_forced(env, choose, T, seed, passes_asked=None, **kw):
        return run(env, choose, T, seed, passes=passes)

    return run_forced


def _close_figures() -> None:
    import matplotlib.pyplot as plt

    plt.close("all")


# ------------------------------------------------------------------ Exercise 5, one seat
def sweep_seat(ns: dict, seat: int, passes: int | None) -> dict:
    """Replay Exercise 5's sweep cell and ex5-run as seat `seat` would run them.

    passes None keeps the lab's launches (PASSES); an int forces that launch everywhere.
    """
    local = dict(ns)
    local.update(RUN_SEED=(seat + 1) * 1000, lab=_Recorder())
    if passes is not None:
        local["run"] = forced(ns, passes)
    margins = Margins()
    with tracked(ns, margins), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(sweep_source(), f"<ex5 sweep seat {seat}>", "exec"), local)  # noqa: S102
        exec(compile(cell_source("ex5-run"), f"<ex5-run seat {seat}>", "exec"), local)  # noqa: S102
    _close_figures()
    assert local["lab"].seen == [(seat + 1) * 1000 + 100 + i for i in range(5)]
    assert local["PASSES"] == LAB_PASSES, "the lab's PASSES changed: update LAB_PASSES"
    return {
        "regrets": {
            w: {m: list(map(float, r)) for m, r in d.items()} for w, d in local["regrets"].items()
        },
        "verdicts": {f"{w}/{m}": word for (w, m), word in local["calls"].items()},
        "widths_end": {w: float(v) for w, v in local["widths_end"].items()},
        "metrics": {k: v for k, v in local["lab"].metrics.items() if not isinstance(v, dict)},
        "margins": margins.as_dict(),
    }


# Process pool: each worker builds the lab's namespace once.
_NS: dict | None = None


def _worker_init() -> None:
    global _NS
    os.environ.setdefault("MPLBACKEND", "Agg")
    with contextlib.redirect_stdout(io.StringIO()):
        _NS = lab_namespace()


def _sweep_task(args: tuple[int, int | None]) -> tuple[int, int | None, dict]:
    seat, passes = args
    assert _NS is not None
    return seat, passes, sweep_seat(_NS, seat, passes)


def sweeps(ns: dict, passes_list: list, jobs: int) -> dict:
    """{passes: [one sweep_seat result per seat in SEATS]}; passes None is the lab's launch."""
    tasks = [(s, p) for p in passes_list for s in SEATS]
    out: dict = {p: {} for p in passes_list}
    if jobs <= 1:
        for s, p in tasks:
            out[p][s] = sweep_seat(ns, s, p)
    else:
        with ProcessPoolExecutor(
            jobs, mp_context=get_context("spawn"), initializer=_worker_init
        ) as ex:
            for s, p, res in ex.map(_sweep_task, tasks):
                out[p][s] = res
    return {p: [out[p][s] for s in SEATS] for p in passes_list}


def _by_seat(results: list[dict]) -> dict:
    keys = ("regrets", "verdicts", "widths_end", "metrics", "margins")
    return {k: [r[k] for r in results] for k in keys}


# ------------------------------------------------------------------ seat 0's other cells
def _mean_weight(ns: dict, log, target) -> float:
    return float(ns["mean_weight"](log, target)[0])


def seat0_values(ns: dict) -> dict:
    """What seat 0's notebook prints in Exercises 1-4, 6 and 7 (from the built namespace)."""
    users_0, users_9 = ns["users_0"], ns["users_9"]
    uniform, new_policy = ns["uniform"], ns["new_policy"]
    return {
        "ex1_same_users": bool(np.array_equal(users_0, users_9)),
        "ex1_clicks_item0": int(ns["clicks_0"]),
        "ex1_clicks_item9": int(ns["clicks_9"]),
        "ex1_max_gap_simulator": float(np.abs(ns["ours"] - ns["sim"]).max()),
        "ex1_mean_rates": [float(v) for v in ns["sim"].mean(axis=0)],
        # The users and their mirror images -x: every item's mean is 0.5 by symmetry.
        "ex1_mirror_mean_gap": float(
            np.abs(
                np.vstack([ns["sim"], [ns["env"].expected_reward(-x) for x in ns["X"]]]).mean(
                    axis=0
                )
                - 0.5
            ).max()
        ),
        "ex2_predict_nine": float(ns["nine"].predict(ns["unit"])),
        "ex3_shown_greedy": [int(m.n) for m in ns["models_greedy"]],
        "ex3_shown_linucb": [int(m.n) for m in ns["models_ucb"]],
        "ex3_regret_greedy": float(ns["reg_greedy"][-1]),
        "ex3_regret_linucb": float(ns["reg_ucb"][-1]),
        "ex4_shown_lints": [int(m.n) for m in ns["models_ts"]],
        "ex4_regret_lints": float(ns["reg_ts"][-1]),
        "ex4_draws_above": float(np.mean(ns["draws"] > ns["ucb"])),
        "ex4_above": int(ns["above"]),
        "ex4_n_draws": int(len(ns["draws"])),
        "ex4_p_theory": float(ns["p_theory"]),
        "ex6_best_rows": float(ns["greedy_rows"].mean()),
        "ex6_propensities": sorted({round(float(p), 6) for p in ns["log"].P}),
        "ex6_weight_uniform": _mean_weight(ns, ns["log"], uniform),
        "ex6_weight_new_policy": _mean_weight(ns, ns["log"], new_policy),
        "ex6_weight_uniform_eps0": _mean_weight(ns, ns["greedy_log"], uniform),
        "ex7_fixed_weight_uniform": _mean_weight(ns, ns["suspect_log"], uniform),
        "ex7_fixed_weight_new_policy": _mean_weight(ns, ns["suspect_log"], new_policy),
    }


SEAT0_CELLS = ("ex0-run", "ex1-run", "ex2-run", "ex3-run", "ex4-run", "ex6-run", "ex7-run")


def seat0(ns: dict) -> dict:
    """Seat 0's cells, recorded twice: as built, then replayed with every decision counted
    (the replay must give the same numbers), plus Exercise 7 with the planted-bug stub."""
    built = seat0_values(ns)
    margins = Margins()
    with tracked(ns, margins), contextlib.redirect_stdout(io.StringIO()):
        for cid in SEAT0_CELLS:
            exec(compile(cell_source(cid), f"<{cid}>", "exec"), ns)  # noqa: S102
            _close_figures()
        again = seat0_values(ns)
        assert again == built, "replaying seat 0's cells changed a number"
        stub = dict(ns)  # the colleague's logger as the notebook ships it (its ex7-stub cell)
        exec(compile(cell_source("ex7-stub"), "<ex7-stub>", "exec"), stub)  # noqa: S102
        exec(compile(cell_source("ex7-run"), "<ex7-run, stub>", "exec"), stub)  # noqa: S102
        _close_figures()
    sus = stub["suspect_log"]
    built["ex7_stub_weight_uniform"] = _mean_weight(ns, sus, ns["uniform"])
    built["ex7_stub_weight_new_policy"] = _mean_weight(ns, sus, ns["new_policy"])
    built["ex7_stub_propensities"] = sorted({round(float(p), 6) for p in sus.P})
    built["margins"] = margins.as_dict()
    return built


# ------------------------------------------------------------------ m05-regimes
def run_regimes(ns: dict, passes: int = 1) -> dict:
    from prl import envs

    assert set(REGIMES) <= set(ns["REGIMES"]), "the lab's regimes"
    run = ns["run"]
    margins = Margins()
    per: dict = {}
    with tracked(ns, margins):
        for regime in REGIMES:
            kw = ns["REGIMES"][regime]
            env = envs.RecSim(n_items=ns["K"], d=ns["D_RAW"], seed=ns["CATALOG"], **kw)
            per[regime] = {}
            for method in METHODS:
                choose = ns["METHODS"][method]
                curves, finals = [], []
                for seed in REGIME_SEEDS:
                    cum, _ = run(env, choose, T, seed, passes=passes)
                    curves.append([float(v) for v in cum[EVERY - 1 :: EVERY]])
                    finals.append(float(cum[-1]))
                per[regime][method] = {"curve": curves, "final": finals}
    per["margins"] = margins.as_dict()
    return per


# ------------------------------------------------------------------ records
def experiment_record(
    name: str, budget: dict, seeds: list[int], per_seed: dict, seconds: float, note: str, git
) -> dict:
    import nbformat

    rt = runtime.detect()
    git_sha, dirty = git
    meta = nbformat.read(NOTEBOOK, as_version=4).metadata.get("prl", {})
    return {
        "schema": record.SCHEMA,
        "kind": "experiment",
        "experiment": name,
        "script": SCRIPT,
        "args": {
            "notebook": "notebooks/05-contextual-bandits.ipynb",
            "notebook_content_sha": meta.get("content_sha"),
            "code": "the worked notebook's reference solutions and run cells, in-process "
            "(live budget)",
            "repo_dirty": dirty,
            "test_doubles": runtime.test_doubles_used(),
        },
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
            "arch": __import__("platform").machine(),
        },
        "python": rt.python,
        "packages": runtime.package_versions(list(runtime.DEFAULT_PACKAGES)),
        "budget": budget,
        "seeds": seeds,
        "per_seed": per_seed,
        "seconds": round(seconds, 1),
        "note": note,
    }


def where() -> str:
    rt = runtime.detect()
    return f"{_default_env(rt, 'colab-cpu')} ({rt.cpu}, {__import__('platform').machine()})"


NOTES = {
    "m05-seats": (
        "Exercise 5's sweep and run cells replayed for seats 0-29 (RUN_SEED = (seat + 1) * "
        "1000, seeds RUN_SEED + 100..104, T = 3000; the lab's conditions: three regimes after "
        "a one-pass launch, and the warm launch, well-specified after 3 passes); `seeds` lists "
        "the seats and every per_seed list is indexed like it. per_seed['seat0'] holds seat "
        "0's Exercise 1-4, 6 and 7 numbers (Exercise 7 also with the planted-bug stub). "
        "margins: argmax_first "
        "decisions, the smallest distance from a tie threshold, and how many decisions the "
        "tie rule settled."
    ),
    "m05-launch": (
        "Exercise 5's sweep and run cells replayed for seats 0-29 with the launch forced to "
        "`passes` rounds per item for every condition (item t % K at rounds t < passes * K), "
        "passes 1, 3 and 10; per_seed['passes p'] has the m05-seats layout, and its 'warm "
        "launch' entries repeat well-specified at that launch."
    ),
    "m05-regimes": (
        f"Seeds {REGIME_SEEDS[0]}-{REGIME_SEEDS[-1]}, T = {T}, the lab's three regimes and "
        f"three methods; per_seed[regime][method]['curve'][i] is seed i's cumulative "
        f"pseudo-regret after rounds {EVERY}, {2 * EVERY}, ..., {T}; 'final' is the last value."
    ),
}


def compute(name: str, ns: dict, jobs: int, cache: dict) -> tuple[dict, list[int], dict]:
    if name in ("m05-seats", "m05-launch"):
        passes = [None] if name == "m05-seats" else LAUNCH_PASSES
        missing = [p for p in passes if p not in cache]
        cache.update(sweeps(ns, missing, jobs) if missing else {})
        if name == "m05-seats":
            per = _by_seat(cache[None])
            per["seat0"] = seat0(ns)
            return SEATS_BUDGET, list(SEATS), per
        return LAUNCH_BUDGET, list(SEATS), {f"passes {p}": _by_seat(cache[p]) for p in passes}
    if name == "m05-regimes":
        return REGIMES_BUDGET, list(REGIME_SEEDS), run_regimes(ns, REGIMES_BUDGET["passes"])
    raise KeyError(name)


EXPERIMENTS = ("m05-seats", "m05-launch", "m05-regimes")


def run(only: list[str] | None, jobs: int, inbox: Path = tp.INBOX) -> list[Path]:
    git = tp._git_state()
    ns = lab_namespace()
    cache: dict = {}
    out = []
    for name in EXPERIMENTS:
        if only and name not in only:
            continue
        start, load = time.perf_counter(), os.getloadavg()[0]
        budget, seeds, per = compute(name, ns, jobs, cache)
        secs = time.perf_counter() - start
        procs = f"{jobs} worker processes" if jobs > 1 and name != "m05-regimes" else "one process"
        note = (
            f"{NOTES[name]} Wall time on {where()}, {procs}, with other builds running (load "
            f"average {load:.1f} at the start, {os.getloadavg()[0]:.1f} at the end): a laptop "
            "time, not a Colab time."
        )
        rec = experiment_record(name, budget, seeds, json.loads(json.dumps(per)), secs, note, git)
        out.append(tp.write_record(rec, inbox))
    return out


def _diff(old, new, path: str, out: list[str], floats: list[float]) -> None:
    """Exact comparison of JSON values; floats are collected as |differences|."""
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


def _margins_in(per: dict) -> list[float]:
    found: list[float] = []
    if isinstance(per, dict):
        if "min_margin" in per and isinstance(per["min_margin"], float):
            found.append(per["min_margin"])
        for v in per.values():
            found += _margins_in(v)
    elif isinstance(per, list):
        for v in per:
            found += _margins_in(v)
    return found


def reproduce(only: list[str] | None, jobs: int) -> bool:
    """Rerun each experiment in memory and compare it with its newest filed record.

    Discrete values (verdicts, show counts, click counts, decision counts) must match
    exactly. Floats are reported: how many are bit-identical and the largest difference;
    the comparison passes when every float is within 1e-9.
    """
    ns = lab_namespace()
    rt = runtime.detect()
    print(f"Reproducing on {where()}, Python {rt.python}, numpy {np.__version__}")
    with contextlib.suppress(Exception):
        import threadpoolctl

        for lib in threadpoolctl.threadpool_info():
            print(
                f"  BLAS: {lib.get('internal_api')} {lib.get('version')} ({lib.get('architecture')})"
            )
    cache: dict = {}
    ok = True
    for name in EXPERIMENTS:
        if only and name not in only:
            continue
        path, rec = newest_record(name)
        budget, seeds, per = compute(name, ns, jobs, cache)
        if rec["budget"] != json.loads(json.dumps(budget)) or rec["seeds"] != seeds:
            print(f"{name}: runs/{path.name} has a different budget or seeds; not comparable")
            ok = False
            continue
        new = json.loads(json.dumps(per))
        exact: list[str] = []
        floats: list[float] = []
        _diff(rec["per_seed"], new, "per_seed", exact, floats)
        same = sum(1 for f in floats if f == 0.0)
        worst = max(floats, default=0.0)
        good = not exact and worst <= 1e-9
        ok &= good
        margins = _margins_in(new)
        print(
            f"{name}: {'REPRODUCED' if good else 'DIFFERS'} against runs/{path.name}: "
            f"{len(exact)} discrete mismatches; {same} of {len(floats)} floats bit-identical, "
            f"max |diff| {worst:.3g}; smallest decision margin {min(margins, default=math.nan):.3g}"
        )
        for line in exact[:10]:
            print(f"  {line}")
    return ok


# ------------------------------------------------------------------ reading the records
def _order(path: Path) -> tuple[str, int]:
    m = re.search(r"-exp(?:-(\d+))?\.json$", path.name)
    return (path.name[:10], int(m.group(1) or 1) if m else 0)


def newest_record(experiment: str) -> tuple[Path, dict]:
    paths = sorted(RUNS.glob(f"*-{experiment}-exp*.json"), key=_order)
    if not paths:
        raise SystemExit(
            f"No filed record of {experiment} in runs/. Run this script with --run, then "
            "scripts/add_run_record.py."
        )
    return paths[-1], json.loads(paths[-1].read_text())


def regimes_data() -> dict:
    path, rec = newest_record("m05-regimes")
    assert rec["budget"] == REGIMES_BUDGET and rec["seeds"] == REGIME_SEEDS, path.name
    rounds = np.arange(0, T + 1, EVERY)
    panels = {}
    for regime in REGIMES:
        panels[regime] = {}
        for method in METHODS:
            c = np.array(rec["per_seed"][regime][method]["curve"])
            c = np.hstack([np.zeros((len(c), 1)), c])  # round 0: no regret yet
            panels[regime][method] = {
                "mean": np.round(c.mean(axis=0), 3),
                "lo": np.round(c.min(axis=0), 3),
                "hi": np.round(c.max(axis=0), 3),
                "raw": c,
                "final": np.array(rec["per_seed"][regime][method]["final"]),
            }
    return {"record": path.name, "rec": rec, "rounds": rounds, "panels": panels}


# ------------------------------------------------------------------ figure 1: the ellipse
LAMBDA = np.array([[5.0, 1.0], [1.0, 2.0]])  # I + 3 (1, 0)(1, 0)^T + (1, 1)(1, 1)^T
B = np.array([3.0, 1.0])  # clicks 1, 1, 0 at (1, 0) and 1 at (1, 1)
DIRECTIONS = {
    "(1, 0)": np.array([1.0, 0.0]),
    "(0, 1)": np.array([0.0, 1.0]),
    "(1, \N{MINUS SIGN}1)/\N{SQUARE ROOT}2": np.array([1.0, -1.0]) / math.sqrt(2.0),
}
CENTER = r"$\hat w$"  # the learner's estimate (the lead's notation: w, not theta)


def ellipse_data() -> dict:
    inv = np.linalg.inv(LAMBDA)
    center = inv @ B
    evals, evecs = np.linalg.eigh(LAMBDA)  # ascending: the first is the long axis
    semi = 1.0 / np.sqrt(evals)
    long_axis = evecs[:, 0] * np.sign(evecs[1, 0])
    tilt = math.degrees(math.atan2(abs(long_axis[0]), abs(long_axis[1])))  # from the vertical
    widths = {k: float(np.sqrt(u @ inv @ u)) for k, u in DIRECTIONS.items()}
    return {
        "inv": inv,
        "center": center,
        "semi": semi,
        "long_axis": long_axis,
        "tilt_from_vertical": tilt,
        "widths": widths,
        "width_11": float(np.sqrt(np.array([1.0, 1.0]) @ inv @ np.array([1.0, 1.0]) / 2.0)),
    }


ELLIPSE_PT = 14  # labels and ticks: the figure sits in the Demo's answer fold (252 px at 360)


def ellipse_figure(t: Theme, data: dict) -> Figure:
    """460 x 400 px, two-line 14 pt labels; the key is in the page's caption (UI review, item 1).

    At 252 px (the fold on a 360 px phone) the text renders at about 10 px. The canvas is
    narrower than the column, so the SVG declares itself a "subfigure panel", the only
    description tests/test_site_figures.py accepts for a narrow figure.
    """
    fig = Figure(figsize=(460 / 96, 400 / 96))
    ax = fig.add_axes((0.17, 0.13, 0.81, 0.85))
    c = np.round(data["center"], 4)
    semi, axis = data["semi"], data["long_axis"]
    angle = math.degrees(math.atan2(axis[1], axis[0]))  # the long axis, from the w1 axis
    shape = {"xy": tuple(c), "width": 2 * semi[0], "height": 2 * semi[1], "angle": angle}
    ax.add_patch(Ellipse(**shape, facecolor=t.series[0], alpha=0.12, edgecolor="none"))
    ax.add_patch(Ellipse(**shape, fill=False, edgecolor=t.series[0], lw=2.2))
    inv = data["inv"]
    offsets = {  # label position relative to the arrow tip, in data units
        "(1, 0)": (0.06, 0.0),
        "(0, 1)": (0.47, 0.0),  # right of its tangent line
        "(1, \N{MINUS SIGN}1)/\N{SQUARE ROOT}2": (0.07, -0.08),
    }
    colors = (t.series[1], t.series[2], t.series[3])
    for (name, u), color in zip(DIRECTIONS.items(), colors, strict=True):
        w = data["widths"][name]
        tip = c + w * u
        # The tangent line perpendicular to u touches the ellipse at c + inv u / w.
        touch = c + inv @ u / w
        perp = np.array([-u[1], u[0]])
        span = np.round([tip + 0.42 * perp, tip - 0.42 * perp], 4)
        ax.plot(span[:, 0], span[:, 1], color=t.neutral, lw=1.4, ls=(0, (4, 2.5)), zorder=2)
        ax.annotate(
            "",
            xy=tuple(np.round(tip, 4)),
            xytext=tuple(c),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.5,head_width=0.25",
                "color": color,
                "lw": 2.4,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )
        ax.plot(*np.round(touch, 4), marker="o", ms=5, color=color, mec=t.bg, mew=0.8, zorder=5)
        dx, dy = offsets[name]
        ax.text(
            tip[0] + dx,
            tip[1] + dy,
            f"{name}\nwidth {w:.3f}",
            ha="left",
            va="center",
            color=t.text,
            fontsize=ELLIPSE_PT,
            linespacing=1.2,
        )
    ax.plot(*c, marker="o", ms=7, color=t.text, zorder=6)
    ax.annotate(
        f"{CENTER}\n({c[0]:.3f}, {c[1]:.3f})",
        xy=tuple(c),
        xytext=(-10, -6),
        textcoords="offset points",
        ha="right",
        va="top",
        color=t.text,
        fontsize=ELLIPSE_PT,
        linespacing=1.2,
        bbox={"boxstyle": "square,pad=0.15", "facecolor": t.bg, "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.45, 1.85)
    ax.set_ylim(-0.75, 1.15)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0, 1.5]))
    ax.yaxis.set_major_locator(FixedLocator([-0.5, 0.0, 0.5, 1.0]))
    fmt = FuncFormatter(lambda v, _p: f"{v:g}".replace("-", "\N{MINUS SIGN}"))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(labelsize=ELLIPSE_PT)
    ax.grid(True, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("$w_1$", fontsize=ELLIPSE_PT)
    ax.set_ylabel("$w_2$", fontsize=ELLIPSE_PT)
    return fig


# ------------------------------------------------------------------ figure 2: three regimes
PX = 1 / 96
PANEL_W, PANEL_H = 264 * PX, 300 * PX  # 34:33:33 of the 778 px column; they stack on phones
STYLES = {"greedy": (0, (5, 2.5)), "LinUCB": "solid", "LinTS": (0, (7, 2.2, 1.6, 2.2))}
LABELS = {"greedy": "Greedy", "LinUCB": "LinUCB", "LinTS": "LinTS"}
Y_MAX = 600


def regime_panel(regime: str):
    """One regime's panel. Every panel has its own legend (they stack on phones, and the
    lightbox opens each alone); band edges repeat each method's dash style; the subcaption
    names the regime, so the panel has no title (UI review, items 2 and 9)."""

    def draw(t: Theme, data: dict) -> Figure:
        fig = Figure(figsize=(PANEL_W, PANEL_H))
        ax = fig.add_axes((0.27, 0.165, 0.68, 0.8))
        x = data["rounds"]
        colors = dict(zip(METHODS, t.series, strict=False))
        band = 0.22 if t.suffix else 0.14  # t.suffix is ".dark" in the dark theme
        for method in METHODS:
            p = data["panels"][regime][method]
            ax.fill_between(x, p["lo"], p["hi"], color=colors[method], alpha=band, lw=0)
            for edge in (p["lo"], p["hi"]):
                ax.plot(x, edge, color=colors[method], lw=0.7, ls=STYLES[method], alpha=0.8)
        for method in METHODS:
            p = data["panels"][regime][method]
            ax.plot(x, p["mean"], color=colors[method], lw=2.0, ls=STYLES[method])
        ax.set_xlim(0, T)
        ax.set_ylim(0, Y_MAX)
        ax.xaxis.set_major_locator(FixedLocator([0, 1000, 2000, 3000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1000:g}k" if v else "0"))
        ax.yaxis.set_major_locator(FixedLocator([0, 200, 400, 600]))
        ax.grid(True, axis="y", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative pseudo-regret")
        handles = [Line2D([], [], color=colors[m], lw=2.0, ls=STYLES[m]) for m in METHODS]
        ax.legend(
            handles,
            [LABELS[m] for m in METHODS],
            loc="upper left",
            frameon=False,
            handlelength=2.6,
            fontsize=MIN_FONT,
            borderaxespad=0.2,
        )
        return fig

    return draw


# ------------------------------------------------------------------ build, report, check
DATA = {"05-ellipse": ellipse_data, "05-regimes": regimes_data}
FIGURES = {
    "05-ellipse": ("05-ellipse", ellipse_figure, True),  # narrow, for its fold (see its docstring)
    "05-regimes-spec": ("05-regimes", regime_panel("well-specified"), True),
    "05-regimes-misspec": ("05-regimes", regime_panel("misspecified"), True),
    "05-regimes-drift": ("05-regimes", regime_panel("drifting"), True),
}


def panel_svg_bytes(fig: Figure) -> bytes:
    """Deterministic SVG that declares itself a subfigure panel (tests/test_site_figures.py)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Description": "subfigure panel"})
    return buf.getvalue()


def build() -> tuple[dict[Path, bytes], dict[str, dict]]:
    computed = {name: make() for name, make in DATA.items()}
    quoted_numbers(computed)
    out: dict[Path, bytes] = {}
    for name, (key, draw, panel) in FIGURES.items():
        for t in themes():
            with mpl.rc_context(rc(name, t)):
                fig = draw(t, computed[key])
                out[IMAGES / f"{name}{t.suffix}.svg"] = (panel_svg_bytes if panel else svg_bytes)(
                    fig
                )
    return out, computed


# ------------------------------------------------------------------ numbers the pages quote
def exact_numbers() -> dict[str, float]:
    """The page's and the knowledge checks' hand computations, recomputed."""
    from scipy.stats import norm

    lam2 = np.eye(2) + np.outer([1, 0], [1, 0]) + np.outer([1, 1], [1, 1])
    inv2 = np.linalg.inv(lam2)
    w2 = inv2 @ np.array([1.0, 0.0])
    sd = math.hypot(0.25 * 0.2, 0.25 * 0.02)
    lam53 = np.eye(2) + np.outer([1, 0], [1, 0]) + np.outer([0, 2], [0, 2])
    return {
        "worked_inv": inv2,
        "worked_w": w2,
        "worked_pred": float(w2 @ np.array([1.0, 1.0])),
        "chk41_scores": (0.6 + 0.5 * 0.2, 0.65 + 0.5 * 0.02),
        "chk41_p": float(norm.sf(0.05 / sd)),
        "chk41_z": 0.05 / sd,
        "li_alpha": 1 + math.sqrt(math.log(2 / 0.05) / 2),
        "ag_v": 0.98 * math.sqrt(9 * 60 * math.log(3000 / 0.05)),
        "predict_ex2": 9 / (9 + 1),
        "predict_ex3": 1 / math.sqrt(101),
        "predict_ex4": float(norm.sf(0.5 / 0.25)),
        "kc53_inv": np.linalg.inv(lam53),
        "kc54_width": math.sqrt(1 / (1 + 3)),
    }


def _r(x: float, digits: int) -> float:
    return round(float(x), digits)


def quoted_numbers(computed: dict[str, dict]) -> None:
    """Assert every number the Module 5 page, its figures, knowledge-checks.qmd's Module 5
    answers and the m05 predict_answer entries quote (Exercise 5's in exercise_5_claims)."""
    ex = exact_numbers()
    cited_records_exist()
    # Section 2's worked example, the demo, Check yourself 4.1 and 6.1, the references.
    assert np.allclose(ex["worked_inv"], [[2 / 5, -1 / 5], [-1 / 5, 3 / 5]])
    assert np.allclose(ex["worked_w"], [2 / 5, -1 / 5]) and abs(ex["worked_pred"] - 1 / 5) < 1e-15
    e = computed["05-ellipse"]
    assert np.allclose(e["inv"] * 9, [[2, -1], [-1, 5]])
    assert np.allclose(e["center"], [5 / 9, 2 / 9])
    assert [_r(v, 3) for v in e["center"]] == [0.556, 0.222]
    assert [_r(v, 3) for v in e["widths"].values()] == [0.471, 0.745, 0.707]
    assert np.allclose(list(e["widths"].values()), np.sqrt([2 / 9, 5 / 9, 1 / 2]))
    assert _r(e["width_11"], 3) == 0.527 and abs(e["width_11"] - math.sqrt(5 / 18)) < 1e-15
    assert [_r(v, 3) for v in sorted(e["semi"])] == [0.434, 0.768]
    assert round(e["tilt_from_vertical"]) == 17  # "the long one ... about 17 degrees from it"
    assert [_r(v, 2) for v in ex["chk41_scores"]] == [0.70, 0.66]
    assert _r(ex["chk41_z"], 3) == 0.995 and _r(ex["chk41_p"], 2) == 0.16
    assert _r(ex["li_alpha"], 2) == 2.36
    assert _r(ex["ag_v"], 1) == 75.5 and round(ex["ag_v"] / 0.25, -2) == 300
    # Predict answers that are exact (Exercises 2, 3, 4, 6, 7).
    assert ex["predict_ex2"] == 0.9
    assert _r(ex["predict_ex3"], 4) == 0.0995 and _r(0.5 * ex["predict_ex3"], 3) == 0.050
    assert _r(ex["predict_ex4"], 4) == 0.0228 and round(1 / ex["predict_ex4"]) == 44
    assert _r(1 - 0.1 + 0.1 / 10, 2) == 0.91 and _r(0.1 / 10, 2) == 0.01
    assert _r(0.1 * 9 / 10, 2) == 0.09 and _r(0.1 / 0.91, 2) == 0.11
    assert abs(0.91 * (0.1 / 0.91) + 0.09 * (0.1 / 0.01) - 1.0) < 1e-12
    # Knowledge checks 5.2-5.4, 5.7 and 5.8.
    assert np.allclose(0.5 + 0.15 * np.array([1, -2, 10]), [0.65, 0.2, 2.0])
    assert np.allclose(ex["kc53_inv"], np.diag([0.5, 0.2]))
    assert ex["kc54_width"] == 0.5 and 0.5 * 0.5 == 0.25 and 0.25 * 0.5 == 0.125
    assert _r(1 - 0.25 + 0.25 / 5, 2) == 0.8 and 0.25 / 5 == 0.05 and round(1 / 0.05) == 20
    assert round(0.91 / 0.01) == 91
    # Seat 0's notebook (m05-seats): Exercises 1, 3, 4, 6 and 7.
    _, rec = newest_record("m05-seats")
    s0 = rec["per_seed"]["seat0"]
    assert s0["ex1_same_users"] and (s0["ex1_clicks_item0"], s0["ex1_clicks_item9"]) == (486, 464)
    # "every item's click probability averaged over users is exactly 0.5": with each user's
    # mirror image -x the mean is 0.5 to rounding (x is symmetric, the clip is about 0.5).
    assert s0["ex1_mirror_mean_gap"] < 1e-12, s0["ex1_mirror_mean_gap"]
    assert s0["ex2_predict_nine"] == 0.9 or abs(s0["ex2_predict_nine"] - 0.9) < 1e-12
    once = [a for a, n in enumerate(s0["ex3_shown_greedy"]) if n == 1]
    assert once == [2, 4, 7, 9], once  # predict_answer Ex3; the page's "four of the ten items"
    # "447 of 20000 draws"
    assert s0["ex4_n_draws"] == 20000 and s0["ex4_above"] == 447
    assert round(s0["ex4_draws_above"] * 20000) == 447
    assert _r(1 - s0["ex6_best_rows"], 3) == 0.093  # "seat 0: 9.3%"
    assert _r(s0["ex7_stub_weight_new_policy"], 2) == 0.62
    assert s0["ex7_stub_propensities"] == [0.91]  # every row records 0.91: weights all 0.11
    assert _r(s0["ex7_stub_weight_uniform"], 2) == 0.11
    # "0.62 against 1.05 on the correct log": Exercise 6's env.log log (the lab prints 1.055)
    assert _r(s0["ex6_weight_new_policy"], 2) == 1.05
    assert _r(s0["ex6_weight_uniform"], 2) == 1.03  # the Gotcha's "My mean weight is 1.03"
    claims = exercise_5_claims(computed, rec, newest_record("m05-launch")[1])
    failed = [c for c in claims if not c[1]]
    if ASSERT_EXERCISE_5:
        assert not failed, failed


CITING = ("modules/05-contextual-bandits.qmd", "knowledge-checks.qmd", "_variables.yml")


def cited_records_exist() -> None:
    """Every runs/*.json that the Module 5 pages cite is a filed record."""
    for name in CITING:
        for cited in re.findall(r"runs/[\w.-]+\.json", (ROOT / name).read_text()):
            assert (ROOT / cited).exists(), f"{name} cites {cited}, which is not in runs/"


def _tally(per: dict, conditions=REGIMES) -> dict:
    """{(condition, method): {word: seats}} for LinUCB and LinTS against greedy."""
    out: dict = {}
    for cond in conditions:
        for method in ("LinUCB", "LinTS"):
            words = [v[f"{cond}/{method}"] for v in per["verdicts"]]
            out[cond, method] = {w: words.count(w) for w in set(words)}
    return out


def leads(per: dict, cond: str, method: str) -> list[float]:
    """Greedy's IQM regret minus the method's, seat by seat (positive: the method did better)."""
    from prl.evaluate import iqm

    return [iqm(r[cond]["greedy"]) - iqm(r[cond][method]) for r in per["regrets"]]


def exercise_5_claims(
    computed: dict[str, dict], seats: dict, launch: dict
) -> list[tuple[str, bool, str]]:
    """Exercise 5's quoted numbers: (claim, holds?, what the records say)."""
    per = seats["per_seed"]
    tally = _tally(per, REGIMES + (WARM,))
    claims: list[tuple[str, bool, str]] = []

    def claim(text: str, ok: bool, detail) -> None:
        claims.append((text, bool(ok), str(detail)))

    # Seat 0 (m05-seats).
    v0 = per["verdicts"][0]
    got = tuple(v0[f"{c}/LinUCB"] for c in REGIMES + (WARM,))
    claim(
        "seat 0: LinUCB beats greedy well-specified and misspecified, ties drifting and warm",
        got == ("LinUCB", "LinUCB", "tie", "tie"),
        got,
    )
    got = tuple(v0[f"{c}/LinTS"] for c in REGIMES + (WARM,))
    claim("seat 0: LinTS ties in all four", got == ("tie",) * 4, got)
    m0 = per["metrics"][0]
    got = (m0["ucb_lead_warm"], m0["ts_lead_warm"])
    claim("seat 0: ucb_lead_warm 14.32, ts_lead_warm -7.36", got == (14.32, -7.36), got)
    # One-pass regimes, seats 0-29 (m05-seats).
    for method, want in (("LinUCB", (18, 25, 9)), ("LinTS", (12, 16, 3))):
        got = tuple(tally[w, method].get(method, 0) for w in REGIMES)
        claim(f"one pass: {method} beats greedy at {want} seats", got == want, got)
    got = tuple(tally[w, "LinUCB"].get("tie", 0) for w in REGIMES)
    claim("one pass: LinUCB's calls are ties at 12, 5, 21 seats", got == (12, 5, 21), got)
    greedy = sum(tally[w, m].get("greedy", 0) for w in REGIMES for m in ("LinUCB", "LinTS"))
    claim("one pass: greedy wins no comparison", greedy == 0, greedy)
    others = {
        w: {m: sorted(set(tally[w, m]) - {m, "tie"}) for m in ("LinUCB", "LinTS")} for w in REGIMES
    }
    claim(
        "one pass: every other call is a tie",
        all(not v for d in others.values() for v in d.values()),
        others,
    )
    lows = {(w, m): min(leads(per, w, m)) for w in REGIMES for m in ("LinUCB", "LinTS")}
    claim(
        "one pass: both methods' IQM below greedy's at every seat",
        min(lows.values()) > 0,
        min(lows.values()),
    )
    drift, spec = leads(per, "drifting", "LinUCB"), leads(per, "well-specified", "LinUCB")
    smaller = sum(d < s_ for d, s_ in zip(drift, spec, strict=True))
    claim("LinUCB's drift lead below its well-specified lead at 27 of 30", smaller == 27, smaller)
    # The warm launch (m05-seats), and m05-launch's three passes.
    ties_warm = tally[WARM, "LinUCB"].get("tie", 0)
    claim(
        "warm launch: LinUCB's call a tie at 29 of 30 (12 of 30 with one pass)",
        (ties_warm, tally["well-specified", "LinUCB"].get("tie", 0)) == (29, 12),
        ties_warm,
    )
    warm = leads(per, WARM, "LinUCB")
    got = (_r(np.median(spec), 1), _r(np.median(warm), 1))
    claim("LinUCB's median lead 121.7 (one pass) to 38.3 (warm)", got == (121.7, 38.3), got)
    above = sum(v < 0 for v in warm)
    claim("warm launch: LinUCB's IQM above greedy's at 2 of 30 seats", above == 2, above)
    p3, p10 = launch["per_seed"]["passes 3"], launch["per_seed"]["passes 10"]
    same = all(
        r[WARM] == q["well-specified"] for r, q in zip(per["regrets"], p3["regrets"], strict=True)
    ) and all(
        v[f"{WARM}/{m}"] == u[f"well-specified/{m}"]
        for v, u in zip(per["verdicts"], p3["verdicts"], strict=True)
        for m in ("LinUCB", "LinTS")
    )
    claim(
        "m05-seats' warm launch equals m05-launch's 3-pass well-specified, seat by seat", same, same
    )
    t3 = _tally(p3)
    got = (
        t3["well-specified", "LinUCB"].get("tie", 0),
        t3["misspecified", "LinUCB"].get("LinUCB", 0),
    )
    claim(
        "three passes: well-specified tie at 29 of 30; misspecified LinUCB wins at 12",
        got == (29, 12),
        got,
    )
    t10 = _tally(p10)
    ties10 = sum(t10[w, m].get("tie", 0) for w in REGIMES for m in ("LinUCB", "LinTS"))
    wins10 = (
        tuple(t10[w, "LinUCB"].get("LinUCB", 0) for w in REGIMES),
        sum(t10[w, "LinTS"].get("LinTS", 0) for w in REGIMES),
    )
    claim(
        "ten passes: 178 of 180 calls ties; LinUCB wins 0 + 1 + 1, LinTS none",
        (ties10, wins10) == (178, ((0, 1, 1), 0)),
        (ties10, wins10),
    )
    w0 = per["widths_end"][0]
    got = (_r(w0["well-specified"], 2), _r(w0["misspecified"], 2))
    claim(
        "seat 0 end widths 0.25 (well-specified) and 0.27 (misspecified)", got == (0.25, 0.27), got
    )
    # The regimes figure: its caption and alt texts (m05-regimes).
    g = computed["05-regimes"]["panels"]

    def slope(regime: str, method: str, end: int) -> float:
        m = g[regime][method]["raw"].mean(axis=0)
        i, j = end // EVERY, (end - 500) // EVERY
        return float((m[i] - m[j]) / 500)

    got = (_r(slope("well-specified", "LinUCB", T), 4), _r(slope("misspecified", "LinUCB", T), 4))
    claim("LinUCB per round over the last 500: 0.0303 and 0.0805", got == (0.0303, 0.0805), got)
    got = (_r(slope("drifting", "LinUCB", 1000), 4), _r(slope("drifting", "LinUCB", T), 4))
    claim("drifting LinUCB: 0.0828 (rounds 501-1000) against 0.1588", got == (0.0828, 0.1588), got)
    ends = {
        m: (
            _r(g["well-specified"][m]["final"].min(), 1),
            _r(g["well-specified"][m]["final"].max(), 1),
        )
        for m in METHODS
    }
    claim(
        "well-specified end ranges 88.4-514.5, 129.4-258.1, 113.4-186.9; greedy's widest",
        ends == {"greedy": (88.4, 514.5), "LinUCB": (129.4, 258.1), "LinTS": (113.4, 186.9)},
        ends,
    )
    finals = {m: _r(g["misspecified"][m]["final"].mean(), 1) for m in METHODS}
    claim(
        "misspecified mean finals 465.0, 341.7, 345.8 (greedy highest)",
        finals == {"greedy": 465.0, "LinUCB": 341.7, "LinTS": 345.8},
        finals,
    )
    higher = all(
        g["misspecified"][m]["final"].mean() > g["well-specified"][m]["final"].mean()
        for m in METHODS
    )
    claim("misspecified: all three end higher than well-specified", higher, None)
    larger = all(slope("misspecified", m, T) > slope("well-specified", m, T) for m in METHODS)
    claim("misspecified: slopes stay larger (last 500 rounds)", larger, None)
    falling = all(
        np.all(np.diff([slope("well-specified", m, e) for e in range(1000, T + 1, 500)]) < 0)
        for m in METHODS
    )
    claim("well-specified: all three bend over (slope per 500 rounds falls)", falling, None)
    rising = all(
        np.all(np.diff([slope("drifting", m, e) for e in range(1000, T + 1, 500)]) > 0)
        for m in METHODS
    )
    claim("drifting: the mean slope per 500 rounds rises for every method", rising, None)
    return claims


def report(computed: dict[str, dict]) -> None:
    e = computed["05-ellipse"]
    print(
        f"05-ellipse: center {np.round(e['center'], 4).tolist()}, semi-axes "
        f"{np.round(e['semi'], 4).tolist()}, long axis {e['tilt_from_vertical']:.1f} degrees "
        f"from the vertical, widths { {k: round(v, 4) for k, v in e['widths'].items()} }"
    )
    path, seats = newest_record("m05-seats")
    print(f"m05-seats from runs/{path.name}")
    per = seats["per_seed"]
    for k, v in per["seat0"].items():
        print(f"  seat 0 {k}: {v}")
    for (w, m), counts in _tally(per, REGIMES + (WARM,)).items():
        lead = leads(per, w, m)
        print(
            f"  seats 0-29, {w}, {m} vs greedy: {dict(sorted(counts.items()))}; lead median "
            f"{np.median(lead):.1f} [{min(lead):.1f}, {max(lead):.1f}], negative at "
            f"{sum(v < 0 for v in lead)}"
        )
    margins = [m["min_margin"] for m in per["margins"]]
    ties = sum(m["tie_rule"] for m in per["margins"])
    print(f"  smallest decision margin over seats: {min(margins):.3g}; tie-rule decisions {ties}")
    lpath, launch = newest_record("m05-launch")
    print(f"m05-launch from runs/{lpath.name}")
    for key, lp in launch["per_seed"].items():
        for (w, m), counts in _tally(lp).items():
            lead = leads(lp, w, m)
            print(
                f"  {key}, {w}, {m} vs greedy: {dict(sorted(counts.items()))}; lead median "
                f"{np.median(lead):.1f}, negative at {sum(v < 0 for v in lead)}"
            )
    print("Exercise 5's quoted numbers:")
    for text, ok, detail in exercise_5_claims(computed, seats, launch):
        print(f"  {'holds  ' if ok else 'DIFFERS'} {text}: {detail}")


def main(argv: list[str]) -> int:
    only = [argv[i + 1] for i, a in enumerate(argv) if a == "--only" and i + 1 < len(argv)]
    jobs = next((int(argv[i + 1]) for i, a in enumerate(argv) if a == "--jobs"), 1)
    if "--run" in argv:
        for p in run(only or None, jobs):
            print(f"Saved {p.name} in runs/inbox/ (file it with scripts/add_run_record.py).")
        return 0
    if "--reproduce" in argv:
        return 0 if reproduce(only or None, jobs) else 1
    check = "--check" in argv
    files, computed = build()
    if check:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m05.py")
        return 1 if stale else 0
    IMAGES.mkdir(exist_ok=True)
    for p, data in files.items():
        p.write_bytes(data)
        print(f"wrote {p.relative_to(ROOT)} ({len(data) / 1024:.0f} KiB)")
    print()
    report(computed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
