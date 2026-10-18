"""Module 4 figures, from exact computations and recorded experiments: light and dark SVGs.

  uv run python scripts/make_figures_m04.py --run [--only NAME]  # run the experiments; records go to runs/inbox/
  uv run python scripts/add_run_record.py             # file them in runs/
  uv run python scripts/make_figures_m04.py           # draw images/04-*.svg from the filed records
  uv run python scripts/make_figures_m04.py --check   # fail if a committed SVG differs from a fresh drawing
  uv run python scripts/make_figures_m04.py --reproduce [--only NAME]  # rerun in memory, compare
                                                      # with the filed records, write nothing

The experiments use the lab's own reference code: the worked notebook's definitions (its
setup, provided cells and solution cells, and the functions its run cells define) run
in-process, with solutions bound. Nothing here reimplements a policy or the loop.

  04-hoeffding        Exact: a fair coin's sample mean after N = 10 and N = 100 tosses, the
                      binomial tail P(mean >= 0.5 + eta) and Hoeffding's bound exp(-2 N eta^2).
  04-beta-posteriors  Exact: the densities of Beta(21, 31), Beta(4, 8) and Beta(4, 2).
  04-regret-curves    m04-regret-curves: greedy, epsilon-greedy (0.1), UCB1 and Thompson
                      sampling on seeds 0-19 for 20 000 rounds; the IQM per round with a 95%
                      bootstrap band (prl.evaluate's bootstrap, vectorized), log rounds.
  04-scale            m04-scale: pseudo-regret at 5000 rounds in the original units, every
                      seed, for UCB1 and epsilon-greedy with rewards x 1 and x 10.

A third experiment has no figure: m04-seats, the 100-seat scan of what every seat's lab
computes (seats 0-99, seeds (seat + 1) * 1000 + i for i = 0..19, the live budget), for
the debrief and the predict_answer entries (OPEN_QUESTIONS B32-B34). Epsilon-greedy and UCB1
run on to 20 000 rounds there too: a longer reward stack starts with the shorter one and the
policy's generator is the same, so each 20 000-round run continues its 5000-round run exactly.

Every build runs quoted_numbers(): each number the page, a caption, an alt text, a knowledge
check or a predict_answer quotes from these records or from an exact computation is asserted,
so a changed record fails here before the page is wrong. The figures read only the newest
filed record of each experiment (runs/*.json), so `--check` never reruns anything. Drawing is
deterministic: svg.hashsalt is the figure name, no date in the metadata, text as paths, values
rounded before plotting.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from fractions import Fraction
from functools import partial
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_protocol as tp  # noqa: E402
from make_expected import m04_prob_best  # noqa: E402
from make_figures_m01 import MIN_FONT, WIDTH_IN, Theme, rc, svg_bytes, themes  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.checks import expected  # noqa: E402
from prl.evaluate import bootstrap_ci, compare, iqm  # noqa: E402
from prl.lab import _default_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
RUNS = ROOT / "runs"
NOTEBOOK = ROOT / "notebooks" / "04-bandits.ipynb"
SCRIPT = "scripts/make_figures_m04.py"

SEEDS = list(range(20))
GRID = sorted(
    set(np.unique(np.round(np.geomspace(1, 20000, 120)).astype(int)).tolist())
    | {1000, 2500, 5000, 20000}
)
HORIZONS = [1000, 2500, 5000, 20000]
CURVES = {
    "T": 20000,
    "methods": {
        "greedy": "eps_greedy, epsilon = 0",
        "eps-greedy": "eps_greedy, epsilon = 0.1",
        "UCB1": "ucb1 (argmax_tie of ucb1_index)",
        "TS": "thompson_step",
    },
    "grid": GRID,
    "horizons": HORIZONS,
}
SCALE = {
    "T": 5000,
    "scale": 10.0,
    "configs": {
        "eps_greedy_x1": "eps_greedy (0.1), rewards x 1",
        "eps_greedy_x10": "eps_greedy (0.1), rewards x 10",
        "ucb1_x1": "ucb1, rewards x 1",
        "ucb1_x10": "ucb1, rewards x 10, the same bonus",
        "ucb1_rescaled_x10": "ucb1_scaled with B = 10, rewards x 10",
    },
}
SEATS = list(range(100))
SEAT = {
    "seeds_per_seat": 20,
    "seat_seeds": "(seat + 1) * 1000 + i, i = 0..19",
    "T": 5000,
    "early": 1000,
    "late": 20000,
    "scale": 10.0,
    "epsilons": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
    "workers": 8,
}
EXPERIMENTS = {
    "04-regret-curves": "m04-regret-curves",
    "04-scale": "m04-scale",
}
SEATS_EXPERIMENT = "m04-seats"


# ------------------------------------------------------------------ the lab's own code
def _defs_only(src: str) -> str:
    """The top-level def and class statements of a run cell (it also runs and plots)."""
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def lab_functions() -> dict:
    """The worked notebook's definitions, solutions bound, without running its experiments.

    Executes the setup stub, the init cell, the provided cells, the solution cells and the
    function definitions of the run cells (ucb1, ucb1_scaled, play). The colleague's
    unfixed agent, the ex6-stub cell, is added as `UCB1SuspectBuggy`.
    """
    import nbformat

    matplotlib_backend = mpl.get_backend()
    mpl.use("Agg", force=True)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {"__name__": "__main__"}
    stub6 = None
    with tp._environment(PRL_QUICK="0", PRL_WORKED="1"), contextlib.redirect_stdout(io.StringIO()):
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            cid, src = cell.get("id", ""), cell.source
            if cid.startswith("stretch"):
                break
            if cid == "gen-setup":
                src = tp.SETUP_STUB
            elif cid == "ex6-stub":
                stub6 = src
                continue
            elif re.fullmatch(r"ex[0-9]+-run", cid):
                src = _defs_only(src)
            elif cid != "gen-init" and not re.fullmatch(r"c[0-9]+|ex[0-9]+-sol", cid):
                continue  # stubs, checkpoints, the finish cell
            exec(compile(src, f"<04-bandits cell {cid}>", "exec"), ns)  # noqa: S102
    assert stub6 is not None
    buggy = dict(ns)
    exec(compile(stub6, "<ex6-stub>", "exec"), buggy)  # noqa: S102
    ns["UCB1SuspectBuggy"] = buggy["UCB1Suspect"]
    if matplotlib_backend.lower() != "agg":
        mpl.use("Agg", force=True)
    return ns


def _r6(x: float) -> float:
    return round(float(x), 6)


def _arms_sha(arms: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arms, dtype=np.int8).tobytes()).hexdigest()[:16]


def run_curves(ns: dict) -> dict:
    """Pseudo-regret at the grid rounds and pull counts at the horizons, per method and seed."""
    T, R, run, stack = CURVES["T"], ns["R_MEAN"], ns["run_bandit"], ns["reward_stack"]
    policies = {
        "greedy": partial(ns["eps_greedy"], epsilon=0.0),
        "eps-greedy": partial(ns["eps_greedy"], epsilon=0.1),
        "UCB1": ns["ucb1"],
        "TS": ns["thompson_step"],
    }
    idx = np.array(GRID) - 1  # regret after t rounds is regret[t - 1]
    per: dict = {}
    for name, policy in policies.items():
        regret, counts = [], []
        for k in SEEDS:
            arms, reg = run(policy, stack(R, T, k), R, k)
            regret.append([float(x) for x in reg[idx]])
            counts.append([np.bincount(arms[:h], minlength=3).tolist() for h in HORIZONS])
        per[name] = {"regret": regret, "counts": counts}
    return per


def run_scale(ns: dict) -> dict:
    """Final pseudo-regret / scale per seed, and a hash of each run's arms."""
    T, B, R, run, stack = (
        SCALE["T"],
        SCALE["scale"],
        ns["R_MEAN"],
        ns["run_bandit"],
        ns["reward_stack"],
    )
    eps = partial(ns["eps_greedy"], epsilon=0.1)
    configs = {
        "eps_greedy_x1": (eps, 1.0),
        "eps_greedy_x10": (eps, B),
        "ucb1_x1": (ns["ucb1"], 1.0),
        "ucb1_x10": (ns["ucb1"], B),
        "ucb1_rescaled_x10": (partial(ns["ucb1_scaled"], B=B), B),
    }
    per: dict = {}
    for name, (policy, scale) in configs.items():
        values, shas = [], []
        for k in SEEDS:
            arms, reg = run(policy, stack(R, T, k, scale=scale), scale * R, k)
            values.append(float(reg[-1] / scale))
            shas.append(_arms_sha(arms))
        per[name] = {"regret": values, "arms_sha": shas}
    summary = {}
    for name, base in (
        ("eps_greedy", "eps_greedy_x1"),
        ("ucb1", "ucb1_x1"),
        ("ucb1_rescaled", "ucb1_x1"),
    ):
        a, b = np.array(per[base]["regret"]), np.array(per[f"{name}_x10"]["regret"])
        summary[name] = {
            "same_arms_seeds": sum(
                x == y
                for x, y in zip(per[base]["arms_sha"], per[f"{name}_x10"]["arms_sha"], strict=True)
            ),
            "mean_x1": float(a.mean()),
            "mean_x10": float(b.mean()),
            "iqm_x1": iqm(a),
            "iqm_x10": iqm(b),
            "seeds_lower": int(np.sum(b < a - 1e-9)),
            "seeds_higher": int(np.sum(b > a + 1e-9)),
            "verdict_x10_vs_x1": ns["verdict"](b, a),
        }
    per["summary"] = summary
    return per


# ------------------------------------------------------------------ the 100-seat scan
_WORKER_NS: dict = {}


def _worker_init() -> None:
    _WORKER_NS.update(lab_functions())


def _seat_worker(seat: int) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return seat_run(_WORKER_NS, seat)


def seat_run(ns: dict, seat: int) -> dict:
    """What seat `seat`'s lab computes at the live budget, per seed, plus its summaries.

    Raw values are rounded to 6 decimals (every pseudo-regret here is a multiple of 0.1).
    The summaries are computed from the unrounded values with the lab's own `verdict` and
    prl.evaluate's iqm and bootstrap_ci, as the run cells do.
    """
    b = SEAT
    T, early, late, B = b["T"], b["early"], b["late"], b["scale"]
    R, run, stack, verdict = ns["R_MEAN"], ns["run_bandit"], ns["reward_stack"], ns["verdict"]
    eps_greedy, ucb1 = ns["eps_greedy"], ns["ucb1"]
    seeds = [(seat + 1) * 1000 + i for i in range(b["seeds_per_seat"])]
    raw: dict = {
        k: []
        for k in (
            "greedy",
            "greedy_best_second_half",
            "eps-greedy",
            "UCB1",
            "TS",
            "ucb1_pulls",
            "ts_pulls",
            "ts_h1_halves",
            "eps_greedy_x10",
            "ucb1_x10",
            "ucb1_rescaled_x10",
            "eps_greedy_x10_same_arms",
            "ucb1_x10_same_arms",
            "ucb1_rescaled_x10_same_arms",
            "suspect_pulls",
        )
    }
    sweep = {e: [] for e in b["epsilons"]}
    full: dict = {"greedy": [], "eps-greedy": [], "UCB1": [], "TS": [], "x10": {}}
    x10 = {"eps_greedy": [], "ucb1": [], "ucb1_rescaled": []}
    for k in seeds:
        arms, reg = run(partial(eps_greedy, epsilon=0.0), stack(R, T, k), R, k)
        full["greedy"].append(reg[-1])
        raw["greedy_best_second_half"].append(int(np.sum(arms[T // 2 :] == 2)))
        base = {}
        for name, policy in (("eps-greedy", partial(eps_greedy, epsilon=0.1)), ("UCB1", ucb1)):
            arms, reg = run(policy, stack(R, late, k), R, k)  # continues the 5000-round run
            full[name].append((reg[early - 1], reg[T - 1], reg[late - 1]))
            base[name] = arms[:T]
            if name == "UCB1":
                raw["ucb1_pulls"].append(np.bincount(arms[:T], minlength=3).tolist())
        arms, reg = run(ns["thompson_step"], stack(R, T, k), R, k)
        full["TS"].append((reg[early - 1], reg[T - 1]))
        raw["ts_pulls"].append(np.bincount(arms, minlength=3).tolist())
        raw["ts_h1_halves"].append(
            [int(np.sum(arms[: T // 2] == 1)), int(np.sum(arms[T // 2 :] == 1))]
        )
        for name, policy, ref in (
            ("eps_greedy", partial(eps_greedy, epsilon=0.1), base["eps-greedy"]),
            ("ucb1", ucb1, base["UCB1"]),
            ("ucb1_rescaled", partial(ns["ucb1_scaled"], B=B), base["UCB1"]),
        ):
            arms, reg = run(policy, stack(R, T, k, scale=B), B * R, k)
            x10[name].append(reg[-1] / B)
            raw[f"{name}_x10_same_arms"].append(bool(np.array_equal(arms, ref)))
        for e in b["epsilons"]:
            if e == 0.0:
                sweep[e].append(full["greedy"][-1])
            elif e == 0.1:
                sweep[e].append(full["eps-greedy"][-1][1])
            else:
                sweep[e].append(run(partial(eps_greedy, epsilon=e), stack(R, T, k), R, k)[1][-1])
        agent = ns["UCB1SuspectBuggy"](3)
        arms, _ = ns["play"](agent, stack(R, T, k), R)
        raw["suspect_pulls"].append(np.bincount(arms, minlength=3).tolist())
    # raw values
    raw["greedy"] = [_r6(x) for x in full["greedy"]]
    for name in ("eps-greedy", "UCB1", "TS"):
        raw[name] = [[_r6(x) for x in row] for row in full[name]]
    for name, vals in x10.items():
        raw[f"{name}_x10"] = [_r6(x) for x in vals]
    raw["sweep"] = {str(e): [_r6(x) for x in vals] for e, vals in sweep.items()}
    # summaries, as the lab computes them
    at = {
        h: {m: np.array([row[j] for row in full[m]]) for m in ("eps-greedy", "UCB1", "TS")}
        for j, h in enumerate((early, T))
    }
    at[late] = {m: np.array([row[2] for row in full[m]]) for m in ("eps-greedy", "UCB1")}
    verdicts = {}
    for h in (early, T):
        for a, c in (("TS", "UCB1"), ("TS", "eps-greedy"), ("UCB1", "eps-greedy")):
            verdicts[f"{a} vs {c} @ {h}"] = verdict(at[h][a], at[h][c])
    verdicts[f"UCB1 vs eps-greedy @ {late}"] = verdict(at[late]["UCB1"], at[late]["eps-greedy"])
    iqms = {
        f"{m} @ {h}": [round(iqm(x), 4), *(round(v, 4) for v in bootstrap_ci(x, iqm))]
        for h, d in at.items()
        for m, x in d.items()
    }
    x1 = {"eps_greedy": at[T]["eps-greedy"], "ucb1": at[T]["UCB1"], "ucb1_rescaled": at[T]["UCB1"]}
    scale = {}
    for name in x10:
        a, c = x1[name], np.array(x10[name])
        same = bool(np.allclose(a, c, rtol=0, atol=1e-9))
        scale[name] = {
            "direction": "same" if same else ("up" if c.mean() > a.mean() else "down"),
            "mean_x1": round(float(a.mean()), 4),
            "mean_x10": round(float(c.mean()), 4),
            "iqm_x1": round(iqm(a), 4),
            "iqm_x10": round(iqm(c), 4),
            "higher": int(np.sum(c > a + 1e-9)),
            "lower": int(np.sum(c < a - 1e-9)),
            "verdict_x10_vs_x1": verdict(c, a),
        }
    best = np.array(raw["greedy_best_second_half"]) / (T - T // 2)
    pulls = np.array(raw["ucb1_pulls"])
    room = {
        "greedy_locked": int(np.sum(best < 0.5)),
        "ucb1_pulls_h1": round(float(pulls[:, 1].mean()), 1),
        "eps_iqm": round(iqm(at[T]["eps-greedy"]), 1),
        "ucb1_iqm": round(iqm(at[T]["UCB1"]), 1),
        "ts_iqm": round(iqm(at[T]["TS"]), 1),
        "ucb1_x10_gap": round(scale["ucb1"]["mean_x10"] - scale["ucb1"]["mean_x1"], 1),
        "ucb1_x10_lower": scale["ucb1"]["lower"],
        "ucb1_late_lead": round(iqm(at[late]["eps-greedy"]) - iqm(at[late]["UCB1"]), 1),
    }
    return {
        "seeds": seeds,
        "raw": raw,
        "verdicts": verdicts,
        "iqm": iqms,
        "scale": scale,
        "room": room,
        "sweep_iqm": {str(e): round(iqm(np.array(v)), 4) for e, v in sweep.items()},
    }


def run_seats(_ns: dict, seats: list[int] | None = None) -> dict:
    seats = SEATS if seats is None else seats
    ctx = mp.get_context("spawn")
    with ctx.Pool(SEAT["workers"], initializer=_worker_init) as pool:
        results = pool.map(_seat_worker, seats, chunksize=1)
    out = {"summary": [seat_summary(r) for r in results]}
    if 0 in seats:
        out["seat_0"] = results[seats.index(0)]
    return out


def seat_summary(r: dict) -> dict:
    """One seat's outcomes, without its per-seed values (the record keeps those for seat 0)."""
    raw = r["raw"]
    h1 = [p[1] for p in raw["ucb1_pulls"]]
    ts_h1 = [p[1] for p in raw["ts_pulls"]]
    suspect = np.array(raw["suspect_pulls"])
    return {
        "verdicts": r["verdicts"],
        "iqm": {k: v[0] for k, v in r["iqm"].items()},
        "scale": r["scale"],
        "room": r["room"],
        "sweep_iqm": r["sweep_iqm"],
        "ucb1_pulls_h1_range": [min(h1), max(h1)],
        "ts_pulls_h1": [round(float(np.mean(ts_h1)), 2), min(ts_h1), max(ts_h1)],
        "x10_same_arms_seeds": {
            k: sum(raw[f"{k}_x10_same_arms"]) for k in ("eps_greedy", "ucb1", "ucb1_rescaled")
        },
        "suspect_lead": np.bincount(suspect.argmax(axis=1), minlength=3).tolist(),
        "suspect_winner_pulls": sorted(set(suspect.max(axis=1).tolist())),
    }


# ------------------------------------------------------------------ records
def experiment_record(
    name: str, budget: dict, seeds: list, per_seed: dict, seconds: float, note: str, git
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
            "notebook": "notebooks/04-bandits.ipynb",
            "notebook_content_sha": meta.get("content_sha"),
            "code": "the worked notebook's definitions and reference solutions, run in-process",
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
    "m04-regret-curves": (
        "Greedy, epsilon-greedy (0.1), UCB1 and Thompson sampling from the worked notebook, on "
        "reward_stack(R_MEAN, 20000, k) with policy seed k, seeds 0-19. per_seed[method]['regret'][i][j] "
        "is the pseudo-regret after budget['grid'][j] rounds on seeds[i]; ['counts'][i][h] the pulls "
        "of each headline in the first budget['horizons'][h] rounds."
    ),
    "m04-scale": (
        "UCB1 and epsilon-greedy (0.1) with rewards x 1 and x 10, and UCB1 with the bonus x 10 "
        "(the lab's ucb1_scaled, B = 10), 5000 rounds, seeds 0-19. per_seed[config]['regret'][i] is "
        "the final pseudo-regret divided by the scale (original units); 'arms_sha' hashes the arms "
        "pulled, so equal hashes mean identical runs. per_seed['summary'] gives the x 10 outcome "
        "per method: mean and IQM with rewards x 1 and x 10, seeds lower and higher, seeds with "
        "identical arms, and the lab's verdict (x 10 vs x 1)."
    ),
    SEATS_EXPERIMENT: (
        "What every seat's lab computes at the live budget, for seats 0-99 (seeds (seat + 1) * 1000 "
        "+ i). per_seed['summary'] is indexed like `seeds` (the seat numbers): each seat's "
        "verdicts at 1000, 5000 and 20 000 rounds, IQMs, the x 10 outcome, the room-code scalars "
        "(including ucb1_x10_lower and ucb1_late_lead) and the epsilon sweep's IQMs. "
        "per_seed['seat_0'] also holds seat 0's per-seed values ('raw'): epsilon-greedy and UCB1 "
        "as [regret at 1000, 5000, 20000] (the 20 000-round runs continue the 5000-round runs "
        "exactly), Thompson sampling as [1000, 5000], the rest at 5000. Everything is computed "
        "from the unrounded values with the lab's verdict and prl.evaluate. "
        f"{SEAT['workers']} worker processes."
    ),
}
RUNNERS = {
    "m04-regret-curves": (run_curves, CURVES, SEEDS),
    "m04-scale": (run_scale, SCALE, SEEDS),
    SEATS_EXPERIMENT: (run_seats, SEAT, SEATS),
}


def run(inbox: Path = tp.INBOX, only: list[str] | None = None) -> list[Path]:
    git = tp._git_state()  # the repo as it is when the run starts
    load_start = os.getloadavg()[0]
    tp.ensure_current(NOTEBOOK)
    ns = lab_functions()
    out = []
    for name, (runner, budget, seeds) in RUNNERS.items():
        if only and name not in only:
            continue
        start = time.perf_counter()
        per = runner(ns)
        secs = time.perf_counter() - start
        load = os.getloadavg()[0]
        note = (
            f"{NOTES[name]} Wall time on {where()}: a laptop time, not a Colab time. One-minute "
            f"load average {load_start:.0f} at the start and {load:.0f} at the end: other builds "
            "were running on this machine, so the time is inflated."
        )
        rec = experiment_record(name, dict(budget), list(seeds), per, secs, note, git)
        out.append(tp.write_record(rec, inbox))
    return out


def _diff(old, new, path: str, out: list[str], floats: list[float]) -> None:
    """Exact comparison of JSON-like values, except floats (collected as |differences|)."""
    if isinstance(old, float) or isinstance(new, float):
        if (
            old is None
            or new is None
            or isinstance(old, (str, bool))
            or isinstance(new, (str, bool))
        ):
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


def reproduce(only: list[str] | None = None) -> bool:
    """Rerun each experiment in memory and compare with its newest filed record."""
    ns = lab_functions()
    ok = True
    for name, (runner, budget, seeds) in RUNNERS.items():
        if only and name not in only:
            continue
        path, rec = newest_record(name)
        if rec["budget"] != json.loads(json.dumps(budget)) or rec["seeds"] != list(seeds):
            print(f"{name}: runs/{path.name} has a different budget or seeds; not comparable")
            ok = False
            continue
        new = json.loads(json.dumps(runner(ns)))  # the record's JSON types
        exact: list[str] = []
        floats: list[float] = []
        _diff(rec["per_seed"], new, "per_seed", exact, floats)
        worst = max(floats, default=0.0)
        good = not exact and worst <= 1e-9
        ok &= good
        print(
            f"{name}: {'REPRODUCED' if good else 'DIFFERS'} against runs/{path.name} "
            f"({len(floats)} floats, max |diff| {worst:.3g}; {len(exact)} exact mismatches)"
        )
        for line in exact[:10]:
            print(f"  {line}")
    return ok


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


# ------------------------------------------------------------------ the data of each figure
ETA = np.round(np.linspace(0.0, 0.5, 501), 6)
HOEFF_N = (10, 100)


def binomial_tail(n: int, eta: float) -> float:
    """P(sample mean of n fair coins >= 0.5 + eta), exactly (rational, then float)."""
    k0 = math.ceil(Fraction(n) * (Fraction(1, 2) + Fraction(str(eta))))
    return float(Fraction(sum(math.comb(n, k) for k in range(k0, n + 1)), 2**n))


def hoeffding_data() -> dict:
    lines = {}
    for n in HOEFF_N:
        lines[n] = {
            "exact": np.array([binomial_tail(n, e) for e in ETA]),
            "bound": np.exp(-2 * n * ETA**2),
        }
    return {"eta": ETA, "lines": lines}


POSTERIORS = [(21, 31), (4, 8), (4, 2)]  # counts (50, 10, 4), clicks (20, 3, 3)
SHOWINGS = [(50, 20), (10, 3), (4, 3)]
X = np.round(np.linspace(0.0, 1.0, 401), 6)


def beta_data() -> dict:
    return {
        "x": X,
        "pdf": [np.round(stats.beta.pdf(X, u, v), 5) for u, v in POSTERIORS],
        "p_best": m04_prob_best(POSTERIORS),
    }


BOOT, LEVEL = 5000, 0.95  # prl.evaluate.bootstrap_ci's defaults


def iqm_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQM over seeds (axis 0) and prl.evaluate.bootstrap_ci(x, iqm) at every column, vectorized:
    the same resampled indices (default_rng(0), n_boot 5000) for every column."""
    n = values.shape[0]
    idx = np.random.default_rng(0).integers(0, n, size=(BOOT, n))
    center = stats.trim_mean(values, 0.25, axis=0)
    lo, hi = np.empty(values.shape[1]), np.empty(values.shape[1])
    for start in range(0, values.shape[1], 32):
        cols = values[:, start : start + 32]
        boots = stats.trim_mean(cols[idx], 0.25, axis=1)  # (BOOT, cols)
        q = np.quantile(boots, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2], axis=0)
        lo[start : start + 32], hi[start : start + 32] = q[0], q[1]
    return center, lo, hi


def curves_data() -> dict:
    path, rec = newest_record(EXPERIMENTS["04-regret-curves"])
    grid = rec["budget"]["grid"]
    lines = {}
    for m in rec["budget"]["methods"]:
        values = np.array(rec["per_seed"][m]["regret"])
        c, lo, hi = iqm_band(values)
        lines[m] = {"values": values, "iqm": c, "lo": lo, "hi": hi}
    return {"record": path.name, "rec": rec, "grid": np.array(grid), "lines": lines}


def scale_data() -> dict:
    path, rec = newest_record(EXPERIMENTS["04-scale"])
    per = rec["per_seed"]
    return {
        "record": path.name,
        "rec": rec,
        "values": {k: np.array(v["regret"]) for k, v in per.items()},
        "sha": {k: v["arms_sha"] for k, v in per.items()},
    }


# ------------------------------------------------------------------ figure 1: Hoeffding
def _sci(v: float, _p=None) -> str:
    if v >= 0.01:
        return f"{v:g}"
    return f"10$^{{{int(round(math.log10(v)))}}}$".replace("-", "\N{MINUS SIGN}")


def hoeffding_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.2))  # 680 x 403 px
    ax = fig.add_axes((0.115, 0.135, 0.52, 0.82))
    handles, labels = [], []
    for i, n in enumerate(HOEFF_N):
        line = data["lines"][n]
        color = t.series[i]
        h1 = ax.plot(data["eta"], line["bound"], color=color, lw=2.2, ls="-")[0]
        h2 = ax.plot(
            data["eta"],
            line["exact"],
            color=color,
            lw=2.0,
            ls=(0, (3.5, 1.8)),
            drawstyle="steps-post",
        )[0]
        handles += [h1, h2]
        labels += [f"N = {n}: bound", f"N = {n}: exact"]
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlim(0, 0.5)
    ax.yaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(_sci))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.grid(True, axis="y", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("\N{GREEK SMALL LETTER ETA}: how far above 0.5")
    ax.set_ylabel("P(sample mean \N{GREATER-THAN OR EQUAL TO} 0.5 + \N{GREEK SMALL LETTER ETA})")
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.66, 0.96),
        frameon=False,
        title="A fair coin, N tosses\nbound: e$^{\N{MINUS SIGN}2N\N{GREEK SMALL LETTER ETA}^2}$\nexact: binomial tail",
        alignment="left",
        handlelength=2.6,
    )
    return fig


# ------------------------------------------------------------------ figure 2: Beta posteriors
BETA_STYLES = ("-", (0, (4, 1.8)), (0, (1, 1.6)))


def beta_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 3.9))  # 680 x 374 px
    ax = fig.add_axes((0.085, 0.15, 0.89, 0.81))
    handles, labels = [], []
    for a, (pdf, (n, s)) in enumerate(zip(data["pdf"], SHOWINGS, strict=True)):
        h = ax.plot(data["x"], pdf, color=t.series[a], lw=2.4, ls=BETA_STYLES[a])[0]
        handles.append(h)
        labels.append(f"headline {a} ({n} showings, {s} clicks)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8.4)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator([0, 2, 4, 6, 8]))
    ax.grid(True, axis="y", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("click rate x")
    ax.set_ylabel("posterior density")
    ax.legend(handles, labels, loc="upper right", frameon=False, handlelength=2.6)
    return fig


# ------------------------------------------------------------------ figure 3: regret curves
CURVE_NAMES = {
    "greedy": "greedy",
    "eps-greedy": "ε-greedy, ε = 0.1",
    "UCB1": "UCB1",
    "TS": "Thompson sampling",
}
CURVE_STYLES = {
    "greedy": (0, (1, 1.6)),
    "eps-greedy": (0, (4.5, 1.8)),
    "UCB1": (0, (6, 1.8, 1.4, 1.8)),
    "TS": "-",
}
YMAX = 400


def curves_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.4))  # 680 x 422 px
    ax = fig.add_axes((0.10, 0.135, 0.87, 0.835))
    x = data["grid"]
    handles, labels = [], []
    order = ["greedy", "eps-greedy", "UCB1", "TS"]
    for i, m in enumerate(order):
        line = data["lines"][m]
        color = t.series[i]
        lo, hi = np.round(line["lo"], 3), np.round(line["hi"], 3)
        ax.fill_between(x, lo, np.minimum(hi, YMAX * 1.2), color=color, alpha=0.18, lw=0)
        h = ax.plot(x, np.round(line["iqm"], 3), color=color, lw=2.2, ls=CURVE_STYLES[m])[0]
        handles.append(h)
        labels.append(CURVE_NAMES[m])
    ax.axvline(5000, color=t.neutral, lw=1.4, ls="-", zorder=0)
    ax.text(
        4600,
        YMAX * 0.97,
        "5000 rounds:\nthe lab's T",
        ha="right",
        va="top",
        color=t.text,
        fontsize=MIN_FONT,
    )
    ax.set_xscale("log")
    ax.set_xlim(1, 20000)
    ax.set_ylim(0, YMAX)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v):,}".replace(",", " ")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0, 100, 200, 300, 400]))
    ax.grid(True, axis="y", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("round t (log scale)")
    ax.set_ylabel("pseudo-regret")
    n = len(data["rec"]["seeds"])
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        title=f"IQM of {n} seeds; bands: 95%\nbootstrap interval of the IQM",
        alignment="left",
        handlelength=2.8,
    )
    return fig


# ------------------------------------------------------------------ figure 4: rewards x 10
SCALE_COLS = [
    ("ucb1_x1", "UCB1\n× 1", "o", False),
    ("ucb1_x10", "UCB1\n× 10", "^", True),
    ("eps_greedy_x1", "ε-greedy\n× 1", "s", False),
    ("eps_greedy_x10", "ε-greedy\n× 10", "D", True),
]


def scale_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.0))  # 680 x 384 px
    ax = fig.add_axes((0.165, 0.17, 0.52, 0.80))
    jitter = (np.arange(len(SEEDS)) % 5 - 2) * 0.07
    for j, (key, _label, marker, filled) in enumerate(SCALE_COLS):
        color = t.series[2] if key.startswith("ucb1") else t.series[1]
        v = np.round(data["values"][key], 2)
        ax.plot(
            j + jitter,
            v,
            ls="none",
            marker=marker,
            ms=6.5,
            mew=1.4,
            mec=color,
            mfc=color if filled else t.bg,
        )
        ax.plot(
            [j - 0.3, j + 0.3],
            [round(float(v.mean()), 2)] * 2,
            color=t.text,
            lw=1.6,
            ls=(0, (3, 1.6)),
        )
        ax.plot([j - 0.3, j + 0.3], [round(iqm(v), 2)] * 2, color=t.text, lw=2.2, ls="-")
    ax.set_xlim(-0.6, len(SCALE_COLS) - 0.4)
    ax.set_ylim(0, 1050)
    ax.xaxis.set_major_locator(FixedLocator(list(range(len(SCALE_COLS)))))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: SCALE_COLS[int(round(v))][1]))
    ax.yaxis.set_major_locator(FixedLocator([0, 250, 500, 750, 1000]))
    ax.grid(True, axis="y", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("pseudo-regret at 5000 rounds\n(original units)")
    handles = [
        Line2D([], [], color=t.text, lw=2.2, ls="-"),
        Line2D([], [], color=t.text, lw=1.6, ls=(0, (3, 1.6))),
        Line2D([], [], ls="none", marker="o", ms=6.5, mew=1.4, mec=t.muted, mfc=t.bg),
        Line2D([], [], ls="none", marker="^", ms=6.5, mew=1.4, mec=t.muted, mfc=t.muted),
    ]
    labels = ["IQM", "mean", "rewards × 1", "rewards × 10"]
    n = len(data["rec"]["seeds"])
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.71, 0.97),
        frameon=False,
        title=f"Every seed ({n}).\nAcross a column:",
        alignment="left",
        handlelength=2.4,
    )
    return fig


# ------------------------------------------------------------------ build, report, check
DATA = {
    "04-hoeffding": hoeffding_data,
    "04-beta-posteriors": beta_data,
    "04-regret-curves": curves_data,
    "04-scale": scale_data,
}
FIGURES = {
    "04-hoeffding": ("04-hoeffding", hoeffding_figure),
    "04-beta-posteriors": ("04-beta-posteriors", beta_figure),
    "04-regret-curves": ("04-regret-curves", curves_figure),
    "04-scale": ("04-scale", scale_figure),
}


def build() -> tuple[dict[Path, bytes], dict[str, dict]]:
    computed = {name: make() for name, make in DATA.items()}
    quoted_numbers(computed)
    out: dict[Path, bytes] = {}
    for name, (key, draw) in FIGURES.items():
        for t in themes():
            with mpl.rc_context(rc(name, t)):
                out[IMAGES / f"{name}{t.suffix}.svg"] = svg_bytes(draw(t, computed[key]))
    return out, computed


# ------------------------------------------------------------------ numbers the pages quote
HALVES: list[float] = []  # quoted values that are exact halves, rounded half up


def _r1(x: float) -> float:
    """Round to 1 decimal. A value within 1e-9 of a half (an IQM of ten multiples of 0.1 can be
    exactly 23.05) is rounded half up, whichever side float rounding left it on; a value near
    a half but not on it is refused, since its rounding could depend on the platform."""
    d = abs(x * 10 - math.floor(x * 10) - 0.5)
    if d < 1e-9:
        HALVES.append(x)
        return math.floor(x * 10 + 0.5) / 10
    assert d > 1e-6, x
    return round(x, 1)


def _iqm_ci(x: np.ndarray) -> tuple[float, float, float]:
    lo, hi = bootstrap_ci(x, iqm)
    return _r1(iqm(x)), _r1(lo), _r1(hi)


def exact_numbers(hoeff: dict, beta: dict) -> None:
    """Closed-form and exact numbers on the page and in the knowledge checks."""
    R = np.array([0.4, 0.3, 0.5])
    gaps = R.max() - R
    assert np.allclose(gaps, [0.1, 0.2, 0.0])
    # Hoeffding figure: values, "the bound lies above the exact tail everywhere", N = 100 falls faster.
    assert round(binomial_tail(10, 0.3), 5) == 0.05469 and round(binomial_tail(10, 0.3), 3) == 0.055
    assert round(math.exp(-2 * 10 * 0.3**2), 4) == 0.1653 and f"{math.exp(-1.8):.3f}" == "0.165"
    assert (
        round(binomial_tail(100, 0.1), 5) == 0.02844 and round(binomial_tail(100, 0.1), 3) == 0.028
    )
    assert round(math.exp(-2 * 100 * 0.1**2), 4) == 0.1353 and f"{math.exp(-2):.3f}" == "0.135"
    for n in HOEFF_N:
        line = hoeff["lines"][n]
        assert np.all(line["bound"] >= line["exact"] - 1e-15), n
    l10, l100 = hoeff["lines"][10], hoeff["lines"][100]
    assert np.all(l100["bound"][1:] < l10["bound"][1:])
    i2 = int(np.flatnonzero(ETA == 0.2)[0])
    assert l100["exact"][i2] < l10["exact"][i2]
    # Check yourself 3.1 and the bonus.
    c, s, t = np.array([90, 9, 1]), np.array([54.0, 4, 0]), 100
    bonus = np.sqrt(2 * math.log(t) / c)
    assert f"{2 * math.log(100):.3f}" == "9.210"
    assert [f"{b:.3f}" for b in bonus] == ["0.320", "1.012", "3.035"]
    assert [f"{v:.3f}" for v in s / c + bonus] == ["0.920", "1.456", "3.035"]
    assert Fraction(100) ** -4 == Fraction(1, 10**8)
    # Beta posteriors figure, Check yourself 4.1, KC 4.6.
    pb = beta["p_best"]
    assert (round(pb[2], 3), round(pb[0], 3), round(pb[1], 3)) == (0.869, 0.077, 0.054)
    assert [round(s / n, 2) for n, s in SHOWINGS] == [0.4, 0.3, 0.75]
    pdf = beta["pdf"]
    peaks = [float(beta["x"][np.argmax(p)]) for p in pdf]
    assert peaks == [0.4, 0.3, 0.75], peaks
    assert max(pdf[2]) < max(pdf[1]) < max(pdf[0])  # lower and wider, then the lowest
    sds = [stats.beta.std(u, v) for u, v in POSTERIORS]
    assert sds[0] < sds[1] < sds[2]
    q01, q99 = stats.beta.ppf([0.01, 0.99], 4, 2)
    assert 0.15 < q01 < 0.25 and q99 > 0.95, (q01, q99)  # "spreads from about 0.2 to 1"
    assert f"{4 / 12:.3f}" == "0.333" and f"{5 / 13:.3f}" == "0.385"
    assert f"{3 / 10:.1f}" == "0.3" and f"{4 / 11:.3f}" == "0.364"
    means = [u / (u + v) for u, v in POSTERIORS]
    assert [f"{m:.3f}" for m in means] == ["0.404", "0.333", "0.667"]
    p = means
    sample_reward = [
        p[0] + (1 - p[0]) * (1 - p[1]) * (1 - p[2]),
        (1 - p[0]) * p[1],
        (1 - p[0]) * (1 - p[1]) * p[2],
    ]
    assert [f"{v:.3f}" for v in sample_reward] == ["0.536", "0.199", "0.265"]
    # Check yourself 5.1, KC 4.8: epsilon-greedy's floor.
    for e, per_round, total in (
        (0.1, 0.01, 50),
        (0.01, 0.001, 5),
        (0.5, 0.05, 250),
        (0.05, 0.005, 25),
        (0.2, 0.02, 100),
    ):
        floor = e / 3 * gaps.sum()
        assert math.isclose(floor, per_round) and math.isclose(floor * 5000, total)
    assert round(3 / 0.01) == 300  # "once every 300 rounds"
    # KC 4.1, 4.2, 4.3, 4.4.
    arms = np.array([0, 1, 2, 2, 2, 0, 2, 2, 2, 2])
    rewards = np.array([1, 0, 0, 1, 1, 0, 1, 1, 1, 0])
    assert math.isclose(gaps[arms].sum(), 0.4) and math.isclose(
        gaps @ np.bincount(arms, minlength=3), 0.4
    )
    assert 10 * 0.5 - rewards.sum() == -1
    assert math.isclose(0.3 + 0.1 * 4997, 500.0) and math.isclose(0.3 + 0.2 * 4997, 999.7)
    assert f"{2 * math.log(1000):.2f}" == "13.82" and math.ceil(2 * math.log(1000) / 0.01) == 1382
    assert f"{math.sqrt(math.log(10)):.2f}" == "1.52"
    # Exercise 2's bound and Exercise 6's bonus (predict_answer).
    assert round(8 * math.log(5000) / 0.2**2 + 1 + math.pi**2 / 3) == 1708
    assert f"{math.sqrt(2 * math.log(5000)):.2f}" == "4.13"


def quoted_numbers(computed: dict[str, dict]) -> None:
    exact_numbers(computed["04-hoeffding"], computed["04-beta-posteriors"])
    # ---- figure 3 (regret curves), its caption and alt text
    c = computed["04-regret-curves"]
    grid = list(c["grid"])
    col = {h: grid.index(h) for h in HORIZONS}
    v = {m: line["values"] for m, line in c["lines"].items()}
    L = c["lines"]

    def stat(m: str, h: int) -> tuple[float, float, float]:  # IQM [95% CI], from the band
        j = col[h]
        return _r1(L[m]["iqm"][j]), _r1(L[m]["lo"][j]), _r1(L[m]["hi"][j])

    def winner(a: str, b: str, h: int) -> str:  # compare()'s rule, lower is better, 20 seeds
        j = col[h]
        if not (L[a]["hi"][j] < L[b]["lo"][j] or L[b]["hi"][j] < L[a]["lo"][j]):
            return "tie"
        return "A" if L[a]["iqm"][j] < L[b]["iqm"][j] else "B"

    # the band is prl.evaluate's bootstrap_ci, and winner() is compare(), where the page quotes them
    for m in ("TS", "eps-greedy", "UCB1"):
        lo, hi = bootstrap_ci(v[m][:, col[5000]], iqm)
        assert abs(L[m]["lo"][col[5000]] - lo) < 1e-9 and abs(L[m]["hi"][col[5000]] - hi) < 1e-9
    assert compare(v["UCB1"][:, -1], v["eps-greedy"][:, -1], higher_is_better=False).winner == "A"
    s5 = {m: stat(m, 5000) for m in ("TS", "eps-greedy", "UCB1")}
    assert s5 == {
        "TS": (23.1, 19.4, 26.8),
        "eps-greedy": (60.4, 53.6, 78.3),
        "UCB1": (113.8, 105.9, 120.0),
    }, s5
    for a, b in (("TS", "eps-greedy"), ("TS", "UCB1"), ("eps-greedy", "UCB1")):
        assert winner(a, b, 5000) == "A"
    s20 = {m: stat(m, 20000) for m in ("UCB1", "eps-greedy")}
    assert s20 == {"UCB1": (184.8, 169.8, 197.4), "eps-greedy": (211.3, 203.1, 228.8)}, s20
    assert winner("UCB1", "eps-greedy", 20000) == "A"
    s1 = {m: stat(m, 1000) for m in ("TS", "eps-greedy")}
    assert s1 == {"TS": (14.9, 11.8, 19.0), "eps-greedy": (17.8, 12.4, 36.3)}, s1
    assert winner("TS", "eps-greedy", 1000) == "tie"
    assert _r1(iqm(v["greedy"][:, col[5000]])) == 300.4
    counts = {m: np.array(c["rec"]["per_seed"][m]["counts"]) for m in c["lines"]}
    best_late = (
        counts["greedy"][:, HORIZONS.index(5000), 2] - counts["greedy"][:, HORIZONS.index(2500), 2]
    )
    assert int(np.sum(best_late / 2500 < 0.5)) == 11
    # alt text: greedy's band is very wide; UCB1 above epsilon-greedy at 5000 and below at
    # 20 000, crossing between; TS lowest throughout and nearly flat after a few hundred rounds
    widths = {m: line["hi"][col[5000]] - line["lo"][col[5000]] for m, line in c["lines"].items()}
    assert widths["greedy"] > 3 * max(w for m, w in widths.items() if m != "greedy")
    ucb, eps, ts = (c["lines"][m]["iqm"] for m in ("UCB1", "eps-greedy", "TS"))
    assert ucb[col[5000]] > eps[col[5000]] and ucb[-1] < eps[-1]
    cross = [g for g, a, b in zip(grid, ucb, eps, strict=True) if g >= 5000 and a < b]
    assert cross and cross[0] > 10000, cross[:1]
    # caption: epsilon-greedy keeps rising at a steady slope while UCB1's flattens
    assert eps[-1] - eps[col[5000]] > 2 * (ucb[-1] - ucb[col[5000]])
    others = np.minimum.reduce([c["lines"][m]["iqm"] for m in ("greedy", "eps-greedy", "UCB1")])
    # The alt text's "lowest throughout" holds from 250 rounds on; before that greedy (to round
    # 17) or epsilon-greedy (rounds 97-206) is lower, by at most 0.51 (reported to the director).
    late = np.array(grid) >= 250
    assert np.all(ts[late] <= others[late] + 1e-9)
    assert np.max(ts[~late] - others[~late]) < 0.6
    i500 = grid.index(min(g for g in grid if g >= 500))
    assert ts[-1] - ts[i500] < 0.05 * YMAX  # nearly flat on the plot's 0-400 scale
    assert eps[-1] - eps[i500] > 10 * (ts[-1] - ts[i500])
    # ---- figure 4 (rewards x 10), its caption and alt text, knowledge check 4.8
    sc = computed["04-scale"]
    x = sc["values"]
    up, down = x["ucb1_x10"] > x["ucb1_x1"] + 1e-9, x["ucb1_x10"] < x["ucb1_x1"] - 1e-9
    assert (int(up.sum()), int(down.sum())) == (7, 13)
    assert (_r1(x["ucb1_x10"][down].min()), _r1(x["ucb1_x10"][down].max())) == (0.3, 31.9)
    assert (_r1(x["ucb1_x10"][up].min()), _r1(x["ucb1_x10"][up].max())) == (342.2, 999.4)
    assert (_r1(x["ucb1_x1"].mean()), _r1(x["ucb1_x10"].mean())) == (112.2, 195.1)
    assert (_r1(iqm(x["ucb1_x1"])), _r1(iqm(x["ucb1_x10"]))) == (113.8, 90.1)
    assert compare(x["ucb1_x10"], x["ucb1_x1"], higher_is_better=False).winner == "tie"
    assert sc["sha"]["eps_greedy_x1"] == sc["sha"]["eps_greedy_x10"]  # 20 of 20 identical
    assert sc["sha"]["ucb1_x1"] == sc["sha"]["ucb1_rescaled_x10"]
    assert len(set(sc["sha"]["ucb1_x1"])) == 20
    assert np.allclose(x["eps_greedy_x1"], x["eps_greedy_x10"], rtol=0, atol=1e-9)
    assert (_r1(x["ucb1_x1"].min()), _r1(x["ucb1_x1"].max())) == (78.8, 140.4)  # "about 80 and 140"
    assert (_r1(x["eps_greedy_x1"].min()), _r1(x["eps_greedy_x1"].max())) == (45.7, 121.6)
    near_zero = x["ucb1_x10"][down]
    assert near_zero.max() < 35 and x["ucb1_x10"][up].min() > 340  # "13 near zero, 7 from 340"
    # the record agrees with checkpoint 5's fixture (make_expected's own implementation) and
    # with the regret-curves record at 5000 rounds (a 20 000-round run continues a 5000 one)
    fx = expected("m04_scale")
    for key in SCALE["configs"]:
        assert np.allclose(x[key], fx[key], rtol=0, atol=1e-9), key
    assert np.allclose(x["ucb1_x1"], v["UCB1"][:, col[5000]], rtol=0, atol=1e-9)
    assert np.allclose(x["eps_greedy_x1"], v["eps-greedy"][:, col[5000]], rtol=0, atol=1e-9)
    # ---- the predict_answer entries' seat 0, from the 100-seat scan (if filed)
    with contextlib.suppress(SystemExit):
        seat_numbers()


def seat_numbers() -> None:
    """Seat 0's numbers in the predict_answer entries (_variables.yml), from m04-seats."""
    _, rec = newest_record(SEATS_EXPERIMENT)
    s0 = rec["per_seed"]["seat_0"]
    i0 = rec["seeds"].index(0)
    assert seat_summary(s0) == rec["per_seed"]["summary"][i0]
    per = {k: [s0[k]] for k in s0}  # seat 0 alone, indexed like the summary lists
    i0 = 0
    raw = per["raw"][i0]
    assert per["seeds"][i0] == list(range(1000, 1020))
    greedy = np.array(raw["greedy"])
    best = np.array(raw["greedy_best_second_half"]) / 2500
    locked = best < 0.5
    assert int(locked.sum()) == 9 == per["room"][i0]["greedy_locked"]
    assert (_r1(greedy[~locked].min()), _r1(greedy[~locked].max())) == (0.3, 4.3)
    assert set(np.round(greedy[locked], 1).tolist()) <= {499.5, 500.0, 500.4, 999.7}
    pulls = np.array(raw["ucb1_pulls"])
    assert round(pulls[:, 1].mean()) == 227 and (pulls[:, 1].min(), pulls[:, 1].max()) == (123, 359)
    ts_pulls, halves = np.array(raw["ts_pulls"]), np.array(raw["ts_h1_halves"])
    assert round(ts_pulls[:, 1].mean()) == 48 and (ts_pulls[:, 1].min(), ts_pulls[:, 1].max()) == (
        15,
        105,
    )
    assert round(halves[:, 0].mean()) == 41 and _r1(halves[:, 1].mean()) == 6.5
    assert int(np.sum(halves[:, 1] == 0)) == 1
    room = per["room"][i0]
    assert (room["ts_iqm"], room["eps_iqm"], room["ucb1_iqm"]) == (22.8, 53.2, 103.7)
    early = {m: np.array([row[0] for row in raw[m]]) for m in ("TS", "eps-greedy")}
    assert (_r1(iqm(early["TS"])), _r1(iqm(early["eps-greedy"]))) == (15.3, 14.5)
    vd = per["verdicts"][i0]
    assert vd["TS vs eps-greedy @ 1000"] == "tie" and vd["UCB1 vs eps-greedy @ 1000"] == "B"
    assert all(
        vd[f"{p} @ 5000"] == w
        for p, w in (("TS vs UCB1", "A"), ("TS vs eps-greedy", "A"), ("UCB1 vs eps-greedy", "B"))
    )
    ucb = per["scale"][i0]["ucb1"]
    assert (_r1(ucb["mean_x1"]), _r1(ucb["mean_x10"])) == (103.8, 191.8)
    assert ucb["lower"] == 13 and ucb["verdict_x10_vs_x1"] == "tie"
    sw = per["sweep_iqm"][i0]
    assert (_r1(sw["0.0"]), _r1(sw["0.05"]), _r1(sw["0.5"])) == (200.9, 35.4, 252.4)
    sp = np.array(raw["suspect_pulls"])
    assert np.sort(sp, axis=1).tolist() == [[1, 1, 4998]] * len(sp)
    lead = np.bincount(sp.argmax(axis=1), minlength=3).tolist()
    assert lead == [11, 5, 4], lead


def report(computed: dict[str, dict]) -> None:
    if HALVES:
        print(
            f"Quoted values that are exact halves, rounded half up: {sorted(set(round(h, 6) for h in HALVES))}"
        )
    c = computed["04-regret-curves"]
    grid = list(c["grid"])
    print(f"04-regret-curves from runs/{c['record']}")
    for h in (1000, 5000, 20000):
        j = grid.index(h)
        parts = []
        for m, line in c["lines"].items():
            x = line["values"][:, j]
            lo, hi = bootstrap_ci(x, iqm)
            parts.append(f"{m} {iqm(x):.1f} [{lo:.1f}, {hi:.1f}]")
        print(f"  t = {h}: " + "; ".join(parts))
    sc = computed["04-scale"]
    print(f"04-scale from runs/{sc['record']}")
    for k, x in sc["values"].items():
        print(
            f"  {k:18s} mean {x.mean():7.1f} IQM {iqm(x):7.1f} min {x.min():7.1f} max {x.max():7.1f}"
        )
    with contextlib.suppress(SystemExit):
        seats_report()


def _tally(values: list) -> str:
    keys = sorted(set(values), key=str)
    return ", ".join(f"{k}: {values.count(k)}" for k in keys)


def seats_report() -> None:
    path, rec = newest_record(SEATS_EXPERIMENT)
    summary = rec["per_seed"]["summary"]
    n, k = len(rec["seeds"]), SEAT["seeds_per_seat"]
    print(f"{SEATS_EXPERIMENT} from runs/{path.name}: {n} seats of {k} seeds")
    for key in summary[0]["verdicts"]:
        print(f"  {key:30s} {_tally([x['verdicts'][key] for x in summary])}")
    for key in summary[0]["room"]:
        vals = np.array([x["room"][key] for x in summary], dtype=float)
        print(
            f"  room {key:16s} min {vals.min():7.1f} median {np.median(vals):7.1f} "
            f"max {vals.max():7.1f} (seat 0: {summary[0]['room'][key]})"
        )
    locked = sum(x["room"]["greedy_locked"] for x in summary)
    print(f"  greedy left headline 2 behind on {locked} of {n * k} seeds ({locked / (n * k):.3f})")
    for name in ("eps_greedy", "ucb1", "ucb1_rescaled"):
        sc = [x["scale"][name] for x in summary]
        same = sum(x["x10_same_arms_seeds"][name] for x in summary)
        print(
            f"  x10 {name:14s} direction {_tally([v['direction'] for v in sc])}; verdict x10 vs "
            f"x1 {_tally([v['verdict_x10_vs_x1'] for v in sc])}; identical arms on {same} of {n * k} seeds"
        )
    sc = [x["scale"]["ucb1"] for x in summary]
    lower = np.array([v["lower"] for v in sc])
    print(
        f"  UCB1 x10: mean up on {sum(v['mean_x10'] > v['mean_x1'] for v in sc)} of {n} seats, IQM "
        f"lower on {sum(v['iqm_x10'] < v['iqm_x1'] for v in sc)}; seeds lower per seat "
        f"{lower.min()}-{lower.max()} (total {lower.sum()} of {n * k})"
    )
    lead = np.sum([x["suspect_lead"] for x in summary], axis=0)
    winners = sorted({w for x in summary for w in x["suspect_winner_pulls"]})
    print(
        f"  buggy suspect: the winning arm's pulls {winners}; lead arm over all seeds {lead.tolist()}"
    )


def main(argv: list[str]) -> int:
    only = [argv[i + 1] for i, a in enumerate(argv) if a == "--only" and i + 1 < len(argv)]
    if "--run" in argv:
        for p in run(only=only or None):
            print(f"Saved {p.name} in runs/inbox/ (file it with scripts/add_run_record.py).")
        return 0
    if "--reproduce" in argv:
        return 0 if reproduce(only or None) else 1
    check = "--check" in argv
    files, computed = build()
    if check:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m04.py")
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
