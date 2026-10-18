"""Module 6 figures: light and dark SVGs from exact computations and one recorded experiment.

  uv run python scripts/make_figures_m06.py --run      # run the experiment; its record goes to runs/inbox/
  uv run python scripts/add_run_record.py              # file it in runs/
  uv run python scripts/make_figures_m06.py            # draw images/06-*.svg
  uv run python scripts/make_figures_m06.py --check    # fail if a committed SVG differs from a fresh drawing
  uv run python scripts/make_figures_m06.py --reproduce   # rerun the experiment in memory and
                                                       # compare with the filed record; write nothing

Same conventions as scripts/make_figures_m01.py (theme, rcParams, deterministic SVG: no date,
text as paths, values rounded before plotting).

  06-riverswim              The six-state river with our probabilities (prl.envs.RiverSwim's
                            arrays): the right action above, the left action below. Exact.
  06-hitting-time           h(0) under uniform random play against the chain length n = 2..16
                            (the linear solve of eq-hitting), log y-axis, the budget of 5000
                            steps as a dashed reference line. Exact.
  06-lines-visitation       Section 5's demo, two subfigure panels (layout [[62,38]]): the
  06-lines-regret           expected fraction of 5000 steps per state from the left end, uniform
                            play and pi*, on the 10-state river; uniform play's expected
                            cumulative regret against pi*'s expected rewards (and pi*'s, 0). Exact.
  06-regret-seeds           Cumulative regret against the step, every seed of the record
                            m06-regret-seeds (seeds 0-9): eps-greedy Q-learning, optimistic
                            Q-learning and MBIE-EB, 5000 steps on the 10-state river.

The experiment uses the lab's own reference code: the worked notebook (solutions bound, live
budget) runs in-process up to the stretch's run cell, as scripts/threshold_protocol.py does,
and its run_q_learning, run_planner, plan_mbie_eb, regret, visitation and first_visit do the
work. The colleague's agent of Exercise 7 (its ex7-stub cell, the planted bug) is recorded as
a fourth learner; the figure does not draw it. Nothing here reimplements a learner. The exact
figures use make_expected.py's solver code (value iteration, the hitting-time solve), which
may live in scripts, never in prl.

Every build runs quoted_numbers(): each number modules/06-exploration-mdps.qmd, its captions,
its alt text, the lab text and the m06 predict_answers quote from an exact computation or from
the record is asserted, and so is each claim about the learners' behavior ("confirm on the
record"). The figures read only the newest filed record (runs/*.json), so --check never
reruns anything. --reproduce compares a rerun with the record: discrete outcomes (first
visits) exactly, floats to 1e-9.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_protocol as tp  # noqa: E402
from make_expected import _hitting, _optimal, _policy_chain, _riverswim  # noqa: E402
from make_figures_m01 import MIN_FONT, WIDTH_IN, Theme, rc, svg_bytes, themes  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.evaluate import bootstrap_ci, compare, iqm  # noqa: E402
from prl.lab import _default_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
RUNS = ROOT / "runs"
NOTEBOOK = ROOT / "notebooks" / "06-exploration-mdps.ipynb"
SCRIPT = "scripts/make_figures_m06.py"

GAMMA = 0.95
N_CHAIN, STEPS = 10, 5000
SEEDS = list(range(10))
EXPERIMENT = "m06-regret-seeds"
EVERY = 25  # the record keeps the cumulative regret after 0, 25, 50, ..., 5000 steps
BUDGET = {
    "steps": STEPS,
    "n_chain": N_CHAIN,
    "gamma": GAMMA,
    "epsilon": 0.1,
    "alpha": 0.1,
    "q0_optimistic": 20.0,
    "c": 1.0,
    "tol": 0.01,
    "curve_every": EVERY,
}
LEARNERS = ("eps-greedy", "optimistic", "MBIE-EB", "suspect")
DRAWN = LEARNERS[:3]


# ------------------------------------------------------------------ the experiment
def lab_namespace() -> dict:
    """The worked notebook's namespace (live budget, solutions bound) before the stretch runs."""
    tp.ensure_current(NOTEBOOK)
    with tp._environment(PRL_QUICK="0", PRL_WORKED="1"):
        ns, _ = tp.solution_namespace(NOTEBOOK, "stretch")
    return ns


def buggy_suspect(ns: dict):
    """The colleague's mbie_eb_suspect as the notebook ships it (its ex7-stub cell).

    The worked namespace binds the fixed reference to that name, so the stub cell is run
    again in a copy of the namespace.
    """
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    (src,) = [c.source for c in nb.cells if c.get("id") == "ex7-stub"]
    local = dict(ns)
    exec(compile(src, "<ex7-stub>", "exec"), local)  # noqa: S102
    return local["mbie_eb_suspect"]


def run_regret(ns: dict) -> dict:
    """Per seed and learner: the regret curve, the final regret, visitation, first visit."""
    b = BUDGET
    assert ns["budget"]["steps"] == b["steps"] and ns["N_CHAIN"] == b["n_chain"]
    assert (ns["GAMMA"], ns["EPSILON"], ns["ALPHA"]) == (b["gamma"], b["epsilon"], b["alpha"])
    assert (ns["C_BONUS"], ns["TOL"]) == (b["c"], b["tol"])
    assert np.allclose(ns["Q0"], b["q0_optimistic"], atol=1e-9)  # 1 / (1 - 0.95), rounded
    suspect = buggy_suspect(ns)
    learners = dict(ns["learners"])  # Exercise 6's three, as the run cell defines them
    learners["suspect"] = lambda env, k: suspect(env, b["steps"], b["c"], b["gamma"], k)
    assert tuple(learners) == LEARNERS
    per: dict[str, list] = {}
    for seed in SEEDS:
        for name, run in learners.items():
            with contextlib.redirect_stdout(io.StringIO()):
                st, ac, _ = run(ns["make_river"](b["n_chain"], b["steps"]), seed)
            reg = ns["regret"](st, ac, ns["river"].R, ns["OPT"])  # Exercise 6's reference
            curve = np.concatenate([[0.0], reg])
            v = ns["visitation"](st[:-1], ns["S"])
            per.setdefault(f"{name}_regret", []).append(round(float(curve[-1]), 9))
            per.setdefault(f"{name}_curve", []).append([round(float(x), 6) for x in curve[::EVERY]])
            per.setdefault(f"{name}_visitation", []).append([round(float(x), 6) for x in v])
            per.setdefault(f"{name}_first_visit", []).append(ns["first_visit"](st, ns["END"]))
    return per


def experiment_record(per_seed: dict, seconds: float, note: str, git) -> dict:
    import nbformat

    rt = runtime.detect()
    git_sha, dirty = git
    meta = nbformat.read(NOTEBOOK, as_version=4).metadata.get("prl", {})
    return {
        "schema": record.SCHEMA,
        "kind": "experiment",
        "experiment": EXPERIMENT,
        "script": SCRIPT,
        "args": {
            "notebook": "notebooks/06-exploration-mdps.ipynb",
            "notebook_content_sha": meta.get("content_sha"),
            "code": "the worked notebook's reference solutions, run in-process (live budget)",
            "learners": list(LEARNERS),
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
        "budget": dict(BUDGET),
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "seconds": round(seconds, 1),
        "note": note,
    }


def where() -> str:
    rt = runtime.detect()
    return f"{_default_env(rt, 'colab-cpu')} ({rt.cpu}, {__import__('platform').machine()})"


NOTE = (
    "Exercise 6's three learners (eps-greedy Q-learning from Q0 = 0, optimistic Q-learning "
    "from Q0 = 20, MBIE-EB with c = 1) and the colleague's agent of Exercise 7 (its ex7-stub "
    "cell, the planted bug), one episode of 5000 steps from the left end of the 10-state "
    "river per seed, seeded with the seed itself. per_seed lists are indexed like `seeds`: "
    "the final cumulative regret against pi*'s expected rewards, the cumulative regret after "
    "every 25th step (0, 25, ..., 5000), the visitation of s_0..s_4999 and the first step at "
    "the right end (null: never)."
)


def run(inbox: Path = tp.INBOX) -> Path:
    git = tp._git_state()  # the repo as it is when the run starts
    ns = lab_namespace()
    start = time.perf_counter()
    per = run_regret(ns)
    secs = time.perf_counter() - start
    note = f"{NOTE} Wall time on {where()}: a laptop time, not a Colab time."
    return tp.write_record(experiment_record(per, secs, note, git), inbox)


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


def reproduce() -> bool:
    """Rerun the experiment in memory and compare with its newest filed record."""
    path, rec = newest_record()
    if rec["budget"] != json.loads(json.dumps(BUDGET)) or rec["seeds"] != SEEDS:
        print(f"{EXPERIMENT}: runs/{path.name} has a different budget or seeds; not comparable")
        return False
    new = json.loads(json.dumps(run_regret(lab_namespace())))  # the record's JSON types
    exact: list[str] = []
    floats: list[float] = []
    _diff(rec["per_seed"], new, "per_seed", exact, floats)
    worst = max(floats, default=0.0)
    good = not exact and worst <= 1e-9
    print(
        f"{EXPERIMENT}: {'REPRODUCED' if good else 'DIFFERS'} against runs/{path.name} "
        f"({len(floats)} floats, max |diff| {worst:.3g}; {len(exact)} exact mismatches)"
    )
    for line in exact[:10]:
        print(f"  {line}")
    return good


# ------------------------------------------------------------------ the record and exact data
def _order(path: Path) -> tuple[str, int]:
    m = re.search(r"-exp(?:-(\d+))?\.json$", path.name)
    return (path.name[:10], int(m.group(1) or 1) if m else 0)


def newest_record() -> tuple[Path, dict]:
    paths = sorted(RUNS.glob(f"*-{EXPERIMENT}-exp*.json"), key=_order)
    if not paths:
        raise SystemExit(
            f"No filed record of {EXPERIMENT} in runs/. Run this script with --run, then "
            "scripts/add_run_record.py."
        )
    return paths[-1], json.loads(paths[-1].read_text())


def expected_rewards(mdp, pi: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """The lab's expected_rewards: the exact expected reward at each step from the start, and
    the expected fraction of steps in each state."""
    P_pi, r_pi = _policy_chain(mdp.P, pi), (pi * mdp.R).sum(axis=1)
    d = np.zeros(mdp.n_states)
    d[mdp.start] = 1.0
    rewards, occupancy = np.empty(steps), np.zeros(mdp.n_states)
    for t in range(steps):
        rewards[t] = d @ r_pi
        occupancy += d
        d = d @ P_pi
    return rewards, occupancy / steps


def uniform(n: int) -> np.ndarray:
    return np.full((n, 2), 0.5)


def hitting_data() -> dict:
    ns = list(range(2, 17))
    h = {n: float(_hitting(_policy_chain(_riverswim(n).P, uniform(n)), n - 1)[0]) for n in ns}
    return {"ns": ns, "h": h}


def lines_data() -> dict:
    river = _riverswim(N_CHAIN)
    _, pi_star, Q_star, _ = _optimal(river.P, river.R, GAMMA, tol=1e-12)
    pi = river.policy_matrix(pi_star)
    opt, occ_star = expected_rewards(river, pi, STEPS)
    uni, occ_uni = expected_rewards(river, uniform(N_CHAIN), STEPS)
    # pi*'s long-run reward per step: the stationary distribution of its chain.
    P_pi = _policy_chain(river.P, pi)
    w, vec = np.linalg.eig(P_pi.T)
    st = np.real(vec[:, np.argmin(np.abs(w - 1))])
    st /= st.sum()
    return {
        "river": river,
        "pi_star": pi_star,
        "Q_star": Q_star,
        "opt": opt,
        "uni": uni,
        "occ_star": occ_star,
        "occ_uni": occ_uni,
        "regret_uni": np.cumsum(opt - uni),
        "gain": float(st @ (pi * river.R).sum(axis=1)),
    }


def regret_data() -> dict:
    path, rec = newest_record()
    per = rec["per_seed"]
    return {
        "record": path.name,
        "rec": rec,
        "x": np.arange(0, rec["budget"]["steps"] + 1, rec["budget"]["curve_every"]),
        "curves": {k: np.round(np.array(per[f"{k}_curve"]), 2) for k in LEARNERS},
        "final": {k: np.array(per[f"{k}_regret"]) for k in LEARNERS},
        "visitation": {k: np.array(per[f"{k}_visitation"]) for k in LEARNERS},
        "first": {k: per[f"{k}_first_visit"] for k in LEARNERS},
    }


# ------------------------------------------------------------------ drawing helpers
PX = 1 / 96


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def _count(v: float, _pos: int | None = None) -> str:
    return f"{v:,.0f}"


# ------------------------------------------------------------------ figure 1: the river
R_NODE = 0.27
LEFT_DASH = (0, (3, 1.5))  # the left action's arrows: dashed as well as another color


def _arrow(ax, a, b, color, *, rad=0.0, ls="solid", lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=6,head_width=3.5",
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def _loop(ax, x, y, color, *, up=True, ls="solid"):
    """A self-loop above (or below) the node at (x, y)."""
    sign = 1 if up else -1
    a = (x - 0.13, y + sign * (R_NODE - 0.03))
    b = (x + 0.13, y + sign * (R_NODE - 0.03))
    _arrow(ax, a, b, color, rad=-1.9 * sign if up else 1.9, ls=ls)


def river_figure(t: Theme, data: dict) -> Figure:
    river = data["river6"]
    n = river.n_states
    P = river.P
    fig = Figure(figsize=(WIDTH_IN, 440 * PX))  # 680 x 440 px, aspect 1.55
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(-0.75, n - 0.25)
    ax.set_ylim(-3.0, 1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    right_c, left_c = t.series[0], t.series[1]
    rows = {"right": 0.0, "left": -2.0}
    label = {"fontsize": MIN_FONT, "color": t.text, "ha": "center"}
    for y in rows.values():
        for s in range(n):
            ax.add_patch(
                Circle((s, y), R_NODE, facecolor=t.surface, edgecolor=t.border, lw=1.6, zorder=3)
            )
            ax.text(s, y, str(s), va="center", zorder=4, fontweight="bold", **label)
    # The right action: forward moves, stays (loops above) and slips (arcs below).
    y = rows["right"]
    ax.text(
        -0.72,
        1.05,
        "Right action",
        ha="left",
        va="center",
        color=t.text,
        fontsize=13,
        fontweight="bold",
    )
    for s in range(n):
        stay = P[s, 1, s]
        _loop(ax, s, y, right_c)
        words = f"{stay:g}" if s < n - 1 else f"{stay:g}, pays {river.r_right:g}"
        ax.text(s + (0.0 if s < n - 1 else -0.05), y + 0.68, words, va="bottom", **label)
        if s < n - 1:
            fwd = P[s, 1, s + 1]
            _arrow(ax, (s + R_NODE, y), (s + 1 - R_NODE, y), right_c)
            ax.text(s + 0.5, y + 0.07, f"{fwd:g}", va="bottom", **label)
        if s > 0:
            slip = P[s, 1, s - 1]
            _arrow(ax, (s - 0.12, y - R_NODE), (s - 1 + 0.12, y - R_NODE), right_c, rad=-0.55)
            ax.text(s - 0.5, y - 0.66, f"{slip:g}", va="top", **label)
    # The left action: one step left with probability 1; state 0 stays and pays.
    y = rows["left"]
    ax.text(
        -0.72,
        y + 0.62,
        "Left action",
        ha="left",
        va="center",
        color=t.text,
        fontsize=13,
        fontweight="bold",
    )
    for s in range(1, n):
        assert P[s, 0, s - 1] == 1.0
        _arrow(ax, (s - R_NODE, y), (s - 1 + R_NODE, y), left_c, ls=LEFT_DASH)
        ax.text(s - 0.5, y + 0.07, "1", va="bottom", **label)
    assert P[0, 0, 0] == 1.0
    _loop(ax, 0, y, left_c, up=False, ls=LEFT_DASH)
    ax.text(0.0, y - 0.68, f"1, pays {river.R[0, 0]:g}", va="top", **label)
    return fig


# ------------------------------------------------------------------ figure 2: hitting times
def hitting_figure(t: Theme, data: dict) -> Figure:
    ns, h = data["ns"], data["h"]
    fig = Figure(figsize=(WIDTH_IN, 400 * PX))  # 680 x 400 px, aspect 1.7
    ax = fig.add_axes((0.13, 0.14, 0.83, 0.82))
    ys = [float(f"{h[n]:.6g}") for n in ns]
    ax.axhline(STEPS, color=t.neutral, lw=1.6, ls=(0, (5, 2.5)), zorder=1)
    ax.text(16.2, STEPS * 0.62, f"budget, {STEPS} steps", ha="right", va="top", color=t.text)
    ax.plot(ns, ys, color=t.series[0], lw=2.0, marker="o", ms=7, mec=t.bg, mew=1.0, zorder=3)
    for n in (2, 6, 10):
        ax.annotate(
            f"{h[n]:,.0f}" if n == 2 else f"{h[n]:,.1f}",
            (n, h[n]),
            xytext=(9, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            color=t.text,
        )
    ax.set_yscale("log")
    ax.set_xlim(1.5, 16.5)
    ax.set_ylim(1, 1e8)
    ax.xaxis.set_major_locator(FixedLocator(list(range(2, 17, 2))))
    ax.xaxis.set_minor_locator(FixedLocator(list(range(3, 17, 2))))
    ax.yaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"$10^{{{int(round(math.log10(v)))}}}$")
    )
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_xlabel("Chain length $n$")
    ax.set_ylabel("Expected steps, left end to right end")
    return fig


# ------------------------------------------------------------------ figure 3: two reference lines
PANEL_VISIT_IN = 470 * PX  # the left subfigure (layout 62:38 of the 778 px column)
PANEL_REGRET_IN = 290 * PX
PANEL_H_IN = 300 * PX


def visitation_panel(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(PANEL_VISIT_IN, PANEL_H_IN))
    ax = fig.add_axes((0.15, 0.17, 0.83, 0.8))
    x = np.arange(N_CHAIN)
    uni = np.round(data["occ_uni"], 4)
    star = np.round(data["occ_star"], 4)
    ax.bar(x - 0.2, uni, width=0.4, color=t.series[0], lw=0, label="uniform random play")
    ax.bar(
        x + 0.2,
        star,
        width=0.4,
        facecolor=t.series[1],
        edgecolor=t.bg,
        hatch="//",
        lw=0,
        label=r"optimal policy $\pi^*$",
    )
    for xi, v, dx in ((0, uni[0], -0.2), (8, star[8], 0.2), (9, star[9], 0.2)):
        ax.text(xi + dx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", color=t.text)
    ax.set_xlim(-0.6, N_CHAIN - 0.4)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(FixedLocator(list(x)))
    ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_xlabel("State (0 = left end)")
    ax.set_ylabel("Fraction of 5000 steps")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=MIN_FONT)
    return fig


def regret_lines_panel(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(PANEL_REGRET_IN, PANEL_H_IN))
    ax = fig.add_axes((0.27, 0.17, 0.62, 0.8))
    steps = np.arange(0, STEPS + 1, 50)  # every 50th step: the expected curve is smooth
    reg = np.round(np.concatenate([[0.0], data["regret_uni"]])[steps], 2)
    ax.plot(steps, reg, color=t.series[0], lw=2.2)
    ax.plot([0, STEPS], [0, 0], color=t.series[1], lw=2.6, ls=(0, (3.2, 1.6)))
    ax.text(150, 4400, f"uniform\nrandom play:\n{reg[-1]:.1f}", ha="left", va="top", color=t.text)
    ax.text(STEPS, 150, r"$\pi^*$: 0", ha="right", va="bottom", color=t.text)
    ax.set_xlim(0, STEPS)
    ax.set_ylim(-60, 4500)
    ax.xaxis.set_major_locator(FixedLocator([0, 2500, 5000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([0, 1000, 2000, 3000, 4000]))
    ax.yaxis.set_major_formatter(FuncFormatter(_count))
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Expected regret")
    return fig


# ------------------------------------------------------------------ figure 4: the record
STYLES = {"eps-greedy": "solid", "optimistic": (0, (5, 2)), "MBIE-EB": (0, (1.2, 1.6))}
NAMES = {"eps-greedy": "ε-greedy Q-learning", "optimistic": "optimistic Q-learning"}
NAMES["MBIE-EB"] = "MBIE-EB"


def regret_seeds_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 420 * PX))  # 680 x 420 px, aspect 1.62
    ax = fig.add_axes((0.12, 0.13, 0.83, 0.83))
    x = data["x"]
    handles, labels = [], []
    for j, name in enumerate(DRAWN):
        for c in data["curves"][name]:
            ax.plot(x, c, color=t.series[j], lw=1.3, ls=STYLES[name], alpha=0.9)
        handles.append(Line2D([], [], color=t.series[j], lw=2.2, ls=STYLES[name]))
        labels.append(f"{NAMES[name]} ({len(data['curves'][name])} seeds)")
    ax.set_xlim(0, STEPS)
    ax.set_ylim(0, 4500)
    ax.xaxis.set_major_locator(FixedLocator([0, 1000, 2000, 3000, 4000, 5000]))
    ax.xaxis.set_major_formatter(FuncFormatter(_count))
    ax.yaxis.set_major_locator(FixedLocator([0, 1000, 2000, 3000, 4000]))
    ax.yaxis.set_major_formatter(FuncFormatter(_count))
    ax.grid(True, axis="y", lw=0.8)
    _style_axes(ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative regret")
    ax.legend(handles, labels, loc="upper left", frameon=False, handlelength=3.0)
    return fig


# ------------------------------------------------------------------ numbers the page quotes
def _r(x: float, d: int) -> float:
    return round(float(x), d)


def _round_range(values, d: int) -> tuple[float, float]:
    return _r(min(values), d), _r(max(values), d)


#: The colleague's agent's fraction of steps at state 0 on the record's seeds (min, max).
SUSPECT_STATE0 = (0.727, 0.745)


def exact_numbers(computed: dict) -> dict:
    """Section 2-5 numbers (the director's m06_page_numbers.py, folded in), all exact."""
    out: dict = {}
    river6 = _riverswim(6)
    P6 = _policy_chain(river6.P, uniform(6))
    out.update(p_right=P6[2, 3], p_stay=P6[2, 2], p_left=P6[2, 1])
    out.update(left_end_right=P6[0, 1], left_end_stay=P6[0, 0])
    out.update(right_end_stay=P6[5, 5], right_end_left=P6[5, 4])
    u = [5.0]
    for _ in range(1, 15):
        u.append((1 + 0.525 * u[-1]) / 0.175)
    out["u"] = u
    closed = [55 / 7 * 3**s - 20 / 7 for s in range(15)]  # u_s = (55/7) 3^s - 20/7
    out["closed_form"] = bool(np.allclose(u, closed, rtol=1e-12, atol=1e-12))
    out["h"] = computed["06-hitting-time"]["h"]
    lines = computed["06-lines"]
    Q_star = lines["Q_star"]
    out["max_Q_star"] = float(Q_star.max())
    out["pi_star_right"] = {
        n: bool(np.all(_optimal(_riverswim(n).P, _riverswim(n).R, GAMMA, 1e-12)[1] == 1))
        for n in range(6, 17)
    }
    beta = (1 / (1 - GAMMA)) * np.sqrt(np.log(2 * 10 * 2 * 5000 / 0.05) / 2)
    out["beta"] = float(beta)
    out["max_plans"] = 1 + 20 * len([k for k in range(13) if 2**k <= STEPS])
    return out


def quoted_numbers(computed: dict) -> dict:
    """Every number the page, its captions and alt text, the lab text and the m06
    predict_answers quote, and every claim about the recorded learners."""
    ex = exact_numbers(computed)
    # Section 1 and figure 1 (the alt text), from the env's arrays.
    river6 = _riverswim(6)
    P = river6.P
    p = np.round(P, 12)  # 0.35 + 0.05 is 0.39999999999999997 in floating point
    assert [p[s, 1, s + 1] for s in (1, 2, 3, 4)] == [0.35] * 4
    assert [p[s, 1, s] for s in (1, 2, 3, 4)] == [0.6] * 4
    assert [p[s, 1, s - 1] for s in (1, 2, 3, 4)] == [0.05] * 4
    assert (p[0, 1, 1], p[0, 1, 0], p[5, 1, 5], p[5, 1, 4]) == (0.4, 0.6, 0.95, 0.05)
    assert river6.r_right == 1.0 and river6.R[0, 0] == 0.005
    assert all(P[s, 0, max(s - 1, 0)] == 1.0 for s in range(6))
    # Section 2 and Exercise 1's predict_answer: uniform play.
    assert (_r(ex["p_right"], 12), _r(ex["p_stay"], 12), _r(ex["p_left"], 12)) == (
        0.175,
        0.3,
        0.525,
    )
    assert (_r(ex["left_end_right"], 12), _r(ex["left_end_stay"], 12)) == (0.2, 0.8)
    assert (_r(ex["right_end_stay"], 12), _r(ex["right_end_left"], 12)) == (0.475, 0.525)
    assert _r(0.525 / 0.175, 9) == 3 and _r(1.05 / 0.35, 9) == 3 and _r(1 / 0.175, 1) == 5.7
    assert [_r(x, 2) for x in ex["u"][:5]] == [5.0, 20.71, 67.86, 209.29, 633.57]
    assert ex["closed_form"]
    h = ex["h"]
    assert _r(sum(ex["u"][:3]), 2) == _r(h[4], 2) == 93.57
    assert _r(h[6], 1) == 936.4 and _r(h[10], 1) == 77296.4 and _r(h[12], 1) == 695899.3
    assert _r(h[2], 9) == 5.0 and _r(h[7], 1) == 2842.9 and _r(h[8], 1) == 8567.9
    assert h[7] < STEPS < h[8] and _r(h[10] / STEPS, 2) == 15.46  # "about 15 times"
    assert round(936 * 3**4, -2) == 75800  # Exercise 2's predict_answer
    # "rising by a factor close to 3 per state once the chain has more than a few states"
    ratios = {n: h[n + 1] / h[n] for n in range(2, 16)}
    assert all(abs(ratios[n] - 3) < 0.1 for n in range(5, 16)), ratios
    assert ratios[2] > 5 and min(ratios.values()) > 2.99
    # Strehl and Littman's chain; eps-greedy once it prefers left (section 2's fold).
    assert _r(0.5 * 0.3, 12) == 0.15 and _r(0.5 + 0.5 * 0.1, 12) == 0.55
    assert _r(0.55 / 0.15, 2) == 3.67 and _r(0.55 / 0.15 - 11 / 3, 12) == 0
    assert _r(0.05 * 0.4, 12) == 0.02 and _r(0.05 * 0.35, 12) == 0.0175
    assert _r(0.95 + 0.05 * 0.05, 12) == 0.9525 and _r(0.9525 / 0.0175, 2) == 54.43
    # Section 3: V_max and Q*.
    assert _r(1 / (1 - GAMMA), 9) == 20 and _r(ex["max_Q_star"], 2) == 16.68
    assert all(ex["pi_star_right"].values())
    # Section 4: Lemma 7's beta, the plans, Check yourself 3.1 and 4.1.
    assert round(ex["beta"]) == 55 and _r(ex["beta"], 1) == 55.1
    assert ex["max_plans"] == 261
    assert _r(2 / (1 - 0.9), 9) == 20 and _r(20 + 0.5 * (0.9 * 20 - 20), 9) == 19
    assert _r(1 / (1 - GAMMA), 9) == 20 and _r(0.505 + GAMMA * 20, 9) == 19.505
    assert _r(0.005 / (1 - GAMMA), 9) == 0.1 and _r(GAMMA * 0.1, 9) == 0.095
    assert _r(0.1 + 0.5, 9) == 0.6 and _r(0.095 + 1, 9) == 1.095
    # Knowledge checks 6.1, 6.3, 6.4 (6.5's visitation is the checkpoint's own case).
    kc_u = [4.0]
    for _ in range(2):
        kc_u.append((1 + 0.4 * kc_u[-1]) / 0.2)
    assert kc_u == [4.0, 13.0, 31.0] and sum(kc_u) == 48 and _r(0.4 / 0.2, 9) == 2
    assert _r(5 / (1 - 0.8), 9) == 25 and _r(25 + 0.2 * (1 + 0.8 * 25 - 25), 9) == 24.2
    assert _r(0.6 / (1 - 0.9), 9) == 6 and _r(0.2 + 0.6 / 3 + 0.9 * 6, 9) == 5.8
    # The value-iteration tolerance Gotcha.
    assert _r(GAMMA * 0.01 / (1 - GAMMA), 9) == 0.19
    # Section 5 and figure 3 (caption, alt text, demo), exact expectations.
    li = computed["06-lines"]
    occ_u, occ_s = li["occ_uni"], li["occ_star"]
    assert _r(occ_u[0], 4) == 0.6366 and _r(occ_u[0], 2) == 0.64  # "64%", "0.64"
    assert _r(occ_u[9] * 1e5, 1) == 3.7  # "3.7 x 10^-5"
    assert _r(occ_s[8], 2) == 0.12 and _r(occ_s[9], 2) == 0.85 and _r(occ_s[9], 3) == 0.852
    assert occ_s[:8].max() < 0.02  # "near 0 up to state 7" (0.018 at state 7)
    assert occ_u[6:].max() < 0.001  # "almost nothing beyond state 5"
    assert _r(li["opt"].sum(), 1) == 4048.2 and _r(li["regret_uni"][-1], 1) == 4040.2
    assert _r(li["opt"][-1] - li["uni"][-1], 4) == 0.8127
    assert _r(li["gain"], 5) == 0.81429 and _r(li["gain"], 2) == 0.81
    # The lab text: optimal and uniform visitation at the right end (the setup cell prints them).
    # ---------------------------------------------------------------- the record (figure 4)
    rg = computed["06-regret-seeds"]
    rec, first, final, visit = rg["rec"], rg["first"], rg["final"], rg["visitation"]
    assert rec["budget"] == BUDGET and rec["seeds"] == SEEDS
    curves = {k: np.array(rec["per_seed"][f"{k}_curve"]) for k in LEARNERS}
    assert all(np.allclose(curves[k][:, -1], final[k], atol=1e-6) for k in LEARNERS)
    n = len(SEEDS)
    # eps-greedy (Exercise 3's predict_answer, the caption): never reaches the right end, more
    # concentrated at the left end than uniform play, a straight line nearly as steep as
    # uniform play's.
    assert all(f is None for f in first["eps-greedy"])
    assert visit["eps-greedy"][:, 0].min() > occ_u[0]
    eps_final = final["eps-greedy"]
    assert eps_final.min() > 0.99 * li["regret_uni"][-1] - 40  # "nearly as steep"
    half = len(rg["x"]) // 2
    for c in curves["eps-greedy"]:
        first_half, second_half = c[half] - c[0], c[-1] - c[half]
        assert abs(second_half / first_half - 1) < 0.02, (first_half, second_half)
    # optimistic (Exercise 4's predict_answer, KC 6.2): reaches the right end on every seed,
    # far sooner than uniform play's 77,296 steps; the curves bend but keep climbing.
    assert all(f is not None for f in first["optimistic"])
    assert max(first["optimistic"]) < STEPS < h[10]
    last = slice(-int(1000 / EVERY) - 1, None)  # the last 1000 steps
    slope = {k: (curves[k][:, -1] - curves[k][:, last.start]) / 1000 for k in LEARNERS}
    early = {k: curves[k][:, int(1000 / EVERY)] / 1000 for k in LEARNERS}  # the first 1000
    assert np.all(slope["optimistic"] < early["optimistic"])  # bends
    assert np.all(slope["optimistic"] > 0.1)  # keeps climbing
    # MBIE-EB (Exercise 5's predict_answer, KC 6.2, the caption): over three quarters of the
    # steps at the right end on every seed; bends early and stays nearly flat.
    assert all(f is not None for f in first["MBIE-EB"])
    assert visit["MBIE-EB"][:, 9].min() > 0.75
    assert np.all(slope["MBIE-EB"] < 0.1)  # an eighth of eps-greedy's 0.81 per step, or less
    assert np.all(slope["MBIE-EB"] < slope["optimistic"])
    # Exercise 6's predict_answer: MBIE-EB lowest, then optimistic, then eps-greedy, no ties.
    for a, b in (
        ("MBIE-EB", "optimistic"),
        ("MBIE-EB", "eps-greedy"),
        ("optimistic", "eps-greedy"),
    ):
        v = compare(final[a], final[b], higher_is_better=False)
        assert v.winner == "A", (a, b, v.sentence)
    # The quoted ranges (the page's section 2 and section 5 folds, the predict_answers of
    # Exercises 3 to 6, knowledge check 6.2) and the room code's reach counts on these seeds.
    assert _round_range(visit["eps-greedy"][:, 0], 3) == (0.972, 0.983)
    reach = {k: sum(f is not None for f in first[k]) for k in LEARNERS}
    assert (reach["eps-greedy"], reach["optimistic"], reach["MBIE-EB"]) == (0, n, n)
    assert (min(first["optimistic"]), max(first["optimistic"])) == (399, 1215)
    assert (min(first["MBIE-EB"]), max(first["MBIE-EB"])) == (82, 269)
    assert _round_range(visit["MBIE-EB"][:, 9], 3) == (0.805, 0.85)
    cis = {k: tuple(_r(x, 1) for x in (iqm(final[k]), *bootstrap_ci(final[k], iqm))) for k in DRAWN}
    assert cis == {
        "MBIE-EB": (133.2, 81.9, 171.9),
        "optimistic": (2611.9, 2513.6, 2672.9),
        "eps-greedy": (4025.0, 4024.9, 4025.1),
    }, cis
    # Knowledge check 6.6 (hypothetical rewards, exact arithmetic): Reg_1..Reg_4.
    kc66 = np.cumsum(np.array([0, 0, 0.1, 0.3]) - np.array([0.005, 0, 0, 0.95]))
    assert [_r(x, 9) for x in kc66] == [-0.005, -0.005, 0.095, -0.555]
    # Exercise 7's predict_answer: the colleague's agent (the ex7-stub cell) never reaches the
    # right end and spends more than half of its steps at state 0, on every seed.
    assert all(f is None for f in first["suspect"]) and visit["suspect"][:, 9].max() == 0.0
    assert _round_range(visit["suspect"][:, 0], 3) == SUSPECT_STATE0, _round_range(
        visit["suspect"][:, 0], 3
    )
    return {
        "ex": ex,
        "slope": slope,
        "early": early,
        "iqm": {k: (iqm(final[k]), *bootstrap_ci(final[k], iqm)) for k in LEARNERS},
        "n": n,
    }


# ------------------------------------------------------------------ build, report, check
def river_data() -> dict:
    return {"river6": _riverswim(6)}


DATA = {
    "06-riverswim": river_data,
    "06-hitting-time": hitting_data,
    "06-lines": lines_data,
    "06-regret-seeds": regret_data,
}
# name: (its data, how to draw it, a subfigure panel?)
FIGURES = {
    "06-riverswim": ("06-riverswim", river_figure, False),
    "06-hitting-time": ("06-hitting-time", hitting_figure, False),
    "06-lines-visitation": ("06-lines", visitation_panel, True),
    "06-lines-regret": ("06-lines", regret_lines_panel, True),
    "06-regret-seeds": ("06-regret-seeds", regret_seeds_figure, False),
}


def panel_svg_bytes(fig: Figure) -> bytes:
    """Deterministic SVG that declares itself a subfigure panel (tests/test_site_figures.py
    then allows it to be narrower than the column)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Description": "subfigure panel"})
    return buf.getvalue()


def build() -> tuple[dict[Path, bytes], dict]:
    computed = {name: make() for name, make in DATA.items()}
    computed["quoted"] = quoted_numbers(computed)
    out: dict[Path, bytes] = {}
    for name, (key, draw, panel) in FIGURES.items():
        for t in themes():
            with mpl.rc_context(rc(name, t)):
                fig = draw(t, computed[key])
                out[IMAGES / f"{name}{t.suffix}.svg"] = (panel_svg_bytes if panel else svg_bytes)(
                    fig
                )
    return out, computed


def report(computed: dict) -> None:
    rg = computed["06-regret-seeds"]
    q = computed["quoted"]
    print(f"{EXPERIMENT} from runs/{rg['record']} (budget {rg['rec']['budget']})")
    for k in LEARNERS:
        c, lo, hi = q["iqm"][k]
        fin = rg["final"][k]
        print(
            f"  {k:10s} final regret by seed {[round(float(x), 1) for x in fin]}; IQM {c:.1f} "
            f"[{lo:.1f}, {hi:.1f}]; min {fin.min():.1f}, max {fin.max():.1f}"
        )
        print(f"  {k:10s} first visit to the right end by seed {rg['first'][k]}")
        v = rg["visitation"][k]
        print(
            f"  {k:10s} visitation at state 0 {v[:, 0].round(3).tolist()}; at state 9 "
            f"{v[:, 9].round(3).tolist()}"
        )
        print(
            f"  {k:10s} regret per step, first 1000 steps {q['early'][k].round(3).tolist()}; "
            f"last 1000 {q['slope'][k].round(3).tolist()}"
        )
    for a, b in (
        ("MBIE-EB", "optimistic"),
        ("MBIE-EB", "eps-greedy"),
        ("optimistic", "eps-greedy"),
    ):
        v = compare(rg["final"][a], rg["final"][b], names=(a, b), higher_is_better=False)
        print(f"  {v.sentence}")


def main(argv: list[str]) -> int:
    if "--run" in argv:
        p = run()
        print(f"Saved {p.name} in runs/inbox/ (file it with scripts/add_run_record.py).")
        return 0
    if "--reproduce" in argv:
        return 0 if reproduce() else 1
    files, computed = build()
    if "--check" in argv:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m06.py")
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
