"""Module 3 figures, drawn from recorded experiments: light and dark SVGs (PLAN.md §8).

  uv run python scripts/make_figures_m03.py --run [--only NAME]  # run the experiments; records go to runs/inbox/
  uv run python scripts/add_run_record.py             # file them in runs/
  uv run python scripts/make_figures_m03.py           # draw images/03-*.svg from the filed records
  uv run python scripts/make_figures_m03.py --check   # fail if a committed SVG differs from a fresh drawing
  uv run python scripts/make_figures_m03.py --reproduce [--only NAME]  # rerun in memory, compare
                                                      # with the filed records, write nothing

The experiments use the lab's own reference code: the worked notebook (solutions bound,
live budget) runs in-process up to the stretch's run cell, as scripts/threshold_protocol.py
does, and its `sarsa`, `q_learning`, `n_step_td_prediction`, `collect` and `rms_on_route`
do the work. Nothing here reimplements a learner.

  03-cliff-contrast  SARSA and Q-learning on the 4x12 cliff (gamma 0.95), 1000 episodes,
                     alpha 0.1, epsilon 0.1, seeds 0-9. Left: both greedy paths of seed 0.
                     Right: online return per episode, moving average of 20, mean over the
                     seeds with the min-max band.
  03-nstep-dial      Offline n-step TD on 1000 episodes of the epsilon-soft safe route:
                     RMS error on the route against n, one line per alpha, the IQM over
                     seeds 0-9 with a 95% bootstrap interval.

One more experiment has no figure: m03-suspect-returns, the planted-bug suspect (the
notebook's ex6-stub cell) against the reference q_learning from Q0 = 10, both trained by the
lab's own value_curve (Exercise 6's left panel). `report` prints its numbers.

`--reproduce` reruns the experiments with the current notebook and compares every per-seed
value with the newest filed record: discrete outcomes (paths, path lengths, whole-number
returns) must match exactly, floats to 1e-9. It checks a new platform or a changed notebook
without filing anything.

The figures read only the newest filed record of each experiment (runs/*.json), so
`--check` never reruns anything. Drawing is deterministic: svg.hashsalt is the figure
name, no date in the metadata, text as paths, values rounded before plotting. The palette
comes from prl._palette; background, text and lines from the brand tokens.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import trim_mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_protocol as tp  # noqa: E402
from make_figures_m01 import MIN_FONT, WIDTH_IN, Theme, rc, svg_bytes, themes  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.lab import _default_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
RUNS = ROOT / "runs"
NOTEBOOK = ROOT / "notebooks" / "03-model-free.ipynb"
SCRIPT = "scripts/make_figures_m03.py"

SEEDS = list(range(10))
CONTRAST = {"episodes": 1000, "alpha": 0.1, "epsilon": 0.1, "gamma": 0.95, "window": 500}
DIAL = {"episodes": 1000, "ns": [1, 2, 4, 8, 16, 32], "alphas": [0.02, 0.05, 0.1, 0.2]}
DIAL.update(epsilon=0.1, gamma=0.95)
SUSPECT = {"episodes": 1000, "alpha": 0.1, "epsilon": 0.1, "gamma": 0.95, "q0": 10.0}
SUSPECT.update(chunk=20, first=100, last=500)
SMOOTH = 20
EXPERIMENTS = {"03-cliff-contrast": "m03-cliff-contrast", "03-nstep-dial": "m03-nstep-dial"}
SUSPECT_EXPERIMENT = "m03-suspect-returns"


# ------------------------------------------------------------------ running the experiments
def lab_namespace() -> dict:
    """The worked notebook's namespace (live budget, solutions bound) before the stretch runs."""
    tp.ensure_current(NOTEBOOK)
    with tp._environment(PRL_QUICK="0", PRL_WORKED="1"):
        ns, _ = tp.solution_namespace(NOTEBOOK, "stretch")
    return ns


def _fresh_env(ns: dict):
    return ns["gym"].make(
        "prl/CliffGridworld-v0", rows=ns["ROWS"], cols=ns["COLS"], gamma=ns["GAMMA"]
    )


def run_contrast(ns: dict) -> dict:
    b = CONTRAST
    per = {k: [] for k in ("sarsa_online", "q_online", "sarsa_path", "q_path")}
    per.update(sarsa_reached=[], q_reached=[])
    for seed in SEEDS:
        for key, fn in (("sarsa", ns["sarsa"]), ("q", ns["q_learning"])):
            with contextlib.redirect_stdout(io.StringIO()):
                Q, online = fn(
                    _fresh_env(ns), b["episodes"], b["alpha"], b["epsilon"], b["gamma"], seed
                )
            ints = [int(round(x)) for x in online]
            assert np.allclose(ints, online), "the cliff's rewards are whole numbers"
            path, reached = ns["greedy_path"](Q)
            per[f"{key}_online"].append(ints)
            per[f"{key}_path"].append([int(s) for s in path])
            per[f"{key}_reached"].append(bool(reached))
    return per


def run_dial(ns: dict) -> dict:
    per: dict[str, list] = {f"alpha {a:g}": [] for a in DIAL["alphas"]}
    for seed in SEEDS:
        with contextlib.redirect_stdout(io.StringIO()):
            episodes = ns["collect"](ns["env"], ns["PI"], DIAL["episodes"], seed)
        for a in DIAL["alphas"]:
            row = [
                round(
                    ns["rms_on_route"](
                        ns["n_step_td_prediction"](episodes, a, DIAL["gamma"], ns["S"], n)
                    ),
                    6,
                )
                for n in DIAL["ns"]
            ]
            per[f"alpha {a:g}"].append(row)
    return per


def buggy_suspect(ns: dict):
    """The colleague's q_learning_suspect as the notebook ships it (its ex6-stub cell).

    The worked namespace binds the fixed reference to that name, so the stub cell is run
    again in a copy of the namespace.
    """
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    (src,) = [c.source for c in nb.cells if c.get("id") == "ex6-stub"]
    local = dict(ns)
    exec(compile(src, "<ex6-stub>", "exec"), local)  # noqa: S102
    return local["q_learning_suspect"]


def run_suspect(ns: dict) -> dict:
    """Online returns and final values of the suspect and of the reference q_learning."""
    b = SUSPECT
    assert ns["budget"]["control_alpha"] == b["alpha"] and ns["EPSILON"] == b["epsilon"]
    assert max(10, b["episodes"] // 50) == b["chunk"], "the lab's chunk size"
    learners = {"suspect": buggy_suspect(ns), "fixed": ns["q_learning"]}
    per: dict[str, list] = {}
    for seed in SEEDS:
        for key, fn in learners.items():
            with contextlib.redirect_stdout(io.StringIO()):
                _, values, online, Q = ns["value_curve"](
                    fn, b["episodes"], b["chunk"], seed, b["q0"]
                )
            ints = [int(round(x)) for x in online]
            assert np.allclose(ints, online), "the cliff's rewards are whole numbers"
            path, reached = ns["greedy_path"](Q)
            per.setdefault(f"{key}_online", []).append(ints)
            per.setdefault(f"{key}_max_q_start", []).append([round(float(v), 9) for v in values])
            per.setdefault(f"{key}_q_last_move", []).append(round(float(Q[ns["ABOVE_GOAL"], 2]), 9))
            per.setdefault(f"{key}_path_moves", []).append(len(path) - 1 if reached else None)
    return per


def experiment_record(
    name: str,
    budget: dict,
    per_seed: dict,
    seconds: float,
    note: str,
    git: tuple[str | None, bool | None],
) -> dict:
    rt = runtime.detect()
    git_sha, dirty = git
    import nbformat

    meta = nbformat.read(NOTEBOOK, as_version=4).metadata.get("prl", {})
    return {
        "schema": record.SCHEMA,
        "kind": "experiment",
        "experiment": name,
        "script": SCRIPT,
        "args": {
            "notebook": "notebooks/03-model-free.ipynb",
            "notebook_content_sha": meta.get("content_sha"),
            "code": "the worked notebook's reference solutions, run in-process (live budget)",
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
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "seconds": round(seconds, 1),
        "note": note,
    }


def where() -> str:
    rt = runtime.detect()
    return f"{_default_env(rt, 'colab-cpu')} ({rt.cpu}, {__import__('platform').machine()})"


def _notes() -> dict[str, str]:
    return {
        "m03-cliff-contrast": (
            f"SARSA and Q-learning from the worked notebook, {len(SEEDS)} seeds, "
            f"{CONTRAST['episodes']} episodes each. per_seed lists are indexed like `seeds`."
        ),
        "m03-nstep-dial": (
            f"Offline n-step TD (the stretch's reference) on {DIAL['episodes']} episodes of the "
            "epsilon-soft safe route per seed; per_seed['alpha a'][i][j] is the RMS error on "
            "the route for seeds[i] and ns[j]."
        ),
        SUSPECT_EXPERIMENT: (
            "The planted-bug suspect (the notebook's ex6-stub cell) and the reference "
            f"q_learning, each trained by the lab's value_curve from Q0 = {SUSPECT['q0']:g}: "
            f"{SUSPECT['episodes'] // SUSPECT['chunk']} chunks of {SUSPECT['chunk']} episodes, "
            "chunk i seeded seed * 1000 + i. per_seed: every episode's online return, max Q(start) "
            "after each chunk, the final Q(above goal, down) and the greedy path's moves."
        ),
    }


RUNNERS = {
    "m03-cliff-contrast": (run_contrast, CONTRAST),
    "m03-nstep-dial": (run_dial, DIAL),
    SUSPECT_EXPERIMENT: (run_suspect, SUSPECT),
}


def run(inbox: Path = tp.INBOX, only: list[str] | None = None) -> list[Path]:
    git = tp._git_state()  # the repo as it is when the run starts
    ns = lab_namespace()
    out = []
    for name, (runner, budget) in RUNNERS.items():
        if only and name not in only:
            continue
        start = time.perf_counter()
        per = runner(ns)
        secs = time.perf_counter() - start
        note = f"{_notes()[name]} Wall time on {where()}: a laptop time, not a Colab time."
        rec = experiment_record(name, dict(budget), per, secs, note, git)
        out.append(tp.write_record(rec, inbox))
    return out


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


def reproduce(only: list[str] | None = None) -> bool:
    """Rerun each experiment in memory and compare with its newest filed record."""
    ns = lab_namespace()
    ok = True
    for name, (runner, budget) in RUNNERS.items():
        if only and name not in only:
            continue
        path, rec = newest_record(name)
        if rec["budget"] != json.loads(json.dumps(budget)) or rec["seeds"] != SEEDS:
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


def smooth(x: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=float), np.ones(k) / k, mode="valid")


def contrast_data() -> dict:
    path, rec = newest_record(EXPERIMENTS["03-cliff-contrast"])
    per, b = rec["per_seed"], rec["budget"]
    curves = {}
    for key in ("sarsa", "q"):
        sm = np.array([smooth(o) for o in per[f"{key}_online"]])
        curves[key] = {
            "mean": np.round(sm.mean(axis=0), 3),
            "lo": np.round(sm.min(axis=0), 3),
            "hi": np.round(sm.max(axis=0), 3),
            "last": [float(np.mean(o[-b["window"] :])) for o in per[f"{key}_online"]],
        }
    i0 = rec["seeds"].index(0)
    return {
        "record": path.name,
        "rec": rec,
        "x": np.arange(SMOOTH, b["episodes"] + 1),
        "curves": curves,
        "paths": {k: (per[f"{k}_path"][i0], per[f"{k}_reached"][i0]) for k in ("sarsa", "q")},
    }


BOOT = 2000  # bootstrap resamples of the seeds, from a fixed generator (deterministic)


def iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Interquartile mean: the mean of the middle half (scipy's trim_mean at 25%)."""
    return trim_mean(x, 0.25, axis=axis)


def iqm_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """IQM over seeds and its 95% percentile bootstrap interval (seeds resampled)."""
    idx = rng.integers(len(values), size=(BOOT, len(values)))
    boots = iqm(values[idx], axis=1)
    return float(iqm(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def dial_data() -> dict:
    path, rec = newest_record(EXPERIMENTS["03-nstep-dial"])
    b = rec["budget"]
    rng = np.random.default_rng(0)
    lines = {}
    for a in b["alphas"]:
        arr = np.array(rec["per_seed"][f"alpha {a:g}"])
        stats = np.array([iqm_ci(arr[:, j], rng) for j in range(arr.shape[1])])
        lines[a] = {
            "iqm": np.round(stats[:, 0], 4),
            "lo": np.round(stats[:, 1], 4),
            "hi": np.round(stats[:, 2], 4),
            "mean": np.round(arr.mean(axis=0), 4),
            "best_n": [b["ns"][int(j)] for j in arr.argmin(axis=1)],
        }
    return {"record": path.name, "rec": rec, "ns": b["ns"], "lines": lines}


# ------------------------------------------------------------------ figure 1: the cliff contrast
ROWS, COLS = 4, 12
STYLES = {"sarsa": "solid", "q": (0, (3.2, 1.6))}
NAMES = {"sarsa": "SARSA", "q": "Q-learning"}


def _cells(path: list[int]) -> list[tuple[float, float]]:
    """(x, y) of cell centers, y up, keeping only the corners of the path."""
    pts = [(s % COLS + 0.5, ROWS - 1 - s // COLS + 0.5) for s in path]
    keep = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:], strict=False):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            keep.append(b)
    keep.append(pts[-1])
    return keep


def _grid(ax, t: Theme) -> None:
    for r in range(ROWS):
        for c in range(COLS):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=t.surface, edgecolor=t.grid, lw=1.0))
    ax.add_patch(
        Rectangle(
            (1, 0), COLS - 2, 1, facecolor=t.cliff_bg, edgecolor=t.cliff_fg, lw=1.5, hatch="//"
        )
    )
    ax.text(
        COLS / 2,
        0.5,
        "Cliff: \N{MINUS SIGN}100",
        ha="center",
        va="center",
        fontsize=MIN_FONT,
        fontweight="bold",
        color=t.cliff_fg,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": t.cliff_bg, "edgecolor": "none"},
    )
    for c, letter in ((0, "S"), (COLS - 1, "G")):
        ax.add_patch(Rectangle((c, 0), 1, 1, facecolor=t.surface, edgecolor=t.border, lw=2.0))
        ax.text(
            c + 0.5,
            0.32,
            letter,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=t.text,
        )


def contrast_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 3.6))  # 680 x 346 px, aspect 1.97
    colors = {"sarsa": t.series[0], "q": t.series[1]}
    # Left: the grid and the two greedy paths of seed 0.
    ax = fig.add_axes((0.01, 0.34, 0.52, 0.56))
    ax.set_xlim(-0.05, COLS + 0.05)
    ax.set_ylim(-0.05, ROWS + 0.05)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _grid(ax, t)
    offsets = {"sarsa": (-0.13, 0.13), "q": (0.13, -0.13)}
    for key in ("sarsa", "q"):
        path, _ = data["paths"][key]
        pts = _cells(path)
        dx, dy = offsets[key]
        xs = [x + dx for x, _ in pts]
        ys = [y + dy for _, y in pts]
        ys[0], ys[-1] = 0.72, 0.62  # leave S and G readable
        ax.plot(
            xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1] + 0.14], color=colors[key], lw=2.6, ls=STYLES[key]
        )
        ax.annotate(
            "",
            xy=(xs[-1], ys[-1]),
            xytext=(xs[-1], ys[-1] + 0.3),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.45,head_width=0.25",
                "color": colors[key],
                "lw": 2.6,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    ax.set_title("Greedy paths after learning, seed 0", fontsize=13, color=t.text, pad=6)
    # Right: the online return while learning.
    ax2 = fig.add_axes((0.655, 0.155, 0.305, 0.715))
    x = data["x"]
    for key in ("q", "sarsa"):
        cur = data["curves"][key]
        ax2.fill_between(x, cur["lo"], cur["hi"], color=colors[key], alpha=0.18, lw=0)
        ax2.plot(x, cur["mean"], color=colors[key], lw=2.0, ls=STYLES[key])
    ax2.set_xlim(0, x[-1])
    ax2.set_ylim(-120, 0)
    ax2.xaxis.set_major_locator(FixedLocator([0, 500, 1000]))
    ax2.yaxis.set_major_locator(FixedLocator([-120, -80, -40, 0]))
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v:g}".replace("-", "\N{MINUS SIGN}"))
    )
    ax2.grid(True, axis="y", lw=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.set_xlabel("Episode")
    ax2.set_title("Online return per episode", fontsize=13, color=t.text, pad=6)
    n = len(data["rec"]["seeds"])
    handles, labels = [], []
    for key in ("sarsa", "q"):
        path, reached = data["paths"][key]
        moves = f"{len(path) - 1} moves" if reached else "does not reach G"
        handles.append(Line2D([], [], color=colors[key], lw=2.6, ls=STYLES[key]))
        labels.append(f"{NAMES[key]} (greedy path: {moves})")
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        frameon=False,
        handlelength=3.0,
        fontsize=MIN_FONT,
        title=(
            f"Right: mean of {n} seeds, band min\N{EN DASH}max;\n"
            f"each a moving average of {SMOOTH} episodes"
        ),
        title_fontsize=MIN_FONT,
        alignment="left",
    )
    return fig


# ------------------------------------------------------------------ figure 2: the n-step dial
MARKERS = ("o", "s", "^", "D")


def dial_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.4))  # 680 x 422 px, aspect 1.61
    ax = fig.add_axes((0.105, 0.135, 0.585, 0.81))
    ns = data["ns"]
    handles, labels = [], []
    for i, (a, line) in enumerate(data["lines"].items()):
        color = t.series[i]
        ax.fill_between(ns, line["lo"], line["hi"], color=color, alpha=0.15, lw=0)
        (h,) = ax.plot(
            ns, line["iqm"], color=color, lw=2.0, marker=MARKERS[i], ms=7, mec=t.bg, mew=1.0
        )
        handles.append(h)
        labels.append(f"\N{GREEK SMALL LETTER ALPHA} = {a:g}")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("n (steps before the bootstrap; log scale)")
    ax.set_ylabel("RMS error on the route")
    n = len(data["rec"]["seeds"])
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.715, 0.95),
        frameon=False,
        title=f"Step size\nIQM of {n} seeds;\nband: 95% bootstrap\ninterval of the IQM",
        title_fontsize=MIN_FONT,
        alignment="left",
        handlelength=2.4,
    )
    return fig


# ------------------------------------------------------------------ build, report, check
FIGURES = {
    "03-cliff-contrast": (contrast_data, contrast_figure),
    "03-nstep-dial": (dial_data, dial_figure),
}


def build() -> tuple[dict[Path, bytes], dict[str, dict]]:
    out: dict[Path, bytes] = {}
    computed: dict[str, dict] = {}
    for name, (make_data, draw) in FIGURES.items():
        data = make_data()
        computed[name] = data
        for t in themes():
            with mpl.rc_context(rc(name, t)):
                out[IMAGES / f"{name}{t.suffix}.svg"] = svg_bytes(draw(t, data))
    return out, computed


def report(computed: dict[str, dict]) -> None:
    c = computed["03-cliff-contrast"]
    print(f"03-cliff-contrast from runs/{c['record']} (budget {c['rec']['budget']})")
    for key in ("sarsa", "q"):
        last = c["curves"][key]["last"]
        print(
            f"  {NAMES[key]:10s} online return, last {c['rec']['budget']['window']} episodes: "
            f"per seed {[round(v, 1) for v in last]}; min {min(last):.1f}, max {max(last):.1f}"
        )
        path, reached = c["paths"][key]
        print(f"  {NAMES[key]:10s} seed-0 greedy path: {len(path) - 1} moves, reaches G: {reached}")
    per = c["rec"]["per_seed"]
    for key in ("sarsa", "q"):
        lens = [
            len(p) - 1 if r else None
            for p, r in zip(per[f"{key}_path"], per[f"{key}_reached"], strict=True)
        ]
        print(f"  {NAMES[key]:10s} greedy-path moves by seed: {lens}")
    gaps = [
        s - q for s, q in zip(c["curves"]["sarsa"]["last"], c["curves"]["q"]["last"], strict=True)
    ]
    print(f"  SARSA minus Q-learning by seed: {[round(g, 1) for g in gaps]}")
    d = computed["03-nstep-dial"]
    print(f"03-nstep-dial from runs/{d['record']} (budget {d['rec']['budget']})")
    for a, line in d["lines"].items():
        for stat in ("iqm", "mean"):
            vals = line[stat]
            best = d["ns"][int(np.argmin(vals))]
            edge = "interior" if best not in (d["ns"][0], d["ns"][-1]) else "AT THE EDGE"
            print(f"  alpha {a:<5g} {stat:4s} RMS by n {vals.tolist()}; best n {best} ({edge})")
        ci = ", ".join(f"[{lo:g}, {hi:g}]" for lo, hi in zip(line["lo"], line["hi"], strict=True))
        print(f"  alpha {a:<5g} 95% bootstrap intervals of the IQM by n: {ci}")
        print(f"  alpha {a:<5g} best n by seed: {line['best_n']}")


def suspect_report() -> None:
    path, rec = newest_record(SUSPECT_EXPERIMENT)
    b, per = rec["budget"], rec["per_seed"]
    print(f"{SUSPECT_EXPERIMENT} from runs/{path.name} (budget {b})")
    windows = {
        f"first {b['first']}": slice(0, b["first"]),
        f"last {b['last']}": slice(-b["last"], None),
    }
    windows["all"] = slice(None)
    for label, sl in windows.items():
        s = [float(np.mean(o[sl])) for o in per["suspect_online"]]
        f = [float(np.mean(o[sl])) for o in per["fixed_online"]]
        above = sum(x > y for x, y in zip(s, f, strict=True))
        ties = sum(x == y for x, y in zip(s, f, strict=True))
        print(
            f"  mean online return, {label} episodes: suspect {[round(x, 1) for x in s]} "
            f"(min {min(s):.1f}, max {max(s):.1f}); fixed {[round(x, 1) for x in f]} "
            f"(min {min(f):.1f}, max {max(f):.1f}); suspect higher on {above} of {len(s)} "
            f"seeds, equal on {ties}"
        )
    for key in ("suspect", "fixed"):
        print(
            f"  {key}: Q(above goal, down) {sorted(set(round(v, 4) for v in per[f'{key}_q_last_move']))}; "
            f"final max Q(start) {[round(c[-1], 3) for c in per[f'{key}_max_q_start']]}; "
            f"greedy-path moves {per[f'{key}_path_moves']}"
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
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m03.py")
        return 1 if stale else 0
    IMAGES.mkdir(exist_ok=True)
    for p, data in files.items():
        p.write_bytes(data)
        print(f"wrote {p.relative_to(ROOT)} ({len(data) / 1024:.0f} KiB)")
    print()
    report(computed)
    with contextlib.suppress(SystemExit):  # no record yet: nothing to report
        suspect_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
