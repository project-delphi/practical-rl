"""Module 9's signals, taken from the lab itself, and the margins its recorded runs must show.

The functions come from labs/src/09-debugging-evaluation.py, not from a copy: the solution
cells of Exercises 1, 2, 5 and 6, and the provided helpers of the data cell (`run_log`,
`q_start`, `smooth_returns`, `last_five`, `second_half_median`), exec'd with NumPy only. So the
margins below, the fixtures in scripts/make_expected.py and the numbers asserted by
scripts/make_figures_m09.py are computed by the code the participants see.

NumPy only (no torch, no gymnasium): scripts/make_datasets.py and scripts/make_expected.py
import this module, and CI's drift gate runs them without torch.

`margins(data)` evaluates every margin of the director's handoff (section 4) on every seed of
the dataset `m09_broken_runs` and returns them; `assert_margins(data)` raises MarginError, with
the numbers, if a hard one fails. Seeds and configurations are fixed: a failed margin is
reported, never fixed by choosing seeds.
"""

from __future__ import annotations

import ast
import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "labs" / "src" / "09-debugging-evaluation.py"
RUNS = ("healthy", "A", "B", "C", "D")
BROKEN = ("A", "B", "C", "D")
GAMMA = 0.99
G_MAX = sum(GAMMA**t for t in range(500))  # the most a 500-step CartPole episode can earn
PROVIDED = ("run_log", "smooth_returns", "q_start", "second_half_median", "last_five")
SOLUTION_CELLS = ("ex1-sol", "ex2-sol", "ex5-sol", "ex6-sol")


class MarginError(AssertionError):
    """A recorded run does not show a margin the lab relies on."""


def lab_cells(path: Path = LAB) -> list[tuple[str, str]]:
    """The jupytext cells of a lab source as (header line, body) pairs."""
    cells: list[tuple[str, list[str]]] = []
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("# %%"):
            cells.append((line.strip(), []))
        elif cells:
            cells[-1][1].append(line)
    return [(head, "".join(body)) for head, body in cells]


def cell_by_role(role: str, path: Path = LAB) -> str:
    found = [body for head, body in lab_cells(path) if re.search(rf'role="{role}"', head)]
    if len(found) != 1:
        raise ValueError(f"{path.name}: expected one cell with role {role}, found {len(found)}")
    return found[0]


def code_cell_with(text: str, path: Path = LAB) -> str:
    found = [
        body
        for head, body in lab_cells(path)
        if "[markdown]" not in head and "role=" not in head and text in body
    ]
    if len(found) != 1:
        raise ValueError(f"{path.name}: expected one provided cell with {text!r}, got {len(found)}")
    return found[0]


def _defs(source: str, names: tuple[str, ...]) -> str:
    """The top-level function definitions `names` from a cell, as source."""
    tree = ast.parse(source)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in keep}
    if missing:
        raise ValueError(f"the lab's data cell defines no {', '.join(sorted(missing))}")
    return "\n\n".join(ast.get_source_segment(source, n) for n in keep)


def lab_functions(data: dict[str, np.ndarray] | None = None) -> types.SimpleNamespace:
    """The lab's solutions and provided helpers. `data` is bound as the lab's RUNS."""
    ns: dict[str, Any] = {"np": np, "RUNS": data}
    lab = types.SimpleNamespace(solution=lambda n: lambda f: f)
    ns["lab"] = lab
    for role in SOLUTION_CELLS:
        exec(compile(cell_by_role(role), f"<lab {role}>", "exec"), ns)  # noqa: S102
    provided = code_cell_with('RUNS = prl.data.load("m09_broken_runs")')
    exec(compile(_defs(provided, PROVIDED), "<lab data cell>", "exec"), ns)  # noqa: S102
    ns.pop("lab")
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if callable(v)})


# ------------------------------------------------------------------ the margins
@dataclass
class Margins:
    """Every margin's numbers, per run and seed, and the problems found."""

    table: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    hard: list[str] = field(default_factory=list)  # failed hard margins: stop
    soft: list[str] = field(default_factory=list)  # failed wording checks: report
    lines: list[str] = field(default_factory=list)  # one line per margin, pass or fail

    def ok(self) -> bool:
        return not self.hard


def signal_table(data: dict[str, np.ndarray]) -> dict[str, dict[str, list[float]]]:
    """Each signal's per-seed summary for every run, as Exercise 3's table computes it,
    plus the rewards, Q at the start states at the end and the greedy return at the end."""
    f = lab_functions(data)
    seeds = [int(s) for s in data["seeds"]]
    out: dict[str, dict[str, list[float]]] = {}
    for run in RUNS:
        logs = [f.run_log(run, s) for s in seeds]
        out[run] = {
            "gather_check": [float(f.gather_check(g)) for g in logs],
            "target_lag_last": [float(f.target_lag(g)[-1]) for g in logs],
            "q_drift_last5": [f.last_five(f.q_drift(g)) for g in logs],
            "td_scale_2nd_half": [f.second_half_median(f.td_error_scale(g)) for g in logs],
            "q_start_end": [float(f.q_start(g)[-1]) for g in logs],
            "eval_return_end": [float(g["eval_return"][-1]) for g in logs],
            "reward_min": [float(g["snap_reward"].min()) for g in logs],
            "reward_max": [float(g["snap_reward"].max()) for g in logs],
            "smoothed_return_end": [float(f.smooth_returns(g)[1][-1]) for g in logs],
        }
    return out


def margins(data: dict[str, np.ndarray]) -> Margins:
    """The handoff's margins (section 4) on every seed. Nothing is chosen or averaged away."""
    t = signal_table(data)
    m = Margins(table=t)
    budget = int(data["budget_steps"])
    first = int(data["step"][0])

    def need(ok: bool, text: str, *, soft: bool = False) -> None:
        m.lines.append(("pass  " if ok else ("SOFT  " if soft else "FAIL  ")) + text)
        if not ok:
            (m.soft if soft else m.hard).append(text)

    def show(run: str, key: str) -> str:
        return "[" + ", ".join(f"{v:.6g}" for v in t[run][key]) + "]"

    need(
        all(v >= 0.9 for v in t["B"]["gather_check"]),
        f"gather_check: run B >= 0.9 on every seed: B {show('B', 'gather_check')}",
    )
    for run in ("healthy", "A", "C", "D"):
        need(
            all(v == 0.0 for v in t[run]["gather_check"]),
            f"gather_check: run {run} exactly 0.0: {show(run, 'gather_check')}",
        )
    need(
        all(v == budget - first for v in t["D"]["target_lag_last"]),
        f"target_lag at the last log point: run D = {budget - first:,}: "
        f"{show('D', 'target_lag_last')}",
    )
    for run in ("healthy", "A", "B", "C"):
        need(
            all(v == 0 for v in t[run]["target_lag_last"]),
            f"target_lag at the last log point: run {run} = 0: {show(run, 'target_lag_last')}",
        )
    for run in RUNS:
        need(
            min(t[run]["reward_min"]) == 1.0 and max(t[run]["reward_max"]) == 1.0,
            f"rewards in the logged batches all exactly 1.0: run {run} min "
            f"{show(run, 'reward_min')}, max {show(run, 'reward_max')}",
        )
    need(
        all(v > G_MAX for v in t["C"]["q_start_end"]),
        f"Q at the start states at the end: run C > {G_MAX:.4f}: {show('C', 'q_start_end')}",
    )
    for run in ("healthy", "A", "B", "D"):
        need(
            all(v <= G_MAX for v in t[run]["q_start_end"]),
            f"Q at the start states at the end: run {run} <= {G_MAX:.4f}: "
            f"{show(run, 'q_start_end')}",
        )
    others = max(abs(v) for run in RUNS if run != "C" for v in t[run]["q_drift_last5"])
    need(
        min(t["C"]["q_drift_last5"]) > 3 * others,
        f"Q drift (last 5): run C's lowest {min(t['C']['q_drift_last5']):.6g} > 3 x the "
        f"largest |value| of every other run ({others:.6g}, so > {3 * others:.6g})",
    )
    healthy_min = min(t["healthy"]["eval_return_end"])
    for run in BROKEN:
        need(
            max(t[run]["eval_return_end"]) < healthy_min,
            f"greedy return at {budget:,}: run {run} below the healthy minimum "
            f"{healthy_min:.6g}: {show(run, 'eval_return_end')}",
        )
    a_low = min(t["A"]["td_scale_2nd_half"])
    rest = max(v for run in ("healthy", "C", "D") for v in t[run]["td_scale_2nd_half"])
    need(
        a_low >= 1.5 * rest,
        f"(wording) relative TD error, second-half median: run A's lowest {a_low:.6g} >= 1.5 x "
        f"the highest over healthy, C and D ({rest:.6g}; ratio {a_low / rest:.3f})",
        soft=True,
    )
    return m


def assert_margins(data: dict[str, np.ndarray]) -> Margins:
    m = margins(data)
    if not m.ok():
        raise MarginError(
            "m09_broken_runs: a margin the lab relies on fails (seeds and configurations are "
            "fixed; report it, do not pick seeds):\n  " + "\n  ".join(m.hard)
        )
    return m
