"""Set the threshold of a stochastic checkpoint from many seeds (PLAN.md §7).

A stochastic checkpoint is a prl.checks.StochasticCheck in prl/checks/mNN.py: a metric of
the participant's function (RMS error, regret, return), its direction, a live and a QUICK
budget, and a stated minimum gap. This script computes the metric over many seeds for the
reference solution and for every mutant of that function in labs/mutants/mNN.py (write
random play as a mutant with id="random"). Then:

  threshold = the midpoint between the solution's 1st percentile and the strongest wrong
              version's 99th (higher is better), or between the solution's 99th and the
              strongest wrong version's 1st (lower is better).

It fails loudly, and writes nothing, when that gap is below the check's min_gap (the
checkpoint cannot tell right from wrong at this budget), when the solution fails the
threshold at any seed, or when a wrong version passes it at any seed. It refuses fewer
than 100 seeds at the QUICK budget and fewer than 20 at the live budget.

The solution is what the worked notebook has bound when exercise N's first run cell
starts: notebooks/<slug>.ipynb runs in-process with solutions bound, checkpoint cells
skipped. On success the script writes an experiment record to runs/inbox/ and prints the
thresholds entry to paste into the check. Its comment states N, the threshold, the gap
and the flake bound (rule of three: no failure in N seeds bounds the flake rate at about
3/N). Run it in colab-sim and on arm64; the threshold must hold on both.

Usage:
  uv run python scripts/build_notebooks.py     # the protocol reads the built notebook
  uv run python scripts/threshold_protocol.py m03 check_mc_prediction --budget quick --seeds 200
  uv run python scripts/threshold_protocol.py m03 check_mc_prediction --budget live
  uv run python scripts/add_run_record.py      # file the record, then paste the entry
  uv run python scripts/threshold_protocol.py --from-record runs/<record>.json   # re-derive
Options: --seeds N (default: the floor), --seed-start S (default 0, which includes the
check's own seed 0), --dry-run (measure and report; no record, nothing to paste).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import inspect
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import textwrap
import time
import tomllib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import nbformat
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_run_record  # noqa: E402
import check_notebooks  # noqa: E402
import variables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prl" / "src"))
import prl  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.checks import CheckFailed, StochasticCheck, assert_threshold  # noqa: E402
from prl.lab import _default_env  # noqa: E402

INBOX = ROOT / "runs" / "inbox"
MIN_SEEDS = {"quick": 100, "live": 20}  # PLAN.md §7
P_LOW, P_HIGH = 1.0, 99.0
PERCENTILE_METHOD = "linear"
SOLUTION = "solution"
SCRIPT = "scripts/threshold_protocol.py"
# Stands in for the generated setup cell: the names it defines, without installing anything.
SETUP_STUB = (
    "import prl\nimport prl.plot\nimport prl.runtime\n\n"
    'PRL_REF = "threshold-protocol"\nWORKED = True\nSEED = 0\n'
)
_MISSING = object()


class ProtocolError(Exception):
    """The protocol cannot set a threshold; the message says why."""


# ---------------------------------------------------------------- the solution's namespace
def _stops(cell_id: str, source: str, ex: str) -> bool:
    """Is this exercise `ex`'s first run or checkpoint cell?"""
    if ex == "stretch":
        return cell_id.startswith("stretch-run") or (
            cell_id.startswith("stretch") and "lab.check(" in source
        )
    return re.fullmatch(rf"ex{re.escape(ex)}-(run|chk)[0-9]*", cell_id) is not None


def _skipped(cell_id: str) -> bool:
    # Checkpoint cells are skipped: an earlier stochastic check may not have its threshold yet.
    return (
        cell_id == "gen-finish"
        or cell_id.startswith("verify-")
        or re.match(r"ex[0-9]+[a-z]?-chk", cell_id) is not None
    )


def _close_figures() -> None:
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


def solution_namespace(notebook: Path, ex: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Run the worked notebook in-process up to exercise `ex`'s first run or checkpoint cell.

    Returns the namespace and the (cell id, source) pairs that ran. Earlier run cells run,
    because later code may use what they define; checkpoint cells do not.
    """
    # Run cells call plt.show(). Under an interactive backend (macOS picks one if pyplot was
    # imported earlier without MPLBACKEND) it opens windows and blocks; Agg makes it a no-op.
    matplotlib.use("Agg", force=True)
    nb = nbformat.read(notebook, as_version=4)
    ns: dict[str, Any] = {"__name__": "__main__"}
    executed: list[tuple[str, str]] = []
    out = io.StringIO()
    for cell in nb.cells:
        cid, src = cell.get("id", ""), cell.source
        if cell.cell_type != "code":
            continue
        if _stops(cid, src, ex):
            break
        if _skipped(cid):
            continue
        if cid == "gen-setup":
            src = SETUP_STUB
        try:
            with contextlib.redirect_stdout(out), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")
                exec(compile(src, f"<{notebook.stem} cell {cid}>", "exec"), ns)  # noqa: S102
        except Exception as exc:
            tail = out.getvalue()[-800:].strip()
            raise ProtocolError(
                f"{notebook.name}, cell {cid} raised {type(exc).__name__}: {exc}"
                + (f"\nIts output ended with:\n{tail}" if tail else "")
            ) from exc
        executed.append((cid, src))
    else:
        raise ProtocolError(f"{notebook.name} has no run or checkpoint cell for exercise {ex}")
    _close_figures()
    return ns, executed


@contextlib.contextmanager
def _swapped_in(ns: dict[str, Any], sources: dict[str, str]) -> Iterator[None]:
    """Bind a mutant's functions in the namespace (as verify mode does), then restore."""
    saved = {name: ns.get(name, _MISSING) for name in sources}
    try:
        for name, src in sources.items():
            exec(compile(src, f"<mutant {name}>", "exec"), ns)  # noqa: S102
        yield
    finally:
        for name, obj in saved.items():
            if obj is _MISSING:
                ns.pop(name, None)
            else:
                ns[name] = obj


@contextlib.contextmanager
def _environment(**values: str) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ---------------------------------------------------------------- measuring
def _name(key: str) -> str:
    if key == SOLUTION:
        return "the solution"
    kind, _, ident = key.partition(":")
    return f"{kind} {ident!r}"


def _sweep(
    check: StochasticCheck,
    fns: tuple[Any, ...],
    seeds: list[int],
    budget: dict[str, Any],
    *,
    key: str,
) -> tuple[list[float | None], float]:
    """The metric at every seed, and the seconds it took. None: the metric itself rejected it."""
    values: list[float | None] = []
    start = time.perf_counter()
    for seed in seeds:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                value = check.measure(*fns, seed=seed, budget=budget)
        except CheckFailed as exc:
            if key == SOLUTION:
                raise ProtocolError(
                    f"the solution fails the check's metric at seed {seed}: {exc}"
                ) from exc
            values.append(None)
            continue
        except Exception as exc:
            raise ProtocolError(
                f"{_name(key)} raised {type(exc).__name__} at seed {seed}: {exc}. "
                "A crash is not a rejection: fix the code, not the threshold."
            ) from exc
        if not math.isfinite(value):
            if key == SOLUTION:
                raise ProtocolError(f"the solution's metric is {value} at seed {seed}")
            if math.isinf(value) and (value > 0) == check.higher_is_better:
                raise ProtocolError(
                    f"{_name(key)} scores {value} at seed {seed}, which passes any threshold"
                )
            values.append(None)  # NaN, or infinitely bad: assert_threshold rejects it
            continue
        values.append(value)
    return values, time.perf_counter() - start


# ---------------------------------------------------------------- the threshold
@dataclass
class Analysis:
    higher_is_better: bool
    min_gap: float
    seeds: list[int]
    solution_tail: float  # the solution's 1st percentile (higher is better) or 99th
    tails: dict[str, float]  # each wrong version's 99th percentile (higher is better) or 1st
    strongest: str
    gap: float  # positive when the two tails are separated
    midpoint: float
    threshold: float  # the midpoint, rounded to within |gap|/10
    solution_failures: list[int]
    escapes: dict[str, list[int]]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def n(self) -> int:
        return len(self.seeds)

    @property
    def flake_bound(self) -> float:
        """Rule of three: no failure in n trials puts the 95% upper bound at about 3/n."""
        return 3 / self.n

    @property
    def quantiles(self) -> tuple[float, float]:
        """(the solution's percentile, the wrong versions' percentile)."""
        return (P_LOW, P_HIGH) if self.higher_is_better else (P_HIGH, P_LOW)


def _percentile(values: list[float | None], q: float, worst: float) -> float:
    arr = np.array([worst if v is None else v for v in values], dtype=float)
    with np.errstate(invalid="ignore"):
        p = float(np.percentile(arr, q, method=PERCENTILE_METHOD))
    return worst if math.isnan(p) else p


def _round(raw: float, gap: float) -> float:
    """The midpoint with the fewest significant digits that stay within |gap|/10 of it."""
    if not (gap != 0 and math.isfinite(raw) and math.isfinite(gap)):
        return raw
    for digits in range(2, 18):
        rounded = float(f"{raw:.{digits}g}")
        if abs(rounded - raw) <= abs(gap) / 10:
            return rounded
    return raw


def _threshold_words(a: Analysis) -> str:
    if a.threshold == a.midpoint:
        return f"the midpoint, {a.threshold!r}"
    return f"the midpoint {a.midpoint:.6g}, rounded to {a.threshold!r}"


def _passes(value: float, threshold: float, higher_is_better: bool) -> bool:
    """Exactly the checkpoint's own comparison."""
    try:
        assert_threshold(value, threshold, provenance="", higher_is_better=higher_is_better)
    except CheckFailed:
        return False
    return True


def _seed_list(seeds: list[int]) -> str:
    shown = ", ".join(map(str, seeds[:8]))
    return f"seeds {shown}" + (", ..." if len(seeds) > 8 else "")


def analyze(
    per_seed: dict[str, list[float | None]],
    seeds: list[int],
    *,
    higher_is_better: bool,
    min_gap: float,
) -> Analysis:
    """Place the threshold and test it. A pure function of the record's data."""
    if SOLUTION not in per_seed:
        raise ProtocolError("there is no per-seed metric for the solution")
    wrong = {k: v for k, v in per_seed.items() if k != SOLUTION}
    if not wrong:
        raise ProtocolError("no wrong version was measured")
    for key, vals in per_seed.items():
        if len(vals) != len(seeds):
            raise ProtocolError(f"{_name(key)} has {len(vals)} values for {len(seeds)} seeds")
    sol = per_seed[SOLUTION]
    if any(v is None for v in sol):
        raise ProtocolError("the solution has seeds without a finite metric")
    hib = higher_is_better
    worst = -math.inf if hib else math.inf
    q_sol, q_wrong = (P_LOW, P_HIGH) if hib else (P_HIGH, P_LOW)
    solution_tail = _percentile(sol, q_sol, worst)
    tails = {key: _percentile(vals, q_wrong, worst) for key, vals in wrong.items()}
    strongest = (max if hib else min)(tails, key=tails.__getitem__)
    best = tails[strongest]
    if not math.isfinite(best):
        raise ProtocolError(
            "the metric itself rejected every wrong version at every seed (CheckFailed or NaN), "
            "so they cannot place a threshold. Add a wrong version the metric can score, "
            'such as random play (a mutant with id="random").'
        )
    gap = solution_tail - best if hib else best - solution_tail
    midpoint = 0.5 * (solution_tail + best)
    threshold = _round(midpoint, gap)
    failures = [s for s, v in zip(seeds, sol, strict=True) if not _passes(v, threshold, hib)]
    escapes = {
        key: [
            s
            for s, v in zip(seeds, vals, strict=True)
            if v is not None and _passes(v, threshold, hib)
        ]
        for key, vals in wrong.items()
    }
    problems = []
    if gap < min_gap:
        problems.append(
            f"the gap {gap:.4g} between the solution's p{q_sol:g} and the p{q_wrong:g} of "
            f"{_name(strongest)} is below the stated minimum {min_gap:g}. "
            "At this budget the checkpoint cannot reliably tell right from wrong: raise the "
            "budget or choose a sharper metric."
        )
    if failures:
        problems.append(
            f"the solution fails the threshold {threshold:.6g} at {len(failures)} of "
            f"{len(seeds)} seeds ({_seed_list(failures)}); the flake bound needs zero failures."
        )
    for key, bad in escapes.items():
        if bad:
            problems.append(
                f"{_name(key)} passes the threshold {threshold:.6g} at {len(bad)} of "
                f"{len(seeds)} seeds ({_seed_list(bad)}); every checkpoint must reject every "
                "mutant."
            )
    return Analysis(
        higher_is_better=hib,
        min_gap=min_gap,
        seeds=list(seeds),
        solution_tail=solution_tail,
        tails=tails,
        strongest=strongest,
        gap=gap,
        midpoint=midpoint,
        threshold=threshold,
        solution_failures=failures,
        escapes=escapes,
        problems=problems,
    )


# ---------------------------------------------------------------- the protocol
@dataclass
class Result:
    module: str
    check_name: str
    check: StochasticCheck
    budget: str
    settings: dict[str, Any]
    notebook: Path
    content_sha: str | None
    mutants_file: Path
    seeds: list[int]
    per_seed: dict[str, list[float | None]]
    why: dict[str, str]
    seconds: float
    seconds_per_call: float
    inputs_sha: str
    analysis: Analysis


def check_seed_floor(budget: str, n_seeds: int) -> None:
    if budget not in MIN_SEEDS:
        raise ProtocolError(f"the budget must be 'quick' or 'live', not {budget!r}")
    if n_seeds < MIN_SEEDS[budget]:
        raise ProtocolError(
            f"refusing to run: the protocol needs at least {MIN_SEEDS[budget]} seeds at the "
            f"{budget} budget (PLAN.md §7); asked for {n_seeds}."
        )


def _source(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return repr(obj)


def run_protocol(
    check: StochasticCheck,
    *,
    check_name: str,
    notebook: Path,
    budget: str,
    n_seeds: int,
    seed_start: int = 0,
    mutants_dir: Path | None = None,
    progress: Callable[[str], None] = print,
) -> Result:
    """Measure the solution and every wrong version over the seeds, then analyze."""
    check_seed_floor(budget, n_seeds)
    meta = nbformat.read(notebook, as_version=4).metadata.get("prl", {})
    module = meta.get("module")
    if not module:
        raise ProtocolError(f"{notebook.name} has no prl.module in its metadata")
    mutants_dir = mutants_dir or check_notebooks.MUTANTS
    wrong = [
        m
        for m in check_notebooks.load_mutants(module, mutants_dir)
        if m["ex"] == check.ex and set(m["sources"]) <= set(check.functions)
    ]
    if not wrong:
        raise ProtocolError(
            f"{module}.py in {mutants_dir.name}/ has no mutant of exercise {check.ex} that "
            f"replaces {' or '.join(check.functions)}. The protocol needs at least one wrong "
            'version: a plausible mistake, or random play as a mutant with id="random".'
        )
    seeds = list(range(seed_start, seed_start + n_seeds))
    if check.seed not in seeds:
        progress(f"Note: the check's own seed {check.seed} is not among the protocol's seeds.")
    settings = dict(check.quick if budget == "quick" else check.live)
    start = time.perf_counter()
    with _environment(PRL_QUICK="1" if budget == "quick" else "0", PRL_WORKED="1"):
        ns, executed = solution_namespace(notebook, check.ex)
        missing = [name for name in check.functions if name not in ns]
        if missing:
            raise ProtocolError(
                f"the worked notebook defines no {', '.join(missing)} before exercise "
                f"{check.ex}'s run cell"
            )
        fns = tuple(ns[name] for name in check.functions)
        per_seed: dict[str, list[float | None]] = {}
        per_seed[SOLUTION], sol_seconds = _sweep(check, fns, seeds, settings, key=SOLUTION)
        progress(f"  the solution: {n_seeds} seeds in {sol_seconds:.1f} s")
        with contextlib.redirect_stdout(io.StringIO()):
            again = check.measure(*fns, seed=seeds[0], budget=settings)
        if not math.isclose(again, per_seed[SOLUTION][0], rel_tol=1e-9, abs_tol=1e-12):
            raise ProtocolError(
                f"the metric is not reproducible: seed {seeds[0]} gave "
                f"{per_seed[SOLUTION][0]!r}, then {again!r}. Take all randomness from `seed`."
            )
        why: dict[str, str] = {}
        for m in wrong:
            key = f"mutant:{m['id']}"
            with _swapped_in(ns, m["sources"]):
                fns = tuple(ns[name] for name in check.functions)
                per_seed[key], seconds = _sweep(check, fns, seeds, settings, key=key)
            why[key] = m["why"]
            progress(f"  {_name(key)}: {n_seeds} seeds in {seconds:.1f} s")
        _close_figures()
    elapsed = time.perf_counter() - start
    analysis = analyze(
        per_seed, seeds, higher_is_better=check.higher_is_better, min_gap=check.min_gap
    )
    inputs = [
        *executed,
        *((f"mutant:{m['id']}", "\n".join(m["sources"].values())) for m in wrong),
        ("metric", _source(check.metric)),
        ("budget", json.dumps(settings, sort_keys=True)),
    ]
    return Result(
        module=module,
        check_name=check_name,
        check=check,
        budget=budget,
        settings=settings,
        notebook=notebook,
        content_sha=meta.get("content_sha"),
        mutants_file=mutants_dir / f"{module}.py",
        seeds=seeds,
        per_seed=per_seed,
        why=why,
        seconds=elapsed,
        seconds_per_call=sol_seconds / n_seeds,
        inputs_sha=record.content_sha(inputs),
        analysis=analysis,
    )


# ---------------------------------------------------------------- the record
def _rel(path: Path) -> str:
    """A repo-relative path, or just the file name: records never hold absolute paths."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _prl_version() -> str:
    if not prl.__version__.endswith("+unknown"):
        return prl.__version__
    with open(ROOT / "prl" / "pyproject.toml", "rb") as fh:  # running from source, not installed
        return tomllib.load(fh)["project"]["version"]


def _git_state() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return sha or None, bool(status.strip())


def _where(rec: dict[str, Any]) -> str:
    hw = rec["hardware"]
    return f"{rec['env']} ({hw['cpu']}, {hw['arch']})"


def _summary(a: Analysis) -> str:
    q_sol, q_wrong = a.quantiles
    direction = "higher" if a.higher_is_better else "lower"
    return (
        f"Threshold {a.threshold!r} ({direction} is better): the solution's p{q_sol:g} is "
        f"{a.solution_tail:.4g}, the p{q_wrong:g} of {_name(a.strongest)} is "
        f"{a.tails[a.strongest]:.4g}, the gap is {a.gap:.4g} (stated minimum {a.min_gap:g}) "
        f"and the threshold is {_threshold_words(a)}. "
        f"The solution fails at 0 of {a.n} seeds and every wrong version is rejected at all "
        f"{a.n}: flake rate at most about 3/{a.n} = {a.flake_bound:.1%} (rule of three). "
        "A null in per_seed is a seed the metric itself rejected."
    )


def build_record(res: Result) -> dict[str, Any]:
    """The experiment record (kind: experiment) of a successful protocol run."""
    rt = runtime.detect()
    git_sha, dirty = _git_state()
    a = res.analysis
    return {
        "schema": record.SCHEMA,
        "kind": "experiment",
        "experiment": f"threshold-{res.module}-{res.check_name}-{res.budget}-n{a.n}",
        "script": SCRIPT,
        "args": {
            "module": res.module,
            "check": res.check_name,
            "exercise": res.check.ex,
            "functions": list(res.check.functions),
            "notebook": _rel(res.notebook),
            "notebook_content_sha": res.content_sha,
            "mutants": _rel(res.mutants_file),
            "wrong_versions": res.why,
            "higher_is_better": res.check.higher_is_better,
            "min_gap": res.check.min_gap,
            "percentiles": [P_LOW, P_HIGH],
            "percentile_method": PERCENTILE_METHOD,
            "check_seed": res.check.seed,
            "inputs_sha": res.inputs_sha,
            "repo_dirty": dirty,
            "test_doubles": runtime.test_doubles_used(),
            "shims": runtime.shims_used(),
        },
        "prl_version": _prl_version(),
        "git_sha": git_sha,
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "platform": rt.platform,
        "env": _default_env(rt, res.check.designed),
        "hardware": {
            "cpu": rt.cpu,
            "n_cpu": rt.n_cpu,
            "ram_gb": rt.ram_gb,
            "accel": rt.accel,
            "gpu": rt.gpu,
            "arch": platform.machine(),
            "torch_threads": rt.torch_threads,
        },
        "python": rt.python,
        "packages": runtime.package_versions(list(runtime.DEFAULT_PACKAGES)),
        "budget": {
            "name": res.budget,
            "quick": res.budget == "quick",
            "designed": res.check.designed,
            "settings": res.settings,
            "solution_seconds_per_call": round(res.seconds_per_call, 6),
        },
        "seeds": res.seeds,
        "per_seed": res.per_seed,
        "seconds": round(res.seconds, 1),
        "note": _summary(a),
    }


def write_record(rec: dict[str, Any], inbox: Path = INBOX) -> Path:
    """Validate the record as scripts/add_run_record.py will, then save it in the inbox."""
    problems = add_run_record.check(rec, variables.load())
    if problems:
        raise ProtocolError("the experiment record is not valid: " + "; ".join(problems))
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / record.record_filename(rec)
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------- output
def format_report(per_seed: dict[str, list[float | None]], a: Analysis, header: str) -> str:
    q_sol, q_wrong = a.quantiles
    worst = -math.inf if a.higher_is_better else math.inf
    direction = "higher" if a.higher_is_better else "lower"
    lines = [
        header,
        f"{a.n} seeds ({a.seeds[0]}-{a.seeds[-1]}) · {direction} is better · "
        f"stated minimum gap {a.min_gap:g}",
        "",
        f"  {'version':30s} {'p1':>10s} {'median':>10s} {'p99':>10s}  at threshold {a.threshold!r}",
    ]
    for key in sorted(per_seed, key=lambda k: k != SOLUTION):  # the solution first
        vals = per_seed[key]
        p1, p50, p99 = (_percentile(vals, q, worst) for q in (P_LOW, 50.0, P_HIGH))
        if key == SOLUTION:
            verdict = f"fails at {len(a.solution_failures)} of {a.n}"
        else:
            verdict = f"passes at {len(a.escapes[key])} of {a.n}"
            rejected = sum(v is None for v in vals)
            if rejected:
                verdict += f" ({rejected} rejected by the metric itself)"
        lines.append(f"  {_name(key):30s} {p1:10.4g} {p50:10.4g} {p99:10.4g}  {verdict}")
    lines += [
        "",
        f"Solution p{q_sol:g} = {a.solution_tail:.4g}; strongest wrong version: "
        f"{_name(a.strongest)}, p{q_wrong:g} = {a.tails[a.strongest]:.4g}.",
        f"Gap {a.gap:.4g} (stated minimum {a.min_gap:g}). Threshold: {_threshold_words(a)}.",
    ]
    if a.ok:
        lines.append(
            f"The solution fails at 0 of {a.n} seeds, so its flake rate is at most about "
            f"3/{a.n} = {a.flake_bound:.1%} (rule of three)."
        )
    else:
        lines.append("FAILED: this budget gives no threshold.")
        lines += [f"  - {p}" for p in a.problems]
    return "\n".join(lines)


def paste_entry(
    a: Analysis,
    *,
    budget: str,
    settings: dict[str, Any],
    record_name: str,
    date: str,
    where: str,
) -> str:
    """The thresholds entry for the StochasticCheck, with its provenance comment."""
    q_sol, q_wrong = a.quantiles
    comment = (
        f"threshold_protocol.py, {budget} budget {json.dumps(settings, sort_keys=True)}, "
        f"N = {a.n} seeds ({a.seeds[0]}-{a.seeds[-1]}), {date}, {where}. "
        f"Solution p{q_sol:g} {a.solution_tail:.4g}; strongest wrong version "
        f"{_name(a.strongest)} p{q_wrong:g} {a.tails[a.strongest]:.4g}; gap {a.gap:.4g} "
        f"(minimum {a.min_gap:g}); threshold: {_threshold_words(a)}. 0 of {a.n} solution seeds fail: "
        f"flake rate at most about 3/N = {a.flake_bound:.1%} (rule of three)."
    )
    lines = [f"        # {line}" for line in textwrap.wrap(comment, width=86)]
    provenance = f"threshold protocol, {a.n} seeds: runs/{record_name}"
    lines.append(f"        {json.dumps(budget)}: ({a.threshold!r}, {json.dumps(provenance)}),")
    return "\n".join(lines)


# ---------------------------------------------------------------- command line
def load_check(module: str, name: str) -> StochasticCheck:
    target = f"prl.checks.{module}"
    try:
        mod = importlib.import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name != target:
            raise
        raise ProtocolError(f"there is no prl/checks/{module}.py") from exc
    obj = getattr(mod, name, None)
    if not isinstance(obj, StochasticCheck):
        found = sorted(k for k, v in vars(mod).items() if isinstance(v, StochasticCheck))
        raise ProtocolError(
            f"{target}.{name} is not a StochasticCheck. "
            f"Stochastic checks in {target}: {', '.join(found) or 'none'}."
        )
    return obj


def notebook_for(module: str) -> Path:
    v = variables.load()
    if module not in v["modules"]:
        raise ProtocolError(f"unknown module {module!r}")
    path = check_notebooks.NB_DIR / f"{v['modules'][module]['slug']}.ipynb"
    if not path.exists():
        raise ProtocolError(f"{path.name} does not exist: run scripts/build_notebooks.py")
    return path


def ensure_current(notebook: Path) -> None:
    """Refuse a notebook that is out of date with its source (needs jupytext)."""
    try:
        import build_notebooks
    except ImportError as exc:  # colab-sim has no jupytext
        print(f"(Not checking that {notebook.name} is current: {exc}.)")
        return
    src = build_notebooks.SRC / f"{notebook.stem}.py"
    try:
        _, built = build_notebooks.build(src, variables.load())
    except build_notebooks.BuildError as exc:
        raise ProtocolError(f"{src.name} does not build: {exc}") from exc
    if built["text"] != notebook.read_text():
        raise ProtocolError(
            f"{notebook.name} is out of date with labs/src/{src.name}: "
            "run scripts/build_notebooks.py first."
        )


def from_record(path: Path) -> int:
    """Re-derive the threshold and the entry from a filed record, without rerunning."""
    rec = json.loads(path.read_text())
    if rec.get("kind") != "experiment" or rec.get("script") != SCRIPT:
        raise ProtocolError(f"{path.name} is not a threshold-protocol experiment record")
    args = rec["args"]
    if args.get("percentiles") != [P_LOW, P_HIGH] or args.get("percentile_method") != (
        PERCENTILE_METHOD
    ):
        raise ProtocolError(
            f"{path.name} used percentiles {args.get('percentiles')} "
            f"({args.get('percentile_method')}); this script uses {[P_LOW, P_HIGH]} "
            f"({PERCENTILE_METHOD})"
        )
    a = analyze(
        rec["per_seed"],
        rec["seeds"],
        higher_is_better=args["higher_is_better"],
        min_gap=args["min_gap"],
    )
    budget = rec["budget"]
    header = (
        f"From {path.name}: prl.checks.{args['module']}.{args['check']} (exercise "
        f"{args['exercise']}: {', '.join(args['functions'])}), {budget['name']} budget "
        f"{json.dumps(budget['settings'], sort_keys=True)}, {rec['date']}, {_where(rec)}"
    )
    print(format_report(rec["per_seed"], a, header))
    if a.ok:
        print("\nThe thresholds entry this record supports:\n")
        print(
            paste_entry(
                a,
                budget=budget["name"],
                settings=budget["settings"],
                record_name=path.name,
                date=rec["date"],
                where=_where(rec),
            )
        )
    return 0 if a.ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("module", nargs="?", help="module id, for example m03")
    ap.add_argument("check", nargs="?", help="a StochasticCheck in prl/checks/<module>.py")
    ap.add_argument("--budget", choices=sorted(MIN_SEEDS))
    ap.add_argument("--seeds", type=int, help="number of seeds (default: the floor)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="measure and report only")
    ap.add_argument("--inbox", type=Path, default=INBOX, help="where to save the record")
    ap.add_argument("--from-record", type=Path, metavar="RECORD", help="re-derive from a record")
    args = ap.parse_args(argv)
    try:
        if args.from_record:
            return from_record(args.from_record)
        if not (args.module and args.check and args.budget):
            ap.error("give MODULE CHECK --budget quick|live, or --from-record RECORD")
        n_seeds = args.seeds if args.seeds is not None else MIN_SEEDS[args.budget]
        check_seed_floor(args.budget, n_seeds)
        check = load_check(args.module, args.check)
        notebook = notebook_for(args.module)
        ensure_current(notebook)
        settings = dict(check.quick if args.budget == "quick" else check.live)
        print(
            f"Threshold protocol: prl.checks.{args.module}.{args.check} (exercise {check.ex}: "
            f"{', '.join(check.functions)}), {args.budget} budget "
            f"{json.dumps(settings, sort_keys=True)}"
        )
        res = run_protocol(
            check,
            check_name=args.check,
            notebook=notebook,
            budget=args.budget,
            n_seeds=n_seeds,
            seed_start=args.seed_start,
        )
    except ProtocolError as exc:
        print(f"threshold_protocol: {exc}", file=sys.stderr)
        return 1
    print()
    print(format_report(res.per_seed, res.analysis, f"Notebook {res.notebook.name}"))
    rt = runtime.detect()
    print(
        f"One checkpoint call at this budget took about {res.seconds_per_call:.3g} s on this "
        f"machine ({_default_env(rt, check.designed)}, {rt.cpu}, {platform.machine()})."
    )
    current = check.thresholds.get(args.budget)
    if current:
        print(f"The check now uses {current[0]!r} ({current[1]}).")
    if not res.analysis.ok:
        print("No record written.")
        return 1
    if args.dry_run:
        print("Dry run: no record written, so there is nothing to paste. Rerun without --dry-run.")
        return 0
    rec = build_record(res)
    try:
        path = write_record(rec, args.inbox)
    except ProtocolError as exc:
        print(f"threshold_protocol: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {path.name} in {_rel(path.parent)}/ (file it with scripts/add_run_record.py).")
    print(f"\nPaste into prl/checks/{args.module}.py, in the thresholds of {args.check}:\n")
    print(
        paste_entry(
            res.analysis,
            budget=args.budget,
            settings=res.settings,
            record_name=path.name,
            date=rec["date"],
            where=_where(rec),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
