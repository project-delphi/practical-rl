"""Set the threshold of a stochastic checkpoint from many seeds (PLAN.md §7).

A stochastic checkpoint is a prl.checks.StochasticCheck in prl/checks/mNN.py: a metric of
the participant's function (RMS error, regret, return), its direction, a live and a QUICK
budget, and a stated minimum gap. This script computes the metric over many seeds for the
reference solution and for every mutant of that function in labs/mutants/mNN.py (write
random play as a mutant with id="random"; a mutant marked exact_only=True is left out,
since only an exact checkpoint can tell it from the solution), plus, when it exists,
labs/mutants/protocol/mNN.py: protocol-only wrong versions of provided code (for a checkpoint
that guards the platform, such as random play in place of a provided training loop), which
verify mode never reads. Mutant ids must be distinct across both files. Then:

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

Inputs, and when a record is stale. A record's inputs_sha hashes what the protocol runs
and compares, in order: the notebook cells it executes (the setup cell as SETUP_STUB),
each wrong version's source, the metric's own source and the budget. The generated init
cell passes the notebook's content_sha to prl.lab.init, and that hash changes with any code
edit anywhere in the lab. So, since inputs hashing 2, that one argument is replaced by a
placeholder before hashing; the init cell's other arguments (the checkpoint list, the
packages) still count. The record keeps the notebook's content_sha as provenance only
(args.notebook_content_sha), marks args.inputs_hashing = 2, and lists each input's own hash
(args.inputs), so a later comparison can name the input that changed.

A threshold's record is current when the inputs the protocol would hash today equal the
ones it hashed; it is stale otherwise, and only a stale record needs refiling. An edit to
an exercise after the checkpoint's does not make the record stale. An edit to a cell the
protocol runs, to a wrong version, to the metric or to the budget does, and so does adding
or removing a wrong version or a checkpoint. The hash does not cover prl code that the
metric calls, or package versions.

--stale checks every threshold that a prl/checks/mNN.py declares: it opens the record its
provenance names and recomputes today's inputs. Records without args.inputs_hashing were
hashed the first way (inputs hashing 1: the init cell as built). They stay valid as filed.
--stale compares them the new way by setting the init cell's content_sha back to the
record's: if today's inputs then reproduce its inputs_sha, only code the protocol does not
run has changed. If they do not, it rebuilds the inputs from the record's commit to name
the input that changed; when that commit does not reproduce the record either, it says the
record is stale without saying why, and that refiling makes it checkable input by input.
It exits 0 only when every record is current.

Usage:
  uv run python scripts/build_notebooks.py     # the protocol reads the built notebook
  uv run python scripts/threshold_protocol.py m03 check_mc_prediction --budget quick --seeds 200
  uv run python scripts/threshold_protocol.py m03 check_mc_prediction --budget live
  uv run python scripts/add_run_record.py      # file the record, then paste the entry
  uv run python scripts/threshold_protocol.py --from-record runs/<record>.json   # re-derive
  uv run python scripts/threshold_protocol.py --stale [MODULE]   # which records need refiling
Options: --seeds N (default: the floor), --seed-start S (default 0, which includes the
check's own seed 0), --dry-run (measure and report; no record, nothing to paste).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import importlib
import inspect
import io
import json
import math
import os
import pkgutil
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
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

RUNS = ROOT / "runs"
INBOX = RUNS / "inbox"
MIN_SEEDS = {"quick": 100, "live": 20}  # PLAN.md §7
# How a record's inputs_sha is computed. 1 (records without args.inputs_hashing): the init
# cell as built, content_sha included. 2: its content_sha replaced by CONTENT_SHA_PLACEHOLDER,
# with each input's own hash in args.inputs.
INPUTS_HASHING = 2
INIT_CELL = "gen-init"
CONTENT_SHA_PLACEHOLDER = "<content_sha>"
# The build writes `prl.lab.init("<slug>", "<content_sha>", ...)` (build_notebooks.py).
_INIT_SHA = re.compile(r'(prl\.lab\.init\(\s*"[^"]*",\s*)"[^"]*"')
PROTOCOL_ONLY = "protocol"  # labs/mutants/protocol/mNN.py: wrong versions for the protocol only
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


def protocol_cells(
    nb: nbformat.NotebookNode, ex: str, name: str = "the notebook"
) -> list[tuple[str, str]]:
    """The (cell id, source) pairs the protocol runs for exercise `ex`, in order.

    Code cells up to exercise `ex`'s first run or checkpoint cell. Earlier run cells are
    included, because later code may use what they define; checkpoint cells, verify cells
    and gen-finish are not. The setup cell is SETUP_STUB.
    """
    cells: list[tuple[str, str]] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cid, src = cell.get("id", ""), cell.source
        if _stops(cid, src, ex):
            return cells
        if _skipped(cid):
            continue
        cells.append((cid, SETUP_STUB if cid == "gen-setup" else src))
    raise ProtocolError(f"{name} has no run or checkpoint cell for exercise {ex}")


def solution_namespace(notebook: Path, ex: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Run the worked notebook in-process up to exercise `ex`'s first run or checkpoint cell.

    Returns the namespace and the (cell id, source) pairs that ran: protocol_cells().
    """
    cells = protocol_cells(nbformat.read(notebook, as_version=4), ex, notebook.name)
    return _run_cells(cells, notebook), cells


def _run_cells(cells: list[tuple[str, str]], notebook: Path) -> dict[str, Any]:
    # Run cells call plt.show(). Under an interactive backend (macOS picks one if pyplot was
    # imported earlier without MPLBACKEND) it opens windows and blocks; Agg makes it a no-op.
    matplotlib.use("Agg", force=True)
    ns: dict[str, Any] = {"__name__": "__main__"}
    out = io.StringIO()
    for cid, src in cells:
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
    _close_figures()
    return ns


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


# ---------------------------------------------------------------- the inputs it hashes
def _source(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return repr(obj)


def with_init_sha(
    inputs: Iterable[tuple[str, str]], value: str | None = CONTENT_SHA_PLACEHOLDER
) -> list[tuple[str, str]]:
    """The inputs with the init cell's content_sha argument set to `value` (None: as built).

    Only that one argument changes; the cell stays, because its other arguments count.
    """
    out = []
    for label, src in inputs:
        if label == INIT_CELL and value is not None:
            src, n = _INIT_SHA.subn(lambda m: f'{m.group(1)}"{value}"', src)
            if n != 1:
                raise ProtocolError(
                    f"the {INIT_CELL} cell has {n} prl.lab.init(slug, content_sha, ...) calls, "
                    "not one: update _INIT_SHA in scripts/threshold_protocol.py to match the "
                    "build's init cell"
                )
        out.append((label, src))
    return out


def hashed_inputs(
    cells: Iterable[tuple[str, str]],
    wrong: Iterable[Mapping[str, Any]],
    metric_source: str,
    settings: Mapping[str, Any],
    *,
    init_sha: str | None = CONTENT_SHA_PLACEHOLDER,
) -> list[tuple[str, str]]:
    """What a record's inputs_sha hashes, as (label, source) pairs in order.

    The default is inputs hashing 2. init_sha=None gives inputs hashing 1 (the init cell as
    built); a record's notebook_content_sha gives hashing 1 as that record would have seen it.
    """
    return with_init_sha(
        [
            *cells,
            *((f"mutant:{m['id']}", "\n".join(m["sources"].values())) for m in wrong),
            ("metric", metric_source),
            ("budget", json.dumps(dict(settings), sort_keys=True)),
        ],
        init_sha,
    )


def input_hashes(inputs: Iterable[tuple[str, str]]) -> list[list[str]]:
    """[label, hash] for each input, in order: a record's args.inputs."""
    return [[label, record.content_sha([(label, src)])] for label, src in inputs]


def wrong_versions(
    check: StochasticCheck, module: str, mutants_dir: Path
) -> tuple[list[dict], list[str], list[str]]:
    """(the wrong versions measured, the exact_only ids left out, ids appearing twice or more).

    Every mutant of the checked functions, whichever exercise declares it (a check of two
    functions scores the mutants of both), from labs/mutants/mNN.py and then
    labs/mutants/protocol/mNN.py; exact_only mutants are the exact checkpoints' job.
    """
    candidates = [
        m
        for m in check_notebooks.load_mutants(module, mutants_dir)
        + check_notebooks.load_mutants(module, mutants_dir / PROTOCOL_ONLY)  # [] when absent
        if set(m["sources"]) <= set(check.functions)
    ]
    ids = [m["id"] for m in candidates]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    excluded = [m["id"] for m in candidates if m.get("exact_only")]
    return [m for m in candidates if not m.get("exact_only")], excluded, duplicated


@dataclass
class ProtocolInputs:
    """What the protocol runs and compares for one check at one budget."""

    module: str
    content_sha: str | None  # the notebook's: provenance only, never hashed
    cells: list[tuple[str, str]]
    wrong: list[dict]
    exact_only: list[str]
    duplicated: list[str]
    metric_source: str
    settings: dict[str, Any]
    mutants_file: Path
    protocol_mutants_file: Path | None

    def hashed(self, init_sha: str | None = CONTENT_SHA_PLACEHOLDER) -> list[tuple[str, str]]:
        return hashed_inputs(
            self.cells, self.wrong, self.metric_source, self.settings, init_sha=init_sha
        )


def protocol_inputs(
    check: StochasticCheck, notebook: Path, budget: str, mutants_dir: Path | None = None
) -> ProtocolInputs:
    """Read, without running anything, what the protocol would run and hash today."""
    nb = nbformat.read(notebook, as_version=4)
    meta = nb.metadata.get("prl", {})
    module = meta.get("module")
    if not module:
        raise ProtocolError(f"{notebook.name} has no prl.module in its metadata")
    mutants_dir = mutants_dir or check_notebooks.MUTANTS
    wrong, excluded, duplicated = wrong_versions(check, module, mutants_dir)
    protocol_file = mutants_dir / PROTOCOL_ONLY / f"{module}.py"
    return ProtocolInputs(
        module=module,
        content_sha=meta.get("content_sha"),
        cells=protocol_cells(nb, check.ex, notebook.name),
        wrong=wrong,
        exact_only=excluded,
        duplicated=duplicated,
        metric_source=_source(check.metric),
        settings=dict(check.quick if budget == "quick" else check.live),
        mutants_file=mutants_dir / f"{module}.py",
        protocol_mutants_file=protocol_file if protocol_file.exists() else None,
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
    exact_only: list[str]
    seconds: float
    seconds_per_call: float
    inputs_sha: str
    analysis: Analysis
    protocol_mutants_file: Path | None = None  # labs/mutants/protocol/mNN.py, if it exists
    git_sha: str | None = None  # read when the run starts, not when it ends
    repo_dirty: bool | None = None
    inputs: list[list[str]] = field(default_factory=list)  # [label, hash] per hashed input


def check_seed_floor(budget: str, n_seeds: int) -> None:
    if budget not in MIN_SEEDS:
        raise ProtocolError(f"the budget must be 'quick' or 'live', not {budget!r}")
    if n_seeds < MIN_SEEDS[budget]:
        raise ProtocolError(
            f"refusing to run: the protocol needs at least {MIN_SEEDS[budget]} seeds at the "
            f"{budget} budget (PLAN.md §7); asked for {n_seeds}."
        )


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
    # The repo state, the cells, the mutants and the metric's source are read before
    # measuring, so editing a file while the protocol runs cannot change what its record
    # says it measured.
    git_sha, repo_dirty = _git_state()
    mutants_dir = mutants_dir or check_notebooks.MUTANTS
    parts = protocol_inputs(check, notebook, budget, mutants_dir)
    module, wrong, excluded, settings = parts.module, parts.wrong, parts.exact_only, parts.settings
    if parts.duplicated:
        # Per-seed values are keyed by id, so a second mutant with the same id would silently
        # replace the first's in the analysis (for example "random" in two exercises).
        raise ProtocolError(
            f"{module}.py in {mutants_dir.name}/ (and {PROTOCOL_ONLY}/): mutant id(s) "
            f"{', '.join(parts.duplicated)} appear more than once among the wrong versions of "
            f"{' and '.join(check.functions)}. Give each a distinct id."
        )
    if excluded:
        progress(f"Left out as exact_only: {', '.join(excluded)}")
    if not wrong:
        raise ProtocolError(
            f"{module}.py in {mutants_dir.name}/ has no mutant of exercise {check.ex} that "
            f"replaces {' or '.join(check.functions)}. The protocol needs at least one wrong "
            'version: a plausible mistake, or random play as a mutant with id="random".'
        )
    inputs = parts.hashed()  # inputs hashing 2; checks the init cell before anything runs
    seeds = list(range(seed_start, seed_start + n_seeds))
    if check.seed not in seeds:
        progress(f"Note: the check's own seed {check.seed} is not among the protocol's seeds.")
    start = time.perf_counter()
    with _environment(PRL_QUICK="1" if budget == "quick" else "0", PRL_WORKED="1"):
        ns = _run_cells(parts.cells, notebook)
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
    return Result(
        module=module,
        check_name=check_name,
        check=check,
        budget=budget,
        settings=settings,
        notebook=notebook,
        content_sha=parts.content_sha,
        mutants_file=parts.mutants_file,
        protocol_mutants_file=parts.protocol_mutants_file,
        seeds=seeds,
        per_seed=per_seed,
        why=why,
        exact_only=excluded,
        seconds=elapsed,
        seconds_per_call=sol_seconds / n_seeds,
        inputs_sha=record.content_sha(inputs),
        inputs=input_hashes(inputs),
        analysis=analysis,
        git_sha=git_sha,
        repo_dirty=repo_dirty,
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
    git_sha, dirty = res.git_sha, res.repo_dirty
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
            **(
                {"protocol_mutants": _rel(res.protocol_mutants_file)}
                if res.protocol_mutants_file
                else {}
            ),
            "wrong_versions": res.why,
            "exact_only_left_out": res.exact_only,
            "higher_is_better": res.check.higher_is_better,
            "min_gap": res.check.min_gap,
            "percentiles": [P_LOW, P_HIGH],
            "percentile_method": PERCENTILE_METHOD,
            "check_seed": res.check.seed,
            "inputs_hashing": INPUTS_HASHING,
            "inputs_sha": res.inputs_sha,
            "inputs": res.inputs,
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


# ---------------------------------------------------------------- staleness
Entry = tuple[str, str, StochasticCheck, str, float, str]


def declared_thresholds(modules: Iterable[str] | None = None) -> list[Entry]:
    """(module, check name, check, budget, threshold, provenance) of every declared threshold."""
    wanted = set(modules) if modules else None
    out: list[Entry] = []
    for info in pkgutil.iter_modules(prl.checks.__path__):
        if wanted is not None and info.name not in wanted:
            continue
        mod = importlib.import_module(f"prl.checks.{info.name}")
        for name, obj in vars(mod).items():
            if isinstance(obj, StochasticCheck):
                for budget, (threshold, provenance) in obj.thresholds.items():
                    out.append((info.name, name, obj, budget, threshold, provenance))
    return out


def _git_show(sha: str, path: str) -> str | None:
    """A file's text at a commit, or None."""
    try:
        out = subprocess.run(
            ["git", "show", f"{sha}:{path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _source_at(sha: str, fn: Any) -> str | None:
    """What inspect.getsource(fn) gave at a commit, if fn's file and name were the same then."""
    try:
        path = Path(inspect.getsourcefile(fn) or "").resolve().relative_to(ROOT.resolve())
    except (TypeError, ValueError):
        return None
    text = _git_show(sha, path.as_posix())
    if text is None:
        return None
    try:
        body: list[ast.stmt] = ast.parse(text).body
    except SyntaxError:
        return None
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None = None
    for part in getattr(fn, "__qualname__", "").split("."):
        found = [
            n
            for n in body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and n.name == part
        ]
        if not found:
            return None
        node = found[-1]  # the last definition is the one bound
        body = node.body
    if node is None:
        return None
    # getsource starts at co_firstlineno, which is the first decorator's line.
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return "".join(inspect.getblock(text.splitlines(keepends=True)[start - 1 :]))


def filed_inputs(rec: dict[str, Any], check: StochasticCheck) -> tuple[list | None, str]:
    """A hashing-1 record's inputs, rebuilt from its commit, in the hashing-2 form.

    Returns (inputs, "") or (None, why not). The rebuilt inputs count only when they
    reproduce the record's inputs_sha, so a rebuild never guesses.
    """
    args, sha = rec["args"], rec.get("git_sha")
    if not sha:
        return None, "the record names no commit"
    short, module = sha[:8], args["module"]
    nb_text = _git_show(sha, args["notebook"])
    if nb_text is None:
        return None, f"commit {short} has no {args['notebook']}"
    try:
        cells = protocol_cells(nbformat.reads(nb_text, as_version=4), str(args["exercise"]))
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp)
            for key, target in (
                ("mutants", here / f"{module}.py"),
                ("protocol_mutants", here / PROTOCOL_ONLY / f"{module}.py"),
            ):
                if args.get(key):
                    text = _git_show(sha, args[key])
                    if text is None:
                        return None, f"commit {short} has no {args[key]}"
                    target.parent.mkdir(exist_ok=True)
                    target.write_text(text)
            loaded = check_notebooks.load_mutants(module, here)
            loaded += check_notebooks.load_mutants(module, here / PROTOCOL_ONLY)
        ids = {key.removeprefix("mutant:") for key in args["wrong_versions"]}
        wrong = [
            m
            for m in loaded
            if m["id"] in ids
            and set(m["sources"]) <= set(args["functions"])
            and not m.get("exact_only")
        ]
        metric = _source_at(sha, check.metric)
        if metric is None:
            name = getattr(check.metric, "__qualname__", "the metric")
            return None, f"commit {short} has no {name} where the metric is today"
        inputs = hashed_inputs(cells, wrong, metric, rec["budget"]["settings"], init_sha=None)
        if record.content_sha(inputs) != args["inputs_sha"]:
            dirty = " (its working tree had uncommitted changes)" if args.get("repo_dirty") else ""
            return None, f"the files at commit {short}{dirty} do not reproduce its inputs_sha"
        return with_init_sha(inputs), ""
    except (ProtocolError, ValueError, SyntaxError, KeyError) as exc:
        return None, f"commit {short} cannot be read the way the protocol reads it ({exc})"


def _input_name(label: str) -> str:
    if label == "metric":
        return "the metric's source"
    if label == "budget":
        return "the budget"
    if label.startswith("mutant:"):
        return f"wrong version {label.removeprefix('mutant:')!r}"
    return f"cell {label}"


def _by_occurrence(pairs: list[tuple[str, ...]]) -> dict[tuple[str, int], str]:
    """{(label, k): hash}, k counting repeats of a label (two mutants with one id, as in B90)."""
    seen: dict[str, int] = {}
    out = {}
    for label, sha in pairs:
        k = seen[label] = seen.get(label, -1) + 1
        out[(label, k)] = sha
    return out


def differences(filed: Iterable[Iterable[str]], now: Iterable[Iterable[str]]) -> list[str]:
    """Which inputs differ, from two [label, hash] lists (a record's args.inputs)."""
    before, after = [tuple(x) for x in filed], [tuple(x) for x in now]
    if before == after:
        return []
    if sorted(before) == sorted(after):
        return ["the same inputs, in a different order"]
    old, new = _by_occurrence(before), _by_occurrence(after)

    def name(key: tuple[str, int]) -> str:
        return _input_name(key[0]) + (f" #{key[1] + 1}" if key[1] else "")

    out = [
        f"{name(key)} {'changed' if key in old else 'is new'}"
        for key, sha in new.items()
        if old.get(key) != sha
    ]
    return out + [f"{name(key)} is gone" for key in old if key not in new]


def _declared_differences(rec: dict[str, Any], entry: Entry) -> list[str]:
    """Where the record's declaration differs from the check's today (not hashed inputs)."""
    module, name, check, budget, _, _ = entry
    args, out = rec["args"], []
    filed_for = (args.get("module"), args.get("check"), rec["budget"].get("name"))
    if filed_for != (module, name, budget):
        out.append("the record is for {} {} at the {} budget".format(*filed_for))
    if args.get("higher_is_better") != check.higher_is_better:
        out.append("the direction (higher_is_better) changed")
    if args.get("min_gap") != check.min_gap:
        out.append(f"min_gap changed from {args.get('min_gap')} to {check.min_gap}")
    return out


def _budget_change(rec: dict[str, Any], today: ProtocolInputs) -> str:
    filed = rec["budget"].get("settings")
    if filed == today.settings:
        return ""
    return f"the budget changed (from {json.dumps(filed)} to {json.dumps(today.settings)})"


def _wrong_version_changes(rec: dict[str, Any], today: ProtocolInputs) -> list[str]:
    filed = {k.removeprefix("mutant:") for k in rec["args"].get("wrong_versions", {})}
    now = {m["id"] for m in today.wrong}
    return [f"wrong version {i!r} is new" for i in sorted(now - filed)] + [
        f"wrong version {i!r} is gone" for i in sorted(filed - now)
    ]


@dataclass
class Freshness:
    """Whether a declared threshold's record matches what the protocol would hash today."""

    module: str
    check_name: str
    exercise: str
    budget: str
    record: str = ""
    status: str = "missing"  # current | stale | unchecked | missing
    hashing: int | None = None  # the record's inputs hashing
    how: str = ""
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def freshness(
    entry: Entry,
    *,
    notebook: Path | None,
    notebook_problem: str = "",
    runs_dir: Path | None = None,
    mutants_dir: Path | None = None,
) -> Freshness:
    """Compare one declared threshold's record with the inputs the protocol would hash today."""
    module, name, check, budget, _, provenance = entry
    f = Freshness(module, name, check.ex, budget)
    found = re.search(r"runs/(\S+\.json)$", provenance)
    if not found:
        f.reasons.append(f"its provenance {provenance!r} names no runs/ record")
        return f
    f.record = found.group(1)
    path = (runs_dir or RUNS) / f.record
    if not path.exists():
        f.reasons.append(f"{f.record} is not filed in runs/")
        return f
    rec = json.loads(path.read_text())
    if rec.get("kind") != "experiment" or rec.get("script") != SCRIPT:
        f.status, f.how = "unchecked", "it is not a threshold-protocol experiment record"
        return f
    args = rec["args"]
    f.hashing = args.get("inputs_hashing", 1)
    declared = _declared_differences(rec, entry)
    try:
        if notebook is None:
            raise ProtocolError(notebook_problem or f"there is no notebook for {module}")
        today = protocol_inputs(check, notebook, budget, mutants_dir)
        now = today.hashed()
    except ProtocolError as exc:
        f.status, f.how, f.reasons = "unchecked", f"today's inputs cannot be read: {exc}", declared
        return f
    if today.duplicated:
        f.notes.append(
            f"the protocol refuses this check today: wrong-version id(s) "
            f"{', '.join(today.duplicated)} appear more than once; rename them before refiling"
        )
    budget_line = _budget_change(rec, today)
    if f.hashing == INPUTS_HASHING:
        f.how = f"compared input by input (inputs hashing {INPUTS_HASHING})"
        changed = []
        if record.content_sha(now) != args["inputs_sha"]:
            changed = differences(args.get("inputs", []), input_hashes(now))
    elif f.hashing == 1:
        filed_sha = args.get("notebook_content_sha")
        replay = None
        if filed_sha:
            replay = record.content_sha(today.hashed(init_sha=filed_sha)) == args["inputs_sha"]
        rebuilt, why = (None, "") if replay else filed_inputs(rec, check)
        if replay:
            f.how = (
                "filed under the old hashing; compared the new way: with the init cell's "
                f"content_sha set back to the record's ({filed_sha}), today's inputs reproduce "
                "its inputs_sha"
            )
            changed = []
        elif rebuilt is not None:
            f.how = (
                "filed under the old hashing; compared the new way, input by input: the files "
                f"at its commit {rec['git_sha'][:8]} reproduce its inputs_sha"
            )
            changed = differences(input_hashes(rebuilt), input_hashes(now))
        elif replay is False:
            f.how = (
                "filed under the old hashing; compared the new way (the init cell's content_sha "
                "set back to the record's): an input changed. The record keeps no per-input "
                f"hashes and {why}, so it cannot say which. Refiling makes it checkable input "
                "by input"
            )
            hints = ([budget_line] if budget_line else []) + _wrong_version_changes(rec, today)
            changed = hints + [
                "whether another input changed too cannot be told"
                if hints
                else "which input changed cannot be told"
            ]
        else:
            f.status, f.reasons = "unchecked", declared
            f.how = (
                "filed under the old hashing, with no notebook_content_sha to set back, and "
                f"{why}: refile to make it checkable"
            )
            return f
    else:
        f.status, f.how = "unchecked", f"unknown inputs_hashing {f.hashing!r}"
        return f
    changed = [budget_line if c == "the budget changed" and budget_line else c for c in changed]
    f.reasons = declared + changed
    f.status = "stale" if f.reasons else "current"
    return f


def stale_report(
    entries: list[Entry],
    *,
    runs_dir: Path | None = None,
    mutants_dir: Path | None = None,
) -> list[Freshness]:
    """freshness() of every entry; each module's notebook is found and checked once."""
    notebooks: dict[str, tuple[Path | None, str]] = {}
    out = []
    for entry in entries:
        module = entry[0]
        if module not in notebooks:
            try:
                nb = notebook_for(module)
                ensure_current(nb)
                notebooks[module] = (nb, "")
            except ProtocolError as exc:
                notebooks[module] = (None, str(exc))
        nb, problem = notebooks[module]
        out.append(
            freshness(
                entry,
                notebook=nb,
                notebook_problem=problem,
                runs_dir=runs_dir,
                mutants_dir=mutants_dir,
            )
        )
    return out


STATUSES = ("current", "stale", "unchecked", "missing")


def format_freshness(results: list[Freshness]) -> str:
    def wrap(text: str, first: str, rest: str) -> list[str]:
        return textwrap.wrap(text, width=96, initial_indent=first, subsequent_indent=rest)

    lines = [
        "Threshold records against the inputs the protocol would hash today "
        f"(inputs hashing {INPUTS_HASHING}).",
        "",
    ]
    for f in results:
        lines.append(
            f"{f.status.upper():9s} {f.module} {f.check_name} (exercise {f.exercise}), "
            f"{f.budget}: {f.record or '(no record)'}"
        )
        if f.how:
            lines += wrap(f.how + ".", " " * 10, " " * 10)
        for reason in f.reasons:
            lines += wrap(reason, " " * 10 + "- ", " " * 12)
        for note in f.notes:
            lines += wrap(note, " " * 10 + "note: ", " " * 16)
    count = {s: sum(f.status == s for f in results) for s in STATUSES}
    old = [f for f in results if f.hashing == 1]
    checked = [f for f in old if f.status in ("current", "stale")]
    by_input = [f for f in checked if "input by input:" in f.how]
    lines += [
        "",
        f"{len(results)} thresholds: " + ", ".join(f"{count[s]} {s}" for s in STATUSES) + ".",
        f"Filed under the old hashing: {len(old)}. Compared the new way: {len(checked)} "
        f"({len(by_input)} input by input, rebuilt from their commits). Not checkable: "
        f"{len(old) - len(checked)}.",
        "The hash does not cover prl code that the metric calls, or package versions.",
    ]
    return "\n".join(lines)


def stale_main(modules: list[str] | None = None) -> int:
    results = stale_report(declared_thresholds(modules))
    print(format_freshness(results))
    return 0 if all(f.status == "current" for f in results) else 1


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
    ap.add_argument(
        "--stale",
        action="store_true",
        help="report which declared thresholds' records are current and which are stale",
    )
    args = ap.parse_args(argv)
    try:
        if args.stale:
            if args.check:
                ap.error("--stale takes at most a MODULE")
            return stale_main([args.module] if args.module else None)
        if args.from_record:
            return from_record(args.from_record)
        if not (args.module and args.check and args.budget):
            ap.error("give MODULE CHECK --budget quick|live, --from-record RECORD, or --stale")
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
