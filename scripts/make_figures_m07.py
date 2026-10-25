"""Module 7 records, figures and quoted numbers: light and dark SVGs.

  uv run python scripts/make_figures_m07.py --run [--only NAME]  # experiments -> runs/inbox/
  uv run python scripts/add_run_record.py              # file them in runs/
  uv run python scripts/export_bands.py                # copy the band into prl
  uv run python scripts/make_figures_m07.py            # draw images/07-*.svg from the records
  uv run python scripts/make_figures_m07.py --check    # fail if a committed SVG differs
  uv run python scripts/make_figures_m07.py --reproduce [--only NAME]  # rerun in memory and
                                                       # compare with the filed records

Everything comes from the lab's own reference code. The worked notebook runs in-process with
solutions bound up to Exercise 6's run cell (threshold_protocol.solution_namespace: Exercises
1-5's run cells run, checkpoints do not). Exercise 6's run cell then contributes only its
constants and function definitions (the features, train_linear_q, greedy_return,
random_return), since the rest of it reads the band these experiments record. The
colleague's unfixed featurizer is the ex7-stub cell.

  m07-linear-q-live, m07-linear-q-quick
      The band m07_linear_q (runs/bands.yml) and the numbers behind Exercises 6 and 7: the
      lab's train_linear_q with linear_q_update and rbf_features at the lab's budget, seeds
      0-19 (no seat uses them: seats draw (SEED + 1) * 1000). Per seed: the log (recent
      return and ‖W‖ every 1000 steps), the greedy and random-play returns over the lab's 10
      evaluation episodes, Q at the seed's first start state; and the same seed trained with
      the colleague's unfixed features (the log with null for non-finite values, the greedy
      return, Q at the start state).

Figures:
  07-legs          Exact: Exercise 5's run cell, the largest value error of Baird's expected
                   update against sweeps, for the original and each leg removed (alpha 0.01).
  07-weight-norms  m07-linear-q-live: ‖W‖ every 1000 steps for the 20 seeds, with
                   rbf_features and with the colleague's unfixed features, log scale.

Every build runs exact_numbers() and record_numbers(): each number the page, a knowledge
check or a predict_answer quotes is asserted, so a changed lab or record fails here before
the page is wrong. The figures read only the newest filed record of each experiment, so
`--check` never trains anything. Drawing is deterministic: svg.hashsalt is the figure name,
no date in the metadata, text as paths, values rounded (and clipped to the axes) before
plotting.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_protocol as tp  # noqa: E402
from make_figures_m01 import MIN_FONT, WIDTH_IN, Theme, rc, svg_bytes, themes  # noqa: E402
from prl import record, runtime  # noqa: E402
from prl.checks import m07 as checks  # noqa: E402
from prl.envs import Baird  # noqa: E402
from prl.lab import _default_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
RUNS = ROOT / "runs"
NOTEBOOK = ROOT / "notebooks" / "07-function-approximation.ipynb"
SCRIPT = "scripts/make_figures_m07.py"

SEEDS = list(range(20))
EXPERIMENTS = {"live": "m07-linear-q-live", "quick": "m07-linear-q-quick"}


# ------------------------------------------------------------------ the lab's own code
def _provided(src: str) -> str:
    """A run cell's constants (ALL_CAPS assignments) and function definitions, nothing else."""

    def caps(target: ast.expr) -> bool:
        if isinstance(target, ast.Name):
            return target.id.isupper()
        return isinstance(target, ast.Tuple) and all(caps(t) for t in target.elts)

    keep = [
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef)
        or (isinstance(n, ast.Assign) and all(caps(t) for t in n.targets))
    ]
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def lab_namespace(*, quick: bool) -> dict:
    """The worked notebook through Exercise 5, plus Exercise 6's plumbing and the colleague's
    unfixed featurizer (as `colleague_features_unfixed`)."""
    import nbformat

    with (
        tp._environment(PRL_QUICK="1" if quick else "0", PRL_WORKED="1"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        ns, _ = tp.solution_namespace(NOTEBOOK, "6")
        cells = {c.get("id"): c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells}
        exec(compile(_provided(cells["ex6-run"]), "<07 ex6-run, provided>", "exec"), ns)  # noqa: S102
        buggy = dict(ns)
        exec(compile(cells["ex7-stub"], "<07 ex7-stub>", "exec"), buggy)  # noqa: S102
        ns["colleague_features_unfixed"] = buggy["colleague_features"]
    tp._close_figures()
    return ns


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _q_start(ns: dict, W: np.ndarray, featurize, seed: int) -> list | None:
    obs, _ = ns["gym"].make("CartPole-v1").reset(seed=seed)
    with np.errstate(over="ignore", invalid="ignore"):
        q = W @ featurize(obs)
    return [float(v) for v in q] if np.all(np.isfinite(q)) else None


def run_linear_q(ns: dict) -> dict:
    """Seeds 0-19 at the lab's budget: the solution with rbf_features, and the colleague's."""
    train, greedy, rand = ns["train_linear_q"], ns["greedy_return"], ns["random_return"]
    update, feats = ns["linear_q_update"], ns["rbf_features"]
    unfixed = ns["colleague_features_unfixed"]
    episodes = ns["budget"].eval_episodes
    per: dict = {
        k: []
        for k in (
            "recent_return",
            "w_norm",
            "greedy_return",
            "random_return",
            "max_w_norm",
            "episodes",
            "q_start",
            "seconds",
        )
    }
    col: dict = {k: [] for k in ("w_norm", "max_finite_w_norm", "greedy_return", "q_start")}
    for seed in SEEDS:
        with contextlib.redirect_stdout(io.StringIO()):
            start = time.perf_counter()
            W, log = train(update, feats, **ns["TRAIN"], seed=seed)
            per["seconds"].append(round(time.perf_counter() - start, 3))  # the band's run
            Wc, clog = train(update, unfixed, **ns["TRAIN"], seed=seed)
        per["recent_return"].append([float(v) for v in log["recent_return"]])
        per["w_norm"].append([float(v) for v in log["w_norm"]])
        per["greedy_return"].append(float(greedy(W, feats, seed, episodes)))
        per["random_return"].append(float(rand(seed, episodes)))
        per["max_w_norm"].append(float(max(log["w_norm"])))
        per["episodes"].append(len(log["returns"]))
        per["q_start"].append(_q_start(ns, W, feats, seed))
        norms = [_finite_or_none(v) for v in clog["w_norm"]]
        finite = [v for v in norms if v is not None]
        col["w_norm"].append(norms)
        col["max_finite_w_norm"].append(max(finite) if finite else None)
        col["greedy_return"].append(float(greedy(Wc, unfixed, seed, episodes)))
        col["q_start"].append(_q_start(ns, Wc, unfixed, seed))
    per["colleague"] = col
    return per


def budget_of(ns: dict) -> dict:
    b = ns["budget"]
    return {
        "steps": b.steps,
        "eps_steps": b.eps_steps,
        "log_every": b.log_every,
        "eval_episodes": b.eval_episodes,
        "grid": list(range(b.log_every, b.steps + 1, b.log_every)),
        "alpha": ns["ALPHA_Q"],
        "gamma": ns["GAMMA_Q"],
        "features": "rbf_features: 625 Gaussian bumps (5^4 grid, width 0.5), unit length",
        "colleague": "the ex7-stub featurizer: the same bumps, not scaled",
        "seeds": "0-19; seats use (SEED + 1) * 1000",
    }


# ------------------------------------------------------------------ records
def experiment_record(
    name: str, budget: dict, per_seed: dict, seconds: float, note: str, git
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
            "notebook": "notebooks/07-function-approximation.ipynb",
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
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "seconds": round(seconds, 1),
        "note": note,
    }


def where() -> str:
    rt = runtime.detect()
    return f"{_default_env(rt, 'colab-cpu')} ({rt.cpu}, {__import__('platform').machine()})"


NOTE = (
    "Linear semi-gradient Q-learning on CartPole-v1 from the worked notebook (Exercise 6's "
    "train_linear_q, greedy_return and random_return with the reference linear_q_update and "
    "rbf_features), at the lab's {label} budget, seeds 0-19. Every per_seed list is indexed "
    "like `seeds`. recent_return[i][j] and w_norm[i][j]: the mean return of the last 10 "
    "finished episodes and ‖W‖ after budget['grid'][j] steps. greedy_return and "
    "random_return: means over the lab's 10 evaluation episodes (reset seeds seed + 10000 + "
    "k). q_start: Q(s0, ·) at reset(seed). seconds: the wall-clock of each seed's training run with rbf_features (a laptop time). colleague: the same seed trained with the "
    "ex7-stub featurizer (unfixed); null marks a non-finite ‖W‖ or Q."
)


def run(inbox: Path = tp.INBOX, only: list[str] | None = None) -> list[Path]:
    git = tp._git_state()  # the repo as it is when the run starts
    tp.ensure_current(NOTEBOOK)
    out = []
    for label, name in EXPERIMENTS.items():
        if only and name not in only:
            continue
        load_start = os.getloadavg()[0]
        ns = lab_namespace(quick=label == "quick")
        start = time.perf_counter()
        per = run_linear_q(ns)
        secs = time.perf_counter() - start
        load = os.getloadavg()[0]
        note = (
            f"{NOTE.format(label=label)} Wall time for the 40 training runs and their "
            f"evaluations on {where()}: a laptop time, not a Colab time. One-minute load "
            f"average {load_start:.1f} at the start and {load:.1f} at the end (other builds "
            "were running on this machine)."
        )
        rec = experiment_record(name, budget_of(ns), per, secs, note, git)
        out.append(tp.write_record(rec, inbox))
    return out


def _diff(old, new, path: str, out: list[str], floats: list[float]) -> None:
    """Exact comparison of JSON-like values, except floats (relative differences collected)."""
    if isinstance(old, float) or isinstance(new, float):
        if (
            old is None
            or new is None
            or isinstance(old, (str, bool))
            or isinstance(new, (str, bool))
        ):
            out.append(f"{path}: {old!r} vs {new!r}")
        else:
            a, b = float(old), float(new)
            floats.append(abs(a - b) / max(1.0, abs(a), abs(b)))
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
    ok = True
    for label, name in EXPERIMENTS.items():
        if only and name not in only:
            continue
        path, rec = newest_record(name)
        ns = lab_namespace(quick=label == "quick")
        if rec["budget"] != json.loads(json.dumps(budget_of(ns))) or rec["seeds"] != SEEDS:
            print(f"{name}: runs/{path.name} has a different budget or seeds; not comparable")
            ok = False
            continue
        new = json.loads(json.dumps(run_linear_q(ns)))
        new["seconds"] = rec["per_seed"]["seconds"]  # wall-clock: not a result to reproduce
        exact: list[str] = []
        floats: list[float] = []
        _diff(rec["per_seed"], new, "per_seed", exact, floats)
        same = sum(f == 0.0 for f in floats)
        worst = max(floats, default=0.0)
        good = not exact and worst <= 1e-9
        ok &= good
        seeds_same = [
            s
            for i, s in enumerate(SEEDS)
            if rec["per_seed"]["w_norm"][i] == new["w_norm"][i]
            and rec["per_seed"]["colleague"]["w_norm"][i] == new["colleague"]["w_norm"][i]
        ]
        print(
            f"{name}: {'REPRODUCED' if good else 'DIFFERS'} against runs/{path.name}: "
            f"{len(exact)} exact mismatches; {same} of {len(floats)} floats bit-identical, "
            f"largest relative difference {worst:.3g}; seeds whose logs are bit-identical: "
            f"{len(seeds_same)} of {len(SEEDS)}"
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


# ------------------------------------------------------------------ exact numbers
def _r(x: float, nd: int) -> float:
    """Round to nd decimals, refusing values that sit on a rounding boundary."""
    scaled = abs(x) * 10**nd
    assert abs(scaled - math.floor(scaled) - 0.5) > 1e-6, (x, nd)
    return round(float(x), nd)


def _sig(x: float, digits: int) -> str:
    return f"{x:.{digits - 1}e}"


def exact_numbers(L: dict) -> dict:
    """Every exact number the page, the predict_answer entries and the knowledge checks quote.

    L is the lab's namespace after Exercise 5's run cell. Returns the platform-dependent
    rounding residuals, which are reported, not asserted to the digit."""
    b = Baird()
    soft: dict = {}
    # ---- Section 1, the Summary, Exercise 1 and knowledge checks 7.5 and 7.6
    spec = L["scale_spec"](3, 50)
    assert spec == {
        "n_states": 132_651,
        "n_actions": 132_651,
        "q_entries": 17_596_287_801,
        "feature_dim": 153,
    }
    assert 51**3 == 132_651 and round(spec["q_entries"] * 8 / 1e9) == 141  # "about 141 GB"
    assert _sig(132_651**3, 2) == "2.3e+15"  # P[s, a, s']
    assert _sig(spec["q_entries"], 3) == "1.76e+10"  # the Summary
    kc5 = L["scale_spec"](4, 20)
    assert (kc5["n_states"], kc5["q_entries"], kc5["feature_dim"]) == (194_481, 37_822_859_361, 84)
    assert round(kc5["q_entries"] * 8 / 1e9, -2) == 300  # "about 300 GB" (302.6)
    assert 3 * 51**2 == 7803  # 7.6: pair indicators
    fits = {}
    for shared in (False, True):
        v, features = L["two_product_values"](shared)
        w_fit, *_ = np.linalg.lstsq(features, v, rcond=None)
        fits[shared] = (float(v.min()), float(v.max()), float(np.abs(features @ w_fit - v).max()))
    lo, hi, gap = fits[True]
    assert (_r(lo, 2), _r(hi, 2), _r(gap, 2)) == (50.91, 75.91, 2.47) and gap < 5  # the fee is 5
    assert fits[False][2] < 1e-12  # "below 1e-12, rounding" (Exercise 1, knowledge check 7.6)
    # ---- Section 3, Check yourself 3.1 and the collapsed argument
    A2 = L["td_matrix"](np.eye(2), np.array([0.5, 0.5]), np.array([[0.0, 1.0], [0.0, 1.0]]), 0.9)
    assert np.allclose(A2, [[0.5, -0.45], [0.0, 0.05]], atol=1e-15)
    lam2 = np.sort(np.linalg.eigvals(A2).real)
    assert np.allclose(lam2, [0.05, 0.5]) and math.isclose(min(2 / lam2), 4.0)  # alpha < 4
    y = np.array([1.0, 2.0])
    assert _r(y @ A2 @ y, 6) == -0.2  # "y = (1, 2) gives y^T A y = -0.2"
    A = L["A"]  # Exercise 3's run cell: Baird's A
    assert np.allclose(A, L["td_matrix"](b.phi, b.mu_behavior, b.P_target, b.gamma))
    sym = np.linalg.eigvalsh((A + A.T) / 2)
    assert _r(sym.min(), 5) == -1.02058 and _r(sym.min(), 2) == -1.02
    # ---- Section 4: the fragment's table (w0 = 10, alpha = 0.1), exact rationals
    table = {0.4: (0.98, 8.17, 1.33), 0.5: (1.00, 10, 10), 0.6: (1.02, 12.19, 72.45)}
    table[0.99] = (1.098, 25.47, 114_877)
    for g, (factor, w10, w100) in table.items():
        f = 1 + Fraction(1, 10) * (2 * Fraction(str(g)) - 1)
        assert float(f) == factor
        assert _r(float(10 * f**10), 2) == w10
        assert round(float(10 * f**100), 0 if w100 > 1000 else 2) == w100
    # ---- Section 4's constants and Section 6's
    assert np.array_equal(b.w0, [1, 1, 1, 1, 1, 1, 10, 1]) and b.gamma == 0.99
    assert L["ALPHA_BAIRD"] == 0.01 and L["SWEEPS"] == 1000
    assert np.array_equal(L["OBS_SCALE"], [2.4, 3.0, 0.2095, 3.5]) and len(L["GRID_POINTS"]) == 625
    assert L["WIDTH"] == 0.5 and (L["ALPHA_Q"], L["GAMMA_Q"]) == (0.3, 0.99)
    assert round(1 / (1 - L["GAMMA_Q"])) == 100
    import gymnasium.envs.classic_control.cartpole as cartpole

    env = cartpole.CartPoleEnv()
    assert env.x_threshold == 2.4  # gymnasium's source; its docstring gives the angle as .2095
    assert "(-.2095, .2095)" in (cartpole.CartPoleEnv.__doc__ or "")
    assert abs(env.theta_threshold_radians - 0.2095) < 1e-4
    # ---- Exercise 2
    w1 = L["w1"]
    phi1, phi7 = b.phi[0], b.phi[6]
    assert math.isclose(0.99 * 12 - 3, 8.88) and np.allclose(w1[[0, 7]], [1.1776, 1.0888])
    assert _r(w1 @ phi1, 4) == 3.444 and _r(w1 @ phi7, 4) == 12.1776
    assert _r(0.99 * (w1 @ phi7) - w1 @ phi1, 4) == 8.6118
    # ---- Exercise 3 and knowledge check 7.2
    assert _r(A[0, 6], 5) == -0.28286 and _r(A[0, 6], 3) == -0.283 and A[6, 0] == 0
    assert _r(np.abs(A - A.T).max(), 4) == 0.8486
    eig = np.sort(np.linalg.eigvals(A).real)
    assert np.allclose(eig, [-0.23925, -0.02218, 0, *[0.57143] * 5], atol=5e-6)
    assert [_r(x, 3) for x in eig[:2]] == [-0.239, -0.022]  # Section 5's fold
    lam_all = np.linalg.eigvals(A)
    assert abs(lam_all[np.argmin(lam_all.real)].imag) < 1e-12  # "the eigenvalue -0.23925"
    assert np.linalg.matrix_rank(b.phi) == 7  # 8 weights, 7 independent feature vectors
    # ---- Exercise 4 and knowledge check 7.3
    norms, path = L["norms"], L["path"]
    assert [_r(norms[k], 3) for k in (0, 10, 100, 1000)] == [10.344, 10.609, 16.826, 436.856]
    w_end = path[-1]
    assert np.all(np.round(w_end[:6], 2) == 114.43) and _r(w_end[7], 2) == 335.00
    assert _r(w_end[6], 2) == 6.85 and path[0][6] == 10
    rho = [np.abs(np.linalg.eigvals(np.eye(8) - a * A)).max() for a in (0.001, 0.01, 0.1)]
    assert [_r(x, 6) for x in rho] == [1.000239, 1.002393, 1.023925]
    # Exercise 4's facilitator note: 1.002393^1000 = 10.9 predicts about 113, but ‖w‖ grows
    # 42-fold; w0's coefficient along the growing unit eigenvector is 44.9.
    assert _r(rho[1] ** 1000, 1) == 10.9 and round(np.linalg.norm(b.w0) * rho[1] ** 1000) == 113
    assert round(norms[1000] / norms[0]) == 42
    lam_A, vec_A = np.linalg.eig(A)
    vec_A = vec_A / np.linalg.norm(vec_A, axis=0)
    coef = np.linalg.solve(vec_A, b.w0.astype(complex))[np.argmin(lam_A.real)]
    assert _r(abs(coef), 1) == 44.9 and abs(coef) > np.linalg.norm(b.w0)
    doubling = math.log(2) / math.log(rho[0])
    assert round(doubling, 1) == 2897.5 and round(doubling, -2) == 2900
    # ---- Exercise 5, Section 5's fold and knowledge check 7.4
    GRID, results = L["GRID"], L["results"]

    def err(name: str, k: int) -> float:
        return float(results[name][1][np.searchsorted(GRID, k)])

    assert float(b.phi[6] @ b.phi[6]) == 5 and math.isclose((1 - b.gamma) * 5, 0.05)
    assert b.phi[6] @ b.w0 == 12
    assert np.allclose(results["on-policy"][0][-1], [1, 1, 1, 1, 1, 1, 7.6, -3.8], atol=1e-9)
    assert math.isclose(2 * 1 - 3.8, -1.8)
    v_on = b.phi @ results["on-policy"][0][-1]
    assert np.allclose(v_on[:6], -1.8, atol=1e-9) and abs(v_on[6]) < 1e-9
    G = b.phi.T @ np.diag(b.mu_behavior) @ b.phi
    lam = np.linalg.eigvalsh(G).max()
    assert _r(lam, 3) == 1.857 and _r(2 / lam, 3) == 1.077
    A_on = L["td_matrix"](b.phi, np.eye(7)[6], b.P_target, b.gamma)
    assert np.allclose(A_on, (1 - b.gamma) * np.outer(b.phi[6], b.phi[6]), atol=1e-15)
    assert np.linalg.matrix_rank(A_on) == 1
    mc = (22 / 13) * np.array([1, 1, 1, 1, 1, 1, 4, -2])
    assert [_r(x, 3) for x in mc[[0, 6, 7]]] == [1.692, 6.769, -3.385]
    assert np.allclose(results["monte-carlo"][0][-1], mc, atol=1e-9)
    assert np.allclose(b.phi @ np.array([1, 1, 1, 1, 1, 1, 4, -2]), 0)
    assert _sig(0.01 * (1 - b.gamma) / 7, 3) == "1.43e-05"
    assert _r(err("tabular", 10**4), 1) == 10.4 and _sig(err("tabular", 10**6), 2) == "7.5e-06"
    assert f"{err('monte-carlo', 10**4):.0e}" == "2e-12"
    assert _sig(err("original", 10**4), 2) == "1.6e+12"
    # Each variant converges: on-policy to an error of 1.8 (the upper states), Monte Carlo to
    # 0 (rounding level), tabular still falling.
    assert _r(err("on-policy", 10**5), 6) == _r(err("on-policy", 10**6), 6) == 1.8
    assert err("monte-carlo", 10**5) < 1e-12 and err("monte-carlo", 10**6) < 1e-12
    assert err("tabular", 10**6) < err("tabular", 10**5) < err("tabular", 10**4)
    assert not np.isfinite(results["original"][1][-1]) or err("original", 10**6) > 1e12
    # Figure 07-legs, caption and alt text. "All four start near 12":
    assert all(round(float(results[n][1][0])) == 12 for n in results)
    # The original passes 1e12 at 1e4 sweeps and leaves the top of the plot (1e15) soon after:
    over = np.nan_to_num(results["original"][1], nan=np.inf) > LEG_YLIM[1]
    assert err("original", 10**4) > 1e12 and 10**4 < GRID[np.argmax(over)] < 2 * 10**4
    # Monte Carlo falls off the bottom of the plot (1e-8) before 1e4 sweeps:
    assert GRID[np.argmax(results["monte-carlo"][1] < LEG_YLIM[0])] < 10**4
    # On-policy: within 1% of 1.8 from about 1e4 sweeps on ("near 1e4 sweeps but at an error of
    # 1.8"). The alt text's "falls to a flat line at 1.8 after about five thousand sweeps" is
    # loose: the error dips to 1.41 near 5000 sweeps and is within 1% of 1.8 from 12,589 on.
    e_on = results["on-policy"][1]
    flat = np.abs(e_on - 1.8) < 0.018
    settle = next(i for i in range(len(GRID)) if flat[i:].all())
    assert GRID[settle] == 12_589 and _r(e_on.min(), 2) == 1.41 and GRID[e_on.argmin()] == 5012
    # Tabular stays near 12 until about 1e4 sweeps (10.4 there), then 7.5e-6 at 1e6. The alt
    # text's "about ten to the minus six" is loose: 7.5e-6 is between 1e-6 and 1e-5.
    assert err("tabular", 10**3) > 11.5 and err("tabular", 10**4) > 10
    assert 1e-6 < err("tabular", 10**6) < 1e-5
    # ---- Exercise 7 and knowledge check 7.8
    raw = L["colleague_features_unfixed"](np.zeros(4))
    sq = float(raw @ raw)
    assert _r(sq, 3) == 9.868 and _r(0.3 * sq, 3) == 2.960 and _r(1 - 0.3 * sq, 2) == -1.96
    assert _r(2 / sq, 2) == 0.20 and _r(0.3 / sq, 2) == 0.03
    assert np.allclose(L["rbf_features"](np.zeros(4)), raw / np.sqrt(sq))
    # ---- knowledge check 7.1
    args = (np.array([2.0, 4.0]), np.array([1.0, 0.0]), 1.0, np.array([0.0, 1.0]), 0.5, 0.1)
    assert np.allclose(L["semi_gradient_td0"](*args), [2.1, 4.0])
    w, phi, r, phi_n, g, a = args
    delta = r + g * w @ phi_n - w @ phi
    assert delta == 1 and np.allclose(w - a * delta * (g * phi_n - phi), [2.1, 3.95])
    # ---- knowledge check 7.7
    W = np.array([[1.0, 0, 2], [0, 1, 1]])
    phi, phi_n = np.array([1.0, 1, 0]), np.array([0.0, 1, 1])
    assert np.array_equal(W @ phi_n, [2, 2]) and W[0] @ phi == 1
    for term, trunc, row0 in ((0, 0, [1.18, 0.18, 2]), (1, 0, [1, 0, 2]), (0, 1, [1.18, 0.18, 2])):
        out = L["linear_q_update"](W, phi, 0, 1.0, phi_n, bool(term), bool(trunc), 0.9, 0.1)
        assert np.allclose(out[0], row0) and np.array_equal(out[1], W[1])
    assert math.isclose(1 + 0.9 * 2 - 1, 1.8)
    return {"soft": soft, "legs": {"grid": GRID, "err": {k: v[1] for k, v in results.items()}}}


# ------------------------------------------------------------------ numbers from the records
def record_numbers(recs: dict[str, dict]) -> dict:
    """The record-based claims of the page, the predict_answer entries and the knowledge checks.

    Returns the values behind each claim, for the report."""
    out: dict = {}
    for label, rec in recs.items():
        p = rec["per_seed"]
        greedy, rand = np.array(p["greedy_return"]), np.array(p["random_return"])
        w_norm = np.array(p["w_norm"])
        maxw = np.array(p["max_w_norm"])
        col = p["colleague"]
        cmax = np.array([np.nan if v is None else v for v in col["max_finite_w_norm"]])
        cgreedy = np.array(col["greedy_return"])
        # Exercise 6: "random play ... about 20 steps"; "far above 3 x random"
        assert 15 <= np.median(rand) <= 25, np.median(rand)
        assert np.all(greedy >= 3 * rand), (greedy / rand).min()
        # "‖W‖ rises and then levels off in the hundreds": the largest ‖W‖ of every seed is in
        # the hundreds, and over the last third of the run it grows by at most 25%.
        third = w_norm.shape[1] * 2 // 3
        growth = w_norm[:, -1] / w_norm[:, third - 1]
        assert np.all((100 <= maxw) & (maxw < 1000)), maxw
        assert np.all(growth <= 1.25), growth.max()
        assert maxw.max() < checks.NORM_BOUND / 10  # the training checks' bound
        # Exercise 7, knowledge check 7.8: ‖W‖ grows by many orders of magnitude with the
        # unfixed features, Q at the start state is far above 100, and the greedy return
        # stays high on some seeds anyway.
        assert np.all(np.isnan(cmax) | (cmax > 1e20)), np.nanmin(cmax)
        assert all(q is None or min(abs(x) for x in q) > 1000 for q in col["q_start"])
        high = int(np.sum(cgreedy >= 3 * np.median(rand)))
        out[label] = {
            "seeds": len(greedy),
            "greedy min/median/max": (greedy.min(), np.median(greedy), greedy.max()),
            "random min/median/max": (rand.min(), np.median(rand), rand.max()),
            "greedy / random, smallest": float((greedy / rand).min()),
            "max ‖W‖ min/max over seeds": (maxw.min(), maxw.max()),
            "‖W‖ at the end min/max": (w_norm[:, -1].min(), w_norm[:, -1].max()),
            "‖W‖ growth over the last third, max": float(growth.max()),
            "colleague: log10 max finite ‖W‖ min/max": (
                float(np.log10(np.nanmin(cmax))),
                float(np.log10(np.nanmax(cmax))),
            ),
            "colleague: seeds with a non-finite ‖W‖": int(
                sum(any(v is None for v in row) for row in col["w_norm"])
            ),
            "colleague: greedy min/median/max": (cgreedy.min(), np.median(cgreedy), cgreedy.max()),
            "colleague: seeds with greedy >= 3 x random's median": high,
        }
    assert out["live"]["colleague: seeds with greedy >= 3 x random's median"] >= 1
    quoted_record_values(recs)
    return out


def quoted_record_values(recs: dict[str, dict]) -> None:
    """The record values the second pass quotes: Section 6's After-the-lab fold, Figure
    07-weight-norms' caption and alt text, the predict_answer entries of Exercises 6 and 7, and
    knowledge check 7.8. Seeds 0-19 of each band record."""

    def r1(xs) -> tuple:
        """(min, median, max) to one decimal; a median exactly on a half keeps two decimals."""
        xs = np.asarray(xs, dtype=float)
        med = float(np.median(xs))
        tie = abs(med * 10 - math.floor(med * 10) - 0.5) < 1e-9
        return (_r(xs.min(), 1), round(med, 2) if tie else _r(med, 1), _r(xs.max(), 1))

    for label, want in {
        "live": {"greedy": (206.6, 493.5, 500.0), "peak": (327, 420), "end": (263, 312)},
        # The QUICK median is 348.45 exactly (the mean of 332.4 and 364.5); the predict_answer
        # writes 348.4, where half-up rounding gives 348.5.
        "quick": {"greedy": (214.9, 348.45, 500.0), "peak": (172, 193), "end": (169, 184)},
    }.items():
        p = recs[label]["per_seed"]
        assert r1(p["greedy_return"]) == want["greedy"], r1(p["greedy_return"])
        assert r1(p["random_return"]) == (12.2, 22.0, 29.0)
        maxw = np.array(p["max_w_norm"])
        assert (round(maxw.min()), round(maxw.max())) == want["peak"]
        end = np.array([row[-1] for row in p["w_norm"]])
        assert (round(end.min()), round(end.max())) == want["end"]
    p = recs["live"]["per_seed"]
    ratio = np.array(p["greedy_return"]) / np.array(p["random_return"])
    # The page and Exercise 6's predict_answer say "at least 8.1 times random on every seed";
    # the smallest ratio is 8.063, so "at least 8" is what holds.
    assert _r(ratio.min(), 2) == 8.06 and ratio.min() > 8
    q = np.array(p["q_start"])
    assert (_r(q.min(), 1), _r(q.max(), 1)) == (74.4, 110.7)
    c = p["colleague"]
    first = np.log10([row[0] for row in c["w_norm"]])
    last = np.log10([row[-1] for row in c["w_norm"]])
    assert first.min() > 16 and round(first.min()) == 17 and round(first.max()) == 36
    assert 85 < last.min() and last.max() < 116  # "ending between about 10^85 and 10^116"
    assert all(row[-1] > row[9] for row in c["w_norm"])  # keeps climbing after 10,000 steps
    assert all(v is not None for row in c["w_norm"] for v in row)  # never inf or nan
    top = np.log10(c["max_finite_w_norm"])
    assert (_r(top.min(), 1), _r(top.max(), 1)) == (85.6, 115.4)
    assert all(q is not None and min(abs(x) for x in q) > 1000 for q in c["q_start"])
    assert r1(c["greedy_return"]) == (9.2, 14.4, 459.3)
    above = [s for s, g in zip(recs["live"]["seeds"], c["greedy_return"], strict=True) if g > 160]
    assert above == [3, 4, 12]  # "above 160 ... on seeds 3, 4 and 12" (3 of 20)
    cq = recs["quick"]["per_seed"]["colleague"]
    top_q = np.log10(cq["max_finite_w_norm"])
    assert (_r(top_q.min(), 1), _r(top_q.max(), 1)) == (32.1, 52.8)


# ------------------------------------------------------------------ figures
LEG_STYLE = {  # name: (series index, dash, label)
    "original": (0, "-", "original (all three legs)"),
    "on-policy": (1, "--", "on-policy weights"),
    "monte-carlo": (2, "-.", "Monte Carlo targets"),
    "tabular": (3, ":", "tabular features"),
}
LEG_FLOOR, LEG_CEIL = 1e-9, 1e16  # values outside the axes are clipped here, off the plot
LEG_YLIM = (1e-8, 1e15)


def _log_round(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to [lo, hi] (inf counts as hi), then round in log space: platform-proof drawing."""
    y = np.where(np.isfinite(y), y, hi)
    return 10 ** np.round(np.log10(np.clip(y, lo, hi)), 3)


def legs_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.2))
    ax = fig.add_axes((0.12, 0.14, 0.85, 0.82))
    grid = data["grid"]
    for name, (i, dash, label) in LEG_STYLE.items():
        y = _log_round(np.asarray(data["err"][name], dtype=float), LEG_FLOOR, LEG_CEIL)
        ax.plot(grid, y, color=t.series[i], lw=2.2, ls=dash, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e6)
    ax.set_ylim(*LEG_YLIM)
    ax.xaxis.set_major_locator(FixedLocator([1, 1e2, 1e4, 1e6]))
    ax.yaxis.set_major_locator(FixedLocator([10.0**e for e in range(-6, 16, 3)]))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(_pow10))
    ax.grid(True, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("sweeps (log scale)")
    ax.set_ylabel(r"largest $|\hat{v}(s) - v_\pi(s)|$ (log scale)")
    ax.legend(
        loc="upper left",
        facecolor=t.bg,  # opaque, so gridlines do not run through the text
        framealpha=1,
        edgecolor="none",
        handlelength=2.8,
        title="Baird, α = 0.01",
        alignment="left",
    )
    return fig


def _pow10(v: float, _p=None) -> str:
    e = int(round(math.log10(v)))
    if e == 0:
        return "1"
    return f"$10^{{{e}}}$".replace("-", "\N{MINUS SIGN}")


NORM_YLIM = (1, 1e125)


def _steps(v: float, _p=None) -> str:
    return f"{int(v):,}"


def norms_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.6))
    ax = fig.add_axes((0.12, 0.13, 0.82, 0.84))
    grid = np.array(data["grid"])
    solution = [np.round(np.array(row, dtype=float), 1) for row in data["solution"]]
    for y in solution:
        ax.plot(grid, _log_round(y, 1, 1e130), color=t.series[0], lw=1.2, alpha=0.8)
    for row in data["colleague"]:
        y = np.array([np.nan if v is None else v for v in row], dtype=float)
        ok = np.isfinite(y)
        ax.plot(
            grid[ok],
            _log_round(y[ok], 1, 1e130),
            color=t.series[1],
            lw=1.2,
            ls="--",
            alpha=0.8,
        )
    ax.axhline(checks.NORM_BOUND, color=t.neutral, lw=1.4, ls=":", zorder=0)
    ax.text(
        grid[-1],
        checks.NORM_BOUND * 30,
        "10⁴: the training checkpoints' bound",
        ha="right",
        va="bottom",
        color=t.text,
        fontsize=MIN_FONT,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(*NORM_YLIM)
    ax.yaxis.set_major_locator(FixedLocator([1, 1e30, 1e60, 1e90, 1e120]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
    ax.xaxis.set_major_locator(FixedLocator(list(range(0, grid[-1] + 1, 10_000))))
    ax.xaxis.set_major_formatter(FuncFormatter(_steps))
    ax.grid(True, axis="y", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(r"$\Vert W \Vert$ (log scale)")
    n = len(data["solution"])
    handles = [
        Line2D([], [], color=t.series[1], lw=2, ls="--"),
        Line2D([], [], color=t.series[0], lw=2),
    ]
    ax.legend(
        handles,
        ["colleague's (not scaled)", "rbf_features (unit length)"],
        loc="lower left",
        bbox_to_anchor=(0.10, 0.12),
        frameon=False,
        handlelength=2.8,
        title=f"{n} seeds each, α = 0.3",
        alignment="left",
    )
    # The unit-length runs on a linear scale: on the log axis they merge into one line.
    ins = ax.inset_axes((0.70, 0.17, 0.28, 0.3))
    for y in solution:
        ins.plot(grid, y, color=t.series[0], lw=1.0, alpha=0.8)
    ins.set_xlim(0, grid[-1])
    ins.set_ylim(0, 500)
    ins.xaxis.set_major_locator(FixedLocator([0, grid[-1]]))
    ins.xaxis.set_major_formatter(FuncFormatter(_steps))
    ins.yaxis.set_major_locator(FixedLocator([0, 250, 500]))
    ins.grid(True, axis="y", lw=0.8)
    ins.set_axisbelow(True)
    ins.set_facecolor(t.bg)
    for side in ("top", "right"):
        ins.spines[side].set_visible(False)
    ins.set_title("unit length, linear scale", fontsize=MIN_FONT, color=t.text, pad=4)
    return fig


FIGURES = {"07-legs": ("legs", legs_figure), "07-weight-norms": ("norms", norms_figure)}


def build() -> tuple[dict[Path, bytes], dict]:
    mpl.use("Agg", force=True)
    L = lab_namespace(quick=False)
    exact = exact_numbers(L)
    recs = {label: newest_record(name)[1] for label, name in EXPERIMENTS.items()}
    from_records = record_numbers(recs)
    live = recs["live"]
    computed = {
        "legs": exact["legs"],
        "norms": {
            "grid": live["budget"]["grid"],
            "solution": live["per_seed"]["w_norm"],
            "colleague": live["per_seed"]["colleague"]["w_norm"],
        },
        "soft": exact["soft"],
        "records": from_records,
    }
    out: dict[Path, bytes] = {}
    for name, (key, draw) in FIGURES.items():
        for t in themes():
            with mpl.rc_context(rc(name, t)):
                out[IMAGES / f"{name}{t.suffix}.svg"] = svg_bytes(draw(t, computed[key]))
    return out, computed


#: Quoted phrases that do not match the computation or record as written; the assertions
#: check the true values. For the academic director (the page is theirs).
FOR_THE_DIRECTOR = [
    "Exercise 6's predict_answer and Section 6's fold: 'at least 8.1 times random on every "
    "seed'; the smallest ratio is 8.063 (live record), so 'at least 8 times'.",
    "Exercise 6's predict_answer: QUICK median '348.4' is 348.45 exactly (half-up: 348.5).",
    "Figure 07-legs alt text: on-policy 'falls to a flat line at 1.8 after about five thousand "
    "sweeps'; it dips to 1.41 near 5,000 sweeps and is within 1% of 1.8 from about 12,600.",
    "Figure 07-legs alt text: tabular 'about ten to the minus six at one million'; it is "
    "7.5e-6, between 1e-6 and 1e-5.",
]


def report(computed: dict) -> None:
    print("Every quoted number is asserted (exact_numbers, record_numbers). Phrases that differ:")
    for line in FOR_THE_DIRECTOR:
        print(f"  - {line}")
    for what, value in computed["soft"].items():
        print(f"  platform-dependent: {what} = {value:.3g} on this machine")
    for label, rows in computed["records"].items():
        print(f"From {newest_record(EXPERIMENTS[label])[0].name}:")
        for key, value in rows.items():
            shown = (
                " / ".join(f"{v:.4g}" for v in value)
                if isinstance(value, tuple)
                else f"{value:.4g}"
            )
            print(f"  {key}: {shown}")


def main(argv: list[str]) -> int:
    only = [argv[i + 1] for i, a in enumerate(argv) if a == "--only" and i + 1 < len(argv)]
    if "--run" in argv:
        for p in run(only=only or None):
            print(f"Saved {p.name} in runs/inbox/ (file it with scripts/add_run_record.py).")
        return 0
    if "--reproduce" in argv:
        return 0 if reproduce(only or None) else 1
    files, computed = build()
    if "--check" in argv:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m07.py")
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
