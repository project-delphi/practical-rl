"""Module 2 figures: deterministic light and dark SVGs (PLAN.md §8).

  uv run python scripts/make_figures_m02.py           # write images/02-*.svg and print the numbers
  uv run python scripts/make_figures_m02.py --check   # fail if a committed SVG differs from a fresh build

Same conventions as scripts/make_figures_m01.py, whose theme, rcParams and SVG writer
this script reuses: a light and a dark copy of each figure, colors from prl._palette and
the brand tokens, 510 pt wide, text drawn as paths, no date in the metadata, computed
values rounded before plotting.

  02-vi-vs-pi       Value iteration on the shop: max|V_k - V_(k-1)| per sweep (log scale)
                    against tol, with the sweep from which the greedy policy of Q_k stays
                    pi*; policy iteration from "never order": ||V^(pi_k) - V*|| per evaluation.
  02-model-error    Plan in a model estimated from N samples per (stock, order), evaluate in
                    the true shop: the true loss of every seed (seeds 0-4), their mean, the
                    rescaled bound of eq-plan-loss; and eps_P against a 1/sqrt(N) reference.

The sampler, the seeds and the solver code are the lab's (reproduced in
scripts/make_expected.py, which this script imports), so the numbers equal the lab's.
Every action choice uses make_expected's _greedy (the first action within a tolerance
of the row's max), so ties that rounding splits differently on another machine are broken
the same way everywhere. The script asserts every count and number the page's captions and the m02 Predict
answers quote, the figures' and the rest of the page's (page_numbers, run by --check too). The numbers are exact computations on a known model, not run
measurements. This script may contain solver code; prl may not.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_expected import (  # noqa: E402
    M01_INVENTORY,
    _estimate_model,
    _exact_value,
    _greedy,
    _optimal,
    _policy_iteration,
    _random_log,
    _sample_transitions,
    _tie_tol,
    _value_iteration,
)
from make_figures_m01 import MIN_FONT, WIDTH_IN, Theme, rc, svg_bytes, themes  # noqa: E402
from prl.envs import CliffGridworld, InventoryMDP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"

GAMMA = 0.95
TOL = 1e-6
NS = (1, 2, 5, 10, 20, 50, 100, 200)
SEEDS = (0, 1, 2, 3, 4)


def _sig(x: float, digits: int = 4) -> float:
    """Round to significant digits, so the SVG does not depend on last-bit differences."""
    return float(f"{x:.{digits}g}")


def _count(v: float, _pos: int | None = None) -> str:
    return f"{v:,.0f}"


def _greedy_is_optimal(pi: np.ndarray, Q_star: np.ndarray) -> bool:
    """pi picks a maximizer of Q* everywhere (the shop's capped orders tie exactly)."""
    S = Q_star.shape[0]
    return bool(np.all(Q_star[np.arange(S), pi] >= Q_star.max(axis=1) - 1e-9))


# ------------------------------------------------------------------ figure 1: VI against PI
def vi_pi_data() -> dict:
    mdp = InventoryMDP(**M01_INVENTORY)
    P, R = mdp.P, mdp.R
    V_star, pi_star, Q_star, _ = _optimal(P, R, GAMMA, TOL)
    _, _, k_vi, log = _value_iteration(P, R, GAMMA, TOL)
    assert k_vi == 299, f"value iteration takes {k_vi} sweeps, not 299"
    assert [round(x, 2) for x in log[:5]] == [15.03, 12.76, 5.72, 4.03, 3.59], log[:5]
    ratios = np.array(log[20:]) / np.array(log[19:-1])
    assert abs(float(np.median(ratios)) - GAMMA) < 0.01, "the change should fall by about gamma"

    # The greedy policy of Q_k = R + gamma P V_k (the Q value_iteration returns after k sweeps).
    V, optimal_at = np.zeros(mdp.n_states), []
    for _ in range(k_vi):
        V = (R + GAMMA * P @ V).max(axis=1)
        optimal_at.append(_greedy_is_optimal(_greedy(R + GAMMA * P @ V), Q_star))
    first_for_good = next(k for k in range(1, k_vi + 1) if all(optimal_at[k - 1 :]))
    first_ever = optimal_at.index(True) + 1
    assert first_for_good == 4, f"greedy(Q_k) is pi* for every k >= {first_for_good}, not 4"
    assert first_ever == 2 and not optimal_at[2], "greedy(Q_2) is pi*, greedy(Q_3) is not"
    V_k = np.zeros(mdp.n_states)
    for _ in range(first_for_good):
        V_k = (R + GAMMA * P @ V_k).max(axis=1)
    err_at_mark = float(np.max(np.abs(V_k - V_star)))
    assert round(err_at_mark) == 68, err_at_mark

    # Policy iteration from "never order": the policy of each evaluation and its distance to V*.
    S = mdp.n_states
    pi, dists, policies = np.zeros(S, dtype=int), [], []
    while True:
        V_pi = _exact_value(P, R, GAMMA, pi)
        dists.append(float(np.max(np.abs(V_pi - V_star))))
        policies.append(pi.copy())
        Q = R + GAMMA * P @ V_pi
        keep = Q[np.arange(S), pi] >= Q.max(axis=1) - _tie_tol(Q)
        pi_new = np.where(keep, pi, _greedy(Q))
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    k_pi = _policy_iteration(P, R, GAMMA, np.zeros(S, dtype=int))[2]
    assert len(dists) == k_pi == 4, f"policy iteration takes {k_pi} evaluations, not 4"
    assert round(dists[0], 2) == 159.75 and dists[-1] < 1e-9, dists
    return {
        "log": [_sig(x) for x in log],
        "k_vi": k_vi,
        "mark": first_for_good,
        "first_ever": first_ever,
        "err_at_mark": err_at_mark,
        "median_ratio": float(np.median(ratios)),
        "dists": [round(d, 4) if d > 1e-9 else 0.0 for d in dists],
        "policies": policies,
        "pi_star": pi_star,
    }


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def vi_pi_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 3.9))  # 680 x 374 px, aspect 1.82
    left = fig.add_axes((0.105, 0.155, 0.505, 0.70))
    right = fig.add_axes((0.735, 0.155, 0.25, 0.70))

    log = data["log"]
    k = np.arange(1, len(log) + 1)
    left.plot(k, log, color=t.series[0], lw=2.0, zorder=3)
    left.axhline(TOL, color=t.neutral, lw=1.6, ls=(0, (5, 2.5)), zorder=2)
    left.text(150, TOL * 1.6, r"tol $= 10^{-6}$", ha="center", va="bottom", color=t.text)
    mark = data["mark"]
    left.axvline(mark, color=t.series[1], lw=1.8, ls=(0, (1.5, 1.5)), zorder=2)
    left.annotate(
        f"greedy policy $= \\pi^*$\nfrom sweep {mark} on",
        xy=(mark, 3.0),
        xytext=(mark + 22, 3.0),
        ha="left",
        va="center",
        color=t.text,
        arrowprops={"arrowstyle": "-|>", "color": t.series[1], "lw": 1.4, "shrinkA": 2},
    )
    left.annotate(
        f"stops at sweep {data['k_vi']}",
        xy=(data["k_vi"], log[-1]),
        xytext=(-4, -9),
        textcoords="offset points",
        ha="right",
        va="top",
        color=t.text,
    )
    left.set_yscale("log")
    left.set_ylim(3e-8, 60)
    left.set_xlim(0, 305)
    left.xaxis.set_major_locator(FixedLocator([1, 50, 100, 150, 200, 250, 299]))
    left.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    left.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"$10^{{{int(round(math.log10(v)))}}}$")
    )
    left.yaxis.set_minor_formatter(NullFormatter())
    left.grid(True, which="major", axis="y", lw=0.8)
    _style_axes(left)
    left.set_xlabel("Sweep $k$")
    left.set_ylabel(r"$\|V_k - V_{k-1}\|_\infty$ (log scale)")
    left.set_title("Value iteration", loc="left", fontsize=13, color=t.text)

    d = data["dists"]
    ev = np.arange(1, len(d) + 1)
    right.plot(ev, d, color=t.series[0], lw=2.0, marker="o", ms=8, mec=t.bg, mew=1.2, zorder=3)
    for x, y in zip(ev, d, strict=True):
        label = "0" if y == 0 else (f"{y:.2f}" if y < 1 else f"{y:.1f}")
        right.annotate(
            label,
            (x, y),
            xytext=(7, 4) if y > 1 else (0, 10),
            textcoords="offset points",
            ha="left" if y > 1 else "center",
            va="bottom",
            color=t.text,
        )
    right.set_xlim(0.6, len(d) + 0.4)
    right.set_ylim(-8, 185)
    right.xaxis.set_major_locator(FixedLocator(list(ev)))
    right.yaxis.set_major_locator(FixedLocator([0, 50, 100, 150]))
    right.grid(True, which="major", axis="y", lw=0.8)
    _style_axes(right)
    right.set_xlabel("Evaluation $k$")
    right.set_ylabel(r"$\|V^{\pi_k} - V^*\|_\infty$")
    right.set_title("Policy iteration", loc="left", fontsize=13, color=t.text)
    return fig


# ------------------------------------------------------------------ figure 2: model error
def model_error_data() -> dict:
    mdp = InventoryMDP(**M01_INVENTORY)
    P, R = mdp.P, mdp.R
    S, A = mdp.n_states, mdp.n_actions
    _, pi_star, _, _ = _optimal(P, R, GAMMA, TOL)
    V_star = _exact_value(P, R, GAMMA, pi_star)
    loss = np.zeros((len(SEEDS), len(NS)))
    eps_P = np.zeros_like(loss)
    eps_R = np.zeros_like(loss)
    for j, N in enumerate(NS):
        for i, seed in enumerate(SEEDS):
            log = _sample_transitions(mdp, N, np.random.default_rng(seed))
            P_hat, R_hat, _ = _estimate_model(log, S, A)
            _, Q_hat, _, _ = _value_iteration(P_hat, R_hat, GAMMA, TOL)
            V_plan = _exact_value(P, R, GAMMA, _greedy(Q_hat))
            loss[i, j] = float(np.max(V_star - V_plan))
            eps_P[i, j] = float(np.abs(P - P_hat).sum(axis=2).max())
            eps_R[i, j] = float(np.abs(R - R_hat).max())
    r_range = float(R.max() - R.min())
    factor = 2 * GAMMA * r_range / (1 - GAMMA) ** 2
    bound = 2 * eps_R.mean(axis=0) / (1 - GAMMA) + factor * eps_P.mean(axis=0)

    # What the page and _variables.yml quote.
    n1 = loss[:, NS.index(1)]
    assert round(float(n1.min())) == 25 and round(float(n1.max())) == 59, n1
    zeros = (loss == 0.0).sum(axis=0)
    for j, N in enumerate(NS):
        if N >= 50:
            assert zeros[j] >= 3, f"N = {N}: only {zeros[j]} of 5 seeds lose exactly 0"
    assert zeros[NS.index(20)] < 3, "the page says the zeros start at N = 50"
    assert bound.min() >= 1_000, f"the rescaled bound drops to {bound.min():.0f}"
    assert round(factor, -2) == 19_200, factor
    big = [j for j, N in enumerate(NS) if N >= 10]
    slope = float(np.polyfit(np.log([NS[j] for j in big]), np.log(eps_P.mean(axis=0)[big]), 1)[0])
    assert -0.6 < slope < -0.4, f"eps_P falls like N^{slope:.2f}, not 1/sqrt(N)"
    assert abs(float(eps_P[:, NS.index(2)].mean()) - 1.869) < 5e-4
    return {
        "loss": np.round(loss, 4),
        "eps_P": np.round(eps_P, 6),
        "eps_R": np.round(eps_R, 4),
        "bound": np.round(bound, 1),
        "factor": factor,
        "r_range": r_range,
        "zeros": zeros,
        "slope": slope,
    }


SYMLOG_TICKS = [0, 1, 10, 100, 1_000, 10_000, 100_000]


def model_error_figure(t: Theme, data: dict) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.0))  # 680 x 384 px, aspect 1.77
    left = fig.add_axes((0.175, 0.145, 0.455, 0.76))
    right = fig.add_axes((0.775, 0.145, 0.215, 0.76))
    Ns = np.array(NS, dtype=float)
    loss = data["loss"]

    # Every seed, dodged sideways on the log x-axis so equal values (the zeros) stay visible.
    dodge = 10.0 ** (np.linspace(-0.06, 0.06, len(SEEDS)))
    for i in range(len(SEEDS)):
        left.plot(
            Ns * dodge[i],
            loss[i],
            ls="none",
            marker="o",
            ms=6.5,
            mfc="none",
            mec=t.series[0],
            mew=1.4,
            zorder=3,
        )
    left.plot(
        Ns,
        loss.mean(axis=0),
        color=t.series[0],
        lw=2.2,
        marker="D",
        ms=6,
        mec=t.bg,
        mew=1.0,
        zorder=4,
    )
    left.plot(Ns, data["bound"], color=t.neutral, lw=2.0, ls=(0, (5, 2.5)), zorder=2)
    left.text(1.0, 1.0e5, "Bound of the planning loss, rescaled", color=t.text, va="center")
    left.set_xscale("log")
    left.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    left.set_ylim(-0.25, 4e5)
    left.set_xlim(0.75, 270)
    left.xaxis.set_major_locator(FixedLocator(list(NS)))
    left.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    left.xaxis.set_minor_formatter(NullFormatter())
    left.yaxis.set_major_locator(FixedLocator(SYMLOG_TICKS))
    left.yaxis.set_major_formatter(FuncFormatter(_count))
    left.yaxis.set_minor_locator(FixedLocator([]))
    left.grid(True, which="major", axis="y", lw=0.8)
    _style_axes(left)
    left.set_xlabel("Samples per (stock, order), $N$")
    left.set_ylabel(r"$\max_s\, V^*(s) - V^{\hat\pi}(s)$")
    left.set_title("True loss of the planned policy", loc="left", fontsize=13, color=t.text)

    eps = data["eps_P"].mean(axis=0)
    right.plot(Ns, eps, color=t.series[1], lw=2.0, marker="o", ms=7, mec=t.bg, mew=1.0, zorder=3)
    ref = eps[-1] * np.sqrt(Ns[-1] / Ns)
    right.plot(Ns, ref, color=t.neutral, lw=1.8, ls=(0, (5, 2.5)), zorder=2)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(0.75, 270)
    right.set_ylim(0.2, 5.5)
    right.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    right.xaxis.set_minor_formatter(NullFormatter())
    right.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    right.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    right.yaxis.set_minor_formatter(NullFormatter())
    right.grid(True, which="major", axis="y", lw=0.8)
    _style_axes(right)
    right.set_xlabel("$N$")
    right.set_ylabel(r"$\varepsilon_P$, mean of 5 seeds")
    right.set_title("Model error", loc="left", fontsize=13, color=t.text)
    right.text(1.6, 4.2, r"$1/\sqrt{N}$", color=t.text, va="center")
    right.text(1.0, 1.35, r"$\varepsilon_P$", color=t.text, va="center")

    marker = {"ls": "none", "marker": "o", "ms": 6.5, "mfc": "none", "mec": t.series[0]}
    left.legend(
        [
            Line2D([], [], mew=1.4, **marker),
            Line2D([], [], color=t.series[0], lw=2.2, marker="D", ms=6, mec=t.bg),
        ],
        ["One seed (seeds 0 to 4)", "Mean of 5 seeds"],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.70),
        frameon=False,
        handlelength=2.4,
        fontsize=MIN_FONT,
    )
    return fig


# ------------------------------------------------------------------ other numbers the page quotes
def _vi_stop(P, R, gamma, tol, stat):
    """Value iteration from V = 0 that stops when stat(|V_new - V|) < tol; (V, k)."""
    V, k = np.zeros(P.shape[0]), 0
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        if stat(np.abs(V_new - V)) < tol:
            return V_new, k
        V = V_new


def _residual(P, R, gamma, V):
    return float(np.max(np.abs((R + gamma * P @ V).max(axis=1) - V)))


def page_numbers(computed: dict[str, dict]) -> dict:
    """Assert the numbers modules/02-planning.qmd and _variables.yml (m02) quote outside the
    figures, so a change to the model or the samplers cannot leave them stale."""
    mdp = InventoryMDP(**M01_INVENTORY)
    P, R = mdp.P, mdp.R
    S = mdp.n_states
    V_star, pi_star, Q_star, _ = _optimal(P, R, GAMMA, TOL)
    out: dict = {}
    # Section 3: Module 1's "order up to 7" against pi*, and the exact ties in Q*.
    pi7 = np.maximum(0, 7 - np.arange(S))
    V7 = _exact_value(P, R, GAMMA, pi7)
    gap = V_star - V7
    out.update(V7_0=V7[0], gap_0=gap[0], gap_max=gap.max(), gap_argmax=int(gap.argmax()))
    assert round(V7[0], 2) == 38.23 and round(gap[0]) == 42, (V7[0], gap[0])
    assert round(gap.max()) == 45 and gap.argmax() == 6, gap
    out["dV_6_7"] = V_star[7] - V_star[6]
    assert round(out["dV_6_7"], 2) == 2.54, out["dV_6_7"]
    # The ties are exact: capped orders have bitwise equal rows of P and R.
    assert np.array_equal(P[1, 9], P[1, 10]) and R[1, 9] == R[1, 10]
    assert all(np.array_equal(P[10, a], P[10, 0]) and R[10, a] == R[10, 0] for a in range(11))
    deliver = np.minimum(pi_star, mdp.capacity - np.arange(S))
    assert deliver.tolist() == [10, 9, 8, 7, 0, 0, 0, 0, 0, 0, 0], "fill the shelf from 0 to 3"
    # The greedy policy of Q_k: pi* at k = 2, stock 3 stops ordering at k = 3.
    V = np.zeros(S)
    for _ in range(3):
        V = (R + GAMMA * P @ V).max(axis=1)
    assert _greedy(R + GAMMA * P @ V)[3] == 0, "at k = 3, stock 3 stops ordering"
    # Policy iteration: the page's sequence of policies (as deliveries) and 3 evaluations from pi7.
    fill = [
        np.minimum(pol, mdp.capacity - np.arange(S)).tolist()
        for pol in computed["02-vi-vs-pi"]["policies"]
    ]
    want = [[0] * 11] + [[10 - s if s <= top else 0 for s in range(S)] for top in (7, 2, 3)]
    assert fill == want, fill
    assert _policy_iteration(P, R, GAMMA, pi7)[2] == 3
    # Exercise 1's Explain: at tol = 1.0 value iteration stops after 29 sweeps with pi*.
    _, Q1, k1, _ = _value_iteration(P, R, GAMMA, 1.0)
    assert k1 == 29 and _greedy_is_optimal(_greedy(Q1), Q_star), k1
    # Section 2: the greedy-loss guarantee at tol = 1e-3.
    assert round(2 * GAMMA**2 * 1e-3 / (1 - GAMMA) ** 2, 2) == 0.72
    out.update(k_tol1=k1)
    # Exercise 3: missed pairs of random_log(mdp, 200, default_rng(seat)), seats 0-30.
    missed = []
    for seat in range(31):
        log = _random_log(mdp, 200, np.random.default_rng(seat))
        counts = _estimate_model(log, S, mdp.n_actions)[2]
        missed.append(int((counts == 0).sum()))
        if seat == 0:
            per_stock = (counts == 0).sum(axis=1)
            assert missed[0] == 38 and per_stock[9:].sum() == 16, per_stock
    assert min(missed) == 20 and max(missed) == 38, missed
    assert sum(m > 23 for m in missed) == 30, missed
    out.update(missed=missed)
    # Exercise 4: seeds 0-4, the mean falls at every step but one (N = 100 to 200).
    me = computed["02-model-error"]
    means = me["loss"].mean(axis=0)
    assert round(means[NS.index(100)], 2) == 0.08 and round(means[NS.index(200)], 2) == 0.35
    assert me["loss"][:, NS.index(200)].round(2).tolist() == [0.0, 1.39, 0.0, 0.38, 0.0]
    rises = [NS[j + 1] for j in range(len(NS) - 1) if means[j + 1] > means[j]]
    assert rises == [200], rises
    # Exercise 5: the suspect (stop on the mean) against value iteration, and its residual.
    cliff = CliffGridworld(4, 12, slip=0.2, gamma=0.95)
    settled_P = np.zeros((100, 2, 100))
    settled_P[np.arange(100), :, np.arange(100)] = 1.0
    settled_R = np.zeros((100, 2))
    settled_R[0] = [1.0, 0.5]
    models = {
        "shop": (P, R, (299, 299), None),
        "slippery cliff": (cliff.P, cliff.R, (72, 76), 1.9),
        "mostly settled": (settled_P, settled_R, (181, 271), 97.8),
    }
    for name, (P_x, R_x, (k_mean, k_max), ratio) in models.items():
        V_mean, km = _vi_stop(P_x, R_x, GAMMA, TOL, np.mean)
        _, kx = _vi_stop(P_x, R_x, GAMMA, TOL, np.max)
        assert (km, kx) == (k_mean, k_max), (name, km, kx)
        r = _residual(P_x, R_x, GAMMA, V_mean) / (GAMMA * TOL)
        if ratio is None:
            assert r < 1, (name, r)
        else:
            assert round(r, 1) == ratio, (name, r)
        out[f"ex5_{name}"] = (km, kx, r)
    return out


# ------------------------------------------------------------------ build and check
FIGURES = {
    "02-vi-vs-pi": (vi_pi_data, vi_pi_figure),
    "02-model-error": (model_error_data, model_error_figure),
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
    computed["page"] = page_numbers(computed)
    return out, computed


def report(computed: dict[str, dict]) -> None:
    vp = computed["02-vi-vs-pi"]
    print(
        "The shop (Module 1's InventoryMDP), gamma 0.95, tol 1e-6. Exact computations, not timings."
    )
    print(
        f"value iteration: {vp['k_vi']} sweeps; first changes {vp['log'][:5]}; last {vp['log'][-1]:.3g}; "
        f"median ratio per sweep after sweep 20: {vp['median_ratio']:.4f}"
    )
    print(
        f"greedy(Q_k), Q_k = R + gamma P V_k: pi* at k = {vp['first_ever']}, not at k = 3, "
        f"pi* for every k >= {vp['mark']}; ||V_{vp['mark']} - V*|| = {vp['err_at_mark']:.2f}"
    )
    print(f"policy iteration from 'never order': {len(vp['dists'])} evaluations")
    for pol, d in zip(vp["policies"], vp["dists"], strict=True):
        print(f"  policy {pol.tolist()}  ||V^pi - V*|| = {d:g}")
    me = computed["02-model-error"]
    print()
    print(
        f"model error (seeds {SEEDS[0]}-{SEEDS[-1]}): r_max - r_min = {me['r_range']:.3f}, "
        f"factor on eps_P = {me['factor']:,.0f}"
    )
    print(f"{'N':>4} | loss per seed{'':27} | mean   | zeros | eps_P mean | eps_R mean | bound")
    for j, N in enumerate(NS):
        loss = me["loss"][:, j]
        print(
            f"{N:4d} | {np.array2string(loss, precision=2, floatmode='fixed'):40} | {loss.mean():6.2f} | "
            f"{me['zeros'][j]:5d} | {me['eps_P'][:, j].mean():10.3f} | {me['eps_R'][:, j].mean():10.2f} | "
            f"{me['bound'][j]:,.0f}"
        )
    print(f"eps_P against N (N >= 10): slope {me['slope']:.3f} on log-log axes (1/sqrt(N) is -0.5)")
    pg = computed["page"]
    print()
    print(
        f"order up to 7: V(0) = {pg['V7_0']:.2f}, loses {pg['gap_0']:.2f} at stock 0 and at most "
        f"{pg['gap_max']:.2f} (stock {pg['gap_argmax']}); V*(7) - V*(6) = {pg['dV_6_7']:.3f}"
    )
    print(f"tol 1.0: {pg['k_tol1']} sweeps, greedy policy optimal")
    print(f"random_log, 200 days, seats 0-30: missed pairs {pg['missed']}")
    for name in ("shop", "slippery cliff", "mostly settled"):
        km, kx, r = pg[f"ex5_{name}"]
        print(f"suspect on the {name}: {km} sweeps against {kx}; residual {r:.2f} x gamma tol")


def main(argv: list[str]) -> int:
    check = "--check" in argv
    files, computed = build()
    if check:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m02.py")
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
