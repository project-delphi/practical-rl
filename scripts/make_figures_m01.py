"""Module 1 figures: deterministic light and dark SVGs (PLAN.md §8).

  uv run python scripts/make_figures_m01.py           # write images/01-*.svg and print the numbers
  uv run python scripts/make_figures_m01.py --check   # fail if a committed SVG differs from a fresh build

Each figure is drawn twice from the same code: `images/NAME.svg` (light) and
`images/NAME.dark.svg` (dark). filters/figures.lua shows the one that matches the
site's theme. Colors: plot series and the neutral reference color from prl._palette
(generated from brand/*.yml); background, text and lines from the brand tokens. The
background is opaque and equals the page background, so a figure also reads in the
lightbox.

  01-cliff-gridworld      The 4x12 cliff layout from prl.envs.CliffGridworld, with a safe
                          path along the top row and an edge path beside the cliff.
  01-iterations-vs-gamma  Iterative policy evaluation on the lab's inventory MDP under the
                          base-stock policy: iterations until the stopping test passes,
                          against gamma, with a bound that follows from eq-iteration-bound.

Deterministic output: svg.hashsalt is the figure name, no date in the metadata, text
drawn as paths from DejaVu Sans (shipped with matplotlib, so no font lookup), and
computed values rounded before plotting. The numbers are exact computations on known
models, not run measurements. This script may contain solver code (iterative
evaluation); prl may not.
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from prl._palette import DARK, LIGHT, NEUTRAL_DARK, NEUTRAL_LIGHT
from prl.envs import CliffGridworld, InventoryMDP

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"

# Sized for the 680 px text column: SVG lengths are points and 1 pt = 4/3 px, so a
# 680 px wide figure is 510 pt = 7.083 in. Text is at least 12 pt. Aspect <= 2:1.
WIDTH_IN = 680 / 96
MIN_FONT = 12


# ------------------------------------------------------------------ theme
@dataclass(frozen=True)
class Theme:
    name: str
    suffix: str  # "" for light, ".dark" for dark
    bg: str
    text: str
    muted: str
    grid: str  # light rules (border-subtle)
    border: str  # strong outlines (border-strong)
    surface: str
    cliff_fg: str
    cliff_bg: str
    series: tuple[str, ...]
    neutral: str


def themes() -> list[Theme]:
    out = []
    for name, suffix, series, neutral in (
        ("light", "", LIGHT, NEUTRAL_LIGHT),
        ("dark", ".dark", DARK, NEUTRAL_DARK),
    ):
        color = yaml.safe_load((ROOT / "brand" / f"{name}.yml").read_text())["color"]
        pal = color["palette"]
        out.append(
            Theme(
                name=name,
                suffix=suffix,
                bg=color["background"],
                text=pal["text"],
                muted=pal["muted"],
                grid=pal["border-subtle"],
                border=pal["border-strong"],
                surface=pal["surface"],
                cliff_fg=pal["chip-stale-fg"],
                cliff_bg=pal["chip-stale-bg"],
                series=tuple(series),
                neutral=neutral,
            )
        )
    return out


def rc(name: str, t: Theme) -> dict:
    """rcParams for one figure: deterministic SVG, themed text and axes, text >= 12 pt."""
    return {
        "svg.hashsalt": name,
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": MIN_FONT,
        "axes.labelsize": 13,
        "xtick.labelsize": MIN_FONT,
        "ytick.labelsize": MIN_FONT,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
        "text.color": t.text,
        "axes.labelcolor": t.text,
        "axes.edgecolor": t.muted,
        "xtick.color": t.muted,
        "ytick.color": t.muted,
        "xtick.labelcolor": t.text,
        "ytick.labelcolor": t.text,
        "grid.color": t.grid,
        "figure.facecolor": t.bg,
        "axes.facecolor": t.bg,
        "savefig.facecolor": t.bg,
        "hatch.color": t.cliff_fg,
        "hatch.linewidth": 0.8,
        "lines.solid_capstyle": "round",
        "path.simplify": False,
    }


def svg_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    return buf.getvalue()


def _minus(x: float) -> str:
    return f"{x:g}".replace("-", "\N{MINUS SIGN}")


# ------------------------------------------------------------------ figure 1: cliff layout
SAFE_ACTIONS = [0, 0, 0] + [1] * 11 + [2, 2, 2]  # up 3, right 11, down 3
EDGE_ACTIONS = [0] + [1] * 11 + [2]  # up 1, right 11, down 1


def walk(env: CliffGridworld, actions: list[int]) -> tuple[list[tuple[int, int]], float]:
    """Follow deterministic actions from the start through env.P; return cells and total reward."""
    s, total, cells = env.start, 0.0, [env.coords(env.start)]
    for a in actions:
        nxt = np.flatnonzero(env.P[s, a] > 0.5)
        assert nxt.size == 1, "the cliff gridworld without slip is deterministic"
        total += float(env.R[s, a])
        s = int(nxt[0])
        cells.append(env.coords(s))
    assert s == env.goal, f"path ends in {env.coords(s)}, not the goal"
    return cells, total


def cliff_data() -> dict:
    env = CliffGridworld(4, 12)
    safe = walk(env, SAFE_ACTIONS)
    edge = walk(env, EDGE_ACTIONS)
    assert not set(safe[0] + edge[0]) & {env.coords(c) for c in env.cliff}
    return {"env": env, "safe": safe, "edge": edge}


def _corners(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep the cells where the path turns, plus both ends."""
    keep = [cells[0]]
    for prev, cur, nxt in zip(cells, cells[1:], cells[2:], strict=False):
        if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
            keep.append(cur)
    keep.append(cells[-1])
    return keep


def _path_xy(cells, rows: int, dx_start: float, dx_end: float) -> tuple[list[float], list[float]]:
    """Polyline through cell centers (y up). The path leaves the start cell and enters the
    goal cell through their top edges, shifted sideways so the two paths do not overlap."""
    pts = _corners(cells)
    xs = [c + 0.5 for _, c in pts]
    ys = [rows - 1 - r + 0.5 for r, _ in pts]
    xs[0] += dx_start
    xs[1] += dx_start
    xs[-1] += dx_end
    xs[-2] += dx_end
    ys[0] = 0.86  # top of the start cell, above the letter S
    ys[-1] = 0.80  # arrow tip in the goal cell, above the letter G
    return xs, ys


def cliff_figure(t: Theme, data: dict) -> Figure:
    env: CliffGridworld = data["env"]
    rows, cols = env.rows, env.cols
    fig = Figure(figsize=(WIDTH_IN, 3.6))  # 680 x 346 px, aspect 1.97
    ax = fig.add_axes((0.012, 0.30, 0.976, 0.68))
    ax.set_xlim(-0.04, cols + 0.04)
    ax.set_ylim(-0.04, rows + 0.04)
    ax.set_aspect("equal")
    ax.set_axis_off()

    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (c, rows - 1 - r), 1, 1, facecolor=t.surface, edgecolor=t.grid, linewidth=1.0
                )
            )
    # The cliff: shaded and hatched, and labeled in words (color is not the only cue).
    ax.add_patch(
        Rectangle(
            (1, 0),
            cols - 2,
            1,
            facecolor=t.cliff_bg,
            edgecolor=t.cliff_fg,
            linewidth=1.5,
            hatch="//",
        )
    )
    ax.text(
        cols / 2,
        0.5,
        "The cliff: reward \N{MINUS SIGN}100, back to S",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=t.cliff_fg,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": t.cliff_bg, "edgecolor": "none"},
    )
    for c, letter in ((0, "S"), (cols - 1, "G")):
        ax.add_patch(
            Rectangle((c, 0), 1, 1, facecolor=t.surface, edgecolor=t.border, linewidth=2.0)
        )
        ax.text(
            c + 0.5,
            0.36,
            letter,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=t.text,
        )

    # Two paths whose color AND line style differ; the legend names them.
    paths = [
        (*data["safe"], -0.17, +0.17, t.series[0], (0, (3.2, 1.6)), "Safe path along the top row"),
        (*data["edge"], +0.17, -0.17, t.series[1], "solid", "Edge path beside the cliff"),
    ]
    handles, labels = [], []
    for cells, ret, dx0, dx1, color, style, label in paths:
        xs, ys = _path_xy(cells, rows, dx0, dx1)
        # the line stops short of the tip; the (solid) arrowhead finishes it
        ax.plot(xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1] + 0.12], color=color, lw=3.0, ls=style)
        ax.annotate(
            "",
            xy=(xs[-1], ys[-1]),
            xytext=(xs[-1], ys[-1] + 0.3),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.55,head_width=0.3",
                "color": color,
                "lw": 3.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        handles.append(Line2D([], [], color=color, lw=3.0, ls=style))
        labels.append(f"{label}: {len(cells) - 1} moves, total reward {_minus(ret)}")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=1,
        frameon=False,
        handlelength=3.6,
        borderaxespad=0.2,
    )
    return fig


# ------------------------------------------------------------------ figure 2: iterations vs gamma
GAMMAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
TOL = 1e-6
ORDER_UP_TO = 7
FINE_STEPS = 197  # gammas 0.5, 0.5025, ..., 0.99 for the two curves


def inventory_policy_arrays() -> tuple[np.ndarray, np.ndarray]:
    """P_pi and r_pi for the lab's inventory MDP under 'order up to 7'."""
    mdp = InventoryMDP(capacity=10, demand_mean=4.0, max_demand=10)
    S = mdp.n_states
    pi = np.zeros((S, mdp.n_actions))
    pi[np.arange(S), [max(0, ORDER_UP_TO - x) for x in range(S)]] = 1.0
    P_pi = (pi[:, :, None] * mdp.P).sum(axis=1)
    r_pi = (pi * mdp.R).sum(axis=1)
    return P_pi, r_pi


def iterate(P_pi: np.ndarray, r_pi: np.ndarray, gamma: float, tol: float) -> tuple[np.ndarray, int]:
    """V_0 = 0, V_{k+1} = r_pi + gamma P_pi V_k; stop at the first k with ||V_k - V_{k-1}||_inf < tol.

    Returns (V_k, k), where k is the number of backups. Elementwise sums rather than BLAS,
    so the count does not depend on the machine.
    """
    V = np.zeros_like(r_pi)
    k = 0
    while True:
        V_next = r_pi + gamma * (P_pi * V[None, :]).sum(axis=1)
        k += 1
        diff = float(np.abs(V_next - V).max())
        V = V_next
        if diff < tol:
            return V, k


def bound_x(P_pi: np.ndarray, r_pi: np.ndarray, gamma: float, tol: float) -> tuple[float, float]:
    """Return (D, x): D = ||V_0 - V^pi||_inf (exact, by a linear solve) and
    x = log((1 + gamma) D / tol) / log(1 / gamma).

    Why floor(x) + 2 (and so x + 2) bounds the iteration count: eq-iteration-bound gives
    ||V_j - V^pi|| <= gamma^j D. The test at iteration k compares V_k with V_{k-1}, and
    ||V_k - V_{k-1}|| <= ||V_k - V^pi|| + ||V_{k-1} - V^pi|| <= (1 + gamma) gamma^(k-1) D,
    which is < tol as soon as k - 1 > x. That is eq-iteration-bound with
    eps = tol / (1 + gamma), plus the one backup that makes the test pass.
    """
    V_pi = np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)
    D = float(np.abs(V_pi).max())  # V_0 = 0
    return D, math.log((1 + gamma) * D / tol) / math.log(1 / gamma)


def iterations_data() -> dict:
    P_pi, r_pi = inventory_policy_arrays()
    rows = []
    for g in GAMMAS:
        V, k = iterate(P_pi, r_pi, g, TOL)
        D, x = bound_x(P_pi, r_pi, g, TOL)
        V_pi = np.linalg.solve(np.eye(len(r_pi)) - g * P_pi, r_pi)
        err = float(np.abs(V - V_pi).max())
        k_bound = math.floor(x) + 2
        assert k <= k_bound, f"gamma={g}: {k} iterations exceed the bound {k_bound}"
        assert err <= g * TOL / (1 - g), f"gamma={g}: the eq-stop guarantee fails"
        rows.append(
            {
                "gamma": g,
                "k": k,
                "D": D,
                "bound": k_bound,
                "bound_curve": x + 2,
                "err": err,
                "eq_stop": g * TOL / (1 - g),
            }
        )
    # The same computation on a fine grid of gammas draws both curves, so the line between
    # two marked gammas is a real count, not an interpolation. The bound is rounded so the
    # SVG does not depend on last-bit differences in the linear solve between machines.
    fine = np.round(np.linspace(0.5, 0.99, FINE_STEPS), 4)
    fine_k, fine_bound = [], []
    for g in fine:
        k = iterate(P_pi, r_pi, float(g), TOL)[1]
        x = bound_x(P_pi, r_pi, float(g), TOL)[1]
        assert k <= math.floor(x) + 2, f"gamma={g}: {k} iterations exceed the bound"
        fine_k.append(k)
        fine_bound.append(round(x + 2, 3))
    return {"rows": rows, "fine": fine, "fine_k": np.array(fine_k), "curve": np.array(fine_bound)}


def _count_label(v: float, _pos: int | None = None) -> str:
    return f"{v:,.0f}"


def iterations_figure(t: Theme, data: dict) -> Figure:
    rows = data["rows"]
    fig = Figure(figsize=(WIDTH_IN, 4.4))  # 680 x 422 px, aspect 1.61
    ax = fig.add_axes((0.115, 0.125, 0.865, 0.665))
    (bound,) = ax.plot(
        data["fine"], data["curve"], color=t.neutral, lw=2.2, ls=(0, (5, 2.5)), zorder=2
    )
    ax.plot(data["fine"], data["fine_k"], color=t.series[0], lw=2.2, zorder=3)
    gs = [r["gamma"] for r in rows]
    ks = [r["k"] for r in rows]
    marker = {"marker": "o", "ms": 7, "mec": t.bg, "mew": 1.2}
    ax.plot(gs, ks, ls="none", color=t.series[0], zorder=4, **marker)
    for g, k in zip(gs, ks, strict=True):
        last = g == gs[-1]
        ax.annotate(
            _count_label(k),
            (g, k),
            xytext=(-9, -2) if last else (0, -11),
            textcoords="offset points",
            ha="right" if last else "center",
            va="center" if last else "top",
            fontsize=MIN_FONT,
            color=t.text,
        )
    ax.set_yscale("log")
    ax.set_ylim(10, 4000)
    ax.set_xlim(0.48, 1.0)
    ax.xaxis.set_major_locator(FixedLocator(list(GAMMAS)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(_count_label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", axis="y", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(r"Discount $\gamma$")
    ax.set_ylabel("Iterations to stop (log scale)")
    fig.legend(
        [bound, Line2D([], [], color=t.series[0], lw=2.2, **marker)],
        [
            r"Bound: $2 + \log\,((1+\gamma)\,\|V_0 - V^\pi\|_\infty\,/\,\mathrm{tol})"
            r"\;/\;\log\,(1/\gamma)$",
            r"Actual: iterations until $\|V_k - V_{k-1}\|_\infty < \mathrm{tol}$",
        ],
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        frameon=False,
        handlelength=3.0,
        title=r"Inventory MDP, order up to 7; $V_0 = 0$, $\mathrm{tol} = 10^{-6}$",
        alignment="left",
    )
    return fig


# ------------------------------------------------------------------ build and check
FIGURES = {
    "01-cliff-gridworld": (cliff_data, cliff_figure),
    "01-iterations-vs-gamma": (iterations_data, iterations_figure),
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
    cliff = computed["01-cliff-gridworld"]
    for key in ("safe", "edge"):
        cells, ret = cliff[key]
        print(f"cliff gridworld 4x12, {key} path: {len(cells) - 1} moves, total reward {ret:g}")
    print()
    print(
        "Inventory MDP (capacity 10, Poisson(4) demand truncated at 10, price 4, unit cost 2,\n"
        "fixed cost 5, holding 0.2, lost sales 1), policy q = max(0, 7 - stock), V_0 = 0, "
        f"tol = {TOL:g}"
    )
    print(
        f"{'gamma':>6} {'iters':>6} {'bound':>6} {'x+2':>9} {'||V0-Vpi||':>11} "
        f"{'final err':>10} {'eq-stop':>10}"
    )
    for r in computed["01-iterations-vs-gamma"]["rows"]:
        print(
            f"{r['gamma']:>6g} {r['k']:>6d} {r['bound']:>6d} {r['bound_curve']:>9.2f} "
            f"{r['D']:>11.4f} {r['err']:>10.2e} {r['eq_stop']:>10.2e}"
        )
    print(
        "bound = floor(x) + 2 with x = log((1 + gamma) ||V_0 - V^pi||_inf / tol) / log(1 / gamma)."
    )
    print(
        f"The figure draws x + 2 and the actual count on {FINE_STEPS} gammas in [0.5, 0.99] "
        "(every count is within its bound)."
    )


def main(argv: list[str]) -> int:
    check = "--check" in argv
    files, computed = build()
    if check:
        stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date; run scripts/make_figures_m01.py")
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
