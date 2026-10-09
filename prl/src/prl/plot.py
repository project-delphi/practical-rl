"""Plotting with one consistent style.

Figures are drawn on an opaque white background so they read in both
Colab themes. Colors come from the site's brand palette (generated into
prl/_palette.py by scripts/gen_tokens.py). At most four categorical series
per axis; reference lines use the neutral color, dashed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ._palette import DARK, LIGHT, NEUTRAL_LIGHT

SERIES = list(LIGHT)


def style() -> None:
    """Apply the workshop plot style (call once, in the setup cell)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.figsize": (7.0, 3.6),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "image.cmap": "cividis",
        }
    )
    try:  # retina output in notebooks
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats("retina")
    except Exception:  # noqa: BLE001 - outside IPython this is a no-op
        pass


def _downsample(x: np.ndarray, *ys: np.ndarray, max_points: int = 1000):
    if x.size <= max_points:
        return (x, *ys)
    idx = np.linspace(0, x.size - 1, max_points).astype(int)
    return (x[idx], *(y[..., idx] for y in ys))


def curves(
    x: Sequence[float],
    runs: Mapping[str, np.ndarray],
    *,
    band: str | None = "minmax",
    ax: plt.Axes | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    budget: str | None = None,
) -> plt.Axes:
    """Plot one line per method; `runs[name]` has shape (n_seeds, len(x)).

    band = "minmax" shades the seed range, "none" draws each seed thinly.
    """
    if len(runs) > 4:
        raise ValueError("at most 4 series per axis; use small multiples")
    ax = ax or plt.gca()
    xs = np.asarray(x, dtype=float)
    for color, (name, arr) in zip(SERIES, runs.items(), strict=False):
        data = np.atleast_2d(np.asarray(arr, dtype=float))
        xx, data = _downsample(xs, data)
        center = data.mean(axis=0)
        ax.plot(xx, center, color=color, label=f"{name} (mean of {data.shape[0]})")
        if band == "minmax" and data.shape[0] > 1:
            ax.fill_between(xx, data.min(axis=0), data.max(axis=0), color=color, alpha=0.2, lw=0)
        elif band == "none":
            for row in data:
                ax.plot(xx, row, color=color, alpha=0.35, lw=1)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(title=budget)
    return ax


def with_recorded(
    band: Mapping[str, Any],
    yours: np.ndarray,
    *,
    x: Sequence[float],
    your_budget: Mapping[str, Any],
    ax: plt.Axes | None = None,
    label: str = "your seed",
    **kw: Any,
) -> plt.Axes:
    """Overlay your single run on a recorded multi-seed band.

    `band` is a recorded band from prl.data (keys: "x", "runs", "budget", "record").
    Refuses to overlay runs made with a different budget (for example QUICK vs live).
    """
    if dict(band["budget"]) != dict(your_budget):
        raise ValueError(
            f"Budgets differ (recorded {dict(band['budget'])} vs yours {dict(your_budget)}); "
            "the overlay would not be comparable."
        )
    ax = ax or plt.gca()
    rec_x = np.asarray(band["x"], dtype=float)
    runs = np.asarray(band["runs"], dtype=float)
    ax.fill_between(
        rec_x,
        runs.min(axis=0),
        runs.max(axis=0),
        color=NEUTRAL_LIGHT,
        alpha=0.25,
        lw=0,
        label=f"recorded: {runs.shape[0]} seeds ({band.get('record', 'record')})",
    )
    ax.plot(
        np.asarray(x, dtype=float), np.asarray(yours, dtype=float), color=SERIES[0], label=label
    )
    ax.set(**{k: v for k, v in kw.items() if k in {"xlabel", "ylabel", "title"}})
    ax.legend()
    return ax


def bars_ci(
    names: Sequence[str],
    centers: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    *,
    ax: plt.Axes | None = None,
    ylabel: str = "",
    title: str = "",
) -> plt.Axes:
    """Bar chart with interval whiskers (CI or seed range)."""
    ax = ax or plt.gca()
    c = np.asarray(centers, dtype=float)
    err = np.vstack([c - np.asarray(lows, dtype=float), np.asarray(highs, dtype=float) - c])
    colors = [SERIES[i % len(SERIES)] for i in range(len(names))]
    ax.bar(names, c, yerr=err, color=colors, capsize=4)
    ax.set(ylabel=ylabel, title=title)
    return ax


def grid_values(
    values: np.ndarray,
    rows: int,
    cols: int,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
    annotate: bool = True,
) -> plt.Axes:
    """Show a value per gridworld cell as a heatmap."""
    ax = ax or plt.gca()
    grid = np.asarray(values, dtype=float).reshape(rows, cols)
    im = ax.imshow(grid, cmap="cividis")
    if annotate and rows * cols <= 60:
        for (r, c), v in np.ndenumerate(grid):
            ax.text(
                c,
                r,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v < np.median(grid) else "black",
            )
    ax.set(xticks=[], yticks=[], title=title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.03)
    return ax


def policy_arrows(
    policy: np.ndarray,
    rows: int,
    cols: int,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
    skip: Sequence[int] = (),
) -> plt.Axes:
    """Draw a deterministic gridworld policy (0 up, 1 right, 2 down, 3 left) as arrows."""
    ax = ax or plt.gca()
    arrows = {0: (0, 0.35), 1: (0.35, 0), 2: (0, -0.35), 3: (-0.35, 0)}
    for s, a in enumerate(np.asarray(policy, dtype=int)):
        if s in skip:
            continue
        r, c = divmod(s, cols)
        dx, dy = arrows[int(a)]
        ax.arrow(
            c - dx / 2,
            (rows - 1 - r) - dy / 2,
            dx,
            dy,
            head_width=0.15,
            color=SERIES[0],
            length_includes_head=True,
        )
    ax.set(
        xlim=(-0.5, cols - 0.5),
        ylim=(-0.5, rows - 0.5),
        xticks=[],
        yticks=[],
        title=title,
        aspect="equal",
    )
    ax.grid(False)
    return ax


__all__ = [
    "DARK",
    "LIGHT",
    "SERIES",
    "bars_ci",
    "curves",
    "grid_values",
    "policy_arrows",
    "style",
    "with_recorded",
]
