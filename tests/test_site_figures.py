"""Figures and demos on the site (PLAN.md §8).

- Every images/*.svg has a dark twin (NAME.dark.svg), or is listed in THEME_NEUTRAL.
- The committed SVGs are current (scripts/make_figures_m01.py --check) and sized for
  the 680 px column with an aspect ratio of at most 2:1.
- filters/figures.lua, rendered with Quarto on a fixture page, emits each paired figure
  as a light and a dark copy inside one figure: one number, one caption, one id.
- The OJS demos follow the demo rules that can be read from the source.

The rendering tests are skipped when Quarto is not on PATH; how the copies switch with
the theme in a browser is covered by scripts/site_check.py.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
FILTER = ROOT / "filters" / "figures.lua"

# SVGs that read on both backgrounds (an opaque background of their own), so they need
# no dark twin. One entry per file, with the reason.
THEME_NEUTRAL: dict[str, str] = {}

FIGURE_SCRIPTS = sorted((ROOT / "scripts").glob("make_figures_m*.py"))


def light_svgs() -> list[Path]:
    return sorted(p for p in IMAGES.glob("*.svg") if not p.name.endswith(".dark.svg"))


def test_every_svg_has_a_dark_twin_or_is_theme_neutral() -> None:
    missing = [
        p.name
        for p in light_svgs()
        if p.name not in THEME_NEUTRAL and not p.with_name(p.stem + ".dark.svg").exists()
    ]
    assert not missing, f"no dark twin (NAME.dark.svg) and not listed as theme-neutral: {missing}"


def test_every_dark_svg_has_a_light_original() -> None:
    orphans = [
        p.name
        for p in IMAGES.glob("*.dark.svg")
        if not p.with_name(p.name.removesuffix(".dark.svg") + ".svg").exists()
    ]
    assert not orphans, f"dark SVGs without a light original: {orphans}"


def test_theme_neutral_entries_exist() -> None:
    stale = [name for name in THEME_NEUTRAL if not (IMAGES / name).exists()]
    assert not stale, f"THEME_NEUTRAL lists files that do not exist: {stale}"


@pytest.mark.parametrize("script", FIGURE_SCRIPTS, ids=lambda p: p.name)
def test_committed_figures_are_current(script: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("svg", sorted(IMAGES.glob("*.svg")), ids=lambda p: p.name)
def test_svg_is_sized_for_the_column(svg: Path) -> None:
    head = svg.read_text()[:2000]
    m = re.search(r'<svg[^>]*\bwidth="([\d.]+)pt"[^>]*\bheight="([\d.]+)pt"', head)
    assert m, "the root <svg> needs width and height in pt (matplotlib's SVG output)"
    width, height = float(m.group(1)), float(m.group(2))
    assert abs(width - 510) < 0.5, f"width {width} pt; the 680 px column is 510 pt"
    assert width / height <= 2.0 + 1e-9, f"aspect {width / height:.2f} is wider than 2:1"
    assert "<dc:date>" not in head, "a date in the metadata makes the file change on every build"


def test_figures_filter_runs_after_quarto() -> None:
    config = yaml.safe_load((ROOT / "_quarto.yml").read_text())
    entries = [f for f in config["filters"] if isinstance(f, dict)]
    assert {"path": "filters/figures.lua", "at": "post-render"} in entries


# ------------------------------------------------------------------ the filter, rendered
FIXTURE = """---
title: "Figures fixture"
format:
  html:
    lightbox: auto
filters:
  - path: {filter}
    at: post-render
---

See @fig-pair and @fig-solo.

![What to notice: the pair.](images/pair.svg){{#fig-pair fig-alt="A light and a dark copy of one figure."}}

![What to notice: no twin.](images/solo.svg){{#fig-solo fig-alt="A figure that has no dark copy."}}

Inline ![inline pair](images/pair.svg){{width=40}} and remote ![remote](https://example.org/r.svg).
"""

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>\n'


@pytest.fixture(scope="module")
def page(tmp_path_factory: pytest.TempPathFactory) -> str:
    quarto = shutil.which("quarto")
    if quarto is None:
        pytest.skip("quarto not on PATH")
    tmp = tmp_path_factory.mktemp("figures")
    (tmp / "images").mkdir()
    for name in ("pair.svg", "pair.dark.svg", "solo.svg"):
        (tmp / "images" / name).write_text(SVG)
    (tmp / "fixture.qmd").write_text(FIXTURE.format(filter=FILTER))
    out = subprocess.run(
        [quarto, "render", "fixture.qmd", "--to", "html"],
        cwd=tmp,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return (tmp / "fixture.html").read_text()


def figure(html: str, fig_id: str) -> str:
    m = re.search(rf'<div id="{fig_id}" class="quarto-float.*?</figure>\s*</div>', html, re.S)
    assert m, f"figure {fig_id} not found"
    return m.group(0)


def test_pair_is_one_figure_with_a_light_and_a_dark_copy(page: str) -> None:
    fig = figure(page, "fig-pair")
    copies = re.findall(
        r'<span class="(light|dark)-content"><a href="([^"]+)" class="lightbox" '
        r'data-gallery="([^"]+)"[^>]*><img src="([^"]+)"[^>]*alt="([^"]*)"',
        fig,
    )
    assert [(c[0], c[1], c[3]) for c in copies] == [
        ("light", "images/pair.svg", "images/pair.svg"),
        ("dark", "images/pair.dark.svg", "images/pair.dark.svg"),
    ]
    light_gallery, dark_gallery = copies[0][2], copies[1][2]
    assert dark_gallery == light_gallery + "-dark", "each copy needs its own lightbox gallery"
    assert {c[4] for c in copies} == {"A light and a dark copy of one figure."}
    assert fig.count("<figcaption") == 1
    assert len(re.findall(r"Figure&nbsp;1:", fig)) == 3  # caption + one lightbox title per copy


def test_numbering_and_ids_are_unchanged(page: str) -> None:
    assert re.findall(r'class="quarto-xref">(Figure&nbsp;\d)', page) == [
        "Figure&nbsp;1",
        "Figure&nbsp;2",
    ]
    assert page.count('id="fig-pair"') == 1 and page.count('id="fig-solo"') == 1
    assert "subfig" not in page and "(a)" not in page


def test_image_without_a_twin_gets_paper_and_no_copy(page: str) -> None:
    fig = figure(page, "fig-solo")
    assert "light-content" not in fig and "dark-content" not in fig
    assert re.search(r'<img src="images/solo.svg" class="[^"]*\bpaper\b', fig)


def test_inline_and_remote_images(page: str) -> None:
    assert re.search(
        r'<span class="light-content"><img src="images/pair.svg"[^>]*></span>'
        r'<span class="dark-content"><img src="images/pair.dark.svg"[^>]*></span>',
        page,
    )
    assert re.search(r'<img src="https://example.org/r.svg" class="img-fluid" alt="remote">', page)


# ------------------------------------------------------------------ OJS demos
DEMOS = sorted((ROOT / "demos").glob("*.qmd"))


@pytest.mark.parametrize("demo", DEMOS, ids=lambda p: p.name)
def test_demo_follows_the_demo_rules(demo: Path) -> None:
    text = demo.read_text()
    assert not text.lstrip().startswith("---"), "an included demo must not have front matter"
    assert "{{<" not in text, "no shortcodes in a demo (an include in a comment still runs)"
    assert "Predict first" in text, "a Predict-first prompt above the controls"
    assert re.search(r'aria-live="polite"', text), "an aria-live readout"
    cells = re.findall(r"```\{ojs\}\n(.*?)```", text, re.S)
    assert cells, "no {ojs} cells"
    assert all(c.startswith("//| echo: false") for c in cells), "every OJS cell hides its code"
    code = re.sub(r"//[^\n]*", "", "\n".join(cells))  # rules apply to code, not comments
    assert "Math.random" not in code and "d3.random" not in code, "no unseeded randomness"
    assert not re.search(r"\b(setInterval|requestAnimationFrame|Promises\.tick|now\b)", code), (
        "nothing may animate or play by itself"
    )
    colors = set(re.findall(r'"(#[0-9A-Fa-f]{3,8}|rgb[^"]*)"', code))
    assert not colors, f"colors must come from var(--prl-*), found {colors}"
