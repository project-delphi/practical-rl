"""The site's Lua filters (filters/), rendered with Quarto on a small fixture page.

disclosure.lua    collapsed callouts become <details>/<summary>
katex.lua         KaTeX is self-hosted (no CDN), only on pages with math
scroll-regions.lua  tables sit in a labeled wrapper that scrolls sideways

Skipped when Quarto is not on PATH (the browser behavior is covered by
scripts/site_check.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILTERS = ROOT / "filters"

FIXTURE = """---
title: "Fixture"
format:
  html:
    html-math-method: katex
filters:
  - {filters}/katex.lua
  - {filters}/disclosure.lua
  - {filters}/scroll-regions.lua
---

## Check yourself 3.2

The return is $G_t = \\sum_k \\gamma^k r_{{t+k}}$.

::: {{.callout-tip collapse="true" title="Answer to 3.2"}}
Because $\\gamma < 1$ the sum converges.
:::

::: {{.callout-note collapse="true"}}
No title here.
:::

::: {{.callout-warning collapse="false" title="Open with $\\alpha$"}}
Starts open.
:::

::: {{.callout-caution collapse="true"}}
## Title from a heading
Body.
:::

::: {{.callout-note collapse="true" #nte-numbered}}
## A numbered note
Left to Quarto.
:::

::: {{.callout-important}}
Not collapsible.
:::

## Labs

| a | b |
|---|---|
| 1 | 2 |

| c | d |
|---|---|
| 3 | 4 |

: A captioned table
"""

NO_MATH = """---
title: "No math"
format: html
filters:
  - {filters}/katex.lua
---

Plain text.
"""


def render(tmp_path: Path, name: str, text: str) -> str:
    quarto = shutil.which("quarto")
    if quarto is None:
        pytest.skip("quarto not on PATH")
    src = tmp_path / f"{name}.qmd"
    src.write_text(text.format(filters=FILTERS))
    out = subprocess.run(
        [quarto, "render", src.name, "--to", "html"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return (tmp_path / f"{name}.html").read_text()


@pytest.fixture(scope="module")
def page(tmp_path_factory: pytest.TempPathFactory) -> str:
    return render(tmp_path_factory.mktemp("filters"), "fixture", FIXTURE)


def details(html: str) -> list[tuple[str, str]]:
    """(opening tag, summary html) for each <details>."""
    return re.findall(r"(<details[^>]*>)\s*<summary>(.*?)</summary>", html, re.S)


def test_collapsed_callouts_become_details(page: str) -> None:
    found = details(page)
    summaries = [re.sub(r"<[^>]+>", "", s).strip() for _, s in found]
    assert summaries[:2] == ["Answer to 3.2", "Note"]
    assert summaries[3] == "Title from a heading"
    assert 'class="disclosure disclosure-tip"' in found[0][0]
    assert " open" not in found[0][0]


def test_collapse_false_starts_open_and_keeps_math(page: str) -> None:
    tag, summary = details(page)[2]
    assert "open" in tag and "disclosure-warning" in tag
    assert 'class="math inline"' in summary


def test_crossref_and_plain_callouts_are_left_alone(page: str) -> None:
    assert len(details(page)) == 4
    assert 'id="nte-numbered"' in page and "Note&nbsp;1" in page
    assert "callout-important" in page


def test_katex_is_self_hosted(page: str) -> None:
    assert "cdn.jsdelivr.net" not in page
    assert re.search(r'<script src="[^"]*katex-0\.18\.9/katex\.min\.js"', page)
    assert re.search(r'<script src="[^"]*katex-0\.18\.9/katex-render\.js"', page)
    assert re.search(r'<link href="[^"]*katex-0\.18\.9/katex\.min\.css"', page)


def test_page_without_math_loads_no_katex(tmp_path: Path) -> None:
    html = render(tmp_path, "nomath", NO_MATH)
    assert "katex" not in html.lower()


def test_tables_get_a_labeled_scroll_wrapper(page: str) -> None:
    labels = re.findall(r'<div class="table-wrap" data-region-label="([^"]*)">\s*<table', page)
    assert labels == ["Labs table", "A captioned table"]
