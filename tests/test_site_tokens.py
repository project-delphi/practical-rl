"""Theme tokens (brand/light.yml, brand/dark.yml): contrast and completeness (PLAN.md §8).

Text pairs need 4.5:1 (WCAG 1.4.3); borders, focus rings and plot series need 3:1
against what they sit on (WCAG 1.4.11).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "brand"
SCSS = ROOT / "styles" / "prl.scss"

TEXT = [  # (foreground, background) palette keys; "bg" is color.background
    ("text", "bg"),
    ("heading", "bg"),
    ("muted", "bg"),
    ("link", "bg"),
    ("text", "surface"),
    ("muted", "surface"),
    ("link", "surface"),
    ("code", "code-bg"),
    ("code", "surface"),
    ("link", "code-bg"),
    ("on-accent", "accent"),
    ("on-navy", "navy"),
    ("navbar-hl", "navy"),
    ("chip-ok-fg", "chip-ok-bg"),
    ("chip-partial-fg", "chip-partial-bg"),
    ("chip-none-fg", "chip-none-bg"),
    ("chip-stale-fg", "chip-stale-bg"),
]

UI = [
    ("border-strong", "bg"),
    ("border-strong", "surface"),
    ("link", "bg"),  # the focus ring
    ("on-navy", "navy"),  # the focus ring on the navbar
    ("series-1", "bg"),
    ("series-2", "bg"),
    ("series-3", "bg"),
    ("series-4", "bg"),
    ("neutral", "bg"),
]


def load(name: str) -> dict[str, str]:
    data = yaml.safe_load((BRAND / f"{name}.yml").read_text())
    colors = dict(data["color"]["palette"])
    colors["bg"] = data["color"]["background"]
    return colors


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize(("fg", "bg"), TEXT)
def test_text_contrast(theme: str, fg: str, bg: str) -> None:
    c = load(theme)
    ratio = contrast(c[fg], c[bg])
    assert ratio >= 4.5, f"{theme}: {fg} {c[fg]} on {bg} {c[bg]} is {ratio:.2f}:1"


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize(("fg", "bg"), UI)
def test_ui_contrast(theme: str, fg: str, bg: str) -> None:
    c = load(theme)
    ratio = contrast(c[fg], c[bg])
    assert ratio >= 3.0, f"{theme}: {fg} {c[fg]} on {bg} {c[bg]} is {ratio:.2f}:1"


def test_themes_define_the_same_tokens() -> None:
    assert set(load("light")) == set(load("dark"))


def test_scss_uses_only_defined_brand_colors() -> None:
    used = set(re.findall(r"\$brand-([a-z0-9-]+)", SCSS.read_text()))
    for theme in ("light", "dark"):
        missing = used - set(load(theme))
        assert not missing, f"styles/prl.scss uses $brand-{sorted(missing)} missing in {theme}.yml"


def test_scss_rules_use_tokens_not_literal_colors() -> None:
    """Every color in the rules section comes from a --prl-* token."""
    rules = SCSS.read_text().split("/*-- scss:rules --*/", 1)[1]
    code = "\n".join(line.split("//", 1)[0] for line in rules.splitlines())
    literals = re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", code)
    assert not literals, f"literal colors in styles/prl.scss rules: {literals}"
