"""Vendored fonts (fonts/, PLAN.md §8): payload, licenses, and glyph coverage of the site.

The coverage test reads the rendered site (_site/) and checks that every character
the pages show is in the subset fonts (fonts/coverage.json, written by
fonts/build.sh), so nothing silently falls back to a system font. It is skipped
when the site has not been rendered.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "fonts"
SITE = ROOT / "_site"
TEXT_FONTS = ("inter-roman.woff2", "inter-italic.woff2")
MONO_FONT = "jetbrains-mono.woff2"
MATH_FONT = "noto-sans-math.woff2"  # fallback for math symbols in both stacks
MAX_PAYLOAD = 400 * 1024

# Characters allowed to come from a system font, with the reason.
FALLBACK_OK = {
    "◐": "◐ readiness chip icon; aria-hidden, in none of the fonts",
}


def coverage() -> dict[str, set[int]]:
    data = json.loads((FONTS / "coverage.json").read_text())
    out: dict[str, set[int]] = {}
    for name, ranges in data.items():
        cps: set[int] = set()
        for r in ranges:
            lo, _, hi = r.partition("-")
            cps.update(range(int(lo, 16), int(hi or lo, 16) + 1))
        out[name] = cps
    return out


def test_font_files_payload_and_licenses() -> None:
    files = [*TEXT_FONTS, MONO_FONT, MATH_FONT]
    for f in files:
        assert (FONTS / f).is_file(), f"fonts/{f} missing; run bash fonts/build.sh"
    total = sum((FONTS / f).stat().st_size for f in files)
    assert total < MAX_PAYLOAD, f"font payload {total} bytes exceeds {MAX_PAYLOAD}"
    for lic in ("OFL-Inter.txt", "OFL-JetBrainsMono.txt", "OFL-NotoSansMath.txt", "LICENSES.md"):
        assert (FONTS / lic).is_file(), f"fonts/{lic} missing"
    assert set(coverage()) == set(files)


def test_stacks_cover_greek_arrows_and_operators() -> None:
    cov = coverage()
    greek = [0x03B3, 0x03C0, 0x03B5, 0x0394, 0x03C1, 0x03BB]
    math = [0x2192, 0x21D2, 0x21A6, 0x2264, 0x2211, 0x221E, 0x2208, 0x2200, 0x2207, 0x221D, 0x22C5]
    math += list(range(0x2190, 0x2200)) + list(range(0x2200, 0x2300))
    for name in [*TEXT_FONTS, MONO_FONT]:
        stack = cov[name] | cov[MATH_FONT]
        missing = [f"U+{c:04X}" for c in greek if c not in cov[name]]
        missing += [f"U+{c:04X}" for c in math if c not in stack]
        assert not missing, f"{name} + {MATH_FONT} lack {missing}"


class _Text(HTMLParser):
    """Collect visible text, split into prose and code; skip scripts, styles and raw TeX."""

    SKIP = {"script", "style", "template", "noscript", "svg", "math"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []  # (tag, skip, code)
        self.prose: list[str] = []
        self.code: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "img", "meta", "link", "input", "hr", "wbr", "source"}:
            return
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        parent_skip = self.stack[-1][1] if self.stack else False
        parent_code = self.stack[-1][2] if self.stack else False
        skip = parent_skip or tag in self.SKIP or "math" in classes
        code = parent_code or tag in {"code", "pre", "kbd", "samp"}
        self.stack.append((tag, skip, code))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1][1]:
            return
        (self.code if self.stack and self.stack[-1][2] else self.prose).append(data)


def site_pages() -> list[Path]:
    if not (SITE / "index.html").exists():
        pytest.skip("_site/ not rendered")
    return sorted(p for p in SITE.rglob("*.html") if "site_libs" not in p.parts)


def test_rendered_text_is_covered_by_the_fonts() -> None:
    cov = coverage()
    text_cps = set.intersection(*(cov[f] for f in TEXT_FONTS)) | cov[MATH_FONT]
    mono_cps = cov[MONO_FONT] | cov[MATH_FONT]
    problems: dict[str, set[str]] = {}
    for page in site_pages():
        if page.name == "welcome.html":  # the revealjs deck has its own theme
            continue
        parser = _Text()
        parser.feed(page.read_text(errors="replace"))
        for chunks, cps, label in ((parser.prose, text_cps, "Inter"), (parser.code, mono_cps, "JetBrains Mono")):
            for ch in set("".join(chunks)):
                if ch.isspace() or ord(ch) < 0x20 or ch in FALLBACK_OK:
                    continue
                if ord(ch) not in cps:
                    problems.setdefault(f"{label}: U+{ord(ch):04X} {ch}", set()).add(
                        str(page.relative_to(SITE))
                    )
    assert not problems, "characters missing from the subset fonts (extend fonts/build.sh): " + "; ".join(
        f"{k} on {', '.join(sorted(v)[:3])}" for k, v in sorted(problems.items())
    )


def test_coverage_json_matches_the_font_files() -> None:
    ttlib = pytest.importorskip("fontTools.ttLib")
    pytest.importorskip("brotli")
    cov = coverage()
    for name, cps in cov.items():
        cmap = ttlib.TTFont(FONTS / name).getBestCmap()
        assert set(cmap) == cps, f"fonts/coverage.json is stale for {name}; run fonts/build.sh"
