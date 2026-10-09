"""Check internal links in the rendered site (_site/).

Every href/src that points inside the site must resolve to a file, and every
#fragment on an internal page link must match an id on the target page.
External links are not fetched (they are checked by hand before release).

  uv run python scripts/link_check.py [SITE_DIR]
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "a" and a.get("name"):
            self.ids.add(a["name"])
        for key in ("href", "src"):
            val = a.get(key)
            if val and tag in {"a", "img", "script", "link", "source", "iframe"}:
                self.links.append(val)


def parse(path: Path) -> Page:
    p = Page()
    p.feed(path.read_text(errors="replace"))
    return p


def base_path() -> str:
    """Path prefix of the published site (e.g. /practical-rl/), from _quarto.yml's site-url."""
    text = (ROOT / "_quarto.yml").read_text()
    m = re.search(r'site-url:\s*"?([^"\n]+)"?', text)
    path = urlparse(m.group(1).strip()).path if m else ""
    return path.rstrip("/") + "/" if path.strip("/") else "/"


def main(argv: list[str]) -> int:
    site = Path(argv[0]) if argv else ROOT / "_site"
    base = base_path()
    if not site.is_dir():
        print(f"{site} does not exist; render first")
        return 1
    pages = {p: parse(p) for p in site.rglob("*.html")}
    problems = []
    for path, page in pages.items():
        for link in page.links:
            u = urlparse(link)
            if (
                u.scheme
                or link.startswith(("//", "mailto:", "javascript:", "data:"))
                or link == "#"
            ):
                continue
            if not u.path and u.fragment:
                if u.fragment not in page.ids:
                    problems.append(f"{path.relative_to(site)}: missing anchor #{u.fragment}")
                continue
            if u.path.startswith(base) and base != "/":
                target = site / unquote(u.path[len(base) :])
            elif u.path.startswith("/"):
                target = site / unquote(u.path.lstrip("/"))
            else:
                target = path.parent / unquote(u.path)
            target = target.resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                problems.append(f"{path.relative_to(site)}: broken link {link}")
                continue
            if u.fragment and target.suffix == ".html" and target in pages:
                if u.fragment not in pages[target].ids and not re.match(
                    r"^(scrollTo|L\d)", u.fragment
                ):
                    problems.append(f"{path.relative_to(site)}: {link} has no anchor #{u.fragment}")
    for p in sorted(set(problems)):
        print(p)
    print(f"checked {len(pages)} pages: {len(set(problems))} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
