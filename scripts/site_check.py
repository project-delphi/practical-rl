"""Browser checks for the rendered site (PLAN.md §8), with Playwright and axe-core.

Every page in _site/ is loaded in Chromium in four configurations,
{light, dark} x {360x800, 1280x800}, from a local server that serves the site at
its published base path (/practical-rl/, from _quarto.yml's site-url). The check
fails on:

  axe       serious or critical axe-core violations for WCAG 2.0/2.1/2.2 A and AA
            (tags wcag2a, wcag2aa, wcag21a, wcag21aa, wcag22aa), unless allow-listed
  theme     the page did not switch to the emulated color scheme
  reflow    page-level horizontal scrolling at 360 px (and at 320 px, WCAG 1.4.10)
  focus     a Tab stop, or a sampled focusable element in the content, without a
            visible focus indicator, or entirely hidden (e.g. under the navbar)
  summary   a <summary> that does not toggle its <details> with Enter and with Space
  katex     a page with math where KaTeX did not typeset every formula
  network   a request to a host outside the allow-list, or a failed local request
  motion    smooth scrolling or endless animation with prefers-reduced-motion
            (light, 1280 only)
  print     navigation still shown, or dark colors, when printing a dark-mode page
            (dark, 1280 only)

axe-core is the copy bundled with Quarto (share/formats/html/axe/axe.min.js, found
with `quarto --paths`), so no extra JavaScript is vendored; pass --axe to use
another copy. Exceptions live in scripts/site_check_allow.yml, one reason each.

  uv run --group browser python scripts/site_check.py [--site _site] [--only PATTERN]
  uv run --group browser playwright install chromium     # once
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import http.server
import shutil
import socketserver
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from link_check import base_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALLOW_FILE = Path(__file__).resolve().parent / "site_check_allow.yml"
AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]
FAIL_IMPACTS = {"serious", "critical"}
SCHEMES = ["light", "dark"]
WIDTHS = [360, 1280]
HEIGHT = 800
TAB_STOPS = 15  # real Tab presses per page
FOCUS_SAMPLE = 40  # content elements focused programmatically after that
SUMMARY_SAMPLE = 10
WORKERS = 4
VERBOSE = False  # --verbose: print axe's explanation for each failing node


# ------------------------------------------------------------------ allow-list
ALLOW_KINDS = {"axe": "rule", "hosts": "host", "missing": "path"}


def load_allow(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    data = data or {}
    unknown = set(data) - set(ALLOW_KINDS)
    if unknown:
        raise SystemExit(f"{path.name}: unknown section(s) {sorted(unknown)}")
    allow = {kind: data.get(kind) or [] for kind in ALLOW_KINDS}
    for kind, key in ALLOW_KINDS.items():
        for e in allow[kind]:
            if not e.get(key):
                raise SystemExit(f"{path.name}: {kind} entry {e} has no {key}")
            if not str(e.get("reason", "")).strip():
                raise SystemExit(f"{path.name}: {kind} entry {e} has no reason")
    return allow


def network_allowed(allow: dict, kind: str, value: str, page: str) -> bool:
    key = ALLOW_KINDS[kind]
    return any(
        fnmatch.fnmatch(value, e[key]) and fnmatch.fnmatch(page, e.get("page", "*"))
        for e in allow[kind]
    )


def axe_allowed(allow: dict, rule: str, page: str, targets: list[str]) -> bool:
    for e in allow["axe"]:
        if e.get("rule") != rule:
            continue
        if not fnmatch.fnmatch(page, e.get("page", "*")):
            continue
        sel = e.get("selector")
        if sel is None or any(sel in t for t in targets):
            return True
    return False


# ------------------------------------------------------------------ server
def serve(site: Path, base: str) -> tuple[socketserver.TCPServer, int]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            p = unquote(urlparse(path).path)
            if base != "/" and not (p + "/").startswith(base):
                return str(site / "__outside_base_path__")
            rel = p[len(base) :] if base != "/" else p.lstrip("/")
            return str(site / rel)

        def log_message(self, *args: Any) -> None:
            pass

    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def find_axe(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    quarto = shutil.which("quarto")
    if quarto:
        out = subprocess.run([quarto, "--paths"], capture_output=True, text=True, check=False)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        for ln in lines:
            cand = Path(ln) / "formats" / "html" / "axe" / "axe.min.js"
            if cand.exists():
                return cand
    raise SystemExit("axe-core not found: put Quarto 1.10 on PATH or pass --axe PATH/axe.min.js")


# ------------------------------------------------------------------ page scripts
JS_SETTLE = """async () => {
  if (document.fonts && document.fonts.ready) await document.fonts.ready;
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
}"""

# Citation and footnote popups (tippy) fade in and out over ~300 ms; a check that
# looks right after each key press would see a half-faded popup. Make them instant.
JS_TIPPY_INSTANT = """() => {
  for (const el of document.querySelectorAll('*')) {
    if (el._tippy) el._tippy.setProps({ duration: 0, delay: 0 });
  }
}"""

JS_STATE = """() => {
  const math = [...document.querySelectorAll('span.math')];
  const typeset = math.filter(m => m.querySelector('.katex')).length;
  const errors = document.querySelectorAll('.katex-error').length;
  const body = document.body.className;
  return {
    reveal: !!document.querySelector('.reveal .slides'),
    theme: body.includes('quarto-dark') ? 'dark' : (body.includes('quarto-light') ? 'light' : 'none'),
    math: math.length, typeset, errors, katex: !!window.katex,
  };
}"""

JS_OVERFLOW = """() => {
  const de = document.documentElement;
  const width = de.clientWidth;
  const over = de.scrollWidth - width;
  const culprits = [];
  if (over > 0) {
    for (const el of document.querySelectorAll('main *, header *, footer *, nav *')) {
      const r = el.getBoundingClientRect();
      if (r.right > width + 1 && r.width > 0) {
        let p = el.parentElement, clipped = false;
        while (p && p !== document.body) {
          const s = getComputedStyle(p);
          if (s.overflowX !== 'visible' && s.overflowX !== 'clip' && p.scrollWidth > p.clientWidth) { clipped = true; break; }
          p = p.parentElement;
        }
        if (!clipped) culprits.push(el);
      }
    }
  }
  const describe = el => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
    (el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\\s+/).join('.') : '');
  // keep the outermost offenders only
  const outer = culprits.filter(el => !culprits.some(o => o !== el && o.contains(el)));
  return { over, culprits: outer.slice(0, 5).map(describe) };
}"""

# Describe the focused element: is a focus indicator drawn, and is any part of it visible?
JS_FOCUS_INFO = """(el) => {
  if (!el || el === document.body || el === document.documentElement) return null;
  const describe = e => {
    const t = (e.getAttribute('aria-label') || e.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 40);
    return e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') +
      (e.classList.length ? '.' + [...e.classList].slice(0, 3).join('.') : '') + (t ? ` "${t}"` : '');
  };
  const cs = getComputedStyle(el);
  const ow = parseFloat(cs.outlineWidth) || 0;
  const outline = cs.outlineStyle !== 'none' && (cs.outlineStyle === 'auto' || ow >= 2);
  const shadow = cs.boxShadow && cs.boxShadow !== 'none';
  const r = el.getBoundingClientRect();
  const rects = [...el.getClientRects()];
  const sized = rects.some(q => q.width > 0 && q.height > 0) || (r.width > 0 && r.height > 0);
  let shown = false, cover = null, inView = false;
  if (sized) {
    const pts = [[0.5, 0.5], [0.15, 0.2], [0.85, 0.2], [0.15, 0.8], [0.85, 0.8]];
    for (const q of (rects.length ? rects : [r])) {
      for (const [fx, fy] of pts) {
        const x = q.left + q.width * fx, y = q.top + q.height * fy;
        if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
        inView = true;
        // popups opened by the focus itself (citation and footnote tooltips) do not
        // cover anything, unless the focused element is in that popup
        const own = el.closest('[data-tippy-root]');
        const hit = document.elementsFromPoint(x, y).find(h => {
          const pop = h.closest('[data-tippy-root]');
          return !pop || pop === own;
        });
        if (hit && (hit === el || el.contains(hit) || hit.contains(el))) { shown = true; break; }
        if (hit && !cover) cover = describe(hit);
      }
      if (shown) break;
    }
  }
  const where = inView ? 'covered by ' + cover : 'outside the viewport, top ' + Math.round(r.top);
  return { what: describe(el), visible: el.matches(':focus-visible'), outline, shadow, sized, shown, where,
           style: `${cs.outlineStyle} ${cs.outlineWidth} / ${cs.boxShadow.slice(0, 40)}` };
}"""

JS_FOCUS_SAMPLE = """([limit, infoSrc]) => {
  const info = eval(infoSrc);
  const sel = 'a[href], button, summary, input, select, textarea, [tabindex="0"]';
  const scope = document.querySelector('main') || document.body;
  const els = [...scope.querySelectorAll(sel)].filter(e => {
    if (e.closest('details:not([open])') && e.tagName !== 'SUMMARY') return false;
    const s = getComputedStyle(e);
    if (s.visibility === 'hidden' || s.display === 'none') return false;
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && !e.closest('[aria-hidden="true"], [inert], [data-tippy-root]');
  });
  // a spread of the page's controls, not just the first few
  const step = Math.max(1, Math.ceil(els.length / limit));
  const out = [];
  for (let i = 0; i < els.length && out.length < limit; i += step) {
    // close the previous element's citation popup at once, as a pause would
    if (window.tippy && window.tippy.hideAll) window.tippy.hideAll({ duration: 0 });
    els[i].focus();
    const got = info(document.activeElement);
    if (got) out.push(got);
  }
  return out;
}"""

JS_SUMMARIES = """(limit) => [...document.querySelectorAll('details > summary')]
  .filter(s => s.getBoundingClientRect().width > 0).slice(0, limit).map((s, i) => {
    s.setAttribute('data-site-check', String(i));
    return i;
  })"""

JS_MOTION = """() => {
  const smooth = getComputedStyle(document.documentElement).scrollBehavior === 'smooth';
  const endless = document.getAnimations().filter(a => {
    const t = a.effect && a.effect.getTiming ? a.effect.getTiming() : {};
    return a.playState === 'running' && t.iterations === Infinity;
  }).length;
  return { smooth, endless };
}"""

JS_PRINT = """() => {
  const shown = sel => [...document.querySelectorAll(sel)].some(e => {
    const s = getComputedStyle(e); const r = e.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
  });
  const lum = c => {
    const m = c.match(/[\\d.]+/g); if (!m) return 1;
    const [r, g, b] = m.slice(0, 3).map(v => { v = v / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const bs = getComputedStyle(document.body);
  const bg = bs.backgroundColor === 'rgba(0, 0, 0, 0)' ? 'rgb(255, 255, 255)' : bs.backgroundColor;
  return {
    nav: ['#quarto-header', '#quarto-sidebar', '#quarto-margin-sidebar', 'footer.footer'].filter(shown),
    bg: lum(bg), fg: lum(bs.color),
  };
}"""


# ------------------------------------------------------------------ results
@dataclass
class Result:
    page: str
    scheme: str
    width: int
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    axe_minor: int = 0

    @property
    def key(self) -> str:
        return f"{self.scheme}-{self.width}"


async def check_page(
    ctx: Any,
    url: str,
    page_name: str,
    scheme: str,
    width: int,
    origin: str,
    axe_src: str,
    allow: dict,
) -> Result:
    res = Result(page_name, scheme, width)
    page = await ctx.new_page()
    foreign: set[str] = set()  # hosts other than the local server
    missing: dict[str, str] = {}  # local path -> what went wrong

    def on_request(req: Any) -> None:
        u = urlparse(req.url)
        if u.scheme in ("data", "blob", "about"):
            return
        if f"{u.scheme}://{u.netloc}" != origin:
            foreign.add(u.hostname or u.netloc)

    def on_response(resp: Any) -> None:
        if resp.url.startswith(origin) and resp.status >= 400:
            missing[urlparse(resp.url).path] = str(resp.status)

    def on_failed(req: Any) -> None:
        if req.url.startswith(origin):
            missing.setdefault(urlparse(req.url).path, req.failure or "failed")

    page.on("request", on_request)
    page.on("response", on_response)
    page.on("requestfailed", on_failed)
    try:
        await page.goto(url, wait_until="load")
        await page.evaluate(JS_SETTLE)
        state = await page.evaluate(JS_STATE)

        # theme
        if not state["reveal"] and state["theme"] != scheme:
            res.failures.append(f"theme: body is {state['theme']}, expected {scheme}")

        # katex
        if state["math"]:
            if not state["katex"]:
                res.failures.append("katex: window.katex missing on a page with math")
            if state["typeset"] < state["math"] or state["errors"]:
                res.failures.append(
                    f"katex: {state['typeset']}/{state['math']} formulas typeset, "
                    f"{state['errors']} KaTeX errors"
                )

        # reflow
        if not state["reveal"]:
            widths = [width] if width > 400 else [width, 320]
            for w in widths:
                if w != width:
                    await page.set_viewport_size({"width": w, "height": HEIGHT})
                    await page.evaluate(JS_SETTLE)
                ov = await page.evaluate(JS_OVERFLOW)
                if ov["over"] > 0:
                    res.failures.append(
                        f"reflow: page scrolls sideways by {ov['over']} px at {w} px; "
                        f"widest: {', '.join(ov['culprits']) or '?'}"
                    )
            if width <= 400:
                await page.set_viewport_size({"width": width, "height": HEIGHT})
                await page.evaluate(JS_SETTLE)

        # axe
        await page.add_script_tag(content=axe_src)
        axe = await page.evaluate(
            """(tags) => axe.run(document, {runOnly: {type: 'tag', values: tags},
                                            resultTypes: ['violations']})""",
            AXE_TAGS,
        )
        for v in axe["violations"]:
            targets = [
                ", ".join(t) if isinstance(t, list) else str(t)
                for n in v["nodes"]
                for t in n["target"]
            ]
            if v.get("impact") in FAIL_IMPACTS:
                if axe_allowed(allow, v["id"], page_name, targets):
                    res.warnings.append(f"axe {v['id']} allow-listed ({len(v['nodes'])} nodes)")
                    continue
                msg = (
                    f"axe {v['id']} ({v['impact']}): {v['help']} — {len(v['nodes'])} node(s): "
                    + "; ".join(targets[:3])
                )
                if VERBOSE:
                    for n in v["nodes"][:6]:
                        why = (n.get("failureSummary") or "").replace("\n", " ")
                        msg += f"\n      {n['target']}: {why}"
                res.failures.append(msg)
            else:
                res.axe_minor += len(v["nodes"])
                res.warnings.append(f"axe {v['id']} ({v.get('impact')}): {len(v['nodes'])} node(s)")

        # focus: real Tab presses from the top of the page, then a sample of the content
        await page.evaluate(JS_TIPPY_INSTANT)
        await page.evaluate(
            "() => { window.scrollTo(0, 0); document.activeElement && document.activeElement.blur(); }"
        )
        seen = []
        for _ in range(TAB_STOPS):
            await page.keyboard.press("Tab")
            info = await page.evaluate(f"({JS_FOCUS_INFO})(document.activeElement)")
            if info is None:
                break
            seen.append(info)
        if not state["reveal"]:
            seen += await page.evaluate(JS_FOCUS_SAMPLE, [FOCUS_SAMPLE, JS_FOCUS_INFO])
        bad = set()
        for f in seen:
            if not f["sized"]:
                bad.add(f"focus: {f['what']} takes focus but has no size")
            elif not (f["visible"] and (f["outline"] or f["shadow"])):
                bad.add(f"focus: no visible indicator on {f['what']} ({f['style']})")
            elif not f["shown"]:
                bad.add(f"focus: {f['what']} is entirely hidden when focused ({f['where']})")
        res.failures.extend(sorted(bad))

        # <summary> toggles with Enter and Space
        ids = await page.evaluate(JS_SUMMARIES, SUMMARY_SAMPLE)
        for i in ids:
            loc = page.locator(f'summary[data-site-check="{i}"]')
            await loc.focus()
            for key in ("Enter", "Space"):
                before = await loc.evaluate("s => s.parentElement.open")
                await page.keyboard.press(key)
                after = await loc.evaluate("s => s.parentElement.open")
                if before == after:
                    what = await loc.evaluate("s => s.textContent.trim().slice(0, 40)")
                    res.failures.append(f'summary: "{what}" does not toggle with {key}')

        # reduced motion (once per page)
        if scheme == "light" and width > 400:
            await page.emulate_media(reduced_motion="reduce")
            m = await page.evaluate(JS_MOTION)
            if m["smooth"] or m["endless"]:
                res.failures.append(f"motion: with reduced motion, {m}")
            await page.emulate_media(reduced_motion="no-preference")

        # print from dark mode (once per page)
        if scheme == "dark" and width > 400 and not state["reveal"]:
            await page.emulate_media(media="print")
            p = await page.evaluate(JS_PRINT)
            if p["nav"]:
                res.failures.append(f"print: still shows {', '.join(p['nav'])}")
            if p["bg"] < 0.8 or p["fg"] > 0.1:
                res.failures.append(
                    f"print: not light (background luminance {p['bg']:.2f}, text {p['fg']:.2f})"
                )
            await page.emulate_media(media="screen")

        await page.wait_for_load_state("networkidle")
    except Exception as e:  # report and carry on with the other pages
        res.failures.append(f"error: {type(e).__name__}: {str(e).splitlines()[0]}")
    finally:
        await page.close()
    for host in sorted(foreign):
        if network_allowed(allow, "hosts", host, page_name):
            res.warnings.append(f"network: request to {host} allow-listed ({page_name})")
        else:
            res.failures.append(f"network: request to {host}")
    for path, why in sorted(missing.items()):
        if path.endswith("/favicon.ico"):
            continue
        if network_allowed(allow, "missing", path, page_name):
            res.warnings.append(f"network: {why} {path} allow-listed")
        else:
            res.failures.append(f"network: {why} {path}")
    return res


async def run(site: Path, pages: list[str], axe_path: Path, allow: dict) -> list[Result]:
    from playwright.async_api import async_playwright

    base = base_path()
    srv, port = serve(site, base)
    origin = f"http://localhost:{port}"  # Quarto treats localhost links as internal
    axe_src = axe_path.read_text()
    results: list[Result] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            for scheme in SCHEMES:
                for width in WIDTHS:
                    ctx = await browser.new_context(
                        viewport={"width": width, "height": HEIGHT},
                        color_scheme=scheme,
                        reduced_motion="no-preference",
                    )
                    queue: asyncio.Queue[str] = asyncio.Queue()
                    for name in pages:
                        queue.put_nowait(name)

                    async def worker(
                        ctx: Any = ctx,
                        scheme: str = scheme,
                        width: int = width,
                        queue: asyncio.Queue[str] = queue,
                    ) -> None:
                        while not queue.empty():
                            name = queue.get_nowait()
                            url = f"{origin}{base}{name}"
                            results.append(
                                await check_page(
                                    ctx, url, name, scheme, width, origin, axe_src, allow
                                )
                            )

                    await asyncio.gather(*(worker() for _ in range(WORKERS)))
                    await ctx.close()
            await browser.close()
    finally:
        srv.shutdown()
    return results


def report(results: list[Result], axe_version: str) -> int:
    keys = [f"{s}-{w}" for s in SCHEMES for w in WIDTHS]
    pages = sorted({r.page for r in results})
    by = {(r.page, r.key): r for r in results}
    col = max(len(p) for p in pages) + 2
    print(
        f"site check: {len(pages)} pages x {len(keys)} configs, axe-core {axe_version}, tags {', '.join(AXE_TAGS)}"
    )
    print("page".ljust(col) + "".join(k.ljust(13) for k in keys))
    for p in pages:
        cells = []
        for k in keys:
            r = by.get((p, k))
            if r is None:
                cells.append("-")
            elif r.failures:
                cells.append(f"FAIL {len(r.failures)}")
            else:
                cells.append("ok" + (f" ({r.axe_minor} minor)" if r.axe_minor else ""))
        print(p.ljust(col) + "".join(c.ljust(13) for c in cells))
    failing = [r for r in results if r.failures]
    if failing:
        print("\nFailures:")
        for r in sorted(failing, key=lambda r: (r.page, r.key)):
            for f in r.failures:
                print(f"  {r.page} [{r.key}] {f}")
    warns = sorted({w for r in results for w in r.warnings})
    if warns:
        print("\nWarnings (moderate/minor axe findings, allow-listed items):")
        for w in warns:
            print(f"  {w}")
    n_fail = sum(len(r.failures) for r in results)
    print(f"\n{len(results)} page loads, {n_fail} failure(s)")
    return 1 if n_fail else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--site", default=str(ROOT / "_site"))
    ap.add_argument("--only", action="append", help="glob of pages to check (repeatable)")
    ap.add_argument("--axe", help="path to axe.min.js (default: Quarto's bundled copy)")
    ap.add_argument("--allow", default=str(ALLOW_FILE))
    ap.add_argument("--verbose", action="store_true", help="explain each failing axe node")
    args = ap.parse_args(argv)
    global VERBOSE
    VERBOSE = args.verbose
    site = Path(args.site).resolve()
    if not site.is_dir():
        print(f"{site} does not exist; render first")
        return 1
    pages = sorted(
        str(p.relative_to(site)).replace("\\", "/")
        for p in site.rglob("*.html")
        if "site_libs" not in p.parts
    )
    if args.only:
        pages = [p for p in pages if any(fnmatch.fnmatch(p, g) for g in args.only)]
    if not pages:
        print("no pages to check")
        return 1
    axe_path = find_axe(args.axe)
    head = axe_path.read_text()[:200]
    version = head.split("axe v", 1)[1].split()[0] if "axe v" in head else "?"
    allow = load_allow(Path(args.allow))
    results = asyncio.run(run(site, pages, axe_path, allow))
    return report(results, version)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
