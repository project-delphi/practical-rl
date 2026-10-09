# Vendored fonts and math typesetting

Everything the site needs to draw text and math is served from the site itself:
no font service, no CDN. `bash fonts/build.sh` rebuilds the fonts from the pinned
releases below (it checks each download's SHA-256).

## Fonts (this directory)

| File | Family | Upstream release | License |
|---|---|---|---|
| `inter-roman.woff2` | Inter 4.1 (variable; weight 400–700, optical size fixed at 14) | <https://github.com/rsms/inter/releases/tag/v4.1> (`Inter-4.1.zip`, `InterVariable.ttf`; SHA-256 of the zip `9883fdd4…b11e`) | SIL Open Font License 1.1, [`OFL-Inter.txt`](OFL-Inter.txt) |
| `inter-italic.woff2` | Inter 4.1 Italic (as above) | same zip, `InterVariable-Italic.ttf` | SIL Open Font License 1.1, [`OFL-Inter.txt`](OFL-Inter.txt) |
| `jetbrains-mono.woff2` | JetBrains Mono 2.304 (variable; weight 400–700) | <https://github.com/JetBrains/JetBrainsMono/releases/tag/v2.304> (`JetBrainsMono-2.304.zip`, `fonts/variable/JetBrainsMono[wght].ttf`; SHA-256 of the zip `6f6376c6…7bbf`) | SIL Open Font License 1.1, [`OFL-JetBrainsMono.txt`](OFL-JetBrainsMono.txt) |
| `noto-sans-math.woff2` | Noto Sans Math 3.000 (regular), math blocks only | <https://github.com/notofonts/math/releases/tag/NotoSansMath-v3.000> (`NotoSansMath-v3.000.zip`, `NotoSansMath/unhinted/ttf/NotoSansMath-Regular.ttf`; SHA-256 of the zip `ac351837…8356`) | SIL Open Font License 1.1, [`OFL-NotoSansMath.txt`](OFL-NotoSansMath.txt) |

- None of the licenses declares a **Reserved Font Name**, so the subset files
  keep their family names. The OFL allows modification and redistribution (including subsetting)
  when the license travels with the fonts, which these files do.
- **Subset** (`fonts/build.sh`, fontTools 4.65.0, WOFF2 with Brotli, hinting dropped):
  Basic Latin, Latin-1, Latin Extended-A/B/Additional, spacing modifiers and combining
  marks, Greek and Coptic (U+0370–03FF), general punctuation, super- and subscripts,
  currency, letterlike symbols, number forms, **arrows (U+2190–21FF)**, **mathematical
  operators (U+2200–22FF)**, miscellaneous technical, geometric shapes, dingbats
  (for the chips' icons) and Latin ligatures. A block keeps only the glyphs the font
  has. Inter has only 18 of the 256 mathematical operators (no ∈, ∀, ∇, ∝, ⋅), so
  **Noto Sans Math** follows Inter and JetBrains Mono in both font stacks, subset to
  letterlike symbols, arrows, operators, miscellaneous technical, miscellaneous math
  A and supplemental arrows A, and script and double-struck letters (𝒜, 𝔼). Its
  `@font-face` has a matching `unicode-range`, so it downloads only on pages that use
  those symbols. `coverage.json` lists the code points each file covers;
  `tests/test_site_fonts.py` checks every character of the rendered pages against it.
- **Payload:** 304,776 bytes for all four files (Inter roman 96 KB, italic 103 KB,
  JetBrains Mono 56 KB, Noto Sans Math 49 KB). A page loads only the faces it uses.
- **Reproducible:** `SOURCE_DATE_EPOCH` is fixed, so two runs of `fonts/build.sh`
  give byte-identical files.
- **Not covered:** U+25D0 ◐ (the readiness "Real path elsewhere" chip icon,
  `aria-hidden`) is in none of these fonts; the browser draws it from a system font.
- Wiring: `styles/prl.scss` declares the `@font-face` rules with
  `url("fonts/…")`; Quarto resolves those paths against the project root and copies
  the files next to the compiled theme (`_site/site_libs/bootstrap/fonts/`).

## KaTeX (`filters/vendor/katex/`)

| What | Version | Source | License |
|---|---|---|---|
| `katex.min.js`, `katex.min.css`, `fonts/*.woff2` (20 files, 296 KB) | KaTeX 0.18.9 (npm, published 2026-09-23) | <https://registry.npmjs.org/katex/-/katex-0.18.9.tgz> (`dist/`; npm integrity `sha512-8ad9RyoK…OakTQ==`, checked) | MIT, [`filters/vendor/katex/LICENSE`](../filters/vendor/katex/LICENSE) |

- Only the WOFF2 fonts are vendored; `katex.min.css` is unmodified, so its WOFF and
  TTF fallbacks point at files that are not shipped. Every browser the site supports
  takes the WOFF2 source first.
- The npm package declares MIT for the whole distribution, fonts included
  (`package.json`, `LICENSE`); its source also notes that some stretchy-glyph paths
  come from fonts under the SIL OFL 1.1. The package has no separate font license
  file, so nothing more was checked. The fonts load only for glyphs a page's math uses.
- `filters/katex.lua` registers the copy as a Quarto HTML dependency (it lands in
  `_site/site_libs/quarto-contrib/katex-0.18.9/`); `filters/katex-render.js`
  typesets the page. To upgrade, replace the files from a newer tarball and change
  `KATEX_VERSION` in `filters/katex.lua`.
