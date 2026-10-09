#!/usr/bin/env bash
# Rebuild the vendored web fonts in fonts/ from their upstream releases.
#
#   bash fonts/build.sh
#
# Downloads each release (pinned URL and SHA-256), pins the variable axes to the
# range the site uses (weight 400-700; Inter's optical size at 14), subsets to
# Latin, Latin Extended, Greek, punctuation, arrows (U+2190-21FF) and
# mathematical operators (U+2200-22FF), and writes WOFF2. Noto Sans Math fills
# in the math symbols Inter lacks (it is subset to math blocks only and loads only
# on pages that use them). Then it writes
# fonts/coverage.json (the code points each file covers), which
# tests/test_site_fonts.py checks the rendered text against.
# Sources and licenses are recorded in fonts/LICENSES.md.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

FONTTOOLS="fonttools[woff]==4.65.0"
# fontTools stamps head.modified with this instead of the clock: byte-identical rebuilds.
export SOURCE_DATE_EPOCH=1704067200
ft() { uvx --quiet --from "$FONTTOOLS" fonttools "$@"; }
subset() { uvx --quiet --from "$FONTTOOLS" pyftsubset "$@"; }

INTER_URL="https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"
INTER_SHA="9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e"
JBM_URL="https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip"
JBM_SHA="6f6376c6ed2960ea8a963cd7387ec9d76e3f629125bc33d1fdcd7eb7012f7bbf"
MATH_URL="https://github.com/notofonts/math/releases/download/NotoSansMath-v3.000/NotoSansMath-v3.000.zip"
MATH_SHA="ac351837b41f8a897f020b97fb0f075ad574c1e9669fb5839ada1f92fd748356"

# Latin, Latin-1, Latin Extended-A/B/Additional, IPA schwa, spacing modifiers,
# combining marks, Greek and Coptic, general punctuation, super/subscripts,
# currency, letterlike symbols, number forms, arrows, mathematical operators,
# miscellaneous technical, geometric shapes, dingbats (for the chips' icons:
# only glyphs a font actually has are kept), Latin Extended-C/D, ligatures.
UNICODES="U+0020-007E,U+00A0-00FF,U+0100-024F,U+0259,U+02B0-02FF,U+0300-036F,\
U+0370-03FF,U+1E00-1EFF,U+2000-206F,U+2070-209F,U+20A0-20C0,U+2100-214F,\
U+2150-218F,U+2190-21FF,U+2200-22FF,U+2300-23FF,U+25A0-25FF,U+2600-26FF,\
U+2700-27BF,U+2C60-2C7F,U+A720-A7FF,U+FB00-FB06,U+FEFF,U+FFFD"
# Math symbols only (keep in sync with the unicode-range in styles/prl.scss):
# letterlike, arrows, operators, misc technical, misc math A and supplemental
# arrows A (angle brackets, long arrows), script and double-struck letters.
MATH_UNICODES="U+2100-214F,U+2190-21FF,U+2200-22FF,U+2300-23FF,U+27C0-27FF,\
U+1D49C-1D4CF,U+1D538-1D56B"

fetch() { # url sha256 dest
  curl -fsSL -o "$3" "$1"
  echo "$2  $3" | shasum -a 256 -c - >/dev/null
}

fetch "$INTER_URL" "$INTER_SHA" "$work/inter.zip"
fetch "$JBM_URL" "$JBM_SHA" "$work/jbm.zip"
fetch "$MATH_URL" "$MATH_SHA" "$work/math.zip"
mkdir -p "$work/inter" "$work/jbm" "$work/math"
unzip -q -o "$work/inter.zip" InterVariable.ttf InterVariable-Italic.ttf LICENSE.txt -d "$work/inter"
unzip -q -o "$work/jbm.zip" 'fonts/variable/*' OFL.txt -d "$work/jbm"
unzip -q -o "$work/math.zip" NotoSansMath/unhinted/ttf/NotoSansMath-Regular.ttf OFL.txt -d "$work/math"

make() { # source.ttf axis-limits... -- output.woff2
  local src="$1"; shift
  local limits=()
  while [ "$1" != "--" ]; do limits+=("$1"); shift; done
  local out="$2"
  ft varLib.instancer "$src" "${limits[@]}" -q -o "$work/instanced.ttf"
  subset "$work/instanced.ttf" \
    --unicodes="$UNICODES" \
    --layout-features+=tnum,case \
    --no-hinting \
    --flavor=woff2 \
    --output-file="$here/$out"
  echo "wrote fonts/$out ($(wc -c <"$here/$out" | tr -d ' ') bytes)"
}

make "$work/inter/InterVariable.ttf" wght=400:700 opsz=14 -- inter-roman.woff2
make "$work/inter/InterVariable-Italic.ttf" wght=400:700 opsz=14 -- inter-italic.woff2
make "$work/jbm/fonts/variable/JetBrainsMono[wght].ttf" wght=400:700 -- jetbrains-mono.woff2
subset "$work/math/NotoSansMath/unhinted/ttf/NotoSansMath-Regular.ttf" \
  --unicodes="$MATH_UNICODES" \
  --no-hinting \
  --flavor=woff2 \
  --output-file="$here/noto-sans-math.woff2"
echo "wrote fonts/noto-sans-math.woff2 ($(wc -c <"$here/noto-sans-math.woff2" | tr -d ' ') bytes)"

cp "$work/inter/LICENSE.txt" "$here/OFL-Inter.txt"
cp "$work/jbm/OFL.txt" "$here/OFL-JetBrainsMono.txt"
cp "$work/math/OFL.txt" "$here/OFL-NotoSansMath.txt"

uv run --quiet --no-project --with "$FONTTOOLS" python -I - "$here" <<'PY'
import json, sys
from pathlib import Path
from fontTools.ttLib import TTFont

here = Path(sys.argv[1])
out = {}
for f in sorted(here.glob("*.woff2")):
    cmap = TTFont(f).getBestCmap()
    cps = sorted(cmap)
    ranges, start, prev = [], cps[0], cps[0]
    for c in cps[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        ranges.append(f"{start:04X}" if start == prev else f"{start:04X}-{prev:04X}")
        if c is not None:
            start = prev = c
    out[f.name] = ranges
(here / "coverage.json").write_text(json.dumps(out, indent=1) + "\n")
print("wrote fonts/coverage.json")
PY

total=$(cat "$here"/*.woff2 | wc -c | tr -d ' ')
echo "total font payload: $total bytes"
