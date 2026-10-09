"""Page-template lint. Placeholder module pages need only the generated parts;
written pages (status other than placeholder) get the full template check in Phase 2."""

import re
from pathlib import Path

import yaml

import variables

ROOT = Path(__file__).resolve().parents[1]
FULL_TEMPLATE = [
    r"\{\{< include /_includes/module-\d\d\.md >\}\}",
    r"\{\{< include /_includes/lab-\d\d\.md >\}\}",
    r"## Agenda \{#agenda\}",
    r"\{\{< include /_includes/live-\d\d\.md >\}\}",
    r"^## 1\. ",
    r"^## Summary",
    r"^### Equations and where you implement them",
    r"^## Gotchas and troubleshooting",
    r"^## Next steps",
    r"^## Further reading",
]


def front(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    return yaml.safe_load(m.group(1)) if m else {}


def test_every_module_has_a_page_with_its_includes():
    v = variables.load()
    for mid in variables.module_ids(v):
        m = v["modules"][mid]
        page = ROOT / "modules" / f"{m['slug']}.qmd"
        assert page.exists(), page
        text = page.read_text()
        nn = f"{m['n']:02d}"
        for inc in (f"module-{nn}", f"lab-{nn}", f"live-{nn}"):
            assert f"/_includes/{inc}.md" in text, f"{page.name} must include {inc}"


def test_written_pages_follow_the_template():
    for page in sorted((ROOT / "modules").glob("*.qmd")):
        text = page.read_text()
        if front(text).get("status") == "placeholder":
            continue
        positions = []
        for pattern in FULL_TEMPLATE:
            m = re.search(pattern, text, re.M)
            assert m, f"{page.name}: missing template element {pattern}"
            positions.append(m.start())
        assert positions == sorted(positions), f"{page.name}: template elements out of order"
        for sec in re.findall(r"^## \d+\. .*\n+(.*)", text, re.M):
            assert sec.startswith("**In this section:**"), (
                f"{page.name}: sections open with **In this section:**"
            )


def test_figures_have_alt_text():
    for qmd in ROOT.rglob("*.qmd"):
        if "_site" in qmd.parts or ".claude" in qmd.parts:
            continue
        for img in re.findall(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", qmd.read_text()):
            assert "fig-alt=" in img, f"{qmd.name}: every figure needs fig-alt"
