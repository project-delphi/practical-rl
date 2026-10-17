"""A module page's code equals the lab solution of the same name.

Module pages show reference code in ```python blocks (for example "Code: value_iteration
(open after Lab 2)"). Every top-level function there whose name matches a solution in the
lab (labs/src/NN-slug.py, cells exN-sol and stretch-sol) must equal that solution, token
for token, decorators and docstrings aside. So the page can never show code that fails the lab's
checkpoints (the Module 2 UI review found the page's policy_iteration had drifted).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import jupytext
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "modules").glob("[0-9][0-9]-*.qmd"))
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)

# Pages whose code predates this test (strict xfail: remove an entry once its page is fixed).
KNOWN_DRIFT: dict[str, str] = {}


def _defs(source: str) -> dict[str, str]:
    """Top-level function definitions as ast dumps keyed by name.

    Decorators and the docstring are removed: a page may show the code without the lab's
    docstring, but every statement must be the solution's.
    """
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
            out[node.name] = ast.dump(node, include_attributes=False)
    return out


def _solutions(lab: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for cell in jupytext.read(lab).cells:
        role = str(cell.metadata.get("role", ""))
        if cell.cell_type == "code" and re.match(r"^(ex[0-9]+[a-z]?|stretch)-sol$", role):
            found.update(_defs(cell.source))
    return found


def _page_defs(page: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for block in BLOCK.findall(page.read_text()):
        found.update(_defs(block))
    return found


@pytest.mark.parametrize(
    "page",
    [
        pytest.param(p, marks=pytest.mark.xfail(reason=KNOWN_DRIFT[p.stem], strict=True))
        if p.stem in KNOWN_DRIFT
        else p
        for p in PAGES
    ],
    ids=[p.stem for p in PAGES],
)
def test_page_code_equals_the_lab_solution(page: Path):
    lab = ROOT / "labs" / "src" / f"{page.stem}.py"
    if not lab.exists():
        pytest.skip(f"no lab source for {page.name}")
    solutions = _solutions(lab)
    shown = {name: dump for name, dump in _page_defs(page).items() if name in solutions}
    for name, dump in shown.items():
        assert dump == solutions[name], (
            f"{page.name}: the code of {name} differs from the solution in {lab.name}; "
            "copy the solution onto the page again (decorators aside)"
        )


def test_the_lab_greedy_is_the_fixtures_tie_rule():
    """Lab 2's reference greedy and make_expected._greedy pick the same action, ties included."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_expected

    ns: dict = {"np": __import__("numpy")}
    exec(_solution_source(ROOT / "labs" / "src" / "02-planning.py", "greedy"), ns)
    np = ns["np"]
    rng = np.random.default_rng(0)
    cases = [rng.normal(size=(6, 4)) for _ in range(20)]
    tied = rng.normal(size=(6, 4)).round(1)  # exact ties after rounding
    tied[2, 1] = tied[2, 3] = tied[2].max() + 1
    split = tied.copy()
    split[2, 3] += np.spacing(split[2, 3]) * 3  # a tie split by a few ulps, as on some BLAS
    for Q in [*cases, tied, split]:
        assert np.array_equal(ns["greedy"](Q), make_expected._greedy(Q))
    assert ns["greedy"](split)[2] == 1  # the first action within rounding, not the argmax


def _solution_source(lab: Path, name: str) -> str:
    for cell in jupytext.read(lab).cells:
        role = str(cell.metadata.get("role", ""))
        if cell.cell_type == "code" and role.endswith("-sol") and f"def {name}(" in cell.source:
            tree = ast.parse(cell.source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    node.decorator_list = []
                    return ast.unparse(node)
    raise LookupError(f"{name} is not a solution in {lab.name}")
