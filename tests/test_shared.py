"""Shared reference code equals the solution it came from (PLAN.md §3.3).

A later lab pulls an earlier lab's algorithm in with `# %% include labs/shared/NAME.py`.
Each shared file names its origin on its first line, `# origin: labs/src/NN-slug.py`, and
every top-level function it defines must match, token for token (decorators aside), the
function of the same name in one of that source's solution cells. So a shared copy can
never drift from the solution participants wrote.
"""

import ast
import re
from pathlib import Path

import jupytext
import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = sorted((ROOT / "labs" / "shared").glob("*.py"))
ORIGIN = re.compile(r"^#\s*origin:\s*(labs/src/\S+\.py)\s*$")


def _defs(tree: ast.Module) -> dict[str, str]:
    """Top-level function sources, with decorators removed, keyed by name."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            out[node.name] = ast.dump(node, include_attributes=False)
    return out


def _solution_defs(source: Path) -> dict[str, str]:
    nb = jupytext.read(source)
    found: dict[str, str] = {}
    for cell in nb.cells:
        role = str(cell.metadata.get("role", ""))
        if cell.cell_type == "code" and re.match(r"^ex[0-9]+[a-z]?-sol$", role):
            found.update(_defs(ast.parse(cell.source)))
    return found


@pytest.mark.parametrize("path", SHARED, ids=[p.name for p in SHARED])
def test_shared_code_equals_its_origin_solution(path: Path):
    lines = path.read_text().splitlines()
    m = ORIGIN.match(lines[0]) if lines else None
    assert m, f"{path.name}: the first line must be '# origin: labs/src/NN-slug.py'"
    origin = ROOT / m.group(1)
    assert origin.exists(), f"{path.name}: origin {m.group(1)} does not exist"
    shared = _defs(ast.parse(path.read_text()))
    assert shared, f"{path.name} defines no top-level function"
    solutions = _solution_defs(origin)
    for name, dumped in shared.items():
        assert name in solutions, f"{path.name}: {name} is not a solution in {origin.name}"
        assert dumped == solutions[name], (
            f"{path.name}: {name} differs from the solution in {origin.name}; copy it again"
        )


def test_shared_files_are_the_only_includes():
    """Every `# %% include` in a lab source points at labs/shared/."""
    for src in sorted((ROOT / "labs" / "src").glob("*.py")):
        for line in src.read_text().splitlines():
            m = re.match(r"^#\s*%%\s*include\s+(\S+)", line)
            if m:
                assert m.group(1).startswith("labs/shared/"), f"{src.name}: {line}"
                assert (ROOT / m.group(1)).exists(), f"{src.name}: {m.group(1)} missing"
