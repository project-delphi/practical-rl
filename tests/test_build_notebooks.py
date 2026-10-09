import json
from pathlib import Path

import build_notebooks as bn
import variables

ROOT = Path(__file__).resolve().parents[1]


def built(name):
    return bn.build(ROOT / "labs" / "src" / name, variables.load())[1]


def test_lint_is_clean():
    v = variables.load()
    for src in sorted((ROOT / "labs" / "src").glob("*.py")):
        _, b = bn.build(src, v)
        assert bn.lint(b, v) == [], src.name


def test_build_is_deterministic():
    a, b = built("00-setup.py"), built("00-setup.py")
    assert a["text"] == b["text"]


def test_generated_cells_and_solution_folds():
    nb = json.loads(built("00-setup.py")["text"])
    ids = [c["id"] for c in nb["cells"]]
    assert ids[:3] == ["gen-header", "gen-setup", "gen-init"] and ids[-1] == "gen-finish"
    sol = next(c for c in nb["cells"] if c["id"] == "ex1-sol")
    assert sol["metadata"]["cellView"] == "form" and sol["metadata"]["jupyter"]["source_hidden"]
    assert "".join(sol["source"]).startswith("#@title Solution 1")
    assert all(c["metadata"]["id"] == c["id"] for c in nb["cells"])
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_content_sha_ignores_prose_but_not_code(tmp_path, monkeypatch):
    v = variables.load()
    src = (ROOT / "labs" / "src" / "01-mdps-bellman.py").read_text()
    base = bn.build(ROOT / "labs" / "src" / "01-mdps-bellman.py", v)[1]["sha"]
    target = tmp_path / "01-mdps-bellman.py"
    target.write_text(src.replace("Work it out before you code.", "Work it out first."))
    assert bn.build(target, v)[1]["sha"] == base
    target.write_text(
        src.replace(
            "print(discounted_return([1, 1, 1], 0.9))", "print(discounted_return([1, 1], 0.9))"
        )
    )
    assert bn.build(target, v)[1]["sha"] != base


def test_lint_catches_missing_checkpoint(tmp_path):
    v = variables.load()
    src = (ROOT / "labs" / "src" / "01-mdps-bellman.py").read_text()
    cut = src.split('# %% role="ex1-chk"')[0]
    target = tmp_path / "01-mdps-bellman.py"
    target.write_text(cut)
    problems = bn.lint(bn.build(target, v)[1], v)
    assert any("missing a chk cell" in p for p in problems)
