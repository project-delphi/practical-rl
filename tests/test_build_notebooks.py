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
    assert ids[:3] == ["gen-header", "gen-setup", "gen-init"] and ids[-2:] == [
        "gen-finish",
        "gen-next",
    ]
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


def test_lint_requires_predict_answer_and_cited_explain_ref(tmp_path):
    v = variables.load()
    src = (ROOT / "labs" / "src" / "01-mdps-bellman.py").read_text()
    target = tmp_path / "01-mdps-bellman.py"
    target.write_text(src.replace('(Gotcha: "γ is a speed knob")', "(no citation)"))
    del v["modules"]["m01"]["exercises"][1]["predict_answer"]  # exercise 2
    problems = bn.lint(bn.build(target, v)[1], v)
    assert any("exercise 1: the Explain does not cite" in p for p in problems)
    assert any("exercise 2: _variables.yml needs a predict_answer" in p for p in problems)


def test_notebooks_end_with_a_link_to_the_next_module():
    """PLAN §8: the last cell links to the next module, also right after the finish cell."""
    import json

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        ids = [c["id"] for c in cells]
        assert ids[ids.index("gen-finish") + 1] == "gen-next", path.name
        assert ids[-1] in ("gen-next", "gen-next-end"), path.name
        assert "**Next:** [Module" in "".join(cells[-1]["source"]), path.name


def test_an_earlier_labs_include_is_folded_under_its_lab_without_the_origin_note():
    nb = json.loads(built("06-exploration-mdps.py")["text"])
    cell = next(c for c in nb["cells"] if c["id"] == "shared-value-iteration")
    src = "".join(cell["source"])
    assert src.startswith("#@title From Lab 2: value_iteration.py, run this cell")
    assert "# origin:" not in src and "def value_iteration" in src
    assert cell["metadata"]["cellView"] == "form" and cell["metadata"]["jupyter"]["source_hidden"]


def test_folding_an_include_leaves_content_sha_unchanged(monkeypatch):
    folded = built("06-exploration-mdps.py")
    monkeypatch.setattr(bn, "_earlier_include", lambda src: None)
    plain = built("06-exploration-mdps.py")
    assert folded["sha"] == plain["sha"] and folded["text"] != plain["text"]


def test_a_provided_cell_with_a_form_title_is_folded(tmp_path):
    v = variables.load()
    src = (ROOT / "labs" / "src" / "01-mdps-bellman.py").read_text()
    target = tmp_path / "01-mdps-bellman.py"
    target.write_text(
        src + '\n# %%\n#@title Open later { display-mode: "form" }\nANSWER = 42\n'
        "\n# %%\n# A comment first: not a form title\nOTHER = 1\n"
    )
    nb = json.loads(bn.build(target, v)[1]["text"])
    cells = {
        "".join(c["source"]).splitlines()[0]: c for c in nb["cells"] if c["cell_type"] == "code"
    }
    assert cells['#@title Open later { display-mode: "form" }']["metadata"]["cellView"] == "form"
    assert "cellView" not in cells["# A comment first: not a form title"]["metadata"]
