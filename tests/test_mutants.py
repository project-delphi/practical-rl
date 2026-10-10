"""Mutant files must parse and target real exercises and functions."""

from pathlib import Path

import check_notebooks
import variables

ROOT = Path(__file__).resolve().parents[1]


def test_mutants_target_declared_exercises_and_functions():
    v = variables.load()
    for path in sorted((ROOT / "labs" / "mutants").glob("m*.py")):
        module = path.stem
        mod = v["modules"][module]
        writes = {str(e["n"]): set(e["writes"]) for e in mod["exercises"]}
        mutants = check_notebooks.load_mutants(module)
        assert mutants, f"{path.name} declares no mutants"
        ids = set()
        for m in mutants:
            assert m["ex"] in writes, f"{path.name}: {m['id']} targets unknown exercise {m['ex']}"
            (name,) = m["sources"]
            assert name in writes[m["ex"]], (
                f"{path.name}: {m['id']} replaces {name}, not written in ex {m['ex']}"
            )
            key = (m["ex"], m["id"])
            assert key not in ids, f"{path.name}: duplicate mutant id {key}"
            ids.add(key)


def test_every_lab_with_writes_has_mutants_once_drafted():
    v = variables.load()
    for mid, mod in v["modules"].items():
        if mod["status"]["lab"] in ("draft", "reviewed", "done") and any(
            e["writes"] for e in mod["exercises"]
        ):
            if mid == "m00":
                continue  # pre-work: stubs only
            assert (ROOT / "labs" / "mutants" / f"{mid}.py").exists(), (
                f"{mid} needs labs/mutants/{mid}.py"
            )
