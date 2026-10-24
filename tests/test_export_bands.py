"""scripts/export_bands.py: records in runs/ -> deterministic band files and manifest entries.

Everything here lives in temporary folders; the package ships no test bands.
"""

import json

import numpy as np
import pytest

import export_bands as eb
import package_data as pd
from prl import data

SPEC = """
m99_return:
  metric: Mean return of the last 10 episodes
  x: budget.grid
  runs: per_seed.return
  budget: [steps]
  records:
    live: 2026-10-09-local-m99-bands-live-exp
    quick: 2026-10-09-local-m99-bands-quick-exp
"""


@pytest.fixture
def repo(tmp_path, experiment_record):
    """runs/ with a band spec and its two records, and an empty datasets folder."""
    runs, datasets = tmp_path / "runs", tmp_path / "datasets"
    runs.mkdir()
    datasets.mkdir()
    (runs / "bands.yml").write_text(SPEC)
    write_record(runs, "2026-10-09-local-m99-bands-live-exp", experiment_record(400))
    write_record(runs, "2026-10-09-local-m99-bands-quick-exp", experiment_record(40))
    return runs, datasets


def write_record(runs, record_id, rec):
    (runs / f"{record_id}.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")


def export(repo, check=False):
    runs, datasets = repo
    return eb.export(spec=runs / "bands.yml", runs=runs, datasets=datasets, check=check)


def test_export_writes_bands_that_prl_loads(repo, monkeypatch):
    runs, datasets = repo
    assert export(repo, check=True)  # nothing exported yet
    export(repo)
    assert export(repo, check=True) == []  # deterministic: a rerun changes nothing
    manifest = json.loads((datasets / "manifest.json").read_text())
    entry = manifest["bands/m99_return.live"]
    rec = json.loads((runs / "2026-10-09-local-m99-bands-live-exp.json").read_text())
    assert entry["budget"] == {"steps": 400} and entry["seeds"] == [0, 1, 2]
    assert entry["record"] == "2026-10-09-local-m99-bands-live-exp"
    assert entry["record_sha"] == pd.record_sha(rec)
    blob = (datasets / entry["file"]).read_bytes()
    assert entry["sha256"] == pd.sha256(blob) and entry["bytes"] == len(blob)

    monkeypatch.setattr(data, "_root", lambda: datasets)
    b = data.band("m99_return", {"steps": 40, "eval_episodes": 2})
    assert b["x"].tolist() == [10, 20, 30, 40] and b["x"].dtype == np.int64
    assert b["runs"].tolist() == [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    assert b["metric"] == "Mean return of the last 10 episodes"


def test_check_fails_when_a_record_or_the_spec_changes(repo, experiment_record):
    runs, datasets = repo
    export(repo)
    rid = "2026-10-09-local-m99-bands-quick-exp"
    write_record(runs, rid, experiment_record(40, note="added later"))
    assert export(repo, check=True) == [f"out of date: {datasets / 'manifest.json'}"]
    write_record(runs, rid, experiment_record(40, per_seed={"return": [[9.0] * 4] * 3}))
    stale = export(repo, check=True)
    assert any("m99_return.quick.npz" in line for line in stale)
    export(repo)
    (runs / "bands.yml").write_text(SPEC.replace("    quick: " + rid + "\n", ""))
    assert export(repo, check=True)[-1].startswith("no longer generated")
    export(repo)
    assert not (datasets / "bands" / "m99_return.quick.npz").exists()
    assert "bands/m99_return.quick" not in json.loads((datasets / "manifest.json").read_text())


def test_other_manifest_entries_are_kept_and_a_conflicted_manifest_is_rebuilt(repo):
    runs, datasets = repo
    dataset_entry = {"kind": "dataset", "file": "x.npz", "sha256": "0", "bytes": 1}
    (datasets / "manifest.json").write_text(json.dumps({"x": dataset_entry}))
    export(repo)
    manifest = json.loads((datasets / "manifest.json").read_text())
    assert manifest["x"] == dataset_entry and len(manifest) == 3
    (datasets / "manifest.json").write_text("<<<<<<< HEAD\n{}")
    assert any("not valid JSON" in line for line in export(repo, check=True))
    export(repo)
    assert len(json.loads((datasets / "manifest.json").read_text())) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(seeds=[0, 1], per_seed={"return": [[1.0] * 4] * 2}), "at least 3 distinct seeds"),
        (dict(per_seed={"return": [[1.0] * 4, [1.0] * 3, [1.0] * 4]}), "seed 1's curve"),
        (dict(per_seed={"return": [[1.0, None, 1.0, 1.0]] * 3}), "finite numbers"),
        (dict(per_seed={"return": [[1.0] * 4] * 2}), "one list per seed"),
        (dict(budget={"grid": [10, 20, 30, 40]}), "budget has no steps"),
        (dict(budget={"steps": 40, "grid": [10, 30, 20, 40]}), "increase strictly"),
        (dict(per_seed={}), "the record has no per_seed.return"),
        (dict(kind="notebook"), "not an experiment record"),
        (dict(date="2030-01-01"), "not a valid record"),
    ],
)
def test_bad_records_are_refused(repo, experiment_record, change, message):
    runs, _ = repo
    write_record(runs, "2026-10-09-local-m99-bands-quick-exp", experiment_record(40, **change))
    with pytest.raises(eb.SpecError, match=message):
        export(repo)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (SPEC.replace("m99_return", "M99-return"), "lowercase"),
        (SPEC.replace("  metric: Mean return of the last 10 episodes\n", ""), "exactly the keys"),
        (SPEC.replace("budget: [steps]", "budget: steps"), "budget must list keys"),
        (SPEC.replace("budget: [steps]", "budget: [lr]"), "same budget"),
        (SPEC.replace("live: 2026-10-09-local-m99-bands-live-exp", "live: nope"), "no record"),
        ("[1, 2]", "must map band names"),
    ],
)
def test_bad_specs_are_refused(repo, spec, message):
    runs, _ = repo
    (runs / "bands.yml").write_text(spec)
    with pytest.raises(eb.SpecError, match=message):
        export(repo)


def test_paths_may_be_key_lists_for_keys_with_dots(repo, experiment_record):
    runs, datasets = repo
    rid = "2026-10-09-local-m99-bands-quick-exp"
    rec = experiment_record(40)
    rec["per_seed"] = {"return.v2": rec["per_seed"]["return"]}
    write_record(runs, rid, rec)
    spec = SPEC.replace("runs: per_seed.return", 'runs: [per_seed, "return.v2"]')
    spec = spec.replace("    live: 2026-10-09-local-m99-bands-live-exp\n", "")
    (runs / "bands.yml").write_text(spec)
    export(repo)
    assert (datasets / "bands" / "m99_return.quick.npz").exists()
