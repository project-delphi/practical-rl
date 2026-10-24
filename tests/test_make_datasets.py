"""scripts/make_datasets.py: registered builders -> deterministic datasets with provenance.

Everything here lives in temporary folders with a test registry; the package ships no
test datasets.
"""

import json

import numpy as np
import pytest

import make_datasets as md
import package_data as pd
from prl import data

RID = "2026-10-09-local-m99-sweep-exp"


def write_record(runs, record_id, rec):
    (runs / f"{record_id}.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def repo(tmp_path, experiment_record):
    runs, datasets = tmp_path / "runs", tmp_path / "datasets"
    runs.mkdir()
    datasets.mkdir()
    write_record(runs, RID, experiment_record(40))
    registry = {}

    @md.dataset(
        "m99_sweep",
        description="Test sweep from a record.",
        modules=["m99"],
        license="CC BY 4.0",
        sources=[RID],
        registry=registry,
    )
    def m99_sweep(records):
        rec = records[RID]
        return {"returns": np.array(rec["per_seed"]["return"]), "seeds": np.array(rec["seeds"])}

    @md.dataset(
        "m99_noise",
        description="Seeded noise.",
        modules=["m99"],
        license="CC BY 4.0",
        seeds=[7],
        registry=registry,
    )
    def m99_noise(records):
        assert records == {}
        return {"z": np.random.default_rng(7).normal(size=5)}

    return runs, datasets, registry


def build(repo, check=False):
    runs, datasets, registry = repo
    return md.build(registry, runs=runs, datasets=datasets, check=check)


def test_build_writes_datasets_that_prl_loads(repo, monkeypatch):
    runs, datasets, _ = repo
    assert build(repo, check=True)
    build(repo)
    assert build(repo, check=True) == []
    manifest = json.loads((datasets / "manifest.json").read_text())
    entry = manifest["m99_sweep"]
    blob = (datasets / "m99_sweep.npz").read_bytes()
    assert entry == {
        "kind": "dataset",
        "file": "m99_sweep.npz",
        "sha256": pd.sha256(blob),
        "bytes": len(blob),
        "provenance": "m99_sweep.provenance.json",
    }
    prov = json.loads((datasets / "m99_sweep.provenance.json").read_text())
    rec = json.loads((runs / f"{RID}.json").read_text())
    assert prov["sources"] == [{"record": RID, "record_sha": pd.record_sha(rec)}]
    assert prov["arrays"]["returns"] == {"dtype": "float64", "shape": [3, 4]}
    assert prov["sha256"] == entry["sha256"] and prov["license"] == "CC BY 4.0"
    assert json.loads((datasets / "m99_noise.provenance.json").read_text())["seeds"] == [7]

    monkeypatch.setattr(data, "_root", lambda: datasets)
    assert data.load("m99_sweep")["returns"].shape == (3, 4)


def test_check_fails_when_a_source_record_changes(repo, experiment_record):
    runs, datasets, _ = repo
    build(repo)
    write_record(runs, RID, experiment_record(40, note="added later"))
    assert build(repo, check=True) == [f"out of date: {datasets / 'm99_sweep.provenance.json'}"]


def test_float_noise_keeps_the_committed_file(repo):
    _, datasets, _ = repo
    build(repo)
    path = datasets / "m99_noise.npz"
    z = np.random.default_rng(7).normal(size=5)
    path.write_bytes(pd.save_npz(z=z * (1 + 4e-16)))  # as if built on another platform
    noisy = path.read_bytes()
    build(repo)  # the manifest and provenance now describe the kept file
    assert path.read_bytes() == noisy and build(repo, check=True) == []
    assert json.loads((datasets / "manifest.json").read_text())["m99_noise"]["sha256"] == (
        pd.sha256(noisy)
    )


def test_unregistered_files_are_removed_and_bands_kept(repo):
    _, datasets, registry = repo
    band = {"kind": "band", "name": "m99_return", "file": "bands/m99_return.live.npz"}
    (datasets / "manifest.json").write_text(json.dumps({"bands/m99_return.live": band}))
    build(repo)
    del registry["m99_noise"]
    assert len(build(repo, check=True)) == 3  # the npz, its provenance and the manifest
    build(repo)
    assert sorted(p.name for p in datasets.iterdir()) == [
        "m99_sweep.npz",
        "m99_sweep.provenance.json",
        "manifest.json",
    ]
    manifest = json.loads((datasets / "manifest.json").read_text())
    assert manifest["bands/m99_return.live"] == band and "m99_noise" not in manifest


def test_registration_and_builder_output_are_checked(repo):
    _, _, registry = repo
    with pytest.raises(ValueError, match="registered twice"):
        md.dataset("m99_sweep", description="d", modules=["m99"], license="l", registry=registry)(
            lambda r: {}
        )
    with pytest.raises(ValueError, match="needs a description, a license"):
        md.dataset("m99_x", description="d", modules=[], license="l", registry=registry)(
            lambda r: {}
        )
    md.dataset("m99_bad", description="d", modules=["m99"], license="l", registry=registry)(
        lambda r: {"x": np.array([{}], dtype=object)}
    )
    with pytest.raises(ValueError, match="object array"):
        build(repo)
