"""prl.data.band: the recorded band at the lab's budget, sha-checked, ready for with_recorded.

Bands here are written into a temporary folder; prl ships no test bands.
"""

import hashlib
import io
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from prl import _budget, data, plot, runtime

LIVE = {"steps": 400, "log_every": 100, "eval_episodes": 10}
QUICK = {"steps": 40, "log_every": 10, "eval_episodes": 2}


def _npz(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _band(root, name, label, budget, steps, seeds=(0, 1, 2), offset=0.0):
    x = np.arange(steps // 4, steps + 1, steps // 4, dtype=np.int64)
    runs = np.array([np.linspace(0, 1, x.size) + s + offset for s in seeds])
    blob = _npz(x=x, runs=runs, seeds=np.array(seeds, dtype=np.int64))
    file = f"bands/{name}.{label}.npz"
    (root / "bands").mkdir(exist_ok=True)
    (root / file).write_bytes(blob)
    return f"bands/{name}.{label}", {
        "kind": "band",
        "name": name,
        "label": label,
        "file": file,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "record": f"2026-10-26-local-{name}-{label}-exp",
        "record_sha": "0" * 64,
        "budget": budget,
        "seeds": list(seeds),
        "metric": "return",
        "hardware": {
            "platform": "local",
            "env": "local",
            "cpu": "x",
            "n_cpu": 2,
            "accel": "cpu",
            "gpu": None,
            "torch_threads": 1,
        },
        "seconds": {"total": 12.5, "per_seed": None},
    }


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """A datasets folder with one band at two budgets and one dataset."""
    entries = dict(
        [
            _band(tmp_path, "m99_return", "live", {"steps": 400, "log_every": 100}, 400),
            _band(tmp_path, "m99_return", "quick", {"steps": 40, "log_every": 10}, 40),
        ]
    )
    blob = _npz(a=np.arange(3))
    (tmp_path / "m99_logs.npz").write_bytes(blob)
    entries["m99_logs"] = {
        "kind": "dataset",
        "file": "m99_logs.npz",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "provenance": "m99_logs.provenance.json",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(entries))
    monkeypatch.setattr(data, "_root", lambda: tmp_path)
    return tmp_path


@pytest.mark.parametrize(("flag", "steps"), [("0", 400), ("1", 40)])
def test_the_same_lab_code_gets_the_band_at_its_budget(shipped, monkeypatch, flag, steps):
    monkeypatch.setenv("PRL_QUICK", flag)
    S = runtime.settings(LIVE, QUICK, designed="colab-cpu")
    b = data.band("m99_return", S)
    keys = {"x", "runs", "seeds", "budget", "metric", "record", "record_sha", "hardware"}
    assert set(b) == keys | {"seconds"}
    assert b["hardware"]["platform"] == "local" and b["seconds"] == {
        "total": 12.5,
        "per_seed": None,
    }
    assert b["budget"]["steps"] == steps and b["x"][-1] == steps
    assert b["runs"].shape == (3, 4) and b["seeds"] == [0, 1, 2]
    assert b["record"].endswith(("live-exp", "quick-exp"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot.with_recorded(b, b["runs"][0], x=b["x"], your_budget=S, ax=ax)  # extra keys are fine
    plt.close(fig)


def test_no_band_at_your_budget_lists_the_recorded_budgets(shipped):
    with pytest.raises(ValueError, match="Recorded budgets: live .*'steps': 400.*; quick") as e:
        data.band("m99_return", {"steps": 41, "log_every": 10})
    assert "'steps': 41" in str(e.value)
    with pytest.raises(ValueError, match="'log_every': '<missing>'"):
        data.band("m99_return", {"steps": 40})


def test_unknown_band_lists_the_bands(shipped):
    with pytest.raises(KeyError, match="Available: m99_return"):
        data.band("m99_nope", QUICK)


def test_band_and_dataset_names_do_not_mix(shipped):
    assert data.load("m99_logs")["a"].tolist() == [0, 1, 2]
    with pytest.raises(KeyError, match="Available: m99_logs."):
        data.load("bands/m99_return.live")
    with pytest.raises(KeyError, match="Available: m99_return."):
        data.band("m99_logs", QUICK)


def test_a_changed_band_file_is_refused(shipped):
    path = shipped / "bands" / "m99_return.quick.npz"
    blob = path.read_bytes()
    path.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    with pytest.raises(ValueError, match="corrupted"):
        data.band("m99_return", QUICK)


def test_two_recordings_matching_one_budget_is_an_error(shipped):
    entries = json.loads((shipped / "manifest.json").read_text())
    key, entry = _band(shipped, "m99_return", "other", {"steps": 40}, 40)
    entries[key] = entry
    (shipped / "manifest.json").write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="2 recordings match"):
        data.band("m99_return", QUICK)


def test_no_bands_shipped_yet_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_root", lambda: tmp_path)
    with pytest.raises(KeyError, match="none yet"):
        data.band("m99_return", QUICK)


def test_budget_differences():
    assert (
        _budget.differences({"hidden": (64, 64), "n": np.int64(3)}, {"hidden": [64, 64], "n": 3})
        == []
    )
    assert _budget.differences({"steps": 5}, {"steps": 5, "other": 1}) == []
    assert _budget.differences({"steps": 5}, {"steps": 6}) == ["steps: recorded 5, yours 6"]
    assert _budget.differences({"steps": 5}, {}) == ["steps: recorded 5, yours has no 'steps'"]


def test_with_recorded_names_the_differing_key():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    band = {"x": np.arange(3), "runs": np.ones((3, 3)), "budget": {"steps": 3}, "record": "r"}
    with pytest.raises(ValueError, match="Budgets differ \\(steps: recorded 3, yours 30\\)"):
        plot.with_recorded(band, np.ones(3), x=np.arange(3), your_budget={"steps": 30}, ax=ax)
    with pytest.raises(ValueError, match="yours has no 'steps'"):
        plot.with_recorded(band, np.ones(3), x=np.arange(3), your_budget={}, ax=ax)
    plt.close(fig)
