import numpy as np
import pytest

from prl import runtime


def test_detect_has_fields():
    rt = runtime.detect()
    assert rt.platform in {"colab", "ci", "aws", "local"}
    assert rt.accel in {"cuda", "mps", "cpu"}
    assert rt.n_cpu >= 1
    assert rt.python.count(".") == 2


def test_platform_override(monkeypatch):
    monkeypatch.setenv("PRL_PLATFORM", "aws")
    assert runtime.detect().platform == "aws"
    monkeypatch.setenv("PRL_PLATFORM", "colab-sim")
    assert runtime.detect().platform == "ci"


@pytest.mark.parametrize(
    ("flag", "designed", "expected"),
    [("1", "colab-cpu", True), ("0", "colab-t4", False), ("", "colab-cpu", False)],
)
def test_quick_rules(monkeypatch, flag, designed, expected):
    monkeypatch.setenv("PRL_QUICK", flag)
    assert runtime.quick(designed)[0] is expected


def test_quick_auto_for_gpu_lab_without_gpu(monkeypatch):
    monkeypatch.delenv("PRL_QUICK", raising=False)
    monkeypatch.setattr(runtime, "_accel", lambda: ("cpu", None, None))
    is_quick, reason = runtime.quick("colab-t4")
    assert is_quick and "no CUDA" in reason
    assert (
        runtime.quick("colab-cpu")[0] is False
    )  # CPU-designed labs never shrink on their own runtime


def test_settings_requires_same_keys(monkeypatch):
    monkeypatch.setenv("PRL_QUICK", "1")
    s = runtime.settings(
        {"steps": 1000, "seeds": 3}, {"steps": 10, "seeds": 1}, designed="colab-cpu"
    )
    assert s.quick and s.steps == 10 and s.source == "quick"
    with pytest.raises(ValueError):
        runtime.settings({"steps": 1}, {"episodes": 1}, designed="colab-cpu")


def test_seed_everything_reproducible():
    a = runtime.seed_everything(7).normal(size=5)
    b = runtime.seed_everything(7).normal(size=5)
    assert np.array_equal(a, b)


def test_seat_seed(monkeypatch):
    monkeypatch.setenv("PRL_SEAT", "12")
    assert runtime.seat_seed() == 12
    monkeypatch.setenv("PRL_SEAT", "x")
    assert runtime.seat_seed(3) == 3


def test_test_doubles_and_shims_are_recorded(capsys):
    runtime.test_double("model", "SmolLM2-135M", "tiny random LM", "offline CI")
    runtime.shim("trl-fused-lm-head->torch", "no CPU path in TRL 1.15")
    out = capsys.readouterr().out
    assert "TEST DOUBLE" in out and "SHIM" in out
    assert any(d["double"] == "tiny random LM" for d in runtime.test_doubles_used())
    assert any(s["name"].startswith("trl") for s in runtime.shims_used())
