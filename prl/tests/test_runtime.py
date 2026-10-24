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


def test_printed_lines_fit_the_notebook_width(capsys, monkeypatch, tmp_path):
    """Generated output stays within 100 columns (the Module 2 UI review)."""
    from prl import lab as harness
    from prl import runtime

    many = [f"package{i}" for i in range(40)]
    assert all(len(line) <= 100 for line in runtime.wrap_items("Packages: ", many).splitlines())
    assert runtime.wrap_items("Packages: ", ["a 1", "b 2"]) == "Packages: a 1 · b 2"
    runtime.print_versions(extra=many)
    monkeypatch.setenv("PRL_WORKED", "1")
    harness.init("03-model-free", "0123456789abcdef", designed="colab-cpu", api=1, ns={})
    out = capsys.readouterr().out
    assert all(len(line) <= 100 for line in out.splitlines()), max(out.splitlines(), key=len)


def test_finish_output_fits_except_the_room_code(capsys, monkeypatch, tmp_path):
    """lab.finish() stays within 100 columns, except the room code.

    The room code is exempt by decision (2026-10-24): it must stay one pasteable line for the
    debrief's parse_room_codes, and with seven metrics it runs to about 140 columns.
    """
    from prl import lab as harness
    from prl import record

    monkeypatch.setenv("PRL_WORKED", "1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "_variables.yml").write_text("x: 1\n")
    (tmp_path / "prl").mkdir()
    ns = {}
    lab = harness.init(
        "06-exploration-mdps",
        "0123456789abcdef",
        designed="colab-cpu",
        api=1,
        ns=ns,
        checkpoints=["1"],
        seat=0,
    )
    for k in ("mbie_end", "reach_eps", "reach_opt", "reach_mbie", "regret_eps", "regret_opt"):
        lab.metric(k, 4025.04)
    lab.finish()
    lines = capsys.readouterr().out.splitlines()
    record_lines = set()
    inside = False
    for i, line in enumerate(lines):  # the JSON between the markers is data, not prose
        if line == record.BEGIN:
            inside = True
        if inside:
            record_lines.add(i)
        if line == record.END:
            inside = False
    prose = [s for i, s in enumerate(lines) if i not in record_lines and not s.startswith("ROOM ")]
    assert all(len(s) <= 100 for s in prose), max(prose, key=len)
    assert any(s.startswith("ROOM ") and len(s) > 100 for s in lines)  # the exemption is real
