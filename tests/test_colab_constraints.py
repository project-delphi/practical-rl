"""The Colab freeze in ci/colab/ and the weekly `bump` (no network: a fake fetcher)."""

import json
import shutil
from pathlib import Path

import pytest

import colab_constraints as cc
import variables

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "ci" / "colab"
NEW_SHA = "0123456789abcdef0123456789abcdef01234567"
NEW_DATE = "2026-10-12T08:00:00Z"


def test_ci_colab_holds_exactly_the_upstream_files_and_commit():
    held = {p.name for p in COLAB.iterdir() if p.is_file() and not p.name.startswith(".")}
    assert held == {*cc.UPSTREAM_FILES, "COMMIT"}


def test_commit_file_round_trips():
    text = (COLAB / "COMMIT").read_text()
    assert cc.format_commit(*cc.read_commit(text)) == text
    with pytest.raises(ValueError):
        cc.read_commit("e39694e267 2026-10-03")


def test_variables_point_at_the_pinned_freeze():
    sha, _ = cc.read_commit((COLAB / "COMMIT").read_text())
    colab = variables.load()["packages"]["colab"]
    assert sha.startswith(str(colab["backend_info_commit"]))
    assert colab["python"] == cc.os_python((COLAB / "os-info.txt").read_text())


def _fake_upstream(sha: str, date: str, files: dict[str, bytes]):
    """A stand-in for http_get: answers only the URLs bump should ask for."""
    calls = []

    def get(url: str) -> bytes:
        calls.append(url)
        if url == f"{cc.API}/commits?per_page=1" or url == f"{cc.API}/commits/{sha[:10]}":
            body = {"sha": sha, "commit": {"committer": {"date": date}}}
            return json.dumps([body] if "per_page" in url else body).encode()
        prefix = f"{cc.RAW}/{sha}/"
        if url.startswith(prefix) and url[len(prefix) :] in files:
            return files[url[len(prefix) :]]
        raise AssertionError(f"unexpected URL {url}")

    get.calls = calls
    return get


@pytest.fixture
def repo(tmp_path):
    shutil.copytree(COLAB, tmp_path / "colab")
    shutil.copy(variables.VARS, tmp_path / "_variables.yml")
    return tmp_path


def test_bump_rewrites_changed_files_commit_and_variables(repo):
    cpu = (COLAB / "pip-freeze.txt").read_text()
    numpy = cc.parse_freeze_text(cpu)["numpy"]
    new_cpu = cpu.replace(f"numpy=={numpy}\n", "numpy==9.9.9\n") + "zzz-new-package==1.0\n"
    os_info = (COLAB / "os-info.txt").read_text()
    new_os = os_info.replace("Python 3.13.16", "Python 3.13.17")
    files = {
        "pip-freeze.txt": new_cpu.encode(),
        "pip-freeze.gpu.txt": (COLAB / "pip-freeze.gpu.txt").read_bytes(),
        "os-info.txt": new_os.encode(),
    }
    get = _fake_upstream(NEW_SHA, NEW_DATE, files)
    vars_before = (repo / "_variables.yml").read_text()

    res = cc.bump(colab_dir=repo / "colab", variables_path=repo / "_variables.yml", get=get)

    assert res.changed == ["pip-freeze.txt", "os-info.txt"]
    assert res.new == (NEW_SHA, NEW_DATE)
    assert (repo / "colab" / "COMMIT").read_text() == f"{NEW_SHA} {NEW_DATE}\n"
    for name, blob in files.items():
        assert (repo / "colab" / name).read_bytes() == blob
    vars_after = (repo / "_variables.yml").read_text()
    assert "backend_info_commit: 0123456789\n" in vars_after
    assert 'python: "3.13.17"\n' in vars_after
    assert len(vars_after.splitlines()) == len(vars_before.splitlines())  # comments kept
    assert f"| numpy | {numpy} | 9.9.9 |" in res.report  # a package the labs use
    assert "| zzz-new-package | (absent) | 1.0 |" in res.report
    assert "+Python 3.13.17" in res.report and "Python moved" not in res.report
    assert all(u.startswith((cc.API, cc.RAW)) for u in get.calls)


def test_bump_with_the_same_files_writes_nothing(repo):
    files = {name: (COLAB / name).read_bytes() for name in cc.UPSTREAM_FILES}
    get = _fake_upstream(NEW_SHA, NEW_DATE, files)
    commit_before = (repo / "colab" / "COMMIT").read_text()

    res = cc.bump(colab_dir=repo / "colab", variables_path=repo / "_variables.yml", get=get)

    assert res.changed == []
    assert (repo / "colab" / "COMMIT").read_text() == commit_before
    assert res.report.startswith("Colab freeze unchanged")


def test_bump_flags_a_python_minor_change_and_takes_an_explicit_commit(repo):
    files = {name: (COLAB / name).read_bytes() for name in cc.UPSTREAM_FILES}
    files["os-info.txt"] = files["os-info.txt"].replace(b"Python 3.13.16", b"Python 3.14.1")
    get = _fake_upstream(NEW_SHA, NEW_DATE, files)

    res = cc.bump(
        NEW_SHA[:10], colab_dir=repo / "colab", variables_path=repo / "_variables.yml", get=get
    )

    assert res.changed == ["os-info.txt"]
    assert "Python moved from 3.13.16 to 3.14.1" in res.report


def test_bump_never_moves_the_freeze_backwards(repo):
    files = {name: (COLAB / name).read_bytes() for name in cc.UPSTREAM_FILES}
    get = _fake_upstream(NEW_SHA, "2020-01-01T00:00:00Z", files)
    with pytest.raises(SystemExit):
        cc.bump(colab_dir=repo / "colab", variables_path=repo / "_variables.yml", get=get)


def test_bump_rejects_a_commit_that_is_not_a_sha(repo):
    with pytest.raises(ValueError):
        cc.bump("main/../x", colab_dir=repo / "colab", get=_fake_upstream(NEW_SHA, NEW_DATE, {}))
