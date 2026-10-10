"""Use Google Colab's pinned package freeze (ci/colab/) in CI.

  constraints --runtime cpu|gpu --out FILE   pip constraints from the freeze (URLs rewritten,
                                             allow-listed and pinned packages left free)
  sim-requirements --out FILE                the Colab versions colab-sim preinstalls
  compat --runtime cpu|gpu                   resolve our install list against Colab's freeze
                                             (uv pip compile); fail if a preinstalled package
                                             would have to change
  bump [--commit SHA] [--report FILE]        fetch the freeze at SHA (default: the latest
                                             upstream commit); if its files differ, rewrite
                                             ci/colab/ and the commit in _variables.yml

The freeze comes from googlecolab/backend-info at the commit in ci/colab/COMMIT. The weekly
health job (.github/workflows/health.yml) bumps it and opens a PR; a person merges it.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "ci" / "colab"
FREEZE = {
    "cpu": COLAB_DIR / "pip-freeze.txt",
    "gpu": COLAB_DIR / "pip-freeze.gpu.txt",
}
# The upstream files ci/colab/ holds, under the same names. ci/colab/COMMIT records which commit.
UPSTREAM = "googlecolab/backend-info"
UPSTREAM_FILES = ("pip-freeze.txt", "pip-freeze.gpu.txt", "os-info.txt")
API = f"https://api.github.com/repos/{UPSTREAM}"
RAW = f"https://raw.githubusercontent.com/{UPSTREAM}"
WHEEL = re.compile(r"/(?P<name>[A-Za-z0-9_.]+)-(?P<ver>[^-]+)-[^/]*\.whl$")
# What colab-sim preinstalls (Colab's versions), on top of which the notebook's setup cell runs.
SIM = [
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "gymnasium",
    "gym",
    "ipykernel",
    "jupyter-client",
    "pyyaml",
    "nbformat",
    "pip",
]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_freeze(runtime: str) -> dict[str, str]:
    """name -> version (pip-style, local versions kept: 2.11.0+cpu)."""
    return parse_freeze_text(FREEZE[runtime].read_text())


def parse_freeze_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            out[norm(name)] = ver.strip()
        elif " @ " in line:
            name, url = line.split(" @ ", 1)
            m = WHEEL.search(unquote(url))
            if m and url.startswith("http"):
                out[norm(name)] = m.group("ver")
    return out


def constraints(runtime: str) -> list[str]:
    v = variables.load()
    free = {norm(p) for p in v["packages"]["colab"]["allow_upgrade"]} | {
        norm(p) for p in v["packages"]["pins"]
    }
    free |= {"prl"}
    return [
        f"{name}=={ver}" for name, ver in sorted(parse_freeze(runtime).items()) if name not in free
    ]


def sim_requirements() -> list[str]:
    frz = parse_freeze("cpu")
    reqs = []
    for name in SIM:
        if norm(name) in frz:
            reqs.append(f"{name}=={frz[norm(name)]}")
        else:
            reqs.append(name)
    return reqs + ["nbclient"]


def compat(runtime: str) -> int:
    v = variables.load()
    pins = [f"{k}=={val}" for k, val in v["packages"]["pins"].items()]
    index = "https://download.pytorch.org/whl/" + ("cpu" if runtime == "cpu" else "cu130")
    with tempfile.TemporaryDirectory() as tmp:
        cons = Path(tmp) / "constraints.txt"
        cons.write_text("\n".join(constraints(runtime)) + "\n")
        req = Path(tmp) / "req.in"
        req.write_text("\n".join(pins + [str(ROOT / "prl")]) + "\n")
        cmd = [
            "uv",
            "pip",
            "compile",
            str(req),
            "-c",
            str(cons),
            "--python-version",
            "3.13",
            "--python-platform",
            "x86_64-manylinux_2_28",
            "--extra-index-url",
            index,
            "--index-strategy",
            "unsafe-best-match",
            "--no-header",
            "--quiet",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-4000:])
        print(f"colab-compat ({runtime}): our pins cannot be installed on Colab's packages.")
        return 1
    frz = parse_freeze(runtime)
    changed, added = [], []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        n = norm(name)
        if n not in frz:
            added.append(f"{name}=={ver}")
        elif frz[n] != ver:
            changed.append(f"{name}: {frz[n]} -> {ver}")
    allowed = {norm(p) for p in v["packages"]["colab"]["allow_upgrade"]}
    bad = [c for c in changed if norm(c.split(":")[0]) not in allowed]
    print(f"colab-compat ({runtime}): changed {changed or 'nothing'}; added {added or 'nothing'}")
    if bad:
        print(f"Not allowed to change on Colab: {bad}")
        return 1
    return 0


# ---------------------------------------------------------------- bump (weekly health job)
SHA = re.compile(r"[0-9a-f]{40}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
Fetch = Callable[[str], bytes]


def read_commit(text: str) -> tuple[str, str]:
    """ci/colab/COMMIT is one line: '<40-hex sha> <committer date, YYYY-MM-DDTHH:MM:SSZ>'."""
    parts = text.split()
    if len(parts) != 2 or not SHA.fullmatch(parts[0]) or not DATE.fullmatch(parts[1]):
        raise ValueError(f"COMMIT should be '<sha> <YYYY-MM-DDTHH:MM:SSZ>', got {text!r}")
    return parts[0], parts[1]


def format_commit(sha: str, date: str) -> str:
    text = f"{sha} {date}\n"
    read_commit(text)
    return text


def os_python(text: str) -> str | None:
    """The Python version in os-info.txt (its 'Python 3.13.16' line)."""
    m = re.search(r"^Python (\d+\.\d+\.\d+)\s*$", text, re.M)
    return m.group(1) if m else None


def diff_versions(
    old: dict[str, str], new: dict[str, str]
) -> list[tuple[str, str | None, str | None]]:
    """(name, before, after) per package that changed; None means absent on that side."""
    return [
        (n, old.get(n), new.get(n)) for n in sorted(set(old) | set(new)) if old.get(n) != new.get(n)
    ]


def update_variables(text: str, sha: str, python: str | None) -> str:
    """Point packages.colab in _variables.yml at the new commit and Python; keep the rest."""
    # Quoted: an all-digit short SHA would otherwise load as a YAML number.
    text, n = re.subn(r"(?m)^(\s+backend_info_commit:\s*)\S+", rf'\g<1>"{sha[:10]}"', text)
    if n != 1:
        raise ValueError(f"_variables.yml: expected one backend_info_commit line, found {n}")
    if python:
        text, n = re.subn(r'(?m)^(\s+python:\s*)"[^"]*"', rf'\g<1>"{python}"', text)
        if n != 1:
            raise ValueError(f"_variables.yml: expected one python line, found {n}")
    return text


def http_get(url: str, attempts: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "prl-colab-freeze-bump"})
    if url.startswith("https://api.github.com/"):
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:  # only for the higher API rate limit; the upstream repo is public
            req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            client_error = isinstance(exc, urllib.error.HTTPError) and exc.code < 500
            if client_error or attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))  # a network blip should not fail the weekly job
    raise AssertionError("unreachable")


def resolve_commit(get: Fetch, commit: str | None) -> tuple[str, str]:
    """(full sha, committer date) of COMMIT, or of the newest commit on the default branch."""
    if commit is None:
        data = json.loads(get(f"{API}/commits?per_page=1"))[0]
    else:
        if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            raise ValueError(f"not a commit SHA: {commit!r}")
        data = json.loads(get(f"{API}/commits/{commit}"))
    return data["sha"], data["commit"]["committer"]["date"]


@dataclass
class Bump:
    old: tuple[str, str]
    new: tuple[str, str]
    changed: list[str] = field(default_factory=list)  # upstream files whose bytes differ
    report: str = ""


def bump(
    commit: str | None = None,
    *,
    colab_dir: Path = COLAB_DIR,
    variables_path: Path = variables.VARS,
    get: Fetch = http_get,
) -> Bump:
    """Fetch the freeze at COMMIT (default: upstream's newest). Rewrite only if a file differs."""
    old = read_commit((colab_dir / "COMMIT").read_text())
    new = resolve_commit(get, commit)
    if commit is None and new[1] < old[1]:
        raise SystemExit(
            f"Upstream's newest commit {new[0][:10]} ({new[1]}) is older than the pinned "
            f"{old[0][:10]} ({old[1]}). Not moving the freeze backwards."
        )
    after = {name: get(f"{RAW}/{new[0]}/{name}") for name in UPSTREAM_FILES}
    before = {name: (colab_dir / name).read_bytes() for name in UPSTREAM_FILES}
    changed = [name for name in UPSTREAM_FILES if after[name] != before[name]]
    result = Bump(old, new, changed, bump_report(old, new, before, after, changed))
    if changed:
        for name in changed:
            (colab_dir / name).write_bytes(after[name])
        (colab_dir / "COMMIT").write_text(format_commit(*new))
        python = os_python(after["os-info.txt"].decode())
        variables_path.write_text(update_variables(variables_path.read_text(), new[0], python))
    return result


def _table(rows: list[tuple[str, str | None, str | None]]) -> list[str]:
    out = ["| Package | Before | After |", "|---|---|---|"]
    return out + [f"| {n} | {a or '(absent)'} | {b or '(absent)'} |" for n, a, b in rows]


def bump_report(
    old: tuple[str, str],
    new: tuple[str, str],
    before: dict[str, bytes],
    after: dict[str, bytes],
    changed: list[str],
) -> str:
    o, n = f"{old[0][:10]} ({old[1]})", f"{new[0][:10]} ({new[1]})"
    if not changed:
        same = "is the pinned commit" if new[0] == old[0] else f"has the same files as {o}"
        return f"Colab freeze unchanged: upstream {n} {same}.\n"
    pk = variables.load()["packages"]
    names = SIM + pk["colab"]["use_preinstalled"] + pk["colab"]["allow_upgrade"] + list(pk["pins"])
    watch = {norm(p) for p in names}
    lines = [
        f"## Colab freeze: {o} to {n}",
        "",
        f"Upstream diff: https://github.com/{UPSTREAM}/compare/{old[0]}...{new[0]}",
        "",
        "Files changed: " + ", ".join(f"`{c}`" for c in changed) + ".",
    ]
    for runtime, name in (("CPU", "pip-freeze.txt"), ("GPU", "pip-freeze.gpu.txt")):
        if name not in changed:
            continue
        rows = diff_versions(
            parse_freeze_text(before[name].decode()), parse_freeze_text(after[name].decode())
        )
        ours = [r for r in rows if r[0] in watch]
        rest = [r for r in rows if r[0] not in watch]
        lines += ["", f"### {runtime} runtime (`{name}`): {len(rows)} package(s) changed", ""]
        lines += ["Packages the labs use:", "", *_table(ours)] if ours else ["None the labs use."]
        if rest:
            lines += ["", f"<details><summary>{len(rest)} other package(s)</summary>", ""]
            lines += [*_table(rest), "", "</details>"]
    if "os-info.txt" in changed:
        a, b = before["os-info.txt"].decode(), after["os-info.txt"].decode()
        diff = [
            d
            for d in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
            if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))
        ]
        lines += ["", "### OS info (`os-info.txt`)", "", "```diff", *diff, "```"]
        pa, pb = os_python(a), os_python(b)
        if pa and pb and pa.split(".")[:2] != pb.split(".")[:2]:
            lines += [
                "",
                f"**Python moved from {pa} to {pb}.** ci.yml, health.yml, the colab-sim action "
                "and `compat` pin the minor version. Update them in this PR before merging.",
            ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("constraints")
    c.add_argument("--runtime", choices=["cpu", "gpu"], default="cpu")
    c.add_argument("--out", required=True)
    s = sub.add_parser("sim-requirements")
    s.add_argument("--out", required=True)
    k = sub.add_parser("compat")
    k.add_argument("--runtime", choices=["cpu", "gpu"], default="cpu")
    b = sub.add_parser("bump")
    b.add_argument("--commit", help="a backend-info commit SHA (default: the newest upstream)")
    b.add_argument("--report", help="also write the Markdown report to this file")
    args = ap.parse_args(argv)
    if args.cmd == "bump":
        res = bump(args.commit)
        print(res.report, end="")
        if args.report:
            Path(args.report).write_text(res.report)
        if os.environ.get("GITHUB_OUTPUT"):  # step outputs for health.yml
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write(f"changed={str(bool(res.changed)).lower()}\n")
                fh.write(f"commit={res.new[0]}\nshort={res.new[0][:10]}\n")
        return 0
    if args.cmd == "constraints":
        Path(args.out).write_text("\n".join(constraints(args.runtime)) + "\n")
        return 0
    if args.cmd == "sim-requirements":
        Path(args.out).write_text("\n".join(sim_requirements()) + "\n")
        return 0
    return compat(args.runtime)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
