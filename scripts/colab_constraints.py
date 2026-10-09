"""Use Google Colab's pinned package freeze (ci/colab/) in CI.

  constraints --runtime cpu|gpu --out FILE   pip constraints from the freeze (URLs rewritten,
                                             allow-listed and pinned packages left free)
  sim-requirements --out FILE                the Colab versions colab-sim preinstalls
  compat --runtime cpu|gpu                   resolve our install list against Colab's freeze
                                             (uv pip compile); fail if a preinstalled package
                                             would have to change

The freeze comes from googlecolab/backend-info at the commit in ci/colab/COMMIT. The weekly
health job bumps it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FREEZE = {
    "cpu": ROOT / "ci" / "colab" / "pip-freeze.txt",
    "gpu": ROOT / "ci" / "colab" / "pip-freeze.gpu.txt",
}
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
    out: dict[str, str] = {}
    for line in FREEZE[runtime].read_text().splitlines():
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
    args = ap.parse_args(argv)
    if args.cmd == "constraints":
        Path(args.out).write_text("\n".join(constraints(args.runtime)) + "\n")
        return 0
    if args.cmd == "sim-requirements":
        Path(args.out).write_text("\n".join(sim_requirements()) + "\n")
        return 0
    return compat(args.runtime)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
