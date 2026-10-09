"""Runtime detection, QUICK settings, seeding, version printing and test doubles.

Nothing here imports torch unless torch is already imported by the notebook,
so tabular labs start fast.
"""

from __future__ import annotations

import os
import platform as _platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any

import numpy as np

#: Designed runtimes that need a GPU. A lab designed for one of these switches
#: to QUICK automatically when no GPU is present.
GPU_RUNTIMES = frozenset({"colab-t4", "aws-g4dn"})

#: Runtimes a lab may be designed for (keys of `runtimes` in _variables.yml).
DESIGNED_RUNTIMES = frozenset({"colab-cpu", "colab-t4"})

_TEST_DOUBLES: list[dict[str, str]] = []
_SHIMS: list[dict[str, str]] = []


@dataclass(frozen=True)
class Runtime:
    """What this process is running on."""

    platform: str  # colab | ci | aws | local
    accel: str  # cuda | mps | cpu
    gpu: str | None
    cpu: str
    n_cpu: int
    ram_gb: float | None
    python: str
    colab_release: str | None = None
    torch_threads: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        gpu = f" · GPU {self.gpu}" if self.gpu else ""
        return (
            f"{self.platform} · {self.accel}{gpu} · {self.n_cpu} CPU · "
            f"{self.ram_gb if self.ram_gb is not None else '?'} GB RAM · Python {self.python}"
        )


def _platform_name() -> str:
    forced = os.environ.get("PRL_PLATFORM", "").strip().lower()
    if forced:
        # colab-sim is CI pretending to be Colab for the install path; it is still CI.
        return "ci" if forced == "colab-sim" else forced
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
        return "ci"
    try:
        with open("/sys/class/dmi/id/sys_vendor") as fh:
            if "Amazon EC2" in fh.read():
                return "aws"
    except OSError:
        pass
    return "local"


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return _platform.processor() or _platform.machine() or "unknown"


def _n_cpu() -> int:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1


def _ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return round(int(out.stdout.strip()) / 2**30, 1)
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 2**30, 1)
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None


def _accel() -> tuple[str, str | None, int | None]:
    """Accelerator, GPU name and torch thread count.

    Uses torch only if the notebook has already imported it.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        threads = torch.get_num_threads()
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0), threads
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", None, threads
        return "cpu", None, threads
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            name = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
            if name:
                return "cuda", name, None
        except (OSError, subprocess.SubprocessError):
            pass
    return "cpu", None, None


def detect() -> Runtime:
    """Describe the current runtime (platform, accelerator, CPU, RAM, Python)."""
    accel, gpu, threads = _accel()
    return Runtime(
        platform=_platform_name(),
        accel=accel,
        gpu=gpu,
        cpu=_cpu_model(),
        n_cpu=_n_cpu(),
        ram_gb=_ram_gb(),
        python=_platform.python_version(),
        colab_release=os.environ.get("COLAB_RELEASE_TAG") or None,
        torch_threads=threads,
    )


def device(prefer: str = "cpu") -> str:
    """Pick a torch device string.

    Small networks (most labs) are faster on CPU than on MPS, so the default is
    "cpu" even on Apple silicon. Pass prefer="auto" to use CUDA when present.
    """
    if prefer == "cpu":
        return "cpu"
    accel, _, _ = _accel()
    if prefer == "auto":
        return "cuda" if accel == "cuda" else "cpu"
    return prefer


def quick(designed: str) -> tuple[bool, str]:
    """Decide QUICK mode for a lab designed for runtime `designed`.

    PRL_QUICK=1 or PRL_QUICK=0 always wins. Otherwise QUICK is on only when a
    lab designed for a GPU finds no GPU. A CPU-designed lab runs its full
    settings on a CPU.
    """
    flag = os.environ.get("PRL_QUICK", "").strip()
    if flag == "1":
        return True, "PRL_QUICK=1 is set"
    if flag == "0":
        return False, "PRL_QUICK=0 is set"
    if designed in GPU_RUNTIMES:
        accel, _, _ = _accel()
        if accel != "cuda":
            return True, f"designed for {designed} but no CUDA GPU was found"
        return False, "GPU present"
    return False, f"designed for {designed}; running full settings"


class Settings(dict):
    """Budgets for a lab: the live (designed) values, or the QUICK values.

    Lab code reads numbers from here and never branches on QUICK.
    """

    def __init__(self, values: dict[str, Any], *, quick: bool, reason: str, source: str):
        super().__init__(values)
        self.quick = quick
        self.reason = reason
        self.source = source

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def settings(live: dict[str, Any], quick_values: dict[str, Any], *, designed: str) -> Settings:
    """Return the live or QUICK budget for this runtime.

    `live` and `quick_values` must have the same keys, so the code path never
    changes between them.
    """
    if set(live) != set(quick_values):
        missing = set(live) ^ set(quick_values)
        raise ValueError(
            f"live and quick settings must have the same keys; differ on {sorted(missing)}"
        )
    is_quick, reason = quick(designed)
    chosen = quick_values if is_quick else live
    return Settings(
        dict(chosen), quick=is_quick, reason=reason, source="quick" if is_quick else "live"
    )


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if imported) torch. Returns a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # legacy global state, for libraries that still use it
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
    return np.random.default_rng(seed)


def seat_seed(default: int = 0) -> int:
    """The participant's seed: PRL_SEAT if set, else `default`.

    In the room, everyone sets SEED to their seat number so pooled results use
    distinct seeds.
    """
    raw = os.environ.get("PRL_SEAT", "").strip()
    return int(raw) if raw.isdigit() else default


def package_versions(names: list[str] | tuple[str, ...]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


DEFAULT_PACKAGES = ("prl", "numpy", "scipy", "matplotlib", "gymnasium", "torch")


def print_versions(extra: list[str] | tuple[str, ...] = ()) -> dict[str, str | None]:
    """Print the runtime and package versions; return the versions."""
    rt = detect()
    names = list(DEFAULT_PACKAGES) + [n for n in extra if n not in DEFAULT_PACKAGES]
    versions = package_versions(names)
    print(f"Runtime: {rt.summary()}")
    print(f"CPU: {rt.cpu}")
    shown = " · ".join(f"{k} {v}" for k, v in versions.items() if v is not None)
    print(f"Packages: {shown}")
    absent = [k for k, v in versions.items() if v is None]
    if absent:
        print(f"Not installed: {', '.join(absent)}")
    return versions


def test_double(kind: str, real: str, double: str, reason: str) -> None:
    """Announce and record that a stand-in replaces the real thing.

    Records that used a test double show that the code runs, not what the
    method does; the readiness page keeps them apart.
    """
    entry = {"kind": kind, "real": real, "double": double, "reason": reason}
    if entry not in _TEST_DOUBLES:
        _TEST_DOUBLES.append(entry)
    print(f"TEST DOUBLE · {kind}: using {double} instead of {real} ({reason})")


def test_doubles_requested() -> bool:
    """True when PRL_TEST_DOUBLES=1 asks labs to use their stand-ins."""
    return os.environ.get("PRL_TEST_DOUBLES", "").strip() == "1"


def shim(name: str, reason: str) -> None:
    """Announce and record a patch applied to a third-party library."""
    entry = {"name": name, "reason": reason}
    if entry not in _SHIMS:
        _SHIMS.append(entry)
    print(f"SHIM · {name}: {reason}")


def test_doubles_used() -> list[dict[str, str]]:
    return list(_TEST_DOUBLES)


def shims_used() -> list[dict[str, str]]:
    return list(_SHIMS)


def worked_mode() -> bool:
    """True when solutions are bound (PRL_WORKED=1)."""
    return os.environ.get("PRL_WORKED", "").strip() == "1"
