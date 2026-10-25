"""Comparing a recorded budget with the lab's settings (used by prl.data.band and prl.plot)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def plain(value: Any) -> Any:
    """`value` as plain JSON-like Python: tuples become lists, NumPy values become Python ones."""
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    tolist = getattr(value, "tolist", None)  # NumPy scalars and arrays
    if callable(tolist):
        return plain(tolist())
    return value


def differences(recorded: Mapping[str, Any], yours: Mapping[str, Any]) -> list[str]:
    """What differs between a recorded budget and yours: one phrase per key, empty if comparable.

    Every key of `recorded` must be in `yours` with an equal value (a tuple equals a list).
    Keys only in `yours` are settings the recording does not depend on; they are ignored, so a
    lab can pass its whole `prl.runtime.settings(...)`.
    """
    out = []
    for key, want in plain(recorded).items():
        if key not in yours:
            out.append(f"{key}: recorded {want!r}, yours has no {key!r}")
            continue
        have = plain(yours[key])
        if have != want:
            out.append(f"{key}: recorded {want!r}, yours {have!r}")
    return out
