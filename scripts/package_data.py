"""Deterministic package data for prl: .npz files, sha256s, records and the shared manifest.

Used by scripts/make_expected.py (checkpoint fixtures), scripts/make_datasets.py (datasets)
and scripts/export_bands.py (recorded bands). Not part of prl.

prl/src/prl/datasets/manifest.json has one entry per shipped file, keyed by entry id, with a
"kind": make_datasets.py owns the "dataset" entries and export_bands.py the "band" entries.
Each generator rewrites only its own kind and keeps the others, so after a merge conflict in
the manifest, running both generators rebuilds it completely.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "prl" / "src" / "prl" / "datasets"
RUNS = ROOT / "runs"
MANIFEST = "manifest.json"

sys.path.insert(0, str(ROOT / "prl" / "src"))
from prl import record  # noqa: E402


def save_npz(**arrays: np.ndarray) -> bytes:
    """An .npz that is byte-identical across runs (fixed zip timestamps, sorted keys)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            arr_buf = io.BytesIO()
            np.lib.format.write_array(arr_buf, np.asarray(arrays[key]), allow_pickle=False)
            zf.writestr(info, arr_buf.getvalue())
    return buf.getvalue()


def same_content(old: bytes, new: bytes) -> bool:
    """True if two .npz files hold the same arrays.

    Byte-identical files match. Otherwise keys, dtypes, shapes and every non-float value must
    match exactly, and floats to 1e-12: LAPACK's linear solve differs in the last bits between
    platforms (Accelerate on macOS, OpenBLAS on Linux), so a file written on one machine is
    not byte-identical to the same file built on another.
    """
    if old == new:
        return True
    a, b = np.load(io.BytesIO(old)), np.load(io.BytesIO(new))
    if sorted(a.files) != sorted(b.files):
        return False
    for key in a.files:
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if np.issubdtype(x.dtype, np.floating):
            if not np.allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def json_bytes(obj: Any) -> bytes:
    """Pretty JSON with sorted keys and a final newline (how manifests and provenance are written)."""
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def record_sha(rec: Mapping[str, Any]) -> str:
    """sha256 of a record's JSON with sorted keys and no whitespace.

    Independent of the file's indentation and line endings. To check one by hand:
    python -c "import json,hashlib,sys; r=json.load(open(sys.argv[1])); print(hashlib.sha256(
    json.dumps(r, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest())" FILE
    """
    text = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(text.encode("utf-8"))


def read_record(runs: Path, record_id: str) -> dict[str, Any]:
    """The experiment record runs/<record_id>.json, which must be a valid experiment record."""
    path = runs / f"{record_id}.json"
    if not path.is_file():
        raise ValueError(
            f"no record {record_id!r} in {rel(runs)}/ (file it with add_run_record.py)"
        )
    rec = json.loads(path.read_text())
    if not isinstance(rec, dict) or rec.get("kind") != "experiment":
        raise ValueError(f"{rel(path)} is not an experiment record")
    problems = record.validate(rec)
    if problems:
        raise ValueError(f"{rel(path)} is not a valid record: {'; '.join(problems)}")
    return rec


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_manifest(path: Path, *, check: bool) -> tuple[dict[str, Any], list[str]]:
    """The manifest at `path` ({} if absent), and any problem reading it.

    An unreadable manifest (say, merge-conflict markers) is a problem in check mode; when
    writing it counts as empty, so each generator rebuilds its own entries.
    """
    if not path.exists():
        return {}, []
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return data, []
    except ValueError as exc:
        problem = f"{rel(path)} is not valid JSON ({exc})"
        if not check:
            print(f"{problem}; rebuilding it. Run the other generator as well.")
        return {}, [problem]


def merged_manifest(current: Mapping[str, Any], kind: str, entries: Mapping[str, Any]) -> dict:
    """`current` with every entry of `kind` replaced by `entries`; other kinds are kept."""
    kept = {k: v for k, v in current.items() if not (isinstance(v, dict) and v.get("kind") == kind)}
    clash = sorted(set(kept) & set(entries))
    if clash:
        raise ValueError(f"manifest ids already used by another kind: {', '.join(clash)}")
    return {**kept, **entries}


def sync_files(
    want: Mapping[Path, bytes],
    generated: Iterable[Path],
    *,
    check: bool,
    same: Callable[[bytes, bytes], bool] | None = None,
) -> list[str]:
    """Make the generated files equal `want`, or (check=True) report how they differ.

    Writes each file of `want` that is missing or differs, and deletes each path in
    `generated` (the existing files this generator owns) that `want` no longer has.
    `same(old, new)` decides whether an existing file is current (default: identical bytes).
    Returns one line per difference; in check mode nothing changes.
    """
    same = same or (lambda old, new: old == new)
    changes = []
    for path, blob in sorted(want.items()):
        old = path.read_bytes() if path.exists() else None
        if old is not None and same(old, blob):
            continue
        changes.append(f"{'out of date' if old is not None else 'missing'}: {rel(path)}")
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
    for path in sorted(set(generated) - set(want)):
        changes.append(f"no longer generated: {rel(path)}")
        if not check:
            path.unlink()
    return changes
