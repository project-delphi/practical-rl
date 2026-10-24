"""Copy recorded bands from experiment records in runs/ into prl (prl/src/prl/datasets/bands/).

  uv run python scripts/export_bands.py           # write every band that runs/bands.yml names
  uv run python scripts/export_bands.py --check   # fail if a band is out of date (drift gate)

A Colab install of prl never sees runs/, so prl.data.band(name, budget) reads these copies.
runs/bands.yml names each band, the experiment record it comes from at each budget (live and
QUICK), and where the grid and the per-seed curves are inside that record; its header shows
the format. Each copy is an .npz holding x, runs and seeds, byte-identical across reruns,
with one entry in prl/src/prl/datasets/manifest.json:

  "bands/<name>.<label>": {"kind": "band", "name", "label", "file", "sha256", "bytes",
                           "record", "record_sha", "budget", "seeds", "metric"}

`record` is the record's id (its file name in runs/ without .json) and `record_sha` the sha256
of its JSON with sorted keys and no whitespace, so --check fails when a record changes (even
a note) or the spec does. After an export, review the diff and bump the prl tag (PLAN.md §7).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from package_data import (
    DATASETS,
    MANIFEST,
    RUNS,
    json_bytes,
    merged_manifest,
    read_manifest,
    read_record,
    record_sha,
    rel,
    save_npz,
    sha256,
    sync_files,
)

SPEC = RUNS / "bands.yml"
KIND = "band"
KEYS = {"metric", "x", "runs", "budget", "records"}
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
MIN_SEEDS = 3  # CLAUDE.md: comparisons need at least 3 seeds per side


class SpecError(ValueError):
    """runs/bands.yml or a record it names cannot make a band."""


def load_specs(path: Path) -> dict[str, dict[str, Any]]:
    """The band specs in `path` (none if the file is absent or empty), checked for shape."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SpecError(f"{rel(path)} must map band names to specs")
    for name, spec in raw.items():
        where = f"{rel(path)}: {name}"
        if not isinstance(name, str) or not NAME.match(name):
            raise SpecError(f"{where}: a band name is lowercase letters, digits and underscores")
        if not isinstance(spec, dict) or set(spec) != KEYS:
            got = sorted(spec) if isinstance(spec, dict) else type(spec).__name__
            raise SpecError(f"{where}: needs exactly the keys {sorted(KEYS)}, got {got}")
        if not isinstance(spec["metric"], str) or not spec["metric"].strip():
            raise SpecError(f"{where}: metric must say in words what the curves hold")
        for key in ("x", "runs"):
            path_ = spec[key]
            if not (
                (isinstance(path_, str) and path_)
                or (isinstance(path_, list) and path_ and all(isinstance(k, str) for k in path_))
            ):
                raise SpecError(f"{where}: {key} must be a dotted path or a list of keys")
        budget = spec["budget"]
        if not (isinstance(budget, list) and budget and all(isinstance(k, str) for k in budget)):
            raise SpecError(f"{where}: budget must list keys of the record's budget")
        records = spec["records"]
        if not isinstance(records, dict) or not records:
            raise SpecError(f"{where}: records must map a budget label to a record id")
        for label, record_id in records.items():
            if not isinstance(label, str) or not NAME.match(label):
                raise SpecError(f"{where}: label {label!r} must be like live or quick")
            if not isinstance(record_id, str):
                raise SpecError(f"{where}: records.{label} must be a record id (a string)")
    return raw


def resolve(rec: dict[str, Any], path: str | list[str], what: str) -> Any:
    """The value at a dotted path ("per_seed.return") or a key list in a record."""
    keys = path.split(".") if isinstance(path, str) else [str(k) for k in path]
    node: Any = rec
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise SpecError(f"{what}: the record has no {'.'.join(keys)}")
        node = node[key]
    return node


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def build_band(
    name: str, label: str, spec: dict[str, Any], rec: dict[str, Any], record_id: str
) -> tuple[bytes, dict[str, Any]]:
    """One band's .npz bytes and manifest entry, from its spec and record."""
    what = f"band {name} ({label}, record {record_id})"
    seeds = rec["seeds"]
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise SpecError(f"{what}: seeds must be integers")
    if len(set(seeds)) != len(seeds) or len(seeds) < MIN_SEEDS:
        raise SpecError(f"{what}: needs at least {MIN_SEEDS} distinct seeds, got {seeds}")
    x = resolve(rec, spec["x"], what)
    if not (isinstance(x, list) and x and all(_number(v) for v in x)):
        raise SpecError(f"{what}: x must be a non-empty list of finite numbers")
    if any(b <= a for a, b in zip(x, x[1:], strict=False)):
        raise SpecError(f"{what}: x must increase strictly")
    runs = resolve(rec, spec["runs"], what)
    if not (isinstance(runs, list) and len(runs) == len(seeds)):
        raise SpecError(f"{what}: runs must hold one list per seed ({len(seeds)})")
    for seed, row in zip(seeds, runs, strict=True):
        if not (isinstance(row, list) and len(row) == len(x) and all(_number(v) for v in row)):
            raise SpecError(f"{what}: seed {seed}'s curve must be {len(x)} finite numbers")
    missing = [k for k in spec["budget"] if k not in rec["budget"]]
    if missing:
        raise SpecError(f"{what}: the record's budget has no {', '.join(missing)}")
    budget = {k: rec["budget"][k] for k in spec["budget"]}
    integer_x = all(isinstance(v, int) for v in x)
    blob = save_npz(
        x=np.array(x, dtype=np.int64 if integer_x else np.float64),
        runs=np.array(runs, dtype=np.float64),
        seeds=np.array(seeds, dtype=np.int64),
    )
    entry = {
        "kind": KIND,
        "name": name,
        "label": label,
        "file": f"bands/{name}.{label}.npz",
        "sha256": sha256(blob),
        "bytes": len(blob),
        "record": record_id,
        "record_sha": record_sha(rec),
        "budget": budget,
        "seeds": list(seeds),
        "metric": spec["metric"].strip(),
    }
    return blob, entry


def export(
    *, spec: Path = SPEC, runs: Path = RUNS, datasets: Path = DATASETS, check: bool
) -> list[str]:
    """Write (or with check=True, compare) every band and its manifest entry.

    Returns one line per difference. Band files that no spec produces any more are deleted.
    """
    want: dict[Path, bytes] = {}
    entries: dict[str, Any] = {}
    for name, band_spec in sorted(load_specs(spec).items()):
        budgets: dict[str, str] = {}
        for label, record_id in sorted(band_spec["records"].items()):
            try:
                rec = read_record(runs, record_id)
            except ValueError as exc:
                raise SpecError(f"band {name} ({label}): {exc}") from exc
            blob, entry = build_band(name, label, band_spec, rec, record_id)
            key = json_bytes(entry["budget"]).decode()
            if key in budgets:
                raise SpecError(
                    f"band {name}: {budgets[key]} and {label} have the same budget "
                    f"{entry['budget']}, so prl.data.band could not choose between them"
                )
            budgets[key] = label
            want[datasets / entry["file"]] = blob
            entries[f"bands/{name}.{label}"] = entry
    manifest_path = datasets / MANIFEST
    current, problems = read_manifest(manifest_path, check=check)
    want[manifest_path] = json_bytes(merged_manifest(current, KIND, entries))
    bands_dir = datasets / "bands"
    generated = sorted(bands_dir.glob("*.npz")) if bands_dir.is_dir() else []
    return problems + sync_files(want, generated, check=check)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="change nothing; fail if out of date")
    args = ap.parse_args(argv)
    try:
        changes = export(check=args.check)
    except SpecError as exc:
        print(f"export_bands: {exc}")
        return 1
    for line in changes:
        print(line)
    if args.check and changes:
        print("Bands are out of date with runs/. Run scripts/export_bands.py.")
        return 1
    if not changes:
        print("Bands are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
