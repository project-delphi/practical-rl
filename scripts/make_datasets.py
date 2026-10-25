"""Build the datasets that ship inside prl (prl/src/prl/datasets/), each with a provenance JSON.

  uv run python scripts/make_datasets.py           # build every registered dataset
  uv run python scripts/make_datasets.py --check   # fail if any is out of date (drift gate)

Datasets ship inside prl because a git install of prl's subdirectory never sees the rest of
the repository (PLAN.md §5). Labs read them with prl.data.load(name), which checks the sha256.

To add a dataset, register a builder in this file, in a section for its module:

    @dataset(
        "m09_sweep",
        description="Module 8's reference DQN, 5 configurations x 10 seeds: ...",
        modules=["m09"],
        license="CC BY 4.0",
        sources=["2026-10-27-local-m09-sweep-exp"],
    )
    def m09_sweep(records):
        rec = records["2026-10-27-local-m09-sweep-exp"]
        return {"returns": np.array(rec["per_seed"]["returns"], dtype=np.float64)}

A builder returns a dict of NumPy arrays (keys are identifiers; no object arrays). It must be
deterministic and cheap: NumPy only, seconds, no torch, because CI's drift gate rebuilds
every dataset in an environment without torch. Expensive runs (training) belong in an
experiment record in runs/, filed with scripts/add_run_record.py; the builder then reads it
from `records`, which holds exactly the records named in `sources` (their ids and sha256s go
into the provenance, so --check fails when a source record changes). Seeds a builder uses
itself go in `seeds`.

Each dataset writes, byte-identical across reruns:
  <name>.npz              the arrays (fixed zip timestamps, sorted keys)
  <name>.provenance.json  name, description, license, modules, seeds, sources (record id and
                          sha256), generator, arrays (dtype and shape), file, sha256, bytes
  a manifest entry        "<name>": {"kind": "dataset", "file", "sha256", "bytes", "provenance"}
in prl/src/prl/datasets/manifest.json, which scripts/export_bands.py shares for its bands.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from package_data import (
    DATASETS,
    MANIFEST,
    RUNS,
    json_bytes,
    merged_manifest,
    read_manifest,
    read_record,
    record_sha,
    same_content,
    save_npz,
    sha256,
    sync_files,
)

KIND = "dataset"
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
ARRAY_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
GENERATOR = "scripts/make_datasets.py"

Builder = Callable[[dict[str, dict[str, Any]]], Mapping[str, Any]]


@dataclass(frozen=True)
class Dataset:
    name: str
    build: Builder
    description: str
    modules: tuple[str, ...]
    license: str
    sources: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()


#: Every dataset this script builds, by name. Add to it with @dataset(...).
REGISTRY: dict[str, Dataset] = {}


def dataset(
    name: str,
    *,
    description: str,
    modules: Iterable[str],
    license: str,
    sources: Iterable[str] = (),
    seeds: Iterable[int] = (),
    registry: dict[str, Dataset] | None = None,
) -> Callable[[Builder], Builder]:
    """Register `build(records) -> {key: array}` as the builder of dataset `name`."""

    def register(build: Builder) -> Builder:
        reg = REGISTRY if registry is None else registry
        if not NAME.match(name):
            raise ValueError(f"dataset name {name!r}: lowercase letters, digits and underscores")
        if name in reg:
            raise ValueError(f"dataset {name!r} is registered twice")
        if not description.strip() or not license.strip() or not list(modules):
            raise ValueError(f"dataset {name!r} needs a description, a license and its modules")
        reg[name] = Dataset(
            name, build, description.strip(), tuple(modules), license, tuple(sources), tuple(seeds)
        )
        return build

    return register


# ------------------------------------------------------------------ datasets, by module
# Module 9 adds m09_broken_runs and m09_sweep (built from records of Module 8's shared DQN);
# Day 5 adds the rest of PLAN.md §5's table.


# ------------------------------------------------------------------ Module 9
# Records of scripts/make_figures_m09.py (torch); the builders below only convert them.
M09_BROKEN = "2026-10-25-local-m09-broken-runs-exp"
M09_SWEEP = "2026-10-25-local-m09-sweep-exp"
M09_RUNS = ("healthy", "A", "B", "C", "D")
M09_SCALARS = (
    "epsilon",
    "loss",
    "td_abs_mean",
    "td_abs_max",
    "q_max_mean",
    "grad_norm",
    "q_terminal",
)


def _m09_float(x: Any, dtype: Any = np.float32) -> np.ndarray:
    """Nested lists of numbers with None for NaN (as the records store them) as an array."""

    def nan(v: Any) -> Any:
        return [nan(u) for u in v] if isinstance(v, list) else (np.nan if v is None else v)

    return np.asarray(nan(x), dtype=dtype)


def _m09_hardware(rec: dict[str, Any]) -> str:
    """Where a record ran, for the lab's budget sentence. A laptop stays a laptop."""
    hw = rec["hardware"]
    where = "a laptop, not Colab" if rec["platform"] == "local" else rec["env"]
    return f"{hw['cpu']} ({hw['arch']}, {rec['torch_threads']} torch threads; {where})"


def m09_broken_runs(records: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    import m09_signals

    rec = records[M09_BROKEN]
    b = rec["budget"]
    seeds = list(rec["seeds"])
    gamma = float(b["gamma"])
    out: dict[str, np.ndarray] = {
        "runs": np.array(M09_RUNS, dtype="<U7"),
        "seeds": np.array(seeds, dtype=np.int64),
        "gamma": np.array(gamma),
        "target_sync": np.array(b["target_sync"], dtype=np.int64),
        "log_every": np.array(b["log_every"], dtype=np.int64),
        "budget_steps": np.array(b["steps"], dtype=np.int64),
        "step": np.array(b["grid"], dtype=np.int64),
        "probe_obs": _m09_float(b["probe_states"]["obs"]),
    }
    for run in M09_RUNS:
        r = rec["per_seed"][run]
        for key in ("q_probe", "q_probe_target", *M09_SCALARS):
            out[f"{run}__{key}"] = _m09_float(r[key])
        out[f"{run}__snap_step"] = np.array(r["snap_step"], dtype=np.int64)
        out[f"{run}__snap_q"] = _m09_float(r["snap_q"])
        out[f"{run}__snap_actions"] = np.array(r["snap_actions"], dtype=np.int64)
        for key in ("snap_gathered", "snap_target", "snap_reward"):
            out[f"{run}__{key}"] = _m09_float(r[key])
        out[f"{run}__snap_terminated"] = np.array(r["snap_terminated"], dtype=bool)
        out[f"{run}__eval_return"] = _m09_float(r["eval_return"])
        # Per episode: CartPole pays 1 a step, so the return is the length and G_0 is
        # (1 - gamma^L) / (1 - gamma); the episodes tile the steps, so each ends at the running
        # sum of the lengths. The recording script checked all three against the raw log.
        lengths, ends, terms, ep_seed = [], [], [], []
        for seed, length, cut in zip(seeds, r["ep_length"], r["ep_truncated"], strict=True):
            length = np.array(length, dtype=np.int64)
            term = np.ones(len(length), dtype=bool)
            term[np.array(cut, dtype=np.int64)] = False
            lengths.append(length)
            ends.append(np.cumsum(length))
            terms.append(term)
            ep_seed.append(np.full(len(length), seed, dtype=np.int64))
        length = np.concatenate(lengths)
        out[f"{run}__ep_length"] = length
        out[f"{run}__ep_end_step"] = np.concatenate(ends)
        out[f"{run}__ep_terminated"] = np.concatenate(terms)
        out[f"{run}__ep_return"] = length.astype(np.float32)
        disc = (1 - gamma ** length.astype(float)) / (1 - gamma)
        out[f"{run}__ep_disc_return"] = disc.astype(np.float32)
        out[f"{run}__ep_seed"] = np.concatenate(ep_seed)
    m09_signals.assert_margins(out)  # every margin on every seed, or no dataset
    return out


M09_BROKEN_SPEC = dict(
    description=(
        "Module 8's reference DQN on CartPole-v1, 50,000 steps: the healthy run and four broken "
        "runs A-D (a learning rate 100 times the reference, a gather along the wrong dimension, "
        "a target without the terminal mask, a target network never synced), seeds 0-2 each, "
        "with Module 9's logging (every 500 steps, 8 probe states, three batch snapshots of 32 "
        "rows). Keys <run>__<log key>; per-episode arrays hold the three seeds one after another "
        "(<run>__ep_seed). Lab 9, Exercises 1-3."
    ),
    modules=["m09"],
    license="CC BY 4.0",
    sources=[M09_BROKEN],
)
# Registered after the lead's day-3 decision 15 (OPEN_QUESTIONS B81): the builder asserts
# every margin on every seed (scripts/m09_signals.py) and builds nothing if one fails.
dataset("m09_broken_runs", **M09_BROKEN_SPEC)(m09_broken_runs)


@dataset(
    "m09_sweep",
    description=(
        "Module 8's reference DQN on CartPole-v1 with only the learning rate changed (0.1, 0.3, "
        "1, 3 and 10 times the reference), seeds 0-9 each, 50,000 steps: the greedy return "
        "(mean of 10 episodes) every 5,000 steps, each run's laptop wall-clock and where it ran. "
        "Lab 9, Exercise 5."
    ),
    modules=["m09"],
    license="CC BY 4.0",
    sources=[M09_SWEEP],
)
def m09_sweep(records: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    rec = records[M09_SWEEP]
    b = rec["budget"]
    rows = [rec["per_seed"][name] for name in b["configs"]]
    curves = _m09_float([row["return"] for row in rows])
    assert curves.shape == (len(rows), len(rec["seeds"]), len(b["grid"]))
    return {
        "lr_factor": np.array(b["lr_factors"], dtype=np.float64),
        "lr": np.array(b["lr"], dtype=np.float64),
        "seeds": np.array(rec["seeds"], dtype=np.int64),
        "final_return": curves[:, :, -1].astype(np.float64),
        "curve_step": np.array(b["grid"], dtype=np.int64),
        "curve_return": curves,
        "seconds": np.array([row["seconds"] for row in rows], dtype=np.float64),
        "budget_steps": np.array(b["steps"], dtype=np.int64),
        "hardware": np.array(_m09_hardware(rec)),
    }


# ------------------------------------------------------------------ building
def _arrays(name: str, built: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(built, Mapping) or not built:
        raise ValueError(f"dataset {name!r}: the builder must return a non-empty dict of arrays")
    out = {}
    for key, value in built.items():
        if not isinstance(key, str) or not ARRAY_KEY.match(key):
            raise ValueError(f"dataset {name!r}: array name {key!r} must be an identifier")
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise ValueError(f"dataset {name!r}: {key} is an object array; use numbers or strings")
        out[key] = arr
    return out


def build(
    registry: Mapping[str, Dataset] | None = None,
    *,
    runs: Path = RUNS,
    datasets: Path = DATASETS,
    check: bool,
) -> list[str]:
    """Write (or with check=True, compare) every dataset, its provenance and manifest entry.

    Returns one line per difference. Dataset files no builder produces any more are deleted.
    An existing .npz that holds the same arrays (floats to 1e-12) is kept as it is.
    """
    registry = REGISTRY if registry is None else registry
    want: dict[Path, bytes] = {}
    entries: dict[str, Any] = {}
    for name, ds in sorted(registry.items()):
        records = {rid: read_record(runs, rid) for rid in ds.sources}
        arrays = _arrays(name, ds.build(records))
        path = datasets / f"{name}.npz"
        blob = save_npz(**arrays)
        if path.exists() and same_content(path.read_bytes(), blob):
            blob = path.read_bytes()
        provenance = {
            "name": name,
            "description": ds.description,
            "license": ds.license,
            "modules": list(ds.modules),
            "seeds": list(ds.seeds),
            "sources": [{"record": rid, "record_sha": record_sha(records[rid])} for rid in records],
            "generator": f"{GENERATOR} ({ds.build.__name__})",
            "arrays": {
                k: {"dtype": str(a.dtype), "shape": list(a.shape)}
                for k, a in sorted(arrays.items())
            },
            "file": path.name,
            "sha256": sha256(blob),
            "bytes": len(blob),
        }
        want[path] = blob
        want[datasets / f"{name}.provenance.json"] = json_bytes(provenance)
        entries[name] = {
            "kind": KIND,
            "file": path.name,
            "sha256": provenance["sha256"],
            "bytes": len(blob),
            "provenance": f"{name}.provenance.json",
        }
    manifest_path = datasets / MANIFEST
    current, problems = read_manifest(manifest_path, check=check)
    want[manifest_path] = json_bytes(merged_manifest(current, KIND, entries))
    generated = sorted([*datasets.glob("*.npz"), *datasets.glob("*.provenance.json")])
    return problems + sync_files(want, generated, check=check)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="change nothing; fail if out of date")
    args = ap.parse_args(argv)
    try:
        changes = build(check=args.check)
    except ValueError as exc:
        print(f"make_datasets: {exc}")
        return 1
    for line in changes:
        print(line)
    if args.check and changes:
        print("Datasets are out of date. Run scripts/make_datasets.py.")
        return 1
    if not changes:
        print(f"Datasets are up to date ({len(REGISTRY)} registered).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
