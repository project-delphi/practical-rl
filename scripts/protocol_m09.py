"""Run scripts/threshold_protocol.py for checkpoint 6.2 of Lab 9 with its protocol-only mutant.

  uv run python scripts/protocol_m09.py --budget quick [--seeds 100] [--dry-run]
  uv run python scripts/protocol_m09.py --budget live  [--seeds 20]
  uv run python scripts/add_run_record.py

Checkpoint 6.2 (prl.checks.m09.check_live_return) guards the platform, not the participant: it
compares the greedy return Exercise 6's run cell computed with a threshold that the protocol sets
by training the lab's provided train_dqn and CONFIG. Its one wrong version, random play, replaces
that provided code, so it lives in labs/mutants/protocol/m09.py instead of labs/mutants/m09.py
(which holds mutants of functions participants write; tests/test_mutants.py). This driver is the
protocol's own command line with that folder as the mutants' source; everything else (seed
floors, analysis, record, the entry to paste) is threshold_protocol.py's.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_protocol as tp  # noqa: E402

MUTANTS = tp.ROOT / "labs" / "mutants" / "protocol"
MODULE, CHECK = "m09", "check_live_return"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--budget", choices=sorted(tp.MIN_SEEDS), required=True)
    ap.add_argument("--seeds", type=int, help="number of seeds (default: the floor)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="measure and report only")
    args = ap.parse_args(argv)
    n_seeds = args.seeds if args.seeds is not None else tp.MIN_SEEDS[args.budget]
    try:
        tp.check_seed_floor(args.budget, n_seeds)
        check = tp.load_check(MODULE, CHECK)
        notebook = tp.notebook_for(MODULE)
        tp.ensure_current(notebook)
        print(
            f"Threshold protocol: prl.checks.{MODULE}.{CHECK}, {args.budget} budget, mutants "
            f"from {tp._rel(MUTANTS)}/{MODULE}.py"
        )
        res = tp.run_protocol(
            check,
            check_name=CHECK,
            notebook=notebook,
            budget=args.budget,
            n_seeds=n_seeds,
            seed_start=args.seed_start,
            mutants_dir=MUTANTS,
        )
    except tp.ProtocolError as exc:
        print(f"protocol_m09: {exc}", file=sys.stderr)
        return 1
    print()
    print(tp.format_report(res.per_seed, res.analysis, f"Notebook {res.notebook.name}"))
    rt = tp.runtime.detect()
    print(
        f"One checkpoint call at this budget took about {res.seconds_per_call:.3g} s on this "
        f"machine ({tp._default_env(rt, check.designed)}, {rt.cpu}, {platform.machine()})."
    )
    if not res.analysis.ok:
        print("No record written.")
        return 1
    if args.dry_run:
        print("Dry run: no record written.")
        return 0
    rec = tp.build_record(res)
    path = tp.write_record(rec)
    print(f"Saved {path.name} in {tp._rel(path.parent)}/ (file it with scripts/add_run_record.py).")
    print(f"\nPaste into prl/checks/{MODULE}.py, in the thresholds of {CHECK}:\n")
    print(
        tp.paste_entry(
            res.analysis,
            budget=args.budget,
            settings=res.settings,
            record_name=path.name,
            date=rec["date"],
            where=tp._where(rec),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
