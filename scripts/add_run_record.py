"""Validate run records and file them into runs/.

uv run python scripts/add_run_record.py                 # file everything in runs/inbox/
uv run python scripts/add_run_record.py rec.json ...    # file specific files
pbpaste | uv run python scripts/add_run_record.py --stdin --source colab-paste
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prl" / "src"))
from prl import record  # noqa: E402

RUNS = ROOT / "runs"


def check(rec: dict, v: dict) -> list[str]:
    problems = record.validate(rec)
    if rec.get("env") not in v["runtimes"]:
        problems.append(f"env {rec.get('env')!r} is not a runtime in _variables.yml")
    slugs = {m["slug"] for m in v["modules"].values()}
    if rec.get("kind") == "notebook" and rec.get("notebook") not in slugs:
        problems.append(f"unknown notebook {rec.get('notebook')!r}")
    if rec.get("platform") == "colab" and not str(rec.get("env", "")).startswith("colab"):
        problems.append("a Colab run must have a colab-* env")
    if rec.get("platform") != "colab" and str(rec.get("env", "")).startswith("colab"):
        problems.append("only runs on Colab may use a colab-* env")
    return problems


def file_record(rec: dict, v: dict, *, dry_run: bool = False) -> Path:
    problems = check(rec, v)
    if problems:
        raise ValueError("; ".join(problems))
    path = RUNS / record.record_filename(rec)
    if not dry_run:
        path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--stdin", action="store_true", help="read printed notebook output from stdin")
    ap.add_argument("--source", choices=sorted(record.SOURCES), help="override the record's source")
    ap.add_argument("--note", help="add a note to every record")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    v = variables.load()
    recs: list[tuple[dict, Path | None]] = []
    if args.stdin:
        recs += [(r, None) for r in record.parse_printed(sys.stdin.read())]
    inbox_files = [Path(f) for f in args.files] or (
        [] if args.stdin else sorted((RUNS / "inbox").glob("*.json"))
    )
    recs += [(json.loads(p.read_text()), p) for p in inbox_files]
    if not recs:
        print("No records to file.")
        return 0
    bad = 0
    for rec, src in recs:
        if args.source:
            rec["source"] = args.source
        if args.note:
            rec["note"] = args.note
        try:
            out = file_record(rec, v, dry_run=args.dry_run)
        except ValueError as exc:
            bad += 1
            print(f"REJECTED {src or 'stdin'}: {exc}")
            continue
        print(f"filed {out.relative_to(ROOT)}")
        if src is not None and src.parent == RUNS / "inbox" and not args.dry_run:
            src.unlink()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
