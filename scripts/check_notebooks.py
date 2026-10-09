"""Execute notebooks and check them.

Modes:
  worked   PRL_WORKED=1 (solutions bound). Every cell must run and every
           checkpoint must pass. With --record, the run record the notebook
           emits is filed into runs/inbox/ for scripts/add_run_record.py.
  learner  The notebook as a participant first sees it (stubs). Execution must
           stop inside the first exercise that has a TODO, with the harness's
           "not written yet" message, not somewhere else.

Mutant checks (every checkpoint rejects every mutant) arrive with Module 1 in Phase 2.

Usage:
  uv run python scripts/check_notebooks.py [--mode worked|learner] [--record] [NOTEBOOK ...]
Environment passed through: PRL_QUICK, PRL_TEST_DOUBLES, PRL_PLATFORM, PRL_SPEC, PRL_SEAT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def _first_todo_exercise(nb: nbformat.NotebookNode) -> str | None:
    for cell in nb.cells:
        m = re.match(r"ex([0-9]+[a-z]?)-stub$", cell.get("id", ""))
        if m:
            return m.group(1)
    return None


def run_notebook(path: Path, *, mode: str, timeout: int, record_dir: Path | None) -> dict:
    nb = nbformat.read(path, as_version=4)
    env = dict(os.environ)
    env["PRL_WORKED"] = "1" if mode == "worked" else "0"
    env.setdefault("MPLBACKEND", "Agg")
    work = Path(tempfile.mkdtemp(prefix=f"prl-{path.stem}-"))
    env["PRL_RECORD_DIR"] = str(work / "records")
    old_env = dict(os.environ)
    os.environ.update(env)  # the kernel inherits the parent's environment
    start = time.perf_counter()
    result = {"notebook": path.stem, "mode": mode, "ok": False, "seconds": 0.0, "detail": ""}
    try:
        client = NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(work)}}
        )
        try:
            client.execute()
            if mode == "learner":
                result["detail"] = "ran to the end, but should have stopped at the first TODO"
            else:
                result["ok"] = True
        except CellExecutionError as exc:
            failed = _failed_cell(nb)
            text = str(exc)
            if mode == "learner":
                first = _first_todo_exercise(nb)
                fid = failed.get("id", "?") if failed else "?"
                in_first = first is not None and re.match(rf"ex{first}-(run|chk)", fid or "")
                says_todo = ("TODO" in text) or ("not written" in text)
                result["ok"] = bool(in_first and says_todo)
                result["detail"] = f"stopped in cell {fid}" + (
                    "" if result["ok"] else f": {text[-400:]}"
                )
            else:
                fid = failed.get("id", "?") if failed else "?"
                result["detail"] = f"failed in cell {fid}: {text[-1500:]}"
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        result["seconds"] = round(time.perf_counter() - start, 1)
    records = sorted((work / "records").glob("*.json")) if (work / "records").exists() else []
    if record_dir is not None and records and mode == "worked":
        record_dir.mkdir(parents=True, exist_ok=True)
        for rec_path in records:
            shutil.copy(rec_path, record_dir / rec_path.name)
        result["records"] = [r.name for r in records]
    if records:
        rec = json.loads(records[-1].read_text())
        result["status"] = rec.get("status")
        if mode == "worked" and rec.get("status") != "pass":
            result["ok"] = False
            result["detail"] += f" run record status is {rec.get('status')!r}"
    shutil.rmtree(work, ignore_errors=True)
    return result


def _failed_cell(nb: nbformat.NotebookNode) -> dict | None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                return cell
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("notebooks", nargs="*", help="notebook names or paths (default: all)")
    ap.add_argument("--mode", choices=["worked", "learner"], default="worked")
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("PRL_CELL_TIMEOUT", "600")))
    ap.add_argument(
        "--record", action="store_true", help="file the emitted run records in runs/inbox/"
    )
    args = ap.parse_args(argv)
    paths = []
    for name in args.notebooks or [p.stem for p in sorted(NB_DIR.glob("*.ipynb"))]:
        p = Path(name)
        if not p.suffix:
            p = NB_DIR / f"{name}.ipynb"
        paths.append(p if p.is_absolute() else (ROOT / p if not p.exists() else p))
    record_dir = ROOT / "runs" / "inbox" if args.record else None
    failures = 0
    for p in paths:
        res = run_notebook(p, mode=args.mode, timeout=args.timeout, record_dir=record_dir)
        mark = "PASS" if res["ok"] else "FAIL"
        print(
            f"{mark} {args.mode:7s} {res['notebook']:32s} {res['seconds']:7.1f} s  {res['detail']}".rstrip()
        )
        failures += 0 if res["ok"] else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
