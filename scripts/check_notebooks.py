"""Execute notebooks and check them.

Modes:
  worked   PRL_WORKED=1 (solutions bound). Every cell must run and every
           checkpoint must pass. With --record, the run record the notebook
           emits is filed into runs/inbox/ for scripts/add_run_record.py.
  learner  The notebook as a participant first sees it (stubs). Execution must
           stop inside the first exercise that has a TODO, with the harness's
           "not written yet" message, not somewhere else.

  verify   Worked mode plus, after each exercise's checkpoints, one cell per wrong
           version of that exercise: its stub (from the notebook) and every mutant in
           labs/mutants/<module>.py. Each swaps the wrong code in and reruns checkpoint
           cells: the stub must be rejected by every checkpoint of its exercise, a
           mutant by every checkpoint that calls the function it replaces. Rejection
           means a CheckFailed (a NameError or a crash does not count).

Usage:
  uv run python scripts/check_notebooks.py [--mode worked|learner] [--record] [NOTEBOOK ...]
Environment passed through: PRL_QUICK, PRL_TEST_DOUBLES, PRL_PLATFORM, PRL_SPEC, PRL_SEAT.
"""

from __future__ import annotations

import argparse
import ast
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


MUTANTS = ROOT / "labs" / "mutants"


def load_mutants(module: str) -> list[dict]:
    """Mutants declared with @mutant(ex=..., replaces=..., id=..., why=...) in labs/mutants/<module>.py."""
    path = MUTANTS / f"{module}.py"
    if not path.exists():
        return []
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and getattr(dec.func, "id", getattr(dec.func, "attr", "")) == "mutant"
            ):
                kw = {k.arg: ast.literal_eval(k.value) for k in dec.keywords}
                clean = ast.parse(ast.unparse(node)).body[0]
                clean.decorator_list = []
                if clean.name != kw["replaces"]:
                    raise ValueError(
                        f"{path.name}: mutant {kw['id']} must be named {kw['replaces']}"
                    )
                out.append(
                    {
                        "ex": str(kw["ex"]),
                        "id": kw["id"],
                        "why": kw.get("why", ""),
                        "sources": {kw["replaces"]: ast.unparse(clean)},
                        "stub": False,
                    }
                )
    return out


def stub_mutants(nb: nbformat.NotebookNode) -> list[dict]:
    out = []
    for cell in nb.cells:
        m = re.match(r"ex([0-9]+[a-z]?)-stub$", cell.get("id", "")) or re.match(
            r"(stretch)-stub$", cell.get("id", "")
        )
        if m:
            names = re.findall(r"^(?:def|class)\s+([A-Za-z_]\w*)", cell.source, re.M)
            out.append(
                {
                    "ex": m.group(1),
                    "id": "stub",
                    "why": "the unwritten TODO",
                    "sources": {name: cell.source for name in names},
                    "stub": True,
                }
            )
    return out


def inject_mutants(nb: nbformat.NotebookNode, module: str) -> tuple[nbformat.NotebookNode, int]:
    """Insert one verification cell per wrong version after its exercise's last checkpoint."""
    wrongs = stub_mutants(nb) + load_mutants(module)
    chk = {}
    for i, cell in enumerate(nb.cells):
        m = re.match(r"ex([0-9]+[a-z]?)-chk", cell.get("id", ""))
        if m:
            chk.setdefault(m.group(1), []).append(i)
        elif cell.get("id", "").startswith("stretch") and "lab.check(" in cell.get("source", ""):
            chk.setdefault("stretch", []).append(i)
    for w in wrongs:
        if w["ex"] not in chk:
            raise ValueError(
                f"{module}: mutant {w['id']} targets exercise {w['ex']}, which has no checkpoint"
            )
    inserts: dict[int, list] = {}
    for w in wrongs:
        cells = chk[w["ex"]]
        if not w["stub"]:
            # A mutant of one function must be rejected by every checkpoint that calls it.
            names = list(w["sources"])
            cells = [
                i
                for i in cells
                if any(re.search(rf"\b{re.escape(n)}\b", nb.cells[i].source) for n in names)
            ]
            if not cells:
                raise ValueError(f"{module}: no checkpoint of exercise {w['ex']} calls {names}")
        sources = [nb.cells[i].source for i in cells]
        code = (
            f"lab._mutant_begin({w['ex']!r}, {w['id']!r}, {w['sources']!r}, stub={w['stub']})\n"
            f"for _src in {sources!r}:\n"
            "    try:\n"
            "        exec(_src, globals())\n"
            "    except Exception as _exc:\n"
            "        lab._s['mutant']['results'].append({'label': '?', 'rejected': True, "
            "'error_type': type(_exc).__name__, 'message': str(_exc)[:200]})\n"
            f"lab._mutant_end({w['ex']!r})"
        )
        cell = nbformat.v4.new_code_cell(code)
        cell["id"] = f"verify-{w['ex']}-{w['id']}"[:64]
        inserts.setdefault(chk[w["ex"]][-1], []).append(cell)
    for idx in sorted(inserts, reverse=True):
        nb.cells[idx + 1 : idx + 1] = inserts[idx]
    return nb, len(wrongs)


def _first_todo_exercise(nb: nbformat.NotebookNode) -> str | None:
    for cell in nb.cells:
        m = re.match(r"ex([0-9]+[a-z]?)-stub$", cell.get("id", ""))
        if m:
            return m.group(1)
    return None


def run_notebook(path: Path, *, mode: str, timeout: int, record_dir: Path | None) -> dict:
    nb = nbformat.read(path, as_version=4)
    n_wrong = 0
    if mode == "verify":
        nb, n_wrong = inject_mutants(nb, nb.metadata.get("prl", {}).get("module", ""))
    env = dict(os.environ)
    env["PRL_WORKED"] = "0" if mode == "learner" else "1"
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
                if mode == "verify":
                    result["detail"] = f"{n_wrong} wrong version(s) rejected (stubs and mutants)"
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
    if record_dir is not None and records and mode == "worked":  # never file a record from verify
        record_dir.mkdir(parents=True, exist_ok=True)
        for rec_path in records:
            shutil.copy(rec_path, record_dir / rec_path.name)
        result["records"] = [r.name for r in records]
    if records:
        rec = json.loads(records[-1].read_text())
        result["status"] = rec.get("status")
        if mode in ("worked", "verify") and rec.get("status") != "pass":
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
    ap.add_argument("--mode", choices=["worked", "learner", "verify"], default="worked")
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
