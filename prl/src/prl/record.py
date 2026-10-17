"""Run records: the only evidence the site uses for "this lab has run".

A record describes one execution of one notebook (kind="notebook") or one
scripted experiment (kind="experiment"). See runs/README.md for the policy.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = 1
BEGIN = "----- BEGIN PRL RUN RECORD -----"
END = "----- END PRL RUN RECORD -----"

PLATFORMS = {"colab", "local", "aws", "ci"}
STATUSES = {"pass", "fail"}
SCOPES = {"notebook", "partial"}
SOURCES = {"tool", "colab-paste", "manual"}

# field -> (type or types, required)
NOTEBOOK_FIELDS: dict[str, tuple[Any, bool]] = {
    "schema": (int, True),
    "kind": (str, True),
    "notebook": (str, True),
    "content_sha": (str, True),
    "prl_version": (str, True),
    "git_sha": ((str, type(None)), False),
    "badge_ref": ((str, type(None)), False),
    "dirty": ((bool, type(None)), False),
    "date": (str, True),
    "platform": (str, True),
    "env": (str, True),
    "colab_release": ((str, type(None)), False),
    "hardware": (dict, True),
    "torch_threads": ((int, type(None)), False),
    "python": (str, True),
    "packages": (dict, True),
    "quick": (bool, True),
    "quick_reason": (str, True),
    "test_doubles": (list, True),
    "shims": (list, True),
    "solutions_bound": (bool, True),
    "scope": (str, True),
    "scope_note": ((str, type(None)), False),
    "seeds": (list, True),
    "seat": ((int, type(None)), False),
    "settings": (dict, True),
    "seconds": ((int, float), True),
    "checkpoints": (list, True),
    "metrics": (dict, True),
    "status": (str, True),
    "source": (str, True),
    "note": ((str, type(None)), False),
}

EXPERIMENT_FIELDS: dict[str, tuple[Any, bool]] = {
    "schema": (int, True),
    "kind": (str, True),
    "experiment": (str, True),
    "script": (str, True),
    "args": (dict, True),
    "prl_version": (str, True),
    "git_sha": ((str, type(None)), False),
    "date": (str, True),
    "platform": (str, True),
    "env": (str, True),
    "hardware": (dict, True),
    "python": (str, True),
    "packages": (dict, True),
    "budget": (dict, True),
    "seeds": (list, True),
    "per_seed": (dict, True),
    "seconds": ((int, float), True),
    "note": ((str, type(None)), False),
}

_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}"
    r"|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]+)"
)
_PRIVATE_PATH = re.compile(r"(/Users/[^/\s]+|/home/(?!runner\b)[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)")
_SHA = re.compile(r"^[0-9a-f]{16}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def validate(rec: dict[str, Any], *, today: _dt.date | None = None) -> list[str]:
    """Return a list of problems with a run record (empty when valid)."""
    problems: list[str] = []
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    kind = rec.get("kind")
    fields = {"notebook": NOTEBOOK_FIELDS, "experiment": EXPERIMENT_FIELDS}.get(kind)
    if fields is None:
        return [f"kind must be 'notebook' or 'experiment', got {kind!r}"]
    for name, (typ, required) in fields.items():
        if name not in rec:
            if required:
                problems.append(f"missing field {name!r}")
            continue
        value = rec[name]
        if isinstance(value, bool) and typ in (int, (int, float)):
            problems.append(f"{name!r} must be a number, not a boolean")
        elif not isinstance(value, typ):
            problems.append(f"{name!r} has type {type(value).__name__}")
    for name in rec:
        if name not in fields:
            problems.append(f"unknown field {name!r}")
    if rec.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    if rec.get("platform") not in PLATFORMS:
        problems.append(f"platform must be one of {sorted(PLATFORMS)}")
    date = rec.get("date", "")
    if isinstance(date, str):
        if not _DATE.match(date):
            problems.append("date must be YYYY-MM-DD")
        else:
            today = today or _dt.datetime.now(_dt.UTC).date()
            if _dt.date.fromisoformat(date) > today + _dt.timedelta(days=1):
                problems.append("date is in the future")
    if kind == "notebook":
        if rec.get("status") not in STATUSES:
            problems.append(f"status must be one of {sorted(STATUSES)}")
        if rec.get("scope") not in SCOPES:
            problems.append(f"scope must be one of {sorted(SCOPES)}")
        if rec.get("scope") == "partial" and not rec.get("scope_note"):
            problems.append("a partial run needs scope_note")
        if rec.get("source") not in SOURCES:
            problems.append(f"source must be one of {sorted(SOURCES)}")
        if isinstance(rec.get("content_sha"), str) and not _SHA.match(rec["content_sha"]):
            problems.append("content_sha must be 16 hex characters")
        for i, cp in enumerate(rec.get("checkpoints") or []):
            if not isinstance(cp, dict) or not {"label", "pass", "whose"} <= set(cp):
                problems.append(f"checkpoints[{i}] needs label, pass and whose")
    for s in _strings(rec):
        if _SECRET.search(s):
            problems.append("record contains something that looks like a credential")
            break
    for s in _strings(rec):
        if _PRIVATE_PATH.search(s):
            problems.append(f"record contains a private path or username: {s[:60]!r}")
            break
    return problems


def _normalize(src: str) -> str:
    lines = src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.rstrip() for line in lines]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def content_sha(cells: Iterable[tuple[str, str]]) -> str:
    """Hash of a notebook's code: (cell id, source) pairs, in order.

    Computed by the build after `# %% include` expansion and before the
    `#@title` lines are injected, over non-generated code cells only. Prose
    edits therefore never make evidence stale.
    """
    joined = "\n\x1e\n".join(f"{cid}\n{_normalize(src)}" for cid, src in cells)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default: cwd) to the directory holding _variables.yml."""
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "_variables.yml").is_file() and (p / "prl").is_dir():
            return p
    return None


def source_commit(start: Path | None = None) -> tuple[str | None, bool | None]:
    """(commit, dirty) of the prl that is running, for a record's git_sha (PLAN.md §7).

    Installed from git (Colab, AWS): the commit pip recorded in prl's direct_url.json,
    with dirty None. Run from a checkout of this repo: its HEAD, and whether the
    working tree has uncommitted changes. Otherwise (None, None).
    """
    try:
        from importlib.metadata import PackageNotFoundError, distribution

        try:
            raw = distribution("prl").read_text("direct_url.json")
        except PackageNotFoundError:
            raw = None
        if raw:
            commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
            if commit:
                return str(commit), None
    except (OSError, ValueError):
        pass
    root = find_repo_root(start) or find_repo_root(Path(__file__).parent)
    if root is None:
        return None, None
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return (sha or None), bool(status.strip())


def record_filename(rec: dict[str, Any]) -> str:
    name = rec.get("notebook") or rec.get("experiment") or "run"
    sha = rec.get("content_sha") or "exp"
    return f"{rec['date']}-{rec['env']}-{name}-{sha[:8]}.json"


def emit(rec: dict[str, Any], *, inbox: Path | None = None) -> Path | None:
    """Print the record between markers and save it where it can be collected.

    Locally it goes to runs/inbox/ (gitignored) for scripts/add_run_record.py.
    On Colab it is also written to the session and offered as a download.
    """
    problems = validate(rec)
    text = json.dumps(rec, indent=2, sort_keys=True)
    print(BEGIN)
    print(text)
    print(END)
    if problems:
        print("This record is not valid yet:")
        for p in problems:
            print(f"  - {p}")
    path: Path | None = None
    if rec.get("platform") == "colab":
        path = Path("/content") / record_filename(rec)
        try:
            path.write_text(text + "\n")
            from google.colab import files  # type: ignore[import-not-found]

            files.download(str(path))
        except Exception as exc:  # noqa: BLE001 - download is a convenience
            print(f"(Could not offer a download: {exc}. Copy the JSON above instead.)")
        return path
    target = inbox
    if target is None:
        env_dir = os.environ.get("PRL_RECORD_DIR")
        if env_dir:
            target = Path(env_dir)
        else:
            root = find_repo_root()
            target = root / "runs" / "inbox" if root else None
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        path = target / record_filename(rec)
        path.write_text(text + "\n")
        print(f"Saved to {path.name} in {target.name}/ (file it with scripts/add_run_record.py).")
    return path


ROOM = "ROOM"
_ROOM_RESERVED = {"seat", "checks", "quick"}
_ROOM_KEY = re.compile(r"[^A-Za-z0-9_]+")
_ROOM_LINE = re.compile(r"\bROOM (m\d\d)((?: [A-Za-z0-9_]+=\S+)*)")


def room_code(
    module: str,
    *,
    seat: int | None,
    passed: int,
    total: int,
    quick: bool,
    metrics: dict[str, Any],
) -> str:
    """The one line a participant pastes for the debrief's room pooling (PLAN.md §3.7).

    ``ROOM m03 seat=7 checks=9/9 quick=0 td_rms=1.23``: the module, the seat (the
    lab's SEED), core checkpoints passed, whether QUICK budgets ran, then every
    scalar metric the lab recorded, to 6 significant digits. Non-scalar metrics are
    left out. A metric named like a fixed field gets the prefix ``m_``.
    """
    parts = [
        ROOM,
        module,
        f"seat={'-' if seat is None else int(seat)}",
        f"checks={int(passed)}/{int(total)}",
        f"quick={int(bool(quick))}",
    ]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, bool):
            text = str(int(value))
        elif isinstance(value, int):
            text = str(value)
        elif isinstance(value, float):
            text = format(value, ".6g")
        else:
            try:  # numpy scalars
                text = format(float(value), ".6g") if getattr(value, "ndim", None) == 0 else ""
            except (TypeError, ValueError):
                text = ""
        if not text:
            continue
        name = _ROOM_KEY.sub("_", str(key)).strip("_") or "metric"
        if name in _ROOM_RESERVED:
            name = f"m_{name}"
        parts.append(f"{name}={text}")
    return " ".join(parts)


def parse_room_codes(text: str) -> list[dict[str, Any]]:
    """Every room code in pasted text, as dicts: module, seat, passed, total, quick, metrics."""
    out = []
    for m in _ROOM_LINE.finditer(text):
        row: dict[str, Any] = {"module": m.group(1), "metrics": {}}
        for pair in m.group(2).split():
            key, _, value = pair.partition("=")
            if key == "seat":
                row["seat"] = None if value == "-" else int(value)
            elif key == "checks":
                passed, _, total = value.partition("/")
                row["passed"], row["total"] = int(passed), int(total)
            elif key == "quick":
                row["quick"] = value == "1"
            else:
                row["metrics"][key] = float(value)
        out.append(row)
    return out


def parse_printed(text: str) -> list[dict[str, Any]]:
    """Extract every record printed between BEGIN and END markers."""
    out = []
    for block in re.findall(re.escape(BEGIN) + r"\s*(\{.*?\})\s*" + re.escape(END), text, re.S):
        out.append(json.loads(block))
    return out
