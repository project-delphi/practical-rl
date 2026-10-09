"""Load and validate _variables.yml, and derive schedule facts from it.

Imported by the generators and the tests; run directly to validate:
    uv run python scripts/variables.py
"""

from __future__ import annotations

import re
import sys
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARS = ROOT / "_variables.yml"
DESIGNED = {"colab-cpu", "colab-t4"}
SLOTS = ("A", "B", "C")


@cache
def load() -> dict[str, Any]:
    return yaml.safe_load(VARS.read_text())


def minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def hhmm(total: int) -> str:
    return f"{total // 60:02d}:{total % 60:02d}"


def module_ids(v: dict[str, Any] | None = None) -> list[str]:
    v = v or load()
    return sorted(v["modules"], key=lambda k: v["modules"][k]["n"])


def slot_times(slot: str, v: dict[str, Any] | None = None) -> dict[str, tuple[str, str]]:
    """Start and end of each part (briefing, lab, debrief) of a module slot."""
    v = v or load()
    out: dict[str, tuple[str, str]] = {}
    for block in v["clock"]["blocks"]:
        if block["kind"] != "module" or block["slot"] != slot:
            continue
        t = minutes(block["start"])
        for part, mins in block["parts"]:
            out[part] = (hhmm(t), hhmm(t + mins))
            t += mins
    return out


def capstone_times(v: dict[str, Any] | None = None) -> dict[str, tuple[str, str]]:
    v = v or load()
    c_start = next(b for b in v["clock"]["blocks"] if b["kind"] == "module" and b["slot"] == "C")[
        "start"
    ]
    t = minutes(c_start)
    out = {}
    for part, mins in v["clock"]["capstone_parts"]:
        out[part] = (hhmm(t), hhmm(t + mins))
        t += mins
    return out


def core_minutes(mod: dict[str, Any]) -> int:
    return sum(int(e["minutes"]) for e in mod.get("exercises", []))


def module_day(mid: str, v: dict[str, Any] | None = None) -> str | None:
    v = v or load()
    for did, day in v["days"].items():
        if mid in day["modules"]:
            return did
    return None


def validate(v: dict[str, Any]) -> list[str]:
    p: list[str] = []
    for key in (
        "repo",
        "prl",
        "workshop",
        "clock",
        "days",
        "levels",
        "packages",
        "models",
        "runtimes",
        "readiness",
        "modules",
    ):
        if key not in v:
            p.append(f"missing top-level key {key!r}")
    if p:
        return p

    # Clock: contiguous from 09:00 to 17:00 (the clinic sits before it), parts fill blocks.
    blocks = [b for b in v["clock"]["blocks"] if b["kind"] != "clinic"]
    for prev, nxt in zip(blocks, blocks[1:], strict=False):
        if prev["end"] != nxt["start"]:
            p.append(f"clock gap or overlap between {prev['end']} and {nxt['start']}")
    if blocks[0]["start"] != "09:00" or blocks[-1]["end"] != "17:00":
        p.append("the day must run 09:00-17:00")
    for b in v["clock"]["blocks"]:
        span = minutes(b["end"]) - minutes(b["start"])
        if b["kind"] == "module" and sum(m for _, m in b["parts"]) != span:
            p.append(f"module block {b['start']}-{b['end']} parts do not fill it")
    for slot in SLOTS:
        parts = slot_times(slot, v)
        got = {k: minutes(e) - minutes(s) for k, (s, e) in parts.items()}
        if got != {"briefing": 40, "lab": 70, "debrief": 10}:
            p.append(f"slot {slot} must have briefing 40, lab 70, debrief 10; has {got}")
    if sum(m for _, m in v["clock"]["capstone_parts"]) != 120:
        p.append("capstone parts must sum to 120 minutes")

    budgets = v["clock"]["budgets"]
    mods = v["modules"]
    ids = module_ids(v)
    if ids != [f"m{i:02d}" for i in range(len(ids))]:
        p.append(f"module keys must be m00..m{len(ids) - 1:02d} in order")
    seen_days = []
    for did, day in v["days"].items():
        if len(day["modules"]) != 3:
            p.append(f"{did} must list three modules")
        for slot, mid in zip(SLOTS, day["modules"], strict=False):
            if mid not in mods:
                p.append(f"{did} lists unknown module {mid}")
                continue
            if mods[mid]["slot"] != slot:
                p.append(f"{mid} is in slot {slot} of {did} but says slot {mods[mid]['slot']}")
            if mods[mid]["day"] != day["n"]:
                p.append(f"{mid} says day {mods[mid]['day']} but is listed on {did}")
            seen_days.append(mid)
    for mid in ids:
        m = mods[mid]
        if m["n"] != int(mid[1:]):
            p.append(f"{mid}: n must be {int(mid[1:])}")
        if not re.fullmatch(rf"{m['n']:02d}-[a-z0-9]+(-[a-z0-9]+)*", m["slug"]):
            p.append(f"{mid}: slug {m['slug']!r} must look like '{m['n']:02d}-kebab-case'")
        if m["level"] not in v["levels"]:
            p.append(f"{mid}: unknown level {m['level']}")
        if m["runtime"] not in DESIGNED or m["runtime"] not in v["runtimes"]:
            p.append(f"{mid}: runtime must be one of {sorted(DESIGNED)}")
        n_obj = len(m.get("objectives", []))
        if not 3 <= n_obj <= 4:
            p.append(f"{mid}: needs 3-4 objectives, has {n_obj}")
        for key in (
            "title",
            "summary",
            "question",
            "where_we_are",
            "cost",
            "without_gpu",
            "scaffold",
        ):
            if not m.get(key):
                p.append(f"{mid}: missing {key}")
        if m.get("compared_with") and m["compared_with"]["kind"] not in {
            "truth",
            "library",
            "band",
        }:
            p.append(f"{mid}: compared_with.kind must be truth, library or band")
        if mid not in ("m00", "m15") and mid not in seen_days:
            p.append(f"{mid} is not scheduled on any day")
        ex = m.get("exercises", [])
        labels = [str(e["n"]) for e in ex]
        if len(labels) != len(set(labels)):
            p.append(f"{mid}: duplicate exercise numbers")
        if m["slot"] in SLOTS and ex:
            limit = budgets["lab_core_max"][m["slot"]]
            core = core_minutes(m)
            if core > limit:
                p.append(f"{mid}: core is {core} min, over the slot-{m['slot']} limit of {limit}")
            for e in ex:
                is_reconnect = m["slot"] == "B" and str(e["n"]) == "0"
                floor = budgets["lab_reconnect_B"] if is_reconnect else budgets["exercise_min"]
                if int(e["minutes"]) < floor:
                    p.append(
                        f"{mid}: exercise {e['n']} is {e['minutes']} min, under the {floor}-min floor"
                    )
            if m["slot"] == "B" and str(ex[0]["n"]) != "0":
                p.append(f"{mid}: a slot-B lab must start with the 3-minute reconnect row (n: 0)")
    if mods["m00"].get("minutes") != core_minutes(mods["m00"]):
        p.append("m00: exercise minutes must sum to its stated minutes")
    for rid, item in v["readiness"]["items"].items():
        if "title" not in item or "check" not in item:
            p.append(f"readiness item {rid} needs title and check")
    return p


def main() -> int:
    problems = validate(load())
    for prob in problems:
        print(f"_variables.yml: {prob}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
