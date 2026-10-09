"""Write the checkpoint fixtures in prl/src/prl/_expected/.

Fixtures are precomputed expected values, so prl ships answers to checks,
never solver code. This script may compute them however it likes; it is
not part of prl. Deterministic: rerunning it must produce identical files
(CI's drift gate checks this).
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "prl" / "src" / "prl" / "_expected"


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


def _save(name: str, **arrays: np.ndarray) -> bytes:
    return save_npz(**arrays)


def m01_returns() -> bytes:
    cases = [
        ([1.0, 1.0, 1.0], 0.9),
        ([0.0, 0.0, 10.0], 0.5),
        ([-1.0, -1.0, -1.0, -1.0, 0.0], 1.0),
        ([5.0], 0.99),
        ([1.0, -2.0, 3.0, -4.0], 0.8),
    ]
    width = max(len(r) for r, _ in cases)
    rewards = np.zeros((len(cases), width))
    lengths = np.zeros(len(cases), dtype=np.int64)
    gammas = np.zeros(len(cases))
    answers = np.zeros(len(cases))
    for i, (rs, g) in enumerate(cases):
        rewards[i, : len(rs)] = rs
        lengths[i] = len(rs)
        gammas[i] = g
        answers[i] = sum(r * g**k for k, r in enumerate(rs))
    return _save("m01_returns", rewards=rewards, lengths=lengths, gammas=gammas, returns=answers)


FIXTURES = {"m01_returns": m01_returns}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, build in FIXTURES.items():
        blob = build()
        path = OUT / f"{name}.npz"
        if "--check" in argv:
            if not path.exists() or path.read_bytes() != blob:
                stale.append(name)
        else:
            path.write_bytes(blob)
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print(f"Fixtures out of date: {', '.join(stale)}. Run scripts/make_expected.py.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
