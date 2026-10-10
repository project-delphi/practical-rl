"""The notebook harness: solutions, checkpoints and the run record.

Rules it enforces:
- A folded solution never replaces participant code. `@lab.solution(n)` stores
  the reference and hands back whatever the participant already defined
  (or a stand-in that says "write your TODO first"). Only worked mode
  (PRL_WORKED=1, or the WORKED checkbox) binds the reference.
- `lab.use_reference(n)` lets a stuck participant continue with the reference.
- `lab.check(n, fn, *objs)` runs a checkpoint and reports whose code ran.
- `lab.finish()` writes the run record.

State lives as plain data in ns["__prl_lab__"], so rerunning the init cell or
reloading prl keeps the stored references.
"""

from __future__ import annotations

import datetime as _dt
import time
from collections.abc import Callable
from typing import Any

from . import API, __version__, record, runtime
from .checks import CheckFailed

_STATE_KEY = "__prl_lab__"
_MISSING = object()
_STANDIN = "__prl_standin__"


class APIVersionError(RuntimeError):
    pass


def _standin(n: int, name: str) -> Callable[..., Any]:
    def not_written(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Exercise {n}: `{name}` is not written yet. Run your `# TODO {n}` cell first, "
            f"or call lab.use_reference({n!r}) to continue with the reference."
        )

    not_written.__name__ = name
    setattr(not_written, _STANDIN, True)
    return not_written


class _NotWrittenValue:
    """Placeholder for a non-function solution value the participant has not set."""

    def __init__(self, n: int, name: str):
        self._n, self._name = n, name

    def _fail(self, *_a: Any, **_k: Any) -> Any:
        raise NotImplementedError(
            f"Exercise {self._n}: `{self._name}` is not set yet. Run your `# TODO {self._n}` "
            f"cell first, or call lab.use_reference({self._n!r})."
        )

    __call__ = _fail
    __array__ = _fail
    __float__ = _fail
    __len__ = _fail
    __iter__ = _fail
    __getitem__ = _fail

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return self._fail()

    def __repr__(self) -> str:
        return f"<not written yet: {self._name} (exercise {self._n})>"


def _tag_reference(obj: Any, n: int | str, name: str) -> None:
    try:
        obj.__prl_ref__ = (str(n), name)
    except (AttributeError, TypeError):  # objects that refuse attributes are compared by identity
        pass


def _is_ref(obj: Any, n: int | str, name: str) -> bool:
    return getattr(obj, "__prl_ref__", None) == (str(n), name)


def _is_standin(obj: Any) -> bool:
    return bool(getattr(obj, _STANDIN, False)) or isinstance(obj, _NotWrittenValue)


class Lab:
    """Handle on the harness state stored in a notebook namespace."""

    def __init__(self, ns: dict[str, Any]):
        self._ns = ns

    # -- state -------------------------------------------------------------
    @property
    def _s(self) -> dict[str, Any]:
        return self._ns[_STATE_KEY]

    @property
    def worked(self) -> bool:
        return bool(self._s["worked"])

    @property
    def notebook(self) -> str:
        return self._s["notebook"]

    # -- solutions ---------------------------------------------------------
    def _refs(self, n: int | str) -> dict[str, Any]:
        return self._s["refs"].setdefault(str(n), {})

    def _say_once(self, key: str, message: str) -> None:
        said = self._s.setdefault("said", set())
        if key not in said:
            said.add(key)
            print(message)

    def solution(self, n: int | str) -> Callable[[Any], Any]:
        """Decorator for a folded solution of exercise n."""

        def decorate(obj: Any) -> Any:
            name = obj.__name__
            _tag_reference(obj, n, name)
            self._refs(n)[name] = obj
            if self.worked:
                return obj
            current = self._ns.get(name)
            if current is None or _is_standin(current):
                self._say_once(
                    f"missing-{n}",
                    f"Solution {n} stored. Run your `# TODO {n}` cell first, "
                    f"or run lab.use_reference({n!r}) to continue with this solution.",
                )
                return _standin(n, name)
            if _is_ref(current, n, name):
                return obj  # they chose the reference; keep it current
            # Running a solution cell never replaces the participant's code; say so plainly.
            self._say_once(
                f"kept-{n}",
                f"Solution {n} stored, not used: the code from your `# TODO {n}` "
                f"cell stays in place. To continue with the solution, run "
                f"lab.use_reference({n!r}).",
            )
            return current

        return decorate

    def solution_value(self, n: int | str, name: str, value: Any) -> Any:
        """Store a non-function reference value for exercise n; return what the participant has."""
        if isinstance(value, (int, float, bool, str)):
            raise TypeError(
                "solution_value is for arrays and objects; check scalars in the checkpoint instead"
            )
        self._refs(n)[name] = value
        if self.worked:
            return value
        current = self._ns.get(name)
        if current is None or _is_standin(current):
            return _NotWrittenValue(n, name)
        return current

    def use_reference(self, n: int | str, *names: str) -> None:
        """Continue with the reference solution of exercise n (optionally only some of its names)."""
        refs = self._refs(n)
        if not refs:
            raise KeyError(f"No reference stored for exercise {n!r}. Run its solution cell first.")
        chosen = names or tuple(refs)
        unknown = [x for x in chosen if x not in refs]
        if unknown:
            raise KeyError(f"Exercise {n!r} has no reference for {unknown}; it has {sorted(refs)}.")
        for name in chosen:
            self._ns[name] = refs[name]
        shown = ", ".join(f"`{k}`" for k in chosen)
        print(f"Exercise {n}: now using the REFERENCE for {shown}. Checkpoints will say so.")

    def _whose(self, n: int | str) -> str:
        refs = self._s["refs"].get(str(n)) or {}
        if not refs:
            return "provided"
        current = [(name, self._ns.get(name)) for name in refs]
        if any(c is None or _is_standin(c) for _, c in current):
            return "not written"
        is_ref = [_is_ref(c, n, name) for name, c in current]
        if all(is_ref):
            return "reference"
        if any(is_ref):
            return "mixed"
        return "yours"

    # -- checkpoints -------------------------------------------------------
    def check(self, n: int, fn: Callable[..., Any], *objs: Any, label: str | None = None) -> None:
        """Run checkpoint `label` (default: str(n)) for exercise n."""
        label = label or str(n)
        whose = self._whose(n)
        start = time.perf_counter()
        error: CheckFailed | None = None
        try:
            fn(*objs)
        except CheckFailed as exc:
            error = exc
        except NotImplementedError as exc:
            error = CheckFailed(str(exc) or f"Exercise {n} is not written yet.")
            error.__cause__ = exc
        except Exception as exc:  # noqa: BLE001 - report any crash as a failed checkpoint
            error = CheckFailed(f"Your code raised {type(exc).__name__}: {exc}")
            error.__cause__ = exc
        seconds = round(time.perf_counter() - start, 3)
        mutant = self._s.get("mutant")
        if mutant is not None:
            cause = error.__cause__ if error is not None and error.__cause__ else None
            mutant["results"].append(
                {
                    "label": label,
                    "rejected": error is not None,
                    "error_type": type(cause).__name__
                    if cause is not None
                    else ("CheckFailed" if error is not None else None),
                    "message": str(error)[:200] if error is not None else "",
                }
            )
            return
        result = {
            "label": label,
            "exercise": n,
            "pass": error is None,
            "whose": whose,
            "seconds": seconds,
        }
        results = [r for r in self._s["checks"] if r["label"] != label]
        results.append(result)
        self._s["checks"] = results
        if error is None:
            print(f"✓ Checkpoint {label} passed ({_whose_words(whose)}) · {seconds:.1f} s")
            return
        print(f"✗ Checkpoint {label} failed: {error}")
        if whose != "reference":
            print(f"  Stuck? Open the Hint, then the Solution, or run lab.use_reference({n!r}).")
        raise error

    # -- mutants (used by scripts/check_notebooks.py --mode verify) -------------
    def _mutant_begin(
        self, n: int, mutant_id: str, sources: dict[str, str], *, stub: bool = False
    ) -> None:
        """Swap in a wrong version of exercise n's functions; checks then record, not raise."""
        saved = {name: self._ns.get(name, _MISSING) for name in sources}
        for name, src in sources.items():
            exec(compile(src, f"<mutant {mutant_id}>", "exec"), self._ns)  # noqa: S102
            if name not in self._ns:
                raise RuntimeError(f"mutant {mutant_id} does not define {name}")
        self._s["mutant"] = {"n": n, "id": mutant_id, "stub": stub, "saved": saved, "results": []}

    def _mutant_end(self, n: int) -> None:
        """Restore the real functions; fail unless every checkpoint rejected the mutant."""
        mutant = self._s.pop("mutant")
        for name, obj in mutant["saved"].items():
            if obj is _MISSING:
                self._ns.pop(name, None)
            else:
                self._ns[name] = obj
        results = mutant["results"]
        allowed = {"CheckFailed"} | ({"NotImplementedError"} if mutant["stub"] else set())
        problems = []
        if not results:
            problems.append("no checkpoint ran")
        for r in results:
            if not r["rejected"]:
                problems.append(f"checkpoint {r['label']} PASSED")
            elif r["error_type"] not in allowed:
                problems.append(
                    f"checkpoint {r['label']} failed with {r['error_type']}, not a check: {r['message']}"
                )
        kind = "stub" if mutant["stub"] else "mutant"
        if problems:
            raise AssertionError(
                f"Exercise {n} {kind} {mutant['id']!r} was not rejected: " + "; ".join(problems)
            )
        print(f"✓ exercise {n}: {kind} {mutant['id']!r} rejected by {len(results)} checkpoint(s)")

    # -- metrics and records --------------------------------------------------
    def metric(self, key: str, value: Any) -> None:
        self._s["metrics"][key] = value

    def seeds(self, seeds: list[int]) -> None:
        self._s["seeds"] = sorted({int(s) for s in seeds} | set(self._s["seeds"]))

    def summary(self) -> None:
        checks = self._s["checks"]
        expected_labels = self._s["checkpoints"] or [r["label"] for r in checks]
        core = [r for r in checks if r["label"] in expected_labels]
        optional = [r for r in checks if r["label"] not in expected_labels]
        passed = {r["label"] for r in core if r["pass"]}
        print(f"{self.notebook}: {len(passed)} of {len(expected_labels)} core checkpoints passed.")
        for r in core:
            mark = "✓" if r["pass"] else "✗"
            print(f"  {mark} {r['label']:>5}  {_whose_words(r['whose'])}")
        for r in optional:
            mark = "✓" if r["pass"] else "✗"
            print(f"  {mark} {r['label']:>5}  {_whose_words(r['whose'])} (optional)")
        missing = [lbl for lbl in expected_labels if lbl not in {r["label"] for r in checks}]
        if missing:
            print(f"  Not reached: {', '.join(missing)}")

    def status(self) -> str:
        checks = {r["label"]: r for r in self._s["checks"]}
        expected_labels = self._s["checkpoints"] or list(checks)
        if expected_labels and all(
            lbl in checks and checks[lbl]["pass"] for lbl in expected_labels
        ):
            return "pass"
        return "fail"

    def build_record(
        self,
        *,
        source: str = "tool",
        scope: str = "notebook",
        scope_note: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        s = self._s
        rt = runtime.detect()
        env = s["env"] or _default_env(rt, s["designed"])
        rec: dict[str, Any] = {
            "schema": record.SCHEMA,
            "kind": "notebook",
            "notebook": s["notebook"],
            "content_sha": s["content_sha"],
            "prl_version": __version__,
            "git_sha": s.get("git_sha"),
            "badge_ref": s.get("badge_ref"),
            "date": _dt.datetime.now(_dt.UTC).date().isoformat(),
            "platform": rt.platform,
            "env": env,
            "colab_release": rt.colab_release,
            "hardware": {
                "cpu": rt.cpu,
                "n_cpu": rt.n_cpu,
                "ram_gb": rt.ram_gb,
                "accel": rt.accel,
                "gpu": rt.gpu,
            },
            "torch_threads": rt.torch_threads,
            "python": rt.python,
            "packages": runtime.package_versions(list(runtime.DEFAULT_PACKAGES) + s["packages"]),
            "quick": bool(s["quick"]),
            "quick_reason": s["quick_reason"],
            "test_doubles": runtime.test_doubles_used(),
            "shims": runtime.shims_used(),
            "solutions_bound": self.worked,
            "scope": scope,
            "seeds": s["seeds"],
            "seat": s["seat"],
            "settings": s["settings"],
            "seconds": round(time.time() - s["started"], 1),
            "checkpoints": s["checks"],
            "metrics": s["metrics"],
            "status": self.status(),
            "source": source,
        }
        if scope_note:
            rec["scope_note"] = scope_note
        if note:
            rec["note"] = note
        return rec

    def finish(self, **kwargs: Any) -> dict[str, Any]:
        """Print the summary and emit this run's record."""
        self.summary()
        rec = self.build_record(**kwargs)
        record.emit(rec)
        return rec

    def use_settings(self, settings: runtime.Settings) -> runtime.Settings:
        """Record which budget this run used."""
        self._s["settings"] = dict(settings)
        self._s["quick"] = settings.quick
        self._s["quick_reason"] = settings.reason
        return settings


def _whose_words(whose: str) -> str:
    return {
        "yours": "your code",
        "reference": "on the REFERENCE solution",
        "mixed": "partly on the REFERENCE solution",
        "provided": "provided code",
        "not written": "not written yet",
    }.get(whose, whose)


def _default_env(rt: runtime.Runtime, designed: str) -> str:
    if rt.platform == "colab":
        return "colab-t4" if rt.accel == "cuda" else "colab-cpu"
    if rt.platform == "aws":
        return "aws-g4dn"
    if rt.platform == "ci":
        return "ci"
    return "local"


def init(
    notebook: str,
    content_sha: str,
    *,
    designed: str,
    api: int,
    ns: dict[str, Any],
    checkpoints: list[str] | None = None,
    packages: list[str] | None = None,
    seat: int | None = None,
    env: str | None = None,
    git_sha: str | None = None,
    badge_ref: str | None = None,
    worked: bool | None = None,
) -> Lab:
    """Start (or resume) the harness in namespace `ns` (pass globals())."""
    if api != API:
        raise APIVersionError(
            f"This notebook was built for prl API {api}, but prl {__version__} provides API {API}. "
            "Rerun the setup cell (it installs the matching prl), then restart the session."
        )
    if designed not in runtime.DESIGNED_RUNTIMES:
        raise ValueError(f"designed must be one of {sorted(runtime.DESIGNED_RUNTIMES)}")
    is_quick, reason = runtime.quick(designed)
    state = ns.get(_STATE_KEY)
    fresh = not isinstance(state, dict) or state.get("notebook") != notebook
    if fresh:
        state = {
            "notebook": notebook,
            "refs": {},
            "checks": [],
            "metrics": {},
            "seeds": [],
            "settings": {},
            "started": time.time(),
        }
    state.update(
        {
            "content_sha": content_sha,
            "designed": designed,
            "api": api,
            "worked": runtime.worked_mode() if worked is None else bool(worked),
            "checkpoints": list(checkpoints or []),
            "packages": list(packages or []),
            "quick": is_quick,
            "quick_reason": reason,
            "seat": seat,
            "env": env,
            "git_sha": git_sha,
            "badge_ref": badge_ref,
        }
    )
    ns[_STATE_KEY] = state
    lab = Lab(ns)
    mode = "WORKED (solutions bound)" if lab.worked else "your code"
    print(
        f"{notebook} · prl {__version__} · designed for {designed} · QUICK = {is_quick} ({reason}) · {mode}"
    )
    return lab


def mutant(*, ex: int | str, replaces: str, id: str, why: str):  # noqa: A002
    """Mark a deliberately wrong version of an exercise function (labs/mutants/mNN.py).

    scripts/check_notebooks.py --mode verify swaps each mutant in and requires every
    checkpoint of that exercise to reject it.
    """

    def mark(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__prl_mutant__ = {"ex": str(ex), "replaces": replaces, "id": id, "why": why}  # type: ignore[attr-defined]
        return fn

    return mark


__all__ = ["APIVersionError", "CheckFailed", "Lab", "init", "mutant"]
