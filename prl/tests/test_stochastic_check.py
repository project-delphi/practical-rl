"""StochasticCheck: a seeded metric compared with the threshold of the budget in use."""

import numpy as np
import pytest

from prl.checks import CheckFailed, StochasticCheck

TRUE_MEAN = 1.0


def abs_error(mean_estimate, *, seed, n):
    rng = np.random.default_rng(seed)
    return abs(mean_estimate(rng.normal(TRUE_MEAN, 1.0, size=n)) - TRUE_MEAN)


def solution(samples):
    return float(np.mean(samples))


def biased(samples):
    return float(np.mean(samples)) + 0.5


def make(**overrides):
    kw = dict(
        ex=2,
        functions="mean_estimate",
        metric=abs_error,
        live={"n": 400},
        quick={"n": 100},
        higher_is_better=False,
        min_gap=0.1,
        what="error of your mean",
        thresholds={"live": (0.25, "runs/live.json"), "quick": (0.3, "runs/quick.json")},
    )
    kw.update(overrides)
    return StochasticCheck(**kw)


def test_declaration_is_normalized_and_validated():
    check = make()
    assert check.functions == ("mean_estimate",) and check.ex == "2"
    with pytest.raises(ValueError, match="min_gap"):
        make(min_gap=0)
    with pytest.raises(ValueError, match="same keys"):
        make(quick={"m": 1})
    with pytest.raises(ValueError, match="seed"):
        make(live={"seed": 1}, quick={"seed": 2})
    with pytest.raises(ValueError, match="'live' or 'quick'"):
        make(thresholds={"fast": (1.0, "x")})


def test_quick_budget_uses_the_quick_threshold(monkeypatch):
    monkeypatch.setenv("PRL_QUICK", "1")
    check = make()
    assert check.settings().source == "quick" and dict(check.settings()) == {"n": 100}
    check(solution)  # |N(0, 0.1)| at seed 0 is far below 0.3
    with pytest.raises(CheckFailed, match=r"at most 0\.3 \(threshold from runs/quick\.json\)"):
        check(biased)


def test_live_budget_uses_the_live_threshold_and_numbers(monkeypatch):
    monkeypatch.setenv("PRL_QUICK", "0")
    seen = {}

    def metric(fn, *, seed, n):
        seen.update(seed=seed, n=n)
        return abs_error(fn, seed=seed, n=n)

    check = make(metric=metric, seed=7)
    check(solution)
    assert seen == {"seed": 7, "n": 400}
    with pytest.raises(CheckFailed, match="runs/live.json"):
        check(biased)


def test_measure_matches_the_metric():
    check = make()
    assert check.measure(solution, seed=3, budget={"n": 50}) == abs_error(solution, seed=3, n=50)


def test_missing_threshold_is_an_authoring_error(monkeypatch):
    monkeypatch.setenv("PRL_QUICK", "1")
    check = make(thresholds={"live": (0.25, "runs/live.json")})
    with pytest.raises(RuntimeError, match="threshold_protocol"):
        check(solution)


def test_higher_is_better_direction(monkeypatch):
    monkeypatch.setenv("PRL_QUICK", "1")

    def closeness(fn, *, seed, n):
        return 1.0 - abs_error(fn, seed=seed, n=n)

    check = make(metric=closeness, higher_is_better=True, thresholds={"quick": (0.7, "r.json")})
    check(solution)
    with pytest.raises(CheckFailed, match="at least 0.7"):
        check(biased)
