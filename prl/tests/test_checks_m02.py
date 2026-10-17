"""prl.checks.m02 accepts correct code (with either tie-break) and rejects wrong versions.

The notebook's verify mode covers Exercises 1-5 against labs/mutants/m02.py. This file
covers what verify mode does not: the stretch's wrong versions (tests/test_mutants.py
accepts only exercises declared in _variables.yml), and a correct greedy that breaks
the shop's exact ties the other way. The reference code here is test code, not prl.
"""

import numpy as np
import pytest

from prl.checks import CheckFailed, m02


def value_iteration(P, R, gamma, tol):
    V, k = np.zeros(P.shape[0]), 0
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, R + gamma * P @ V_new, k
        V = V_new


def greedy_first(Q):
    return np.argmax(Q, axis=1)


def greedy_last(Q):
    """Also correct: the LAST maximizer (the shop's capped orders tie exactly)."""
    Q = np.asarray(Q)
    return Q.shape[1] - 1 - np.argmax(Q[:, ::-1], axis=1)


def make_mpi(greedy, *, sweeps=True, count_sweeps=False):
    def modified_policy_iteration(P, R, gamma, m, tol):
        S = P.shape[0]
        V, rounds = np.zeros(S), 0
        while True:
            Q = R + gamma * P @ V
            pi = greedy(Q)
            V_new = Q.max(axis=1)
            P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
            for _ in range(m - 1 if sweeps else 0):
                V_new = r_pi + gamma * P_pi @ V_new
            rounds += m if count_sweeps else 1
            if np.max(np.abs(V_new - V)) < tol:
                return V_new, greedy(R + gamma * P @ V_new), rounds
            V = V_new

    return modified_policy_iteration


def test_value_iteration_and_greedy_pass():
    m02.check_value_iteration(value_iteration)
    m02.check_greedy(greedy_first)
    m02.check_greedy(greedy_last)


def test_value_iteration_rejects_the_mean_stop():
    def mean_stop(P, R, gamma, tol):
        V, k = np.zeros(P.shape[0]), 0
        while True:
            V_new = (R + gamma * P @ V).max(axis=1)
            k += 1
            if np.mean(np.abs(V_new - V)) < tol:
                return V_new, R + gamma * P @ V_new, k
            V = V_new

    with pytest.raises(CheckFailed, match="guarantees at most"):
        m02.check_value_iteration(mean_stop)


@pytest.mark.parametrize("greedy", [greedy_first, greedy_last])
def test_modified_policy_iteration_passes_with_either_tie_break(greedy):
    m02.check_modified_policy_iteration(make_mpi(greedy))


@pytest.mark.parametrize(
    "wrong", [{"sweeps": False}, {"count_sweeps": True}], ids=["no-eval-sweeps", "counts-sweeps"]
)
def test_modified_policy_iteration_rejects_the_stretch_mutants(wrong):
    with pytest.raises(CheckFailed, match="rounds"):
        m02.check_modified_policy_iteration(make_mpi(greedy_first, **wrong))
