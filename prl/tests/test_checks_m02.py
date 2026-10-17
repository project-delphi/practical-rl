"""prl.checks.m02 accepts correct code, whichever way it breaks the shop's exact ties.

The notebook's verify mode rejects every stub and every mutant in labs/mutants/m02.py
(the stretch's included). This file covers what verify mode does not: correct code other
than the reference, such as a greedy that picks the LAST maximizer. The reference code
here is test code, not prl.
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


def make_mpi(greedy):
    def modified_policy_iteration(P, R, gamma, m, tol):
        S = P.shape[0]
        V, rounds = np.zeros(S), 0
        while True:
            Q = R + gamma * P @ V
            pi = greedy(Q)
            V_new = Q.max(axis=1)
            P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
            for _ in range(m - 1):
                V_new = r_pi + gamma * P_pi @ V_new
            rounds += 1
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


def test_planning_loss_bound_and_optimality_residual_pass():
    def bound(eps_P, eps_R, r_range, gamma):
        return 2 * eps_R / (1 - gamma) + 2 * gamma * eps_P * r_range / (1 - gamma) ** 2

    def residual(P, R, gamma, V):
        return float(np.max(np.abs((R + gamma * P @ V).max(axis=1) - V)))

    m02.check_planning_loss_bound(bound)
    m02.check_optimality_residual(residual)


def test_optimality_residual_rejects_a_vector():
    with pytest.raises(CheckFailed, match="one number"):
        m02.check_optimality_residual(lambda P, R, gamma, V: np.abs((R + gamma * P @ V).max(1) - V))


def make_policy_iteration(*, tie_tol, noise_seed=None):
    """Policy iteration whose P @ V may carry a few ulps of noise per entry, as another BLAS
    kernel would produce (OpenBLAS splits the shop's exact ties this way)."""
    rng = np.random.default_rng(noise_seed)

    def matvec(P, V):
        PV = P @ V
        if noise_seed is None:
            return PV
        return PV + np.spacing(np.abs(PV)) * rng.integers(-3, 4, size=PV.shape)

    def policy_iteration(P, R, gamma, pi0):
        S = P.shape[0]
        pi, k = np.asarray(pi0, dtype=int).copy(), 0
        while k < 100:
            P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
            V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
            k += 1
            Q = R + gamma * matvec(P, V)
            tol = 1e-9 * (1 + np.abs(Q).max()) if tie_tol else 0.0
            pi_new = np.where(Q[np.arange(S), pi] >= Q.max(axis=1) - tol, pi, Q.argmax(axis=1))
            if np.array_equal(pi_new, pi):
                break
            pi = pi_new
        return V, pi, k

    return policy_iteration


@pytest.mark.parametrize("noise_seed", [None, *range(10)])
def test_policy_iteration_with_a_tolerance_passes_under_rounding_noise(noise_seed):
    m02.check_policy_iteration(make_policy_iteration(tie_tol=True, noise_seed=noise_seed))


@pytest.mark.parametrize("noise_seed", [None, 0, 1])
def test_policy_iteration_with_an_exact_tie_rule_fails_everywhere(noise_seed):
    with pytest.raises(CheckFailed, match="1e-13"):
        m02.check_policy_iteration(make_policy_iteration(tie_tol=False, noise_seed=noise_seed))
