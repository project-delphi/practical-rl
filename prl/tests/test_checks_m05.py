"""prl.checks.m05: the fixtures' contracts and the checks' guards.

The checks' behavior on the lab's solutions and mutants is tested where those live
(scripts/check_notebooks.py --mode verify). Here: the fixtures' values, the checks' guards
against None, NaN and wrong shapes (each must raise CheckFailed itself, because verify mode
counts nothing else), and correct code other than the reference. The ridge and sampler
code here is test code, not prl.
"""

import numpy as np
import pytest

from prl.checks import CheckFailed, expected, m05


def test_fixtures_hold_the_handoff_values():
    fx = expected("m05_reward")
    assert np.allclose(fx["hand_expected"][0], [0.575, 0.65, 0.5], atol=1e-15)
    assert np.allclose(fx["hand_expected"][1], [0.65, 0.2, 0.98], atol=1e-15)
    assert np.allclose(fx["hand_expected"][2], [0.98, 0.8, 0.02], atol=1e-15)
    assert fx["hand_no_clip"][1, 2] == pytest.approx(2.0)  # knowledge check 5.2's case
    fx = expected("m05_linucb")
    assert np.allclose(fx["unit_scores"][0] / 0.5, 1 / np.sqrt(1 + fx["unit_ns"]), atol=1e-15)
    fx = expected("m05_logging")
    assert np.allclose(fx["expected4"], [0.025, 0.925, 0.025, 0.025], atol=1e-15)
    assert np.allclose(fx["expected10"].sum(axis=1), 1.0, atol=1e-15)
    one_hot = fx["expected10"][0]  # epsilon = 0
    assert one_hot.max() == 1.0 and np.count_nonzero(one_hot) == 1
    fx = expected("m05_verdicts")
    assert fx["expected"].tolist() == [
        ["LinUCB", "tie"],
        ["greedy", "tie"],
        ["LinUCB", "tie"],
        ["single seed", "single seed"],
    ]


def test_reference_margins_are_wide():
    # The fixture builders assert these; here they are read back, so a regenerated fixture
    # that narrowed a margin fails a test as well as the builder.
    fx = expected("m05_lints")
    assert fx["ref_max_z"] < 3 < m05.LINTS_MEAN_SE and fx["ref_max_dev"] < 0.1 < m05.LINTS_COV_TOL
    fx = expected("m05_logging")
    assert abs(fx["ref_log_z"]) < 3 < m05.WEIGHT_SE
    assert abs(fx["ref_suspect_z"]) < 3 and abs(fx["ref_suspect_share_z"]) < 3 < m05.SHARE_SE


def _ridge_with(**override):
    class R:
        def __init__(self, d, lam=1.0):
            self.A, self.A_inv, self.b, self.n = lam * np.eye(d), np.eye(d) / lam, np.zeros(d), 0

        def update(self, x, r):
            self.A += np.outer(x, x)
            self.b += r * x
            self.A_inv = np.linalg.inv(self.A)
            self.n += 1

        def predict(self, x):
            return float(self.A_inv @ self.b @ x)

    for name, fn in override.items():
        setattr(R, name, fn)
    return R


@pytest.mark.parametrize(
    "check, bad",
    [
        (m05.check_reward_fn, lambda *a: None),
        (m05.check_reward_fn, lambda theta, x, s: np.full(len(theta), np.nan)),
        (m05.check_reward_fn, lambda theta, x, s: np.zeros((len(theta), 1))),
        (m05.check_ridge, _ridge_with(predict=lambda self, x: None)),
        (m05.check_ridge, _ridge_with(predict=lambda self, x: self.A_inv @ self.b * x)),
        (m05.check_linucb_scores, lambda models, x, c: [m.predict(x) for m in models[:-1]]),
        (m05.check_linucb_scores, lambda models, x, c: np.full(len(models), np.inf)),
        (m05.check_lints_sample, lambda model, v, rng: None),
        (m05.check_lints_sample, lambda model, v, rng: rng.standard_normal(2)),
        (m05.check_verdicts, lambda regrets, budget: None),
        (m05.check_verdicts, lambda regrets, budget: [("clear", "LinUCB")]),
        (m05.check_verdicts, lambda regrets, budget: {}),
        (m05.check_log_policy, lambda scores, eps: None),
        (m05.check_log_policy, lambda scores, eps: np.full(len(scores), -1.0)),
        (m05.check_log_suspect, lambda env, policy, n, seed: None),
        (m05.check_log_suspect, lambda env, policy, n, seed: (1, 2, 3)),
    ],
)
def test_exact_checks_reject_wrong_answers(check, bad):
    with pytest.raises(CheckFailed):
        check(bad)


def test_ridge_check_rejects_a_singular_inverse_without_a_linalg_error():
    # sm-no-one: dividing by x^T A^-1 x alone makes A_inv singular after the first update.
    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += r * x
        Ax = self.A_inv @ x
        self.A_inv -= np.outer(Ax, Ax) / (x @ Ax)
        self.n += 1

    with pytest.raises(CheckFailed, match="not the inverse"):
        m05.check_ridge(_ridge_with(update=update))
    m05.check_ridge(_ridge_with())  # an inverse from np.linalg.inv is accepted


def test_log_policy_check_catches_an_unnormalized_policy_before_env_log():
    def unnormalized(scores, eps):
        probs = np.full(len(scores), eps / len(scores))
        probs[int(np.argmax(scores))] = 1.0
        return probs

    with pytest.raises(CheckFailed, match="sums to"):
        m05.check_log_policy(unnormalized)


def test_any_correct_sampler_passes_the_lints_check():
    # The checkpoint tests moments, not values: draws in another order still pass.
    def multivariate(model, v, rng):
        return rng.multivariate_normal(model.A_inv @ model.b, v**2 * model.A_inv)

    m05.check_lints_sample(multivariate)
