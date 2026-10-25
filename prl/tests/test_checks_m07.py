"""prl.checks.m07: the exact checks against independent implementations, their guards, the
copied plumbing, and the training checkpoints' metric.

The checks' behavior on the lab's solutions and mutants is tested where those live:
scripts/check_notebooks.py --mode verify (exact checks) and scripts/threshold_protocol.py
(checkpoints 6.2 and 7.2). Here: the parts that do not need the lab. The implementations below
are written for the tests, differently from the lab's (tests may hold algorithms; prl may not).
"""

import ast
import math
import re
from pathlib import Path

import numpy as np
import pytest

from prl import envs
from prl.checks import CheckFailed, StochasticCheck, expected, m07

LAB = Path(__file__).resolve().parents[2] / "labs" / "src" / "07-function-approximation.py"


# ------------------------------------------------------------------ independent implementations
def scale_spec(n_products, capacity):
    states = math.prod([capacity + 1] * n_products)
    return {
        "n_states": states,
        "n_actions": states,
        "q_entries": states * states,
        "feature_dim": sum([capacity + 1] * n_products),
    }


def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):
    def dot(x, y):
        return sum(a * b for a, b in zip(x, y, strict=True))

    delta = r + gamma * dot(w, phi_next) - dot(w, phi)
    return np.array([wi + alpha * delta * fi for wi, fi in zip(w, phi, strict=True)])


def td_matrix(phi, mu, P, gamma):
    S, d = phi.shape
    A = np.zeros((d, d))
    for s in range(S):  # sum over states of mu(s) phi(s) (phi(s) - gamma E[phi(s')])^T
        A += mu[s] * np.outer(phi[s], phi[s] - gamma * sum(P[s, t] * phi[t] for t in range(S)))
    return A


def baird_expected_update(w, phi, mu, P, r, gamma, alpha):
    step = np.zeros_like(w)
    for s in range(len(mu)):
        v_next = sum(P[s, t] * (phi[t] @ w) for t in range(len(mu)))
        step += mu[s] * (r[s] + gamma * v_next - phi[s] @ w) * phi[s]
    return w + alpha * step


def remove_one_leg(env):
    S, P = env.n_states, env.P_target
    vals, vecs = np.linalg.eig(P.T)
    mu = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return {
        "on-policy": {"mu": mu / mu.sum()},
        "tabular": {"phi": np.eye(S)},
        "monte-carlo": {"gamma": 0, "r": np.linalg.inv(np.eye(S) - env.gamma * P) @ env.rewards},
    }


def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):
    target = r if terminated else r + gamma * max(row @ phi_next for row in W)
    out = np.array(W, dtype=float)
    out[a] += alpha * (target - W[a] @ phi) * phi
    return out


def unit_features(obs):
    z = np.asarray(obs, dtype=float) / m07.OBS_SCALE
    bumps = np.exp(-((m07.GRID_POINTS - z) ** 2).sum(axis=1) / (2 * m07.WIDTH**2))
    return bumps / math.sqrt(float(bumps @ bumps))


def tdc_expected_update(w, v, phi, mu, P, r, gamma, alpha, beta):
    D = np.diag(mu)
    e = phi.T @ D @ (r + gamma * P @ phi @ w - phi @ w)
    return (
        w + alpha * (e - gamma * phi.T @ P.T @ D @ phi @ v),
        v + beta * (e - phi.T @ D @ phi @ v),
    )


EXACT = [
    (m07.check_scale_spec, scale_spec),
    (m07.check_semi_gradient_td0, semi_gradient_td0),
    (m07.check_td_matrix, td_matrix),
    (m07.check_baird_expected_update, baird_expected_update),
    (m07.check_remove_one_leg, remove_one_leg),
    (m07.check_linear_q_update, linear_q_update),
    (m07.check_colleague_features, unit_features),
    (m07.check_tdc_expected_update, tdc_expected_update),
]


@pytest.mark.parametrize("check, fn", EXACT, ids=lambda x: getattr(x, "__name__", ""))
def test_exact_checks_accept_independent_implementations(check, fn):
    check(fn)


def _bumps(obs):
    z = np.asarray(obs, dtype=float) / m07.OBS_SCALE
    return np.exp(-((m07.GRID_POINTS - z) ** 2).sum(axis=1) / (2 * m07.WIDTH**2))


WRONG = [
    (m07.check_scale_spec, lambda n, c: {**scale_spec(n, c), "n_states": c**n}, "stock 0"),
    (m07.check_scale_spec, lambda n, c: [1, 2, 3, 4], "should return a dict"),
    (
        m07.check_scale_spec,
        lambda n, c: {**scale_spec(n, c), "q_entries": float(scale_spec(n, c)["n_states"])},
        "every action",
    ),
    (
        m07.check_semi_gradient_td0,
        lambda w, phi, r, pn, g, a: w + a * (r + g * w @ pn - w @ phi) * (phi - g * pn),
        "residual",
    ),
    (m07.check_td_matrix, lambda phi, mu, P, g: td_matrix(phi, mu, P, g).T, "Aᵀ"),
    (
        m07.check_baird_expected_update,
        lambda w, phi, mu, P, r, g, a: baird_expected_update(w, phi, np.ones_like(mu), P, r, g, a),
        "mu(s)",
    ),
    (
        m07.check_remove_one_leg,
        lambda env: {**remove_one_leg(env), "on-policy": {"mu": np.eye(env.n_states)[-1]}},
        "fits Baird but not",
    ),
    (
        m07.check_remove_one_leg,
        lambda env: {
            **remove_one_leg(env),
            "monte-carlo": {"gamma": 0.0, "r": np.array(env.rewards, dtype=float)},
        },
        "one-step rewards",
    ),
    (
        m07.check_remove_one_leg,
        lambda env: {**remove_one_leg(env), "extra": {}},
        "exactly the keys",
    ),
    (
        m07.check_linear_q_update,
        lambda W, phi, a, r, pn, term, trunc, g, al: linear_q_update(
            W, phi, a, r, pn, term or trunc, False, g, al
        ),
        "still bootstraps",
    ),
    (
        m07.check_linear_q_update,
        lambda W, phi, a, r, pn, term, trunc, g, al: linear_q_update(
            W, phi, a, r, pn, False, trunc, g, al
        ),
        "value is 0",
    ),
    (m07.check_colleague_features, _bumps, "alpha * ‖φ(s)‖²"),
    (m07.check_colleague_features, lambda o: _bumps(o) / _bumps(o).sum(), "divided by the sum"),
    (
        m07.check_colleague_features,
        lambda o: _bumps(o) / (_bumps(o) @ _bumps(o)),
        "point the right way",
    ),
    (
        m07.check_tdc_expected_update,
        lambda w, v, phi, mu, P, r, g, al, be: (
            w + al * phi.T @ (mu * (r + g * P @ phi @ w - phi @ w)),
            tdc_expected_update(w, v, phi, mu, P, r, g, al, be)[1],
        ),
        "semi-gradient TD again",
    ),
]


@pytest.mark.parametrize("check, bad, match", WRONG)
def test_exact_checks_reject_wrong_answers_and_name_the_mistake(check, bad, match):
    with pytest.raises(CheckFailed, match=re.escape(match)):
        check(bad)


@pytest.mark.parametrize(
    "check, fn",
    [
        (m07.check_semi_gradient_td0, semi_gradient_td0),
        (m07.check_baird_expected_update, baird_expected_update),
        (m07.check_linear_q_update, linear_q_update),
    ],
)
def test_updates_that_change_their_input_in_place_are_named(check, fn):
    def in_place(first, *rest):
        out = fn(first, *rest)
        first[...] = out
        return out

    with pytest.raises(CheckFailed, match="in place"):
        check(in_place)


def test_a_crash_in_the_participants_code_is_a_check_failure_not_chained():
    with pytest.raises(CheckFailed, match="should return shape"):
        m07.check_td_matrix(lambda phi, mu, P, g: phi.T @ P)  # (d, S), not (d, d)
    with pytest.raises(CheckFailed, match="ValueError") as info:
        m07.check_td_matrix(lambda phi, mu, P, g: phi @ P)
    assert info.value.__cause__ is None  # verify mode counts it as a rejection


def test_a_stub_raises_not_implemented_unchanged():
    def stub(*args):
        raise NotImplementedError("TODO 3")

    with pytest.raises(NotImplementedError):
        m07.check_td_matrix(stub)


def test_scale_spec_accepts_numpy_ints_and_integral_floats_only():
    m07.check_scale_spec(lambda n, c: {k: np.int64(v) for k, v in scale_spec(n, c).items()})
    m07.check_scale_spec(lambda n, c: {k: float(v) for k, v in scale_spec(n, c).items()})
    with pytest.raises(CheckFailed, match="feature_dim"):
        m07.check_scale_spec(lambda n, c: {**scale_spec(n, c), "feature_dim": 153.5})


# ------------------------------------------------------------------ fixtures
def test_fixtures_use_prl_envs_baird():
    b = envs.Baird()
    A = expected("m07_td_matrix")
    assert np.array_equal(A["baird_phi"], b.phi) and np.array_equal(A["baird_P"], b.P_target)
    assert np.array_equal(A["baird_mu"], b.mu_behavior) and float(A["baird_gamma"]) == b.gamma
    td0 = expected("m07_td0")
    assert np.array_equal(td0["w_0"], b.w0) and np.array_equal(td0["phi_next_0"], b.phi[6])
    legs = expected("m07_legs")
    assert np.array_equal(legs["baird_P"], b.P_target)
    assert np.array_equal(legs["baird_mu"], np.eye(7)[6])


def test_fixtures_hold_the_numbers_the_lab_and_page_quote():
    sweeps = expected("m07_expected_update")
    norms = [round(float(np.linalg.norm(sweeps[f"baird_{k}"])), 3) for k in (1, 10, 100, 1000)]
    assert norms == [10.366, 10.609, 16.826, 436.856]
    assert np.allclose(expected("m07_td0")["want_0"][[0, 7]], [1.1776, 1.0888])
    feats = expected("m07_features")
    assert np.allclose(np.linalg.norm(feats["features"], axis=1), 1.0)
    assert np.allclose(feats["features"][0], m07.rbf_features(np.zeros(4)), atol=1e-15)


# ------------------------------------------------------------------ the copied plumbing
def _top_level(path: Path) -> dict[str, str]:
    """Each top-level function (without its docstring) and each ALL_CAPS assignment, as AST."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                node.body = body[1:]
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign):
            names = [n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if names and all(n.isupper() for n in names):
                out[",".join(names)] = ast.dump(node)
    return out


@pytest.mark.skipif(not LAB.exists(), reason="the lab source is not next to prl")
def test_the_plumbing_equals_the_labs_provided_cells():
    lab, here = _top_level(LAB), _top_level(Path(m07.__file__))
    for name in (
        "OBS_SCALE",
        "CENTERS",
        "WIDTH",
        "GRID_POINTS",
        "ALPHA_Q,GAMMA_Q,EPS_END",
        "rbf_raw",
        "rbf_features",
        "train_linear_q",
        "greedy_return",
    ):
        assert lab[name] == here[name], f"{name} differs from the lab's provided ex6-run code"


def _lab_settings() -> tuple[dict, dict]:
    for node in ast.walk(ast.parse(LAB.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "settings"
        ):
            live, quick = (
                {kw.arg: ast.literal_eval(kw.value) for kw in arg.keywords} for arg in node.args
            )
            return live, quick
    raise AssertionError("no prl.runtime.settings(...) call in the lab")


@pytest.mark.skipif(not LAB.exists(), reason="the lab source is not next to prl")
def test_the_training_checks_use_the_labs_live_budget():
    live, quick = _lab_settings()
    for check in (m07.check_linear_q_training, m07.check_colleague_training):
        assert dict(check.live) == {k: live[k] for k in check.live}
        assert dict(check.quick) == dict(check.live)  # see the comment on LIVE
    assert live["eval_episodes"] == quick["eval_episodes"] == m07.EVAL_EPISODES


# ------------------------------------------------------------------ the training metric
SHORT = {"steps": 3000, "eps_steps": 1500, "log_every": 1000}


def test_the_metric_takes_all_randomness_from_the_seed():
    a = m07._linear_q_return(linear_q_update, seed=3, **SHORT)
    assert a == m07._linear_q_return(linear_q_update, seed=3, **SHORT)
    W, log = m07.train_linear_q(linear_q_update, m07.rbf_features, seed=3, **SHORT)
    assert log["step"] == [1000, 2000, 3000] and len(log["w_norm"]) == 3
    assert a == m07.greedy_return(W, m07.rbf_features, 3, m07.EVAL_EPISODES)


def test_a_diverging_run_fails_with_the_weight_norm_message_whatever_its_return():
    def explodes(W, phi, a, *rest):
        out = 3.0 * W
        out[a] += phi
        return out

    with pytest.raises(CheckFailed, match="‖W‖ reached") as info:
        m07._linear_q_return(explodes, seed=0, **SHORT)
    assert "above 10,000" in str(info.value) and "step-size rule" in str(info.value)


def test_a_threshold_failure_ends_with_the_checks_hint():
    with pytest.raises(CheckFailed) as info:  # never learns: greedy picks action 0
        m07.check_linear_q_training(lambda W, *rest: W.copy())
    msg = str(info.value)
    assert msg.startswith(m07.check_linear_q_training.what) and "returns W unchanged" in msg
    with pytest.raises(CheckFailed) as info:  # the colleague's unfixed features diverge
        m07.check_colleague_training(linear_q_update, _bumps)
    assert "‖W‖ reached" in str(info.value) and "Hint: with the fixed" not in str(info.value)


def test_a_correct_update_passes_checkpoint_6_2_at_its_seed():
    m07.check_linear_q_training(linear_q_update)


@pytest.mark.parametrize("name", ["check_linear_q_training", "check_colleague_training"])
def test_stochastic_checks_have_thresholds_for_both_budgets(name):
    check = getattr(m07, name)
    assert isinstance(check, StochasticCheck)
    assert set(check.thresholds) == {"live", "quick"}
    for threshold, provenance in check.thresholds.values():
        assert threshold > 3 * 22  # far above 3 x random play (median 22 in the band's record)
        assert provenance.startswith("threshold protocol, 100 seeds: runs/")
