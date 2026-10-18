"""Plausible wrong versions of Module 5's functions (first draft, academic director).

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires
every exact checkpoint that calls the function to reject it. All seven of Module 5's
checkpoints are exact, so no mutant here needs the threshold protocol and none is marked
`random`. Each mutant is a mistake a participant could plausibly make; `why` says which.
Names the notebook defines (`argmax_first`, `compare`, `envs`, `textwrap`) are resolved in
the notebook's namespace when a mutant is swapped in.
"""

import numpy as np

from prl.lab import mutant


# ------------------------------------------------------------------ 1 reward_fn
@mutant(ex=1, replaces="reward_fn", id="no-clip", why="forgets the clip to [0.02, 0.98]")
def reward_fn(theta, x, scale):
    return 0.5 + scale * theta @ x


@mutant(ex=1, replaces="reward_fn", id="no-intercept", why="forgets the 0.5")
def reward_fn(theta, x, scale):  # noqa: F811
    return np.clip(scale * theta @ x, 0.02, 0.98)


@mutant(ex=1, replaces="reward_fn", id="no-scale", why="forgets the scale s")
def reward_fn(theta, x, scale):  # noqa: F811
    return np.clip(0.5 + theta @ x, 0.02, 0.98)


# ------------------------------------------------------------------ 2 Ridge
@mutant(ex=2, replaces="Ridge", id="b-without-reward", why="adds x to b, not r x")
class Ridge:
    def __init__(self, d, lam=1.0):
        self.A = lam * np.eye(d)
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)
        self.n = 0

    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += x
        Ax = self.A_inv @ x
        self.A_inv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.n += 1

    def predict(self, x):
        return float(self.A_inv @ self.b @ x)


@mutant(ex=2, replaces="Ridge", id="sm-sign", why="adds the Sherman-Morrison term")
class Ridge:  # noqa: F811
    def __init__(self, d, lam=1.0):
        self.A = lam * np.eye(d)
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)
        self.n = 0

    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += r * x
        Ax = self.A_inv @ x
        self.A_inv += np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.n += 1

    def predict(self, x):
        return float(self.A_inv @ self.b @ x)


@mutant(ex=2, replaces="Ridge", id="sm-no-one", why="drops the 1 in 1 + x^T A^-1 x")
class Ridge:  # noqa: F811
    def __init__(self, d, lam=1.0):
        self.A = lam * np.eye(d)
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)
        self.n = 0

    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += r * x
        Ax = self.A_inv @ x
        self.A_inv -= np.outer(Ax, Ax) / (x @ Ax)
        self.n += 1

    def predict(self, x):
        return float(self.A_inv @ self.b @ x)


@mutant(ex=2, replaces="Ridge", id="stale-inverse", why="updates A and b but never A_inv")
class Ridge:  # noqa: F811
    def __init__(self, d, lam=1.0):
        self.A = lam * np.eye(d)
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)
        self.n = 0

    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += r * x
        self.n += 1

    def predict(self, x):
        return float(self.A_inv @ self.b @ x)


@mutant(ex=2, replaces="Ridge", id="predict-with-A", why="multiplies b by A, not A^-1")
class Ridge:  # noqa: F811
    def __init__(self, d, lam=1.0):
        self.A = lam * np.eye(d)
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)
        self.n = 0

    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += r * x
        Ax = self.A_inv @ x
        self.A_inv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.n += 1

    def predict(self, x):
        return float(self.A @ self.b @ x)


# ------------------------------------------------------------------ 3 linucb_scores
@mutant(ex=3, replaces="linucb_scores", id="no-sqrt", why="adds c x^T A^-1 x, a variance")
def linucb_scores(models, x, c):
    means = np.array([m.predict(x) for m in models])
    widths = np.array([x @ m.A_inv @ x for m in models])
    return means + c * widths


@mutant(ex=3, replaces="linucb_scores", id="A-not-inverse", why="uses A where A^-1 belongs")
def linucb_scores(models, x, c):  # noqa: F811
    means = np.array([m.predict(x) for m in models])
    widths = np.array([np.sqrt(x @ m.A @ x) for m in models])
    return means + c * widths


@mutant(ex=3, replaces="linucb_scores", id="width-only", why="returns the bonus alone")
def linucb_scores(models, x, c):  # noqa: F811
    return c * np.array([np.sqrt(x @ m.A_inv @ x) for m in models])


@mutant(ex=3, replaces="linucb_scores", id="c-inside-sqrt", why="takes sqrt(c x^T A^-1 x)")
def linucb_scores(models, x, c):  # noqa: F811
    means = np.array([m.predict(x) for m in models])
    return means + np.array([np.sqrt(c * x @ m.A_inv @ x) for m in models])


# ------------------------------------------------------------------ 4 lints_sample
@mutant(ex=4, replaces="lints_sample", id="returns-mean", why="returns theta_hat: greedy")
def lints_sample(model, v, rng):
    return model.A_inv @ model.b


@mutant(ex=4, replaces="lints_sample", id="sqrt-v", why="scales by sqrt(v), covariance v A^-1")
def lints_sample(model, v, rng):  # noqa: F811
    theta_hat = model.A_inv @ model.b
    L = np.linalg.cholesky(model.A_inv)
    return theta_hat + np.sqrt(v) * L @ rng.standard_normal(len(theta_hat))


@mutant(ex=4, replaces="lints_sample", id="A-not-inverse", why="factors A, not A^-1")
def lints_sample(model, v, rng):  # noqa: F811
    theta_hat = model.A_inv @ model.b
    L = np.linalg.cholesky(model.A)
    return theta_hat + v * L @ rng.standard_normal(len(theta_hat))


@mutant(ex=4, replaces="lints_sample", id="isotropic", why="ignores A^-1: covariance v^2 I")
def lints_sample(model, v, rng):  # noqa: F811
    theta_hat = model.A_inv @ model.b
    return theta_hat + v * rng.standard_normal(len(theta_hat))


@mutant(ex=4, replaces="lints_sample", id="own-rng", why="ignores rng: the same draw every call")
def lints_sample(model, v, rng):  # noqa: F811
    theta_hat = model.A_inv @ model.b
    L = np.linalg.cholesky(model.A_inv)
    z = np.random.default_rng(0).standard_normal(len(theta_hat))
    return theta_hat + v * L @ z


# ------------------------------------------------------------------ 5 verdicts
@mutant(ex=5, replaces="verdicts", id="higher-is-better", why="treats regret as a reward")
def verdicts(regrets, budget):
    out = {}
    for regime, by_method in regrets.items():
        for method in ("LinUCB", "LinTS"):
            v = compare(
                by_method[method], by_method["greedy"], names=(method, "greedy"), budget=budget
            )
            out[regime, method] = {"A": method, "B": "greedy"}.get(v.winner, v.winner)
    return out


@mutant(ex=5, replaces="verdicts", id="lower-mean-wins", why="no intervals: the lower mean wins")
def verdicts(regrets, budget):  # noqa: F811
    out = {}
    for regime, by_method in regrets.items():
        for method in ("LinUCB", "LinTS"):
            better = np.mean(by_method[method]) < np.mean(by_method["greedy"])
            out[regime, method] = method if better else "greedy"
    return out


@mutant(ex=5, replaces="verdicts", id="always-tie", why="calls every comparison a tie")
def verdicts(regrets, budget):  # noqa: F811
    return {(regime, m): "tie" for regime in regrets for m in ("LinUCB", "LinTS")}


# ------------------------------------------------------------------ 6 log_policy
@mutant(ex=6, replaces="log_policy", id="no-exploration", why="ignores epsilon: greedy logging")
def log_policy(scores, epsilon):
    probs = np.zeros(len(scores))
    probs[argmax_first(scores)] = 1.0
    return probs


@mutant(ex=6, replaces="log_policy", id="epsilon-on-others", why="eps/(K-1) on the others only")
def log_policy(scores, epsilon):  # noqa: F811
    n = len(scores)
    probs = np.full(n, epsilon / (n - 1))
    probs[argmax_first(scores)] = 1.0 - epsilon
    return probs


@mutant(ex=6, replaces="log_policy", id="unnormalized", why="sets the best to 1: sums to 1 + eps")
def log_policy(scores, epsilon):  # noqa: F811
    n = len(scores)
    probs = np.full(n, epsilon / n)
    probs[argmax_first(scores)] = 1.0
    return probs


# ------------------------------------------------------------------ 7 log_suspect
# The stub itself is the planted bug (it records probs.max()); verify mode runs it as "stub".
@mutant(ex=7, replaces="log_suspect", id="uniform-propensity", why="records 1/K on every row")
def log_suspect(env, policy, n, seed):
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _ in range(n):
        x = env.context()
        probs = policy(x, rng)
        a = int(rng.choice(len(probs), p=probs))
        r = env.step(a)
        rows.append((x, a, r, 1.0 / len(probs), probs))
    X, A, R, P, probs = (np.array(k) for k in zip(*rows, strict=True))
    return envs.LoggedBandit(X=X, A=A, R=R, P=P, probs=probs, T=np.arange(n))


@mutant(ex=7, replaces="log_suspect", id="shows-argmax", why="shows the best item, never samples")
def log_suspect(env, policy, n, seed):  # noqa: F811
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _ in range(n):
        x = env.context()
        probs = policy(x, rng)
        a = int(np.argmax(probs))
        r = env.step(a)
        rows.append((x, a, r, float(probs[a]), probs))
    X, A, R, P, probs = (np.array(k) for k in zip(*rows, strict=True))
    return envs.LoggedBandit(X=X, A=A, R=R, P=P, probs=probs, T=np.arange(n))


@mutant(ex=7, replaces="log_suspect", id="propensity-one", why="records 1 for every shown item")
def log_suspect(env, policy, n, seed):  # noqa: F811
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _ in range(n):
        x = env.context()
        probs = policy(x, rng)
        a = int(rng.choice(len(probs), p=probs))
        r = env.step(a)
        rows.append((x, a, r, 1.0, probs))
    X, A, R, P, probs = (np.array(k) for k in zip(*rows, strict=True))
    return envs.LoggedBandit(X=X, A=A, R=R, P=P, probs=probs, T=np.arange(n))
