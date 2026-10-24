# ---
# jupyter:
#   prl:
#     module: m05
#     pins:
#       mabwiser: "2.7.4"
# ---

# %% [markdown]
# In this lab you build a recommender that learns which of 10 items each user is likely to
# click. You frame the problem, write incremental ridge regression, and turn it into two
# exploring methods: LinUCB, which adds a confidence width to each prediction, and linear
# Thompson sampling, which draws a plausible parameter instead. You compare both with a
# click model that never explores, in three regimes: one where the linear model is right,
# one where it is wrong, and one where users' tastes drift; then once more after a longer
# launch. Then you log data with the propensities that Day 5 needs, and find a bug in a
# colleague's logger.
#
# The briefing's equations are named in brackets, for example (ridge) or (LinUCB). Its table
# "Equations and where you implement them" gives each name's equation number.

# %%
import math
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np

from prl import envs
from prl.checks import m05
from prl.evaluate import Budget, bootstrap_ci, compare, iqm
from prl.plot import NEUTRAL_LIGHT

K, D_RAW = 10, 5  # items, and the dimension of a user's context
D = D_RAW + 1  # the briefing's d: the context and a constant 1 (features below)
CATALOG = 0  # the catalog: every item's parameters are fixed by this seed
LAM = 1.0  # lambda_ridge, the ridge penalty
C_UCB = 0.5  # LinUCB's width multiplier c
V_TS = 0.25  # linear Thompson sampling's scale v
EPSILON = 0.1  # the logging policy's exploration rate

# Budgets. Lab code reads them from here and never branches on QUICK.
budget = lab.use_settings(
    prl.runtime.settings(
        dict(T=3000, n_log=5000),  # live
        dict(T=1000, n_log=2000),  # QUICK
        designed="colab-cpu",
    )
)
# Your seat's seeds start here, clear of the briefing's, the figures' and the checks' seeds
# (0-99), so every seat in the room adds new evidence.
RUN_SEED = (SEED + 1) * 1000
print(f"Budget: {budget.T} rounds per run; {budget.n_log} rounds per log.")
print(f"K = {K} items; contexts of dimension {D_RAW}, features d = {D}")
print(f"c = {C_UCB} (LinUCB), v = {V_TS} (LinTS), epsilon = {EPSILON} (logging)")


# %%
def features(x):
    """Provided. phi(x) = (x, 1): the context with a constant feature, so a linear model
    can hold the 0.5 every item's click probability starts from (click model)."""
    return np.append(x, 1.0)


def argmax_first(values):
    """Provided. The first index within rounding of the max (1e-9 of the values), so
    rounding never decides between two scores that are equal on paper."""
    v = np.asarray(values, dtype=float)
    tol = 1e-9 * (1.0 + np.abs(v).max())
    return int(np.flatnonzero(v >= v.max() - tol)[0])


# %% [markdown] role="ex0-head"
# Welcome back. The briefing ended at logging: a stochastic behavior policy and the
# propensity of every item it shows. The lab follows the briefing in order, so you start with
# section 1's framing. If the runtime restarted over lunch, rerun the setup cell at the top
# and the two cells above, then run the cell below. It rebuilds the recommender from
# `prl.envs`, so this notebook depends on no other notebook.

# %% role="ex0-run"
env = envs.RecSim(n_items=K, d=D_RAW, seed=CATALOG)
env.reset(seed=RUN_SEED)
x = env.context()
print(f"RecSim: {env.n_items} items, contexts of dimension {env.d}, scale {env.scale}")
print("one user's context x:", np.round(x, 3))
print("click probability of each item for this user:")
print("  ", np.round(env.expected_reward(x), 3))
print("show item 0 ->", "click" if env.step(0) else "no click")

# %% [markdown] role="ex1-head"
# **The click model.** A user arrives with a context x. You show one item a; the user clicks
# with probability r(x, a) and you see the click, 0 or 1, for that item only. In RecSim's
# well-specified regime, r(x, a) = clip(0.5 + 0.15 θ_aᵀ x, 0.02, 0.98), where θ_a is item a's
# hidden parameter and 0.15 is `env.scale` (click model). Fill in the framing card below,
# then write the click model for every item at once.

# %% [markdown]
# **Your framing card.** Double-click this cell to edit it.
#
# - Context (what you see about the user): …
# - Action: …
# - Reward, and when you see it: …
# - Next context: does it depend on what you showed? …
# - One thing a real recommender has that this frame leaves out: …

# %% [markdown] role="ex1-predict"
# Two fixed policies serve the same 1000 users: one always shows item 0, the other always
# item 9. About how many clicks will each get: about 150, about 500, or about 850? What would
# Module 4's bandit, which learns one mean per item and ignores the user, conclude about
# which item to show?


# %% role="ex1-stub"
def reward_fn(theta, x, scale):
    """Click probability of every item: clip(0.5 + scale theta_a^T x, 0.02, 0.98).

    theta has shape (K, 5), RecSim's `env.theta`; x is a raw context of shape (5,), without
    the constant 1. Returns shape (K,).
    """
    # TODO 1: one line with theta @ x and np.clip.
    raise NotImplementedError(
        "TODO 1: write reward_fn, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# `theta @ x` has one entry per item. Scale it, add 0.5, and clip each entry to the interval
# [0.02, 0.98].


# %% role="ex1-sol"
@lab.solution(1)
def reward_fn(theta, x, scale):
    """r(x, a) for every item a: the click model, clipped."""
    return np.clip(0.5 + scale * theta @ x, 0.02, 0.98)


# %% role="ex1-run"
rng = np.random.default_rng(RUN_SEED)
X = np.array([env.sample_context(rng) for _ in range(1000)])
ours = np.array([reward_fn(env.theta, x, env.scale) for x in X])
sim = np.array([env.expected_reward(x) for x in X])
print(f"max |reward_fn - simulator| over 1000 users: {np.abs(ours - sim).max():.1e}")
print("each item's click probability, averaged over these users:")
print("  ", np.round(sim.mean(axis=0), 3))
print("how often each item is the best one, over these users:")
print("  ", np.bincount(sim.argmax(axis=1), minlength=K))


def serve(item, n, seed):
    """Provided. Show the same item to n users of the stream `seed`; return users, clicks."""
    env.reset(seed=seed)
    users, clicks = [], 0
    for _ in range(n):
        users.append(env.context())
        clicks += int(env.step(item))
    return np.array(users), clicks


users_0, clicks_0 = serve(0, 1000, RUN_SEED)
users_9, clicks_9 = serve(9, 1000, RUN_SEED)
print("the same 1000 users under both policies:", np.array_equal(users_0, users_9))
print(f"clicks: always item 0 -> {clicks_0}; always item 9 -> {clicks_9}")

# %% [markdown] role="ex1-explain"
# The users were identical under both policies: the item you show never changes who comes
# next, so there is no transition to model (click model). The two items earned about the same
# clicks, yet each is the best item for a different share of users (the counts above): one
# mean per item cannot tell them apart. Now rewrite your card's "Next context" line for a
# recommender in which a user who sees the same item twice clicks less the second time. What
# must the state hold, and what does the transition do with the item you showed? (Gotcha:
# "Every recommender is a contextual bandit.")

# %% role="ex1-chk"
lab.check(1, m05.check_reward_fn, reward_fn)

# %% [markdown] role="ex2-head"
# **Incremental ridge regression, one model per item.** Each item keeps two statistics of the
# features φ = φ(x) it has seen: A = λ_ridge I + Σ φφᵀ, the briefing's Λ (Li et al. call it A),
# and b = Σ r φ. It estimates ŵ = A⁻¹ b (ridge). Updating A and b costs O(d²) per
# observation. Inverting A from scratch would cost O(d³), so keep A⁻¹ current too, with the
# Sherman–Morrison formula A⁻¹ ← A⁻¹ − (A⁻¹φ)(A⁻¹φ)ᵀ / (1 + φᵀA⁻¹φ) (Sherman–Morrison).
# The constructor is written; you write `update` and `predict`.

# %% [markdown] role="ex2-predict"
# λ_ridge = 1. A Ridge model sees the same unit vector φ nine times, with a click every time.
# What does it predict at φ: 1, 0.9, or 0.5?


# %% role="ex2-stub"
class Ridge:
    """Ridge regression for one item, updated one observation at a time.

    A = lam I + sum phi phi^T, b = sum r phi, A_inv = A^-1 (Sherman-Morrison), w_hat = A_inv b.
    """

    def __init__(self, d, lam=1.0):
        self.A = lam * np.eye(d)  # the briefing's Lambda
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)
        self.n = 0  # observations so far

    def update(self, phi, r):
        """Add one observation: A += phi phi^T, b += r phi, A_inv by Sherman-Morrison, n += 1."""
        # TODO 2 (part 1): four lines. Compute A_inv @ phi once and reuse it.
        raise NotImplementedError(
            "TODO 2: write Ridge.update, or run lab.use_reference(2) to continue with the solution"
        )

    def predict(self, phi):
        """w_hat^T phi, as a float, with w_hat = A_inv @ b."""
        # TODO 2 (part 2): one line.
        raise NotImplementedError(
            "TODO 2: write Ridge.predict, or run lab.use_reference(2) to continue with the solution"
        )


# %% [markdown] role="ex2-hint"
# `np.outer(phi, phi)` is φφᵀ. Compute `Ap = self.A_inv @ phi` once. A⁻¹ is symmetric, so
# both the outer product and the denominator in (Sherman–Morrison) can be built from it.
# Update `A_inv` from its old value, before or after `A`; it never reads `A`. `predict`
# returns a float, not an array.


# %% role="ex2-sol"
@lab.solution(2)
class Ridge:
    """Ridge regression for one item, one observation at a time."""

    def __init__(self, d, lam=1.0):
        self.A = lam * np.eye(d)  # Lambda = lam I + sum phi phi^T
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)  # sum r phi
        self.n = 0

    def update(self, phi, r):
        """Add (phi, r) to A and b; keep A_inv current."""
        self.A += np.outer(phi, phi)
        self.b += r * phi
        Ap = self.A_inv @ phi  # (Sherman-Morrison)
        self.A_inv -= np.outer(Ap, Ap) / (1.0 + phi @ Ap)
        self.n += 1

    def predict(self, phi):
        """w_hat^T phi with w_hat = A_inv b (ridge)."""
        return float(self.A_inv @ self.b @ phi)


# %% role="ex2-run"
env.reset(seed=RUN_SEED)
item0, Phi, clicks = Ridge(D, LAM), [], []
print("   n   max abs(w_hat - closed form)   max abs(A_inv - inv(A))")
for n in range(1, 1001):
    phi = features(env.context())
    click = env.step(0)
    item0.update(phi, click)
    Phi.append(phi)
    clicks.append(click)
    if n in (10, 100, 1000):
        P_, y_ = np.array(Phi), np.array(clicks)
        closed = np.linalg.solve(LAM * np.eye(D) + P_.T @ P_, P_.T @ y_)
        gap = np.abs(item0.A_inv @ item0.b - closed).max()
        drift_inv = np.abs(item0.A_inv - np.linalg.inv(item0.A)).max()
        print(f"{n:4d}   {gap:28.1e}   {drift_inv:23.1e}")
truth = np.append(env.scale * env.theta[0], 0.5)  # r(x, 0) = truth . phi(x), inside the clip
print("item 0, w_hat after 1000 users:     ", np.round(item0.A_inv @ item0.b, 3))
print("item 0, w*_0 = (0.15 theta_0, 0.5): ", np.round(truth, 3))
unit = np.zeros(D)
unit[-1] = 1.0  # a unit vector: the constant feature alone
nine = Ridge(D, LAM)
for _ in range(9):
    nine.update(unit, 1.0)
print(f"unit phi seen 9 times, clicked every time: predict(phi) = {nine.predict(unit):.4f}")

# %% [markdown] role="ex2-explain"
# The incremental ŵ equals the closed form to rounding, and so does A⁻¹: Sherman–Morrison
# is exact algebra, not an approximation (Sherman–Morrison). After nine clicks at φ the
# prediction is not 1. Where did the missing tenth go? Read λ_ridge as a count: how many
# observations of what reward does the penalty λ_ridge I add in every direction?

# %% role="ex2-chk"
lab.check(2, m05.check_ridge, Ridge)

# %% [markdown] role="ex3-head"
# **LinUCB.** Score every item by its prediction plus c times its width, and show the best
# score: ŵ_aᵀφ + c √(φᵀA_a⁻¹φ) (LinUCB). The width √(φᵀA_a⁻¹φ) is large in directions item a
# has seen little of, and shrinks as data arrive (width). This is Li et al.'s Algorithm 1,
# with their α written c. The cell after your function runs LinUCB and predict-then-greedy
# on the same stream of users. Both start with a launch: each item gets one impression (one
# showing), in order. Exercise 5 also tries a longer launch.

# %% [markdown] role="ex3-predict"
# λ_ridge = 1. An item has seen the same unit vector φ 100 times. What is its width at φ,
# √(φᵀA⁻¹φ)? And LinUCB's bonus there, c times the width, at c = 0.5? (The run cell prints
# the width for n = 0, 1, 10 and 100.)


# %% role="ex3-stub"
def linucb_scores(models, phi, c):
    """LinUCB score of every item: models[a].predict(phi) + c sqrt(phi^T A_a^-1 phi). Shape (K,)."""
    # TODO 3: one prediction and one width per model; use m.A_inv.
    raise NotImplementedError(
        "TODO 3: write linucb_scores, or run lab.use_reference(3) to continue with the solution"
    )


# %% [markdown] role="ex3-hint"
# Build two arrays with one entry per model: the predictions, and the widths from each
# model's `A_inv`. The width is a standard deviation, so it has a square root, and it uses
# A⁻¹, not A.


# %% role="ex3-sol"
@lab.solution(3)
def linucb_scores(models, phi, c):
    """w_hat_a^T phi + c sqrt(phi^T A_a^-1 phi) for every item a."""
    means = np.array([m.predict(phi) for m in models])
    widths = np.array([np.sqrt(phi @ m.A_inv @ phi) for m in models])
    return means + c * widths  # (LinUCB)


# %% role="ex3-run"
unit = np.zeros(D)
unit[-1] = 1.0  # a unit vector: the constant feature alone
print("width at a unit phi seen n times (no clicks, so the score is c times the width):")
for n in (0, 1, 10, 100):
    m = Ridge(D, LAM)
    for _ in range(n):
        m.update(unit, 0.0)
    score = linucb_scores([m], unit, C_UCB)[0]
    print(f"  n = {n:3d}: width {score / C_UCB:.4f}   1/sqrt(1 + n) = {1 / math.sqrt(1 + n):.4f}")


def run(env, choose, T, seed, passes=1):
    """Provided. T rounds; returns (cumulative pseudo-regret, models).

    Rounds 0..passes * K - 1 are the launch: item t % K is shown at round t, as if the whole
    catalog were new. Then choose(models, phi, rng) picks the item. The regret uses the
    simulator's known click probabilities, which no method sees (regret).
    """
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)  # the method's own coin (LinTS uses it)
    models = [Ridge(D, LAM) for _ in range(K)]
    regret = np.empty(T)
    for t in range(T):
        x = env.context()
        r_true = env.expected_reward(x)  # for the regret only
        phi = features(x)
        a = t % K if t < passes * K else choose(models, phi, rng)
        click = env.step(a)
        models[a].update(phi, click)
        regret[t] = r_true.max() - r_true[a]
    return np.cumsum(regret), models


def greedy_choice(models, phi, rng):
    """Provided. Predict-then-greedy: the item with the highest predicted click rate."""
    return argmax_first([m.predict(phi) for m in models])


def linucb_choice(models, phi, rng):
    """Provided. LinUCB: the item with the highest score."""
    return argmax_first(linucb_scores(models, phi, C_UCB))


lab.seeds([RUN_SEED])
reg_greedy, models_greedy = run(env, greedy_choice, budget.T, RUN_SEED)
reg_ucb, models_ucb = run(env, linucb_choice, budget.T, RUN_SEED)
print(f"Seed {RUN_SEED}, {budget.T} rounds, well-specified regime. Regret at the end:")
print(f"  predict-then-greedy {reg_greedy[-1]:6.1f}   LinUCB {reg_ucb[-1]:6.1f}")
print("Single seed: a fact to put beside the room's seeds, not a ranking.")
print("rounds each item was shown, greedy:", [m.n for m in models_greedy])
print("rounds each item was shown, LinUCB:", [m.n for m in models_ucb])
fig, ax = plt.subplots()
ax.plot(reg_greedy, ls="--", label="predict-then-greedy")
ax.plot(reg_ucb, label=f"LinUCB, c = {C_UCB}")
ax.set(xlabel="round", ylabel="cumulative pseudo-regret", title=f"One seed ({RUN_SEED})")
ax.legend()
plt.show()

# %% [markdown] role="ex3-explain"
# Check your Predict against the printed widths: for large n, each tenfold increase in n
# shrinks the width by about √10 (width). Now read the two "rounds shown" rows. Which items did
# predict-then-greedy stop showing after the launch, and what was their prediction after
# one no-click? Why can greedy never find out it was wrong about them, while LinUCB's width
# keeps an unshown item's score up? You met this trap in Module 4 with one mean per
# headline. What does a model with features change, and what does it not? (Gotcha: "A
# better click model is a better recommender.")

# %% role="ex3-chk"
lab.check(3, m05.check_linucb_scores, linucb_scores)

# %% [markdown] role="ex4-head"
# **Linear Thompson sampling.** Instead of adding a width, draw a plausible parameter for
# each item, w̃_a ~ N(ŵ_a, v² A_a⁻¹), and show the item whose drawn score w̃_aᵀφ is highest
# (LinTS). This is Agrawal and Goyal's algorithm, one model per item. Draw with the
# Cholesky factor L of A⁻¹ (L Lᵀ = A⁻¹): w̃ = ŵ + v L z with z standard normal. Use the
# `rng` you are given, so every run can be repeated.

# %% [markdown] role="ex4-predict"
# With c = 0.5 and v = 0.25, an item's drawn score w̃ᵀφ is normal with mean ŵᵀφ and
# standard deviation v √(φᵀA⁻¹φ). How often is it above the same item's LinUCB score,
# ŵᵀφ + c √(φᵀA⁻¹φ)? About 1 draw in 3, 1 in 44, or 1 in 30,000?


# %% role="ex4-stub"
def lints_sample(model, v, rng):
    """One draw of w from N(w_hat, v^2 A^-1) for one item's model. Shape (d,)."""
    # TODO 4: w_hat, a Cholesky factor of A_inv, and one standard-normal z drawn from rng.
    raise NotImplementedError(
        "TODO 4: write lints_sample, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# `z = rng.standard_normal(len(w_hat))`. If L Lᵀ = A⁻¹, then v L z has covariance
# v² L Lᵀ = v² A⁻¹. Do not use `rng.multivariate_normal`: it factors the covariance with an
# SVD, whose signs can differ between machines, so the same seed can give different draws.


# %% role="ex4-sol"
@lab.solution(4)
def lints_sample(model, v, rng):
    """w ~ N(w_hat, v^2 A^-1), drawn with a Cholesky factor."""
    w_hat = model.A_inv @ model.b
    L = np.linalg.cholesky(model.A_inv)  # L L^T = A^-1
    z = rng.standard_normal(len(w_hat))
    return w_hat + v * L @ z  # (LinTS)


# %% role="ex4-run"
def lints_choice(models, phi, rng):
    """Provided. LinTS: one draw per item, then the highest drawn score."""
    return argmax_first([lints_sample(m, V_TS, rng) @ phi for m in models])


reg_ts, models_ts = run(env, lints_choice, budget.T, RUN_SEED)
print(f"Seed {RUN_SEED}, {budget.T} rounds. Regret at the end (single seed, not a ranking):")
print(f"  greedy {reg_greedy[-1]:6.1f}   LinUCB {reg_ucb[-1]:6.1f}   LinTS {reg_ts[-1]:6.1f}")
print("rounds each item was shown, LinTS:", [m.n for m in models_ts])
# The Predict: how often a draw beats LinUCB's score, for one item and one user.
probe = features(env.sample_context(np.random.default_rng(RUN_SEED + 7)))
m = models_ts[int(np.argmin([mm.n for mm in models_ts]))]  # the least-shown item
ucb = linucb_scores([m], probe, C_UCB)[0]
draws_rng = np.random.default_rng(RUN_SEED + 8)
draws = np.array([lints_sample(m, V_TS, draws_rng) @ probe for _ in range(20000)])
above = int(np.sum(draws > ucb))
p_theory = 0.5 * math.erfc(C_UCB / V_TS / math.sqrt(2))  # P(Z > c / v)
print(f"draws above LinUCB's score: {above} of {len(draws)}; P(Z > c/v) = {p_theory:.4f}")

# %% [markdown] role="ex4-explain"
# A drawn score sits about c/v = 2 standard deviations below LinUCB's optimistic score, and
# beats it in about 1 draw in 44 (LinTS). So LinTS explores by chance rather than by a
# bonus: an item it is unsure about wins when its draw happens to be high. Both depend on
# the same width √(φᵀA⁻¹φ). Which of the two methods can show different items to two
# identical users in a row?

# %% role="ex4-chk"
lab.check(4, m05.check_lints_sample, lints_sample)

# %% [markdown] role="ex5-head"
# **Three regimes, five seeds, and a longer launch.** The cell below runs predict-then-greedy,
# LinUCB and LinTS for the lab's budget, with the same catalog, in four conditions.
# **Well-specified:** the click model of Exercise 1. **Misspecified:** a quadratic term that
# no linear model can represent (misspec = 1). **Drifting:** every θ_a turns half a circle
# over the run, so tastes at the end are the reverse of those at the start (drift). **Warm
# launch:** the well-specified regime again, after a launch of three passes (30 rounds, 1% of
# the run) instead of one. Run it now: it takes about a minute (an estimate) and prints a
# line as each method finishes, but no results. Read the Predict while it runs. Then you write `verdicts`:
# for each condition, LinUCB against greedy and LinTS against greedy, with
# `prl.evaluate.compare` and the workshop's rule. Lower regret is better. Every verdict
# states its budget.

# %%
REGIMES = {
    "well-specified": dict(),
    "misspecified": dict(misspec=1.0),
    "drifting": dict(drift=math.pi / budget.T),  # half a turn over the run
    "warm launch": dict(),  # well-specified, after a longer launch
}
PASSES = {"warm launch": 3}  # launch passes per condition; the others have 1
METHODS = {"greedy": greedy_choice, "LinUCB": linucb_choice, "LinTS": lints_choice}
seeds5 = [RUN_SEED + 100 + i for i in range(5)]
lab.seeds(seeds5)
regrets, curves, widths_end = {}, {}, {}
started = time.perf_counter()
for regime, kw in REGIMES.items():
    r_env = envs.RecSim(n_items=K, d=D_RAW, seed=CATALOG, **kw)
    passes = PASSES.get(regime, 1)
    regrets[regime], curves[regime] = {}, {}
    for method, choose in METHODS.items():
        t0 = time.perf_counter()
        runs = [run(r_env, choose, budget.T, s, passes=passes) for s in seeds5]
        regrets[regime][method] = [float(r[-1]) for r, _ in runs]
        curves[regime][method] = np.array([r for r, _ in runs])
        if method == "LinUCB":  # the widths LinUCB believes at the end, for one user
            probe_x = features(r_env.sample_context(np.random.default_rng(RUN_SEED + 9)))
            ms = runs[0][1]
            widths_end[regime] = float(np.mean([np.sqrt(probe_x @ m.A_inv @ probe_x) for m in ms]))
        print(f"{regime}: {method} done (5 seeds, {time.perf_counter() - t0:.1f} s)")
seconds5 = time.perf_counter() - started

# %% [markdown] role="ex5-predict"
# While the cell above runs: (1) In the first three regimes, will LinUCB's IQM regret be higher
# than greedy's in any of them? (2) Will its lead over greedy be larger or smaller under drift
# than in the well-specified regime? (3) The warm launch repeats the well-specified regime
# after three passes instead of one: will LinUCB's lead over greedy grow, shrink or vanish?
# Then write down how many of your LinUCB calls you expect to be ties at 5 seeds.


# %% role="ex5-stub"
def verdicts(regrets, budget):
    """Verdicts for LinUCB and for LinTS against greedy, in every regime.

    regrets[regime][method] holds one final regret per seed, for method in "greedy",
    "LinUCB", "LinTS". Returns {(regime, method): word} for method in ("LinUCB", "LinTS"),
    where word is the better method's name, "tie" or "single seed", from
    prl.evaluate.compare with the budget. Print each verdict's sentence.
    """
    # TODO 5: compare(method's regrets, greedy's, names=..., budget=..., higher_is_better=...)
    raise NotImplementedError(
        "TODO 5: write verdicts, or run lab.use_reference(5) to continue with the solution"
    )


# %% [markdown] role="ex5-hint"
# `v = compare(a, b, names=(method, "greedy"), budget=budget, higher_is_better=...)` returns
# a Verdict whose `winner` is "A", "B", "tie" or "single seed" and whose `sentence` says why.
# Map "A" to the method's name and "B" to "greedy". Regret is a cost: which way does
# `higher_is_better` go?


# %% role="ex5-sol"
@lab.solution(5)
def verdicts(regrets, budget):
    """LinUCB and LinTS against greedy in every regime (compare)."""
    out = {}
    for regime, by_method in regrets.items():
        for method in ("LinUCB", "LinTS"):
            v = compare(
                by_method[method],
                by_method["greedy"],
                names=(method, "greedy"),
                budget=budget,
                higher_is_better=False,  # regret is a cost
            )
            line = f"{regime}, {method} vs greedy: {v.sentence}"
            print(textwrap.fill(line, 100, subsequent_indent="  "))
            names = {"A": method, "B": "greedy"}
            out[regime, method] = names.get(v.winner, v.winner)
    return out


# %% role="ex5-run"
rt = prl.runtime.detect()
BUDGET5 = Budget(
    steps=budget.T,
    seeds=5,
    wall_clock_s=seconds5,
    hardware=f"{rt.platform}, {rt.n_cpu} CPU",
)
print(BUDGET5.sentence())
print(f"Seeds {seeds5[0]}-{seeds5[-1]}; final pseudo-regret per seed:")
for regime in REGIMES:
    print(f"{regime}:")
    for method in METHODS:
        x5 = np.array(regrets[regime][method])
        lo, hi = bootstrap_ci(x5, iqm)
        seeds_txt = " ".join(f"{r:6.1f}" for r in x5)
        print(f"  {method:7s} {seeds_txt} | IQM {iqm(x5):6.1f} [{lo:6.1f}, {hi:6.1f}]")
    print(f"  LinUCB's mean width at the end, one user: {widths_end[regime]:.3f}")
calls = verdicts(regrets, BUDGET5)
print("\nVerdicts: the better method, or tie (5 vs 5 seeds; the budget is printed above)")
print(f"  {'condition':15s} {'LinUCB vs greedy':>17s} {'LinTS vs greedy':>16s}")
for regime in REGIMES:
    print(f"  {regime:15s} {calls[regime, 'LinUCB']:>17s} {calls[regime, 'LinTS']:>16s}")
ties = sum(word == "tie" for (_, m), word in calls.items() if m == "LinUCB")
print(f"  LinUCB calls that are ties: {ties} of {len(REGIMES)}")
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
styles = {"greedy": "--", "LinUCB": "-", "LinTS": "-."}
for ax, regime in zip(axes.flat, REGIMES, strict=True):
    for method, ls in styles.items():
        c5 = curves[regime][method]
        middle = np.sort(c5, axis=0)[1:-1].mean(axis=0)  # the IQM of 5 seeds, round by round
        line = ax.plot(middle, ls=ls, lw=2.2, label=f"{method}, IQM")[0]
        for k in range(len(c5)):
            ax.plot(c5[k], ls=ls, lw=0.6, alpha=0.5, color=line.get_color())
    ax.set(title=regime, xlabel="round")
for ax in axes[:, 0]:
    ax.set_ylabel("cumulative pseudo-regret")
axes[0, 0].legend(title="thin: every seed")
plt.tight_layout()
plt.show()
iqms = {w: {m: iqm(regrets[w][m]) for m in METHODS} for w in REGIMES}
lab.metric("regret_iqm", {w: {m: round(v, 2) for m, v in d.items()} for w, d in iqms.items()})
lab.metric("verdicts", {f"{w}/{m}": word for (w, m), word in calls.items()})
# Room code: greedy's IQM regret minus the method's (positive: the method does better).
for short, regime in (
    ("spec", "well-specified"),
    ("misspec", "misspecified"),
    ("drift", "drifting"),
):
    lab.metric(f"ucb_lead_{short}", round(iqms[regime]["greedy"] - iqms[regime]["LinUCB"], 2))
lab.metric("ucb_lead_warm", round(iqms["warm launch"]["greedy"] - iqms["warm launch"]["LinUCB"], 2))
lab.metric(
    "ts_lead_spec", round(iqms["well-specified"]["greedy"] - iqms["well-specified"]["LinTS"], 2)
)
lab.metric("ts_lead_warm", round(iqms["warm launch"]["greedy"] - iqms["warm launch"]["LinTS"], 2))

# %% [markdown] role="ex5-explain"
# Read your verdicts against your Predict. Misspecified: compare LinUCB's width at the end
# with the well-specified one, and its regret. What does a narrow width promise, and under
# which assumption? (Gotcha: "Narrow widths mean the model is right.") Drifting: the width
# shrinks with the amount of data in a direction, not with its age (ridge), so once it is
# small LinUCB explores no more than greedy, and both serve tastes that are turning (drift).
# What did its early exploration buy, once those tastes had reversed? Warm launch: what did
# 30 rounds of forced showing give greedy that one pass did not? A tie is five seeds that
# cannot separate two methods: write down your calls, and the debrief pools the room's.

# %% role="ex5-chk"
lab.check(5, m05.check_verdicts, verdicts)

# %% [markdown] role="ex6-head"
# **Logging for later.** Day 5 evaluates new policies from a log, and needs, for every
# logged round, the probability with which the logged item was shown: its *propensity*.
# Write the behavior policy as probabilities: ε-greedy over any scores puts 1 − ε on the
# best-scoring item and spreads ε evenly over all K (ε-greedy logging). The run cell builds
# it over your LinUCB models from Exercise 3, frozen, logs with `env.log`, which records
# each round's propensity, and checks the log with importance weights π/π_b (importance
# weight). In the log, `log.A` holds the items shown and `log.P` the propensities.

# %% [markdown] role="ex6-predict"
# K = 10 and ε = 0.1. About what fraction of logged rows show an item other than the
# best-scoring one: 10%, 9% or 1%? What propensity does the log record on those rows, and on
# the others?


# %% role="ex6-stub"
def log_policy(scores, epsilon):
    """epsilon-greedy over scores: 1 - epsilon + epsilon/K on argmax_first(scores),
    epsilon/K on every other item. Returns probabilities, shape (K,)."""
    # TODO 6: start from epsilon / K everywhere, then add 1 - epsilon to the best item.
    raise NotImplementedError(
        "TODO 6: write log_policy, or run lab.use_reference(6) to continue with the solution"
    )


# %% [markdown] role="ex6-hint"
# Test it before the run cell. Scores (0.1, 0.5, 0.3, 0.2) with ε = 0.1 must give
# (0.025, 0.925, 0.025, 0.025). Does yours sum to 1? Which item wins when two scores are
# equal?


# %% role="ex6-sol"
@lab.solution(6)
def log_policy(scores, epsilon):
    """epsilon-greedy over scores, as probabilities."""
    n = len(scores)
    probs = np.full(n, epsilon / n)  # every item: epsilon / K
    probs[argmax_first(scores)] += 1.0 - epsilon  # (ε-greedy logging)
    return probs


# %% role="ex6-run"
def behavior(x, rng):
    """The deployed logger: epsilon-greedy over your frozen LinUCB models."""
    return log_policy(linucb_scores(models_ucb, features(x), C_UCB), EPSILON)


def uniform(x):
    """A target policy: every item with probability 1/K."""
    return np.full(K, 1.0 / K)


def new_policy(x):
    """A target policy for Day 5: greedy on your LinTS models' predictions."""
    probs = np.zeros(K)
    probs[argmax_first([m.predict(features(x)) for m in models_ts])] = 1.0
    return probs


def mean_weight(log, target):
    """Mean of pi(A_i | X_i) / P_i over the log's rows, with a 95% normal interval."""
    pi = np.array([target(x)[a] for x, a in zip(log.X, log.A, strict=True)])
    w = pi / log.P  # (importance weight)
    half = 1.96 * w.std(ddof=1) / math.sqrt(len(w))
    return w.mean(), w.mean() - half, w.mean() + half


env.reset(seed=RUN_SEED)
log = env.log(behavior, budget.n_log, seed=RUN_SEED + 200)
greedy_rows = log.probs.argmax(axis=1) == log.A
print(f"{len(log)} logged rounds; rows showing the best-scoring item: {greedy_rows.mean():.3f}")
print("propensities recorded:", np.unique(np.round(log.P, 6)))
for name, target in (("uniform", uniform), ("new_policy", new_policy)):
    m_, lo_, hi_ = mean_weight(log, target)
    print(f"  mean weight, target {name:10s}: {m_:.3f}  [95% CI {lo_:.3f}, {hi_:.3f}]")
greedy_log = env.log(
    lambda x, rng: log_policy(linucb_scores(models_ucb, features(x), C_UCB), 0.0),
    budget.n_log,
    seed=RUN_SEED + 200,
)
w_eps0 = mean_weight(greedy_log, uniform)[0]
print(f"the same logger with epsilon = 0: mean weight, uniform target {w_eps0:.3f}")
lab.metric("mean_weight", round(float(mean_weight(log, uniform)[0]), 4))  # room code

# %% [markdown] role="ex6-explain"
# For any target π, the mean of π/π_b over a correct log estimates Σ_a π(a | x) over the
# items the logger can show, which is 1 when π_b > 0 wherever π > 0 (importance weight).
# With ε = 0 the logger shows one item per user, and the mean weight for the uniform target
# fell to 1/K: the other items' probability has nowhere to come from. Which policies could
# Day 5 still evaluate from that greedy log, and which not?

# %% role="ex6-chk"
lab.check(6, m05.check_log_policy, log_policy)

# %% [markdown] role="ex7-head"
# **Planted bug.** A colleague's `log_suspect` runs a behavior policy on the live stream and
# records each round's context, item, click and propensity, in the same format as
# `env.log`. It runs, its rows look plausible, and every recorded propensity is a valid
# probability. Check it with the mean weight, which must be about 1 for any target on a
# correct log, then find the line and fix it.

# %% [markdown] role="ex7-predict"
# Your ε-greedy logger (K = 10, ε = 0.1) and the uniform target, π(a | x) = 1/10. On a
# correct log, what is a row's weight π/π_b when the best item was shown, and when another
# item was? About what share of rows gets each weight?


# %% role="ex7-stub"
def log_suspect(env, policy, n, seed):
    """A colleague's logger: n rounds of policy(x, rng) on env's stream `seed`."""
    # TODO 7: once the diagnostics below show the problem, fix the one wrong line.
    # Then rerun this cell and the run cell.
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _ in range(n):
        x = env.context()
        probs = policy(x, rng)
        a = int(rng.choice(len(probs), p=probs))
        r = env.step(a)
        rows.append((x, a, r, float(probs.max()), probs))
    X, A, R, P, probs = (np.array(k) for k in zip(*rows, strict=True))
    T = np.arange(n)
    return envs.LoggedBandit(X=X, A=A, R=R, P=P, probs=probs, T=T)


# %% [markdown] role="ex7-hint"
# Split the suspect's rows in two: rows where the shown item was the logger's best
# (`suspect_log.A == suspect_log.probs.argmax(axis=1)`), and the rest. What propensity does
# `log_suspect` record in each group, and what does your Predict say it should be?


# %% role="ex7-sol"
@lab.solution(7)
def log_suspect(env, policy, n, seed):
    """The colleague's logger, fixed: the shown item's propensity."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _ in range(n):
        x = env.context()
        probs = policy(x, rng)
        a = int(rng.choice(len(probs), p=probs))
        r = env.step(a)
        rows.append((x, a, r, float(probs[a]), probs))
    X, A, R, P, probs = (np.array(k) for k in zip(*rows, strict=True))
    T = np.arange(n)
    return envs.LoggedBandit(X=X, A=A, R=R, P=P, probs=probs, T=T)


# %% role="ex7-run"
suspect_log = log_suspect(env, behavior, budget.n_log, RUN_SEED + 300)
print(f"{len(suspect_log)} rounds logged by log_suspect")
for name, target in (("uniform", uniform), ("new_policy", new_policy)):
    m_, lo_, hi_ = mean_weight(suspect_log, target)
    print(f"  suspect's mean weight, target {name:10s}: {m_:.3f}  [95% CI {lo_:.3f}, {hi_:.3f}]")
fig, ax = plt.subplots()
for lg, label, ls in ((log, "env.log", "-"), (suspect_log, "log_suspect", "--")):
    pi = np.array([uniform(x)[a] for x, a in zip(lg.X, lg.A, strict=True)])
    w = pi / lg.P
    ax.plot(np.cumsum(w) / np.arange(1, len(w) + 1), ls=ls, label=label)
ax.axhline(1.0, ls=":", color=NEUTRAL_LIGHT, label="1: every correct log")
ax.set(xlabel="logged rounds", ylabel="running mean weight, uniform target", ylim=(0, 2))
ax.legend()
plt.show()

# %% [markdown] role="ex7-explain"
# Before your fix, which rows had the wrong propensity, and by what factor were their
# weights too small? Why does the bug hide from a target that agrees with the logger, such
# as `new_policy`? (Gotcha: "The propensity is bookkeeping.")
#
# **Diagnostics for this lab.** For a learner, plot cumulative pseudo-regret against the
# known means, every seed shown. Healthy: its slope, the regret per round, keeps falling.
# Broken: a slope that stops falling or grows, as when a method has written an item off or
# its model is wrong (misspecified) or stale (drifting), while its widths keep shrinking.
# Print each item's show count: an item stuck at its launch count was written off. For
# ridge, compare ŵ with the closed form, and A⁻¹ with `np.linalg.inv(A)`. For a log, check
# that every propensity is the probability of the shown item and is positive, and that the
# mean weight is about 1 for a target that differs from the logger, such as uniform; a
# target that agrees with the logger hides the bug.

# %% role="ex7-chk"
lab.check(7, m05.check_log_suspect, log_suspect)

# %% [markdown]
# **Your room code.** The last cell, `lab.finish()`, prints one line that starts with
# `ROOM m05`. It holds your seat, the checkpoints you passed, whether QUICK budgets ran, and
# the numbers the debrief pools: greedy's IQM regret minus LinUCB's in each condition
# (`ucb_lead_spec`, `ucb_lead_misspec`, `ucb_lead_drift`, `ucb_lead_warm`) and minus LinTS's
# in two (`ts_lead_spec`, `ts_lead_warm`); positive means the method's IQM regret was below
# greedy's (Exercise 5). And the mean weight of your correct log for the uniform target (`mean_weight`,
# Exercise 6). Paste it where your instructor says. Your run record keeps every number the
# lab printed, the other LinTS leads and your verdicts.

# %% [markdown] role="stretch"
# **Parity with MABWiser.** MABWiser 2.7.4 is a maintained bandit library whose LinUCB keeps,
# for every arm, the same statistics as your `Ridge`: `A_inv` and `Xty` (your `b`). The cell
# below replays one LinUCB run, fits MABWiser's LinUCB with the same c and λ_ridge on the same
# stream of (item, click, features), and compares the statistics and the scores. On Colab
# the setup cell installs MABWiser; elsewhere, if it is not installed, the cell says so and
# skips. **Predict first:** MABWiser recomputes A⁻¹ with `np.linalg.inv`; you kept it with
# Sherman–Morrison over 3000 updates. Will the largest difference in A⁻¹ be exactly 0, at the
# level of rounding (about 1e−15), or about 1e−6?

# %% role="stretch-run"
try:
    from mabwiser.mab import MAB, LearningPolicy
except ImportError:
    MAB = None
    print("MABWiser is not installed, so this stretch is skipped. To run it, install it")
    print("(pip install mabwiser==2.7.4), restart the session and rerun the notebook.")
if MAB is not None:
    env.reset(seed=RUN_SEED)
    rng_s = np.random.default_rng(RUN_SEED + 1)
    mine = [Ridge(D, LAM) for _ in range(K)]
    stream_a, stream_r, stream_phi = [], [], []
    for t in range(budget.T):
        phi = features(env.context())
        a = t if t < K else linucb_choice(mine, phi, rng_s)
        click = env.step(a)
        mine[a].update(phi, click)
        stream_a.append(a)
        stream_r.append(click)
        stream_phi.append(phi)
    mab = MAB(
        arms=list(range(K)), learning_policy=LearningPolicy.LinUCB(alpha=C_UCB, l2_lambda=LAM)
    )
    mab.fit(decisions=stream_a, rewards=stream_r, contexts=np.array(stream_phi))
    theirs = mab._imp.arm_to_model  # MABWiser 2.7.4's per-arm ridge models (private API)
    d_inv = max(np.abs(theirs[a].A_inv - mine[a].A_inv).max() for a in range(K))
    d_b = max(np.abs(theirs[a].Xty - mine[a].b).max() for a in range(K))
    test = np.array([features(env.sample_context(rng_s)) for _ in range(200)])
    their_scores = mab.predict_expectations(test)
    d_score = max(
        abs(their_scores[i][a] - linucb_scores(mine, test[i], C_UCB)[a])
        for i in range(len(test))
        for a in range(K)
    )
    print(f"max |A_inv difference| = {d_inv:.1e}; max |Xty - b| = {d_b:.1e}")
    print(f"max |score difference| over 200 users and {K} items = {d_score:.1e}")
    print("MABWiser inverts A with np.linalg.inv; you kept A_inv with Sherman-Morrison.")
