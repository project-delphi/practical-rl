# ---
# jupyter:
#   prl:
#     module: m07
#     pins:
#       gymnasium: "1.4.0"
# ---

# %% [markdown]
# In this lab a table of values is replaced by a linear function of features. You count what
# a table would cost for a three-product shop, write the semi-gradient TD(0) update, and show
# that its expected update is not the gradient of anything. Then you run it on Baird's
# counterexample, where every reward is 0 and the weights still grow without bound, and remove
# one leg of the deadly triad at a time. Last, you train linear Q-learning on CartPole and find
# the bug in a colleague's features from the weight-norm log.
#
# The briefing's equations are named in brackets, for example (semi-gradient TD(0)) or
# (TD matrix). Its table "Equations and where you implement them" gives each name's equation
# number. States and weights of Baird's counterexample are numbered from 1 on the page and in
# the text, as in Sutton and Barto; in code they are indexed from 0, so state 7 is index 6.

# %%
import textwrap
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from prl.envs import Baird, InventoryMDP
from prl.evaluate import Budget

# Budgets. Lab code reads them from here and never branches on QUICK. The Baird exercises are
# exact and cost well under a second, so only CartPole's training steps change.
budget = lab.use_settings(
    prl.runtime.settings(
        dict(steps=30_000, eps_steps=10_000, log_every=1000, eval_episodes=10),  # live
        dict(steps=10_000, eps_steps=4_000, log_every=1000, eval_episodes=10),  # QUICK
        designed="colab-cpu",
    )
)
TRAIN = dict(
    steps=budget.steps, eps_steps=budget.eps_steps, log_every=budget.log_every
)  # loop args
# Your seat's seed. Seat 0 does not reuse the seeds of the briefing's figures, the recorded
# experiments or the checkpoints (0-99), so every seat adds new evidence.
RUN_SEED = (SEED + 1) * 1000
lab.seeds([RUN_SEED])
_rt = prl.runtime.detect()
HARDWARE = f"{_rt.platform}, {_rt.cpu}, {_rt.n_cpu} CPU"
baird = Baird()  # Sutton and Barto's Figure 11.1: 7 states, 8 weights, gamma = 0.99
print(f"CartPole budget: {budget.steps} steps, epsilon decays over {budget.eps_steps}")
print(f"Your seed: {RUN_SEED}")

# %% [markdown] role="ex1-head"
# **Frame inventory at scale.** Module 1's shop had one product with stock 0 to 20. Now it
# sells three products, each with stock 0 to 50, and each period it orders 0 to 50 units of
# each. Write `scale_spec`, which counts what a table needs for n products of a given
# capacity, and the length of the one-hot features the briefing proposes: for each product,
# a one-hot vector of its stock level, the three vectors side by side (one-hot features).
# The run cell then builds a two-product shop small enough to solve exactly, twice: once
# with independent products, and once where the two products share one delivery fee.

# %% [markdown] role="ex1-predict"
# The second shop pays its fixed delivery fee of 5 once when both products are ordered in
# the same period, instead of once per product. With one-hot features (12 for two products
# of capacity 5), will the best linear fit of the second shop's exact values be exact? If
# not, will its largest error be smaller or larger than the fee itself?


# %% role="ex1-stub"
def scale_spec(n_products, capacity):
    """The size of inventory control with `n_products` products, each holding 0..capacity.

    Every period the shop orders 0..capacity units of each product. Returns a dict of ints:
      "n_states": joint stock levels; "n_actions": joint orders;
      "q_entries": entries of a table Q[s, a]; "feature_dim": the length of the one-hot
      features (one indicator per product and stock level).
    """
    # TODO 1: four counts. Python ints do not overflow.
    raise NotImplementedError(
        "TODO 1: write scale_spec, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# A product has capacity + 1 stock levels, 0 to capacity. Joint stock levels multiply across
# products; one-hot indicators add up. The same holds for orders.


# %% role="ex1-sol"
@lab.solution(1)
def scale_spec(n_products, capacity):
    """Counts for a table and for one-hot features."""
    levels = capacity + 1  # stock 0..capacity
    n_states = levels**n_products  # joint stock levels
    n_actions = levels**n_products  # joint orders
    return {
        "n_states": n_states,
        "n_actions": n_actions,
        "q_entries": n_states * n_actions,
        "feature_dim": n_products * levels,  # (one-hot features)
    }


# %% role="ex1-run"
spec = scale_spec(3, 50)
print(f"Three products, capacity 50: {spec['n_states']:,} states, {spec['n_actions']:,} orders")
print(f"  Q table: {spec['q_entries']:,} entries, {spec['q_entries'] * 8 / 1e9:,.0f} GB in float64")
print(f"  one-hot features: {spec['feature_dim']} weights per value function")


def two_product_values(shared_fee, capacity=5, reorder=2):
    """Provided: the exact values of a two-product shop under an order-up-to rule.

    Each product follows Module 1's InventoryMDP (demand Poisson with mean 2). The rule: when
    stock is below `reorder`, order up to capacity; otherwise order nothing. With
    `shared_fee`, the fixed delivery fee is paid once when both products order. Returns
    (values over the 36 joint states, the one-hot feature matrix)."""
    shop = InventoryMDP(capacity=capacity, demand_mean=2.0, max_demand=8)
    levels = capacity + 1
    rule = np.zeros((levels, levels))  # rule[x, q] = 1 for the order q at stock x
    for x in range(levels):
        rule[x, capacity - x if x < reorder else 0] = 1.0
    P1 = np.einsum("sa,sat->st", rule, shop.P)  # one product's chain under the rule
    r1 = (rule * shop.R).sum(axis=1)
    P = np.kron(P1, P1)  # the products move independently: joint state (x1, x2)
    ones = np.ones(levels)
    r = np.kron(r1, ones) + np.kron(ones, r1)  # the rewards add up
    if shared_fee:
        orders = (np.arange(levels) < reorder).astype(float)  # does this stock order?
        r = r + shop.params["fixed_cost"] * np.kron(orders, orders)  # one fee saved
    v = np.linalg.solve(np.eye(levels**2) - shop.gamma * P, r)
    eye = np.eye(levels)
    features = np.hstack([np.kron(eye, ones[:, None]), np.kron(ones[:, None], eye)])
    return v, features


for shared in (False, True):
    v, features = two_product_values(shared)
    w_fit, *_ = np.linalg.lstsq(features, v, rcond=None)
    gap = np.abs(features @ w_fit - v).max()
    name = "shared delivery fee" if shared else "independent products"
    print(f"Two products, capacity 5, {name}: values {v.min():.2f} to {v.max():.2f};")
    print(f"  best fit with {features.shape[1]} one-hot features: largest error {gap:.3g}")

# %% [markdown] role="ex1-explain"
# Why does a sum of one product's values fit the independent shop exactly (one-hot
# features)? Where does the shared fee make the value of one product's stock depend on the
# other's? For the three-product shop, how many weights would one indicator per pair of
# stock levels add, and is that still far below 132,651? Which coupling would you expect in
# a real shop (a shared truck, a shared warehouse, a budget), and which features would
# represent it?

# %% role="ex1-chk"
# The checkpoints of this lab. (Technical expert: move this import to the first code cell
# once prl.checks.m07 exists.)
from prl.checks import m07

lab.check(1, m07.check_scale_spec, scale_spec)

# %% [markdown] role="ex2-head"
# **The semi-gradient TD(0) update.** With linear values v̂(s, w) = wᵀφ(s) (linear value),
# one transition s → s' with reward r gives the TD error δ = r + γ wᵀφ(s') − wᵀφ(s), and
# semi-gradient TD(0) moves the weights along the features of s only: w ← w + α δ φ(s)
# (semi-gradient TD(0)). The target r + γ wᵀφ(s') also depends on w; the update ignores that.
# A terminal s' has value 0: pass φ(s') = 0.

# %% [markdown] role="ex2-predict"
# Baird's state 1 has features φ = (2, 0, 0, 0, 0, 0, 0, 1) and state 7 has
# φ = (0, 0, 0, 0, 0, 0, 1, 2). At the start, w = (1, 1, 1, 1, 1, 1, 10, 1), so v̂(1) = 3 and
# v̂(7) = 12. One transition 1 → 7 with reward 0, γ = 0.99 and α = 0.01: what are δ and the new
# v̂(1)? Does v̂(7), the target's value, move too, and which way?


# %% role="ex2-stub"
def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):
    """One semi-gradient TD(0) update for linear values (semi-gradient TD(0)).

    w, phi, phi_next: float arrays of shape (d,): the weights, φ(s) and φ(s'), with
    phi_next all zeros when s' is terminal. r, gamma, alpha: floats.
    Returns the new weights, a new array of shape (d,); do not change w in place.
    """
    # TODO 2: the TD error, then a step along phi.
    raise NotImplementedError(
        "TODO 2: write semi_gradient_td0, or run lab.use_reference(2) to continue with the solution"
    )


# %% [markdown] role="ex2-hint"
# The values are dot products, `w @ phi`. The gradient of wᵀφ(s) with respect to w is φ(s).
# `w + ...` makes a new array; `w += ...` would change the caller's array.


# %% role="ex2-sol"
@lab.solution(2)
def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):
    """w + alpha * delta * phi(s); the target is held fixed."""
    delta = r + gamma * w @ phi_next - w @ phi  # TD error
    return w + alpha * delta * phi  # (semi-gradient TD(0))


# %% role="ex2-run"
w0 = baird.w0.copy()
phi1, phi7 = baird.phi[0], baird.phi[6]  # states 1 and 7
w1 = semi_gradient_td0(w0, phi1, 0.0, phi7, baird.gamma, 0.01)
print(f"Before: v(1) = {w0 @ phi1:.4f}, v(7) = {w0 @ phi7:.4f}")
print(f"After one update 1 -> 7: v(1) = {w1 @ phi1:.4f}, v(7) = {w1 @ phi7:.4f}")
print(f"  w = {np.round(w1, 4).tolist()}")
print(
    f"  TD error before {0.99 * (w0 @ phi7) - w0 @ phi1:.4f}, after {0.99 * (w1 @ phi7) - w1 @ phi1:.4f}"
)

# %% [markdown] role="ex2-explain"
# Both values rose, because states 1 and 7 share weight 8. The update moved v̂(1) toward its
# target and moved the target up as well. Did the TD error of this transition shrink as much
# as v̂(1) moved? In a table, could an update at state 1 change v̂(7)? Which term of the full
# gradient of ½δ² does (semi-gradient TD(0)) leave out?

# %% role="ex2-chk"
lab.check(2, m07.check_semi_gradient_td0, semi_gradient_td0)

# %% [markdown] role="ex3-head"
# **Why it is not a gradient.** Average the update over states s weighted by μ(s) and over
# next states s' ~ P_π(s, ·). Stack the features as the rows of Φ (one row per state), put μ
# on the diagonal of D, and the expected update is w ← w + α(b − Aw) (expected update), with
# A = ΦᵀD(I − γP_π)Φ and b = ΦᵀD r_π (TD matrix). Write `td_matrix`. If the expected update
# were gradient descent on some objective J(w), then A would be the Hessian of J, and a
# Hessian is symmetric. The run cell compares A with Aᵀ for Baird.

# %% [markdown] role="ex3-predict"
# Weight 1 appears only in state 1's value, and weight 7 only in state 7's. State 1's target
# is state 7's value, and state 7's target is its own value. Before you run: will entry
# A[1, 7] (how much w₇ moves the expected update of w₁) be zero or not? And A[7, 1]? (In code
# these are A[0, 6] and A[6, 0].)


# %% role="ex3-stub"
def td_matrix(phi, mu, P, gamma):
    """The matrix A of the expected semi-gradient TD(0) update (TD matrix).

    phi: (S, d) features, one row per state. mu: (S,) the weight of each state's update.
    P: (S, S) state-to-state transitions under the policy being evaluated. gamma: float.
    Returns A = phiᵀ D (I − γ P) phi, shape (d, d), with D = diag(mu).
    """
    # TODO 3: one line of matrix products.
    raise NotImplementedError(
        "TODO 3: write td_matrix, or run lab.use_reference(3) to continue with the solution"
    )


# %% [markdown] role="ex3-hint"
# `np.diag(mu)` builds D and `np.eye(S)` builds I; `@` multiplies matrices. Check the shape:
# (d, S) @ (S, S) @ (S, S) @ (S, d) is (d, d).


# %% role="ex3-sol"
@lab.solution(3)
def td_matrix(phi, mu, P, gamma):
    """A = phi^T D (I - gamma P) phi."""
    S = len(mu)
    return phi.T @ np.diag(mu) @ (np.eye(S) - gamma * P) @ phi  # (TD matrix)


# %% role="ex3-run"
A = td_matrix(baird.phi, baird.mu_behavior, baird.P_target, baird.gamma)
print("7 A for Baird's counterexample (rows and columns are weights 1-8):")
print(np.array2string(7 * A, precision=2, suppress_small=True))
print(
    f"A[1, 7] = {A[0, 6]:.5f}, A[7, 1] = {A[6, 0]:.5f}; largest |A - Aᵀ| = {np.abs(A - A.T).max():.4f}"
)
eig = np.linalg.eigvals(A)
print(f"Eigenvalues of A, real parts: {np.round(np.sort(eig.real), 5).tolist()}")

# %% [markdown] role="ex3-explain"
# A[1, 7] is not zero because state 1's target uses w₇; A[7, 1] is zero because state 7's
# target never uses w₁. Bootstrapping looks in one direction, from a state to its successor,
# and that breaks the symmetry. Why can no J(w) have −(b − Aw) as its gradient (no
# objective)? Two eigenvalues of A are negative: what does that predict for
# w ← w + α(b − Aw) along their eigenvectors, whatever α > 0? (Gotcha: "Semi-gradient TD is
# gradient descent.")

# %% role="ex3-chk"
lab.check(3, m07.check_td_matrix, td_matrix)

# %% [markdown] role="ex4-head"
# **Baird's counterexample with expected updates.** Every reward is 0, so v_π = 0, and w = 0
# represents it exactly. The behavior policy picks the dashed action with probability 6/7
# and the solid one with 1/7, so the next state is uniform over all seven and every state is
# updated with the same weight, μ = 1/7. The target policy always takes the solid action, so
# P_π sends every state to state 7. Write `baird_expected_update`: one sweep that updates
# every state at once with its expected TD error under the target policy, weighted by μ
# (expected update). This is Sutton and Barto's Eq. 11.9: no sampling, no randomness. The
# run cell sweeps 1000 times with α = 0.01 from w = (1, 1, 1, 1, 1, 1, 10, 1).

# %% [markdown] role="ex4-predict"
# After 1000 sweeps with α = 0.01, is ‖w‖ nearer 10, 40 or 400? Which weights grow, and does
# any weight fall?


# %% role="ex4-stub"
def baird_expected_update(w, phi, mu, P, r, gamma, alpha):
    """One expected semi-gradient TD(0) sweep over all states at once (expected update).

    w: (d,). phi: (S, d). mu: (S,) weights of the states' updates. P: (S, S) transitions
    under the target policy. r: (S,) expected rewards under the target policy.
    Returns w + alpha * sum_s mu(s) * (r(s) + gamma * sum_s' P(s, s') v(s') - v(s)) * phi(s),
    with v = phi @ w: a new array of shape (d,).
    """
    # TODO 4: the vector of expected TD errors, one per state, then the weighted sum.
    raise NotImplementedError(
        "TODO 4: write baird_expected_update, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# `v = phi @ w` gives every state's value. The expected TD errors are one vector of length S.
# Weighting row s of phi by mu(s) and summing the rows is a single product, `phi.T @ (...)`.


# %% role="ex4-sol"
@lab.solution(4)
def baird_expected_update(w, phi, mu, P, r, gamma, alpha):
    """w + alpha * phi^T D (r + gamma P phi w - phi w)."""
    v = phi @ w  # every state's value
    delta = r + gamma * P @ v - v  # expected TD error per state
    return w + alpha * phi.T @ (mu * delta)  # (expected update)


# %% role="ex4-run"
ALPHA_BAIRD, SWEEPS = 0.01, 1000
base = dict(phi=baird.phi, mu=baird.mu_behavior, P=baird.P_target, r=baird.rewards)
path = [baird.w0.copy()]
for k in range(SWEEPS):
    path.append(baird_expected_update(path[-1], **base, gamma=baird.gamma, alpha=ALPHA_BAIRD))
path = np.array(path)
norms = np.linalg.norm(path, axis=1)
for k in (0, 1, 10, 100, 1000):
    print(f"sweep {k:4d}: ‖w‖ = {norms[k]:9.3f}   w = {np.round(path[k], 2).tolist()}")
# The same sweeps from your td_matrix: w_k = (I - alpha A)^k w_0, since b = 0 here.
M = np.eye(8) - ALPHA_BAIRD * A
same = np.allclose(path[-1], np.linalg.matrix_power(M, SWEEPS) @ baird.w0, rtol=1e-9, atol=1e-9)
print(f"Your update and your td_matrix agree after {SWEEPS} sweeps: {same}")
for a_try in (0.001, 0.01, 0.1):
    rho = np.abs(np.linalg.eigvals(np.eye(8) - a_try * A)).max()
    print(f"alpha = {a_try:<5}: the fastest-growing direction grows by {rho:.6f} per sweep")
lab.metric("baird_norm_1000", round(float(norms[-1]), 3))
fig, (ax_w, ax_n) = plt.subplots(1, 2, figsize=(10, 3.8))
sweeps = np.arange(SWEEPS + 1)
ax_w.plot(sweeps, path[:, 0], color=prl.plot.SERIES[0], label="w₁ to w₆ (equal)")
ax_w.plot(sweeps, path[:, 6], color=prl.plot.SERIES[1], ls="--", label="w₇")
ax_w.plot(sweeps, path[:, 7], color=prl.plot.SERIES[2], ls=":", label="w₈")
ax_w.set(xlabel="sweep", ylabel="weight", title="Expected semi-gradient TD(0), α = 0.01")
ax_w.legend()
ax_n.plot(sweeps, norms, color=prl.plot.SERIES[0])
ax_n.set(xlabel="sweep", ylabel="‖w‖ (log scale)", yscale="log", title="The norm of w")
plt.show()

# %% [markdown] role="ex4-explain"
# Nothing here is random: every sweep is the exact expectation. Which weight grows fastest,
# and why does w₇ fall while w₈ rises? The run printed how fast the fastest direction grows
# per sweep at three step sizes: what does a smaller α buy you? Where in A (TD matrix) is the
# growth decided? (Gotcha: "A bigger network will fix divergence.")

# %% role="ex4-chk"
lab.check(4, m07.check_baird_expected_update, baird_expected_update)

# %% [markdown] role="ex5-head"
# **Remove one leg of the triad.** The divergence needs function approximation,
# bootstrapping and off-policy updates together (deadly triad). Each leg is one argument of
# your `baird_expected_update` and `td_matrix`. Write `remove_one_leg`: it returns, for each
# leg, the arguments that change.
# - **On-policy updates:** weight the states by the target policy's own long-run
#   distribution μ_π, the stationary distribution of P_π, in place of the behavior's.
# - **Tabular features:** one feature per state, so Φ is the identity.
# - **Monte Carlo targets:** no bootstrapping. The expected Monte Carlo target of state s is
#   v_π(s), the solution of v = r_π + γ P_π v. In the update that is γ = 0 and r = v_π.
#
# The run cell starts each variant from the same values and runs it to a million sweeps
# with α = 0.01, using powers of I − αA from your `td_matrix`.

# %% [markdown] role="ex5-predict"
# The briefing says each variant converges. For each one: will w reach 0? Will all seven
# values reach v_π = 0? Which variant will be slowest?


# %% role="ex5-stub"
def remove_one_leg(env):
    """The changes to Baird's expected update that remove one leg of the deadly triad each.

    env has env.n_states, env.gamma, env.P_target (S, S) and env.rewards (S,). Returns
    {"on-policy": {"mu": ...}, "tabular": {"phi": ...},
     "monte-carlo": {"gamma": 0.0, "r": ...}}: mu is P_target's stationary distribution,
    shape (S,); phi is the (S, S) identity; r is v_pi = (I - gamma P_target)^-1 rewards.
    Compute mu and v_pi from env; do not type in Baird's numbers.
    """
    # TODO 5: three small dicts.
    raise NotImplementedError(
        "TODO 5: write remove_one_leg, or run lab.use_reference(5) to continue with the solution"
    )


# %% [markdown] role="ex5-hint"
# A stationary distribution satisfies μᵀP = μᵀ: start from the uniform distribution and
# repeat `mu = mu @ P` a few hundred times. `np.linalg.solve(np.eye(S) - gamma * P, r)` solves
# the Bellman equation for v_π. The checkpoint also tries a different target policy and
# nonzero rewards.


# %% role="ex5-sol"
@lab.solution(5)
def remove_one_leg(env):
    """On-policy weights, tabular features, Monte Carlo targets."""
    S, P = env.n_states, env.P_target
    mu = np.full(S, 1.0 / S)
    for _ in range(1000):
        mu = mu @ P  # converges to the stationary distribution
    v_pi = np.linalg.solve(np.eye(S) - env.gamma * P, env.rewards)
    return {
        "on-policy": {"mu": mu},
        "tabular": {"phi": np.eye(S)},
        "monte-carlo": {"gamma": 0.0, "r": v_pi},
    }


# %% role="ex5-run"
legs = remove_one_leg(baird)
v_pi = np.linalg.solve(np.eye(7) - baird.gamma * baird.P_target, baird.rewards)
GRID = np.unique(np.round(np.logspace(0, 6, 61)).astype(int))  # sweeps 1 to 10^6


def run_variant(cfg, w_start, ks, alpha=ALPHA_BAIRD):
    """Provided: weights after k sweeps for every k in ks, from powers of the affine map
    w -> (I - alpha A) w + alpha b, with A from your td_matrix. Overflow gives inf or nan."""
    A_v = td_matrix(cfg["phi"], cfg["mu"], cfg["P"], cfg["gamma"])
    b_v = cfg["phi"].T @ (cfg["mu"] * cfg["r"])
    d = len(w_start)
    step = np.eye(d + 1)
    step[:d, :d] -= alpha * A_v
    step[:d, d] = alpha * b_v
    out = []
    with np.errstate(over="ignore", invalid="ignore"):
        for k in ks:
            out.append((np.linalg.matrix_power(step, int(k)) @ np.append(w_start, 1.0))[:d])
    return np.array(out), A_v


variants = {"original": {}, **legs}
results, agree = {}, []
print(f"{'variant':12s} {'min Re eig(A)':>13s}   max|v - v_pi| after 10^3, 10^4, 10^5, 10^6 sweeps")
for name, change in variants.items():
    cfg = {**base, "gamma": baird.gamma, **change}
    w_start = baird.phi @ baird.w0 if name == "tabular" else baird.w0  # same starting values
    w_first = w_start.copy()
    for _ in range(100):  # your update, 100 sweeps, against the matrix powers
        w_first = baird_expected_update(w_first, **cfg, alpha=ALPHA_BAIRD)
    ws, A_v = run_variant(cfg, w_start, GRID)
    w100, _ = run_variant(cfg, w_start, [100])
    agree.append(np.allclose(w_first, w100[0], rtol=1e-8, atol=1e-8))
    with np.errstate(over="ignore", invalid="ignore"):
        err = np.abs(ws @ cfg["phi"].T - v_pi).max(axis=1)
    results[name] = (ws, err)
    at = [err[np.searchsorted(GRID, 10**j)] for j in (3, 4, 5, 6)]
    cells = "  ".join(f"{x:10.3g}" for x in at)
    print(f"{name:12s} {np.linalg.eigvals(A_v).real.min():13.5f}   {cells}")
print(f"Your update and your td_matrix agree on all four variants (100 sweeps): {all(agree)}")
for name in ("on-policy", "monte-carlo", "tabular"):
    w_end = results[name][0][-1]
    print(f"{name:12s} after 10^6 sweeps: w = {np.round(w_end, 3).tolist()}")
at_1e4 = np.searchsorted(GRID, 10**4)
lab.metric("legs_error_1e4", {k: float(v[1][at_1e4]) for k, v in results.items()})
fig, ax = plt.subplots()
styles = {"original": "-", "on-policy": "--", "tabular": ":", "monte-carlo": "-."}
for i, (name, (ws, err)) in enumerate(results.items()):
    ok = np.isfinite(err)
    ax.plot(GRID[ok], err[ok], ls=styles[name], color=prl.plot.SERIES[i], label=name)
ax.set(xscale="log", yscale="log", xlabel="sweeps (log scale)", ylabel="max over states |v - v_π|")
ax.set_title("Each leg removed, α = 0.01")
ax.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# Read each variant's smallest real part of an eigenvalue of A against its curve. With
# on-policy weights all of μ_π sits on state 7, so A = (1 − γ)φ₇φ₇ᵀ (TD matrix): which part
# of w does an update change, and which parts never move? Which values are then right, and
# which stay wrong? The Monte Carlo variant follows the gradient of an objective: why does
# w stop where it does and not at 0? The tabular variant is Module 1's policy evaluation,
# damped: why is it the slowest at γ = 0.99?

# %% role="ex5-chk"
lab.check(5, m07.check_remove_one_leg, remove_one_leg)

# %% [markdown] role="ex6-head"
# **Linear Q-learning on CartPole.** CartPole's state is four numbers: cart position and
# velocity, pole angle and angular velocity. The provided features scale each number, place
# a Gaussian bump on each point of a 5 × 5 × 5 × 5 grid, and scale the 625 bumps to unit
# length (RBF features). Each action a has its own weights W[a], so Q(s, a) = W[a]ᵀφ(s).
# Write `linear_q_update`, semi-gradient Q-learning (linear Q-learning): the target is
# r + γ max_b Q(s', b), and 0 replaces the max only when the episode *terminated*. A
# truncated episode (the 500-step limit) ended for a reason the state does not show, so its
# target still bootstraps. The provided loop picks actions ε-greedily, ε falling from 1 to
# 0.05, with α = 0.3 and γ = 0.99.

# %% [markdown] role="ex6-predict"
# A random policy balances the pole for about 20 steps. After your training budget, will the
# greedy policy's return be below 3 times random, around it, or far above? Will ‖W‖ keep
# growing all run, or level off?


# %% role="ex6-stub"
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):
    """One semi-gradient Q-learning update for linear action values (linear Q-learning).

    W: (n_actions, d) weights, Q(s, b) = W[b] @ phi(s). phi, phi_next: (d,) features of s and
    s'. a: the action taken (int). r, gamma, alpha: floats. terminated, truncated: bools from
    env.step. Returns the new W, a new array; only row a changes.
    """
    # TODO 6: the target (which flag stops the bootstrap?), the TD error, then row a.
    raise NotImplementedError(
        "TODO 6: write linear_q_update, or run lab.use_reference(6) to continue with the solution"
    )


# %% [markdown] role="ex6-hint"
# `W @ phi_next` gives Q(s', b) for every b at once. The gradient of W[a] @ phi with respect
# to W[a] is phi, and with respect to every other row it is 0. One of the two flags does not
# enter the target at all. Start from `W.copy()`.


# %% role="ex6-sol"
@lab.solution(6)
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):
    """W[a] += alpha * delta * phi; bootstrap unless terminated."""
    bootstrap = 0.0 if terminated else gamma * np.max(W @ phi_next)
    delta = r + bootstrap - W[a] @ phi  # TD error; truncated still bootstraps
    W_new = W.copy()
    W_new[a] = W[a] + alpha * delta * phi  # (linear Q-learning)
    return W_new


# %% role="ex6-run"
OBS_SCALE = np.array([2.4, 3.0, 0.2095, 3.5])  # cart x and pole angle: where CartPole ends
CENTERS = np.linspace(-1.0, 1.0, 5)  # five bumps per scaled number
WIDTH = CENTERS[1] - CENTERS[0]  # sigma = the spacing, 0.5
GRID_POINTS = np.stack(np.meshgrid(*[CENTERS] * 4, indexing="ij"), axis=-1).reshape(-1, 4)
ALPHA_Q, GAMMA_Q, EPS_END = 0.3, 0.99, 0.05


def rbf_raw(obs):
    """Provided: 625 Gaussian bumps on a 5^4 grid of the scaled observation, each in [0, 1]."""
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    per_dim = np.exp(-((z[:, None] - CENTERS) ** 2) / (2 * WIDTH**2))  # (4, 5)
    return np.einsum("i,j,k,l->ijkl", *per_dim).ravel()


def rbf_features(obs):
    """Provided: the bumps scaled to unit length (RBF features)."""
    f = rbf_raw(obs)
    return f / np.linalg.norm(f)


def train_linear_q(update, featurize, steps, eps_steps, seed, log_every=1000):
    """Provided: epsilon-greedy linear Q-learning on CartPole-v1 with your update.

    Returns (W, log). Every `log_every` steps the log records the step, ‖W‖ and the mean
    return of the last 10 finished episodes; it also keeps every episode's return."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng([seed, 7])
    obs, _ = env.reset(seed=seed)
    phi = featurize(obs)
    W = np.zeros((env.action_space.n, len(phi)))
    log = {"step": [], "w_norm": [], "recent_return": [], "returns": []}
    episode_return, start, shown = 0.0, time.perf_counter(), 0
    for t in range(steps):
        epsilon = max(EPS_END, 1.0 - (1.0 - EPS_END) * t / eps_steps)
        with np.errstate(over="ignore", invalid="ignore"):
            q = W @ phi
        explore = rng.random() < epsilon or not np.all(np.isfinite(q))
        a = int(rng.integers(len(W))) if explore else int(np.argmax(q))
        obs, r, terminated, truncated, _ = env.step(a)
        phi_next = featurize(obs)
        with np.errstate(over="ignore", invalid="ignore"):
            W = update(W, phi, a, r, phi_next, terminated, truncated, GAMMA_Q, ALPHA_Q)
        episode_return += r
        if terminated or truncated:
            log["returns"].append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
            phi_next = featurize(obs)
        phi = phi_next
        if (t + 1) % log_every == 0:
            recent = log["returns"][-10:]
            log["step"].append(t + 1)
            log["w_norm"].append(float(np.linalg.norm(W)))
            log["recent_return"].append(float(np.mean(recent)) if recent else np.nan)
            if time.perf_counter() - start > 10 * (shown + 1):  # a line about every 10 s
                shown += 1
                print(f"  step {t + 1:6d}: last 10 returns {log['recent_return'][-1]:6.1f}")
    env.close()
    return W, log


def greedy_return(W, featurize, seed, episodes):
    """Provided: the mean return of the greedy policy over `episodes` seeded episodes."""
    env = gym.make("CartPole-v1")
    total = 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        done = False
        while not done:
            with np.errstate(over="ignore", invalid="ignore"):
                q = W @ featurize(obs)
            a = int(np.argmax(q)) if np.all(np.isfinite(q)) else 0
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    env.close()
    return total / episodes


def random_return(seed, episodes):
    """Provided: the mean return of uniform random actions on the same start states."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng([seed, 8])
    total = 0.0
    for i in range(episodes):
        env.reset(seed=seed + 10_000 + i)
        done = False
        while not done:
            _, r, terminated, truncated, _ = env.step(int(rng.integers(2)))
            total += r
            done = terminated or truncated
    env.close()
    return total / episodes


print(f"Training your linear Q on seed {RUN_SEED}: {budget.steps} steps", flush=True)
start = time.perf_counter()
W, qlog = train_linear_q(linear_q_update, rbf_features, **TRAIN, seed=RUN_SEED)
train_s = time.perf_counter() - start
score = greedy_return(W, rbf_features, RUN_SEED, budget.eval_episodes)
baseline = random_return(RUN_SEED, budget.eval_episodes)
print(
    f"Done in {train_s:.1f} s: {len(qlog['returns'])} episodes; ‖W‖ at the end {qlog['w_norm'][-1]:.1f}"
)
start_obs, _ = gym.make("CartPole-v1").reset(seed=RUN_SEED)
print(f"Q at your first start state: {np.round(W @ rbf_features(start_obs), 2).tolist()}")
print(f"Greedy return over {budget.eval_episodes} episodes: {score:.1f}")
print(f"Random actions on the same start states: {baseline:.1f}; ratio {score / baseline:.1f}")
used = Budget(steps=budget.steps, seeds=1, wall_clock_s=train_s, hardware=HARDWARE)
print(textwrap.fill(used.sentence() + " Single seed, not a ranking.", 98))
lab.metric("greedy_return", round(score, 1))  # room code
lab.metric("random_return", round(baseline, 1))  # room code
lab.metric("w_norm_end", round(qlog["w_norm"][-1], 1))  # room code
lab.metric("q_log", {k: qlog[k] for k in ("step", "w_norm", "recent_return")})
# Your seed against the recorded band: linear Q with these features, step size and budget.
band = prl.data.band("m07_linear_q", budget)
fig, (ax_r, ax_n) = plt.subplots(1, 2, figsize=(10, 3.8))
prl.plot.with_recorded(
    band,
    qlog["recent_return"],
    x=qlog["step"],
    your_budget=budget,
    ax=ax_r,
    label=f"your seed ({RUN_SEED})",
    xlabel="environment steps",
    ylabel="mean return, last 10 episodes",
    title="Linear Q on CartPole-v1",
)
ax_n.plot(qlog["step"], qlog["w_norm"], color=prl.plot.SERIES[0])
ax_n.set(xlabel="environment steps", ylabel="‖W‖", title="Weight norm, your seed")
plt.show()
inside = band["runs"].min(axis=0)[-1] <= qlog["recent_return"][-1] <= band["runs"].max(axis=0)[-1]
where = "inside" if inside else "outside"
print(f"At the last step your seed is {where} the recorded {len(band['seeds'])}-seed range.")

# %% [markdown] role="ex6-explain"
# Q-learning with linear features has all three legs of the triad: features, a bootstrapped
# target, and a greedy target policy learned from ε-greedy behavior. Baird shows such a
# combination can diverge, and here it did not: what about this run's features and step
# size kept the weights bounded (linear Q-learning)? Is that a guarantee for another seed or
# another α? Your seed is one run: is landing outside the recorded band evidence that your code
# is wrong?

# %% role="ex6-chk1"
lab.check(6, m07.check_linear_q_update, linear_q_update, label="6.1")

# %% role="ex6-chk2"
lab.check(6, m07.check_linear_q_training, linear_q_update, label="6.2")

# %% [markdown] role="ex7-head"
# **Planted bug.** A colleague wrote a faster featurizer for the same 625 bumps: it measures
# each grid point's distance to the scaled observation in one array operation. They removed
# the last scaling step, because "each bump is already between 0 and 1". It runs without a
# warning. Train with it at the same α = 0.3 and read the weight-norm log, then find the
# problem and fix the function.

# %% [markdown] role="ex7-predict"
# With the colleague's features and the same step size, will the log show ‖W‖ about the same
# as in Exercise 6, slowly growing, or growing by many orders of magnitude?


# %% role="ex7-stub"
def colleague_features(obs):
    """A colleague's RBF featurizer for linear Q: the same 625 bumps. Is it right?"""
    # TODO 7: once the weight-norm log shows the problem, fix this function, rerun this
    # cell and the run cell.
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    sq_dist = ((GRID_POINTS - z) ** 2).sum(axis=1)  # one squared distance per grid point
    return np.exp(-sq_dist / (2 * WIDTH**2))  # each bump is in [0, 1]: no scaling needed


# %% [markdown] role="ex7-hint"
# The run printed ‖φ(s)‖² for the start state. For one state whose target is held fixed, an
# update changes Q(s, a) by α δ ‖φ(s)‖², so the error becomes (1 − α‖φ(s)‖²) δ (step-size
# rule). What is that factor here, and what must ‖φ(s)‖ be for Exercise 6's α to work?


# %% role="ex7-sol"
@lab.solution(7)
def colleague_features(obs):
    """The colleague's featurizer, fixed: scaled to unit length."""
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    sq_dist = ((GRID_POINTS - z) ** 2).sum(axis=1)  # one squared distance per grid point
    bumps = np.exp(-sq_dist / (2 * WIDTH**2))
    return bumps / np.linalg.norm(bumps)  # (step-size rule): unit length


# %% role="ex7-run"
upright = colleague_features(np.zeros(4))
sq = float(upright @ upright)
print(f"‖φ(s)‖² at the upright start state: {sq:.4f}; α ‖φ(s)‖² = {ALPHA_Q * sq:.4f}")
print(f"Training on seed {RUN_SEED} with colleague_features:", flush=True)
start = time.perf_counter()
W_c, clog = train_linear_q(linear_q_update, colleague_features, **TRAIN, seed=RUN_SEED)
suspect_s = time.perf_counter() - start
score_c = greedy_return(W_c, colleague_features, RUN_SEED, budget.eval_episodes)
print("Weight-norm log (‖W‖ every 1000 steps; inf or nan means it overflowed):")
for row in range(0, len(clog["w_norm"]), 8):  # eight entries per line
    print("  " + " ".join(f"{x:9.3g}" for x in clog["w_norm"][row : row + 8]))
start_obs, _ = gym.make("CartPole-v1").reset(seed=RUN_SEED)
with np.errstate(over="ignore", invalid="ignore"):
    q_start = W_c @ colleague_features(start_obs)
print(f"Q at your first start state: {np.array2string(q_start, precision=3)}")
print(f"  (no discounted return of CartPole exceeds 1 / (1 - γ) = {1 / (1 - GAMMA_Q):.0f})")
print(f"Greedy return: {score_c:.1f} (Exercise 6: {score:.1f}); {suspect_s:.1f} s")
finite = [x for x in clog["w_norm"] if np.isfinite(x)]
lab.metric("colleague_log10_max_norm", round(float(np.log10(max(finite))), 1) if finite else None)
fig, ax = plt.subplots()
ax.plot(qlog["step"], qlog["w_norm"], color=prl.plot.SERIES[0], label="rbf_features (Exercise 6)")
ok = np.isfinite(clog["w_norm"])
steps_c = np.array(clog["step"])[ok]
ax.plot(steps_c, np.array(clog["w_norm"])[ok], ls="--", color=prl.plot.SERIES[1], label="colleague")
ax.set(yscale="log", xlabel="environment steps", ylabel="‖W‖ (log scale)", title="Weight norms")
ax.legend()
plt.show()

# %% [markdown] role="ex7-explain"
# Where in the log did ‖W‖ leave Exercise 6's range, and how fast did it grow after that?
# Each bump is between 0 and 1, yet ‖φ(s)‖² was close to 10: why does the length of the whole
# vector matter, not the range of each entry (step-size rule)? On some seeds the colleague's
# greedy return is still high: is that evidence that its Q-values mean anything? Besides
# unit-length features, what else would have fixed it? (Gotcha: "Features are just
# preprocessing.")
#
# **Diagnostics for this lab.** Log ‖w‖ (or ‖W‖) every few thousand steps, next to the
# returns. Healthy: it rises while Q grows toward the size of a discounted return (at most
# 1/(1 − γ) = 100 here), then levels off, in the hundreds for this lab. Broken: it jumps by
# orders of magnitude and keeps climbing, on a log scale far above the healthy curve, until it
# may overflow to inf or nan. The greedy return is not a reliable alarm: it can collapse, or
# stay high on a lucky seed. Print Q at a start state too; far above 100, it estimates no
# return. Before blaming the algorithm, print ‖φ(s)‖² on a few states and multiply it by α:
# above 2, a single state's error grows with each update. On Baird, ‖w‖ grows by a constant
# factor per sweep with ‖φ(s)‖² small and α tiny: there the cause is the triad, and no step
# size fixes it.

# %% role="ex7-chk1"
lab.check(7, m07.check_colleague_features, colleague_features, label="7.1")

# %% role="ex7-chk2"
lab.check(7, m07.check_colleague_training, linear_q_update, colleague_features, label="7.2")

# %% [markdown]
# **Your room code.** The last cell, `lab.finish()`, prints one line that starts with
# `ROOM m07`. It holds your seat, the checkpoints you passed, whether QUICK budgets ran, and the
# numbers the debrief pools: your greedy return on CartPole (`greedy_return`, Exercise 6),
# random play's return on the same start states (`random_return`), ‖W‖ at the end of training
# (`w_norm_end`), and the base-10 logarithm of the largest finite ‖W‖ the colleague's features
# reached (`colleague_log10_max_norm`, Exercise 7). Baird's ‖w‖ after 1000 sweeps
# (`baird_norm_1000`) is the same for everyone: it is exact. Paste the line where your
# instructor says.

# %% [markdown] role="stretch"
# **TDC on Baird.** Gradient-TD methods do follow the gradient of an objective, the projected
# Bellman error (PBE), and they stay stable off-policy. TDC keeps a second weight vector v,
# which estimates E[φφᵀ]⁻¹ E[ρ δ φ] (Sutton and Barto, Eq. 11.28), and corrects the
# semi-gradient update with it (Sutton and Barto, §11.7). In expected form, with your
# matrices: e = b − Aw, then w ← w + α(e − γ (P_πΦ)ᵀ D Φ v) and v ← v + β(e − ΦᵀDΦ v). Write it
# and run Sutton and Barto's Figure 11.5 setting: α = 0.005, β = 0.05, 1000 sweeps and more.
# Watch √PBE and the value error √VE separately.


# %% role="stretch-stub"
def tdc_expected_update(w, v, phi, mu, P, r, gamma, alpha, beta):
    """One expected TDC sweep. Returns (new w, new v), arrays of shape (d,)."""
    # Stretch: e = phi.T @ (mu * (r + gamma * P @ phi @ w - phi @ w)); then the two updates.
    raise NotImplementedError(
        "Stretch: write tdc_expected_update, or run lab.use_reference('stretch') to see the solution"
    )


# %% role="stretch-sol"
@lab.solution("stretch")
def tdc_expected_update(w, v, phi, mu, P, r, gamma, alpha, beta):
    """Expected TDC: a semi-gradient step plus the gradient correction."""
    e = phi.T @ (mu * (r + gamma * P @ phi @ w - phi @ w))  # b - A w
    cross = (P @ phi).T @ (mu[:, None] * phi)  # E[rho phi(s') phi(s)^T]
    gram = phi.T @ (mu[:, None] * phi)  # E[phi phi^T]
    return w + alpha * (e - gamma * cross @ v), v + beta * (e - gram @ v)


# %% role="stretch-run"
lab.check("stretch", m07.check_tdc_expected_update, tdc_expected_update, label="stretch")
gram = baird.phi.T @ (baird.mu_behavior[:, None] * baird.phi)
gram_pinv = np.linalg.pinv(gram)  # Phi^T D Phi is singular here: 8 weights, 7 states


def sqrt_pbe(w):
    """Provided: sqrt(PBE) = sqrt((b - Aw)^T (Phi^T D Phi)^+ (b - Aw)), Sutton and Barto Eq. 11.26."""
    v = baird.phi @ w
    delta = baird.rewards + baird.gamma * baird.P_target @ v - v
    e = baird.phi.T @ (baird.mu_behavior * delta)  # b - A w
    return float(np.sqrt(e @ gram_pinv @ e))


def sqrt_ve(w):
    """Provided: the mu-weighted value error against v_pi = 0."""
    return float(np.sqrt(baird.mu_behavior @ (baird.phi @ w) ** 2))


w_t, v_t = baird.w0.copy(), np.zeros(8)
for k in range(1, 100_001):
    w_t, v_t = tdc_expected_update(w_t, v_t, **base, gamma=baird.gamma, alpha=0.005, beta=0.05)
    if k in (100, 1000, 10_000, 100_000):
        print(
            f"sweep {k:6d}: √PBE {sqrt_pbe(w_t):8.4f}  √VE {sqrt_ve(w_t):6.3f}  ‖w‖ {np.linalg.norm(w_t):6.3f}"
        )
print("All seven values are 0 only when w is proportional to (1, 1, 1, 1, 1, 1, 4, -2).")
print(f"TDC's w after 100,000 sweeps: {np.round(w_t, 3).tolist()}")
