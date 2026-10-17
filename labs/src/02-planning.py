# ---
# jupyter:
#   prl:
#     module: m02
# ---

# %% [markdown]
# In this lab you find the best policy for the shop of Lab 1 in two ways, value iteration
# and policy iteration, and check that they agree. Then you take the model away: you
# estimate P and R from samples, plan in the estimate, and measure in the true model what
# the estimate cost you, as the data shrinks. You finish by finding a bug in someone else's
# value iteration from its own log.
#
# The briefing's equations are named in brackets, for example (VI update) or (greedy loss).
# Its table "Equations and where you implement them" gives each name's equation number.

# %%
import matplotlib.pyplot as plt
import numpy as np

from prl import envs
from prl.checks import m02

# The shop of Lab 1, with the same numbers (the briefing uses them too).
INVENTORY = dict(
    capacity=10,
    demand_mean=4.0,
    max_demand=10,
    price=4.0,
    unit_cost=2.0,
    fixed_cost=5.0,
    holding_cost=0.2,
    lost_sales_penalty=1.0,
    gamma=0.95,
)
# Room pooling: SEED is your seat number (set in the setup cell), so every seat samples
# different data in Exercises 3 and 4 and the room's results can be pooled.
SEEDS = [SEED * 1000 + i for i in range(5)]
lab.seeds(SEEDS)

# %% [markdown] role="ex0-head"
# Welcome back. The briefing stopped at the bridge: with P and R you can plan; without
# them you estimate P and R first, and the simulation lemma says what that costs. If the
# runtime restarted over lunch, rerun the setup cell at the top, then run the cell below. It
# rebuilds the shop's arrays from `prl.envs` (the reference version of your Lab 1 arrays)
# and validates them, so this notebook does not depend on your Lab 1 notebook.

# %% role="ex0-run"
mdp = envs.InventoryMDP(**INVENTORY)
mdp.validate()
P, R, gamma = mdp.P, mdp.R, INVENTORY["gamma"]
S, A = R.shape
print(f"shop: {S} states (stock 0..{S - 1}), {A} actions (order 0..{A - 1}), gamma = {gamma}")
print("P:", P.shape, " R:", R.shape, " validate(): rows of P sum to 1")
print("expected one-day reward lies in", (round(float(R.min()), 2), round(float(R.max()), 2)))

# %% [markdown]
# Policy iteration's evaluation step is Lab 1's exact evaluator. The next cell is the
# reference version of your Lab 1 code, followed by a helper that turns a deterministic
# policy (one action per state) into the (S, A) matrix the evaluator expects.

# %% include labs/shared/evaluate_exact.py


# %%
def policy_matrix(actions, n_actions):
    """(S, A) action probabilities for a deterministic policy given as one action per state."""
    actions = np.asarray(actions, dtype=int)
    pi = np.zeros((len(actions), n_actions))
    pi[np.arange(len(actions)), actions] = 1.0
    return pi


# %% [markdown] role="ex1-head"
# Value iteration sweeps the optimality backup from V = 0: Q_k(s, a) = R[s, a] +
# γ Σ_s' P[s, a, s'] V_k(s') and V_{k+1}(s) = max_a Q_k(s, a) (VI update). It stops when
# max_s |V_{k+1}(s) − V_k(s)| < tol (VI stopping rule). The greedy policy reads the best
# action off Q (greedy): π_Q(s) = argmax_a Q(s, a). Return the Q computed from the final V,
# so that `greedy(Q)` is the policy your sweeps produced.

# %% [markdown] role="ex1-predict"
# Lab 1's "order up to 7" orders from every stock level below 7. Guess the shape of the
# optimal policy: from which stock levels does it order, and up to what level? (The fixed
# cost of 5 per delivery is the clue.)


# %% role="ex1-stub"
def value_iteration(P, R, gamma, tol):
    """Sweep V <- max_a [R + gamma P V] from V = 0 until max|V_new - V| < tol.

    Build all of V_new from the old V. Returns (V, Q, k): the final V, the Q computed from
    it (Q = R + gamma P V, so greedy(Q) is the policy), and the number of sweeps k.
    """
    # TODO 1 (part 1): Q from V is R + gamma * P @ V, shape (S, A); V_new is its max over
    # actions. Count one sweep per backup, and test the stopping rule after each.
    raise NotImplementedError(
        "TODO 1: write value_iteration, or run lab.use_reference(1) to continue with the solution"
    )


def greedy(Q):
    """The greedy policy for Q: the action with the largest Q[s, a] in each state, as an int array (S,)."""
    # TODO 1 (part 2): one argmax per state (which axis?).
    raise NotImplementedError(
        "TODO 1: write greedy, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# `P @ V` has shape (S, A): NumPy treats the first axis of `P` as a batch of (A, S)
# matrices. The max over actions is `Q.max(axis=1)` and the greedy action `Q.argmax(axis=1)`.
# Keep the old `V` until the test has run, then set `V = V_new`. The Q you return is built
# from the V you return.


# %% role="ex1-sol"
@lab.solution(1)
def value_iteration(P, R, gamma, tol):
    """Sweep V <- max_a [R + gamma P V] from V = 0 until max|V_new - V| < tol."""
    V, k = np.zeros(P.shape[0]), 0
    while True:
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        k += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, R + gamma * P @ V_new, k
        V = V_new


@lab.solution(1)
def greedy(Q):
    """The greedy policy for Q: argmax over actions in each state."""
    return np.argmax(Q, axis=1)


# %% role="ex1-run"
tol = 1e-6
V_star, Q_star, k_vi = value_iteration(P, R, gamma, tol)
pi_star = greedy(Q_star)
print(f"value iteration: {k_vi} sweeps to tol = {tol:g}")
print("V*(stock):      ", np.round(V_star, 2))
print("pi*(stock):     order", pi_star, " -> shelf after ordering", pi_star + np.arange(S))
pi7 = np.maximum(0, 7 - np.arange(S))
V7 = evaluate_exact(P, R, gamma, policy_matrix(pi7, A))
print(
    f"'order up to 7' loses up to {np.max(V_star - V7):.2f} against V* (at stock {np.argmax(V_star - V7)})"
)
fig, ax = plt.subplots()
ax.plot(range(S), V_star, marker="o", label="V* (greedy policy of your sweeps)")
ax.plot(range(S), V7, marker="s", label='Lab 1: "order up to 7"')
ax.set(xlabel="stock", ylabel="value", title="The best policy against Lab 1's policy")
ax.legend()
plt.show()
# The greedy-loss bound (greedy loss) on a perturbed Q: ||Q_noisy - Q*|| <= eps + the error
# left in Q_star by the stopping rule, so the bound is 2 (eps + gamma tol / (1 - gamma)) / (1 - gamma).
rng = np.random.default_rng(SEED)
print("Greedy policy of Q* + noise in [-eps, eps]: worst true loss over 20 draws, and the bound")
for eps in (0.5, 1.0, 2.0, 5.0):
    worst = 0.0
    for _ in range(20):
        Q_noisy = Q_star + rng.uniform(-eps, eps, size=Q_star.shape)
        V_noisy = evaluate_exact(P, R, gamma, policy_matrix(greedy(Q_noisy), A))
        worst = max(worst, float(np.max(V_star - V_noisy)))
    bound = 2 * (eps + gamma * tol / (1 - gamma)) / (1 - gamma)
    print(f"  eps = {eps:3.1f}: worst loss {worst:6.2f}   bound {bound:6.1f}")

# %% [markdown] role="ex1-explain"
# The optimal policy orders from stocks 0 to 3 only, and each order fills the shelf: fewer,
# larger deliveries, because every delivery costs a fixed 5. Lab 1's policy pays that fixed
# cost almost every day. Now run `value_iteration(P, R, gamma, 1.0)` and read off its policy:
# how many sweeps did it take, and is the policy already π*? The values are still far from
# V*, yet the policy is right. The bound (greedy loss) is what you can *guarantee* from a
# value error; read the table: it holds at every eps, and it is loose. What is the guarantee
# for the policy you just extracted at tol = 1e−6? (Use ‖Q − Q*‖ ≤ γ · tol / (1 − γ).)

# %% role="ex1-chk1"
lab.check(1, m02.check_value_iteration, value_iteration, label="1.1")

# %% role="ex1-chk2"
lab.check(1, m02.check_greedy, greedy, label="1.2")

# %% [markdown] role="ex2-head"
# Policy iteration alternates two steps from a starting policy: evaluate it exactly (Lab 1's
# linear solve, included above), then improve it by acting greedily with respect to its Q
# (improvement). It stops when the greedy policy is the one it already has. Ties: when the
# current action is among the best, keep it; otherwise the loop could alternate between two
# equally good policies forever.

# %% [markdown] role="ex2-predict"
# Start from "never order" (action 0 in every state). How many evaluations, that is linear
# solves, before the policy stops changing: about 2, about 4, or about 40? Your value
# iteration took the number of sweeps printed in Exercise 1.


# %% role="ex2-stub"
def policy_iteration(P, R, gamma, pi0):
    """Alternate exact evaluation and greedy improvement from the deterministic policy pi0.

    pi0 is an int array (S,). Returns (V, pi, k): the value and the policy at the end, and
    the number of evaluations k. Keep the current action when it ties for the best.
    """
    S, A = R.shape
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while True:
        V = evaluate_exact(P, R, gamma, policy_matrix(pi, A))
        k += 1
        # TODO 2: Q from V, then the improved policy: greedy(Q), except keep pi[s] where
        # Q[s, pi[s]] already equals the row's max. Return (V, pi, k) once nothing changes.
        raise NotImplementedError(
            "TODO 2: write policy_iteration, or run lab.use_reference(2) to continue with the solution"
        )


# %% [markdown] role="ex2-hint"
# `Q[np.arange(S), pi]` picks each state's current action value. `np.where(keep, pi,
# greedy(Q))` keeps the current action where `keep` is true. Compare the new and the old
# policy with `np.array_equal`.


# %% role="ex2-sol"
@lab.solution(2)
def policy_iteration(P, R, gamma, pi0):
    """Alternate exact evaluation and greedy improvement from the deterministic policy pi0."""
    S, A = R.shape
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while True:
        V = evaluate_exact(P, R, gamma, policy_matrix(pi, A))
        k += 1
        Q = R + gamma * P @ V
        keep = Q[np.arange(S), pi] >= Q.max(axis=1)
        pi_new = np.where(keep, pi, greedy(Q))
        if np.array_equal(pi_new, pi):
            return V, pi, k
        pi = pi_new


# %% role="ex2-run"
V_pi, pi_pi, k_pi = policy_iteration(P, R, gamma, np.zeros(S, dtype=int))
print(f"policy iteration: {k_pi} evaluations (linear solves); value iteration: {k_vi} sweeps")
print("same policy as value iteration:", np.array_equal(pi_pi, pi_star))
print(
    f"max |V_PI - V_VI| = {np.max(np.abs(V_pi - V_star)):.2e}  (the stopping rule allows {gamma * tol / (1 - gamma):.2e})"
)
print("from 'order up to 7':", policy_iteration(P, R, gamma, pi7)[2], "evaluations")
print(
    "work per iteration, counting multiply-adds: solve ~ S^3 =",
    S**3,
    "; one sweep ~ S^2 A =",
    S * S * A,
)

# %% [markdown] role="ex2-explain"
# Four evaluations against a few hundred sweeps, and the same policy. Each evaluation is a
# linear solve, about S³ multiply-adds, where a sweep costs about S²A: for the shop both are
# tiny. Now imagine S = 10⁶ states and A = 11: how many multiply-adds is one solve, and how
# many is one sweep? Which method can you still afford? (Gotcha: "PI is always faster".)
# Why does keeping the current action on ties matter for termination?

# %% role="ex2-chk"
lab.check(2, m02.check_policy_iteration, policy_iteration)

# %% [markdown] role="ex3-head"
# Now suppose nobody gave you P and R. You have a log: 200 days of the shop under a
# colleague's random ordering, each row (stock, order, reward, next stock). The counts
# estimator (counts) is the model you can build from it: P̂(s' | s, a) is the fraction of the
# visits to (s, a) that went on to s', and R̂[s, a] the mean reward seen there. A pair the log
# never visited gets a self-loop, P̂(s | s, a) = 1, with R̂ = 0: a default, not knowledge.
# The cell below makes the log.


# %%
def random_log(mdp, days, rng):
    """(days, 4) rows (stock, order, reward, next stock) under uniformly random orders from stock 0."""
    rows, s = [], 0
    for _ in range(days):
        a = int(rng.integers(0, mdp.n_actions))
        s2, r, _ = mdp.sample_step(s, a, rng)
        rows.append((s, a, r, s2))
        s = s2
    return np.array(rows, dtype=float)


# %% [markdown] role="ex3-predict"
# 200 days of random orders over 121 (stock, order) pairs: will the log miss some pairs, and
# roughly how many? For a missed pair the rule gives P̂(s | s, a) = 1 and R̂ = 0. If the
# planner's V̂(s) is positive, will it ever choose that action? (Compare Q̂(s, a) = 0 + γ V̂(s)
# with V̂(s).)


# %% role="ex3-stub"
def estimate_model(log, n_states, n_actions):
    """Counts -> (P_hat, R_hat, counts) from rows (s, a, r, s2).

    P_hat[s, a, :] is the fraction of the visits to (s, a) that went to each s2, and
    R_hat[s, a] the mean reward seen at (s, a). An unvisited pair gets P_hat[s, a, s] = 1
    and R_hat[s, a] = 0. counts[s, a] is the number of rows with that pair.
    """
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    # TODO 3: tally the rows (cast s, a, s2 to int), divide by the counts where they are
    # positive, and give every unvisited pair its self-loop.
    raise NotImplementedError(
        "TODO 3: write estimate_model, or run lab.use_reference(3) to continue with the solution"
    )


# %% [markdown] role="ex3-hint"
# One pass over the rows adds 1 to `counts[s, a]` and to `P_hat[s, a, s2]` and adds `r` to
# `R_hat[s, a]`. Then loop over (s, a): divide both by `counts[s, a]` if it is positive, else
# set `P_hat[s, a, s] = 1.0`.


# %% role="ex3-sol"
@lab.solution(3)
def estimate_model(log, n_states, n_actions):
    """Counts -> (P_hat, R_hat, counts) from rows (s, a, r, s2); unvisited pairs self-loop with reward 0."""
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    for s, a, r, s2 in log:
        s, a, s2 = int(s), int(a), int(s2)
        counts[s, a] += 1
        P_hat[s, a, s2] += 1
        R_hat[s, a] += r
    for s in range(n_states):
        for a in range(n_actions):
            if counts[s, a] > 0:
                P_hat[s, a] /= counts[s, a]
                R_hat[s, a] /= counts[s, a]
            else:
                P_hat[s, a, s] = 1.0
    return P_hat, R_hat, counts


# %% role="ex3-run"
log = random_log(mdp, 200, np.random.default_rng(SEED))
P_hat, R_hat, counts = estimate_model(log, S, A)
envs.TabularMDP(P_hat, R_hat, gamma).validate()
print(
    f"visited {int((counts > 0).sum())} of {S * A} (stock, order) pairs; most visits to one pair: {int(counts.max())}"
)
eps_P = np.max(np.abs(P - P_hat).sum(axis=2))
eps_R = np.max(np.abs(R - R_hat))
print(
    f"eps_P = max_(s,a) ||P - P_hat||_1 = {eps_P:.3f} (2 is the largest possible); eps_R = max |R - R_hat| = {eps_R:.2f}"
)
V_hat, Q_hat, _ = value_iteration(P_hat, R_hat, gamma, tol)
pi_hat = greedy(Q_hat)
V_hat_true = evaluate_exact(P, R, gamma, policy_matrix(pi_hat, A))
print("planned policy (order by stock):", pi_hat)
print(f"its TRUE loss, max_s V*(s) - V^pi_hat(s): {np.max(V_star - V_hat_true):.2f}")
fig, ax = plt.subplots()
im = ax.imshow(counts, origin="lower", aspect="auto")
ax.set(xlabel="order size", ylabel="stock", title="Visits per (stock, order) in the 200-day log")
plt.colorbar(im, label="count")
plt.show()

# %% [markdown] role="ex3-explain"
# Look at the heat map: which pairs are unvisited, and why those? (Where does random
# ordering from an empty shelf spend its days?) The estimator (counts) is exact for what it
# saw and silent about the rest; the self-loop is our choice, and the planner treats it as a
# fact. In the shop every value is positive, so Q̂(s, a) = γ V̂(s) < V̂(s) and an unvisited
# action is never chosen. In a world where values are negative (the cliff of Lab 1), the same
# default looks like a free safe harbor. Module 6 turns this default into a deliberate
# choice. Your planned policy's true loss is the number printed above; Exercise 4 measures
# how it shrinks with data.

# %% role="ex3-chk"
lab.check(3, m02.check_estimate_model, estimate_model)

# %% [markdown] role="ex4-head"
# The experiment of the briefing's section 4. The sampler below asks the shop for N
# transitions from every (stock, order) pair, so each P̂(· | s, a) is a mean of N draws and
# nothing is unvisited. You plan in the estimate and evaluate the plan in the truth, then
# repeat over N and over 5 seeds. The simulation lemma bounds the loss by a multiple of
# ε_P = max_(s,a) ‖P(· | s, a) − P̂(· | s, a)‖₁ (planning loss); the experiment measures it.


# %%
def sample_transitions(mdp, N, rng):
    """N sampled transitions from every (s, a), as (S * A * N, 4) rows (s, a, r, s2)."""
    rows = []
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            for _ in range(N):
                s2, r, _ = mdp.sample_step(s, a, rng)
                rows.append((s, a, r, s2))
    return np.array(rows, dtype=float)


# %% [markdown] role="ex4-predict"
# From N = 20 samples per pair to N = 10: does the mean loss double? Does the bound
# (planning loss) double? And at N = 200, will the planned policy still lose anything?


# %% role="ex4-stub"
def plan_and_evaluate(mdp, N, seed, tol=1e-6):
    """Plan in a model estimated from N samples per (s, a); evaluate the plan in the truth.

    Returns (loss, eps_P): loss = max_s [V*(s) - V^pi_hat(s)] with V^pi_hat computed in the
    TRUE model (mdp.P, mdp.R), and eps_P = max_(s,a) ||P(.|s,a) - P_hat(.|s,a)||_1.
    """
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)
    # TODO 4: estimate_model from the log; value_iteration and greedy in the estimate;
    # evaluate_exact of that policy in mdp.P, mdp.R. For V*, plan in the true model the same
    # way and evaluate ITS greedy policy exactly, so the loss is exactly 0 when the plans agree.
    raise NotImplementedError(
        "TODO 4: write plan_and_evaluate, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# Four calls you already have: `estimate_model(log, mdp.n_states, mdp.n_actions)`,
# `value_iteration(P_hat, R_hat, mdp.gamma, tol)`, `greedy(Q_hat)`, and
# `evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(pi_hat, mdp.n_actions))`. Do the
# same with `mdp.P, mdp.R` in place of the estimate to get π* and its exact value V*.
# `eps_P` is `np.abs(mdp.P - P_hat).sum(axis=2).max()`.


# %% role="ex4-sol"
@lab.solution(4)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):
    """Plan in a model estimated from N samples per (s, a); evaluate the plan in the truth."""
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)
    pi_hat = greedy(Q_hat)
    _, Q_true, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)
    V_star_true = evaluate_exact(
        mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_true), mdp.n_actions)
    )
    V_hat_true = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(pi_hat, mdp.n_actions))
    loss = float(np.max(V_star_true - V_hat_true))
    eps_P = float(np.abs(mdp.P - P_hat).sum(axis=2).max())
    return loss, eps_P


# %% role="ex4-run"
Ns = [1, 2, 5, 10, 20, 50, 100, 200]
losses = np.zeros((len(SEEDS), len(Ns)))
eps = np.zeros_like(losses)
print("Budget: 5 seeds per N; the shop's 121 pairs times N samples each; your seat's seeds.")
print("   N | true loss per seed (every seed shown)      | mean   | eps_P mean")
for j, N in enumerate(Ns):
    for i, seed in enumerate(SEEDS):
        losses[i, j], eps[i, j] = plan_and_evaluate(mdp, N, seed)
    print(
        f"{N:4d} | {np.array2string(losses[:, j], precision=2, floatmode='fixed')} | {losses[:, j].mean():6.2f} | {eps[:, j].mean():.3f}"
    )
lab.metric("loss_by_N", {str(N): losses[:, j].round(4).tolist() for j, N in enumerate(Ns)})
r_range = float(R.max() - R.min())
print(
    f"At N = {Ns[-1]}: the bound 2 gamma eps_P (r_max - r_min) / (1 - gamma)^2 = "
    f"{2 * gamma * eps[:, -1].mean() * r_range / (1 - gamma) ** 2:,.0f}, against a largest V* of {V_star.max():.0f}"
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for i, seed in enumerate(SEEDS):
    ax1.plot(Ns, losses[i], marker="o", alpha=0.6, label=f"seed {seed}")
ax1.plot(Ns, losses.mean(axis=0), color="black", lw=2.5, label="mean of 5 seeds")
ax1.set(
    xscale="log",
    xlabel="samples per (stock, order)",
    ylabel="true loss  max_s V*(s) - V^pi_hat(s)",
    title="Plan in P_hat, evaluate in P",
)
ax1.legend(fontsize=8)
ax2.plot(Ns, eps.mean(axis=0), marker="o", label="eps_P, mean of 5 seeds")
ax2.plot(
    Ns,
    eps[:, -1].mean() * np.sqrt(Ns[-1] / np.array(Ns)),
    ls="--",
    label="1 / sqrt(N), matched at N = 200",
)
ax2.set(
    xscale="log",
    yscale="log",
    xlabel="samples per (stock, order)",
    ylabel="eps_P",
    title="Model error",
)
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown] role="ex4-explain"
# Read your table. Did halving N from 20 to 10 double the mean loss? ε_P shrinks like
# 1 / √N (right panel), so the bound (planning loss) rises by a factor of about √2 when N
# halves, and the loss itself does whatever the policy does: above some N the planned
# policy is π* on most seeds and the loss is exactly 0, below it the loss jumps in steps.
# The printed bound is thousands against values near 100: it is a worst case over every
# MDP with that ε_P, with two factors of the horizon 1 / (1 − γ) = 20 in it. With five
# seeds, the comparison rule is: show every seed, and call a tie when the ranges overlap.
# Which pairs of N are ties in your table? (Gotcha: "95%-accurate model → 95%-as-good plan".)

# %% role="ex4-chk"
lab.check(4, m02.check_plan_and_evaluate, plan_and_evaluate)

# %% [markdown] role="ex5-head"
# **Planted bug.** A colleague's `value_iteration_suspect` logs max |V_new − V| at every
# sweep and returns finite, plausible values. Whether a value function solves the
# optimality equation can be tested without a reference answer: the optimality residual,
# max_s |max_a Q(s, a) − V(s)| with Q = R + γ P V, is zero only at V*, and value iteration
# that stops when max |ΔV| < tol returns a V with residual below γ · tol (VI stopping rule).
# The helper below computes it. Run the suspect on the shop, then on a cliff gridworld whose
# moves slip, with the residual as the diagnostic. Then fix the one wrong line.


# %%
def optimality_residual(P, R, gamma, V):
    """max over states of |max_a (R + gamma P V)[s, a] - V[s]|: zero exactly at V*."""
    return float(np.max(np.abs((R + gamma * P @ V).max(axis=1) - V)))


# %% [markdown] role="ex5-predict"
# A correct value iteration stopping at tol = 1e−6 returns a V whose optimality residual is
# below γ · tol = 9.5e−7 (why?). The suspect prints the last entry of its own log. If that
# number is above tol, what has happened?


# %% role="ex5-stub"
def value_iteration_suspect(P, R, gamma, tol):
    """A colleague's value iteration. Returns (V, Q, k, log): log[i] is max|V_new - V| at sweep i + 1."""
    V, k, log = np.zeros(P.shape[0]), 0, []
    while True:
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        k += 1
        change = np.abs(V_new - V)
        log.append(float(change.max()))
        # TODO 5: once the residual shows the problem, fix the one wrong line below, then
        # rerun this cell and the run cell.
        if change.mean() < tol:
            return V_new, R + gamma * P @ V_new, k, log
        V = V_new


# %% [markdown] role="ex5-hint"
# Compare the suspect's last logged max |ΔV| with tol, and its residual with γ · tol. The
# log records one statistic of the change; the `if` tests another. Which statistic does the
# stopping rule (VI stopping rule) need?


# %% role="ex5-sol"
@lab.solution(5)
def value_iteration_suspect(P, R, gamma, tol):
    """The colleague's value iteration, fixed: stop on the max change, not the mean."""
    V, k, log = np.zeros(P.shape[0]), 0, []
    while True:
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        k += 1
        change = np.abs(V_new - V)
        log.append(float(change.max()))
        if change.max() < tol:
            return V_new, R + gamma * P @ V_new, k, log
        V = V_new


# %% role="ex5-run"
tol = 1e-6
V_s, _, k_s, log_s = value_iteration_suspect(P, R, gamma, tol)
print(
    f"shop:  suspect stopped after {k_s} sweeps (yours: {k_vi}); last logged max|dV| = {log_s[-1]:.2e};"
    f" residual = {optimality_residual(P, R, gamma, V_s):.2e}"
)
cliff = envs.CliffGridworld(4, 12, slip=0.2, gamma=0.95)
V_c, _, k_c, log_c = value_iteration_suspect(cliff.P, cliff.R, cliff.gamma, tol)
V_ok, _, k_ok = value_iteration(cliff.P, cliff.R, cliff.gamma, tol)
print(
    f"cliff: suspect stopped after {k_c} sweeps (yours: {k_ok}); last logged max|dV| = {log_c[-1]:.2e}"
    f" (tol = {tol:.0e})"
)
print(
    f"       residual: suspect {optimality_residual(cliff.P, cliff.R, cliff.gamma, V_c):.2e},"
    f" yours {optimality_residual(cliff.P, cliff.R, cliff.gamma, V_ok):.2e}, guarantee gamma * tol = {cliff.gamma * tol:.2e}"
)
fig, ax = plt.subplots()
ax.semilogy(
    range(1, len(log_c) + 1), log_c, marker=".", label="suspect's logged max|dV| per sweep (cliff)"
)
ax.axhline(tol, ls="--", color="gray", label="tol")
ax.set(xlabel="sweep", ylabel="max |V_new - V|", title="Where the suspect stopped")
ax.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# Before your fix: on the shop the suspect stopped at the same sweep as your value
# iteration, with the same residual. On the cliff it stopped a few sweeps early, while its
# own log still read above tol, and its residual broke the guarantee γ · tol. Why does the
# shop hide the bug? (Near the end, every state's change is almost the same number, so the
# mean equals the max.) The mean is at least the max divided by the number of states, so the
# residual can reach S times γ · tol: on an MDP where most states have settled, that is the
# whole factor. (Gotcha: "Stop on the mean change".)
#
# **Diagnostics for this lab.** Every value function you compute has a residual you can
# check without a reference: the Bellman residual of Lab 1 for a policy's value, the
# optimality residual above for V*. Healthy value iteration: a residual below γ · tol, a
# logged max |ΔV| that falls by about a factor γ per sweep once past the first few, and a
# greedy policy that stops changing long before the values settle. Broken: a residual above
# γ · tol (the stopping statistic is wrong, as here); a sweep count that does not grow with
# 1 / (1 − γ) (the test never fires, or fires on the wrong quantity); a greedy policy that
# differs between value iteration and policy iteration (one of the two is wrong). For an
# estimated model, `validate()` checks that P̂ is a model, and ε_P and ε_R say how far it is
# from the truth when you have the truth to compare with; the true loss of the planned
# policy is the number that matters, and only the true model can give it.

# %% role="ex5-chk"
lab.check(5, m02.check_value_iteration_suspect, value_iteration_suspect)

# %% [markdown] role="stretch"
# **Modified policy iteration.** Policy iteration's solve costs about S³; value iteration's
# sweep about S²A, and it needs many sweeps. In between: evaluate the current greedy policy
# with m sweeps of T^π instead of a solve. With m = 1 this is value iteration; as m grows it
# approaches policy iteration. Write `modified_policy_iteration(P, R, gamma, m, tol)`: start
# from V = 0; each round takes Q = R + γ P V, π = greedy(Q), one optimality sweep V ← max_a Q,
# then m − 1 sweeps of T^π; stop when the round changed V by less than tol in the max norm.
# Return (V, pi, rounds) with pi the greedy policy of the final V.


# %% role="stretch-stub"
def modified_policy_iteration(P, R, gamma, m, tol):
    """Policy iteration whose evaluation step is m sweeps of T^pi; m = 1 is value iteration."""
    # Stretch: P[np.arange(S), pi] is P_pi (S, S) and R[np.arange(S), pi] is r_pi (S,).
    raise NotImplementedError(
        "Stretch: write modified_policy_iteration, or run lab.use_reference('stretch') to see the solution"
    )


# %% role="stretch-sol"
@lab.solution("stretch")
def modified_policy_iteration(P, R, gamma, m, tol):
    """Policy iteration whose evaluation step is m sweeps of T^pi; m = 1 is value iteration."""
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


# %% role="stretch-run"
for m in (1, 5, 20, 100):
    V_m, pi_m, rounds = modified_policy_iteration(P, R, gamma, m, 1e-6)
    print(
        f"m = {m:3d}: {rounds:4d} rounds = {rounds * m:5d} sweeps; policy equals pi*: {np.array_equal(pi_m, pi_star)};"
        f" max |V - V*| = {np.max(np.abs(V_m - V_star)):.1e}"
    )
lab.check(
    "stretch", m02.check_modified_policy_iteration, modified_policy_iteration, label="stretch"
)
