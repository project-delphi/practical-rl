# ---
# jupyter:
#   prl:
#     module: m01
# ---

# %% [markdown]
# In this lab you write two Markov decision processes (MDPs) as arrays, then evaluate a
# policy in them twice: exactly, with one linear solve, and by iteration, watching how the
# discount sets the number of sweeps. You finish by finding a bug in someone else's evaluator
# with a check that needs no reference answer: its Bellman residual.
#
# The briefing's equations are named in brackets, for example (return) or (exact). Its table
# "Equations and where you implement them" gives each name's equation number.

# %%
import matplotlib.pyplot as plt
import numpy as np

from prl import envs
from prl.checks import m01

# The inventory problem's numbers (the briefing uses the same ones).
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

# %% [markdown] role="ex1-head"
# The **return** is what almost every method this week estimates or maximizes: the discounted sum
# of the rewards from now on, G_0 = r_0 + γ r_1 + γ² r_2 + … (return).

# %% [markdown] role="ex1-predict"
# Rewards 1, 1, 1 with γ = 0.9: what is the return? Work it out before you code.


# %% role="ex1-stub"
def discounted_return(rewards, gamma):
    """Return G_0 = r_0 + gamma r_1 + gamma^2 r_2 + ... for a list of rewards."""
    # TODO 1: the first reward is not discounted; each later one is discounted once more.
    raise NotImplementedError(
        "TODO 1: write discounted_return, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# Use G_t = r_t + γ G_{t+1} (return). After the last reward G is 0. Which end of the list do
# you start from?


# %% role="ex1-sol"
@lab.solution(1)
def discounted_return(rewards, gamma):
    """Return G_0 = r_0 + gamma r_1 + gamma^2 r_2 + ... for a list of rewards."""
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


# %% role="ex1-run"
print(discounted_return([1, 1, 1], 0.9))
print(discounted_return([0, 0, 10], 0.5))

# %% [markdown] role="ex1-explain"
# Did you get 2.71 and 2.5? In the second case the only reward comes late: what does a small
# γ do to it? γ decides how much the future counts. It is part of the objective, not a speed
# setting (Gotcha: "γ is a speed knob").

# %% role="ex1-chk"
lab.check(1, m01.check_discounted_return, discounted_return)

# %% [markdown] role="ex2-head"
# The cliff gridworld (Sutton and Barto, Example 6.6). States are cells, numbered
# `row * cols + col` with row 0 at the top. Start bottom-left, goal bottom-right; the cells
# between them on the bottom row are the cliff. Actions: 0 up, 1 right, 2 down, 3 left.
#
# The rules, which your arrays must encode exactly:
#
# - every move costs −1; a move off the grid leaves you where you are;
# - moving into the cliff costs −100 and puts you back at the start;
# - the goal is absorbing: every action keeps you there with reward 0;
# - you can never be *in* a cliff cell; to keep the arrays simple, a cliff cell's row says
#   "go to the start, reward 0".
#
# The helpers below find a cell's index and where a move goes.

# %%
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def index(r, c, cols):
    """State number of cell (r, c)."""
    return r * cols + c


def move(r, c, a, rows, cols):
    """The cell action a leads to from (r, c); a move off the grid stays put."""
    dr, dc = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}[a]
    return min(max(r + dr, 0), rows - 1), min(max(c + dc, 0), cols - 1)


# %% [markdown] role="ex2-predict"
# In a 3 × 4 grid: how many states are there, and how many nonzero entries does each row
# `P[s, a, :]` have?


# %% role="ex2-stub"
def gridworld_arrays(rows, cols):
    """P[s, a, s'] and R[s, a] for the cliff gridworld (rules above). Returns (P, R)."""
    S, A = rows * cols, 4
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    start, goal = index(rows - 1, 0, cols), index(rows - 1, cols - 1, cols)
    cliff = [index(rows - 1, c, cols) for c in range(1, cols - 1)]
    for s in range(S):
        r, c = divmod(s, cols)
        for a in range(A):
            if s == goal:
                P[s, a, goal] = 1.0  # the goal absorbs, reward 0
            elif s in cliff:
                P[s, a, start] = 1.0  # never occupied: send it to the start, reward 0
            else:
                # TODO 2: where does action a lead? Into the cliff: back to the start with
                # reward -100. Anywhere else: that cell, with reward -1.
                raise NotImplementedError(
                    "TODO 2: fill P and R, or run lab.use_reference(2) to continue with the solution"
                )
    return P, R


# %% [markdown] role="ex2-hint"
# `s2 = index(*move(r, c, a, rows, cols), cols)` is the cell action `a` leads to. If `s2` is a
# cliff cell, the move goes to `start` instead and costs −100.


# %% role="ex2-sol"
@lab.solution(2)
def gridworld_arrays(rows, cols):
    """P[s, a, s'] and R[s, a] for the cliff gridworld (rules above). Returns (P, R)."""
    S, A = rows * cols, 4
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    start, goal = index(rows - 1, 0, cols), index(rows - 1, cols - 1, cols)
    cliff = [index(rows - 1, c, cols) for c in range(1, cols - 1)]
    for s in range(S):
        r, c = divmod(s, cols)
        for a in range(A):
            if s == goal:
                P[s, a, goal] = 1.0
            elif s in cliff:
                P[s, a, start] = 1.0
            else:
                s2 = index(*move(r, c, a, rows, cols), cols)
                if s2 in cliff:
                    P[s, a, start], R[s, a] = 1.0, -100.0
                else:
                    P[s, a, s2], R[s, a] = 1.0, -1.0
    return P, R


# %% role="ex2-run"
P_grid, R_grid = gridworld_arrays(3, 4)
print("P:", P_grid.shape, " R:", R_grid.shape)
print("Nonzeros per row of P:", np.unique((P_grid > 0).sum(axis=2)))
print("Reward of moving RIGHT from each cell:")
print(R_grid[:, RIGHT].reshape(3, 4))
terminal = np.zeros(12, dtype=bool)
terminal[index(2, 3, 4)] = True
envs.TabularMDP(P_grid, R_grid, 0.9, terminal=terminal, start=index(2, 0, 4)).validate()
print("validate(): rows sum to 1 and the goal absorbs.")

# %% [markdown] role="ex2-explain"
# Twelve states, one nonzero per row: this world is deterministic. Read the printed rewards:
# which cell's "move right" costs −100, and why do the goal and the two cliff cells show 0?
# `P_grid.sum(axis=2)` is all ones; `P_grid.sum(axis=0)` is not. Which axis is the next
# state? (Gotcha: "Rows or columns of P_π, it makes no difference.")

# %% role="ex2-chk"
lab.check(2, m01.check_gridworld_arrays, gridworld_arrays)

# %% [markdown] role="ex3a-head"
# **Inventory control, a real-shaped problem.** A shop stocks one product and can hold at
# most 10 units. Each morning you see the stock, place an order that arrives at once
# (capped so stock never exceeds capacity), then customers arrive: demand is random,
# Poisson with mean 4 (demand of 10 or more counts as 10). You sell what you can; unmet
# demand is lost. Money: you earn 4 per unit sold, pay 2 per unit delivered plus a fixed 5
# whenever any units are delivered (an order the cap cuts to 0 costs nothing),
# pay 0.2 per unit left on the shelf overnight, and take a penalty of 1 per unit of demand
# you cannot meet. The discount is γ = 0.95 per day.
#
# Before you write code, fill in the framing card below.

# %% [markdown]
# **Your framing card.** Double-click this cell to edit it.
#
# - State: …
# - Action: …
# - Reward: …
# - γ: …
# - One judgment call you made, and why: …

# %% [markdown] role="ex3a-predict"
# Stock 0, order 7, demand 4: what is the day's reward? Positive or negative?


# %% role="ex3a-stub"
def inventory_spec(capacity):
    """The framing card's counts: {'n_states': ..., 'n_actions': ...}."""
    # TODO 3a (part 1): the counts from your framing card.
    raise NotImplementedError(
        "TODO 3a: write inventory_spec, or run lab.use_reference('3a') to continue with the solution"
    )


def period_reward(stock, order, demand, p):
    """One day's reward: revenue - ordering cost - holding cost - lost-sales penalty.

    Orders are capped so that stock + order <= p['capacity'], and costs are charged on the
    capped order q = min(order, capacity - stock).
    """
    # TODO 3a (part 2): cap the order, sell min(stock + order, demand), then add up the money.
    raise NotImplementedError(
        "TODO 3a: write period_reward, or run lab.use_reference('3a') to continue with the solution"
    )


# %% [markdown] role="ex3a-hint"
# Cap first: `q = min(order, p["capacity"] - stock)`. Then count the units sold, left and
# lost. The fixed cost applies only if `q > 0`; the unit cost is paid on `q`. Check: stock 3,
# order 4, demand 5 gives 6.6.


# %% role="ex3a-sol"
@lab.solution("3a")
def inventory_spec(capacity):
    """The framing card's counts: {'n_states': ..., 'n_actions': ...}."""
    return {"n_states": capacity + 1, "n_actions": capacity + 1}


@lab.solution("3a")
def period_reward(stock, order, demand, p):
    """One day's reward: revenue - ordering cost - holding cost - lost-sales penalty."""
    q = min(order, p["capacity"] - stock)
    y = stock + q
    sales = min(y, demand)
    left = y - sales
    cost = (p["fixed_cost"] if q > 0 else 0.0) + p["unit_cost"] * q
    return (
        p["price"] * sales
        - cost
        - p["holding_cost"] * left
        - p["lost_sales_penalty"] * (demand - sales)
    )


# %% role="ex3a-run"
print(inventory_spec(10))
for stock, order, demand in [(0, 7, 4), (3, 0, 5), (6, 1, 2)]:
    print(
        f"stock {stock}, order {order}, demand {demand}: reward "
        f"{period_reward(stock, order, demand, INVENTORY):+.2f}"
    )

# %% [markdown] role="ex3a-explain"
# The first case loses 3.60, even though restocking an empty shelf pays off on later days:
# one day's reward is not the value of a decision. Demand is drawn fresh each day,
# independent of the past, so today's stock and order are all the future depends on: the
# state is Markov. Which line of your framing card is a judgment call rather than a fact?
# (Gotcha: "The reward is what I want.")

# %% role="ex3a-chk1"
lab.check("3a", m01.check_inventory_spec, inventory_spec, label="3a.1")

# %% role="ex3a-chk2"
lab.check("3a", m01.check_period_reward, period_reward, label="3a.2")

# %% [markdown] role="ex3b-head"
# Now turn the story into arrays. `pmf[d]` is the probability that demand is `d`
# (provided by `envs.demand_pmf`). R must be the **expected** reward of each (stock, order).

# %% [markdown] role="ex3b-predict"
# From stock 3, ordering 4: which next-morning stocks have nonzero probability?


# %% role="ex3b-stub"
def inventory_arrays(capacity, pmf, p):
    """P[s, a, s'] and expected R[s, a] for the inventory MDP. Returns (P, R)."""
    S = A = capacity + 1
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            for d, prob in enumerate(pmf):
                # TODO 3b: find tomorrow's stock after ordering a (capped) and selling to demand d;
                # add prob to P[s, a, tomorrow] and prob * period_reward(s, a, d, p) to R[s, a].
                raise NotImplementedError(
                    "TODO 3b: write inventory_arrays, or run lab.use_reference('3b') to continue "
                    "with the solution"
                )
    return P, R


# %% [markdown] role="ex3b-hint"
# Tomorrow's stock is `y - min(y, d)` with `y = s + min(a, capacity - s)`. Reuse your
# `period_reward` for the money.


# %% role="ex3b-sol"
@lab.solution("3b")
def inventory_arrays(capacity, pmf, p):
    """P[s, a, s'] and expected R[s, a] for the inventory MDP. Returns (P, R)."""
    S = A = capacity + 1
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            y = s + min(a, capacity - s)
            for d, prob in enumerate(pmf):
                P[s, a, y - min(y, d)] += prob
                R[s, a] += prob * period_reward(s, a, d, p)
    return P, R


# %% role="ex3b-run"
pmf = envs.demand_pmf(INVENTORY["demand_mean"], INVENTORY["max_demand"])
P_inv, R_inv = inventory_arrays(INVENTORY["capacity"], pmf, INVENTORY)
envs.TabularMDP(P_inv, R_inv, INVENTORY["gamma"]).validate()
print("From stock 3, ordering 4, next stock can be:", np.nonzero(P_inv[3, 4])[0])
print("Expected one-day reward from empty stock, by order size:")
print(np.round(R_inv[0], 2))
fig, ax = plt.subplots()
im = ax.imshow(R_inv, origin="lower", aspect="auto")
ax.set(xlabel="order size", ylabel="stock", title="Expected one-day reward R[s, a]")
plt.colorbar(im)
plt.show()

# %% [markdown] role="ex3b-explain"
# Starting empty, the best order for *today alone* is 4 units (−1.06), yet the shop must
# keep stock for tomorrow too. Why can't we pick orders by one-day reward? In Exercise 4,
# which term of (matrix form) carries tomorrow?

# %% role="ex3b-chk"
lab.check("3b", m01.check_inventory_arrays, inventory_arrays)

# %% [markdown] role="ex4-head"
# A **policy** π here is an (S, A) matrix of action probabilities. For a fixed policy the
# Bellman equation is linear, V = r_π + γ P_π V (matrix form), so one solve gives
# V^π = (I − γ P_π)⁻¹ r_π (exact). Below, the "order up to 7" policy orders max(0, 7 − stock).

# %% [markdown] role="ex4-predict"
# Under "order up to 7", how much lower is V(6) than V(7)? (After ordering, how many units
# does the shop open with?)


# %% role="ex4-stub"
def evaluate_exact(P, R, gamma, pi):
    """V^pi from (I - gamma P_pi) V = r_pi, where pi is an (S, A) matrix of probabilities."""
    # TODO 4: average P and R over actions with pi to get P_pi (S, S) and r_pi (S,), then solve.
    raise NotImplementedError(
        "TODO 4: write evaluate_exact, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# `r_pi[s]` is a row sum of `pi[s, a] * R[s, a]`. `P_pi[s, s2]` sums `pi[s, a] * P[s, a, s2]`
# over `a` (a loop over `a`, or `np.einsum`). Solve the system with `np.linalg.solve` rather
# than forming an inverse.


# %% role="ex4-sol"
@lab.solution(4)
def evaluate_exact(P, R, gamma, pi):
    """V^pi from (I - gamma P_pi) V = r_pi, where pi is an (S, A) matrix of probabilities."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


# %% role="ex4-run"
def base_stock_policy(level, capacity):
    """Order up to `level`: a deterministic policy as an (S, A) probability matrix."""
    S = capacity + 1
    pi = np.zeros((S, S))
    for s in range(S):
        pi[s, max(0, level - s)] = 1.0
    return pi


pi7 = base_stock_policy(7, INVENTORY["capacity"])
V7 = evaluate_exact(P_inv, R_inv, INVENTORY["gamma"], pi7)
print("V^pi by stock level:", np.round(V7, 2))
fig, ax = plt.subplots()
ax.plot(range(len(V7)), V7, marker="o")
ax.set(xlabel="stock", ylabel="V^pi(stock)", title='Value of "order up to 7"')
plt.show()
V_grid = evaluate_exact(P_grid, R_grid, 0.9, np.full((12, 4), 0.25))
print("3x4 gridworld, uniform random policy, gamma 0.9 (rows as on the grid):")
print(np.round(V_grid.reshape(3, 4), 2))

# %% [markdown] role="ex4-explain"
# From stock 6 and from stock 7 the shop opens with 7 units, so rows 6 and 7 of P_π are
# equal, and (matrix form) gives V(7) − V(6) = r_π(7) − r_π(6) = 5 + 2 = 7. Why do stocks 0
# to 6 differ by exactly 2? Is one unit at stock 6 really worth 7? Module 2 answers. (In the
# gridworld, random play from the start is very costly: why?)

# %% role="ex4-chk"
lab.check(4, m01.check_evaluate_exact, evaluate_exact)

# %% [markdown] role="ex5-head"
# Iterative evaluation applies T^π V = r_π + γ P_π V (operator) again and again from V = 0.
# Because T^π is a γ-contraction in the max norm (contraction; the briefing's ∞-norm), the
# error shrinks by at least a factor γ per sweep. From V = 0, after k sweeps the error is at
# most γ^k max_s |V^π(s)|, so reaching an error η takes about
# log(max_s |V^π(s)| / η) / log(1/γ) sweeps (iteration bound). While iterating you cannot
# measure the error, only the change: stopping when max |V_new − V| < tol guarantees an
# error of at most γ·tol / (1 − γ) (stopping rule).

# %% [markdown] role="ex5-predict"
# The briefing's demo counted sweeps on a two-state model. The shop has 11 states. At
# γ = 0.9 and tol = 1e−6, will it need tens, hundreds or thousands of sweeps? And from
# γ = 0.9 to 0.99, will its count grow by a bigger factor than the demo's, a smaller one, or
# about the same?


# %% role="ex5-stub"
def evaluate_iterative(P, R, gamma, pi, tol):
    """Repeat V <- r_pi + gamma P_pi V from V = 0 until max|V_new - V| < tol.

    Build all of V_new from the old V (not in place, state by state).
    Returns (V_new, k), where k counts the sweeps (applications of T^pi).
    """
    # TODO 5: build P_pi and r_pi once, then loop.
    raise NotImplementedError(
        "TODO 5: write evaluate_iterative, or run lab.use_reference(5) to continue with the solution"
    )


# %% [markdown] role="ex5-hint"
# Keep the old `V`, a `V_new` and a counter, and test the stopping rule after each sweep. If
# the cell never finishes, stop it with the stop button, then check that `V` moves forward.


# %% role="ex5-sol"
@lab.solution(5)
def evaluate_iterative(P, R, gamma, pi, tol):
    """Repeat V <- r_pi + gamma P_pi V from V = 0 until max|V_new - V| < tol."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    V, k = np.zeros(len(r_pi)), 0
    while True:
        V_new = r_pi + gamma * P_pi @ V
        k += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, k
        V = V_new


# %% role="ex5-run"
gammas = [0.5, 0.7, 0.9, 0.95, 0.99]
tol = 1e-6
counts, bounds = [], []
for g in gammas:
    V_it, k = evaluate_iterative(P_inv, R_inv, g, pi7, tol)
    V_ex = evaluate_exact(P_inv, R_inv, g, pi7)
    err = np.max(np.abs(V_it - V_ex))
    guarantee = g * tol / (1 - g)  # (stopping rule)
    # (iteration bound) with eta = tol / (1 + gamma), plus the sweep that passes the test
    bounds.append(2 + np.log((1 + g) * np.max(np.abs(V_ex)) / tol) / np.log(1 / g))
    counts.append(k)
    print(
        f"gamma {g}: {k:5d} sweeps (bound {bounds[-1]:7.1f}), error {err:.3e}"
        f" = {err / guarantee:.3f} x the guarantee {guarantee:.3e}"
    )
horizon = [1 / (1 - g) for g in gammas]
fig, ax = plt.subplots()
ax.plot(horizon, counts, marker="o", label="your sweeps")
ax.plot(horizon, bounds, linestyle="--", label="(iteration bound)")
ax.set(
    xlabel="effective horizon 1 / (1 - gamma)",
    ylabel="sweeps to tol = 1e-6",
    title="Sweeps grow with the horizon",
)
ax.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# At γ = 0.99 the change fell below tol, yet your error is about 98 × tol. Which Gotcha is
# that? Then use (iteration bound): how many factors of γ shrink an error by 10⁻⁶ at γ = 0.9,
# and how many at γ = 0.99?

# %% role="ex5-chk"
lab.check(5, m01.check_evaluate_iterative, evaluate_iterative)

# %% [markdown] role="ex6-head"
# **Planted bug.** A colleague wrote `evaluate_suspect`. It runs and returns finite numbers.
# Here you could compare them with Exercise 4, but usually you have no reference to compare
# with. A value function can always be tested with its **Bellman residual**,
# max |T^π V − V| (residual): exactly zero at V^π, and not zero anywhere else. It needs no
# reference. Write the residual, use it as the diagnostic, then fix the bug.

# %% [markdown] role="ex6-predict"
# What residual will `V7`, your exact answer from Exercise 4, have: exactly 0, about 1e−14,
# or about 1e−6? Why not exactly 0?


# %% role="ex6-stub"
def bellman_residual(P, R, gamma, pi, V):
    """max over states of |r_pi + gamma P_pi V - V|: zero exactly at V^pi."""
    # TODO 6 (part 1): compute T^pi V and compare it with V in the max norm.
    raise NotImplementedError(
        "TODO 6: write bellman_residual, or run lab.use_reference(6, 'bellman_residual')"
    )


def evaluate_suspect(P, R, gamma, pi):
    """A colleague's exact evaluator. It runs and looks plausible. Is it right?"""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    # TODO 6 (part 2): once your residual shows the problem, fix the one wrong line below.
    # After fixing it, rerun this cell and the run cell.
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi.T, r_pi)


# %% [markdown] role="ex6-hint"
# Compare the residual of `evaluate_suspect` with that of your `evaluate_exact`. Then look at
# how `P_pi` enters the solve: `P_pi[s, s']` is "from s to s'". Which way does `V` get averaged?


# %% role="ex6-sol"
@lab.solution(6)
def bellman_residual(P, R, gamma, pi, V):
    """max over states of |r_pi + gamma P_pi V - V|: zero exactly at V^pi."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return float(np.max(np.abs(r_pi + gamma * P_pi @ V - V)))


@lab.solution(6)
def evaluate_suspect(P, R, gamma, pi):
    """The colleague's evaluator, fixed: P_pi, not its transpose."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


# %% role="ex6-run"
g = INVENTORY["gamma"]
V_suspect = evaluate_suspect(P_inv, R_inv, g, pi7)
print("suspect values:", np.round(V_suspect, 2))
print("residual of evaluate_suspect:", bellman_residual(P_inv, R_inv, g, pi7, V_suspect))
print("residual of evaluate_exact:  ", bellman_residual(P_inv, R_inv, g, pi7, V7))

# %% [markdown] role="ex6-explain"
# Before your fix, the suspect's (residual) was far above 1. When would this bug be
# invisible, with a residual of zero? (Hint: when is P_π equal to its transpose? Is that the
# only case?) A zero residual means the output is V^π for *this* MDP, not that the code is
# right, so test evaluators on an MDP whose P_π is not symmetric.
#
# **Diagnostics for this lab.** An exact evaluator's residual is near machine precision
# (about 1e−14 here; it depends on your machine). An iterative one that stopped at tol has a
# residual below γ·tol. A residual far above these means V does not satisfy the Bellman
# equation for the P, R and π you passed: the evaluator is wrong, however plausible the
# values look. The residual cannot tell you whether P and R describe the right problem;
# `validate()` and the checkpoints of Exercises 2 and 3 do. A residual check costs about one
# sweep and needs no reference answer: run it on every evaluator you write.

# %% role="ex6-chk1"
lab.check(6, m01.check_bellman_residual, bellman_residual, label="6.1")

# %% role="ex6-chk2"
lab.check(6, m01.check_evaluate_suspect, evaluate_suspect, label="6.2")

# %% [markdown] role="stretch"
# **Contraction depends on the norm.** T^π shrinks distances by at least a factor γ in the
# max norm, because
# each row of P_π is a probability vector. In the ordinary Euclidean (2-) norm it can
# stretch them. Take two states that both move to state 1: P_π = [[0, 1], [0, 1]] with
# γ = 0.9. Write `operator_gains(P_pi, gamma)` returning the max-norm and 2-norm "gain" of
# γ P_π, and see which one exceeds 1. (Weight each state by its stationary probability μ(s),
# the long-run fraction of time π spends there, and γ P_π is again a γ-contraction:
# ‖γ P_π x‖_μ ≤ γ ‖x‖_μ. Module 7 uses this as the key step in showing that on-policy linear
# TD is stable; the full convergence argument needs more conditions. Here μ = (0, 1), so
# this weighted norm ignores state 0.)


# %% role="stretch-stub"
def operator_gains(P_pi, gamma):
    """(max-norm gain, 2-norm gain) of the matrix gamma P_pi."""
    # Stretch: the max-norm of a matrix is its largest absolute row sum; the 2-norm is its
    # largest singular value.
    raise NotImplementedError(
        "Stretch: write operator_gains, or run lab.use_reference('stretch') to see the solution"
    )


# %% role="stretch-sol"
@lab.solution("stretch")
def operator_gains(P_pi, gamma):
    """(max-norm gain, 2-norm gain) of the matrix gamma P_pi."""
    A = gamma * np.asarray(P_pi, dtype=float)
    return float(np.max(np.abs(A).sum(axis=1))), float(np.linalg.norm(A, 2))


# %% role="stretch-run"
print("(max-norm gain, 2-norm gain):", operator_gains(np.array([[0.0, 1.0], [0.0, 1.0]]), 0.9))
lab.check("stretch", m01.check_operator_gains, operator_gains, label="stretch")
