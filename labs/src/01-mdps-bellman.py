# ---
# jupyter:
#   prl:
#     module: m01
# ---

# %% [markdown]
# In this lab you write two Markov decision processes (MDPs) as arrays, then evaluate a
# policy in them twice: exactly, with one linear solve, and by iteration, watching the
# discount set the speed. You finish by finding a bug in someone else's evaluator using the
# one number that never lies about a value function: its Bellman residual.
#
# The briefing's equations are named in brackets, for example (return) or (exact).

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
# The **return** is what every method this week estimates or maximizes: the discounted sum
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
# Loop backward: `g = r + gamma * g` processes the list from the end and needs no powers.


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
# Did you get 2.71 and 2.5? The second case gives a large reward late: what does a small γ
# do to it? A γ close to 1 means the agent cares about the long run; γ defines what is
# valued, it is not a speed setting.

# %% role="ex1-chk"
lab.check(1, m01.check_discounted_return, discounted_return)

# %% [markdown] role="ex2-head"
# The cliff gridworld (Sutton and Barto, Example 6.6). States are cells, numbered
# `row * cols + col` with row 0 at the top. Start bottom-left, goal bottom-right; the cells
# between them on the bottom row are the cliff. Actions: 0 up, 1 right, 2 down, 3 left.
#
# The rules, which your arrays must encode exactly:
#
# - every move costs −1; moving into a wall leaves you where you are;
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
    """The cell action a leads to from (r, c), staying put at the walls."""
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
            # TODO 2: set one entry of P[s, a, :] to 1 and set R[s, a], following the rules:
            # goal -> stay, reward 0; cliff cell -> start, reward 0;
            # otherwise move(r, c, a, rows, cols): into the cliff -> start with -100, else -1.
            raise NotImplementedError(
                "TODO 2: fill P and R, or run lab.use_reference(2) to continue with the solution"
            )
    return P, R


# %% [markdown] role="ex2-hint"
# Handle the two special states first (`if s == goal: ...` and `elif s in cliff: ...`), then
# compute `s2 = index(*move(r, c, a, rows, cols), cols)` and check whether `s2` is in the cliff.


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
# which cell's "move right" costs −100, and why does the goal's row show 0?

# %% role="ex2-chk"
lab.check(2, m01.check_gridworld_arrays, gridworld_arrays)

# %% [markdown] role="ex3a-head"
# **Inventory control, a real-shaped problem.** A shop stocks one product and can hold at
# most 10 units. Each morning you see the stock, place an order that arrives at once
# (capped so stock never exceeds capacity), then customers arrive: demand is random,
# Poisson with mean 4 (truncated at 10). You sell what you can; unmet demand is lost.
# Money: you earn 4 per unit sold, pay 2 per unit ordered plus a fixed 5 for any order,
# pay 0.2 per unit left on the shelf overnight, and lose 1 of goodwill per customer turned
# away. The discount is γ = 0.95 per day.
#
# Before you write arrays, write the framing card: what is a state, what is an action, what
# is the reward?

# %% [markdown] role="ex3a-predict"
# Is today's stock enough to predict the future, or does the state also need yesterday's
# demand? Write one sentence why.


# %% role="ex3a-stub"
def inventory_spec(capacity):
    """The framing card's counts: {'n_states': ..., 'n_actions': ...}."""
    # TODO 3a (part 1): states are stock levels 0..capacity; actions are order sizes.
    raise NotImplementedError(
        "TODO 3a: write inventory_spec, or run lab.use_reference('3a') to continue with the solution"
    )


def period_reward(stock, order, demand, p):
    """One day's reward: revenue - ordering cost - holding cost - lost-sales penalty.

    Orders are capped so that stock + order <= p['capacity'].
    """
    # TODO 3a (part 2): cap the order, sell min(stock + order, demand), then add up the money.
    raise NotImplementedError(
        "TODO 3a: write period_reward, or run lab.use_reference('3a') to continue with the solution"
    )


# %% [markdown] role="ex3a-hint"
# With `q = min(order, p["capacity"] - stock)`, `y = stock + q`, `sales = min(y, demand)` and
# `left = y - sales`, the reward is `price * sales - (fixed_cost if q > 0 else 0) - unit_cost * q
# - holding_cost * left - lost_sales_penalty * (demand - sales)`.


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
# Demand is drawn fresh each day, independent of the past, so today's stock is all the
# future depends on: the state is Markov. The first case's reward is negative (−3.60) even
# though ordering 7 is sensible: one day's reward is not the value of a decision. That gap
# is what the Bellman equation closes.

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
# keep stock for tomorrow too. Why can't we pick orders by one-day reward? What would we
# need to know about tomorrow?

# %% role="ex3b-chk"
lab.check("3b", m01.check_inventory_arrays, inventory_arrays)

# %% [markdown] role="ex4-head"
# A **policy** π here is an (S, A) matrix of action probabilities. For a fixed policy the
# Bellman equation is linear, V = r_π + γ P_π V (Bellman, matrix form), so one solve gives
# V^π = (I − γ P_π)⁻¹ r_π (exact). Below, the "order up to 7" policy orders max(0, 7 − stock).

# %% [markdown] role="ex4-predict"
# Under "order up to 7", is V(6) much smaller than V(7), or about the same? Why?


# %% role="ex4-stub"
def evaluate_exact(P, R, gamma, pi):
    """V^pi from (I - gamma P_pi) V = r_pi, where pi is an (S, A) matrix of probabilities."""
    # TODO 4: average P and R over actions with pi to get P_pi (S, S) and r_pi (S,), then solve.
    raise NotImplementedError(
        "TODO 4: write evaluate_exact, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# `P_pi = np.einsum("sa,sat->st", pi, P)` and `r_pi = (pi * R).sum(axis=1)`. Then
# `np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)`; never invert the matrix explicitly.


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

# %% [markdown] role="ex4-explain"
# V rises with stock, and jumps by about 7 between stock 6 and 7. At 6 the policy still
# orders one unit, paying the fixed 5 plus 2 for the unit; at 7 it orders nothing. What does
# that suggest about this policy at stock 6?

# %% role="ex4-chk"
lab.check(4, m01.check_evaluate_exact, evaluate_exact)

# %% [markdown] role="ex5-head"
# Iterative evaluation applies T^π V = r_π + γ P_π V (operator) again and again from V = 0.
# Because T^π is a γ-contraction in the max norm (contraction), the error shrinks by at
# least γ per step, and stopping when max |V_new − V| < tol guarantees an error of at most
# γ·tol / (1 − γ) (stopping rule).

# %% [markdown] role="ex5-predict"
# Going from γ = 0.9 to γ = 0.99, how many times more updates do you need to reach
# tol = 1e−6? Write a number.


# %% role="ex5-stub"
def evaluate_iterative(P, R, gamma, pi, tol):
    """Repeat V <- r_pi + gamma P_pi V from V = 0 until max|V_new - V| < tol.

    Build all of V_new from the old V (not in place, state by state).
    Returns (V_new, k), where k counts the updates.
    """
    # TODO 5: build P_pi and r_pi once, then loop.
    raise NotImplementedError(
        "TODO 5: write evaluate_iterative, or run lab.use_reference(5) to continue with the solution"
    )


# %% [markdown] role="ex5-hint"
# Inside a `while True:` loop, compute `V_new`, increment `k`, and `return V_new, k` as soon as
# `np.max(np.abs(V_new - V)) < tol`; otherwise set `V = V_new`. If the cell never finishes,
# you probably forgot `V = V_new` (interrupt it with the stop button).


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
counts, errors = [], []
for g in gammas:
    V_it, k = evaluate_iterative(P_inv, R_inv, g, pi7, 1e-6)
    err = np.max(np.abs(V_it - evaluate_exact(P_inv, R_inv, g, pi7)))
    counts.append(k)
    errors.append(err)
    bound = g * 1e-6 / (1 - g)
    print(
        f"gamma {g}: {k:5d} updates, error {err:.3e} = {err / bound:.2f} x the guarantee {bound:.3e}"
    )
fig, ax = plt.subplots()
ax.plot([1 / (1 - g) for g in gammas], counts, marker="o")
ax.set(
    xlabel="effective horizon 1 / (1 - gamma)",
    ylabel="updates to tol = 1e-6",
    title="Iterations grow with the horizon",
)
plt.show()

# %% [markdown] role="ex5-explain"
# From γ = 0.9 to 0.99 the count grows about tenfold, close to linear in 1 / (1 − γ). Use the
# briefing's iteration bound to explain why: how many factors of γ does it take to shrink an
# error by 10⁻⁶?

# %% role="ex5-chk"
lab.check(5, m01.check_evaluate_iterative, evaluate_iterative)

# %% [markdown] role="ex6-head"
# **Planted bug.** A colleague wrote `evaluate_suspect`. It runs and returns finite numbers.
# Here you could compare them with Exercise 4, but usually you have no reference to compare
# with. A value function can always be tested with its **Bellman residual**,
# max |T^π V − V| (residual): exactly zero at V^π, and not zero anywhere else. It needs no
# reference. Write the residual, use it as the diagnostic, then fix the bug.
#
# **Diagnostics for this lab.** A healthy evaluator has a residual near machine precision
# (about 1e−14 here); an iterative one, about tol. A residual of order 1 or more means the
# arrays or the evaluator are wrong, however plausible the values look.

# %% [markdown] role="ex6-predict"
# If V is exactly V^π, what is its Bellman residual? What would a residual of 5 tell you?


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
# Before your fix, the suspect's residual was large. When would this bug be invisible, with a
# residual of zero? (Hint: when is P_π equal to its transpose?) A residual check costs one
# matrix-vector product and needs no reference answer; run it on every evaluator you write.

# %% role="ex6-chk1"
lab.check(6, m01.check_bellman_residual, bellman_residual, label="6.1")

# %% role="ex6-chk2"
lab.check(6, m01.check_evaluate_suspect, evaluate_suspect, label="6.2")

# %% [markdown] role="stretch"
# **Contraction depends on the norm.** T^π shrinks distances by γ in the max norm, because
# each row of P_π is a probability vector. In the ordinary Euclidean (2-) norm it can
# stretch them. Take two states that both move to state 1: P_π = [[0, 1], [0, 1]] with
# γ = 0.9. Write `operator_gains(P_pi, gamma)` returning the max-norm and 2-norm "gain" of
# γ P_π, and see which one exceeds 1. (It contracts again in a norm weighted by the
# stationary distribution, which is why on-policy linear TD converges in Module 7.)


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
