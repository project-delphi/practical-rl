# ---
# jupyter:
#   prl:
#     module: m06
#     pins:
#       gymnasium: "1.4.0"
# ---

# %% [markdown]
# In this lab the reward is many steps away. RiverSwim is a chain of states: a tiny reward at
# the left end, a large one at the right end, and a current that makes swimming right slow.
# You first compute, exactly, how long a random swimmer takes to reach the right end, and
# watch that time grow by a factor of about 3 per state. Then you run three learners on a
# 10-state river for 5000 steps: Lab 3's ε-greedy Q-learning, the same learner with an
# optimistic start, and MBIE-EB, which plans in an estimated model with a count bonus inside
# the backup. You finish by finding a bonus in the wrong place in someone else's agent.
#
# The briefing's equations are named in brackets, for example (hitting time) or (MBIE-EB
# backup). Its table "Equations and where you implement them" gives each name's equation
# number.

# %%
import textwrap
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from prl import envs
from prl.evaluate import Budget, bootstrap_ci, compare, iqm

GAMMA = 0.95  # the discount of Strehl and Littman's experiments
R_MAX = 1.0  # every reward in RiverSwim lies in [0, R_MAX]
N_CHAIN = 10  # the river of Exercises 3-7 (Strehl and Littman's has 6 states)
LEFT, RIGHT = 0, 1
EPSILON, ALPHA = 0.1, 0.1  # Lab 3's exploration rate and step size, for Q-learning
C_BONUS = 1.0  # MBIE-EB's bonus coefficient c; c = R_MAX (the briefing says why)
TOL = 0.01  # value iteration stops when no value changes by more than this (S&L, s6)

river = envs.RiverSwim(N_CHAIN, gamma=GAMMA)
S, A = river.n_states, river.n_actions  # Lab 3's q_learning reads S and A
END = N_CHAIN - 1  # the right end

# Budgets. Lab code reads them from here and never branches on QUICK. Every run is cheap, so
# the QUICK budget is the live one.
budget = lab.use_settings(
    prl.runtime.settings(
        dict(steps=5000, seeds=10),  # live
        dict(steps=5000, seeds=10),  # QUICK
        designed="colab-cpu",
    )
)
# Your seat's seeds start here, clear of the briefing's, the figures' and the checkpoints'
# seeds (0-99), so every seat in the room adds new evidence.
RUN_SEED = (SEED + 1) * 1000
print(f"Budget: {budget.steps} steps per run; {budget.seeds} seeds per method in Exercise 6.")
print(f"River of {N_CHAIN} states, gamma = {GAMMA}, rewards in [0, {R_MAX}].")

# %% [markdown]
# ### The reference versions of your Lab 2 and Lab 3 code
#
# MBIE-EB needs Lab 2's counts estimator and value iteration, and the first two learners are
# Lab 3's Q-learning. The cells below are the reference versions of that code, included as
# they are. `epsilon_greedy` is the helper Lab 3 provided.

# %% include labs/shared/value_iteration.py

# %% include labs/shared/estimate_model.py

# %% include labs/shared/q_learning.py


# %%
def epsilon_greedy(Q, s, epsilon, rng):
    """Provided: the greedy action with probability 1 - epsilon, else uniform (ties at random)."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def make_river(n, steps):
    """Provided: a RiverSwim of n states as a Gymnasium env; one episode lasts `steps` steps."""
    return envs.TabularEnv(envs.RiverSwim(n, gamma=GAMMA), max_episode_steps=steps)


print("RiverSwim's probabilities (ours; the briefing compares them with the papers'):")
print(
    f"  right from an interior state: right {river.meta['p_right']}, stay {river.meta['p_stay']},"
)
print(f"  left {river.meta['p_left']}; the left action always moves left (and stays at 0).")
print(f"  reward {river.meta['r_left']} for left at the left end, {river.meta['r_right']} for")
print("  holding position at the right end with the right action.")

# %% [markdown] role="ex1-head"
# **The chain a fixed policy makes.** Fix a policy π, an (S, A) matrix of action
# probabilities. The next state then depends only on the current one, through
# P_π(s, s') = Σ_a π(a | s) P(s' | s, a) (policy chain). Uniform random play, π = ½ for each
# action, is the policy of the briefing's first-step analysis.

# %% [markdown] role="ex1-predict"
# Under uniform random play on the 6-state river, what is the chance of staying at the right
# end for one step, P_π(5, 5), and of moving left from it, P_π(5, 4)?


# %% role="ex1-stub"
def policy_transition(P, pi):
    """P_pi[s, s'] = sum over a of pi[s, a] P[s, a, s'] (policy chain).

    P has shape (S, A, S); pi has shape (S, A), each row a distribution over actions.
    Returns an (S, S) array whose rows sum to 1.
    """
    # TODO 1: mix the rows P[s, a, :] with the weights pi[s, a]: one line.
    raise NotImplementedError(
        "TODO 1: write policy_transition, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# `np.einsum("sa,sat->st", pi, P)` sums over the action index a. So does
# `(pi[:, :, None] * P).sum(axis=1)`. Check that every row of your result sums to 1.


# %% role="ex1-sol"
@lab.solution(1)
def policy_transition(P, pi):
    """P_pi[s, s'] = sum_a pi[s, a] P[s, a, s'] (policy chain)."""
    return np.einsum("sa,sat->st", pi, P)


# %% role="ex1-run"
river6 = envs.RiverSwim(6, gamma=GAMMA)
P_pi6 = policy_transition(river6.P, np.full((6, 2), 0.5))
print("P_pi under uniform random play, 6 states (row s: where s goes in one step):")
print(np.array2string(P_pi6, precision=3, suppress_small=True))
right, stay, left = P_pi6[2, 3], P_pi6[2, 2], P_pi6[2, 1]
print(f"An interior state: right {right:.3f}, stay {stay:.3f}, left {left:.3f}")
print(f"Left over right: {left / right:.3f}")
print(f"The left end: right {P_pi6[0, 1]:.3f}, stay {P_pi6[0, 0]:.3f}")
print(f"The right end: stay {P_pi6[5, 5]:.3f}, left {P_pi6[5, 4]:.3f}")

# %% [markdown] role="ex1-explain"
# The left action always succeeds, so a step left combines a sure move with the right
# action's slip: ½ · 1 + ½ · 0.05. A step right needs the right action and its 0.35. Which
# ratio of two numbers in an interior row of (policy chain) will set how fast the hitting
# time grows? If the left action also failed 5% of the time, would that ratio go up or down?

# %% role="ex1-chk"
# The checkpoints of this lab. (Technical expert: move this import to the first code cell
# once prl.checks.m06 exists.)
from prl.checks import m06

lab.check(1, m06.check_policy_transition, policy_transition)

# %% [markdown] role="ex2-head"
# **Hitting time by first-step analysis.** Let h(s) be the expected number of steps to first
# reach the right end from s. At the target h = 0; from any other state one step is spent,
# then the walk continues from wherever it lands: h(s) = 1 + Σ_s' P_π(s, s') h(s') (hitting
# time). Over the states other than the target this is one linear system, (I − M) h = 1,
# where M is P_π with the target's row and column removed. You also get a simulation to
# compare with.


# %%
def simulate_hitting(P_pi, target, walks, rng, start=0):
    """Provided: run `walks` independent walks from `start` until each reaches `target`;
    return the number of steps each took."""
    cum = np.cumsum(P_pi, axis=1)
    s = np.full(walks, start)
    steps = np.zeros(walks, dtype=int)
    active = s != target
    while active.any():
        u = rng.random(int(active.sum()))
        nxt = (u[:, None] > cum[s[active]]).sum(axis=1)
        s[active] = np.minimum(nxt, len(P_pi) - 1)  # guard against rounding in cum
        steps[active] += 1
        active = s != target
    return steps


# %% [markdown] role="ex2-predict"
# The briefing worked out h(6) ≈ 936 steps for the 6-state river, and showed that each extra
# state multiplies the hitting time by about 3. Without solving anything, predict h(10) for
# the 10-state river of Exercises 3 to 7. How many times longer is that than their budget of
# 5000 steps?


# %% role="ex2-stub"
def hitting_times(P_pi, target):
    """Expected steps to first reach `target`, from every state (hitting time).

    P_pi is an (S, S) transition matrix. Returns h of shape (S,): h[target] = 0 and, for every
    other state s, h[s] = 1 + sum over s' of P_pi[s, s'] h[s'].
    """
    n = len(P_pi)
    others = [s for s in range(n) if s != target]
    h = np.zeros(n)
    # TODO 2: M = P_pi restricted to `others` (rows and columns); solve (I - M) x = 1 and put
    # x into h[others].
    raise NotImplementedError(
        "TODO 2: write hitting_times, or run lab.use_reference(2) to continue with the solution"
    )


# %% [markdown] role="ex2-hint"
# `P_pi[np.ix_(others, others)]` is M. `np.linalg.solve(np.eye(n - 1) - M, np.ones(n - 1))`
# is x. Why is h[target] not one of the unknowns? Because nothing is spent once you are there.


# %% role="ex2-sol"
@lab.solution(2)
def hitting_times(P_pi, target):
    """First-step analysis: (I - M) h = 1 off the target (hitting time)."""
    n = len(P_pi)
    others = [s for s in range(n) if s != target]
    h = np.zeros(n)
    M = P_pi[np.ix_(others, others)]  # moves that have not yet hit the target
    h[others] = np.linalg.solve(np.eye(n - 1) - M, np.ones(n - 1))
    return h


# %% role="ex2-run"
lengths, h_start = list(range(2, 13)), []
print("Uniform random play, expected steps from the left end to the right end:")
for n in lengths:
    P_n = policy_transition(envs.RiverSwim(n, gamma=GAMMA).P, np.full((n, 2), 0.5))
    h_start.append(hitting_times(P_n, n - 1)[0])
    ratio = f"   x {h_start[-1] / h_start[-2]:.3f}" if len(h_start) > 1 else ""
    print(f"  n = {n:2d}: h = {h_start[-1]:12,.1f}{ratio}")
P_5 = policy_transition(envs.RiverSwim(5, gamma=GAMMA).P, np.full((5, 2), 0.5))
walks = simulate_hitting(P_5, 4, 2000, np.random.default_rng(RUN_SEED))
se = walks.std() / np.sqrt(len(walks))
print(f"n = 5: linear solve {hitting_times(P_5, 4)[0]:.1f}; 2000 simulated walks")
print(f"       mean {walks.mean():.1f} (standard error {se:.1f}), shortest {walks.min()},")
print(f"       longest {walks.max()}")
fig, ax = plt.subplots()
ax.semilogy(lengths, h_start, marker="o", label="uniform random play (linear solve)")
ax.axhline(
    budget.steps, ls="--", color=prl.plot.NEUTRAL_LIGHT, label=f"budget, {budget.steps} steps"
)
ax.set(xlabel="chain length n", ylabel="expected steps to the right end", title="Hitting time")
ax.legend()
plt.show()

# %% [markdown] role="ex2-explain"
# On a log scale a constant ratio is a straight line (hitting time). Which chain lengths can
# uniform random play be expected to finish within the 5000-step budget? The simulated walks
# agree with the solve to within a few standard errors, yet the longest walk is many times
# the mean: what does that spread say about a single run? (Gotcha: "ε-greedy gets there
# eventually.")

# %% role="ex2-chk"
lab.check(2, m06.check_hitting_times, hitting_times)

# %% [markdown]
# ### The runners, the visitation counter and the two reference lines
#
# A `Recorder` wrapped around the environment keeps every state and action, so Lab 3's
# `q_learning` runs unchanged. Every learner runs one episode of `budget.steps` steps from
# the left end, as in Strehl and Littman's experiments (no resets). The two reference lines
# are exact expectations: uniform random play, and the DP-optimal policy from Lab 2's value
# iteration on the true model.


# %%
class Recorder(gym.Wrapper):
    """Provided: keeps the states and actions of the current episode."""

    def __init__(self, env):
        super().__init__(env)
        self.states, self.actions = [], []

    def reset(self, **kwargs):
        s, info = self.env.reset(**kwargs)
        self.states, self.actions = [int(s)], []  # a reset starts a new trajectory
        return s, info

    def step(self, action):
        out = self.env.step(action)
        self.actions.append(int(action))
        self.states.append(int(out[0]))
        return out


def run_q_learning(env, steps, q0, seed):
    """Provided: Lab 3's q_learning for one episode of `steps` steps.
    Returns (states, actions, Q): states s_0..s_T and actions a_0..a_{T-1}."""
    rec = Recorder(env)
    Q, _ = q_learning(rec, 1, ALPHA, EPSILON, GAMMA, seed, q0=q0)
    return np.array(rec.states), np.array(rec.actions), Q


def first_visit(states, target):
    """Provided: the first step at which `states` reaches `target`, or None."""
    hits = np.flatnonzero(np.asarray(states) == target)
    return int(hits[0]) if len(hits) else None


def expected_rewards(mdp, pi, steps):
    """Provided: the exact expected reward at each step, and the expected fraction of steps
    in each state, of the policy pi from the start (two reference lines)."""
    P_pi, r_pi = policy_transition(mdp.P, pi), (pi * mdp.R).sum(axis=1)
    d = np.zeros(mdp.n_states)
    d[mdp.start] = 1.0
    rewards, occupancy = np.empty(steps), np.zeros(mdp.n_states)
    for t in range(steps):
        rewards[t] = d @ r_pi
        occupancy += d
        d = d @ P_pi
    return rewards, occupancy / steps


V_STAR, Q_STAR, _ = value_iteration(river.P, river.R, GAMMA, 1e-10)
PI_STAR = river.policy_matrix(greedy(Q_STAR))
OPT, OCC_STAR = expected_rewards(river, PI_STAR, budget.steps)
_, OCC_UNIFORM = expected_rewards(river, np.full((S, A), 0.5), budget.steps)
print("DP-optimal policy is 'right' in every state:", bool(np.all(greedy(Q_STAR) == RIGHT)))
print(f"Its expected reward over {budget.steps} steps from the left end: {OPT.sum():.1f}")
print(
    f"Expected fraction of steps at the right end: optimal {OCC_STAR[END]:.3f}, "
    f"uniform random {OCC_UNIFORM[END]:.2e}"
)

# %% [markdown] role="ex3-head"
# **ε-greedy and where it goes.** Lab 3's Q-learning, from Q₀ = 0 with ε = 0.1, swims the
# 10-state river for 5000 steps. Write `visitation`: the fraction of the steps spent in each
# state (visitation). The run cell passes it the states where an action was taken,
# s_0, …, s_{T−1}.

# %% [markdown] role="ex3-predict"
# Where will ε-greedy Q-learning spend most of its 5000 steps? Uniform random play, the
# briefing's reference line, spends about 64% of its steps at the left end: will ε-greedy be
# more concentrated there, or less? Will it reach the right end at all?


# %% role="ex3-stub"
def visitation(states, n_states):
    """Fraction of the entries of `states` equal to each state 0..n_states-1 (visitation).

    Returns an array of shape (n_states,) that sums to 1; a state never visited gets 0.
    """
    # TODO 3: count each state, then divide by the number of entries.
    raise NotImplementedError(
        "TODO 3: write visitation, or run lab.use_reference(3) to continue with the solution"
    )


# %% [markdown] role="ex3-hint"
# `np.bincount(states)` counts each value, but its length is the largest state that occurs
# plus one: a state never reached would be missing from the end. Pass `minlength=n_states`.


# %% role="ex3-sol"
@lab.solution(3)
def visitation(states, n_states):
    """Fraction of time steps in each state (visitation)."""
    counts = np.bincount(np.asarray(states, dtype=int), minlength=n_states)
    return counts / counts.sum()


# %% role="ex3-run"
states_eps, actions_eps, Q_eps = run_q_learning(
    make_river(N_CHAIN, budget.steps), budget.steps, 0.0, seed=RUN_SEED
)
lab.seeds([RUN_SEED])
v_eps = visitation(states_eps[:-1], S)
hit = first_visit(states_eps, END)
print(f"eps-greedy Q-learning, seed {RUN_SEED}, {budget.steps} steps:")
print("  fraction of steps in each state:", " ".join(f"{v:.3f}" for v in v_eps))
print(f"  first visit to the right end: {'never' if hit is None else f'step {hit}'}")
print(f"  Q at the left end: left {Q_eps[0, LEFT]:.4f}, right {Q_eps[0, RIGHT]:.4f}")
lab.metric("eps_visitation", [round(float(x), 4) for x in v_eps])
fig, ax = plt.subplots()
x = np.arange(S)
ax.bar(x, v_eps, color=prl.plot.SERIES[0], label=f"eps-greedy, seed {RUN_SEED}")
ax.plot(
    x,
    OCC_UNIFORM,
    ls="--",
    marker="o",
    color=prl.plot.NEUTRAL_LIGHT,
    label="uniform random play (expected)",
)
ax.plot(
    x, OCC_STAR, ls=":", marker="s", color=prl.plot.NEUTRAL_LIGHT, label="optimal policy (expected)"
)
ax.set(xlabel="state (0 = left end)", ylabel="fraction of steps", title="Visitation")
ax.legend()
plt.show()

# %% [markdown] role="ex3-explain"
# Read Q at the left end: once the left action has paid 0.005, it beats the right action's
# 0, and the greedy choice there is left. How often does a right move happen at the left end
# now, against ½ under uniform random play? Which number in the hitting-time ratio got worse,
# and by how much? (Gotcha: "ε-greedy gets there eventually.")

# %% role="ex3-chk1"
lab.check(3, m06.check_visitation, visitation, label="3.1")

# %% role="ex3-chk2"
lab.check(3, m06.check_eps_greedy_end, visitation, q_learning, label="3.2")

# %% [markdown] role="ex4-head"
# **Optimism in tabular form.** Same Q-learning, same ε, same steps; only the start changes.
# Every reward lies in [0, r_max], so no return can exceed r_max + γ r_max + γ² r_max + … :
# a start at that value is at least Q*(s, a) for every pair (optimistic start). A pair the
# agent has not tried keeps it, so sooner or later the greedy choice tries it.

# %% [markdown] role="ex4-predict"
# With the optimistic start, in how many of your 10 seeds will Q-learning reach the right end
# within 5000 steps? Will its first visit take tens of steps, hundreds, or tens of thousands?


# %% role="ex4-stub"
def optimistic_q0(r_max, gamma, n_states, n_actions):
    """An (n_states, n_actions) start for Q: in every entry, the largest discounted return
    possible when every reward lies in [0, r_max] (optimistic start)."""
    # TODO 4: the sum r_max + gamma r_max + gamma^2 r_max + ... in closed form, everywhere.
    raise NotImplementedError(
        "TODO 4: write optimistic_q0, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# A geometric series: 1 + γ + γ² + … = 1/(1 − γ). `np.full((n_states, n_actions), value)`.


# %% role="ex4-sol"
@lab.solution(4)
def optimistic_q0(r_max, gamma, n_states, n_actions):
    """Q_0 = r_max / (1 - gamma) everywhere (optimistic start)."""
    return np.full((n_states, n_actions), r_max / (1 - gamma))


# %% role="ex4-run"
Q0 = optimistic_q0(R_MAX, GAMMA, S, A)
print(f"Optimistic start: Q0 = {Q0[0, 0]:.2f} everywhere; the largest Q* is {Q_STAR.max():.2f}")
seeds4 = [RUN_SEED + 10 + i for i in range(budget.seeds)]
lab.seeds(seeds4)
firsts, v_opt = [], None
for k in seeds4:
    st, ac, _ = run_q_learning(make_river(N_CHAIN, budget.steps), budget.steps, Q0, seed=k)
    firsts.append(first_visit(st, END))
    if v_opt is None:
        v_opt = visitation(st[:-1], S)
reached = sum(f is not None for f in firsts)
shown = ", ".join("never" if f is None else str(f) for f in firsts)
print(f"Reached the right end within {budget.steps} steps on {reached} of {len(seeds4)} seeds.")
print(f"First visit, seed by seed: {shown}")
lab.metric("opt_reached", reached)  # room code
fig, ax = plt.subplots()
ax.bar(x - 0.2, v_eps, width=0.4, color=prl.plot.SERIES[0], label="eps-greedy, Q0 = 0")
ax.bar(
    x + 0.2,
    v_opt,
    width=0.4,
    color=prl.plot.SERIES[1],
    hatch="//",
    label=f"optimistic, Q0 = {Q0[0, 0]:.0f}",
)
ax.set(
    xlabel="state (0 = left end)",
    ylabel="fraction of steps",
    title=f"Visitation, first seed of each ({seeds4[0]}, {RUN_SEED})",
)
ax.legend()
plt.show()

# %% [markdown] role="ex4-explain"
# Each update pulls Q(s, a) toward r + γ max Q(s', ·), and every pair not yet tried still
# holds 20 (optimistic start). Why does that keep pulling the agent into states it has not
# seen? Its visitation is spread over the whole river: what is still pulling it away from the
# right end after it has found the reward there? Sutton and Barto call optimistic initial
# values a simple trick for stationary problems: what would happen to its exploration if the
# river changed after step 5000? (Gotcha: "Optimistic initialization is a hack.")

# %% role="ex4-chk1"
lab.check(4, m06.check_optimistic_q0, optimistic_q0, label="4.1")

# %% role="ex4-chk2"
lab.check(4, m06.check_optimistic_reach, optimistic_q0, q_learning, label="4.2")

# %% [markdown] role="ex5-head"
# **The bonus belongs in the backup.** A model-based agent counts every transition, estimates
# P̂ and r̂ with Lab 2's `estimate_model`, plans, and acts greedily on the plan. MBIE-EB plans
# with a bonus c/√N(s, a) added to the reward of every pair, inside the backup, so the bonus
# of a pair far away reaches the states that lead to it (MBIE-EB backup). A pair never tried
# keeps Lab 2's default, a self-loop with reward 0, and gets the bonus of a pair tried once.
# The runner below is provided; you write the plan.


# %%
def run_planner(env, steps, plan, seed):
    """Provided: count, estimate (Lab 2), plan, act greedily (Lab 2's greedy and its tie rule).

    plan(P_hat, R_hat, counts) returns a (S, A) array Q. The agent re-plans when the count of
    the pair it just tried reaches 1, 2, 4, 8, ... (Strehl and Littman re-plan every step).
    Returns (states, actions, Q): states s_0..s_T, actions a_0..a_{T-1}, and the last plan.
    """
    n_s, n_a = env.observation_space.n, env.action_space.n
    log, states, actions = [], [], []
    counts = np.zeros((n_s, n_a), dtype=int)
    s, _ = env.reset(seed=seed)
    stale = True
    for _ in range(steps):
        if stale:  # plan again, from everything seen so far
            Q = plan(*estimate_model(log, n_s, n_a))
        a = int(greedy(Q[[s]])[0])
        s2, r, terminated, truncated, _ = env.step(a)
        log.append((s, a, r, s2))
        states.append(s)
        actions.append(a)
        counts[s, a] += 1
        stale = counts[s, a] & (counts[s, a] - 1) == 0  # a power of 2
        s = s2
    states.append(s)
    return np.array(states), np.array(actions), Q


def pair_counts(states, actions):
    """Provided: N(s, a) of a run, as an (S, A) integer array."""
    N = np.zeros((S, A), dtype=int)
    np.add.at(N, (np.asarray(states)[:-1], np.asarray(actions)), 1)
    return N


# %% [markdown] role="ex5-predict"
# Where will MBIE-EB spend most of its 5000 steps, and roughly what fraction of them: under a
# quarter, about half, or over three quarters? (The optimal policy's reference line spends
# 0.85 of its steps at the right end.)


# %% role="ex5-stub"
def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):
    """Q~ of the estimated model with the bonus c / sqrt(N) inside the backup (MBIE-EB backup).

    P_hat, R_hat and counts come from estimate_model (Lab 2). A pair with count 0 gets the
    bonus of a pair tried once. Solve with value_iteration(P, R, gamma, tol); return its Q.
    """
    # TODO 5: the bonus array, then one call to value_iteration with the bonus in the reward.
    raise NotImplementedError(
        "TODO 5: write mbie_eb_backup, or run lab.use_reference(5) to continue with the solution"
    )


# %% [markdown] role="ex5-hint"
# `c / np.sqrt(np.maximum(counts, 1))` has the shape of R_hat. value_iteration returns
# (V, Q, sweeps); you need the Q, built from the reward plus the bonus.


# %% role="ex5-sol"
@lab.solution(5)
def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):
    """Q~ = R_hat + c / sqrt(N) + gamma P_hat max Q~ (MBIE-EB backup)."""
    bonus = c / np.sqrt(np.maximum(counts, 1))  # N = 0 gets the bonus of N = 1
    return value_iteration(P_hat, R_hat + bonus, gamma, tol)[1]


# %% role="ex5-run"
def plan_mbie_eb(P_hat, R_hat, counts):
    """MBIE-EB's plan at the lab's c, gamma and tol."""
    return mbie_eb_backup(P_hat, R_hat, counts, GAMMA, C_BONUS, TOL)


states_mb, actions_mb, Q_mb = run_planner(
    make_river(N_CHAIN, budget.steps), budget.steps, plan_mbie_eb, seed=RUN_SEED
)
v_mb = visitation(states_mb[:-1], S)
hit = first_visit(states_mb, END)
print(f"MBIE-EB (c = {C_BONUS}), seed {RUN_SEED}, {budget.steps} steps:")
print(f"  first visit to the right end: {'never' if hit is None else f'step {hit}'}")
print(f"  fraction of steps at the right end: {v_mb[END]:.3f}")
N_mb = pair_counts(states_mb, actions_mb)
print("  N(s, a), one column per state (0 = left end):")
print("    left  ", " ".join(f"{n:5d}" for n in N_mb[:, LEFT]))
print("    right ", " ".join(f"{n:5d}" for n in N_mb[:, RIGHT]))
lab.metric("mbie_end", round(float(v_mb[END]), 3))  # room code
fig, ax = plt.subplots()
ax.bar(x, v_mb, color=prl.plot.SERIES[2], label=f"MBIE-EB, seed {RUN_SEED}")
ax.plot(
    x, OCC_STAR, ls=":", marker="s", color=prl.plot.NEUTRAL_LIGHT, label="optimal policy (expected)"
)
ax.set(xlabel="state (0 = left end)", ylabel="fraction of steps", title="Visitation")
ax.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# Read the counts: which pairs did MBIE-EB try only a few times, and why did it stop trying
# them while their bonus c/√N was still large? At the right end the bonus shrinks like
# 1/√N; why does the agent stay anyway (MBIE-EB backup)?

# %% role="ex5-chk"
lab.check(5, m06.check_mbie_eb_backup, mbie_eb_backup)

# %% [markdown] role="ex6-head"
# **Regret over 10 seeds.** The provided cell runs the three learners on 10 seeds each and
# computes each run's cumulative regret against the DP-optimal policy: the optimal policy's
# expected reward at each step from the left end, minus r(s_t, a_t), the mean reward of what
# the learner did, summed (regret). With 10 seeds per side, compare interquartile means
# (IQMs) with 95% bootstrap intervals, and call a tie when the intervals overlap. Write that
# rule as `verdict`.

# %% [markdown] role="ex6-predict"
# Rank ε-greedy Q-learning, optimistic Q-learning and MBIE-EB by their regret after 5000
# steps. Will any of the three pairs be a tie under the rule?


# %% role="ex6-stub"
def verdict(a, b):
    """'A', 'B' or 'tie' for two arrays of per-seed regrets (lower is better).

    The rule for 5 or more seeds: the IQM of each side with a 95% bootstrap interval,
    prl.evaluate.bootstrap_ci(x, iqm) at its defaults; overlapping intervals are a tie;
    otherwise the side with the lower IQM wins.
    """
    # TODO 6: two intervals, one overlap test, one comparison of the IQMs.
    raise NotImplementedError(
        "TODO 6: write verdict, or run lab.use_reference(6) to continue with the solution"
    )


# %% [markdown] role="ex6-hint"
# `lo, hi = bootstrap_ci(a, iqm)`. Two intervals [lo_a, hi_a] and [lo_b, hi_b] overlap when
# hi_a ≥ lo_b and hi_b ≥ lo_a. Lower regret is better.


# %% role="ex6-sol"
@lab.solution(6)
def verdict(a, b):
    """The rule for 5+ seeds: IQMs with bootstrap CIs; overlap is a tie."""
    lo_a, hi_a = bootstrap_ci(a, iqm)
    lo_b, hi_b = bootstrap_ci(b, iqm)
    if hi_a >= lo_b and hi_b >= lo_a:  # the intervals overlap
        return "tie"
    return "A" if iqm(a) < iqm(b) else "B"


# %% role="ex6-run"
def regret(states, actions):
    """Provided: cumulative regret against the optimal policy's expected rewards (regret)."""
    return np.cumsum(OPT - river.R[np.asarray(states)[:-1], np.asarray(actions)])


learners = {
    "eps-greedy": lambda env, k: run_q_learning(env, budget.steps, 0.0, k),
    "optimistic": lambda env, k: run_q_learning(env, budget.steps, Q0, k),
    "MBIE-EB": lambda env, k: run_planner(env, budget.steps, plan_mbie_eb, k),
}
seeds6 = [RUN_SEED + 100 + k for k in range(budget.seeds)]
lab.seeds(seeds6)
start = time.perf_counter()
curves = {name: [] for name in learners}
for name, run in learners.items():
    for k in seeds6:
        st, ac, _ = run(make_river(N_CHAIN, budget.steps), k)
        curves[name].append(regret(st, ac))
seconds = time.perf_counter() - start
final = {name: np.array([c[-1] for c in cs]) for name, cs in curves.items()}
cost = Budget(
    steps=budget.steps,
    seeds=budget.seeds,
    wall_clock_s=seconds,
    hardware=prl.runtime.detect().cpu,
)
print(f"Regret after {budget.steps} steps, seeds {seeds6[0]}-{seeds6[-1]}:")
for name, r in final.items():
    lo, hi = bootstrap_ci(r, iqm)
    print(
        f"  {name:10s} IQM {iqm(r):7.1f}  [{lo:7.1f}, {hi:7.1f}]  "
        f"min {r.min():7.1f}  max {r.max():7.1f}"
    )
pairs = [("MBIE-EB", "optimistic"), ("MBIE-EB", "eps-greedy"), ("optimistic", "eps-greedy")]
for a, b in pairs:
    print(f"{a} (A) vs {b} (B): your verdict {verdict(final[a], final[b])}")
    v = compare(final[a], final[b], names=(a, b), budget=cost, higher_is_better=False)
    print(
        textwrap.fill(f"compare: {v.sentence}", 96, initial_indent="  ", subsequent_indent="    ")
    )
lab.metric("regret", {name: [round(float(x), 1) for x in r] for name, r in final.items()})
lab.metric("regret_eps", round(iqm(final["eps-greedy"]), 1))  # room code: the three IQMs
lab.metric("regret_opt", round(iqm(final["optimistic"]), 1))
lab.metric("regret_mbie", round(iqm(final["MBIE-EB"]), 1))
fig, ax = plt.subplots()
for j, (name, cs) in enumerate(curves.items()):
    for i, c in enumerate(cs):  # every seed, one line each
        ax.plot(
            c,
            ls=["-", "--", ":"][j],
            lw=1.2,
            color=prl.plot.SERIES[j],
            label=name if i == 0 else None,
        )
ax.set(xlabel="step", ylabel="cumulative regret", title=f"Regret, {budget.seeds} seeds each")
ax.legend()
plt.show()

# %% [markdown] role="ex6-explain"
# Each verdict comes with its budget. Which learner's regret grows in a straight line to the
# end, and what is that line's slope in reward per step (regret)? Optimistic Q-learning
# reaches the right end, yet its regret is far above MBIE-EB's: compare their visitation in
# Exercises 4 and 5. Where do its extra steps go?

# %% role="ex6-chk"
lab.check(6, m06.check_verdict, verdict)

# %% [markdown] role="ex7-head"
# **Planted bug.** A colleague wrote their own MBIE-EB agent. It counts, estimates the model
# with Lab 2's `estimate_model`, plans with value iteration, adds the bonus c/√N(s, a), and
# acts greedily. It runs, and it uses the same c. Run it, read its visitation and the value
# its plan gives each state, find the line, and fix it.

# %% [markdown] role="ex7-predict"
# How many of the 5000 steps will the colleague's agent spend at the right end? Will it spend
# more or less than half of them at the left end?


# %% role="ex7-stub"
def mbie_eb_suspect(env, steps, c, gamma, seed):
    """A colleague's MBIE-EB. Returns (states, actions, Q) like run_planner."""
    # TODO 7: once the diagnostics below show the problem, find what is wrong and fix it. Then
    # rerun this cell and the run cell.
    n_s, n_a = env.observation_space.n, env.action_space.n
    log, states, actions = [], [], []
    counts = np.zeros((n_s, n_a), dtype=int)
    s, _ = env.reset(seed=seed)
    stale = True
    for _ in range(steps):
        if stale:
            P_hat, R_hat, N = estimate_model(log, n_s, n_a)
            Q = value_iteration(P_hat, R_hat, gamma, TOL)[1]
            bonus = c / np.sqrt(np.maximum(N, 1))
        a = int(greedy(Q[[s]] + bonus[[s]])[0])
        s2, r, terminated, truncated, _ = env.step(a)
        log.append((s, a, r, s2))
        states.append(s)
        actions.append(a)
        counts[s, a] += 1
        stale = counts[s, a] & (counts[s, a] - 1) == 0
        s = s2
    states.append(s)
    return np.array(states), np.array(actions), Q


# %% [markdown] role="ex7-hint"
# Print the value the colleague's plan gives a state the agent has never reached. Then find
# where the bonus enters: in the reward that value iteration backs up, or only in the
# comparison that picks the action?


# %% role="ex7-sol"
@lab.solution(7)
def mbie_eb_suspect(env, steps, c, gamma, seed):
    """The colleague's agent, fixed: the bonus goes into the backup."""
    n_s, n_a = env.observation_space.n, env.action_space.n
    log, states, actions = [], [], []
    counts = np.zeros((n_s, n_a), dtype=int)
    s, _ = env.reset(seed=seed)
    stale = True
    for _ in range(steps):
        if stale:
            P_hat, R_hat, N = estimate_model(log, n_s, n_a)
            Q = mbie_eb_backup(P_hat, R_hat, N, gamma, c, TOL)
        a = int(greedy(Q[[s]])[0])
        s2, r, terminated, truncated, _ = env.step(a)
        log.append((s, a, r, s2))
        states.append(s)
        actions.append(a)
        counts[s, a] += 1
        stale = counts[s, a] & (counts[s, a] - 1) == 0
        s = s2
    states.append(s)
    return np.array(states), np.array(actions), Q


# %% role="ex7-run"
states_s, actions_s, Q_s = mbie_eb_suspect(
    make_river(N_CHAIN, budget.steps), budget.steps, C_BONUS, GAMMA, seed=RUN_SEED
)
v_s = visitation(states_s[:-1], S)
hit = first_visit(states_s, END)
print(f"The colleague's agent, seed {RUN_SEED}:")
print(f"  first visit to the right end: {'never' if hit is None else f'step {hit}'}")
print("  fraction of steps in each state:", " ".join(f"{v:.3f}" for v in v_s))
print("  value of each state in its last plan, max over a of Q:")
print("   ", " ".join(f"{v:6.2f}" for v in Q_s.max(axis=1)))
print("  the same for MBIE-EB's last plan (Exercise 5):")
print("   ", " ".join(f"{v:6.2f}" for v in Q_mb.max(axis=1)))
lab.metric("suspect_end", round(float(v_s[END]), 3))  # room code: after your fix
fig, (ax_v, ax_q) = plt.subplots(1, 2, figsize=(10, 3.6))
ax_v.bar(x - 0.2, v_s, width=0.4, color=prl.plot.SERIES[3], hatch="//", label="colleague")
ax_v.bar(x + 0.2, v_mb, width=0.4, color=prl.plot.SERIES[2], label="MBIE-EB (Ex. 5)")
ax_v.set(xlabel="state", ylabel="fraction of steps", title="Where it went")
ax_v.legend()
ax_q.plot(x, Q_s.max(axis=1), ls="--", marker="^", color=prl.plot.SERIES[3], label="colleague")
ax_q.plot(x, Q_mb.max(axis=1), marker="o", color=prl.plot.SERIES[2], label="MBIE-EB (Ex. 5)")
ax_q.set(xlabel="state", ylabel="max over a of the plan's Q", title="What its plan believes")
ax_q.legend()
plt.show()

# %% [markdown] role="ex7-explain"
# Before your fix, the colleague's plan gave every state it had not reached a value of 0, and
# its visitation piled up near the left end, as ε-greedy's did. Its bonus was not idle: at
# each state it made the agent switch between the two actions as their counts grew. Why was
# that not enough to reach the right end? Which line puts the bonus where (MBIE-EB backup)
# says it belongs? (Gotcha: "Add the bonus when acting.")
#
# **Diagnostics for this lab.** For an exploring agent, plot its visitation against the two
# reference lines, and print the value its plan or its Q gives each state. A learner that
# never leaves the first few states, while its plan values every unreached state at 0, is
# not being drawn anywhere new: its optimism, if any, does not reach past the states it has
# seen. A healthy optimistic learner's plan values unreached states highly (MBIE-EB: at
# least c/(1 − γ)), its visitation moves right state by state, and its regret curve bends
# flat once it holds the right end. A regret curve that stays a straight line to the end
# means the agent never found the reward, whatever its code says about exploring.

# %% role="ex7-chk"
lab.check(7, m06.check_mbie_eb_suspect, mbie_eb_suspect)

# %% [markdown]
# **Your room code.** The last cell, `lab.finish()`, prints one line that starts with
# `ROOM m06`. It holds your seat, the checkpoints you passed, whether QUICK budgets ran, and
# the numbers the debrief pools: how many of your 10 optimistic seeds reached the right end
# (`opt_reached`, Exercise 4), MBIE-EB's fraction of steps at the right end (`mbie_end`,
# Exercise 5), the three regret IQMs (`regret_eps`, `regret_opt`, `regret_mbie`,
# Exercise 6), and your fixed agent's fraction at the right end (`suspect_end`, Exercise 7).
# Paste it where your instructor says. Your run record keeps every number the lab printed.

# %% [markdown] role="stretch"
# **Posterior sampling (PSRL).** Instead of a bonus, keep a posterior over models, sample one
# model each time you plan, and act greedily on the sample's optimal plan (Osband et al.
# 2013, s3). Write `psrl_sample` with our prior. Each row P̃(· | s, a) ~ Dirichlet(1/S +
# N(s, a, ·)), where N(s, a, s') counts the transitions seen (their Dirichlet prior is
# 1/S). Each reward r̃(s, a) ~ Normal((1 + N r̂)/(N + 1), 1/(N + 1)): the posterior of a
# mean with a Normal(1, 1) prior and noise of variance 1. The prior mean of 1 = r_max
# follows their footnote 4 (their reward prior is normal-gamma; ours fixes the variance).
# The runner re-samples when a count reaches a power of 2, close to the doubling rule they
# use without resets (s6.1). One seed: a fact beside MBIE-EB's, not a ranking.


# %% role="stretch-stub"
def psrl_sample(P_hat, R_hat, counts, rng):
    """One MDP (P~, R~) from the posterior, given estimate_model's output and an rng."""
    # Stretch: N(s, a, s') = P_hat * counts[:, :, None]; one rng.dirichlet per (s, a);
    # one rng.normal for all the rewards.
    raise NotImplementedError(
        "Stretch: write psrl_sample, or run lab.use_reference('stretch') to see the solution"
    )


# %% role="stretch-sol"
@lab.solution("stretch")
def psrl_sample(P_hat, R_hat, counts, rng):
    """Sample (P~, R~) from the posterior given the counts."""
    n_s, n_a = counts.shape
    N_next = P_hat * counts[:, :, None]  # N(s, a, s'); 0 for a pair never tried
    P_tilde = np.empty_like(P_hat)
    for s in range(n_s):
        for a in range(n_a):
            P_tilde[s, a] = rng.dirichlet(1.0 / n_s + N_next[s, a])
    R_tilde = rng.normal((1 + counts * R_hat) / (counts + 1), 1 / np.sqrt(counts + 1))
    return P_tilde, R_tilde


# %% role="stretch-run"
lab.check("stretch", m06.check_psrl_sample, psrl_sample, label="stretch")
rng_psrl = np.random.default_rng(RUN_SEED)


def plan_psrl(P_hat, R_hat, counts):
    """PSRL's plan: the optimal Q of one sampled model."""
    P_tilde, R_tilde = psrl_sample(P_hat, R_hat, counts, rng_psrl)
    return value_iteration(P_tilde, R_tilde, GAMMA, TOL)[1]


st, ac, _ = run_planner(make_river(N_CHAIN, budget.steps), budget.steps, plan_psrl, RUN_SEED)
hit = first_visit(st, END)
print(
    f"PSRL, seed {RUN_SEED}: first visit {'never' if hit is None else f'step {hit}'}; "
    f"regret {regret(st, ac)[-1]:.1f} (MBIE-EB on the same seed: "
    f"{regret(states_mb, actions_mb)[-1]:.1f}). Single seed, not a ranking."
)
