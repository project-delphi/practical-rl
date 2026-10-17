# ---
# jupyter:
#   prl:
#     module: m03
#     pins:
#       gymnasium: "1.4.0"
# ---

# %% [markdown]
# In this lab the model is gone. You estimate the value of a fixed policy from sampled
# episodes twice, by Monte Carlo and by TD(0), and measure the bias and variance of each
# against the exact answer from Lab 1. Then you learn to act: SARSA and Q-learning on the
# 4 × 12 cliff, where the two methods walk different paths. You finish by finding a bug in
# someone else's Q-learning that only an optimistic start can reveal.
#
# The briefing's equations are named in brackets, for example (MC) or (TD(0)). Its table
# "Equations and where you implement them" gives each name's equation number.

# %%
from typing import NamedTuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from prl import envs
from prl.checks import m03

ROWS, COLS, GAMMA = 4, 12, 0.95  # the cliff of Module 1's figure, discounted
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

envs.register()
env = gym.make("prl/CliffGridworld-v0", rows=ROWS, cols=COLS, gamma=GAMMA)
mdp = env.unwrapped.mdp  # the same P, R and gamma you wrote as arrays in Lab 1
S, A = mdp.n_states, mdp.n_actions
START, GOAL = mdp.start, mdp.goal
ABOVE_GOAL = GOAL - COLS  # the cell whose "down" is the final move of every route

# Budgets. Lab code reads them from here and never branches on QUICK. The control budget is
# the same in both: the design scan found the learners' greedy paths unreliable below about
# 1000 episodes at alpha 0.1, and 1000 episodes cost well under a second of compute.
budget = lab.use_settings(
    prl.runtime.settings(
        dict(
            episodes=1000, alpha=0.05, control_episodes=1000, control_alpha=0.1, window=500
        ),  # live
        dict(
            episodes=200, alpha=0.05, control_episodes=1000, control_alpha=0.1, window=500
        ),  # QUICK
        designed="colab-cpu",
    )
)
EPSILON = 0.1  # the exploration rate, constant throughout (Day 2 is about doing better)
# Your seat's seeds start here. Seat 0 does not reuse the recorded experiments' seeds
# (0-9) or the checkpoints' seeds, so every seat in the room adds new evidence.
RUN_SEED = (SEED + 1) * 1000
print(dict(budget))

# %% include labs/shared/evaluate_exact.py

# %% [markdown]
# ### The policy, the runner and the ground truth
#
# Prediction needs a fixed policy whose episodes end. The **ε-soft safe route** takes the
# route action (up the left column, right along the top row, down the right column) with
# probability 1 − ε, and a uniformly random action otherwise. Its exact value comes from
# Lab 1's `evaluate_exact`, included above as the reference version of your code.
#
# The runner uses the Gymnasium environment, so every transition carries `terminated`
# and `truncated` separately. Errors are measured on the 17 states of the route, which
# the policy visits on nearly every episode.


# %%
def safe_route_policy(epsilon):
    """(S, A) action probabilities of the epsilon-soft safe route."""
    pi = np.full((S, A), epsilon / A)
    for s in range(S):
        r, c = divmod(s, COLS)
        route_action = DOWN if c == COLS - 1 else (RIGHT if r == 0 else UP)
        pi[s, route_action] += 1.0 - epsilon
    return pi


def route_states(pi):
    """The states the route visits when no random move happens, start first, goal excluded."""
    s, route = START, [START]
    while s != GOAL:
        s = int(np.argmax(mdp.P[s, int(np.argmax(pi[s]))]))
        route.append(s)
    return route[:-1]


class Episode(NamedTuple):
    states: list  # s_0, ..., s_T
    actions: list  # a_0, ..., a_{T-1}
    rewards: list  # r_0, ..., r_{T-1}
    terminated: bool  # s_T is a terminal state
    truncated: bool  # the episode was cut by the step limit; s_T is not terminal


def run_episode(env, pi, rng):
    """One episode under the policy matrix pi, with explicit terminated / truncated flags."""
    s, _ = env.reset()
    states, actions, rewards = [int(s)], [], []
    while True:
        a = int(rng.choice(A, p=pi[s]))
        s, r, terminated, truncated, _ = env.step(a)
        actions.append(a)
        rewards.append(float(r))
        states.append(int(s))
        if terminated or truncated:
            return Episode(states, actions, rewards, bool(terminated), bool(truncated))


def collect(env, pi, n, seed):
    """n episodes under pi; all randomness comes from seed."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    return [run_episode(env, pi, rng) for _ in range(n)]


def rms_on_route(V):
    """Root-mean-square error of V against the exact values, over the route states."""
    return float(np.sqrt(np.mean((V[ROUTE] - V_EXACT[ROUTE]) ** 2)))


PI = safe_route_policy(EPSILON)
ROUTE = route_states(PI)
V_EXACT = evaluate_exact(mdp.P, mdp.R, GAMMA, PI)
P_pi = np.einsum("sa,sat->st", PI, mdp.P)
alive = [s for s in range(S) if s != GOAL]  # P(episode longer than n steps), exactly from P_pi
tail = np.zeros(len(alive))
tail[alive.index(START)] = 1.0
tail = tail @ np.linalg.matrix_power(P_pi[np.ix_(alive, alive)], 50)
print(f"Route: {len(ROUTE)} states; exact V(start) = {V_EXACT[START]:.3f}")
print(
    f"Return of a clean episode on the safe route (17 moves): {-(1 - GAMMA**17) / (1 - GAMMA):.3f}"
)
print(f"P(episode longer than 50 steps) = {tail.sum():.1e}; the step limit is 500, so no")
print("episode is truncated in practice, and the returns are complete (the briefing says why).")

# %% [markdown] role="ex1-head"
# **First-visit Monte Carlo.** The return G_t after a visit to s is an unbiased sample of
# V^π(s). First-visit MC averages, for each state, the return after its first visit in each
# episode (MC). Work backward through each episode with G = r_t + γ G, then average.

# %% [markdown] role="ex1-predict"
# Every episode starts at the start cell, and a fall sends the agent back there. After N
# episodes, how many returns does first-visit MC average for the start: N, more than N, or
# fewer? Will your first episode's return be a clean episode's −11.64 (printed above), or
# lower?


# %% role="ex1-stub"
def mc_prediction(episodes, gamma, n_states):
    """First-visit Monte Carlo: V[s] = mean return after the first visit to s in each episode.

    Each episode has states s_0..s_T, rewards r_0..r_{T-1}; s_T is never updated.
    """
    V, N = np.zeros(n_states), np.zeros(n_states)
    # TODO 1: for each episode, compute G_t for every t from the end backward, then for each
    # state's FIRST visit add its return to the running mean (keep the count in N).
    raise NotImplementedError(
        "TODO 1: write mc_prediction, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# Two passes per episode. Backward: `G = r + gamma * G` fills a list of returns, one per
# time step. Forward: a `set` of states seen so far in this episode tells you which visit is
# the first. The running mean is `V[s] += (G_t - V[s]) / N[s]` after `N[s] += 1`.


# %% role="ex1-sol"
@lab.solution(1)
def mc_prediction(episodes, gamma, n_states):
    """First-visit MC: V[s] averages the returns after first visits."""
    V, N = np.zeros(n_states), np.zeros(n_states)
    for ep in episodes:
        G, returns = 0.0, [0.0] * len(ep.rewards)
        # G_t = r_t + gamma G_{t+1}, from the end of the episode back
        for t in reversed(range(len(ep.rewards))):
            G = ep.rewards[t] + gamma * G
            returns[t] = G
        seen = set()
        for t, s in enumerate(ep.states[:-1]):
            if s in seen:  # first visit only (MC)
                continue
            seen.add(s)
            N[s] += 1
            V[s] += (returns[t] - V[s]) / N[s]
    return V


# %% role="ex1-run"
episodes = collect(env, PI, budget.episodes, seed=RUN_SEED)
lab.seeds([RUN_SEED])
lengths = [len(ep.rewards) for ep in episodes]
print(
    f"{len(episodes)} episodes; mean length {np.mean(lengths):.1f}, longest {max(lengths)}, "
    f"truncated: {sum(ep.truncated for ep in episodes)}"
)
first = episodes[0]
print(f"Return of your first episode: {mc_prediction([first], GAMMA, S)[START]:.3f}")
n_start = sum(START in ep.states[:-1] for ep in episodes)
print(f"First-visit returns averaged for the start: {n_start} (episodes: {len(episodes)})")
V_mc = mc_prediction(episodes, GAMMA, S)
mc_rms = rms_on_route(V_mc)
lab.metric("mc_rms", round(mc_rms, 4))
print(
    f"MC: V(start) = {V_mc[START]:.3f} (exact {V_EXACT[START]:.3f}); RMS on the route {mc_rms:.3f}"
)
fig, ax = plt.subplots()
ax.plot(V_EXACT[ROUTE], marker="o", label="exact (Lab 1)")
ax.plot(V_mc[ROUTE], marker="s", label=f"first-visit MC, {budget.episodes} episodes")
ax.set(xlabel="position along the route (0 = start)", ylabel="V", title="Monte Carlo on the route")
ax.legend()
plt.show()

# %% [markdown] role="ex1-explain"
# Your first episode's return is one sample of V(start); the estimate is the average of all of
# them (MC). The exact V(start), about −15.4, is below a clean episode's −11.64, the single most
# likely return: which rare episodes pull the expectation down? And as the number of episodes
# grows, what shrinks: the bias, the variance, or both?

# %% role="ex1-chk1"
lab.check(1, m03.check_mc_prediction, mc_prediction, label="1.1")

# %% role="ex1-chk2"
lab.check(1, m03.check_mc_rms, mc_prediction, label="1.2")

# %% [markdown] role="ex2a-head"
# **TD(0).** Replace the tail of the return by the current estimate: the target is
# r_t + γ V(s_{t+1}), and V(s_t) moves a fraction α of the way toward it (TD(0)). If
# s_{t+1} is terminal its value is 0, so the target is r_t alone; if the episode was merely
# truncated, keep the bootstrap (terminal mask). Process transitions in time order.

# %% [markdown] role="ex2a-predict"
# TD(0) with α = 0.05 starts from V = 0 and processes your first episode in time order.
# Afterward, what is V at the cell directly above the goal, and what is V(start)? Compare
# V(start) with the MC return of the same episode, which Exercise 1 printed.


# %% role="ex2a-stub"
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):
    """TD(0): for each transition (s, r, s2) in order, V[s] += alpha (target - V[s]).

    The target bootstraps from V[s2] unless s2 is terminal (the last state of an episode
    with terminated=True). A truncated episode's last transition still bootstraps.
    v0 is the starting value of every state.
    """
    V = np.full(n_states, float(v0))
    # TODO 2a: loop over episodes and their transitions; build the target with the terminal
    # rule, then move V[s] toward it.
    raise NotImplementedError(
        "TODO 2a: write td0_prediction, or run lab.use_reference('2a') to continue with the solution"
    )


# %% [markdown] role="ex2a-hint"
# `zip(ep.states[:-1], ep.rewards, ep.states[1:], strict=True)` walks the transitions. The next state is
# terminal only on the last transition of an episode whose `terminated` flag is True.


# %% role="ex2a-sol"
@lab.solution("2a")
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):
    """TD(0): move V[s] toward r + gamma V[s2], step by step."""
    V = np.full(n_states, float(v0))
    for ep in episodes:
        last = len(ep.rewards) - 1
        s_next = ep.states[1:]
        steps = zip(ep.states[:-1], ep.rewards, s_next, strict=True)
        for t, (s, r, s2) in enumerate(steps):
            # (terminal mask): bootstrap unless s2 is terminal
            terminal = ep.terminated and t == last
            target = r + gamma * (0.0 if terminal else V[s2])  # (TD(0))
            V[s] += alpha * (target - V[s])
    return V


# %% role="ex2a-run"
V_one = td0_prediction(episodes[:1], budget.alpha, GAMMA, S)
print(
    f"After one episode: V(cell above the goal) = {V_one[ABOVE_GOAL]:.4f}, V(start) = {V_one[START]:.4f}"
)
first_states = episodes[0].states[:-1]
print(
    f"  (your first episode visited the cell above the goal {first_states.count(ABOVE_GOAL)} "
    f"time(s) and the start {first_states.count(START)} time(s); each later visit moves V further)"
)
V_td = td0_prediction(episodes, budget.alpha, GAMMA, S)
td_rms = rms_on_route(V_td)
lab.metric("td_rms", round(td_rms, 4))
print(
    f"TD(0), alpha {budget.alpha}: V(start) = {V_td[START]:.3f} (exact {V_EXACT[START]:.3f}); "
    f"RMS on the route {td_rms:.3f} (MC: {mc_rms:.3f})"
)
fig, ax = plt.subplots()
ax.plot(V_EXACT[ROUTE], marker="o", label="exact (Lab 1)")
ax.plot(V_mc[ROUTE], marker="s", label="first-visit MC")
ax.plot(V_td[ROUTE], marker="^", label=f"TD(0), alpha {budget.alpha}")
ax.set(xlabel="position along the route (0 = start)", ylabel="V", title="MC and TD(0) on the route")
ax.legend()
plt.show()

# %% [markdown] role="ex2a-explain"
# Every target in your first episode was built from values still at 0, so the start moved only
# α of the way toward −1, while MC's estimate from the same episode is the whole return. That
# gap is the bias of the TD target (TD bias): which values must improve before the start's
# target is right? After all the episodes, is your TD(0) error above or below MC's?

# %% role="ex2a-chk1"
lab.check("2a", m03.check_td0_prediction, td0_prediction, label="2a.1")

# %% role="ex2a-chk2"
lab.check("2a", m03.check_td0_rms, td0_prediction, label="2a.2")

# %% [markdown] role="ex2b-head"
# **Bias² and variance over 10 seeds.** Run a method on 10 seeds' worth of episodes and you get
# 10 estimates of each state's value. Their mean squared error against the exact V splits
# exactly into the squared bias of their mean and their spread around that mean
# (decomposition). Write the split; the provided cell then runs MC (1/N), constant-α MC and
# TD(0) on the same 10 seeds and reports both terms for each.

# %% [markdown] role="ex2b-predict"
# Which method will have the lowest variance: MC (1/N), constant-α MC or TD(0)? Which will
# have the largest bias²?


# %% role="ex2b-stub"
def bias2_variance(estimates, v_true):
    """Squared bias and variance of estimates over seeds (decomposition).

    estimates has shape (seeds, states), one row per seed; v_true has shape (states,).
    Returns (bias2, variance): the mean over states of (mean over seeds - v_true)^2, and the
    mean over states of the spread over seeds around that mean (divide by the number of
    seeds, not seeds - 1).
    """
    estimates = np.asarray(estimates, dtype=float)
    v_true = np.asarray(v_true, dtype=float)
    # TODO 2b: two lines: the mean over seeds for each state, then the two averages over states.
    raise NotImplementedError(
        "TODO 2b: write bias2_variance, or run lab.use_reference('2b') to continue with the solution"
    )


# %% [markdown] role="ex2b-hint"
# `estimates.mean(axis=0)` is one number per state. bias² averages `(mean - v_true) ** 2` over
# states; the variance averages `estimates.var(axis=0)` over states (NumPy divides by the
# number of seeds by default). Check yourself: the two add up to the average over seeds and
# states of `(estimates - v_true) ** 2`.


# %% role="ex2b-sol"
@lab.solution("2b")
def bias2_variance(estimates, v_true):
    """Squared bias and variance over seeds (decomposition)."""
    estimates = np.asarray(estimates, dtype=float)
    v_true = np.asarray(v_true, dtype=float)
    mean = estimates.mean(axis=0)  # one number per state
    bias2 = float(np.mean((mean - v_true) ** 2))
    variance = float(np.mean(estimates.var(axis=0)))  # divide by seeds
    return bias2, variance


# %% role="ex2b-run"
def mc_constant_alpha(episodes, alpha, gamma, n_states):
    """Provided: first-visit MC with a constant step size (the fair partner of TD(0) at alpha)."""
    V = np.zeros(n_states)
    for ep in episodes:
        G, returns = 0.0, [0.0] * len(ep.rewards)
        for t in reversed(range(len(ep.rewards))):
            G = ep.rewards[t] + gamma * G
            returns[t] = G
        seen = set()
        for t, s in enumerate(ep.states[:-1]):
            if s in seen:
                continue
            seen.add(s)
            V[s] += alpha * (returns[t] - V[s])
    return V


# 10 seeds per seat, after your seat's base seed, so the room's seeds differ.
seeds = [RUN_SEED + 100 + k for k in range(10)]
lab.seeds(seeds)
estimates = {"mc": [], "mcc": [], "td": []}
for k in seeds:
    eps_k = collect(env, PI, budget.episodes, seed=k)
    estimates["mc"].append(mc_prediction(eps_k, GAMMA, S)[ROUTE])
    estimates["mcc"].append(mc_constant_alpha(eps_k, budget.alpha, GAMMA, S)[ROUTE])
    estimates["td"].append(td0_prediction(eps_k, budget.alpha, GAMMA, S)[ROUTE])
table = {}
names = {"mc": "MC (1/N)", "mcc": "MC (constant alpha)", "td": "TD(0)"}
print(f"{budget.episodes} episodes, alpha {budget.alpha}, 10 seeds; averages over the route states")
print(f"{'method':22s} {'bias^2':>8s} {'variance':>9s} {'sum':>8s} {'MSE':>8s}")
for key, rows in estimates.items():
    est = np.array(rows)
    bias2, var = bias2_variance(est, V_EXACT[ROUTE])
    mse = float(np.mean((est - V_EXACT[ROUTE]) ** 2))  # computed directly, to compare
    table[key] = {"bias2": round(bias2, 4), "variance": round(var, 4)}
    print(f"{names[key]:22s} {bias2:8.3f} {var:9.3f} {bias2 + var:8.3f} {mse:8.3f}")
lab.metric("decomposition", table)  # all six numbers, in the run record
lab.metric("td_bias2", table["td"]["bias2"])  # the three the debrief pools, in the room code
lab.metric("td_var", table["td"]["variance"])
lab.metric("mcc_var", table["mcc"]["variance"])

# %% [markdown] role="ex2b-explain"
# TD(0)'s bias² is the largest in the table. Which quantity in (TD bias) is still large after
# this many episodes, and why does it shrink as the budget grows? At the same α, is TD(0)'s
# variance below constant-α MC's on your seat? Ten seeds give one rough number per method, and
# some seats will see the order flip: write down your seat's order; the debrief counts how
# many seats got each order before anyone concludes. (Gotcha: "TD is biased, so MC is
# better.")

# %% role="ex2b-chk"
lab.check("2b", m03.check_bias2_variance, bias2_variance)

# %% [markdown] role="ex3-head"
# **The n-step target.** Between TD(0) (n = 1) and Monte Carlo (n ≥ T) sits the n-step
# return: n sampled rewards, then a bootstrap discounted by γⁿ (n-step). Two end cases: if
# t + n reaches the end of a *terminated* episode, there is nothing to bootstrap from; if it
# reaches the end of a *truncated* one, bootstrap from the last state with γ^(T − t).

# %% [markdown] role="ex3-predict"
# An episode of 4 moves, each −1, that terminates. V = −2 for every nonterminal state and
# γ = 0.9. With n = 2, what is the target for t = 0? And for t = 2, where the two steps run
# into the terminal state?


# %% role="ex3-stub"
def n_step_target(rewards, states, t, n, V, gamma, terminated):
    """G_t^(n): rewards r_t..r_{t+n-1} discounted, plus gamma^n V[s_{t+n}] if t + n < T.

    rewards has T entries, states has T + 1 (the last is s_T). If t + n >= T, the sum runs to
    the end; then add gamma^(T - t) V[s_T] if the episode was truncated (not terminated),
    and nothing if it terminated.
    """
    # TODO 3: sum the rewards from t to min(t + n, T) - 1 with increasing powers of gamma,
    # then handle the three cases for the bootstrap term.
    raise NotImplementedError(
        "TODO 3: write n_step_target, or run lab.use_reference(3) to continue with the solution"
    )


# %% [markdown] role="ex3-hint"
# Let `T = len(rewards)` and `end = min(t + n, T)`. The rewards run from `t` to `end - 1`. The
# bootstrap, if there is one, is `V[states[end]]`, discounted by γ^(end − t). Now list the
# three cases: `end < T`; `end == T` and terminated; `end == T` and truncated. Which of them
# add the bootstrap?


# %% role="ex3-sol"
@lab.solution(3)
def n_step_target(rewards, states, t, n, V, gamma, terminated):
    """G_t^(n): up to n discounted rewards, then a bootstrap."""
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
    # bootstrap from s_end unless the episode terminated there
    if end < T or not terminated:
        G += gamma ** (end - t) * V[states[end]]
    return G


# %% role="ex3-run"
corridor = Episode(
    states=[0, 1, 2, 3, 4],
    actions=[1, 1, 1, 1],
    rewards=[-1.0] * 4,
    terminated=True,
    truncated=False,
)
V_flat = np.full(5, -2.0)
print("Terminated 4-step episode, V = -2, gamma 0.9, n = 2:")
for t in range(4):
    print(
        f"  t = {t}: G^(2) = {n_step_target(corridor.rewards, corridor.states, t, 2, V_flat, 0.9, True):.4f}"
    )
cut = corridor._replace(terminated=False, truncated=True)
print(
    f"The same episode truncated instead: t = 2 gives {n_step_target(cut.rewards, cut.states, 2, 2, V_flat, 0.9, False):.4f}"
)
ep = episodes[0]
print(f"Your first episode (T = {len(ep.rewards)}), targets for t = 0 with your TD(0) values:")
for n in (1, 2, 4, 8, 16, 1000):
    print(
        f"  n = {n:4d}: {n_step_target(ep.rewards, ep.states, 0, n, V_td, GAMMA, ep.terminated):.4f}"
    )
print(f"  MC return of the episode:   {mc_prediction([ep], GAMMA, S)[START]:.4f}")

# %% [markdown] role="ex3-explain"
# At n = 1 the target is TD(0)'s; once n reaches T it equals the Monte Carlo return, and
# the truncated version keeps a bootstrap where the terminated one has none (n-step). As n
# grows, which part of the target carries more sampled randomness, and which part carries
# less of V's error? The stretch sweeps n and α to see where the two meet.

# %% role="ex3-chk"
lab.check(3, m03.check_n_step_target, n_step_target)

# %% [markdown] role="ex4-head"
# **SARSA.** Now the policy is not fixed: act ε-greedily on Q while learning Q. SARSA's
# target bootstraps from the action the agent actually takes next, Q(s', a') (SARSA), so it
# learns the value of the ε-greedy policy itself, exploration included. `epsilon_greedy`,
# a greedy-path extractor and a grid printer are provided.


# %%
def epsilon_greedy(Q, s, epsilon, rng):
    """Provided: the greedy action with probability 1 - epsilon, else uniform (ties at random)."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def greedy_path(Q, max_steps=None):
    """Provided: the states visited from the start by always taking argmax Q, until the goal,
    a revisit (a loop) or max_steps. Returns (path, reached_goal)."""
    s, path = START, [START]
    for _ in range(max_steps or S):
        if s == GOAL:
            return path, True
        s = int(np.argmax(mdp.P[s, int(np.argmax(Q[s]))]))
        if s in path:
            return path + [s], False
        path.append(s)
    return path, s == GOAL


def show_path(path, label):
    """Provided: draw the grid as text; '.' free, 'C' cliff, '*' the path, S start, G goal."""
    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
    for s in mdp.cliff:
        grid[s // COLS][s % COLS] = "C"
    for s in path:
        grid[s // COLS][s % COLS] = "*"
    grid[START // COLS][START % COLS], grid[GOAL // COLS][GOAL % COLS] = "S", "G"
    print(f"{label}: {len(path) - 1} moves")
    print("\n".join(" ".join(row) for row in grid))


def smooth(x, k=20):
    """Provided: moving average over k episodes, for plotting noisy online returns."""
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(k) / k, mode="valid")


# %% [markdown] role="ex4-predict"
# SARSA's greedy path will take some number of moves m (the run prints it). Over the last 500
# episodes, will SARSA's online return per episode be above −m, about −m, or below −m? Why?


# %% role="ex4-stub"
def sarsa(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    """SARSA on a Gymnasium env. Returns (Q, online), online = the reward sum of each episode.

    Start Q at q0 (a number, or an array to continue from). All randomness comes from seed:
    env.reset(seed=seed) once, and rng = np.random.default_rng(seed) for epsilon_greedy.
    """
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)
    online = []
    env.reset(seed=seed)
    # TODO 4: for each episode: reset, pick a; loop: step, pick a2 from s2, update Q[s, a]
    # toward r + gamma Q[s2, a2] (0 if terminated), move on; stop on terminated or truncated.
    raise NotImplementedError(
        "TODO 4: write sarsa, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# The next action a2 is chosen before the update and then *used* as the next step's action:
# that is what makes it on-policy. The bootstrap is `0.0 if terminated else Q[s2, a2]`;
# `truncated` only ends the episode.


# %% role="ex4-sol"
@lab.solution(4)
def sarsa(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    """SARSA. Returns (Q, online): one reward sum per episode."""
    rng = np.random.default_rng(seed)
    if np.isscalar(q0):
        Q = np.full((S, A), float(q0))
    else:
        Q = np.array(q0, dtype=float)
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        total, done = 0.0, False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            # the action actually taken next (SARSA)
            a2 = epsilon_greedy(Q, s2, epsilon, rng)
            # (terminal mask): no bootstrap from a terminal state
            target = r + gamma * (0.0 if terminated else Q[s2, a2])
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, a, done = s2, a2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


# %% role="ex4-run"
Q_sarsa, online_sarsa = sarsa(
    env, budget.control_episodes, budget.control_alpha, EPSILON, GAMMA, seed=RUN_SEED
)
path_sarsa, reached = greedy_path(Q_sarsa)
show_path(path_sarsa, "SARSA's greedy path" + ("" if reached else " (does not reach the goal)"))
edge = [s for s in path_sarsa if s // COLS == ROWS - 2 and 1 <= s % COLS <= COLS - 2]
print(f"Cells of the path directly above the cliff: {len(edge)}")
print(
    f"Online return, mean of the last {budget.window} episodes: {online_sarsa[-budget.window :].mean():.1f}"
)
fig, ax = plt.subplots()
ax.plot(smooth(online_sarsa), label="SARSA (moving average of 20)")
ax.set(xlabel="episode", ylabel="reward sum per episode", title="Online return while learning")
ax.legend()
plt.show()

# %% [markdown] role="ex4-explain"
# A non-greedy "down" fires with probability ε/4 = 0.025 and costs 100, so an edge cell is
# worth about 2.5 less per visit than one farther up, under the policy SARSA evaluates. Which
# of the two bootstraps, Q(s', a') (SARSA) or max Q(s', ·), is the one that sees this cost,
# and why? Read your path: how many rows up does it go before heading right?

# %% role="ex4-chk"
lab.check(4, m03.check_sarsa, sarsa)

# %% [markdown] role="ex5-head"
# **Q-learning and the cliff contrast.** One symbol changes: the target bootstraps from
# max over a' of Q(s', a'), whatever action the agent takes next (Q-learning). It estimates
# Q* no matter how it explores. Then compare the two methods on your seed: the greedy paths,
# and the online returns earned by their ε-greedy behavior while learning.

# %% [markdown] role="ex5-predict"
# By about how much per episode will Q-learning's online return over the last 500 episodes be
# below SARSA's: about 3, about 30, or about 300? Use the briefing's chance of a fall on one
# traverse of the edge, 0.24.


# %% role="ex5-stub"
def q_learning(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    """Q-learning on a Gymnasium env. Returns (Q, online), online = the reward sum per episode.

    Same conventions as sarsa: start at q0; env.reset(seed=seed) once; rng from seed.
    """
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)
    online = []
    env.reset(seed=seed)
    # TODO 5: as in sarsa, but the target uses Q[s2].max() (0 if terminated), and the next
    # action is chosen from s2 after the update.
    raise NotImplementedError(
        "TODO 5: write q_learning, or run lab.use_reference(5) to continue with the solution"
    )


# %% [markdown] role="ex5-hint"
# Choose `a = epsilon_greedy(Q, s, ...)` at the top of each step, take it, then update with
# `Q[s2].max()`. The action chosen next is not part of the target.


# %% role="ex5-sol"
@lab.solution(5)
def q_learning(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    """Q-learning. Returns (Q, online): one reward sum per episode."""
    rng = np.random.default_rng(seed)
    if np.isscalar(q0):
        Q = np.full((S, A), float(q0))
    else:
        Q = np.array(q0, dtype=float)
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = epsilon_greedy(Q, s, epsilon, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            # the best next action, taken or not (Q-learning)
            target = r + gamma * (0.0 if terminated else Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, done = s2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


# %% role="ex5-run"
Q_q, online_q = q_learning(
    env, budget.control_episodes, budget.control_alpha, EPSILON, GAMMA, seed=RUN_SEED
)
path_q, reached_q = greedy_path(Q_q)
show_path(path_q, "Q-learning's greedy path" + ("" if reached_q else " (does not reach the goal)"))
show_path(path_sarsa, "SARSA's greedy path")
mean_sarsa = float(online_sarsa[-budget.window :].mean())
mean_q = float(online_q[-budget.window :].mean())
moves_sarsa = len(path_sarsa) - 1 if reached else None  # None: the path loops
moves_q = len(path_q) - 1 if reached_q else None
lab.metric("online", {"sarsa": round(mean_sarsa, 2), "q_learning": round(mean_q, 2)})
lab.metric("path_moves", {"sarsa": moves_sarsa, "q_learning": moves_q})
lab.metric("sarsa_lead", round(mean_sarsa - mean_q, 2))  # room code
lab.metric("sarsa_moves", moves_sarsa)  # room code; left out when the path loops
print(f"Online return over the last {budget.window} episodes, your seed {RUN_SEED}:")
print(
    f"  SARSA {mean_sarsa:7.1f}   Q-learning {mean_q:7.1f}   difference {mean_sarsa - mean_q:+.1f}"
)
print("Single seed: a fact to put beside the threshold and the room's pooled seeds, not a ranking.")
fig, ax = plt.subplots()
ax.plot(smooth(online_sarsa), label="SARSA")
ax.plot(smooth(online_q), label="Q-learning")
ax.set(
    xlabel="episode",
    ylabel="reward sum per episode (moving average of 20)",
    title=f"The cliff contrast, epsilon {EPSILON}, alpha {budget.control_alpha}, seed {RUN_SEED}",
)
ax.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# Q-learning's greedy path is the 13-move edge route, SARSA's is longer, and yet SARSA collects
# more reward while learning. Both agents explore with the same ε: whose exploration is
# expensive, and why does Q-learning's target not care? If ε were decayed to 0, what would
# happen to the gap? (Gotcha: "Q-learning is better than SARSA.")

# %% role="ex5-chk1"
lab.check(5, m03.check_q_learning_path, q_learning, label="5.1")

# %% role="ex5-chk2"
lab.check(5, m03.check_cliff_contrast, sarsa, q_learning, label="5.2")

# %% [markdown] role="ex6-head"
# **Planted bug.** A colleague wrote `q_learning_suspect`. It runs, its greedy path is the
# optimal route, and it passes any test that starts Q at 0. It is wrong. Train it from an
# optimistic start, Q₀ = 10, and read three things against what they must be: the return it
# earns while learning, max Q at the start against V*(start) = −(1 − γ¹³)/(1 − γ) ≈ −9.73 (the
# 13-move route), and Q of its last move (the cell above the goal, action down). Find the
# line, fix it.

# %% [markdown] role="ex6-predict"
# With Q₀ = 10 and γ = 0.95, what should Q(cell above the goal, down) converge to in a correct
# learner? Can any return in this world, from any state, be above −1? Write both, then run the
# cell below.


# %% role="ex6-stub"
def q_learning_suspect(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    """A colleague's Q-learning. It learns the optimal path. Is it right?"""
    # TODO 6: once the diagnostics below show the problem, find the one wrong line and fix it.
    # Then rerun this cell and the run cell.
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = epsilon_greedy(Q, s, epsilon, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            target = r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, done = s2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


# %% [markdown] role="ex6-hint"
# The suspect's last move settled at −1 + γ · 10. Which Q-values in the grid are still 10
# after training, and where in the target line could one of them come in?


# %% role="ex6-sol"
@lab.solution(6)
def q_learning_suspect(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    """The colleague's Q-learning, fixed: no bootstrap through a terminal state."""
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = epsilon_greedy(Q, s, epsilon, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            target = r + gamma * (0.0 if terminated else Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, done = s2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


# %% role="ex6-run"
V_STAR_START = -(1 - GAMMA**13) / (1 - GAMMA)  # the optimal route has 13 moves
Q0 = 10.0


def value_curve(learner, episodes, chunk, seed, q0):
    """Provided: train in chunks; record max_a Q(start, a) after each chunk and every episode's
    online return. Returns (xs, values, online, Q)."""
    Q, xs, values, online = q0, [], [], []
    for i in range(episodes // chunk):
        Q, on = learner(
            env, chunk, budget.control_alpha, EPSILON, GAMMA, seed=seed * 1000 + i, q0=Q
        )
        online.extend(on)
        xs.append((i + 1) * chunk)
        values.append(Q[START].max())
    return xs, values, np.array(online), Q


chunk = max(10, budget.control_episodes // 50)
xs, v_suspect, on_suspect, Q_suspect = value_curve(
    q_learning_suspect, budget.control_episodes, chunk, RUN_SEED, Q0
)
_, v_fixed, on_fixed, Q_fixed = value_curve(
    q_learning, budget.control_episodes, chunk, RUN_SEED, Q0
)
print(f"Optimistic start Q0 = {Q0}; after {xs[-1]} episodes:")
print(
    f"  suspect: Q(above goal, down) = {Q_suspect[ABOVE_GOAL, DOWN]:8.4f}   max Q(start) = {Q_suspect[START].max():8.4f}"
)
print(
    f"  yours:   Q(above goal, down) = {Q_fixed[ABOVE_GOAL, DOWN]:8.4f}   max Q(start) = {Q_fixed[START].max():8.4f}"
)
print(f"  truth:   Q*(above goal, down) = {-1.0:8.4f}   V*(start) = {V_STAR_START:8.4f}")
print(
    f"  Q(goal, ·) in your q_learning: {Q_fixed[GOAL]} (never updated: no action is taken in "
    "the goal, and with the mask no target reads it)"
)
show_path(greedy_path(Q_suspect)[0], "suspect's greedy path")
fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(10, 3.6))
ax_r.plot(smooth(on_suspect), label="q_learning_suspect")
ax_r.plot(smooth(on_fixed), label="your q_learning")
ax_r.set(xlabel="episode", ylabel="online return (moving average of 20)", title="What it earns")
ax_r.legend()
ax_v.plot(xs, v_suspect, label="q_learning_suspect")
ax_v.plot(xs, v_fixed, label="your q_learning")
ax_v.axhline(V_STAR_START, linestyle="--", color="gray", label="V*(start)")
ax_v.set(xlabel="episodes", ylabel="max Q(start, a)", title=f"What it believes, from Q0 = {Q0}")
ax_v.legend()
plt.show()

# %% [markdown] role="ex6-explain"
# Before your fix, the suspect's last move settled at −1 + γ · 10 = 8.5 and its start at about
# −4.6, both above the truth, yet its greedy path was optimal: a bonus that shrinks with
# distance cannot reorder the routes. Look at the left panel first: could you have found this
# bug from the returns? (Gotcha: "My returns went up, so it works.") Why did the same bug
# change nothing when Q started at 0, and would Sutton and Barto's choice, Q(terminal, ·) = 0
# instead of a mask, have fixed it here? (terminal mask) Would `0.0 if (terminated or
# truncated)` also pass this checkpoint, and why is it still wrong in CartPole? (Gotcha:
# "`done` means stop bootstrapping.")
#
# **Diagnostics for this lab.** For a learner, plot max Q(start) over episodes against a known
# V*(start), and print Q of the last move against its reward: a value curve that flattens at
# the wrong level, or a last move worth more than its reward, means the target is wrong, not
# the exploration. The return curve cannot show this: it measures the policy you follow, not
# the values you learned. Run every such test from an optimistic start as well as from zero;
# the terminal bug is invisible from zero. For a predictor, plot the estimate along the route
# against the exact V and read the error's shape: a constant offset points to the discount or
# the terminal rule, an offset that grows with distance from the goal points to a bootstrap
# that forgets its start too slowly (small α, few episodes), and a ragged curve with a correct
# mean points to variance, which more episodes or more seeds reduce. A greedy path that loops
# means Q is noisy where the agent rarely goes; it is not a bug.

# %% role="ex6-chk"
lab.check(6, m03.check_q_learning_suspect, q_learning_suspect)

# %% [markdown]
# **Your room code.** The last cell, `lab.finish()`, prints one line that starts with `ROOM m03`.
# It holds your seat, the checkpoints you passed, whether QUICK budgets ran, and the numbers the
# debrief pools: `mc_rms` and `td_rms` (Exercises 1 and 2a); TD(0)'s bias² and variance and
# constant-α MC's variance (`td_bias2`, `td_var`, `mcc_var`, Exercise 2b); SARSA's lead in
# online return and its greedy path's moves (`sarsa_lead`, `sarsa_moves`, Exercise 5;
# `sarsa_moves` is left out when the path loops). Paste it where your instructor says. Your
# run record keeps every number the lab printed.

# %% [markdown] role="stretch"
# **The n-step dial on your seed.** Build n-step TD prediction on your `n_step_target`: for
# each episode and each t in order, move V[s_t] a fraction α toward G_t^(n), computed with the
# values as they stand (the briefing's n-step update; Sutton and Barto's Eq. 7.2 makes the same
# updates in the same order, n steps later in time). Then the provided sweep runs n in
# {1, 2, 4, 8, 16} and several α on 5 seeds and plots the RMS error on the route: the
# briefing's figure, on your seat's episodes.


# %% role="stretch-stub"
def n_step_td_prediction(episodes, alpha, gamma, n_states, n):
    """n-step TD: V[s_t] += alpha (G_t^(n) - V[s_t]) for every t of every episode, in order."""
    # Stretch: loop over episodes and t; call n_step_target with the current V.
    raise NotImplementedError(
        "Stretch: write n_step_td_prediction, or run lab.use_reference('stretch') to see the solution"
    )


# %% role="stretch-sol"
@lab.solution("stretch")
def n_step_td_prediction(episodes, alpha, gamma, n_states, n):
    """n-step TD: V[s_t] += alpha (G_t^(n) - V[s_t]) for every t of every episode, in order."""
    V = np.zeros(n_states)
    for ep in episodes:
        for t, s in enumerate(ep.states[:-1]):
            G = n_step_target(ep.rewards, ep.states, t, n, V, gamma, ep.terminated)
            V[s] += alpha * (G - V[s])
    return V


# %% role="stretch-run"
lab.check("stretch", m03.check_n_step_td_prediction, n_step_td_prediction, label="stretch")
ns, alphas = [1, 2, 4, 8, 16], [0.02, 0.05, 0.1, 0.2]
stretch_seeds = [RUN_SEED + 200 + k for k in range(5)]
episodes_by_seed = [collect(env, PI, budget.episodes, seed=k) for k in stretch_seeds]
fig, ax = plt.subplots()
for alpha in alphas:
    errs = np.array(
        [
            [rms_on_route(n_step_td_prediction(eps_k, alpha, GAMMA, S, n)) for n in ns]
            for eps_k in episodes_by_seed
        ]
    )
    ax.plot(ns, errs.mean(axis=0), marker="o", label=f"alpha {alpha} (mean of 5 seeds)")
    ax.fill_between(ns, errs.min(axis=0), errs.max(axis=0), alpha=0.15)
ax.set(
    xscale="log",
    xlabel="n",
    ylabel="RMS error on the route",
    title=f"n-step TD after {budget.episodes} episodes",
)
ax.set_xticks(ns, [str(n) for n in ns])
ax.legend()
plt.show()
