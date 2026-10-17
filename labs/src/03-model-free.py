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
# the policy visits on every episode.


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
print(f"Return of a clean route (17 moves): {-(1 - GAMMA**17) / (1 - GAMMA):.3f}")
print(f"P(episode longer than 50 steps) = {tail.sum():.1e}; the step limit is 500, so no")
print("episode is truncated in practice, and the returns are complete (the briefing says why).")

# %% [markdown] role="ex1-head"
# **First-visit Monte Carlo.** The return G_t after a visit to s is an unbiased sample of
# V^π(s). First-visit MC averages, for each state, the return after its first visit in each
# episode (MC). Work backward through each episode with G = r_t + γ G, then average.

# %% [markdown] role="ex1-predict"
# Every episode starts at the start cell. After N episodes, how many returns does first-visit
# MC average for the start? And what is the return of an episode that follows the route with
# no random move at all: 17 moves of −1 at γ = 0.95? Compare it with the exact V(start)
# printed above before you run anything.


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
    """First-visit Monte Carlo: V[s] = mean return after the first visit to s in each episode."""
    V, N = np.zeros(n_states), np.zeros(n_states)
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
            N[s] += 1
            V[s] += (returns[t] - V[s]) / N[s]
    return V


# %% role="ex1-run"
episodes = collect(env, PI, budget.episodes, seed=SEED)
lengths = [len(ep.rewards) for ep in episodes]
print(
    f"{len(episodes)} episodes; mean length {np.mean(lengths):.1f}, longest {max(lengths)}, "
    f"truncated: {sum(ep.truncated for ep in episodes)}"
)
first = episodes[0]
print(f"Return of your first episode: {mc_prediction([first], GAMMA, S)[START]:.3f}")
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
# Your first episode's return is one sample of V(start); the average of all of them is the
# estimate. The clean route returns −11.64, yet the exact V(start) is about −15.4: where do
# the extra costs come from, and why are they in the expectation but not in every sample?
# Each return is unbiased for V(start) (MC), so what shrinks as the number of episodes grows:
# the bias, the variance, or both?

# %% role="ex1-chk1"
lab.check(1, m03.check_mc_prediction, mc_prediction, label="1.1")

# %% role="ex1-chk2"
lab.check(1, m03.check_mc_rms, mc_prediction, label="1.2")

# %% [markdown] role="ex2-head"
# **TD(0).** Replace the tail of the return by the current estimate: the target is
# r_t + γ V(s_{t+1}), and V(s_t) moves a fraction α of the way toward it (TD(0)). If
# s_{t+1} is terminal its value is 0, so the target is r_t alone; if the episode was merely
# truncated, keep the bootstrap (terminal mask). Process transitions in time order.

# %% [markdown] role="ex2-predict"
# TD(0) with α = 0.05 starts from V = 0 and processes one episode in time order. Suppose the
# cell directly above the goal is visited once, as the last state before the goal. What is V
# at that cell after the episode? Write the number.


# %% role="ex2-stub"
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):
    """TD(0): for each transition (s, r, s2) in order, V[s] += alpha (target - V[s]).

    The target bootstraps from V[s2] unless s2 is terminal (the last state of an episode
    with terminated=True). A truncated episode's last transition still bootstraps.
    v0 is the starting value of every state.
    """
    V = np.full(n_states, float(v0))
    # TODO 2: loop over episodes and their transitions; build the target with the terminal
    # rule, then move V[s] toward it.
    raise NotImplementedError(
        "TODO 2: write td0_prediction, or run lab.use_reference(2) to continue with the solution"
    )


# %% [markdown] role="ex2-hint"
# `zip(ep.states[:-1], ep.rewards, ep.states[1:], strict=True)` walks the transitions. The next state is
# terminal only on the last transition of an episode whose `terminated` flag is True.


# %% role="ex2-sol"
@lab.solution(2)
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):
    """TD(0): for each transition (s, r, s2) in order, V[s] += alpha (target - V[s])."""
    V = np.full(n_states, float(v0))
    for ep in episodes:
        last = len(ep.rewards) - 1
        for t, (s, r, s2) in enumerate(zip(ep.states[:-1], ep.rewards, ep.states[1:], strict=True)):
            terminal = ep.terminated and t == last
            target = r + gamma * (0.0 if terminal else V[s2])
            V[s] += alpha * (target - V[s])
    return V


# %% role="ex2-run"
V_one = td0_prediction(episodes[:1], budget.alpha, GAMMA, S)
print(
    f"After one episode: V(cell above the goal) = {V_one[ABOVE_GOAL]:.4f}, V(start) = {V_one[START]:.4f}"
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


# %% role="ex2-run2"
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


# Bias^2 and variance over 10 seeds (decomposition): rough with 10 seeds, and the room pools them.
seeds = [SEED * 100 + k for k in range(10)]
lab.seeds(seeds)
estimates = {"MC (1/N)": [], "MC (constant alpha)": [], "TD(0)": []}
for k in seeds:
    eps_k = collect(env, PI, budget.episodes, seed=k)
    estimates["MC (1/N)"].append(mc_prediction(eps_k, GAMMA, S)[ROUTE])
    estimates["MC (constant alpha)"].append(mc_constant_alpha(eps_k, budget.alpha, GAMMA, S)[ROUTE])
    estimates["TD(0)"].append(td0_prediction(eps_k, budget.alpha, GAMMA, S)[ROUTE])
decomposition = {}
print(f"{budget.episodes} episodes, alpha {budget.alpha}, 10 seeds; averages over the route states")
print(f"{'method':22s} {'bias^2':>8s} {'variance':>9s} {'MSE':>8s}")
for name, rows in estimates.items():
    est = np.array(rows)
    bias2 = float(np.mean((est.mean(axis=0) - V_EXACT[ROUTE]) ** 2))
    var = float(np.mean(est.var(axis=0)))
    decomposition[name] = dict(bias2=round(bias2, 4), variance=round(var, 4))
    print(f"{name:22s} {bias2:8.3f} {var:9.3f} {bias2 + var:8.3f}")
lab.metric("decomposition", decomposition)

# %% [markdown] role="ex2-explain"
# After one episode the cell above the goal holds α × (−1 + γ · 0) = −0.05: the goal is
# terminal, so its value is 0 by definition, not by estimate. Now read the table. For TD(0),
# which term is larger, bias² or variance, and where does the bias come from at this budget:
# the targets (TD bias) or the zero start that bootstrapping forgets one step at a time? At
# the same α, whose variance is lower, TD(0)'s or constant-α MC's? Ten seeds make rough
# numbers; compare with the room's pooled table in the debrief before you conclude.
# (Gotcha: "TD is biased, so MC is better.")

# %% role="ex2-chk1"
lab.check(2, m03.check_td0_prediction, td0_prediction, label="2.1")

# %% role="ex2-chk2"
lab.check(2, m03.check_td0_rms, td0_prediction, label="2.2")

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
# `T = len(rewards)`, `end = min(t + n, T)`. The bootstrap index is `end`; it is a real
# bootstrap when `end < T`, or when `end == T` and the episode was truncated.


# %% role="ex3-sol"
@lab.solution(3)
def n_step_target(rewards, states, t, n, V, gamma, terminated):
    """G_t^(n): rewards r_t..r_{t+n-1} discounted, plus gamma^n V[s_{t+n}] if t + n < T."""
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
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
# ε-greedy with ε = 0.1 and four actions, in a cell directly above the cliff where "down"
# is not the greedy action: what is the probability of stepping into the cliff on this move?
# Multiply by the cost of a fall to see what SARSA will think of that cell.


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
    """SARSA on a Gymnasium env. Returns (Q, online), online = the reward sum of each episode."""
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        total, done = 0.0, False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = epsilon_greedy(Q, s2, epsilon, rng)
            target = r + gamma * (0.0 if terminated else Q[s2, a2])
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, a, done = s2, a2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


# %% role="ex4-run"
Q_sarsa, online_sarsa = sarsa(
    env, budget.control_episodes, budget.control_alpha, EPSILON, GAMMA, seed=SEED
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
# How many moves does Q-learning's greedy path take from the start to the goal? Is that fewer
# or more than SARSA's? Then: whose online return over the last episodes will be higher?


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
    """Q-learning on a Gymnasium env. Returns (Q, online), online = the reward sum per episode."""
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


# %% role="ex5-run"
Q_q, online_q = q_learning(
    env, budget.control_episodes, budget.control_alpha, EPSILON, GAMMA, seed=SEED
)
path_q, reached_q = greedy_path(Q_q)
show_path(path_q, "Q-learning's greedy path" + ("" if reached_q else " (does not reach the goal)"))
show_path(path_sarsa, "SARSA's greedy path")
mean_sarsa = float(online_sarsa[-budget.window :].mean())
mean_q = float(online_q[-budget.window :].mean())
lab.metric("sarsa_online", round(mean_sarsa, 2))
lab.metric("q_online", round(mean_q, 2))
lab.metric("sarsa_path_moves", len(path_sarsa) - 1 if reached else None)
lab.metric("q_path_moves", len(path_q) - 1 if reached_q else None)
print(f"Online return over the last {budget.window} episodes, your seed {SEED}:")
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
    title=f"The cliff contrast, epsilon {EPSILON}, alpha {budget.control_alpha}, seed {SEED}",
)
ax.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# Q-learning's greedy path is the 13-move edge route, SARSA's is longer, and yet SARSA
# collects more reward while learning. Both agents explore with the same ε; whose
# exploration is expensive, and why does Q-learning's target not care? If ε were decayed to
# 0, what would happen to the two paths and to the gap? (Gotcha: "Q-learning is better than
# SARSA.")

# %% role="ex5-chk1"
lab.check(5, m03.check_q_learning_path, q_learning, label="5.1")

# %% role="ex5-chk2"
lab.check(5, m03.check_cliff_contrast, sarsa, q_learning, label="5.2")

# %% [markdown] role="ex6-head"
# **Planted bug.** A colleague wrote `q_learning_suspect`. It runs, its greedy path is the
# optimal route, and it passes any test that starts Q at 0. It is wrong. Start it from an
# optimistic Q₀ = 10 instead and read two numbers: Q of the final transition (the cell above
# the goal, action down), whose true value is its reward, −1; and max Q at the start against
# V*(start) = −(1 − γ¹³)/(1 − γ) ≈ −9.73 (the 13-move route). Find the line, fix it.

# %% [markdown] role="ex6-predict"
# With Q₀ = 10 and γ = 0.95, what value does Q(cell above the goal, down) converge to under
# the terminal mask? And without it, if the update bootstraps from Q(goal, ·), which nothing
# ever updates?


# %% role="ex6-stub"
def q_learning_suspect(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    """A colleague's Q-learning. It learns the optimal path. Is it right?"""
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
            # TODO 6: once the diagnostics below show the problem, fix the one wrong line.
            # After fixing it, rerun this cell and the run cell.
            target = r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, done = s2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


# %% [markdown] role="ex6-hint"
# Compare the target line with your own `q_learning`. Which flag is missing, and which of
# the two flags should it be? (terminal mask)


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
    """Provided: train in chunks and record max_a Q(start, a) after each; returns (xs, values, Q)."""
    Q, xs, values = q0, [], []
    for i in range(episodes // chunk):
        Q, _ = learner(env, chunk, budget.control_alpha, EPSILON, GAMMA, seed=seed * 1000 + i, q0=Q)
        xs.append((i + 1) * chunk)
        values.append(Q[START].max())
    return xs, values, Q


chunk = max(10, budget.control_episodes // 50)
xs, v_suspect, Q_suspect = value_curve(q_learning_suspect, budget.control_episodes, chunk, SEED, Q0)
_, v_fixed, Q_fixed = value_curve(q_learning, budget.control_episodes, chunk, SEED, Q0)
print(f"Optimistic start Q0 = {Q0}; after {xs[-1]} episodes:")
print(
    f"  suspect: Q(above goal, down) = {Q_suspect[ABOVE_GOAL, DOWN]:8.4f}   max Q(start) = {Q_suspect[START].max():8.4f}"
)
print(
    f"  yours:   Q(above goal, down) = {Q_fixed[ABOVE_GOAL, DOWN]:8.4f}   max Q(start) = {Q_fixed[START].max():8.4f}"
)
print(f"  truth:   Q*(above goal, down) = {-1.0:8.4f}   V*(start) = {V_STAR_START:8.4f}")
show_path(greedy_path(Q_suspect)[0], "suspect's greedy path")
fig, ax = plt.subplots()
ax.plot(xs, v_suspect, label="q_learning_suspect")
ax.plot(xs, v_fixed, label="your q_learning")
ax.axhline(V_STAR_START, linestyle="--", color="gray", label="V*(start)")
ax.set(xlabel="episodes", ylabel="max Q(start, a)", title=f"Value of the start from Q0 = {Q0}")
ax.legend()
plt.show()

# %% [markdown] role="ex6-explain"
# Before your fix, the suspect's final transition settled at −1 + γ · 10 = 8.5 and its start
# at about −4.6, both above the truth, and its greedy path was still optimal: a bonus that
# shrinks with distance cannot reorder the routes. Why did the same bug change nothing when
# Q started at 0? Which fix did Sutton and Barto's pseudocode use instead, and when does it
# fail? (terminal mask)
#
# **Diagnostics for this lab.** For a learner, plot max Q(start) over episodes against a known
# V*(start) and print Q of the final transition against its reward: a value curve that
# flattens at the wrong level, or a final transition worth more than its reward, means the
# target is wrong, not the exploration. Run every such test from an optimistic start as well
# as from zero; the terminal bug is invisible from zero. For a predictor, plot the estimate
# along the route against the exact V and read the error's shape: a constant offset points to
# the discount or the terminal rule, an offset that grows with distance from the goal points
# to a bootstrap that forgets its start too slowly (small α, few episodes), and a ragged
# curve with a correct mean points to variance, which more episodes or more seeds reduce.
# A greedy path that loops means Q is noisy where the agent rarely goes; it is not a bug.

# %% role="ex6-chk"
lab.check(6, m03.check_q_learning_suspect, q_learning_suspect)

# %%
# The room code: one line for the debrief's pooled table (seat = SEED).
print(
    f"ROOM CODE | seat {SEED} | mc_rms {mc_rms:.3f} | td_rms {td_rms:.3f} | "
    f"sarsa_online {mean_sarsa:.1f} | q_online {mean_q:.1f} | "
    f"sarsa_moves {len(path_sarsa) - 1 if reached else 'loop'} | q_moves {len(path_q) - 1 if reached_q else 'loop'}"
)

# %% [markdown] role="stretch"
# **The n-step dial on your seed.** Build n-step TD prediction on your `n_step_target`: for
# each episode and each t, move V[s_t] a fraction α toward G_t^(n), computed with the values
# as they stand (the briefing's Eq. for n-step TD; Sutton and Barto Eq. 7.2 update the value
# n steps late within an episode, which this offline version ignores). Then the provided
# sweep runs n in {1, 2, 4, 8, 16} and several α on 5 seeds and plots the RMS error on the
# route: the curves of the briefing's figure, on your seed's episodes.


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
stretch_seeds = [SEED * 100 + 50 + k for k in range(5)]
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
