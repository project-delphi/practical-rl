"""Plausible wrong versions of Module 3's functions.

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires
every exact checkpoint that calls the function to reject it; stochastic checkpoints are
skipped there, because scripts/threshold_protocol.py has already required every mutant of
a stochastic checkpoint's functions to fail its threshold at every seed, with a stated
gap. Each mutant is a mistake a participant could plausibly make; `why` says which.
`id="random"` is the never-learns version the protocol needs.

`exact_only=True` marks wrong versions that a budgeted RMS cannot separate from the
solution but that the exact checkpoints reject by construction (OPEN_QUESTIONS C14):
every-visit MC (checkpoint 1.1 uses episodes with repeated states); MC that keeps only the
latest return (its error does not shrink with the budget and overlaps the solution's at
QUICK; checkpoint 1.1 averages over several episodes); TD(0) without the
discount, treating a truncation as terminal, or bootstrapping through the terminal
(checkpoint 2.1 uses v0 != 0 and a truncated episode). The protocol leaves them out.

Not listed: a Q-learning fix that masks on `truncated` too. No cliff episode is ever
truncated at this budget, so no checkpoint can tell it from the solution.
"""

import numpy as np

from prl.lab import mutant


@mutant(ex=1, replaces="mc_prediction", id="no-discount", why="sums rewards without the discount")
def mc_prediction(episodes, gamma, n_states):
    V, N = np.zeros(n_states), np.zeros(n_states)
    for ep in episodes:
        G, returns = 0.0, [0.0] * len(ep.rewards)
        for t in reversed(range(len(ep.rewards))):
            G = ep.rewards[t] + G
            returns[t] = G
        seen = set()
        for t, s in enumerate(ep.states[:-1]):
            if s in seen:
                continue
            seen.add(s)
            N[s] += 1
            V[s] += (returns[t] - V[s]) / N[s]
    return V


@mutant(
    ex=1,
    replaces="mc_prediction",
    id="last-return",
    why="overwrites V with the latest return",
    exact_only=True,
)
def mc_prediction(episodes, gamma, n_states):  # noqa: F811
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
            V[s] = returns[t]
    return V


@mutant(
    ex=1,
    replaces="mc_prediction",
    id="every-visit",
    why="averages the return after every visit, not only the first",
    exact_only=True,
)
def mc_prediction(episodes, gamma, n_states):  # noqa: F811
    V, N = np.zeros(n_states), np.zeros(n_states)
    for ep in episodes:
        G, returns = 0.0, [0.0] * len(ep.rewards)
        for t in reversed(range(len(ep.rewards))):
            G = ep.rewards[t] + gamma * G
            returns[t] = G
        for t, s in enumerate(ep.states[:-1]):
            N[s] += 1
            V[s] += (returns[t] - V[s]) / N[s]
    return V


@mutant(ex=1, replaces="mc_prediction", id="random", why="never looks at the episodes")
def mc_prediction(episodes, gamma, n_states):  # noqa: F811
    return np.random.default_rng(0).normal(size=n_states)


@mutant(ex=2, replaces="td0_prediction", id="no-bootstrap", why="the target is the reward alone")
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):
    V = np.full(n_states, float(v0))
    for ep in episodes:
        for s, r in zip(ep.states[:-1], ep.rewards, strict=True):
            V[s] += alpha * (r - V[s])
    return V


@mutant(ex=2, replaces="td0_prediction", id="updates-next-state", why="moves V[s2] instead of V[s]")
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):  # noqa: F811
    V = np.full(n_states, float(v0))
    for ep in episodes:
        last = len(ep.rewards) - 1
        for t, (s, r, s2) in enumerate(zip(ep.states[:-1], ep.rewards, ep.states[1:], strict=True)):
            terminal = ep.terminated and t == last
            target = r + gamma * (0.0 if terminal else V[s2])
            V[s2] += alpha * (target - V[s])
    return V


@mutant(
    ex=2,
    replaces="td0_prediction",
    id="no-discount",
    why="bootstraps from V[s2] without gamma",
    exact_only=True,
)
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):  # noqa: F811
    V = np.full(n_states, float(v0))
    for ep in episodes:
        last = len(ep.rewards) - 1
        for t, (s, r, s2) in enumerate(zip(ep.states[:-1], ep.rewards, ep.states[1:], strict=True)):
            terminal = ep.terminated and t == last
            target = r + (0.0 if terminal else V[s2])
            V[s] += alpha * (target - V[s])
    return V


@mutant(
    ex=2,
    replaces="td0_prediction",
    id="truncation-as-terminal",
    why="drops the bootstrap at the end of every episode, truncated or not",
    exact_only=True,
)
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):  # noqa: F811
    V = np.full(n_states, float(v0))
    for ep in episodes:
        last = len(ep.rewards) - 1
        for t, (s, r, s2) in enumerate(zip(ep.states[:-1], ep.rewards, ep.states[1:], strict=True)):
            terminal = (ep.terminated or ep.truncated) and t == last
            target = r + gamma * (0.0 if terminal else V[s2])
            V[s] += alpha * (target - V[s])
    return V


@mutant(
    ex=2,
    replaces="td0_prediction",
    id="bootstraps-through-terminal",
    why="always bootstraps from V[s2], even when s2 is terminal",
    exact_only=True,
)
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):  # noqa: F811
    V = np.full(n_states, float(v0))
    for ep in episodes:
        for s, r, s2 in zip(ep.states[:-1], ep.rewards, ep.states[1:], strict=True):
            V[s] += alpha * (r + gamma * V[s2] - V[s])
    return V


@mutant(ex=2, replaces="td0_prediction", id="random", why="never looks at the episodes")
def td0_prediction(episodes, alpha, gamma, n_states, v0=0.0):  # noqa: F811
    return np.random.default_rng(0).normal(size=n_states)


@mutant(
    ex=3,
    replaces="n_step_target",
    id="always-bootstraps",
    why="bootstraps from s_T even when the episode terminated",
)
def n_step_target(rewards, states, t, n, V, gamma, terminated):
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
    return G + gamma ** (end - t) * V[states[end]]


@mutant(
    ex=3,
    replaces="n_step_target",
    id="truncation-as-terminal",
    why="drops the bootstrap at a truncation",
)
def n_step_target(rewards, states, t, n, V, gamma, terminated):  # noqa: F811
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
    if end < T:
        G += gamma ** (end - t) * V[states[end]]
    return G


@mutant(
    ex=3, replaces="n_step_target", id="discount-off-by-one", why="discounts the first reward too"
)
def n_step_target(rewards, states, t, n, V, gamma, terminated):  # noqa: F811
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t + 1) * rewards[k] for k in range(t, end))
    if end < T or not terminated:
        G += gamma ** (end - t) * V[states[end]]
    return G


@mutant(ex=4, replaces="sarsa", id="max-target", why="bootstraps from max Q: it is Q-learning")
def sarsa(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)  # noqa: F821
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)  # noqa: F821
        total, done = 0.0, False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = epsilon_greedy(Q, s2, epsilon, rng)  # noqa: F821
            target = r + gamma * (0.0 if terminated else Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, a, done = s2, a2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


@mutant(ex=4, replaces="sarsa", id="random", why="acts at random and never updates Q")
def sarsa(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):  # noqa: F811
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)  # noqa: F821
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, r, terminated, truncated, _ = env.step(int(rng.integers(A)))  # noqa: F821
            total += r
            done = terminated or truncated
        online.append(total)
    return Q, np.array(online)


@mutant(
    ex=5,
    replaces="q_learning",
    id="taken-action-target",
    why="bootstraps from the action taken next: it is SARSA",
)
def q_learning(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)  # noqa: F821
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)  # noqa: F821
        total, done = 0.0, False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = epsilon_greedy(Q, s2, epsilon, rng)  # noqa: F821
            target = r + gamma * (0.0 if terminated else Q[s2, a2])
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, a, done = s2, a2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


@mutant(ex=5, replaces="q_learning", id="random", why="acts at random and never updates Q")
def q_learning(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):  # noqa: F811
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)  # noqa: F821
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, r, terminated, truncated, _ = env.step(int(rng.integers(A)))  # noqa: F821
            total += r
            done = terminated or truncated
        online.append(total)
    return Q, np.array(online)


@mutant(
    ex=6,
    replaces="q_learning_suspect",
    id="still-bootstraps",
    why="leaves the colleague's bug in place",
)
def q_learning_suspect(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)  # noqa: F821
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = epsilon_greedy(Q, s, epsilon, rng)  # noqa: F821
            s2, r, terminated, truncated, _ = env.step(a)
            target = r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, done = s2, terminated or truncated
        online.append(total)
    return Q, np.array(online)


@mutant(ex=6, replaces="q_learning_suspect", id="random", why="acts at random and never updates Q")
def q_learning_suspect(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):  # noqa: F811
    rng = np.random.default_rng(seed)
    Q = np.full((S, A), float(q0)) if np.isscalar(q0) else np.array(q0, dtype=float)  # noqa: F821
    online = []
    env.reset(seed=seed)
    for _ in range(episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, r, terminated, truncated, _ = env.step(int(rng.integers(A)))  # noqa: F821
            total += r
            done = terminated or truncated
        online.append(total)
    return Q, np.array(online)
