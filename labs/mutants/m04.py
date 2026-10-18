"""Plausible wrong versions of Module 4's functions (first draft by the academic director;
the technical expert completes it).

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires
every exact checkpoint of its exercise that calls the function to reject it; stochastic
checkpoints are skipped there, because scripts/threshold_protocol.py has already required
every mutant of a stochastic checkpoint's functions to fail its threshold at every seed,
with a stated gap. Each mutant is a mistake a participant could plausibly make; `why` says
which. `id="random"` is the never-learns version the protocol needs.

Only checkpoint 3.3 (Thompson sampling's regret) is stochastic. It calls `thompson_step`
and `run_bandit`, so the protocol would score every `run_bandit` mutant against it as well:
they are all `exact_only=True`, and checkpoint 1.1 (exact, on fixed reward stacks with a
policy that checks what it is shown) rejects them. `thompson_step`'s "posterior-mean"
(Russo et al.'s greedy BernGreedy) is `exact_only=True` too: the design scan found its regret
overlapping the solution's at every budget tried, because it does well on the seeds where it
does not lock in; checkpoint 3.2 (exact choice frequencies on fixed posteriors) rejects it.

Not listed: a `verdict` that uses the IQM's bootstrap interval with 2 to 4 seeds too. With so
few seeds a resample made of copies of the extreme value has probability of at least 2.5%, so
the percentile interval reaches the [min, max] range and the verdict is the same on every input.
"""

import numpy as np

from prl.lab import mutant


# ------------------------------------------------------------------ 1: run_bandit
@mutant(
    ex=1,
    replaces="run_bandit",
    id="reward-by-round",
    why="reads row t of the stack instead of the top of the pulled arm's own column",
    exact_only=True,
)
def run_bandit(policy, stack, r_mean, seed):
    rng = np.random.default_rng(seed)
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a = policy(counts, sums, t, rng)
        reward = stack[t, a]
        counts[a] += 1
        sums[a] += reward
        arms[t] = a
    gaps = np.max(r_mean) - np.asarray(r_mean)
    return arms, np.cumsum(gaps[arms])


@mutant(
    ex=1,
    replaces="run_bandit",
    id="count-before-reward",
    why="increments the count before reading the stack, so the first pull takes row 1",
    exact_only=True,
)
def run_bandit(policy, stack, r_mean, seed):  # noqa: F811
    rng = np.random.default_rng(seed)
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a = policy(counts, sums, t, rng)
        counts[a] += 1
        reward = stack[min(counts[a], T - 1), a]
        sums[a] += reward
        arms[t] = a
    gaps = np.max(r_mean) - np.asarray(r_mean)
    return arms, np.cumsum(gaps[arms])


@mutant(
    ex=1,
    replaces="run_bandit",
    id="regret-from-rewards",
    why="sums r* minus the reward received (random regret), not the gap of the arm pulled",
    exact_only=True,
)
def run_bandit(policy, stack, r_mean, seed):  # noqa: F811
    rng = np.random.default_rng(seed)
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=int)
    rewards = np.zeros(T)
    for t in range(T):
        a = policy(counts, sums, t, rng)
        reward = stack[counts[a], a]
        counts[a] += 1
        sums[a] += reward
        arms[t] = a
        rewards[t] = reward
    return arms, np.cumsum(np.max(r_mean) - rewards)


@mutant(
    ex=1,
    replaces="run_bandit",
    id="rng-every-round",
    why="makes a new default_rng(seed) in every round, so the policy's coin flips repeat",
    exact_only=True,
)
def run_bandit(policy, stack, r_mean, seed):  # noqa: F811
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a = policy(counts, sums, t, np.random.default_rng(seed))
        reward = stack[counts[a], a]
        counts[a] += 1
        sums[a] += reward
        arms[t] = a
    gaps = np.max(r_mean) - np.asarray(r_mean)
    return arms, np.cumsum(gaps[arms])


@mutant(
    ex=1,
    replaces="run_bandit",
    id="round-from-one",
    why="passes t + 1 to the policy, so UCB1's log and every round-based rule are off by one",
    exact_only=True,
)
def run_bandit(policy, stack, r_mean, seed):  # noqa: F811
    rng = np.random.default_rng(seed)
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a = policy(counts, sums, t + 1, rng)
        reward = stack[counts[a], a]
        counts[a] += 1
        sums[a] += reward
        arms[t] = a
    gaps = np.max(r_mean) - np.asarray(r_mean)
    return arms, np.cumsum(gaps[arms])


@mutant(
    ex=1,
    replaces="run_bandit",
    id="per-round-regret",
    why="returns the gap of each round, gaps[arms], and forgets the running sum",
    exact_only=True,
)
def run_bandit(policy, stack, r_mean, seed):  # noqa: F811
    rng = np.random.default_rng(seed)
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a = policy(counts, sums, t, rng)
        reward = stack[counts[a], a]
        counts[a] += 1
        sums[a] += reward
        arms[t] = a
    gaps = np.max(r_mean) - np.asarray(r_mean)
    return arms, gaps[arms]


# ------------------------------------------------------------------ 1: eps_greedy
@mutant(
    ex=1,
    replaces="eps_greedy",
    id="explores-with-1-minus-eps",
    why="explores when u < 1 - epsilon: the comparison the wrong way round",
)
def eps_greedy(counts, sums, t, rng, epsilon=0.1):
    if np.any(counts == 0):
        return int(np.argmin(counts))
    if rng.random() < 1 - epsilon:
        return int(rng.integers(len(counts)))
    return argmax_tie(sums / counts)  # noqa: F821


@mutant(
    ex=1,
    replaces="eps_greedy",
    id="no-first-pulls",
    why="skips 'each arm once' and guards the division instead, so greedy starts on arm 0",
)
def eps_greedy(counts, sums, t, rng, epsilon=0.1):  # noqa: F811
    if rng.random() < epsilon:
        return int(rng.integers(len(counts)))
    return argmax_tie(sums / np.maximum(counts, 1))  # noqa: F821


@mutant(
    ex=1,
    replaces="eps_greedy",
    id="greedy-on-sums",
    why="exploits the arm with the largest total reward, not the largest mean",
)
def eps_greedy(counts, sums, t, rng, epsilon=0.1):  # noqa: F811
    if np.any(counts == 0):
        return int(np.argmin(counts))
    if rng.random() < epsilon:
        return int(rng.integers(len(counts)))
    return argmax_tie(sums)  # noqa: F821


@mutant(
    ex=1,
    replaces="eps_greedy",
    id="explores-other-arms",
    why="explores only among the arms that are not greedy (the greedy arm's share falls)",
)
def eps_greedy(counts, sums, t, rng, epsilon=0.1):  # noqa: F811
    if np.any(counts == 0):
        return int(np.argmin(counts))
    best = argmax_tie(sums / counts)  # noqa: F821
    if rng.random() < epsilon:
        others = [a for a in range(len(counts)) if a != best]
        return int(others[rng.integers(len(others))])
    return best


@mutant(
    ex=1,
    replaces="eps_greedy",
    id="explore-first",
    why="flips the coin before checking for unpulled arms (the three cases in the wrong order)",
)
def eps_greedy(counts, sums, t, rng, epsilon=0.1):  # noqa: F811
    if rng.random() < epsilon:
        return int(rng.integers(len(counts)))
    if np.any(counts == 0):
        return int(np.argmin(counts))
    return argmax_tie(sums / counts)  # noqa: F821


# ------------------------------------------------------------------ 2: ucb1_index
@mutant(ex=2, replaces="ucb1_index", id="log10", why="takes log base 10 instead of ln")
def ucb1_index(counts, sums, t):
    index = np.full(len(counts), np.inf)
    seen = counts > 0
    if seen.any():
        index[seen] = sums[seen] / counts[seen] + np.sqrt(2 * np.log10(t) / counts[seen])
    return index


@mutant(ex=2, replaces="ucb1_index", id="no-sqrt", why="forgets the square root of the bonus")
def ucb1_index(counts, sums, t):  # noqa: F811
    index = np.full(len(counts), np.inf)
    seen = counts > 0
    if seen.any():
        index[seen] = sums[seen] / counts[seen] + 2 * np.log(t) / counts[seen]
    return index


@mutant(ex=2, replaces="ucb1_index", id="log-t-plus-one", why="uses ln(t + 1), one round ahead")
def ucb1_index(counts, sums, t):  # noqa: F811
    index = np.full(len(counts), np.inf)
    seen = counts > 0
    if seen.any():
        index[seen] = sums[seen] / counts[seen] + np.sqrt(2 * np.log(t + 1) / counts[seen])
    return index


@mutant(
    ex=2,
    replaces="ucb1_index",
    id="unpulled-zero",
    why="gives an arm never pulled the index 0 instead of infinity",
)
def ucb1_index(counts, sums, t):  # noqa: F811
    index = np.zeros(len(counts))
    seen = counts > 0
    if seen.any():
        index[seen] = sums[seen] / counts[seen] + np.sqrt(2 * np.log(t) / counts[seen])
    return index


@mutant(ex=2, replaces="ucb1_index", id="sums-not-means", why="adds the bonus to the total reward")
def ucb1_index(counts, sums, t):  # noqa: F811
    index = np.full(len(counts), np.inf)
    seen = counts > 0
    if seen.any():
        index[seen] = sums[seen] + np.sqrt(2 * np.log(t) / counts[seen])
    return index


# ------------------------------------------------------------------ 3: beta_posterior
@mutant(ex=3, replaces="beta_posterior", id="swapped", why="returns (1 + F, 1 + S)")
def beta_posterior(counts, sums):
    clicks = np.asarray(sums, dtype=float)
    return 1 + (counts - clicks), 1 + clicks


@mutant(ex=3, replaces="beta_posterior", id="no-prior", why="forgets the prior's 1 + on both")
def beta_posterior(counts, sums):  # noqa: F811
    clicks = np.asarray(sums, dtype=float)
    return clicks, counts - clicks


@mutant(
    ex=3,
    replaces="beta_posterior",
    id="failures-are-counts",
    why="uses the pull count as the number of failures (forgets to subtract the clicks)",
)
def beta_posterior(counts, sums):  # noqa: F811
    clicks = np.asarray(sums, dtype=float)
    return 1 + clicks, 1 + np.asarray(counts, dtype=float)


# ------------------------------------------------------------------ 3: thompson_step
@mutant(
    ex=3,
    replaces="thompson_step",
    id="posterior-mean",
    why="plays the largest posterior mean, not a draw (greedy; Russo et al.'s BernGreedy)",
    exact_only=True,
)
def thompson_step(counts, sums, t, rng):
    shape_s, shape_f = beta_posterior(counts, sums)  # noqa: F821
    return argmax_tie(shape_s / (shape_s + shape_f))  # noqa: F821


@mutant(
    ex=3,
    replaces="thompson_step",
    id="sample-reward",
    why="draws a 0/1 outcome per arm instead of a rate (Russo et al. 2018, p. 14)",
)
def thompson_step(counts, sums, t, rng):  # noqa: F811
    shape_s, shape_f = beta_posterior(counts, sums)  # noqa: F821
    outcomes = rng.random(len(counts)) < shape_s / (shape_s + shape_f)
    return argmax_tie(outcomes)  # noqa: F821


@mutant(
    ex=3,
    replaces="thompson_step",
    id="swapped-draw",
    why="draws rng.beta(1 + F, 1 + S): the posterior of the miss rate, so it hunts the worst arm",
)
def thompson_step(counts, sums, t, rng):  # noqa: F811
    shape_s, shape_f = beta_posterior(counts, sums)  # noqa: F821
    return argmax_tie(rng.beta(shape_f, shape_s))  # noqa: F821


@mutant(ex=3, replaces="thompson_step", id="random", why="pulls a uniformly random arm")
def thompson_step(counts, sums, t, rng):  # noqa: F811
    return int(rng.integers(len(counts)))


# ------------------------------------------------------------------ 4: verdict
@mutant(ex=4, replaces="verdict", id="higher-is-better", why="calls the higher regret the winner")
def verdict(regret_a, regret_b):
    xa, xb = np.asarray(regret_a, dtype=float), np.asarray(regret_b, dtype=float)
    n = min(xa.size, xb.size)
    if n == 1:
        return "single seed"
    if n >= 5:
        sa, sb = (iqm(xa), *bootstrap_ci(xa, iqm)), (iqm(xb), *bootstrap_ci(xb, iqm))  # noqa: F821
    else:
        sa, sb = (xa.mean(), xa.min(), xa.max()), (xb.mean(), xb.min(), xb.max())
    if sa[1] <= sb[2] and sb[1] <= sa[2]:
        return "tie"
    return "A" if sa[0] > sb[0] else "B"


@mutant(ex=4, replaces="verdict", id="no-ties", why="never calls a tie: the lower center wins")
def verdict(regret_a, regret_b):  # noqa: F811
    xa, xb = np.asarray(regret_a, dtype=float), np.asarray(regret_b, dtype=float)
    if min(xa.size, xb.size) == 1:
        return "single seed"
    ca = iqm(xa) if min(xa.size, xb.size) >= 5 else xa.mean()  # noqa: F821
    cb = iqm(xb) if min(xa.size, xb.size) >= 5 else xb.mean()  # noqa: F821
    return "A" if ca < cb else "B"


@mutant(
    ex=4,
    replaces="verdict",
    id="ranges-always",
    why="compares [min, max] ranges at every seed count, never the IQM's bootstrap interval",
)
def verdict(regret_a, regret_b):  # noqa: F811
    xa, xb = np.asarray(regret_a, dtype=float), np.asarray(regret_b, dtype=float)
    if min(xa.size, xb.size) == 1:
        return "single seed"
    sa, sb = (xa.mean(), xa.min(), xa.max()), (xb.mean(), xb.min(), xb.max())
    if sa[1] <= sb[2] and sb[1] <= sa[2]:
        return "tie"
    return "A" if sa[0] < sb[0] else "B"


@mutant(
    ex=4,
    replaces="verdict",
    id="mean-interval",
    why="bootstraps the mean, not the IQM (an outlier seed widens or shifts the interval)",
)
def verdict(regret_a, regret_b):  # noqa: F811
    xa, xb = np.asarray(regret_a, dtype=float), np.asarray(regret_b, dtype=float)
    n = min(xa.size, xb.size)
    if n == 1:
        return "single seed"
    if n >= 5:
        sa, sb = (xa.mean(), *bootstrap_ci(xa)), (xb.mean(), *bootstrap_ci(xb))  # noqa: F821
    else:
        sa, sb = (xa.mean(), xa.min(), xa.max()), (xb.mean(), xb.min(), xb.max())
    if sa[1] <= sb[2] and sb[1] <= sa[2]:
        return "tie"
    return "A" if sa[0] < sb[0] else "B"


# ------------------------------------------------------------------ 5: scale_directions
@mutant(ex=5, replaces="scale_directions", id="ucb1-down", why="expects a smaller bonus to help")
def scale_directions():
    return {"eps_greedy": "same", "ucb1": "down", "ucb1_rescaled": "same"}


@mutant(
    ex=5,
    replaces="scale_directions",
    id="eps-up",
    why="expects every method's regret to grow with the rewards, even in the original units",
)
def scale_directions():  # noqa: F811
    return {"eps_greedy": "up", "ucb1": "up", "ucb1_rescaled": "up"}


@mutant(
    ex=5,
    replaces="scale_directions",
    id="ucb1-same",
    why="believes UCB1 is free of parameters, so nothing changes",
)
def scale_directions():  # noqa: F811
    return {"eps_greedy": "same", "ucb1": "same", "ucb1_rescaled": "same"}


# ------------------------------------------------------------------ 6: UCB1Suspect
@mutant(
    ex=6,
    replaces="UCB1Suspect",
    id="counts-every-arm",
    why="increments every arm's count in each update, so the bonuses never differ",
)
class UCB1Suspect:
    def __init__(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def act(self, t):
        if t < len(self.counts):
            return t
        n = np.maximum(self.counts, 1)
        return argmax_tie(self.sums / n + np.sqrt(2 * np.log(t) / n))  # noqa: F821

    def update(self, arm, reward):
        self.counts += 1
        self.sums[arm] += reward


@mutant(
    ex=6,
    replaces="UCB1Suspect",
    id="counted-twice",
    why="adds the count in act as well as in update, so every bonus is too small by sqrt(2)",
)
class UCB1Suspect:  # noqa: F811
    def __init__(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def act(self, t):
        if t < len(self.counts):
            self.counts[t] += 1
            return t
        n = np.maximum(self.counts, 1)
        a = argmax_tie(self.sums / n + np.sqrt(2 * np.log(t) / n))  # noqa: F821
        self.counts[a] += 1
        return a

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sums[arm] += reward


# ------------------------------------------------------------------ stretch: gaussian_posterior
@mutant(
    ex="stretch",
    replaces="gaussian_posterior",
    id="variance-not-sd",
    why="returns the posterior variance where the standard deviation belongs",
)
def gaussian_posterior(counts, sums, noise_sd, prior_sd):
    precision = 1 / prior_sd**2 + counts / noise_sd**2
    return (sums / noise_sd**2) / precision, 1 / precision


@mutant(
    ex="stretch",
    replaces="gaussian_posterior",
    id="sample-mean",
    why="uses the sample mean as the posterior mean, ignoring the prior's pull toward 0",
)
def gaussian_posterior(counts, sums, noise_sd, prior_sd):  # noqa: F811
    precision = 1 / prior_sd**2 + counts / noise_sd**2
    return sums / np.maximum(counts, 1), np.sqrt(1 / precision)
