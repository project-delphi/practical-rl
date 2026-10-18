# ---
# jupyter:
#   prl:
#     module: m04
# ---

# %% [markdown]
# In this lab you choose which of three headlines to show, one visitor at a time; a click pays
# 1 and no click pays 0. You write the bandit loop and measure pseudo-regret against the known
# click rates. Then you write three ways to choose: ε-greedy, UCB1 and Thompson sampling. You
# compare them over 20 seeds with the workshop's tie rule, multiply every reward by 10 to see
# which method notices, and find the bug in a colleague's UCB1.
#
# The briefing's equations are named in brackets, for example (pseudo-regret) or (UCB1). Its
# table "Equations and where you implement them" gives each name's equation number.

# %%
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from prl.checks import m04  # the checkpoints of this lab
from prl.evaluate import Budget, bootstrap_ci, iqm

R_MEAN = np.array([0.4, 0.3, 0.5])  # click rates r(a) of headlines 0, 1, 2; 2 is the best
GAPS = R_MEAN.max() - R_MEAN  # Delta_a = r* - r(a): 0.1, 0.2 and 0
N_ARMS = len(R_MEAN)

# Budgets. Lab code reads them from here and never branches on QUICK. `early` is the second
# horizon at which Exercise 4 reads the same runs.
budget = lab.use_settings(
    prl.runtime.settings(
        dict(T=5000, seeds=20, early=1000),  # live
        dict(T=1000, seeds=10, early=200),  # QUICK
        designed="colab-cpu",
    )
)
T = budget.T
# Your seat's seeds start here. Seat 0 does not reuse the seeds of the briefing's figures,
# the recorded experiments or the checkpoints (0-99), so every seat adds new evidence.
RUN_SEED = (SEED + 1) * 1000
SEEDS = [RUN_SEED + i for i in range(budget.seeds)]
_rt = prl.runtime.detect()
HARDWARE = f"{_rt.platform}, {_rt.cpu}, {_rt.n_cpu} CPU"
print(f"Click rates {R_MEAN.tolist()}; gaps {GAPS.tolist()}")
print(f"Budget: {T} rounds per run, {budget.seeds} seeds per method: {SEEDS[0]}-{SEEDS[-1]}")


# %% [markdown]
# ### Provided: the tie rule for arms, the reward stack and a plot
#
# **Rewards are drawn up front.** For each seed, `reward_stack` draws a column of rewards for
# every arm: `stack[n, a]` is what the (n + 1)-th pull of arm `a` pays. A method that pulls arm
# `a` takes the reward on top of its column. Two methods run on the same seed therefore see the
# same rewards whenever they pull the same arm, which makes their comparison fairer, and it
# changes nothing about what each method sees on its own (Lattimore and Szepesvári call this
# the reward-stack model). The policy's own coin flips come from a separate generator.
#
# **Ties go to the lowest arm.** `argmax_tie` returns the first arm within rounding of the
# largest value, so a tie never depends on how the last bit of a float was rounded.


# %%
def argmax_tie(x):
    """Provided: the first index within rounding of the max. An infinite entry wins at once."""
    x = np.asarray(x, dtype=float)
    top = x.max()
    if np.isinf(top):
        return int(np.flatnonzero(x == top)[0])
    tol = 1e-9 * (1 + np.abs(x).max())
    return int(np.flatnonzero(x >= top - tol)[0])


def reward_stack(r_mean, T, seed, scale=1.0):
    """Provided: Bernoulli rewards drawn up front, 0 or `scale`. stack[n, a] is the reward of
    the (n + 1)-th pull of arm a. Its stream is seeded by [seed, 1], apart from the policy's."""
    rng = np.random.default_rng([seed, 1])
    r_mean = np.asarray(r_mean, dtype=float)
    return scale * (rng.random((T, len(r_mean))) < r_mean)


def plot_regret(ax, curves, label, style="-"):
    """Provided: every seed as a thin line and the IQM over seeds as a thick one."""
    curves = np.asarray(curves, dtype=float)
    rounds = np.arange(1, curves.shape[1] + 1)
    center = np.array([iqm(curves[:, j]) for j in range(curves.shape[1])])
    line = ax.plot(rounds, center, ls=style, lw=2.5, label=f"{label} (IQM)")[0]
    for c in curves:
        ax.plot(rounds, c, ls=style, lw=0.6, alpha=0.35, color=line.get_color())
    ax.set(xlabel="round t", ylabel="pseudo-regret")


# %% [markdown] role="ex1-head"
# **The bandit loop and ε-greedy.** In round t a policy sees what has happened so far, the
# pull counts N_t(a) and the reward sums of every arm, and picks an arm; that arm's next reward
# comes off its stack. The pseudo-regret adds the gap of every arm pulled,
# Reg_T = Σ_t (r* − r(a_t)) (pseudo-regret): it uses the known click rates, never the rewards.
# Write the loop, `run_bandit`, then ε-greedy, `eps_greedy`: each arm once, then with
# probability ε a uniformly random arm and otherwise the arm with the best mean so far.
# With ε = 0 it is greedy.

# %% [markdown] role="ex1-predict"
# Greedy (ε = 0) shows each headline once, in rounds 0, 1 and 2, then always the one with the
# best click rate so far. Suppose that on some seed it shows headline 0 in every round from
# round 3 on. What is its pseudo-regret after T = 5000 rounds? And if it shows headline 1 from
# round 3 on? And if it shows headline 2, the best, from round 3 on?


# %% role="ex1-stub"
def run_bandit(policy, stack, r_mean, seed):
    """Play len(stack) rounds of a bandit. Returns (arms, regret).

    Make rng = np.random.default_rng(seed) once. In round t = 0, 1, ...: a = policy(counts, sums,
    t, rng); the reward is the top of arm a's stack, stack[counts[a], a]; then counts[a] += 1 and
    sums[a] += reward. counts (ints) and sums (floats) have one entry per arm and start at 0.
    arms[t] is the arm pulled in round t. regret[t] is the pseudo-regret after round t: the sum of
    the gaps max(r_mean) - r_mean[a] of the arms pulled in rounds 0..t (pseudo-regret).
    """
    # TODO 1 (part 1): the loop above, then the running sum of the gaps of the arms you pulled.
    raise NotImplementedError(
        "TODO 1: write run_bandit, or run lab.use_reference(1) to continue with the solution"
    )


def eps_greedy(counts, sums, t, rng, epsilon=0.1):
    """epsilon-greedy as a policy. If an arm has never been pulled, return the first such arm.
    Otherwise draw u = rng.random(): if u < epsilon return a uniform arm, rng.integers(len(counts));
    else the arm with the best mean so far, argmax_tie(sums / counts)."""
    # TODO 1 (part 2): three cases, in that order.
    raise NotImplementedError(
        "TODO 1: write eps_greedy, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# `T, n_arms = stack.shape`. Inside the loop, read the reward *before* you increment
# `counts[a]`: the first pull of arm `a` takes row 0. After the loop, `gaps[arms]` is the gap of
# every round in order, and `np.cumsum` turns it into the running total. In `eps_greedy`,
# `np.argmin(counts)` is the first arm with the smallest count, which is a never-pulled arm if
# there is one.


# %% role="ex1-sol"
@lab.solution(1)
def run_bandit(policy, stack, r_mean, seed):
    """The bandit loop: the arm of every round and the pseudo-regret."""
    rng = np.random.default_rng(seed)
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a = policy(counts, sums, t, rng)
        reward = stack[counts[a], a]  # the top of arm a's stack
        counts[a] += 1
        sums[a] += reward
        arms[t] = a
    gaps = np.max(r_mean) - np.asarray(r_mean)  # Delta_a
    return arms, np.cumsum(gaps[arms])  # (pseudo-regret)


@lab.solution(1)
def eps_greedy(counts, sums, t, rng, epsilon=0.1):
    """Each arm once; then explore with probability epsilon."""
    if np.any(counts == 0):
        return int(np.argmin(counts))  # an arm never pulled
    if rng.random() < epsilon:
        return int(rng.integers(len(counts)))  # any arm
    return argmax_tie(sums / counts)  # the best mean so far


# %% role="ex1-run"
greedy = partial(eps_greedy, epsilon=0.0)
eps_01 = partial(eps_greedy, epsilon=0.1)
lab.seeds(SEEDS)
final = {}  # final pseudo-regret per method, one entry per seed; Exercise 4 compares them
curves = {}  # regret after every round, for the plots and for Exercise 4's early horizon
best_share = []  # greedy's share of headline 2 in the second half of each run
for name, policy in [("greedy", greedy), ("eps-greedy", eps_01)]:
    curves[name] = []
    for k in SEEDS:
        arms, regret = run_bandit(policy, reward_stack(R_MEAN, T, k), R_MEAN, k)
        curves[name].append(regret)
        if name == "greedy":
            best_share.append(np.mean(arms[T // 2 :] == 2))
    final[name] = np.array([c[-1] for c in curves[name]])
locked = int(np.sum(np.array(best_share) < 0.5))
lab.metric("greedy_locked", locked)  # room code
for name in ("greedy", "eps-greedy"):
    print(f"{name}, pseudo-regret after {T} rounds, seed by seed:")
    for row in range(0, len(SEEDS), 10):  # ten seeds per line
        print("  " + " ".join(f"{x:7.1f}" for x in final[name][row : row + 10]))
print(f"Greedy left headline 2 behind (under half of the last {T // 2} pulls) on {locked} seeds.")
fig, ax = plt.subplots()
plot_regret(ax, curves["greedy"], "greedy")
plot_regret(ax, curves["eps-greedy"], "ε-greedy, ε = 0.1", style="--")
ax.set_title(f"{len(SEEDS)} seeds, {T} rounds")
ax.legend()
plt.show()

# %% [markdown] role="ex1-explain"
# On the seeds where greedy left headline 2 behind, its curve is a straight line: every round
# adds the same gap (pseudo-regret). Why does greedy never try headline 2 again once its first
# showing got no click? ε-greedy never stays stuck for good: what does it pay in every round
# instead (exploration floor), and does its curve flatten? (Gotcha: "ε = 0.1 is a safe default.")

# %% role="ex1-chk1"
lab.check(1, m04.check_run_bandit, run_bandit, label="1.1")

# %% role="ex1-chk2"
lab.check(1, m04.check_eps_greedy, eps_greedy, label="1.2")

# %% [markdown] role="ex2-head"
# **UCB1.** Pull each arm once; then pull the arm with the largest index
# U_t(a) = r̂_t(a) + √(2 ln t / N_t(a)) (UCB1), where t is the number of rounds played so far
# and N_t(a) the pulls of arm a among them. The bonus is the width of a Hoeffding confidence
# bound (Hoeffding) whose failure probability is t⁻⁴. Write the index of every arm at once; an
# arm never pulled gets +∞. The policy `ucb1` (provided) pulls the first arm with the largest
# index.

# %% [markdown] role="ex2-predict"
# By round T = 5000, about how many times will UCB1 have pulled headline 1, the worst arm
# (gap 0.2): about 20, about 200, or about 2000? First compute the bound that the briefing's
# Theorem 1 puts on the expected count, 8 ln T / Δ² plus a constant of about 4.3.


# %% role="ex2-stub"
def ucb1_index(counts, sums, t):
    """UCB1's index of every arm after t rounds (UCB1): sums / counts + sqrt(2 ln t / counts),
    and np.inf for an arm never pulled. counts and sums have one entry per arm."""
    # TODO 2: start from an array of np.inf; fill in the arms with counts > 0. Compute the log
    # only when some arm has been pulled (at t = 0 none has, and log(0) is -inf).
    raise NotImplementedError(
        "TODO 2: write ucb1_index, or run lab.use_reference(2) to continue with the solution"
    )


# %% [markdown] role="ex2-hint"
# `seen = counts > 0` selects the pulled arms; `index[seen] = ...` fills only them. The
# natural log is `np.log`. Check one number by hand: one pull with reward 1 at t = 3 gives
# 1 + √(2 ln 3) ≈ 2.482.


# %% role="ex2-sol"
@lab.solution(2)
def ucb1_index(counts, sums, t):
    """r_hat + sqrt(2 ln t / N); inf for an arm never pulled."""
    index = np.full(len(counts), np.inf)  # never pulled: first
    seen = counts > 0
    if seen.any():
        mean = sums[seen] / counts[seen]
        bonus = np.sqrt(2 * np.log(t) / counts[seen])
        index[seen] = mean + bonus  # (UCB1)
    return index


# %% role="ex2-run"
def ucb1(counts, sums, t, rng):
    """Provided: UCB1 as a policy. It uses no randomness; rng is there to fit run_bandit."""
    return argmax_tie(ucb1_index(counts, sums, t))


stack = reward_stack(R_MEAN, T, SEEDS[0])
print(f"Seed {SEEDS[0]}: first rewards of the three headlines {stack[0].tolist()}")
print(f"  index at t = 3 after one pull each: {np.round(ucb1_index(np.ones(3), stack[0], 3), 3)}")
curves["UCB1"], pulls = [], []
for k in SEEDS:
    arms, regret = run_bandit(ucb1, reward_stack(R_MEAN, T, k), R_MEAN, k)
    curves["UCB1"].append(regret)
    pulls.append(np.bincount(arms, minlength=N_ARMS))
final["UCB1"] = np.array([c[-1] for c in curves["UCB1"]])
pulls = np.array(pulls)
bound = 8 * np.log(T) / GAPS[1] ** 2 + 1 + np.pi**2 / 3
print(f"UCB1, pulls of each headline by round {T}, over {len(SEEDS)} seeds:")
for a in range(N_ARMS):
    print(
        f"  headline {a}: mean {pulls[:, a].mean():7.1f}, from {pulls[:, a].min()} to {pulls[:, a].max()}"
    )
print(f"Theorem 1's bound on the expected pulls of headline 1: {bound:.0f}")
lab.metric("ucb1_pulls_h1", round(float(pulls[:, 1].mean()), 1))  # room code
fig, ax = plt.subplots()
plot_regret(ax, curves["UCB1"], "UCB1")
plot_regret(ax, curves["eps-greedy"], "ε-greedy, ε = 0.1", style="--")
ax.set_title(f"{len(SEEDS)} seeds, {T} rounds")
ax.legend()
plt.show()

# %% [markdown] role="ex2-explain"
# Each pull of an arm shrinks its bonus by a factor √(N/(N + 1)), while the bonus of an arm
# you leave alone grows like √(ln t): no arm is written off for good, which is what greedy got
# wrong (UCB1). Read the counts: is headline 1's count near Theorem 1's bound, or far below
# it? A bound can be true and still far from what happens; what does Theorem 1 promise, and
# what does it not?

# %% role="ex2-chk1"
lab.check(2, m04.check_ucb1_index, ucb1_index, label="2.1")

# %% role="ex2-chk2"
lab.check(2, m04.check_ucb1_counts, ucb1_index, label="2.2")

# %% [markdown] role="ex3-head"
# **Thompson sampling.** Give each arm's click rate a uniform prior, Beta(1, 1). For rewards
# in {0, 1}, Bayes' rule turns it into Beta(1 + S, 1 + F) after S clicks and F non-clicks
# (Beta posterior). Each round, draw one rate r̃(a) from every arm's posterior and pull the arm
# with the largest draw (Thompson sampling). Write the posterior, `beta_posterior`, then the
# policy, `thompson_step`, which draws with `rng.beta`.

# %% [markdown] role="ex3-predict"
# UCB1 pulled headline 1 a few hundred times by round 5000 (Exercise 2 printed the mean).
# Will Thompson sampling pull it more often or less often? Will it ever stop pulling it?


# %% role="ex3-stub"
def beta_posterior(counts, sums):
    """The Beta posterior of every arm from a Beta(1, 1) prior, for rewards in {0, 1} (Beta
    posterior). Returns two arrays, 1 + S and 1 + F, with S = sums and F = counts - sums."""
    # TODO 3 (part 1): two lines.
    raise NotImplementedError(
        "TODO 3: write beta_posterior, or run lab.use_reference(3) to continue with the solution"
    )


def thompson_step(counts, sums, t, rng):
    """Thompson sampling as a policy: draw one rate per arm, rng.beta(1 + S, 1 + F) with the two
    arrays from beta_posterior, and return argmax_tie of the draws."""
    # TODO 3 (part 2): two lines.
    raise NotImplementedError(
        "TODO 3: write thompson_step, or run lab.use_reference(3) to continue with the solution"
    )


# %% [markdown] role="ex3-hint"
# For Bernoulli rewards the sum of an arm's rewards is its number of clicks, so S = `sums`
# and F = `counts - sums`. `rng.beta` accepts arrays and draws one value per entry, so a single
# call draws for every arm. An arm never pulled has the posterior Beta(1, 1): a uniform draw.


# %% role="ex3-sol"
@lab.solution(3)
def beta_posterior(counts, sums):
    """Beta(1 + S, 1 + F) for every arm, from a uniform prior."""
    clicks = np.asarray(sums, dtype=float)  # S: rewards of 1
    misses = counts - clicks  # F: rewards of 0
    return 1 + clicks, 1 + misses  # (Beta posterior)


@lab.solution(3)
def thompson_step(counts, sums, t, rng):
    """One draw per arm from its posterior; the largest wins."""
    shape_s, shape_f = beta_posterior(counts, sums)
    draws = rng.beta(shape_s, shape_f)  # one r~(a) per arm
    return argmax_tie(draws)  # (Thompson sampling)


# %% role="ex3-run"
counts_ex, sums_ex = np.array([50, 10, 4]), np.array([20.0, 3.0, 3.0])
s_post, f_post = beta_posterior(counts_ex, sums_ex)
probe = np.random.default_rng(RUN_SEED)
picks = np.bincount(
    [thompson_step(counts_ex, sums_ex, 64, probe) for _ in range(4000)], minlength=3
)
print(f"Counts {counts_ex.tolist()}, clicks {sums_ex.astype(int).tolist()}:")
print(f"  posteriors Beta{list(zip(s_post.tolist(), f_post.tolist(), strict=True))}")
print(f"  means so far {np.round(sums_ex / counts_ex, 3).tolist()}; greedy would pull headline 2")
print(f"  share of 4000 Thompson draws won by each headline: {np.round(picks / 4000, 3).tolist()}")
curves["TS"], pulls_ts, halves = [], [], []
for k in SEEDS:
    arms, regret = run_bandit(thompson_step, reward_stack(R_MEAN, T, k), R_MEAN, k)
    curves["TS"].append(regret)
    pulls_ts.append(np.bincount(arms, minlength=N_ARMS))
    halves.append((np.sum(arms[: T // 2] == 1), np.sum(arms[T // 2 :] == 1)))
final["TS"] = np.array([c[-1] for c in curves["TS"]])
pulls_ts, halves = np.array(pulls_ts), np.array(halves)
print(f"Thompson sampling, pulls of each headline by round {T}, over {len(SEEDS)} seeds:")
for a in range(N_ARMS):
    print(
        f"  headline {a}: mean {pulls_ts[:, a].mean():7.1f}, from {pulls_ts[:, a].min()} "
        f"to {pulls_ts[:, a].max()} (UCB1: mean {pulls[:, a].mean():.1f})"
    )
print(
    f"  headline 1, first {T // 2} rounds: mean {halves[:, 0].mean():.1f} pulls; "
    f"last {T // 2}: mean {halves[:, 1].mean():.1f} (none on {np.sum(halves[:, 1] == 0)} seeds)"
)
fig, ax = plt.subplots()
plot_regret(ax, curves["TS"], "Thompson sampling")
plot_regret(ax, curves["UCB1"], "UCB1", style="--")
ax.set_title(f"{len(SEEDS)} seeds, {T} rounds")
ax.legend()
plt.show()

# %% [markdown] role="ex3-explain"
# Thompson sampling has no bonus. It pulls an arm about as often as the posterior gives that
# arm a chance of being the best (Thompson sampling): with counts (50, 10, 4), compare the
# share of draws headline 2 won with what greedy would do. As headline 1's posterior narrows
# below headline 2's, why do its pulls thin out without ever stopping?

# %% role="ex3-chk1"
lab.check(3, m04.check_beta_posterior, beta_posterior, label="3.1")

# %% role="ex3-chk2"
lab.check(3, m04.check_thompson_choices, thompson_step, label="3.2")

# %% role="ex3-chk3"
lab.check(3, m04.check_thompson_regret, thompson_step, run_bandit, label="3.3")

# %% [markdown] role="ex4-head"
# **Three methods over 20 seeds, and the tie rule.** With 5 or more seeds per side the
# workshop compares two methods by the interquartile mean (IQM) of their per-seed results,
# each with a 95% bootstrap interval; if the two intervals overlap, the verdict is a tie. With
# 2 to 4 seeds it shows every seed and compares the [min, max] ranges; with one seed it says
# "single seed, not a ranking". Write that rule, `verdict`, for pseudo-regret, where lower is
# better. Use `iqm(x)` and `bootstrap_ci(x, iqm)` (imported above) with their defaults, not
# `prl.evaluate.compare`: the checkpoint compares your function with it.

# %% [markdown] role="ex4-predict"
# Rank ε-greedy (ε = 0.1), UCB1 and Thompson sampling by pseudo-regret after 5000 rounds
# over your 20 seeds. Then read the same runs after 1000 rounds: will any pair be a tie there
# that is decided at 5000?


# %% role="ex4-stub"
def verdict(regret_a, regret_b):
    """The workshop's rule for two methods' per-seed regrets; lower is better. Returns "A", "B",
    "tie" or "single seed".

    One seed on either side: "single seed". 2 to 4 seeds (on the smaller side): each side's mean
    and its [min, max]. 5 or more: each side's iqm(x) and its interval bootstrap_ci(x, iqm).
    Overlapping intervals: "tie". Otherwise "A" if regret_a's center is lower, else "B".
    """
    # TODO 4: summarize each side as (center, low, high), then test the overlap.
    raise NotImplementedError(
        "TODO 4: write verdict, or run lab.use_reference(4) to continue with the solution"
    )


# %% [markdown] role="ex4-hint"
# Two intervals [lo_a, hi_a] and [lo_b, hi_b] overlap when `lo_a <= hi_b and lo_b <= hi_a`.
# Decide the seed count from the smaller side, `min(len(a), len(b))`.


# %% role="ex4-sol"
@lab.solution(4)
def verdict(regret_a, regret_b):
    """The tie rule; lower regret wins."""
    xa = np.asarray(regret_a, dtype=float)
    xb = np.asarray(regret_b, dtype=float)
    n = min(xa.size, xb.size)
    if n == 1:
        return "single seed"
    if n >= 5:  # IQM and a 95% bootstrap interval
        sa = (iqm(xa), *bootstrap_ci(xa, iqm))
        sb = (iqm(xb), *bootstrap_ci(xb, iqm))
    else:  # every seed: the mean and the range
        sa = (xa.mean(), xa.min(), xa.max())
        sb = (xb.mean(), xb.min(), xb.max())
    if sa[1] <= sb[2] and sb[1] <= sa[2]:
        return "tie"  # the intervals overlap
    return "A" if sa[0] < sb[0] else "B"


# %% role="ex4-run"
pairs = [("TS", "UCB1"), ("TS", "eps-greedy"), ("UCB1", "eps-greedy")]
verdicts = {}
for horizon in (budget.early, T):
    at = {m: np.array([c[horizon - 1] for c in curves[m]]) for m in ("eps-greedy", "UCB1", "TS")}
    print(f"After {horizon} rounds:")
    for m, x in at.items():
        lo, hi = bootstrap_ci(x, iqm)
        print(f"  {m:10s} IQM {iqm(x):7.1f}  [{lo:7.1f}, {hi:7.1f}]")
    for a_name, b_name in pairs:
        v = verdict(at[a_name], at[b_name])
        word = {"A": f"{a_name} lower", "B": f"{b_name} lower"}.get(v, v)
        verdicts[f"{a_name} vs {b_name} @ {horizon}"] = v
        print(f"  {a_name} vs {b_name}: {word}")
    print("  " + Budget(steps=horizon, seeds=len(SEEDS), hardware=HARDWARE).sentence())
lab.metric("verdicts", verdicts)
lab.metric("eps_iqm", round(iqm(final["eps-greedy"]), 1))  # room code
lab.metric("ucb1_iqm", round(iqm(final["UCB1"]), 1))  # room code
lab.metric("ts_iqm", round(iqm(final["TS"]), 1))  # room code
fig, ax = plt.subplots()
for m, style in [("eps-greedy", "--"), ("UCB1", ":"), ("TS", "-")]:
    plot_regret(ax, curves[m], m, style=style)
ax.axvline(budget.early, color=prl.plot.NEUTRAL_LIGHT, ls="-.", lw=1.5, label="early horizon")
ax.set_xscale("log")
ax.set_title(f"{len(SEEDS)} seeds; log-scale rounds")
ax.legend()
plt.show()

# %% [markdown] role="ex4-explain"
# Read the verdicts at both horizons. Which pair changed between 1000 and 5000 rounds, and
# why can a verdict depend on the budget at all? UCB1's regret grows like ln t and ε-greedy's
# like t: which curve would you bet on at 50 000 rounds, and what would you need to run before
# saying so? (Gotcha: "One seed is enough to compare.")

# %% role="ex4-chk"
lab.check(4, m04.check_verdict, verdict, final)

# %% [markdown] role="ex5-head"
# **Reward scale and ε.** Multiply every reward by 10: a click now pays 10. The provided cell
# reruns ε-greedy (ε = 0.1), UCB1 with its bonus unchanged, and UCB1 with its bonus multiplied
# by 10 as well, the width Hoeffding gives for rewards in [0, B] with B = 10 (scaled bonus).
# Every regret is divided by 10 again, so all are in the original units. Then it sweeps ε.
# Before you run it, write your predictions in `scale_directions`.

# %% [markdown] role="ex5-predict"
# For each method, will the mean pseudo-regret over your seeds, in the original units, go up,
# go down, or stay exactly the same on every seed when every reward is multiplied by 10?
# ε-greedy; UCB1 with the same bonus; UCB1 with the bonus × 10. Write the three answers in
# `scale_directions` below, with one sentence of reason each in a comment.


# %% role="ex5-stub"
def scale_directions():
    """Your predictions, written before the run: a dict with the keys "eps_greedy", "ucb1" and
    "ucb1_rescaled", each "up", "down" or "same" ("same" means identical on every seed)."""
    # TODO 5: return {"eps_greedy": ..., "ucb1": ..., "ucb1_rescaled": ...}, a reason for each
    # in a comment.
    raise NotImplementedError(
        "TODO 5: write scale_directions, or run lab.use_reference(5) to continue with the solution"
    )


# %% [markdown] role="ex5-hint"
# ε-greedy compares means with means: does multiplying all of them by 10 change which is
# largest? UCB1 adds a bonus to a mean: which of the two is in reward units, and which is not?


# %% role="ex5-sol"
@lab.solution(5)
def scale_directions():
    """The directions the theory predicts and the run shows."""
    return {
        "eps_greedy": "same",  # it only compares means: same order
        "ucb1": "up",  # the bonus is now small next to the gaps
        "ucb1_rescaled": "same",  # mean and bonus both x 10
    }


# %% role="ex5-run"
def ucb1_scaled(counts, sums, t, rng, B=1.0):
    """Provided: UCB1 for rewards in [0, B]: the bonus times B (scaled bonus)."""
    index = np.full(len(counts), np.inf)
    seen = counts > 0
    if seen.any():
        bonus = B * np.sqrt(2 * np.log(t) / counts[seen])
        index[seen] = sums[seen] / counts[seen] + bonus
    return argmax_tie(index)


predicted = scale_directions()
x10 = {"eps_greedy": eps_01, "ucb1": ucb1, "ucb1_rescaled": partial(ucb1_scaled, B=10.0)}
x1 = {"eps_greedy": final["eps-greedy"], "ucb1": final["UCB1"], "ucb1_rescaled": final["UCB1"]}
scaled = {}
for name, policy in x10.items():
    regrets = []
    for k in SEEDS:
        _, regret = run_bandit(policy, reward_stack(R_MEAN, T, k, scale=10.0), 10 * R_MEAN, k)
        regrets.append(regret[-1] / 10)  # back to the original units
    scaled[name] = np.array(regrets)
directions = {}
print(f"Rewards x 10, regret / 10, mean over {len(SEEDS)} seeds:")
for name in x10:
    a, b = x1[name], scaled[name]
    if np.allclose(a, b, rtol=0, atol=1e-9):
        directions[name] = "same"
    else:
        directions[name] = "up" if b.mean() > a.mean() else "down"
    print(
        f"  {name:14s} x1 {a.mean():7.1f}   x10 {b.mean():7.1f}   {directions[name]:4s} "
        f"(you predicted {predicted.get(name)})"
    )
    print(
        f"  {'':14s} x10 higher on {int(np.sum(b > a + 1e-9))} seeds, "
        f"lower on {int(np.sum(b < a - 1e-9))}; tie rule, x10 vs x1: {verdict(b, a)}"
    )
lab.metric("scale_directions", directions)
lab.metric("ucb1_x10_gap", round(float(scaled["ucb1"].mean() - final["UCB1"].mean()), 1))
try:
    stack_x10 = reward_stack(R_MEAN, 50, SEEDS[0], scale=10.0)
    run_bandit(thompson_step, stack_x10, 10 * R_MEAN, SEEDS[0])
except ValueError as exc:
    print(f"Thompson sampling with rewards x 10 stops: ValueError: {exc}")
epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
sweep = {}
for e in epsilons:
    sweep[e] = np.array(
        [
            run_bandit(partial(eps_greedy, epsilon=e), reward_stack(R_MEAN, T, k), R_MEAN, k)[1][-1]
            for k in SEEDS
        ]
    )
lab.metric("eps_sweep_iqm", {str(e): round(iqm(x), 1) for e, x in sweep.items()})
fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(10, 3.8))
for i, (name, label) in enumerate([("ucb1", "UCB1, same bonus"), ("eps_greedy", "ε-greedy")]):
    ax_s.plot(np.full(len(SEEDS), 2 * i), x1[name], "o", mfc="none", label=f"{label}, x 1")
    ax_s.plot(np.full(len(SEEDS), 2 * i + 1), scaled[name], "^", label=f"{label}, x 10")
ax_s.set(xticks=[0, 1, 2, 3], xticklabels=["UCB1\nx1", "UCB1\nx10", "ε-gr.\nx1", "ε-gr.\nx10"])
ax_s.set(ylabel="pseudo-regret at T (original units)", title="Every seed, rewards x 1 and x 10")
centers = [iqm(sweep[e]) for e in epsilons]
cis = np.array([bootstrap_ci(sweep[e], iqm) for e in epsilons])
xs = np.arange(len(epsilons))
ax_e.errorbar(
    xs,
    centers,
    yerr=[np.subtract(centers, cis[:, 0]), cis[:, 1] - centers],
    marker="o",
    capsize=3,
    label="ε-greedy, IQM [95% CI]",
)
ax_e.axhline(iqm(final["UCB1"]), ls="--", color=prl.plot.NEUTRAL_LIGHT, label="UCB1, IQM")
ax_e.set(
    xticks=xs,
    xticklabels=[str(e) for e in epsilons],
    xlabel="ε",
    ylabel="pseudo-regret at T",
    title=f"ε sweep, {len(SEEDS)} seeds, {T} rounds",
)
ax_e.legend()
plt.show()

# %% [markdown] role="ex5-explain"
# Which of the three answers did the run confirm, and why does ε-greedy pull exactly the same
# arms? UCB1's bonus is the width of (Hoeffding) for rewards in [0, 1]; next to gaps 10 times
# larger it acts like a small bonus. Look at UCB1 × 10 seed by seed: why is the mean up while
# the tie rule may call it a tie, and some seeds even improve? In the sweep, why is regret high
# at both ends of ε? (Gotcha: "UCB has no parameters.")

# %% role="ex5-chk"
lab.check(5, m04.check_scale_directions, scale_directions)

# %% [markdown] role="ex6-head"
# **Planted bug.** A colleague wrote a UCB1 agent in another style: an object that keeps its
# own counts and reward sums, with `act(t)` to choose and `update(arm, reward)` to learn.
# `play` (provided) runs it on a reward stack. It runs without a warning and its regret curve
# looks plausible on some seeds. It is wrong. Read the histogram of pulls per headline
# against your UCB1's from Exercise 2, find the line, fix it.

# %% [markdown] role="ex6-predict"
# In your UCB1, the worse headlines kept being pulled now and then, a few hundred times by
# round 5000 for headline 1. If an agent's counts never grew past their starting value, what
# would its bonus be for every arm in round t, and what would decide which arm it pulls? What
# would the histogram of pulls look like on one seed?


# %% role="ex6-stub"
class UCB1Suspect:
    """A colleague's UCB1 agent. It keeps its own counts and reward sums. Is it right?"""

    # TODO 6: once the histogram below shows the problem, find the missing line and add it.
    # Then rerun this cell and the run cell.
    def __init__(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def act(self, t):
        if t < len(self.counts):
            return t  # play each arm once
        n = np.maximum(self.counts, 1)  # never divide by zero
        index = self.sums / n + np.sqrt(2 * np.log(t) / n)
        return argmax_tie(index)

    def update(self, arm, reward):
        self.sums[arm] += reward


# %% [markdown] role="ex6-hint"
# Print `agent.counts` after a run. Which attribute is read in `act` but never written in
# `update`? The guard `np.maximum(self.counts, 1)` is why nothing ever warned you.


# %% role="ex6-sol"
@lab.solution(6)
class UCB1Suspect:
    """The colleague's UCB1, fixed: update counts the pull."""

    def __init__(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def act(self, t):
        if t < len(self.counts):
            return t  # play each arm once
        n = np.maximum(self.counts, 1)  # never divide by zero
        index = self.sums / n + np.sqrt(2 * np.log(t) / n)
        return argmax_tie(index)

    def update(self, arm, reward):
        self.counts[arm] += 1  # the missing line: N_t(a) grows
        self.sums[arm] += reward


# %% role="ex6-run"
def play(agent, stack, r_mean):
    """Provided: run an agent with act(t) and update(arm, reward) on a reward stack.
    Returns (arms, regret), as run_bandit does."""
    T_run, n_arms = stack.shape
    top = np.zeros(n_arms, dtype=int)  # the next row of each arm's stack
    arms = np.zeros(T_run, dtype=int)
    for t in range(T_run):
        a = agent.act(t)
        agent.update(a, stack[top[a], a])
        top[a] += 1
        arms[t] = a
    gaps = np.max(r_mean) - np.asarray(r_mean)
    return arms, np.cumsum(gaps[arms])


hist, final_suspect = [], []
for k in SEEDS:
    agent = UCB1Suspect(N_ARMS)
    arms, regret = play(agent, reward_stack(R_MEAN, T, k), R_MEAN)
    hist.append(np.bincount(arms, minlength=N_ARMS))
    final_suspect.append(regret[-1])
hist = np.array(hist)
print(f"The agent after seed {SEEDS[-1]}: counts {agent.counts}, sums {agent.sums}")
print(
    f"  its bonus in the last round: {np.round(np.sqrt(2 * np.log(T - 1) / np.maximum(agent.counts, 1)), 3)}"
)
print(f"Pulls per headline by round {T}, first 5 seeds (your UCB1 in brackets):")
for i in range(5):
    print(f"  seed {SEEDS[i]}: {hist[i].tolist()}   [{pulls[i].tolist()}]")
lab.metric("suspect_pulls", hist.mean(axis=0).round(1).tolist())  # per headline, not pooled
fig, ax = plt.subplots()
xs = np.arange(N_ARMS)
for shift, (h, label, hatch) in zip(
    (-0.2, 0.2), [(pulls, "your UCB1 (Exercise 2)", ""), (hist, "UCB1Suspect", "//")], strict=True
):
    ax.bar(xs + shift, h.mean(axis=0), width=0.4, hatch=hatch, label=f"{label}, mean")
    ax.errorbar(
        xs + shift,
        h.mean(axis=0),
        yerr=[h.mean(0) - h.min(0), h.max(0) - h.mean(0)],
        fmt="none",
        color=prl.plot.NEUTRAL_LIGHT,
        capsize=4,
    )
ax.set(
    xticks=xs,
    xticklabels=[f"headline {a}" for a in xs],
    ylabel=f"pulls by round {T}",
    title=f"Pulls per headline, {len(SEEDS)} seeds (bars: min to max)",
)
ax.legend()
plt.show()

# %% [markdown] role="ex6-explain"
# Before the fix, the counts stayed at 0, so the guard divided by 1: the "mean" was the total
# reward, and the bonus was the same √(2 ln t) for every arm. Which arm won, and why did it
# keep winning? Why is a histogram with one arm near T and the others near 1 a sign that the
# bonus is not shrinking, and not a sign that UCB1 found the best arm? (UCB1)
#
# **Diagnostics for this lab.** Plot the pull counts per arm, not only the regret. A healthy
# optimistic method keeps pulling every worse arm, a number of times that grows slowly with t
# (hundreds by round 5000 here for UCB1); a method that stops pulling some arm forever is
# greedy somewhere, whatever its name. Print the bonus at the end of a run: it must differ
# between arms and shrink with each arm's count. Run every method on several seeds, never one:
# greedy looks perfect on the seeds where it happens to keep the best arm. Compare pseudo-regret,
# not reward, because it removes the noise of the clicks; and before comparing methods,
# check that their rewards are on the scale their assumptions name.

# %% role="ex6-chk"
lab.check(6, m04.check_ucb1_suspect, UCB1Suspect)

# %% [markdown]
# **Your room code.** The last cell, `lab.finish()`, prints one line that starts with `ROOM m04`.
# It holds your seat, the checkpoints you passed, whether QUICK budgets ran, and the numbers the
# debrief pools: on how many seeds greedy left headline 2 behind (`greedy_locked`, Exercise 1);
# UCB1's mean pulls of headline 1 (`ucb1_pulls_h1`, Exercise 2); the IQM of each method's
# pseudo-regret at T (`eps_iqm`, `ucb1_iqm`, `ts_iqm`, Exercise 4); and how much UCB1's mean
# regret changed with rewards × 10 (`ucb1_x10_gap`, Exercise 5). Paste it where your instructor
# says. Your run record keeps every number the lab printed.

# %% [markdown] role="stretch"
# **Gaussian Thompson sampling.** Thompson sampling needs a likelihood. The Beta posterior
# assumes rewards in {0, 1}; with rewards × 10 it stops. A Gaussian model accepts any reward:
# give arm a's mean the prior N(0, s₀²) and treat each reward as its mean plus Gaussian noise
# with standard deviation σ. The posterior is Gaussian with precision 1/s₀² + N/σ² and mean
# (sum of rewards / σ²) / precision (Lattimore and Szepesvári, Example 34.7, with unit noise).
# Write it, draw one mean per arm, pull the largest. Then run it on rewards × 1 with σ = 0.5,
# on rewards × 10 with σ = 0.5 and with σ = 5: where did the reward scale go this time?


# %% role="stretch-stub"
def gaussian_posterior(counts, sums, noise_sd, prior_sd):
    """Posterior mean and standard deviation of every arm's mean: prior N(0, prior_sd^2),
    rewards = mean + N(0, noise_sd^2) noise. Returns two arrays (mean, sd)."""
    # Stretch: precision = 1/prior_sd^2 + counts/noise_sd^2; mean = (sums/noise_sd^2)/precision.
    raise NotImplementedError(
        "Stretch: write gaussian_posterior, or run lab.use_reference('stretch') to see the solution"
    )


# %% role="stretch-sol"
@lab.solution("stretch")
def gaussian_posterior(counts, sums, noise_sd, prior_sd):
    """Gaussian prior and noise: the posterior of each arm's mean."""
    precision = 1 / prior_sd**2 + counts / noise_sd**2
    mean = (sums / noise_sd**2) / precision
    return mean, np.sqrt(1 / precision)


# %% role="stretch-run"
lab.check("stretch", m04.check_gaussian_posterior, gaussian_posterior, label="stretch")


def gaussian_ts(counts, sums, t, rng, noise_sd=0.5, prior_sd=1.0):
    """Provided: Gaussian Thompson sampling on your posterior."""
    mean, sd = gaussian_posterior(counts, sums, noise_sd, prior_sd)
    return argmax_tie(rng.normal(mean, sd))


runs = {
    "x1, sigma 0.5": (partial(gaussian_ts, noise_sd=0.5, prior_sd=1.0), 1.0),
    "x10, sigma 0.5": (partial(gaussian_ts, noise_sd=0.5, prior_sd=1.0), 10.0),
    "x10, sigma 5": (partial(gaussian_ts, noise_sd=5.0, prior_sd=10.0), 10.0),
}
for name, (policy, scale) in runs.items():
    regrets = [
        run_bandit(policy, reward_stack(R_MEAN, T, k, scale=scale), scale * R_MEAN, k)[1][-1]
        / scale
        for k in SEEDS
    ]
    print(f"Gaussian TS, rewards {name}: IQM regret {iqm(regrets):7.1f} (original units)")
print(f"Beta TS, rewards x1: IQM regret {iqm(final['TS']):7.1f}")
