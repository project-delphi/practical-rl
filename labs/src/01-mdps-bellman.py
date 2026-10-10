# ---
# jupyter:
#   prl:
#     module: m01
# ---

# %% [markdown]
# > **Phase 1 stub.** This notebook holds one exercise so the build, checks and CI can be
# > tested end to end. The full lab (seven exercises) arrives in Phase 2.

# %% [markdown] role="ex1-head"
# The **return** is the quantity every method this week estimates or maximizes:
# the discounted sum of rewards from now on.

# %% [markdown] role="ex1-predict"
# Rewards 1, 1, 1 with discount 0.9: what is the return? Work it out before you code.

# %% role="ex1-stub"
def discounted_return(rewards, gamma):
    """Return G_0 = r_1 + gamma r_2 + gamma^2 r_3 + ... for a list of rewards."""
    # TODO 1: the first reward is not discounted; each later one is discounted once more.
    raise NotImplementedError(
        "TODO 1: write discounted_return, or run lab.use_reference(1) to continue with the solution"
    )


# %% [markdown] role="ex1-hint"
# Loop backward: `g = r + gamma * g` processes the list from the end, and needs no powers.


# %% role="ex1-sol"
@lab.solution(1)
def discounted_return(rewards, gamma):
    """Return G_0 = r_1 + gamma r_2 + gamma^2 r_3 + ... for a list of rewards."""
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


# %% role="ex1-run"
print(discounted_return([1, 1, 1], 0.9))

# %% [markdown] role="ex1-explain"
# Did you get 2.71? Which reward would you have to change to make the return negative, and
# why does the last one matter least?

# %% role="ex1-chk"
from prl.checks import m01

lab.check(1, m01.check_discounted_return, discounted_return)
