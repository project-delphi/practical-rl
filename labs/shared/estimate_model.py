# origin: labs/src/02-planning.py
# The reference version of your Lab 2 code (Exercise 3): the counts estimator. A pair never
# tried gets a self-loop with reward 0.
def estimate_model(log, n_states, n_actions):
    """Counts -> (P_hat, R_hat, counts); unvisited pairs self-loop."""
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    for s, a, r, s2 in log:  # tally N(s, a), N(s, a, s'), reward sums
        s, a, s2 = int(s), int(a), int(s2)
        counts[s, a] += 1
        P_hat[s, a, s2] += 1
        R_hat[s, a] += r
    for s in range(n_states):
        for a in range(n_actions):
            if counts[s, a] > 0:  # eq-counts
                P_hat[s, a] /= counts[s, a]
                R_hat[s, a] /= counts[s, a]
            else:  # unvisited: a self-loop with reward 0
                P_hat[s, a, s] = 1.0
    return P_hat, R_hat, counts
