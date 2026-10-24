# origin: labs/src/02-planning.py
# The reference version of your Lab 2 code (Exercise 1): value iteration and the greedy
# policy with its tie rule.
def value_iteration(P, R, gamma, tol):
    """Value iteration from V = 0; stop when max|V_new - V| < tol."""
    V, k = np.zeros(P.shape[0]), 0  # V_0 = 0; k counts sweeps
    while True:
        Q = R + gamma * P @ V  # Q_k, shape (S, A) (eq-vi)
        V_new = Q.max(axis=1)  # V_{k+1} = T* V_k
        k += 1
        if np.max(np.abs(V_new - V)) < tol:  # eq-vi-stop
            # Return the Q of the returned V, for greedy().
            return V_new, R + gamma * P @ V_new, k
        V = V_new


def greedy(Q):
    """The first action within rounding of each row's max."""
    tie_tol = 1e-9 * (1 + np.abs(Q).max())
    return (Q >= Q.max(axis=1, keepdims=True) - tie_tol).argmax(axis=1)
