# The reference version of your Lab 1 code (Exercise 4): exact policy evaluation.
def evaluate_exact(P, R, gamma, pi):
    """V^pi from (I - gamma P_pi) V = r_pi, where pi is an (S, A) matrix of probabilities."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)
