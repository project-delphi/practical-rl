# The reference version of Lab 1's evaluate_exact (Module 1, Exercise 4), copied verbatim from
# its solution cell. Later labs include it with `# %% include labs/shared/m01_evaluate_exact.py`.


def evaluate_exact(P, R, gamma, pi):
    """V^pi from (I - gamma P_pi) V = r_pi, where pi is an (S, A) matrix of probabilities."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)
