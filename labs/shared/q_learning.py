# origin: labs/src/03-model-free.py
# The reference version of your Lab 3 code (Exercise 5): Q-learning. It reads S, A and
# epsilon_greedy from the notebook (a scalar q0 needs S and A).
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
