# origin: labs/src/08-dqn.py
# The reference version of your Lab 8 code: the replay buffer (Exercise 1), the masked target
# and the Huber loss (Exercise 3), the training loop (Exercise 4), the Double DQN target
# (Exercise 5) and Monte Carlo returns (Exercise 6). Include labs/shared/dqn_setup.py first:
# train_dqn reads CONFIG's fields and calls make_q_net, epsilon_at, act, gather_q, dqn_target,
# DQNLogger and to_batch from it, each by name when it runs.
class ReplayBuffer:
    """A ring buffer of transitions in preallocated NumPy arrays."""

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0  # the slot the next transition goes to
        self.size = 0  # how many slots hold a transition

    def add(self, obs, act, rew, next_obs, terminated, truncated):
        """Write one transition at slot ptr; advance ptr and size."""
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i] = next_obs
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.ptr = (i + 1) % self.capacity  # the oldest goes next
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Uniform over the filled slots (replay sampling)."""
        idx = rng.integers(0, self.size, size=batch_size)
        return to_batch(self, idx)


def td_target(batch, q_next, gamma):
    """r + gamma (1 - terminated) q_next: truncation still bootstraps."""
    return batch.rew + gamma * (1.0 - batch.terminated) * q_next


def huber(x, threshold=1.0):
    """Elementwise Huber loss with the given threshold (Huber)."""
    a = x.abs()
    quadratic = 0.5 * x**2
    linear = threshold * (a - 0.5 * threshold)
    return torch.where(a <= threshold, quadratic, linear)


def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):
    """DQN (Mnih et al. 2015, Algorithm 1). Returns (q_net, log)."""
    target_fn = target_fn or dqn_target
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env()
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q_net = make_q_net(obs_dim, n_actions, config)
    target_net = make_q_net(obs_dim, n_actions, config)
    target_net.load_state_dict(q_net.state_dict())  # w- = w
    opt = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size, obs_dim)
    log = DQNLogger(config, steps, q_net, target_net, make_env, **log_options)
    obs, _ = env.reset(seed=seed)
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)
        a = act(q_net, obs, eps, n_actions, rng)
        next_obs, r, terminated, truncated, _ = env.step(a)
        buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = buffer.sample(config.batch_size, rng)
            q_values = q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)
            y = target_fn(batch, q_net, target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()
            opt.zero_grad()
            loss.backward()
            params = q_net.parameters()
            norm = nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            opt.step()
            log.update(t, batch, q_values, q_sa, y, loss, norm)
        if config.target_sync and t % config.target_sync == 0:
            target_net.load_state_dict(q_net.state_dict())  # w- <- w
            log.sync(t)
        log.step(t, eps, r, terminated, truncated)
    env.close()
    return q_net, log.result()


def double_dqn_target(batch, q_net, target_net, gamma):
    """Select with the online network, evaluate with the target."""
    with torch.no_grad():
        a_next = q_net(batch.next_obs).argmax(dim=1)  # w selects
        q_next = gather_q(target_net(batch.next_obs), a_next)  # w- values
        return td_target(batch, q_next, gamma)  # (Double DQN target)


def mc_return(rewards, gamma):
    """Returns-to-go, computed backward (MC return)."""
    g = np.zeros(len(rewards))
    total = 0.0
    for t in reversed(range(len(rewards))):
        total = rewards[t] + gamma * total
        g[t] = total
    return g
