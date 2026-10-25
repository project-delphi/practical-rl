# origin: labs/src/08-dqn.py
# The reference version of your Lab 8 code: the replay buffer (Exercise 1), the masked target
# (Exercise 2), the training loop (Exercise 3), the Double DQN target (Exercise 4) and Monte
# Carlo returns (Exercise 5). Include labs/shared/dqn_setup.py first: train_dqn calls
# dqn_parts, epsilon_at, act, gather_q, huber and dqn_target from it, each by name when it
# runs, and dqn_parts builds this file's ReplayBuffer.
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


def train_dqn(make_env, config, seed, steps, *, target_fn=None, **log_options):
    """DQN (Mnih et al. 2015, Algorithm 1). Returns (q_net, log)."""
    target_fn = target_fn or dqn_target
    p = dqn_parts(make_env, config, seed, steps, **log_options)
    obs = p.obs
    for t in range(1, steps + 1):
        eps = epsilon_at(t, steps, config)
        a = act(p.q_net, obs, eps, p.n_actions, p.rng)
        next_obs, r, terminated, truncated, _ = p.env.step(a)
        p.buffer.add(obs, a, r, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = p.env.reset()
        if t > config.learning_starts and t % config.train_freq == 0:
            batch = p.buffer.sample(config.batch_size, p.rng)
            q_values = p.q_net(batch.obs)
            q_sa = gather_q(q_values, batch.act)
            y = target_fn(batch, p.q_net, p.target_net, config.gamma)
            loss = huber(y - q_sa, config.huber_threshold).mean()  # (DQN loss)
            p.opt.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(p.q_net.parameters(), config.max_grad_norm)
            p.opt.step()
            p.log.update(t, batch, q_values, q_sa, y, loss, norm)
        if config.target_sync and t % config.target_sync == 0:
            p.target_net.load_state_dict(p.q_net.state_dict())  # w- <- w
            p.log.sync(t)
        p.log.step(t, eps, r, terminated, truncated)
    p.env.close()
    return p.q_net, p.log.result()


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
