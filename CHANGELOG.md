# Changelog

All notable changes to this workshop. Dates are UTC.

## Unreleased

### Added
- Phase 0 (2026-10-09): `PLAN.md`, `OPEN_QUESTIONS.md`, `references.bib`, persona agents in `.claude/agents/`.
- Phase 1 (merged 2026-10-09; its two human Colab smoke runs are not filed yet):
  - `prl` package: runtime, run records, lab harness, checks, five environments, evaluation, data and plotting, with 53 tests.
  - Notebook build from jupytext sources, with a rhythm lint. Worked and learner checks. `00-setup` and a Module 1 stub.
  - `_variables.yml` with a validator, and generated includes for every page. Readiness built from `runs/`.
  - CI: lint, tests and drift gate; colab-compat against Colab's freeze; colab-sim notebook jobs; infra lint. A Pages workflow.
  - `infra/aws` (CloudFormation, `prl-aws`, price table from the AWS Price List).
- Phase 2 (merged 2026-10-10): the golden Module 1, "MDPs, returns and the Bellman equation".
  - Briefing with two light and dark figures, an OJS contraction demo, Gotchas and annotated further reading.
  - Lab: seven exercises and a stretch, with a planted bug (an evaluator that uses `P_pi.T`).
  - `prl.checks.m01` with fixtures from `scripts/make_expected.py`; 28 mutants in `labs/mutants/m01.py`; a verify mode that must reject every stub and mutant.
  - Eight knowledge checks for Module 1. Technical, academic and pedagogy reviews applied.
- Phase 3, Day 1 (2026-10-17): Module 2, "Planning: value iteration, policy iteration and model error".
  - Briefing with a worked two-state example, the greedy-loss and planning-loss bounds, two subfigured figures and a "ties need a tolerance" section.
  - Lab: six exercises (core 3 + 50) and a stretch (modified policy iteration). The planted bug, VI stopping on the mean change, runs on three models.
  - `prl.checks.m02` with fixtures, 49 mutants (verify rejects 55 wrong versions), knowledge checks, and a laptop run record. Pedagogy, UI and academic reviews applied.
- Phase 3, Day 1 (2026-10-17): Module 3, "Model-free prediction and control".
  - Briefing on MC, TD(0) with its bias and variance, SARSA vs Q-learning on the 4 × 12 cliff, terminated vs truncated, and the n-step dial, with two figures drawn from recorded experiments.
  - Lab: seven exercises (core 50) and a stretch (an n-step sweep). The planted bug, Q-learning that bootstraps through the terminal, is visible only from an optimistic start; its return curve shows nothing.
  - `prl.checks.m03`: exact checks plus six stochastic checkpoints with thresholds from 100-seed protocol runs at both budgets (arm64; colab-sim still owed). Verify mode skips stochastic checkpoints, and `exact_only` mutants are left to the exact checks. Knowledge checks and laptop run records. Pedagogy, UI and academic reviews applied.
- Phase 3, Day 2 (2026-10-24): Module 4, "Bandits from first principles".
  - Briefing on pseudo-regret, greedy's trap, Hoeffding's inequality and the UCB1 bonus, Beta–Bernoulli Thompson sampling, and how ε and the reward scale break each method's assumptions.
  - Lab: six exercises (core 52) and a Gaussian Thompson sampling stretch, on a per-arm reward stack. Exercise 4 continues ε-greedy and UCB1 to 20,000 rounds, so the budget lesson is shown. The planted bug, a UCB1 whose counts never grow, shows in a per-seed pull histogram.
  - `prl.checks.m04`: exact replays plus one stochastic checkpoint (Thompson sampling's regret, thresholds from 100-seed protocol runs, arm64), figures from three recorded experiments including 100 seats, knowledge checks and laptop run records. Pedagogy, UI and academic reviews applied.
- Phase 3, Day 2 (2026-10-24): Module 5, "Contextual bandits for recommendation".
  - Briefing on framing a recommender as a contextual bandit, incremental ridge regression, the confidence ellipsoid and LinUCB's bonus, linear Thompson sampling, misspecification, drift and the launch, and why logging must be stochastic.
  - Lab: eight exercises (core 3 + 50) on `RecSim`, with a warm-launch condition chosen from a record, a MABWiser parity stretch (pinned for Colab) and a planted logger that records the greedy action's propensity.
  - `prl.checks.m05`: seven exact checkpoints with fixtures, figures from three recorded experiments, knowledge checks and laptop run records. Pedagogy, UI and academic reviews applied.
- Phase 3, Day 2 (2026-10-24): Module 6, "Exploration in MDPs".
  - Briefing on RiverSwim's hitting time by first-step analysis, optimistic initialization, MBIE-EB's bonus inside the backup, visitation and regret against the DP optimum, and where count bonuses stop.
  - Lab: seven exercises (core 52) and a PSRL stretch on a 10-state river. The planted bug applies the bonus only at action selection; its agent never reaches the right end.
  - `prl.checks.m06`: exact checks plus three stochastic checkpoints with thresholds from 100-seed protocol runs (arm64), figures from a recorded 10-seed experiment, knowledge checks and laptop run records. Pedagogy, UI and academic reviews applied.
- Phase 3, Day 3 (2026-10-25): Module 7, "Function approximation and why it can diverge".
  - Briefing on counting the three-product shop's states, linear values, the semi-gradient TD(0) update and its non-symmetric matrix A, a two-state divergence demo, Baird's counterexample, the deadly triad and feature scale.
  - Lab: seven exercises (core 51) and a TDC stretch. Baird with expected updates, each leg of the triad removed (followed to 10⁶ sweeps with matrix powers), and linear Q-learning on CartPole with 625 unit-length RBF features against a recorded 20-seed band. The planted bug, a colleague's featurizer whose bumps are not unit length, drives ‖W‖ to 10⁸⁵ or more while the greedy return can still look fine; the weight-norm log exposes it.
  - `prl.checks.m07`: exact checks plus two training checkpoints (greedy return ≥ 160 from 100-seed protocol runs at both budgets, arm64; a weight-norm bound rejects the planted bug), 40 mutants (verify rejects 48 wrong versions), the band `m07_linear_q`, figures that assert every quoted number, knowledge checks and laptop run records. Pedagogy, UI and academic reviews applied.
- Phase 3, Day 3 (2026-10-25): Module 8, "DQN from scratch".
  - Briefing on replay, the target network and the Huber loss (Mnih et al.'s clipped error is its gradient), masks on `terminated`, the max bias and Double DQN (van Hasselt et al.'s Theorem 1 with its conditions), and equal-budget bands.
  - Lab: seven exercises (core 3 + 51) and a LunarLander stretch, built on one provided configuration that Module 9 reuses through `labs/shared/`. The planted bug, a colleague's buffer that samples with its write pointer, is diagnosed from the histogram of drawn ages. Participants place their live DQN and Double DQN seeds over recorded bands, measure Q(s₀) against the MC return (below on most seats: Q tracks the target copies' lag), and call the reference against SB3 2.9 DQN at the same budget (a tie).
  - `prl.checks.m08`: exact checks plus a training checkpoint (best evaluation minus the first ≥ 91.0, 200-seed protocol runs at both budgets, arm64), 34 wrong versions rejected in verify mode, seven bands (reference DQN, Double DQN, SB3), a 20-seat record, figures that assert every quoted number, knowledge checks and laptop run records. Pedagogy, UI and academic reviews applied.
- `health.yml` (weekly): the locked-environment run on Linux and macOS arm64 with live budgets; `colab_constraints.py bump`, which refreshes the Colab freeze, runs colab-compat and colab-sim on it and opens a PR with their results; and a `uv lock --upgrade` report. The colab-sim steps moved to a composite action, with the same checks (2026-10-10).
- `scripts/threshold_protocol.py` and `prl.checks.StochasticCheck`: thresholds for stochastic checkpoints, set from at least 100 (QUICK) or 20 (live) seeds of the solution and its mutants, and recorded as experiment records (2026-10-10).

### Fixed
- Colab setup: one pin per package, so `pip` no longer fails with ResolutionImpossible on gymnasium 1.4.0 (2026-10-09).
- A stuck participant is pointed to `lab.use_reference(N)` by the stubs and the solution cells (2026-10-09).
- CI's test job reinstalled torch on every `uv run` despite its dev-only install; `UV_NO_SYNC=1` keeps it dev-only (2026-10-10).
- `add_run_record.py` no longer overwrites a filed record that has the same file name; the new one gets `-2`, `-3`, ... (2026-10-10).
