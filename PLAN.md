# Practical Reinforcement Learning: plan

A five-day, practical, first-principles reinforcement learning workshop by Genial Labs.

| | |
|---|---|
| Repository | `project-delphi/practical-rl` (public) |
| Site | `https://project-delphi.github.io/practical-rl/` |
| License | Text CC BY 4.0, code MIT |
| Language | Python only |
| Spelling | American English |

**Status: Phase 0 (research and plan), 2026-10-09.**

- The draft was reviewed by four persona agents (§11): academic director, pedagogy expert, technical expert and UI expert. Their findings are folded in.
- Nothing in this file is a measurement. Every minute count is a design budget. It includes the time participants spend on the Predict → Run → Explain → Check rhythm. Phase 3 timed runs confirm it for compute, and a human pilot confirms it for people.
- Open decisions, departures from the brief, and anything not yet verified are in [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).
- Citations are in [`references.bib`](references.bib). Each entry's `annote` says what was opened and which location we cite.

---

## 0. Decisions already taken (2026-10-09)

| Topic | Decision |
|---|---|
| Repository | `project-delphi/practical-rl`, public, published to GitHub Pages |
| AWS infrastructure as code | **CloudFormation** with AWS CLI wrapper scripts. The end-to-end AWS run is a human item on the release checklist. Phase 1 only lints the templates (`cfn-lint`, `shellcheck`). |
| Module 12 | **The core runs on CPU.** It covers an exact toy, plus checks that log-probs and the DPO loss on a small language model match TRL. The T4 training run is the stretch; its 3-seed evidence comes from run records. |
| Deep-RL labs | **"Recorded bands + your seed."** At most 10–12 minutes of live compute per lab, and no cell runs longer than 3 minutes. The participant's live seed is overlaid on bands from scripted experiment runs stored in `runs/`. |
| Spelling | American English |
| Quarto | 1.10.19, pinned in CI. The local copy is upgraded in Phase 1. |
| Roles | Persona subagents in `.claude/agents/`: `academic-director`, `pedagogy-expert`, `technical-expert`, `ui-expert`, plus a fresh-context `verifier` for Phase 5 (§11). |
| Parallel work | **Git worktrees**, one branch per module or task. The lead merges after review. `prl/` changes happen on `main` only (§11). |

---

## 1. Conventions from `project-delphi/nlp-llms`

The reference was read from its committed `main` at `3022908` (2026-10-07). `origin/main` has since moved to `af3dabe` (2026-10-09). Phase 1 re-reads anything we copy from it.

### Kept as they are
1. **`_variables.yml` is the single source of truth.** It feeds the schedule, the day pages, module headers, slides, notebook headers and the readiness page. CI regenerates everything from it and fails on any diff (the *drift gate*).
2. **Module page skeleton.** Three parts are generated:
   - the header and facts box (`_includes/module-NN.md`);
   - "In the lab" (`_includes/lab-NN.md`), built from the notebook's `## Exercise N · title (M min)` headings;
   - "Agenda" (`_includes/live-NN.md`), built from the page's `live:` front matter.

   The rest is hand-written:
   - "Where we are"
   - numbered sections, each opening with **In this section:**
   - labeled equations
   - "In the lab." callouts
   - "Check yourself" boxes with folded answers
   - worked examples
   - optional collapsed callouts
   - Summary, the equations-to-lab table, Gotchas, Next steps and an annotated Further reading
3. **Lua filters.** `disclosure.lua` turns collapsed callouts into native `<details>`, so they work from the keyboard. `live.lua` adds the "In the room" badges.
4. **Harness semantics.**
   - A folded solution never overwrites participant code.
   - `use_reference(N)` exists for participants who are stuck.
   - Every checkpoint reports whose code it ran.
   - Test modes: worked, learner and verify.
5. **Evidence.**
   - Run records carry `content_sha` and are validated, including a credential reject.
   - A run counts as *teaching-eligible* only when all of these hold: designed runtime, real path, worked mode, whole notebook, not QUICK, current hash.
   - A release check gates the GitHub Release.
6. **Instructor patterns.**
   - Warm-up: 6 minutes alone, 4 with a neighbor, 5 on the two most-missed questions.
   - Debrief: the room's numbers (4), one misconception (3), the bridge (3).
   - Running results table.
   - "When the clock slips" rules.
   - Per-module facilitator notes and a "When things fail" table.
   - An entry check with a rule: miss 2 of 3 in an area and you get reading.
7. **Tooling.** uv groups; a CPU torch index on Linux CI; a render job with `--no-project`; `uv.lock` committed and used with `--locked`; deterministic SVG figure scripts; OJS demos with a seeded RNG.

### Changed, and why
| Reference | Here | Why |
|---|---|---|
| Harness, loaders and interfaces inlined into each notebook (about 310 lines) | **`prl/` package**, installed from a pinned git ref | Brief §3, and labs share environments and data |
| Raw `.ipynb` is the source | **jupytext percent `.py` in `labs/src/`**, built one way into `notebooks/*.ipynb` with deterministic, role-based cell IDs | Reviewable diffs |
| Multi-clock schedule model with a 1,156-line generator | **One fixed daily clock** with per-day `modules: [A, B, C]`. Only the capstone has a custom split. | All five days share one shape |
| IPython hooks plus a patched `print` attribute checkpoint results | Explicit `lab.check(n, fn, *objs)` | The `.py` source also runs under pytest |
| "Fails on stub" is checked only at the first checkpoint | **Mutants** (`labs/mutants/mNN.py`): the planted bug plus one plausible wrong variant per exercise. Every checkpoint must reject every mutant. | Otherwise "fails on stub" proves little |
| QUICK means "running on a CPU" | QUICK means `PRL_QUICK=1`, or automatic only when a GPU-designed lab finds no GPU. It changes numbers only (`prl.settings(live=…, quick=…)`). A QUICK seed is never overlaid on a full-budget band. | CPU-designed labs must run in full on their designed runtime (OPEN_QUESTIONS A1) |
| Many offline flags | `PRL_QUICK`, `PRL_WORKED`, `PRL_TEST_DOUBLES`, `PRL_PLATFORM`. Test doubles are explicit and recorded. | One convention |
| Readiness ranked with 8 keys, plus prose | **One matrix** (lab × evidence column) with a details block per lab | The same honesty, far less code |
| `unittest` | **pytest**, including seeded statistical tests | Brief §3 |
| References written as Markdown | **`references.bib` + citeproc**, with verification in `annote` | One source, auditable |
| `**In this section:**` and `**Objective:**` mixed; only alt text is linted | **Page-template lint** plus a **notebook rhythm lint** | Brief template items 2–3 |
| Desktop only; color-only schedule bars; Google Fonts and CDN requests | **Mobile-first**, two narrow schedule tables, self-hosted fonts (with Greek coverage) and KaTeX, `_brand.yml` tokens, automated WCAG 2.2 AA checks (§8) | Brief §5 site audit; the UI review |
| Colab only | Colab, local uv, and AWS | Brief §3 |
| `AGENTS.md` personas and `briefs/` | `CLAUDE.md` plus `.claude/agents/`. Lab specs live in this file. | Simpler |
| Mirror workflow, migration script, "restated verbatim" contracts | Dropped | Not needed |

**Lesson from the reference's RLHF lab.** Its reward-hacking result was GPU-only and seen with a single seed. Ours is **exact and shows on a CPU**: the gold-versus-proxy frontier over β in Module 12, Part A.

---

## 2. Audience, format, daily clock

**Audience.**
- ML engineers, data scientists and graduate students who know supervised learning, Python, NumPy and PyTorch (they have trained an `nn.Module`).
- They know probability, basic linear algebra and calculus.
- No prior RL.

**Entry check** (on "Start here", 15 minutes). Three questions in each of four areas. Each question is chosen to head off a trap later in the week:

| Area | Questions |
|---|---|
| Probability | E_q[(p/q)f]; the variance of a sample mean; a Beta–Bernoulli update |
| Linear algebra and calculus | When is (I−A)x = b solvable; ∇ log softmax; the chain rule |
| PyTorch | `detach` for targets; the shape of a `gather`; the order of `zero_grad`/`backward`/`step` |
| Python and NumPy | Broadcasting `P[s,a,:] @ V`; fancy indexing; `default_rng` |

Missing 2 or more in an area links to specific free reading. The links are re-verified by the academic director.

**What this workshop is not** (on the home page, one line each):
- deep-learning basics
- heavy theory: PAC bounds, regret proofs beyond UCB intuition, stochastic approximation, Bellman rank
- robotics simulator engineering
- distributed RL infrastructure
- continuous-time control (LQR, HJB)
- MCTS and AlphaZero
- pretraining and SFT mechanics

All of these go to an annotated "Where to go next" page.

**Daily clock** (all five days):

| Time | Block |
|---|---|
| 08:00–09:00 | Day 1 only: optional setup clinic |
| 09:00–09:15 | Warm-up (Day 1: welcome and setup check) |
| 09:15–11:15 | **Module A**: briefing 40 · lab 70 · debrief 10 |
| 11:15–11:30 | Break |
| 11:30–12:10 | **Module B** briefing 40. Its last 2 minutes are "open the notebook and run the setup cell". |
| 12:10–13:10 | Lunch |
| 13:10–14:30 | **Module B** lab 70 · debrief 10. The lab's first row is a 3-minute reconnect: rerun setup, and recap where the briefing stopped. |
| 14:30–14:45 | Break |
| 14:45–16:45 | **Module C**: briefing 40 · lab 70 · debrief 10. On Day 5 this slot is the capstone: kickoff 10 · work 80 · share-outs 30. |
| 16:45–17:00 | Wrap-up: running results table; "what is still broken" |

**Budgets** (all estimates until measured):

| Block | Budget |
|---|---|
| Briefing | 40 minutes = at most 30 of exposition + at least 8 of live activity (+ 2 of setup in B slots). It is planned in the page's `live:` front matter, with block kinds `section`, `activity` and `setup`, and tested. |
| Lab, A and C slots | Core at most **52** |
| Lab, B slots | Core at most **51 + 3** (reconnect) |
| Lab, everything else | The rest of the 70 minutes is slack plus one optional stretch section. Nothing later depends on a stretch. |
| Exercise | At least 5 minutes, including the rhythm |
| Live compute | At most 10–12 minutes per lab. No cell over 3 minutes. Progress printed about every 10 seconds (at most 20 lines per cell). No background training. |

---

## 3. Rules every lab follows

1. **Predict → Run → Explain → Check.** Each exercise contains, in order:
   - a *Predict* cell (`predict_answer` is recorded for the pace sheet);
   - a `# TODO` stub, always a top-level `def` or `class`;
   - a two-level fold: Hint, then Solution;
   - *Run*;
   - an *Explain* prompt that cites an equation label or a Gotcha (`explain_ref`);
   - a checkpoint.

   The build lints the order and the references. Every exercise has a checkpoint that fails on its stub and on every mutant. If an exercise writes two functions, it uses sub-checkpoints (3a/3b) so a failure names the broken piece.
2. **Scaffolding fades** (`scaffold` field):
   - **S3** on Days 1–2: skeleton, shapes and helpers given.
   - **S2** on Day 3: signature, docstring and a Hint.
   - **S1** on Days 4–5: for losses and estimators, only the signature and the equation reference.

   Plumbing (runners, plotting, environment loops) is provided throughout.
3. **From scratch first** (brief rule 7). Participants write the core algorithm.
   - Algorithms a later lab needs arrive as "the reference version of your Lab N code", through `# %% include labs/shared/*.py`. A test checks each one equals its origin solution.
   - `prl/` contains no algorithms.
4. **Diagnostics.** Each lab has a Diagnostics cell: what to plot, and what healthy and broken runs look like. Each lab has one **planted bug**, and the diagnostic that exposes it.
5. **Evidence and comparisons** (brief rule 6):
   - Fewer than 5 seeds: show every seed. Overlapping ranges are a **tie**. No CI.
   - 5 or more seeds: IQM with bootstrap CIs. With one task the bootstrap is ordinary, not stratified.
   - One seed: "single seed, not a ranking".
   - Every comparison states its budget: steps, seeds, wall-clock and hardware.
6. **Bands are never pass/fail.** A correct new seed falls outside a 5-seed min–max band with probability 2/6. Checkpoints test thresholds from `scripts/threshold_protocol.py`. Where a seed lands relative to the band is reported as a fact. A participant computes the IQM and the bootstrap from a record's per-seed array themselves, with and without their own seed.
7. **Room pooling.** `SEED = seat number`, set in the setup cell and recorded. The lab ends by printing a one-line "room code". The debrief pools 15–25 seeds from the room into a room IQM shown beside the recorded band. The room runs the multi-seed experiment together.
8. **Termination vs truncation.** Every CartPole lab bootstraps on `truncated` and not on `terminated`. With Gymnasium's `SAME_STEP` autoreset, the bootstrap uses `info["final_obs"]`. Checkpoints test this.
9. **Framing.** Each day frames at least one real-shaped problem. Framing exercises are checked through artifacts: state counts, rewards on hand-worked cases, `validate()`. The prose framing is discussed in the debrief.
10. **No cross-day file dependencies.** Labs load committed datasets that ship inside `prl`. "Use your own log" is optional.

---

## 4. Curriculum

Each module lists:
- **Where we are**: what the previous module left broken.
- **Question**: the one question this module answers.
- **Objectives**: measurable, each exercised by a named checkpoint.
- **Briefing**: as section (minutes, activity). Totals are exposition + activity [+ setup]. Collapsed and `{.reference}` material is noted.
- **Lab**: an exercise table. Minutes include the rhythm.
- **Compare**, **Stretch** and **Scaffold**.

Day questions are the brief's, verbatim.

### Module 0 · Setup and "hello RL" (pre-work, 20 min, Colab CPU, level 100)

Objectives:
- Run the setup notebook on at least one run path, and read its versions and runtime.
- Plot returns from a random CartPole agent, with seed bands.
- Submit a run record.

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 0 | Runtime check | — | `detect()` fields present; versions printed | 5 |
| 1 | Random agent | `run_episode` | Returns list length and type | 5 |
| 2 | Seed bands | `returns_by_seed` | Shape, and seeds distinct | 5 |
| 3 | Record | — | The record validates | 5 |

---

### Day 1 · From the Bellman equation to learning from experience

*What does "optimal" mean, and how do we find it, first with a model and then without one?* The framing problem is **inventory control**.

#### M1 · MDPs, returns and the Bellman equation (A, level 200, Colab CPU, S3)

- **Where we are.** The start of the week.
- **Question.** What quantity are we estimating, and why can we compute it exactly when the model is known?

**Objectives:**
1. Encode a finite MDP as `P[s,a,s']` and `R[s,a]`, and validate it.
2. Frame inventory control as an MDP (state, action, reward, γ) and justify the choices.
3. Evaluate a policy exactly, and iteratively. Bound the iteration count: k ≥ log(‖V₀−V^π‖∞/ε)/log(1/γ). Stopping when ‖V_{k+1}−V_k‖∞ < tol guarantees ‖V_{k+1}−V^π‖∞ ≤ γ·tol/(1−γ).
4. Find a wrong evaluator from its Bellman residual.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Inventory: what must you track? | 6 (predict 3) |
| 2 | Returns, γ, MDP as arrays | 6 |
| 3 | Bellman expectation equation (S&B Eq. 3.14); 2-state example by hand | 10 (chk 3) |
| 4 | T^π is a γ-contraction in ‖·‖∞ (one line: ‖γP^π(V−V′)‖∞ ≤ γ‖V−V′‖∞); iteration bound; slider demo | 12 (demo 4) |
| 5 | Optimality equation stated (S&B Eq. 3.19–3.20); bridge | 6 |

Total: 30 + 10. **Collapsed:** uniqueness via Banach; the stopping-rule proof.

**Lab (core 52):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | Discounted return | `discounted_return` | Exact on 3 fixtures | 5 |
| 2 | Tiny cliff gridworld as arrays (index and move helpers provided) | `gridworld_arrays` (transition rule: cliff → start with −100; goal absorbs) | Equals the fixture; `validate()` | 8 |
| 3a | Framing card for inventory | `inventory_spec` | State count; reward on 3 hand-worked cases | 5 |
| 3b | Inventory MDP (demand pmf provided) | `inventory_arrays` | `validate()`; equals the fixture | 12 |
| 4 | Exact evaluation | `evaluate_exact` | Equals the fixture V | 5 |
| 5 | Iterative evaluation (γ sweep and plot provided) | `evaluate_iterative → (V, k)` | k within the bound; error ≤ γ·tol/(1−γ) | 9 |
| 6 | **Planted bug**: evaluator uses `P_pi.T` | Diagnose and fix | Bellman residual < tol | 8 |

- **Compare:** exact fixtures.
- **Stretch:** T^π can expand in the unweighted 2-norm. Example: two states both moving to state 1 with γ = 0.9 give ‖γP‖₂ ≈ 1.27. It contracts in the stationary-weighted norm, which is why on-policy linear TD converges (M7).

#### M2 · Planning: value iteration, policy iteration and model error (B, level 200, Colab CPU, S3)

- **Where we are.** M1 evaluates a *given* policy.
- **Question.** Given a model, how do we find the best policy, and what happens when the model is wrong?

**Objectives:**
1. Implement value iteration (VI) and policy iteration (PI), and compare their iteration counts.
2. Extract a greedy policy. Check that if ‖Q−Q*‖∞ ≤ ε, the greedy policy loses at most 2γε/(1−γ). The lemma number will be cited only after it has been opened.
3. Estimate a model from samples. Measure the true loss of the policy planned in it as the data shrinks, over 5 seeds.
4. State the simulation lemma (AJKS V3 Lemma 2.2, in identity form). With rewards in [0,1] and ε_P = max‖P(·|s,a)−P̂(·|s,a)‖₁, every policy's value error is at most γε_P/(1−γ)², and the planned policy loses at most 2γε_P/(1−γ)².

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | VI as repeated optimality backups | 8 |
| 2 | Greedy extraction; loss bound stated | 7 (chk 2) |
| 3 | PI; improvement argument in 3 lines | 8 (chk 2) |
| 4 | Counts → P̂; simulation lemma and horizon factor | 11 (predict 4) |
| 5 | Bridge | 4 |
| — | Setup | 2 |

Total: 30 + 8 + 2. **Collapsed:** modified policy iteration; asynchronous VI; the lemma's proof.

**Lab (core 3 + 49):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 0 | Reconnect, rerun setup | — | Setup check | 3 |
| 1 | VI and greedy extraction | `value_iteration` (1a), `greedy` (1b) | V\* and π\* equal the fixture; greedy-loss bound holds on a perturbed Q | 12 |
| 2 | Policy iteration | `policy_iteration` | Same π\*; fewer iterations than VI | 9 |
| 3 | Estimate the model. Unvisited (s,a) becomes a self-loop with R̂ = 0, which is also the Predict. | `estimate_model` | Exact on a fixed sample | 8 |
| 4 | Plan in P̂, evaluate in P (runner provided) | `plan_and_evaluate` | At the smallest N, mean loss exceeds the mean at the largest N by more than the seed spread | 12 |
| 5 | **Planted bug**: VI stops on mean \|ΔV\|, not max | Diagnose from the logged max\|ΔV\| per sweep | Residual of the returned V < tol | 8 |

- **Compare:** exact fixtures.
- **Stretch:** modified policy iteration.

#### M3 · Model-free prediction and control (C, level 200, Colab CPU, S3)

- **Where we are.** M2 needs P and R.
- **Question.** Without a model, how do we learn values and policies from experience?

**Objectives:**
1. Measure the RMS error, squared bias and variance of Monte Carlo (MC) and TD(0) estimates across 10 seeds, against the exact V from M1.
2. Implement SARSA and Q-learning, and explain the cliff contrast under a constant ε: Q-learning's greedy path is optimal, while SARSA has the higher online return.
3. Diagnose a terminal-bootstrap bug from a learning curve and the Q-values.
4. Compute n-step TD targets, and place them on the dial between MC and TD.

**Briefing.** The MC target G_t is unbiased for v_π(S_t). The TD target R + γV(S′) has bias γ·E[V(S′) − v_π(S′)], and usually, though not always, lower variance. First-visit MC is unbiased; every-visit MC is biased but consistent.

| § | Topic | Min |
|---|---|---|
| 1 | No model | 3 |
| 2 | MC | 6 |
| 3 | TD(0); bias and variance | 10 (chk 3) |
| 4 | SARSA vs Q-learning; ε-greedy | 11 (predict 4) |
| 5 | Terminated vs truncated | 5 (chk 3) |
| 6 | The n-step dial, one figure | 3 |
| 7 | Bridge | 2 |

Total: 30 + 10. **Reference:** the n-step derivation; TD(λ).

**Lab (core 49):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | First-visit MC prediction | `mc_prediction` | RMS below the protocol threshold | 9 |
| 2 | TD(0); bias² and variance over 10 seeds | `td0_prediction` | Threshold; decomposition reported | 10 |
| 3 | n-step TD target | `n_step_target` | Exact on a fixed trajectory | 5 |
| 4 | SARSA (ε-greedy provided) | `sarsa` | Greedy path is the safe path | 8 |
| 5 | Q-learning and the cliff contrast | `q_learning` | Greedy path optimal; online return below SARSA's (protocol) | 9 |
| 6 | **Planted bug**: bootstraps through the terminal (bug made visible by nonzero/optimistic init) | Diagnose and fix | Q(goal, ·) = 0 under optimistic init | 8 |

- **Compare:** the exact V and the optimal policy.
- **Stretch:** an n-step sweep (RMS error over n and α).

---

### Day 2 · Exploration and bandits

*How should an agent gather data when every action costs something?* The framing problem is a **recommender**.

#### M4 · Bandits from first principles (A, level 200, Colab CPU, S3)

- **Where we are.** M3's ε-greedy explores blindly.
- **Question.** Each pull earns reward and also reveals information. How should we choose?

**Objectives:**
1. Define pseudo-regret from the known arm means, and measure it with seed bands.
2. Derive the UCB1 bonus √(2 ln t / n_j) from Hoeffding's inequality, which assumes rewards in [0,1] (Auer et al. 2002, §2, Fig. 1). Implement it.
3. Implement Beta–Bernoulli Thompson sampling, which assumes rewards in {0,1}, from Bayes' rule.
4. Show how ε-greedy and UCB1 depend on ε and on the reward scale, and explain why through the assumptions above.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Regret | 5 |
| 2 | Why greedy fails | 6 (predict 3) |
| 3 | Hoeffding → UCB1 | 11 (demo 3) |
| 4 | TS from Bayes | 8 (chk 2) |
| 5 | Scale and ε | 5 (chk 2) |
| 6 | Bridge to context | 5 |

Total: 30 + 10. Sources: Lattimore & Szepesvári Ch. 5, 7, 36; Russo et al. 2018.

**Lab (core 51):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | Bandit loop, ε-greedy, pseudo-regret | `run_bandit` (1a), `eps_greedy` (1b) | Exact on a seeded instance; greedy is linear on the trap instance | 11 |
| 2 | UCB1 | `ucb1_index` | Bonus exact; counts on a fixed seed | 10 |
| 3 | Thompson sampling | `thompson_step` | Posterior update exact; regret below threshold | 10 |
| 4 | 3 methods × 20 seeds; the tie rule | Verdict sentences | Match `prl.evaluate.compare` | 7 |
| 5 | Reward ×10 and ε sweep (runner provided) | Predicted direction per method | Checked against the computed direction | 6 |
| 6 | **Planted bug**: UCB counts never incremented | Diagnose from the arm-pull histogram | Regret below threshold | 7 |

- **Compare:** pseudo-regret from the known means.
- **Stretch:** Gaussian Thompson sampling.

#### M5 · Contextual bandits for recommendation (B, level 200, Colab CPU, S3)

- **Where we are.** M4 learns nothing across users.
- **Question.** How do we explore when the best action depends on the context, and log data we can learn from later?

**Objectives:**
1. Frame a recommender as a contextual bandit, and state what is lost compared with full RL.
2. Implement incremental ridge regression, LinUCB (Li et al. 2010, Alg. 1, disjoint) and linear Thompson sampling (Agrawal & Goyal 2013).
3. Compare them with predict-then-greedy under misspecification and drift, over 5 seeds.
4. Log data with correct propensities from a stochastic behavior policy, for Day 5.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Frame RecSim | 7 (predict 2) |
| 2 | Incremental ridge | 7 |
| 3 | Ellipsoid → LinUCB width | 8 (demo 3) |
| 4 | LinTS | 5 (chk 2) |
| 5 | Misspecification and drift | 4 |
| 6 | Propensities; why logging must be stochastic | 7 (chk 2) |
| — | Setup | 2 |

Total: 29 + 9 + 2. **Collapsed:** L&S Thm 20.5; delayed feedback.

**Lab (core 3 + 49):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 0 | Reconnect | — | Setup check | 3 |
| 1 | Frame RecSim | `reward_fn`, framing card | Reward on 3 hand-worked cases | 5 |
| 2 | Incremental ridge | `Ridge.update`, `Ridge.predict` | Equals the closed form | 8 |
| 3 | LinUCB | `linucb_scores` | Widths exact; a unit x seen n times has width α/√(1+n) | 10 |
| 4 | Linear TS | `lints_sample` | Sample moments within tolerance | 7 |
| 5 | Three regimes (runner provided) | Verdict sentences | Match `compare` | 8 |
| 6 | Logging policy (ε-greedy over LinUCB) | `log_policy` | Propensities sum to 1; mean importance weight π/π_b ≈ 1 | 6 |
| 7 | **Planted bug**: logs the greedy action's propensity | Diagnose and fix | Mean weight's CI covers 1 | 5 |

- **Compare:** pseudo-regret from the simulator's known means.
- **Stretch:**
  - Parity with MABWiser 2.7.4's LinUCB on the same stream: compare its `A_inv` and `Xty` with your ridge statistics. The technical review verified that it installs and runs on Python 3.13 with NumPy 2.
  - Delayed (batched) feedback.

#### M6 · Exploration in MDPs (C, level 300, Colab CPU, S3)

- **Where we are.** Bandit bonuses look one step ahead.
- **Question.** How do we explore when the reward is many steps away?

**Objectives:**
1. Compute the expected hitting time of RiverSwim's right end under a *fixed* policy (uniform random) by first-step analysis. Show it grows like (p_L/p_R)^n with chain length n. Chain length is our generalization; the original has 6 states.
2. Implement optimistic initialization, and MBIE-EB (Strehl & Littman 2008, §3.2, Eq. 8: a bonus β/√N(s,a) inside the backup of the estimated model), using the reference `estimate_model` and VI from M2.
3. Measure state visitation and cumulative regret against the DP-optimal policy over 10 seeds (S&L report cumulative reward over the first 5000 steps).
4. Explain where count bonuses stop working, and what pseudo-counts and RND attempt (Bellemare et al. 2016; Burda et al. 2019).

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | RiverSwim | 2 |
| 2 | First-step hitting time | 10 (predict 3) |
| 3 | Optimistic initialization | 5 (chk 2) |
| 4 | The bonus belongs in the backup (MBIE-EB) | 8 (chk 2) |
| 5 | Visitation and regret | 5 (demo 3) |
| 6 | Where counts stop | 5 |
| 7 | Bridge | 5 |

Total: 30 + 10. **Collapsed:** RND; pseudo-counts; PSRL.

**Lab (core 51):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | Uniform-random policy on RiverSwim | `policy_transition(P, π)` | Exact | 5 |
| 2 | Hitting time against chain length | `hitting_times` | Linear solve = simulation within tolerance; growth ratio | 10 |
| 3 | ε-greedy Q-learning (reference) and visitation | `visitation` | Right end rarely visited (protocol) | 7 |
| 4 | Optimistic initialization | `optimistic_q0` | Right end reached in ≥ k of 10 seeds | 6 |
| 5 | MBIE-EB backup | `mbie_eb_backup` | Exact on fixed counts | 10 |
| 6 | Cumulative regret, 10 seeds (runner provided) | Verdict | Matches `compare` | 6 |
| 7 | **Planted bug**: bonus applied only at action selection | Diagnose from visitation | Visitation recovers | 7 |

- **Compare:** the DP optimum.
- **Stretch:** PSRL (Osband et al. 2013).
- *RiverSwim transition probabilities are ours and stated on the page. The S&L figure is an image and has not yet been read (OPEN_QUESTIONS B11).*

---

### Day 3 · Value-based deep RL

*What changes when the value function is a neural network?*

Framing:
- M7: **inventory at scale**.
- M9: "a teammate's DQN stopped learning".

#### M7 · Function approximation and why it can diverge (A, level 300, Colab CPU, S2)

- **Where we are.** Tables do not scale.
- **Question.** What breaks when a parameterized function replaces the table?

**Objectives:**
1. Derive semi-gradient TD(0) for linear values. Show it is not the gradient of any objective, because A = ΦᵀD(I−γP)Φ is not symmetric.
2. Reproduce Baird's divergence with expected (DP-style) semi-gradient updates (S&B §11.2, Fig. 11.1). Show which leg of the deadly triad (§11.3), when removed, restores convergence. With on-policy weighting on state 7, A = (1−γ)φ₇φ₇ᵀ ⪰ 0.
3. Frame inventory at scale (3 products, capacity 50 each: 51³ = 132,651 states), and justify a feature representation.
4. Train linear semi-gradient Q-learning on CartPole, and report the weight norm and returns.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Count the states | 6 (predict 2) |
| 2 | Linear V | 5 |
| 3 | Semi-gradient derivation | 9 (chk 2) |
| 4 | Baird, live | 10 (demo 4) |
| 5 | Remove one leg | 5 (predict 2) |
| 6 | Feature scaling; RBF | 5 |

Total: 30 + 10. **Collapsed:** the asymmetric-Jacobian argument in full; TDC.

**Gotcha:** "A bigger network will fix divergence." Baird's rewards are all 0, so w = 0 is representable, and it still diverges.

**Lab (core 49):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | Frame inventory at scale | `scale_spec` | State count 132,651; feature dimension | 5 |
| 2 | Semi-gradient TD(0) update | `semi_gradient_td0` | Equals the hand-computed value | 7 |
| 3 | Not a gradient | `td_matrix` | Exact; A ≠ Aᵀ | 6 |
| 4 | Baird with expected updates | `baird_expected_update` | Exact sequence; ‖w‖ grows | 8 |
| 5 | Remove one leg: on-policy weighting, MC targets, tabular features | Three variants | Which converge (exact) | 9 |
| 6 | Linear Q on CartPole (RBF provided); one live seed | `linear_q_update` | ≥ 3× random (protocol) | 8 |
| 7 | **Planted bug**: unnormalized features | Diagnose from the weight-norm log; fix the featurizer | Norm bounded; ≥ 3× random | 6 |

- **Compare:** a recorded band.
- **Stretch:** TDC on Baird. It drives the projected Bellman error to 0, slowly; there is no promise that w converges fast.

#### M8 · DQN from scratch (B, level 300, Colab CPU, S2)

- **Where we are.** Linear Q is limited, and naive neural Q is unstable.
- **Question.** Which changes make neural Q-learning stable, and what bias remains?

**Objectives:**
1. Implement a replay buffer, and explain the two problems it fixes: correlated samples and data reuse.
2. Implement DQN with a target network and Huber loss, masking only on `terminated`.
   - *Wording for the page:* Mnih et al. 2015 clip the TD error to [−1, 1] in the update. That is exactly the gradient of the Huber loss with threshold 1. The gradients are equal; the loss values are not. Clipping the *loss* is not equivalent.
3. Derive E[max_a Q̂] ≥ max_a E[Q̂] (Jensen), and state van Hasselt et al. 2016 Thm 1 with its conditions: equal true values, zero-mean errors. Implement Double DQN. Measure Q(s₀) against the discounted MC return of the greedy policy.
4. Place your single live seed against recorded 5-seed bands of the reference DQN and SB3's DQN at the same budget. Call the reference-vs-SB3 comparison a tie or a ranking from the bands.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Linear → neural | 3 |
| 2 | Replay | 6 (chk 2) |
| 3 | Target network and Huber | 8 |
| 4 | Masks | 5 (chk 2) |
| 5 | Max bias; Double DQN | 11 (demo 3) |
| 6 | Equal-budget bands | 5 (predict 2) |
| — | Setup | 2 |

Total: 29 + 9 + 2. **Reference:** prioritized replay (PER), dueling networks.

**Gotcha:** bootstrapping through truncation at γ = 0.99 makes an exact Q exceed a 500-step MC return by about γ⁵⁰⁰/(1−γ) ≈ 0.66.

**Lab (core 3 + 51):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 0 | Reconnect | — | Setup check | 3 |
| 1 | Replay buffer (preallocated arrays) | `ReplayBuffer.add`, `.sample` | Shapes; uniform (χ²) | 8 |
| 2 | **Planted bug**: "recent-only" buffer | Diagnose (multiple choice) and fix | The fixed buffer passes χ² | 5 |
| 3 | Masked TD target (3a) and Huber loss (3b) | `td_target`, `huber` | Exact on fixed tensors; truncation bootstrapped | 9 |
| 4 | Training loop with target sync; one live seed (30–50k steps, `train_freq=4`) | `train_dqn` | ≥ 3× random (protocol) | 12 |
| 5 | Double DQN target | `double_dqn_target` | Exact | 5 |
| 6 | Overestimation: Q(s₀) vs discounted MC return (your seed over recorded DQN and DDQN bands) | `mc_return` | Exact on fixed rewards; the gap is reported | 7 |
| 7 | SB3 at equal budget: your seed over recorded bands | Budget sentence | Generated from the records | 5 |

- **Compare:** SB3 2.9 DQN (recorded bands).
- **Stretch:** `LunarLander-v3`, installing only `box2d`.

#### M9 · Debugging and evaluating RL (C, level 300, Colab CPU, S2)

- **Where we are.** M8 works on one seed.
- **Question.** How do we know an agent works, and find out why when it doesn't?

**Objectives:**
1. Diagnose four DQN failures from their logged metrics, by writing the signal that exposes each one.
2. Confirm a diagnosis with probe environments (Jones 2021: constant reward; observation-dependent reward) and the checklist: a trivial environment, overfitting one transition, terminal handling, reward scale.
3. Report a comparison with IQM and bootstrap CIs (Agarwal et al. 2021, §4.1 and §4.3), a budget sentence, and ties called.
4. Interpret a hyperparameter sensitivity sweep.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | "A teammate's DQN stopped" | 3 |
| 2 | What to log | 8 (demo 3) |
| 3 | The checklist and probes | 8 (chk 2) |
| 4 | Seeds | 6 (predict 3) |
| 5 | IQM and the bootstrap | 8 (chk 2) |
| 6 | Budget sentences; ties | 7 |

Total: 30 + 10.

**Lab (core 52):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | Healthy logs: record your predictions | Multiple-choice answers | Recorded | 5 |
| 2–5 | Four **recorded** broken runs (no terminal mask; target never synced; learning rate ×100; wrong `gather` index): write the signal, then diagnose | `q_drift`, `target_lag`, `td_error_scale`, `gather_check` | Value on the record; diagnosis (multiple choice) | 4 × 6 |
| 6 | Two probe environments, run against the buggy agents | `ConstantRewardEnv`, `ObsRewardEnv` | Q = 1 (and 100 without the mask, at γ = 0.99) | 10 |
| 7 | Recorded sweep (5 configs × 10 seeds): IQM, bootstrap CI, budget sentence | `iqm`, `bootstrap_ci` | Equal `prl.evaluate` | 8 |
| 8 | One live config × 1 seed | — | ≥ protocol threshold; position against the band reported | 5 |

- **Compare:** recorded bands.
- **Stretch:** Jones's probe 3 (two steps, discounting).

---

### Day 4 · Policy optimization

*How do we improve a policy directly, and how far can we safely step?* The framing problem is **pricing**, done as a pair framing card in M10's briefing.

The day's storyline is three ways to bound an update:
- the baseline bounds variance (M10);
- the clip bounds the ratio (M11);
- the KL to π_ref bounds the distance (M12).

#### M10 · Policy gradients (A, level 300, Colab CPU, S1)

- **Where we are.** Value methods need a max over actions, and fail on continuous actions.
- **Question.** Can we improve the policy directly?

**Objectives:**
1. Derive the policy gradient theorem with the log-derivative trick (Sutton et al. 1999, Thm 1; S&B §13.2), and implement REINFORCE (Williams 1992).
   - *Wording:* implementations usually drop γ^t, which is then not the exact gradient of the discounted objective.
2. Show that an action-independent baseline b(s) leaves the expected gradient unchanged, because E_a[b(s)∇log π] = 0. Measure how it changes the variance. Fitting b on the same batch, or normalizing advantages, adds a small bias.
3. Implement GAE (Schulman et al. 2016, Eq. 16; λ = 0 is Eq. 17, λ = 1 is Eq. 18) with `terminated`/`truncated` handled. Measure its bias and variance across λ on the gridworld, with exact A^π and a deliberately wrong V.
4. Implement an actor-critic loss, and compare it with REINFORCE at a fixed budget.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Pricing framing card (pairs) | 6 (activity 3) |
| 2 | Log-derivative trick on a bandit; theorem stated | 10 (chk 2) |
| 3 | REINFORCE | 5 |
| 4 | Baseline: no bias | 7 (demo 3) |
| 5 | Actor-critic; the GAE λ dial | 8 (chk 2) |
| 6 | Bridge | 4 |

Total: 30 + 10. **Collapsed:** the full trajectory proof; GAE telescoping.

**Lab (core 51):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | Score-function estimate vs exact gradient (tiny bandit; idea from nlp-llms Lab 9, attributed) | `score_function_grad` | Within MC error | 6 |
| 2 | REINFORCE with reward-to-go | `rewards_to_go` (2a), `reinforce_loss` (2b) | Exact on fixed tensors; ≥ 3× random within 150 episodes | 12 |
| 3 | Learned baseline; gradient variance over 50 batches | `baseline_loss` | Mean unchanged within MC error; variance ratio < 1 | 10 |
| 4 | GAE, plus a λ sweep on gridworld rollouts with exact A^π and V = 0.8·V^π | `gae` | Exact on a fixed batch of 2 episodes; Var(λ=1) > Var(λ=0); bias(λ=1) ≈ 0 | 10 |
| 5 | **Planted bug**: GAE not reset at episode boundaries | Diagnose from advantages plotted by step | Boundary fixture | 5 |
| 6 | Actor-critic loss (the loop is Ex 2's reference) | `actor_critic_loss` | Exact on fixed tensors; ≥ 3× random | 8 |

- **Compare:** recorded bands.
- **Stretch:** a Gaussian policy on Pendulum. Expect slow progress at this budget; M11 asks why.

#### M11 · PPO from scratch (B, level 300, Colab CPU, S1)

- **Where we are.** M10's updates can overshoot.
- **Question.** How far can one batch move the policy before performance drops?

**Objectives:**
1. Show that a large policy-gradient step can collapse performance, and measure the step's KL.
2. Implement PPO's clipped surrogate (Schulman et al. 2017, Eq. 7) with value and entropy terms (Eq. 9), and the update loop over vectorized environments.
   - *Wording:* L^CLIP removes the incentive to push ρ_t outside [1−ε_clip, 1+ε_clip]. It does not bound the ratio or the KL.
3. Ablate clipping live over 3 seeds. Read two implementation details from a recorded ablation: advantage normalization (#7) and LR annealing (#4) (Huang et al. 2022). State seeds and budget.
4. Place your seed against recorded SB3-PPO and reference-PPO bands. CleanRL's `ppo.py` is annotated reading.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Step size | 3 |
| 2 | Big-step collapse | 8 (demo 3) |
| 3 | Ratio surrogate | 6 |
| 4 | The clip case table | 9 (chk 3) |
| 5 | Value and entropy | 4 |
| 6 | Implementation details; healthy clipfrac and KL | 8 (predict 2) |
| — | Setup | 2 |

Total: 30 + 8 + 2. **Collapsed:** TRPO (Schulman et al. 2015, Thm 1); the full Huang et al. list.

**Lab (core 3 + 51):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 0 | Reconnect | — | Setup check | 3 |
| 1 | Big steps hurt (step sweep provided) | `categorical_kl` | Exact; return falls as KL grows (reported) | 6 |
| 2 | Boundary masks, and truncation bootstrap from `final_obs` (rollout collector provided). The collector uses `SyncVectorEnv(..., autoreset_mode=AutoresetMode.SAME_STEP)`. `info["final_obs"]` is an object array, so stack with `np.stack(info["final_obs"][info["_final_obs"]])`. | `boundary_masks` | Exact | 5 |
| 3 | Clipped surrogate (3a), value loss (3b), entropy (3c) | `ppo_loss` | Exact on fixed tensors | 10 |
| 4 | Update loop: epochs × minibatches, advantage normalization, approx-KL and clipfrac logging | `ppo_update` | Logs in range on a fixed rollout | 10 |
| 5 | **Planted bug**: old log-probs recomputed from current parameters at every minibatch | Diagnose (clipfrac = 0, KL = 0) and fix | clipfrac > 0 and KL > 0 | 5 |
| 6 | Clip on vs off, 3 seeds × about 20k stress steps, one cell per seed | — | Runs complete; verdict or tie via `compare` (reported) | 10 |
| 7 | Your seed over recorded SB3 and reference bands | Budget sentence | Generated | 5 |

- **Compare:** SB3 2.9 PPO (recorded bands).
- **Stretch:** SAC on Pendulum (Haarnoja et al. 2018a; automatic temperature from 2018b, §5).

#### M12 · RL for language models: preferences, RLHF, DPO, GRPO (C, level 400, Colab CPU core, T4 stretch, S1)

- **Where we are.** M11 bounds a step with the clip.
- **Question.** What if the reward is learned from comparisons, and how far should we trust it?
- The KL to π_ref is Day 4's third way of bounding a step.

**Objectives:**
1. Fit a Bradley–Terry reward model from comparisons, and read its accuracy against the label-noise ceiling.
2. Derive π\*_β ∝ π_ref·exp(r/β) (Rafailov et al. 2023, Eq. 4) and the DPO loss (Eq. 7). Verify on an exact toy that DPO reaches π\*.
   - *Conditions to state:* BT-generated preferences with finite reward gaps; exact preference probabilities, or infinite data; a connected comparison graph; π_ref with full support.
   - With hard, deterministic labels, the BT maximum-likelihood estimate diverges, and DPO pushes π(y_l) → 0 at any β (a Gotcha; the link to Azar et al. 2024, §4.2, is still to be verified).
3. Compute GRPO's group-relative advantages Â_i = (r_i − mean r)/std r (Shao et al. 2024, §4.1.2). Explain what the group replaces: the critic.
   - Mean-centering alone gives exactly (1−1/K) × the true gradient. Dividing by the std adds bias.
   - GRPO puts the KL in the loss (Eq. 4 estimator), not in the reward.
4. Show reward hacking as the exact gold-vs-proxy reward curve against KL(π\*_β‖π_ref) as β varies. Then compute sequence log-probs and your DPO loss on SmolLM2-135M-Instruct, and check both against TRL's.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | An LM as a policy; sequence log-prob | 5 |
| 2 | Bradley–Terry | 6 (chk 2) |
| 3 | The KL closed form, derived; worked example with 3 responses at β = 1 and 0.25 (adapted from nlp-llms M10, attributed) | 9 (chk 3) |
| 4 | DPO: Z cancels | 6 |
| 5 | RLHF and GRPO in one picture (critic vs group) | 5 (predict 2) |
| 6 | Reward hacking over β (Stiennon et al. 2020, §4.3, Fig. 5) | 9 (demo 4) |

Total: 29 + 11. **Collapsed:** the InstructGPT pipeline (Ouyang et al. 2022, arXiv v1, §3.5, Eq. 2); GRPO's full objective; Gao et al. 2023 functional forms.

**Lab (core 52):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | Frame the LM as RL: prompt = state, response = action, BT score = reward, gold = evaluation. This cell also starts the model download. | `framing` | Exact | 5 |
| 2 | Bradley–Terry reward model on toy pairs | `bt_loss` | Accuracy vs ceiling | 8 |
| 3 | Closed form; gold and proxy reward vs KL(π\*_β‖π_ref) | `optimal_policy`, `kl` | Exact | 8 |
| 4 | DPO loss; the toy policy reaches π\* | `dpo_loss` | Total variation < tol | 9 |
| 5 | GRPO advantages; mean-centered estimate = (1−1/K) × exact gradient | `grpo_advantages` | Exact | 7 |
| 6 | Masked sequence log-probs on SmolLM2-135M (CPU) | `sequence_logprobs` | Equal transformers' own token cross-entropy (atol 2e-3 per sequence; no TRL needed) | 8 |
| 7 | Your DPO loss vs TRL's on 8 pairs. **Planted bug**: the reference *is* the policy, so the loss reads exactly ln 2 at every step while the weights still change. | `ref_logprobs` | Equals TRL's (atol 1e-5); bug fixed | 7 |

- **Compare:** TRL 1.15's DPO loss on the same batch.
- **TRL on CPU (technical review, measured on M1 Pro):**
  - `DPOTrainer.__init__` always patches in a Triton fused LM head (`trl/trainer/dpo_trainer.py:936`). It has no CPU path, so `compute_loss` fails on any CPU.
  - The notebook replaces `trl.trainer.utils._ChunkedLogProbFunction` with a 12-line pure-PyTorch stand-in. It is recorded as a shim in the run record (`shims: ["trl-fused-lm-head→torch"]`). It matched TRL's loss to 4.8e-7.
  - Always pass `bf16=False, fp16=False` on CPU. `DPOConfig` defaults to `bf16=True`, which made the reference forward 17× slower and changed the logits.
  - Cost: `compute_loss` on 8 pairs took 0.41 s. One forward+backward at 16 × 128 tokens took 5–7 s, at a peak RSS of 5.4 GB.
  - OPEN_QUESTIONS A11 asks whether to use this shim or a T4-recorded loss fixture.
- **Stretch (T4):**
  - A proxy reward model on the synthetic preference set, trained by TRL `RewardTrainer` in a provided cell.
  - Your DPO loop vs `DPOTrainer`: proxy score, gold reward, KL, and an inspected hacked sample.
  - The 3-seed evidence comes from records.
- **Without GPU:** the core is complete.

---

### Day 5 · Learning from logged data

*Can we evaluate and improve a policy we never deployed?* The framing problem is **shipping a recommender policy**.

#### M13 · Off-policy evaluation (A, level 300, Colab CPU, S1)

- **Where we are.** M5 logged data with propensities.
- **Question.** How good is a new policy, judged only from logs of an old one?

**Objectives:**
1. Derive the importance-sampling identity under overlap (π_b(a|x) > 0 wherever π(a|x) > 0). Implement IS and weighted IS (Swaminathan & Joachims 2015, Eq. 7), and the effective sample size (Σw)²/Σw².
2. Implement the direct method and doubly robust estimation (Dudík et al. 2011, Eq. 1).
   - *Wording:* bias = E_x[Δ·δ], where Δ is the reward model's error at the target action and δ = 1 − p/p̂. It is zero if the propensities are exact and positive wherever π acts, *or* if the reward model is exact for the actions π takes. This assumes the models are fit independently of the evaluation sample; the paper uses a deterministic π.
   - The ridge model is fit on a separate split.
3. Implement per-decision IS (Precup et al. 2000; S&B §5.9), Σ_t γ^t ρ_{0:t} r_t, and tabular FQE (Le et al. 2019, Alg. 3). Show how variance grows with the horizon (Liu et al. 2018, §2).
4. Measure bias, variance and MSE against ground truth as π moves away from π_b, with bootstrap CIs.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | What must the logs contain? | 4 (predict 2) |
| 2 | IS identity | 8 (chk 3) |
| 3 | WIS | 3 |
| 4 | DM → DR | 9 (chk 2) |
| 5 | IS vs PDIS over horizon | 7 (demo 3) |
| 6 | FQE as regression | 5 |
| 7 | Overlap; CIs | 4 |

Total: 30 + 10. IS and DR are derived; PDIS and FQE are stated. **Collapsed:** SNIPS asymptotics; sequential DR (Jiang & Li 2016, Eq. 10); marginalized IS.

**Lab (core 50):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 1 | IS (1a) and WIS (1b) on `recsim_logs` | `is_estimate`, `wis_estimate` | Exact on a fixture; IS unbiased over replicates | 9 |
| 2 | DM, with ridge fit on a separate split | `dm_estimate` | Fixture | 6 |
| 3 | DR | `dr_estimate` | Fixture; unbiased with true propensities | 6 |
| 4 | Overlap and ESS; mismatch sweep with bootstrap CIs (runner provided) | `ess` | Exact; MSE ordering reported | 8 |
| 5 | Trajectory IS vs PDIS over horizon (gridworld logs) | `pdis_estimate` | Exact on a fixture; variance grows | 8 |
| 6 | Tabular FQE | `fqe_tabular` | Within tolerance of the true value | 8 |
| 7 | **Planted bug**: logged propensities are for the wrong action | Diagnose (mean weight ≠ 1) and fix | Mean weight's CI covers 1 | 5 |

- **Compare:** simulator ground truth. Open Bandit Pipeline is unmaintained, and the page says so.
- **Stretch:** clipped IS.

#### M14 · Offline RL (B, level 300, Colab CPU, S1)

- **Where we are.** M13 can evaluate a policy from logs.
- **Question.** How do we *improve* a policy from a fixed log, without new data?

**Objectives:**
1. Show extrapolation error (Fujimoto et al. 2019, §3). Q at actions absent from the data is unconstrained, and the max in the target picks overestimates. So the Q-gap (mean max-Q over unseen actions minus Q at the data actions) grows while the greedy policy's true return does not.
2. Implement behavior cloning. Predict, before running, on which dataset it will match the best offline-RL method (expert-narrow) and on which it won't (mixed). Check against ground truth (Kumar et al. 2022, §4.2).
3. Implement the discrete CQL penalty (Kumar et al. 2020, §3.2, Eq. 4) and IQL's expectile loss and AWR weights (Kostrikov et al. 2022, §4.1–4.3, Eq. 5–7). Explain each from the failure it fixes.
   - *Link:* AWR's weights exp(β_IQL(Q−V)) are M12's closed form with π_ref = π_b and β_KL = 1/β_IQL.
4. Compare methods by ground-truth rollouts and by FQE (the reference from M13). Note that the deterministic expert dataset has no overlap, so only FQE applies. State seeds and budget.

**Briefing:**

| § | Topic | Min |
|---|---|---|
| 1 | Evaluate → improve | 3 |
| 2 | Extrapolation error | 8 (predict 3) |
| 3 | BC | 5 (chk 2) |
| 4 | CQL penalty | 9 (demo 2) |
| 5 | IQL expectiles; AWR = closed form | 8 (chk 2) |
| 6 | OPE vs truth | 5 |
| — | Setup | 2 |

Total: 29 + 9 + 2.

**Lab (core 3 + 51):**

| # | Exercise | Writes | Checkpoint | Min |
|---|---|---|---|---|
| 0 | Reconnect | — | Setup check | 3 |
| 1 | Coverage of each dataset | `coverage(ds)` | Exact | 5 |
| 2 | BC, plus a written prediction of where it matches | `bc_loss` | Exact on fixed tensors; ≥ protocol threshold on expert-narrow | 7 |
| 3 | Naive offline DQN (reference) and the Q-gap | `q_gap` | Exact; the gap grows on the record | 6 |
| 4 | CQL penalty | `cql_penalty` | Exact | 8 |
| 5 | **Planted bug**: CQL sign flipped | Diagnose from the Q-gap; fix | Q-gap bounded | 5 |
| 6 | IQL expectile loss (6a) and AWR weights (6b) | `expectile_loss`, `awr_weights` | Exact | 9 |
| 7 | Live, one dataset: {BC, naive, CQL} × 3 seeds, plus IQL × 1 ("single seed, not a ranking"); rollouts and FQE vs truth | — | Verdicts via `compare`; FQE error reported | 11 |

- **Compare:** BC and ground truth. The page explains why there is no library comparison: d3rlpy 2.8.1 runs a health check at import that requires the old `gym>=0.26`, which on Colab would replace the preinstalled `gym==0.25.2`; it also pins `gymnasium==1.0.0`, and its IQL is continuous-only. The technical review verified this, and we dropped d3rlpy (OPEN_QUESTIONS A8).
- **Stretch:** neural FQE against rollouts.

#### M15 · Capstone (C, pairs)

**Schedule:**
- The capstone page is published at the **Day 3 wrap-up** as reading.
- The Day 4 wrap-up gives it 4 minutes plus a sign-up board, and pairs are fixed by 17:00.
- On Day 5: kickoff 10 · work 80 · share-outs 30.
- With more than 8 pairs, use track panels: 2 minutes per pair, one Q&A per track.

**Objectives:** frame a new problem; run one controlled change within a stated budget; report per-seed results (≥ 3 seeds; CIs only with ≥ 5); give an honest analysis of what failed.

**Tracks** (each starter runs a baseline end to end in under 5 minutes):

| Track | Task | Starter |
|---|---|---|
| 15a · Ship a recommender policy | Offline candidates from `recsim_logs`, chosen with OPE and CIs | Simulator truth is the test set and is called **once**, enforced by a counter |
| 15b · New control task | `Acrobot-v1` with your own DQN or PPO, 3 seeds, one cell per seed | — |
| 15c · Preference tuning | A reward-hacking mitigation measured on the exact toy (the default). On a T4: a new preference set (`Anthropic/hh-rlhf`, MIT, or `Intel/orca_dpo_pairs`, Apache-2.0), reporting held-out preference accuracy and KL. It has no gold reward, so it is labeled exploratory. | Ships the reference DPO loop, so it doesn't depend on M12's stretch |

**The 80 minutes:**

| Minutes | Activity |
|---|---|
| 0–10 | Baseline |
| 10–20 | Framing, plus a written prediction card |
| 20–60 | One change |
| 60–65 | Freeze |
| 65–70 | One ground-truth evaluation |
| 70–80 | One-pager, auto-filled from records |

**Rubric** (4 criteria × 3 levels, equal weight):

| Criterion | 1 | 2 | 3 |
|---|---|---|---|
| Framing | S, A, R named | All four plus evaluation, justified | Plus assumptions (overlap, stationarity, termination) and what would break them |
| Method and budget | Unmotivated; partial budget | Tied to a failure mode from the week; full budget sentence | Plus one controlled change predicted beforehand; own vs reference code declared |
| Evidence | One seed, or a ranking inside overlapping spreads | ≥ 3 seeds shown, or IQM + CI; ties called; OPE with a CI | Plus a ground-truth or held-out check, and a health diagnostic plot |
| Honesty about failure | None, or "needs tuning" | One failure with its symptom | Diagnosed with evidence; says what would change the conclusion |

**One-pager headings:**
- Problem and decision-maker
- Framing table
- Prediction (written first)
- Method and what changed
- Budget (auto)
- Results and verdict
- Ground-truth / sanity check
- What failed and why
- Next day
- Run-record IDs

---

## 5. Testbeds, datasets and dependencies

### Testbeds
Gymnasium environments are marked; the rest are in `prl.envs`.

| Testbed | Used in |
|---|---|
| `CliffGridworld` (parametric; S&B Example 6.6 layout) | M1, M2, M3, M10 (λ sweep), M13 |
| `InventoryMDP` (Poisson demand; DP optimum) | M1, M2, M7 framing |
| `RiverSwim` (layout from Strehl & Littman 2008, §6, Fig. 1; length parameter is ours) | M6 |
| `RecSim` (contextual bandit: misspecification knob, drift, logged mode) | M5, M13, 15a |
| `Baird` (S&B §11.2) | M7 |
| Probe environments (Jones 2021) | M9 |
| `CartPole-v1` (Gymnasium) | M0, M7–M11, M14 |
| `Pendulum-v1` (Gymnasium) | M10 and M11 stretches |
| `Acrobot-v1` (Gymnasium) | 15b |
| `LunarLander-v3` (Gymnasium) | M8 stretch |

### Datasets
Datasets ship **inside `prl`** as package data (`prl/src/prl/datasets/`), because a `pip install …#subdirectory=prl` does not include the repo's root `data/`.
- Each is built by `scripts/make_datasets.py`, with a sha256 and a provenance JSON.
- The total is an estimated 5–10 MB.

| Dataset | Behavior policy / provenance | Used in |
|---|---|---|
| `gridworld_logs` | ε-soft on `CliffGridworld`; true values known | M13 |
| `recsim_logs` | ε-greedy over LinUCB, with propensities | M13, 15a |
| `cartpole_{expert_narrow,medium,mixed}` | Hand-coded linear controller (deterministic); noisy mixes log π_b | M14 |
| `m09_broken_runs`, `m09_sweep` | Reference DQN (4 planted bugs); 5 configs × 10 seeds | M9 |
| `m11_pretrained_policy.npz` | MLP distilled from the controller by BC | M11 |
| `m12_preferences` | SmolLM2 samples labeled by a gold rule that a verbosity exploit can game (our license) | M12 stretch, 15c |
| `_expected/*.npz` | Fixtures for `prl.checks` (precomputed values, never solver code) | All checkpoints |

### Artifact dependencies
Each arrives as reference code included in the later notebook, never as a participant's file.

| From | Used by |
|---|---|
| M1 environments and evaluator | M2, M3, M6, M10, M13 |
| M2 `estimate_model` and VI | M6 (MBIE-EB, optimum) |
| M3 Q-learning | M6, M7 |
| M5 ridge and logging | M13, 15a |
| M8 DQN | M9, M14, 15b |
| M10 GAE | M11 |
| M11 PPO | 15b |
| M12 | 15c |
| M13 FQE | M14, 15a |

---

## 6. Run paths and hardware

| Lab | Designed runtime | Without GPU | QUICK in CI |
|---|---|---|---|
| M0–M11, M13, M14, 15a, 15b | Colab CPU | Whole lab | Smaller budgets |
| M12 core | Colab CPU | Whole core | Tiny test-double LM, recorded as such |
| M12 stretch; 15c GPU option | Colab T4 | Toy only | Test double |

**Why CPU.** Small MLPs are bound by environment steps, and Colab's T4 runtime has the same 2 vCPUs. This is an expectation that Phase 3 measures (OPEN_QUESTIONS A5).

**Throughput estimate.** These are rough M1 Pro numbers from the technical review, taken with torch at 2 threads. They are not a Colab time, and none is a run record.

| Workload | Steps/s |
|---|---|
| Raw `CartPole-v1` | ≈ 184k |
| Minimal DQN (`train_freq=4`) | ≈ 9.6k |
| SB3 DQN | ≈ 4.7k |
| SB3 PPO | ≈ 3.6k |

- A 50k-step run takes 5–14 s here. Even at 3–5× slower on Colab, the 10–12 minute live cap is comfortable.
- `00-setup` therefore includes a 20-second throughput benchmark cell. Its Colab run record is the first real budgeting evidence.
- The "recorded bands + your seed" decision stands. If the Colab numbers allow, labs run more seeds live within the cap (OPEN_QUESTIONS A12).

**Colab.** Colab's runtime state is pinned from `googlecolab/backend-info@e39694e267` (2026-10-03):

| | |
|---|---|
| OS and Python | Ubuntu 24.04.5, Python 3.13.16 |
| CPU runtime | torch 2.11.0+cpu (no triton) |
| GPU runtime | torch 2.11.0+cu130, triton 3.6.0 |
| Other packages | numpy 2.1.3, scipy 1.16.3, pandas 2.2.3, matplotlib 3.10.0, transformers 5.18.0, accelerate 1.15.0, datasets 4.8.5, gymnasium 1.3.0, gym 0.25.2, pip 24.1.2 |

The install cell:
- Is generated, and calls pip through `subprocess` with `-c constraints`. The constraints list **every** installed distribution as `name==version` (from `importlib.metadata`, never URLs), except an allow-list: `gymnasium`.
- Pins exactly what it adds: `gymnasium==1.4.0`, `stable-baselines3==2.9.0`, `trl==1.15.0` (M12, 15c) and `prl@{ref}`.
- On `ResolutionImpossible`, stops with "Colab changed; tell the instructor".
- If a module that is already imported changes version, prints "Restart session".

`prl`'s dependency floors must sit at or below Colab's versions (numpy 2.1, scipy 1.16, matplotlib 3.10). The SciPy-1.18 requirement applies only to our lock, never to `prl`.

The technical review simulated this with Colab's versions and pip 24.1.2: only gymnasium changed, and `pip check` was clean. SB3 2.9 prints the "Gym has been unmaintained" banner because Colab preinstalls `gym`. The setup cell silences that import, and it is a Gotcha.

**Local.**
- `uv run --group notebooks jupyter lab` on macOS, Linux or WSL 2.
- The setup cell checks pins and warns rather than installing.
- The default device is CPU.

**AWS** (`infra/aws/`, CloudFormation).
- **Instance:** `g4dn.xlarge` (T4, compute capability 7.5).
- **AMI:** the Base OSS Nvidia Driver GPU DLAMI (Ubuntu 24.04), from the SSM parameter `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id`. The latest release notes (2026-10-08) list G4dn, driver 595.91.07, CUDA 13.2 and system Python 3.12.
  - The `create` script **resolves the AMI once** and passes the literal ID.
  - It does not use a `AWS::SSM::Parameter::Value<…>` parameter. That re-resolves on every stack update and would replace the instance, losing the participant's work.
- **Storage:** EBS ≥ 100 GB with `DeleteOnTermination: true`.
- **Network:** no inbound rules. Egress goes through a public IP in the default VPC, or through NAT or endpoints (documented).
- **Access:**
  - An SSM instance role.
  - Users connect by SSM port forwarding, which needs the Session Manager plugin locally.
  - JupyterLab binds to 127.0.0.1, with its token read through SSM.
- **Setup:**
  - User-data clones the repo and runs `uv python install 3.13` before `uv sync`.
  - JupyterLab runs as a **systemd unit** with `PRL_PLATFORM=aws`, so it survives a stop and start.
- **Cost controls:**
  - **Idle stop:** a CloudWatch alarm stops the instance when CPU stays below 5% for about 90 minutes, which spans lunch. The action is `!Sub "arn:aws:automate:${AWS::Region}:ec2:stop"`.
  - **Tags** on every resource.
  - **Account budget:** a one-time `account-budget.yaml` (`AWS::Budgets::Budget`). Its region requirement is open: OPEN_QUESTIONS B21.
- **Scripts:** `create | connect | stop | start | destroy | cohort-up N | cohort-down`.
- **Pricing:**
  - `infra/aws/price_table.py` runs by hand or in the weekly job, never at render.
  - It streams the EC2 offer CSV (303 MB) or calls `pricing get-products`, filters to g4dn.xlarge / Linux / shared / on-demand, and commits the price, date and source URL to `_variables.yml`.
- **Quota and alternative:** the quota request is documented. SageMaker AI Studio (JupyterLab space with built-in idle shutdown) is documented as an alternative.
- **Known risk:** TRL's Triton kernel on a T4 (OPEN_QUESTIONS B22).

---

## 7. Engineering design (Phase 1)

### Layout
```
CLAUDE.md PLAN.md OPEN_QUESTIONS.md CHANGELOG.md LICENSE (MIT) LICENSE-CONTENT (CC BY 4.0)
_quarto.yml _variables.yml brand/{light,dark}.yml prl.scss filters/*.lua _includes/ (generated)
index prepare setup schedule day-1..5 notebooks readiness references notation next-steps faq
teach facilitator-guide instructor-pace knowledge-checks welcome (revealjs)   modules/00..15
labs/src/*.py labs/shared/*.py labs/mutants/*.py   notebooks/*.ipynb (generated, committed)
prl/{pyproject.toml, src/prl/…, tests/}   runs/ (records; runs/inbox gitignored)   data/README.md
images/*.svg (+ .dark.svg)   fonts/ (vendored)   scripts/   infra/aws/   .github/workflows/   .claude/agents/
```

### `prl` public API
No algorithms. Dependencies: numpy, scipy, gymnasium, matplotlib. No torch.

| Module | Contents |
|---|---|
| `runtime` | `detect()`: platform, accelerator, CPU model, n_cpu, RAM. `device()`. `quick(designed)` → (flag, reason). `settings(live, quick)`. `seed_everything()`. `print_versions()`. `test_double(...)`. `API`. |
| `lab` | `init(notebook, content_sha, designed, api, ns=globals())` → `Lab`, with `.solution(n)`, `.solution_value()`, `.use_reference(n)`, `.check(n, fn, *objs, label)`, `.metric()`, `.summary()`, `.finish()`. State lives as plain dicts in `ns["__prl_lab__"]`, so rerunning `init` keeps the stored references and stubs, and reloading `prl` cannot break `isinstance`. `check` decides "whose code" by identity against the stored references. It catches `CheckFailed`, prints the help message, records the result, then re-raises. A record's `status` is "pass" only if every checkpoint was reached and passed. |
| `checks` | `CheckFailed`, `assert_close/shape/simplex`, `assert_threshold(v, t, provenance)`, `mNN.check_*` (loading `_expected` fixtures) |
| `envs` | `TabularMDP(P, R, γ).validate()`, `CliffGridworld`, `RiverSwim`, `InventoryMDP`, `RecSim`, `Baird`, `register()`. Probe environments are *not* in `prl`: Module 9 participants write them (decided in Phase 1) |
| `data` | `load(name)` (sha-checked), `LoggedBandit`, `Transitions`, `generate_logs()` |
| `evaluate` | `run_seeds`, `bootstrap_ci`, `iqm` (`scipy.stats.trim_mean(x, 0.25)`), `stratified_bootstrap`, `compare(a, b, budget)` → A, B or tie plus a sentence, `rollout_returns`. Replaces rliable, which was archived 2025-10-15. |
| `plot` | `style()` (opaque white, retina, palette generated from `brand/`), `curves(band=)`, `with_recorded(band, yours)` (refuses a budget mismatch), `bars_ci`, `grid_values`, `policy_arrows`. Bands come from `prl/src/prl/datasets/bands/`, which `scripts/export_bands.py` copies from `runs/`. Each band carries its source record ID and sha, and the `prl` tag is bumped after every export. A Colab install never sees `runs/`. |
| `record` | `SCHEMA`, `validate`, `content_sha`, `emit` |

### Notebooks
- **Build:** `scripts/build_notebooks.py` turns `labs/src/NN-slug.py` into `notebooks/NN-slug.ipynb`.
  - Deterministic: nbformat 4.5, sorted keys, no outputs.
  - Role-based cell IDs: `ex03-stub`, `ex03-sol`, `ex03-chk1`.
  - Colab `metadata.id` equals the cell ID, so the site can deep-link with `#scrollTo=` (verify in Phase 1).
- **Generated cells:**
  - **Header:** title, badge, day, module, core minutes, designed runtime, a link back to the module's "In the lab" section and to the next module, objectives, and the rhythm line.
  - **Install/setup:** on Colab, installs; elsewhere, checks the pins. Then sets `SEED = seat`.
  - **`lab.init(...)`**
  - **`lab.finish()`:** prints the record's JSON between markers. On Colab it also triggers a `files.download`; locally it writes to `runs/inbox/`.
- **Solution cells:** a two-level fold, Hint then Solution, using `#@title … { display-mode: "form" }`, `cellView: form` and `jupyter.source_hidden`. A solution cell prints one line per exercise: that it stored the solution without using it, and how to use it (`lab.use_reference(N)`, optionally naming one function).
- **Checkpoint output:**
  - Pass: `✓ Checkpoint 3 passed (your code) · 0.4 s`.
  - Fail: `✗ … expected … · Hint … · Stuck? lab.use_reference(3)`, in words, not ANSI color.
- **Lints:**
  - the rhythm, plus `predict_answer` and `explain_ref`;
  - top-level TODOs;
  - no Quarto syntax and no inline HTML styling in notebook markdown;
  - `# %% include` expansion.
- **Jupytext source:** a percent-header metadata key (`# %% role="ex03-stub"`) becomes the nbformat cell id.
  - Ids must match `^[A-Za-z0-9_-]{1,64}$` and be unique.
  - Jupytext metadata is stripped, and `kernelspec` is set to python3.
  - GPU notebooks get both `accelerator: "GPU"` and `colab: {gpuType: "T4"}`.
- **The "solutions bound" variant** is the same notebook run with `PRL_WORKED=1` (OPEN_QUESTIONS A4).
- **`content_sha`:**
  - sha256[:16] of `"\n\x1e\n".join(f"{id}\n{src}")` over the code cells.
  - Taken **after** `# %% include` expansion and **before** `#@title` injection, with LF line endings and trailing whitespace stripped.
  - A pytest test pins the definition.
  - A record is stale ("before the notebook last changed") when its `content_sha` **or** its `prl_version` differs from the current one.
- **`scripts/check_notebooks.py`** (nbclient) has four modes:
  - *worked*;
  - *learner*: must stop at the first TODO;
  - *verify*: right after each exercise's checkpoints, swap in its stub and each mutant and rerun the checkpoint cells (checks call the participant's function themselves, so rerunning run cells is unnecessary). The stub must be rejected by every checkpoint of its exercise; a mutant by every checkpoint that calls the function it replaces. Only a `CheckFailed` counts as rejection;
  - *record*.
- **Mutant mechanism:**
  - `labs/mutants/mNN.py` holds functions decorated `@mutant(ex=3, replaces="evaluate_exact", id="P_pi_T", why="…")`.
  - In verify mode, for each mutant, the tool inserts after exercise n's last checkpoint:
    1. `lab._mutant_begin(n, <mutant source>, id)`, which saves the replaced names in `ns` and rebinds them;
    2. copies of the `exNN-run*` and `exNN-chk*` cells;
    3. `lab._mutant_end(n)`.
  - In mutant mode, `check` records results instead of raising.
  - `_mutant_end` restores the names, then fails unless **every** checkpoint raised `CheckFailed`. A `NameError` or `NotImplementedError` does not count as a rejection.
  - Mutants of stochastic, training-based checkpoints are rejected once, by the threshold protocol (recorded), not on every CI run.
- **`scripts/threshold_protocol.py`** sets thresholds for stochastic checkpoints:
  - **Seeds:** ≥ 100 for cheap (QUICK-budget) checks, ≥ 20 for live-budget checks.
  - **Threshold:** the midpoint between the solution's 1st percentile and the 99th percentile of the strongest mutant (or random). The protocol requires a stated gap between the two.
  - **Platforms:** it runs in the colab-sim environment plus arm64.
  - **Output:** an experiment record. The checkpoint's provenance comment states N and the implied flake bound (the rule of three: with no failures in N, the upper bound is about 3/N).

### `_variables.yml` (schema)
| Key | Contents |
|---|---|
| `repo` | owner, name, url, site_url, colab_base, badge_ref |
| `prl` | ref, version |
| `workshop` | title, org, licenses |
| `clock` | clinic, warmup, A, B (around lunch), C, wrapup |
| `days.dN` | title, question, modules, framing |
| `packages` | checked, `colab.use_preinstalled`, pins |
| `models` | id, revision, license |
| `datasets` | file, sha256, bytes, license, generator, seed, modules |
| `runtimes` | colab-cpu, colab-t4, local, aws-g4dn, ci, ci-proxy |
| `readiness` | max_run_age_days, items |

Each `modules.mNN` entry has:
- **Identity:** n, slug, day, slot, level, title, summary, question, where_we_are.
- **Page content:** build, before_you_start, objectives.
- **Runtime facts:** runtime, cost, without_gpu.
- **Lab design:** testbeds, framing, from_scratch, compared_with {lib|truth, why}, planted_bug {what, diagnostic}, diagnostics, stretch, scaffold.
- **Exercises:** a list of {n, title, minutes, writes, checkpoint, predict_answer, explain_ref}.
- **Budgets:** {live, quick}.
- **Status:** status, estimate_minutes (null until measured).

### Run records (`runs/*.json`)
| Group | Fields |
|---|---|
| Identity | schema, kind=notebook, notebook, content_sha, prl_version, git_sha (= `prl_commit`, read from `direct_url.json` on Colab and AWS; the repo HEAD locally), badge_ref, dirty (local only), date |
| Where | platform (colab, local, aws, ci), env, colab_release, hardware {cpu, n_cpu, ram_gb, accel, gpu}, torch_threads, python, packages |
| How | quick, quick_reason, test_doubles[], shims[], solutions_bound, scope, seeds[], seat, settings |
| Results | seconds, checkpoints[{label, pass, whose, seconds}], metrics{}, status, source |

**Experiment records** (`kind=experiment`) have their own schema: script, args, budget, seeds, per-seed metrics, hardware and platform.

`scripts/add_run_record.py` validates each record and rejects credentials, absolute paths and usernames.

### CI (GitHub-hosted, 4 vCPU)
- **`ci.yml`**, on every push and PR:
  - ruff and pytest; `prl` tests on Python 3.12 and 3.13.
  - The drift gate.
  - **colab-compat:** `uv pip compile req.in -c <Colab freeze> --python-version 3.13 --python-platform x86_64-manylinux_2_28`, with the PyTorch `cpu` and `cu130` extra indexes and `--index-strategy unsafe-best-match`, for both the CPU and GPU freezes.
    - `req.in` includes `prl`.
    - The freeze comes from `pip-freeze.txt` and `pip-freeze.gpu.txt` at a pinned `backend-info` commit. Its `torch @ URL` lines are rewritten to `torch==2.11.0+cpu` or `+cu130`, and its `file://` and `git+` lines are dropped.
    - The technical review dry-ran this; only gymnasium changes.
  - **colab-sim notebook jobs (the primary gate):** a **per-notebook** matrix, each job ≤ 8 minutes.
    - Environment: Python 3.13 on Linux x86_64, with exactly the pinned Colab CPU freeze (torch 2.11.0+cpu) plus our pins.
    - Flags: `PRL_QUICK=1`.
    - Each job runs worked + verify + mutants in one pass, then learner.
    - Caches: uv, and `HF_HOME` keyed on the model revision.
    - M12 uses the TRL shim and a cached model, or a recorded tiny-LM test double.
  - The **locked-environment** notebook run (torch 2.14.1, numpy 2.5.3…) moves to the weekly job. It differs from Colab, and fixtures, thresholds and `prl` must hold in both.
- **`pages.yml`:** render → link-check → **site-check** (§8) → deploy. Nothing executes at render. The site-check also runs on PRs that touch the site.
- **`health.yml`**, weekly:
  - the locked-environment notebook run;
  - a bump of the pinned `backend-info` commit, then colab-compat and colab-sim on the new freeze;
  - the M12 Hub path;
  - the `uv lock --upgrade` diff;
  - macOS arm64;
  - a **ci-proxy** job (`taskset` to 2 cores, live budgets), recorded as `ci-proxy` and never counted as "designed";
  - the full UI matrix (WebKit, 200% zoom).

---

## 8. Site and UI design (`ui-expert`)

### Information architecture
| Element | Design |
|---|---|
| Brand link | Home. There is no separate "Home" item. |
| Navbar | Start here · Schedule · Days ▾ (Module 0, Day 1–5) · Notebooks · Reference ▾ (references, notation, next steps, FAQ) · Instructors ▾ (teach, facilitator guide, pace sheet, knowledge checks, welcome slides, readiness) · GitHub icon (with aria-label) |
| Sidebar (generated) | Module 0, then Day N → its three modules |
| Navigation aids | Breadcrumbs on; page navigation runs Day 1 → M1 → M2 → M3 → Day 2 … |
| Footer | Setup, FAQ, Notation, Readiness |
| Home | 2 CTAs (Start here, See the schedule); the five days; the path (every module); outcomes; who it is for; how a module works; "What this workshop is not". No readiness verdict in the hero. |
| Day page | Question and framing → the day's clock → module cards (Read the briefing first, Colab second, a runtime chip) → warm-up → wrap-up and the running results table. The facilitator run sheet lives in the facilitator guide. |
| Schedule | Two narrow tables ("Daily clock"; "What runs in each slot"). Module B's lunch split and the capstone split appear as text, never as color bars. |

**Participant flow.**
- Module page → "Open Lab N in Colab" (opens a new tab).
- The Colab header links back to "In the lab"; the last cell links to the next module.
- Each "In the lab" row deep-links to its exercise.

### Module template components
- **Header strip** (`.module-header`) says *when*: chips for Day, Module and Slot, plus the clock, e.g. "Briefing 11:30 · lunch · Lab 13:10 · Debrief 14:20". It also holds the summary and the CTA group:
  - a text button: Open *Lab N: title* in Colab ↗, with a visually hidden "(opens in a new tab)";
  - "Download .ipynb";
  - "Other ways to run" → `setup.qmd#local`.
- **Facts box** (`.facts`) says *what*, in the brief's order: Duration · Level (linked to the levels table) · Lab runtime ("Colab CPU · ≤ 12 min live compute", labeled [estimate] until a teaching-eligible record exists, then "measured X min, env, date [record]") · Cost · Without GPU. The page lint checks the order.
- **In the lab:** `# | Exercise | Briefing § | Min`, plus a "Core N of 70 min" total row and a "Planted bug" chip. Rows link to the Colab `scrollTo` anchor.
- **"In the lab." callouts** become `::: {.in-lab ex=3 eq="eq-eval"}`. A filter renders them as a labeled box with an Exercise chip, and the "Equations and where you implement them" table is **generated** from them. The lint proves every exercise and equation is covered.
- **Agenda:** `## Agenda {#agenda}` with the caption "Briefing, 40 min" and the total row "40 = 30 exposition + 10 activities".
- **Check yourself** is numbered ("Check yourself 3.2"), with the summary "Answer to 3.2". It uses native `<details>` (disclosure.lua) and opens on print.
- **Equations:** long equations are split with `aligned`. Only displays that actually overflow get `tabindex="0" role="region" aria-label="Equation N"`. The site-check lists any that overflow at 360 px.
- **Sentence-case headings** (linted); `lang: en-US`.

### Theme and tokens
- `brand/light.yml` and `brand/dark.yml` (`_brand.yml` format) are the base tokens. `prl.scss` emits `:root{--prl-*}`, and every rule uses `var(--prl-*)`.
- `respect-user-color-scheme: true`. No cosmo theme, so no Google Fonts import.
- Fonts are vendored woff2, subset to Latin, Latin-ext, Greek and math arrows/operators. A test checks every glyph in the rendered text is covered.
- KaTeX and the OJS libraries are vendored, or the site-check enforces an explicit host allow-list.
- Token values (contrast vs background noted) are in the UI review appendix; examples:

| Token | Light | Contrast | Dark | Contrast |
|---|---|---|---|---|
| text | `#1d2430` | 15.6 | `#d7dee6` | 13.2 |
| link and focus | `#0b5c99` | 7.0 | `#8cc4ee` | 9.6 |
| muted | `#56606c` | 6.4 | `#9ba8b6` | 7.4 |

- Chips use icon + word, and the icon is `aria-hidden`.
- Focus ring: `outline:2px solid var(--prl-focus); outline-offset:2px; box-shadow:0 0 0 4px var(--prl-bg)`. `html{scroll-padding-top:4.5rem}` keeps focus from being hidden under the navbar (WCAG 2.4.11).

### Plot palette
Shared by the site, OJS and `prl.plot`. It is generated from `brand/` by `scripts/gen_tokens.py`, behind the drift gate.

| Series | Light | Dark |
|---|---|---|
| 1 | `#0072B2` | `#3F97DB` |
| 2 | `#D55E00` | `#E0702F` |
| 3 | `#009E73` | `#22AE80` |
| 4 | `#AA3377` | `#A9529E` |
| Neutral (dashed reference lines) | `#6B7480` | `#8C98A5` |

- The UI expert checked this palette with a color-vision-deficiency (CVD) validator. Worst ΔE: light 10.3, dark 9.8. Contrast with the background: ≥ 3.4 (light) and ≥ 3.7 (dark). Phase 1 re-runs the test.
- At most 4 series; beyond that, use small multiples.
- Sequential: cividis. Diverging: blue–grey–vermillion centered at 0.
- Notebooks always use the light set, on an opaque white figure.

### Figures and demos
- **Figure files:** deterministic SVG (`svg.hashsalt`, no date, paths for text). There is a light `NN-x.svg` and a dark `NN-x.dark.svg`, rendered as `.light-content`/`.dark-content` by `filters/figures.lua`. `.paper` is the fallback.
- **Figure format:** sized for a 680 px column; text ≥ 12 pt; aspect ratio ≤ 2:1; `lightbox: auto`.
- **Alt text and captions:**
  - `fig-alt` describes what is drawn, in 1–2 sentences.
  - The caption opens with **What to notice:**, then gives the budget, then the record ID if measured.
  - Lint: alt ≥ 40 characters, not equal to the caption, never "image of".
- **OJS demos.** Every demo has a Predict-first prompt, an `aria-live` readout, labeled inputs, a seeded RNG and colors from `var(--prl-series-N)`. None autoplays (each respects reduced motion), and each takes ≤ 4 minutes of the agenda.

| Demo | Module | What it shows |
|---|---|---|
| Contraction vs γ | M1 | 2-state MDP; log axis; bound vs actual; tolerance slider |
| Max bias | M8 | E[max Q̂] − max Q vs number of actions and σ |
| UCB over time | M4 | Scrub t over a precomputed seeded 3-arm run |
| IS variance vs mismatch | M13 | Plus a horizon slider, weight histogram and ESS |
| PPO clip region | M11 | Sign of A, ε slider, zero-gradient shading |
| β-frontier | M12 | Exact π\* bars plus gold, proxy and KL readout |

### Readiness, notebooks, slides and print
- **Readiness:**
  - A verdict sentence, then the legend, then one matrix: Lab | Teach on Colab? | Designed runtime | On its designed runtime | Real path elsewhere | CI | Gaps.
  - Chip states: `✓ Teaching-eligible`, `◐ Real path elsewhere`, `● CI (QUICK / test doubles)`, `⚠ Before the notebook last changed`, `○ Not run`.
  - A sticky first column, and a `<details>` block of records per lab.
- **Notebooks page:** cards per day. Each card has a title (linking to the briefing), chips for runtime, live-compute estimate and status (linking to `readiness#mNN`), plus Open in Colab and Download.
- **Welcome deck:** about 10 slides, built from includes. `theme: [brand, slides.scss]`, `::: notes` with minutes, and a test that `?print-pdf` works.
- **Print:** handout pages set `body-classes: handout`. Print forces light tokens and hides navigation. External link URLs are shown, and blocks don't break across pages. The capstone template must fit on one Letter page and one A4 page.

### Automated site-check (Playwright + Quarto 1.10's bundled axe-core, WCAG 2.2 AA)
Pages × {light, dark} × {360 px, 1280 px}. The check fails on any of the following:
- serious or critical axe violations (exceptions need an allow-list entry with a reason);
- token contrast (text ≥ 4.5, UI ≥ 3) or palette CVD;
- reflow: page-level horizontal scroll at 320 px;
- keyboard and focus: visible and not obscured, `<summary>` toggles with Enter and Space;
- targets smaller than 24 px;
- headings out of order, or a missing table header or caption;
- duplicate link names pointing to different targets;
- reduced motion not respected;
- KaTeX not typeset;
- requests to non-allow-listed hosts;
- print page counts.

---

## 9. Tooling and versions (verified 2026-10-09)
| Tool | Version | Decision |
|---|---|---|
| Quarto | 1.10.19 (local 1.6.40) | Pin in CI; upgrade locally in Phase 1. Since 1.7, render fails on IPython display errors. |
| Python | Colab 3.13 | `requires-python >=3.12` (SciPy 1.18). CI notebooks run on 3.13. *Changes the brief's ≥3.11.* |
| PyTorch | 2.14.1 (Colab 2.11.0+cu130) | Locked for local and CI. On Colab, use the preinstalled version (≥2.8) and record it. |
| Gymnasium | 1.4.0 (Colab 1.3.0) | `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `LunarLander-v3`. Vector autoreset defaults to `NEXT_STEP`. |
| Box2D | `box2d==2.3.10` | Install alone. `gymnasium[box2d]` pulls `pygame-ce`, which conflicts with Colab's `pygame`. |
| SB3 | 2.9.0 | DQN, PPO, SAC comparisons |
| CleanRL | Needs gymnasium 0.29.1 and Python <3.11 | Annotated reading only |
| TRL | 1.15.0 (`PPOTrainer` removed in 1.13; `processing_class`; no `max_prompt_length`; `bf16` defaults to True). `DPOTrainer` always uses a Triton fused LM head, so it has **no CPU path**, and on a T4 (compute capability 7.5) it is unverified. | Pin exactly. On CPU: pure-torch shim plus `bf16=False`, recorded. On T4: a human smoke run in Phase 1. |
| MABWiser | 2.7.4 (2024-08-30); works on Python 3.13 with NumPy 2 (verified) | Optional M5 parity cell (stretch) |
| transformers, accelerate, peft, datasets | 5.19.0, 1.15.0, 0.21.2, 5.1.0 | No bf16 on T4: use fp16 or fp32 |
| uv, jupytext, nbclient | 0.12.24, 1.19.6, 0.11.0 | — |
| rliable | Archived | Own IQM and bootstrap |
| Open Bandit Pipeline, SCOPE-RL | Unmaintained | Not used |
| d3rlpy | 2.8.1; pins gymnasium 1.0.0; imports `gym>=0.26` at load (would replace Colab's gym 0.25.2); IQL continuous-only | **Dropped** (verified by the technical review) |
| Minari | 0.5.4; no CartPole data | Own datasets |
| LM | `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0, 134.5M, ungated, about 260 MB) | Pin revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` (last modified 2025-09-22) |
| AWS AMI | `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id` (Base OSS Nvidia Driver GPU, Ubuntu 24.04; release 2026-10-08) | Resolved once by `create`, then passed as a literal ID |
| Colab | `googlecolab/backend-info@e39694e267` (2026-10-03): Python 3.13.16; CPU torch 2.11.0+cpu (no triton); GPU torch 2.11.0+cu130, triton 3.6.0 | Pinned freeze drives colab-compat and colab-sim |

Prices come from the AWS Price List at build time. Third-party price trackers are never used.

---

## 10. Notation (decided on `notation.qmd`, followed everywhere)
Rule: when a paper's symbol collides with ours, keep its letter with a method subscript, and list the paper's original symbol on `notation.qmd`.

| Ours | Meaning | Replaces |
|---|---|---|
| r_t | reward after a_t (S&B's R_{t+1}). r(s,a) is its mean; the array `R[s,a]` holds r(s,a). | — |
| r_φ; r̂(x,a) | learned reward model; DM/DR reward model | Dudík's ϱ̂ |
| γ | discount only | Ouyang's γ → c_ptx |
| α | step size only | SAC α → α_ent; CQL α → α_cql |
| c | exploration-bonus coefficient | LinUCB α; MBIE-EB β |
| c_V, c_H | PPO value and entropy weights | c₁, c₂ |
| β | KL coefficient (RLHF, DPO). AWR weights are written exp(Â/β). | IQL inverse temperature |
| λ | λ-return and GAE (the same object) | ridge λ → λ_ridge |
| ρ_t, ρ_{0:t} | importance ratio and its product. In PPO, ρ_t(θ) = π_θ/π_θold. | PPO r_t(θ) |
| π_b | behavior policy (frees b for the baseline b(s)) | Dudík p; CQL π_β |
| δ_t | TD error | Hoeffding δ → p_fail |
| ε, ε_clip | exploration; PPO clip | — |
| K | GRPO group size | Shao's G (clashes with G_t) |
| N(s,a) | visit count (n stays n-step) | MBIE-EB n(s,a) |
| Λ_a | LinUCB Gram matrix | Li et al.'s A_a |
| μ, d | state weighting; feature dimension | — |
| τ, τ_e | trajectory; IQL expectile | — |
| ε_P | model error (M2) | — |
| also | θ, w, φ(s), π_ref, Z(x), H, 𝒜, V^π, Q^π, A^π, G_t, Â_t, KL | — |

---

## 11. Roles and parallel work

| Agent (`.claude/agents/`) | Owns | Reviews |
|---|---|---|
| `academic-director` | Curriculum, objectives, derivations and assumptions, notation, citations, briefing content, knowledge-check answers | All pages, for correctness |
| `pedagogy-expert` | Rhythm, timing, cognitive load, scaffolding, misconceptions, Check yourself, warm-ups, debriefs, facilitator guide, pace sheet, capstone rubric | Every page and lab, for learnability |
| `technical-expert` | `prl`, build and harness, checkpoints, mutants, thresholds, CI, records and readiness, dependencies, Colab compatibility, compute, `infra/aws` | Every notebook, for correctness and runtime |
| `ui-expert` | IA, template rendering, theme, mobile, accessibility, figures, OJS, slides, readiness layout, notebook presentation | Every rendered page |
| `verifier` | Phase 5 audits, in a fresh context | Everything |

**Build flow per module:**
1. The AD drafts the page and lab spec.
2. The PE reviews timing and rhythm.
3. The TE builds the notebook, checks, mutants and recorded experiments.
4. The UI expert reviews the rendered page.
5. The AD signs off on correctness.

The lead (the main session) integrates, owns `prl/`, and stops at each phase gate.

**Git worktrees.** `main` holds approved work only. The lead commits shared infrastructure there: `prl/`, `scripts/`, `_variables.yml` schema changes and workflows.
- **Branches:** in Phase 3, each module (or module × role task) runs in its own worktree. That is either an agent with `isolation: "worktree"` or `git worktree add ../prl-wt/mNN -b module/mNN-slug`. Branch names are `module/mNN-slug`, `site/<topic>` and `infra/<topic>`.
- **Scope:** a worktree touches only its module's files: `modules/NN-*.qmd`, `labs/src/NN-*.py`, `labs/mutants/mNN.py`, `images/NN-*`, and its `modules.mNN` entry. A needed `prl` change goes in as a request in the PR description.
- **Merge:** the lead rebases onto `main`, regenerates the includes and notebooks (which must pass the drift gate), runs the day's checks, then merges.
- **Generated files** are never hand-merged.
- **Cleanup:** worktrees are removed after the merge.

---

## 12. Risks
| Risk | Mitigation |
|---|---|
| Labs overrun (2 vCPUs; novices) | Budgets include the rhythm (≤52, or 51+3). Recorded bands plus your seed. Live compute ≤ 10–12 min. A "behind → demo" rule per lab in the pace sheet. Human pilot. |
| Flaky stochastic checkpoints | Exact checks on fixed tensors where possible. Thresholds from `threshold_protocol`. Bands never pass/fail. |
| Colab, TRL or Gymnasium drift | Exact pins for what we add; colab-compat gate; weekly health job |
| Box2D/pygame conflict on Colab | LunarLander only in a stretch; install `box2d` alone |
| Hugging Face Hub outage | M12's core runs on the toy. Part B falls back to a labeled test double. Facilitators pre-cache. |
| TRL's Triton-only DPO path (no CPU path; unverified on T4) | A pure-torch shim, recorded in `shims[]`. B6 is checked against transformers' own cross-entropy, which needs no TRL. A human T4 smoke run in Phase 1. A recorded T4 loss fixture as the fallback. |
| CI differs from Colab (lock vs Colab freeze) | colab-sim notebook jobs are the primary gate; the locked environment runs weekly |
| AWS stack update replaces the instance | The AMI is resolved once and passed as a literal ID |
| Reward hacking not visible | Exact frontier on the toy; T4 stretch evidence from records |
| Citation errors | `annote` records; the Phase 5 fresh-context audit |
| AWS quota, cost or access | Documented quota, budget, idle stop, `cohort-down`. The access model is in OPEN_QUESTIONS C1. |
| No lab teaching-eligible without Colab | A human Colab pass on the release checklist; the readiness page says so plainly |
| Accessibility or mobile regressions | Automated site-check gates deploys |
| Scope (16 pages + notebooks) | The golden Module 1 is reviewed before anything copies it |

## 13. Phases and exit criteria
| Phase | Work | Exit |
|---|---|---|
| 0 | Research, plan, persona reviews | `PLAN.md`, `OPEN_QUESTIONS.md`, `references.bib`, `.claude/agents/` committed. **Stop.** |
| 1 | Scaffold, in this order:<br>1. repo hygiene<br>2. `_variables.yml` + schema test<br>3. `prl` runtime, record, lab + tests<br>4. all five envs + M1 fixtures<br>5. evaluate, plot, data, checks<br>6. `brand/` + tokens + palette generation<br>7. build script with `00-setup` (including a throughput benchmark cell) and a stub M1<br>8. `check_notebooks` (worked and learner modes), `add_run_record`<br>9. `gen_includes`; every page as a placeholder<br>10. readiness generator, link check, page lint, basic site-check (axe at 2 widths × 2 themes)<br>11. workflows (`ci.yml` with colab-compat and colab-sim; `pages.yml`)<br>12. `infra/aws` templates and scripts, linted<br><br>Deferred to Phase 2 or later: mutant machinery (it starts with M1), threshold protocol, release check, `health.yml`, ci-proxy, macOS job, the full UI matrix. | CI green; the site renders and passes the basic site-check; 00-setup passes in colab-sim. **Human smoke runs:** 00-setup on Colab CPU (records the vCPU count, throughput and `scrollTo` behavior) and one `DPOTrainer` step on a Colab T4. **Stop.** |
| 2 | Golden Module 1: page, figures, notebook, mutants (mutant machinery built here), checks, stretch, knowledge checks, a local run record | **Stop** for close review |
| 3 | Day by day: three modules, drafted in parallel in worktrees by the personas; the day's notebooks run locally; recorded experiments and thresholds; readiness update | **Stop after each day** |
| 4 | Instructor kit: welcome slides, facilitator guide, pace sheet ("behind → demo" rule), knowledge checks with coverage test, warm-ups for Days 2–5, capstone kit, running results table | **Stop** |
| 5 | Fresh-context `verifier` audits: derivation, citation, code (worked, learner, mutants, QUICK in budget), practicality, timing, site | `RELEASE_CHECKLIST.md` lists the human-only tasks |

---

## Appendix A · One Predict prompt per lab
| Lab | Predict | Checkable answer |
|---|---|---|
| M1 | γ from 0.9 to 0.99: how many times more iterations to reach 1e-6? | About 10× or more. The measured count stays within the bound. |
| M2 | Does halving the data double the loss? | No. The bound rises by about √2, and the actual loss is often exactly 0 above a data threshold. |
| M3 | Whose greedy path hugs the edge? Whose online return is higher? | Q-learning's path hugs the edge; SARSA's return is higher (constant ε) |
| M4 | Rewards ×10 with the same UCB1 bonus: does regret go up or down? | Up. The bonus is now relatively small, so it behaves nearly greedily. |
| M5 | Unit x seen 100 times, λ_ridge = 1: what is LinUCB's width? | c/√101 ≈ 0.1c |
| M6 | Under uniform random play, does hitting time grow linearly or exponentially in chain length? | Exponentially (ratio p_L/p_R) |
| M7 | Baird's ‖w‖: does it go to 0, stay flat, or grow without bound? | It grows without bound |
| M8 | Is Q(s₀) above or below the MC return? Does Double DQN shrink the gap? | Above; yes (on the recorded band) |
| M9 | Constant-reward probe with reward 1, γ = 0.99: what is Q? And without the terminal mask? | 1; 100 |
| M10 | With a learned baseline, does the mean gradient change? Does the variance? | The mean doesn't; the variance ratio is < 1 |
| M11 | Â > 0, ρ = 1.5, ε_clip = 0.2: what is the gradient of the clipped term? | 0 |
| M12 | What is the DPO loss when π = π_ref? | ln 2 ≈ 0.693 |
| M13 | Whose MSE grows fastest as π moves away from π_b? | IS (confirm on the record) |
| M14 | Where does BC match the best offline-RL method? | Expert-narrow; not on mixed (Kumar et al. 2022, §4.2) |

For planted bugs, the Predict is the symptom's value: Q(goal) = 0, mean weight = 1, residual ≈ 0. For band exercises, it is where the seed lands, P(outside) = 2/(n+1). That is graded by the Explain prompt, never by a checkpoint.

## Appendix B · Misconceptions for Gotchas and debriefs (quoted misconception → correction)
| Module | Misconceptions |
|---|---|
| M0 | "A T4 speeds up CartPole" → small nets are bound by environment steps on the same 2 vCPUs (expected; Phase 3 measures it). · "A random agent is no baseline" → every "× random" check is built on it. |
| M1 | "γ is a speed knob" → it defines what is valued, and the horizon is 1/(1−γ). · "Converged because ΔV is small" → error ≤ γ/(1−γ)·‖ΔV‖∞. · "The reward is what I want" → the reward is a framing choice. |
| M2 | "95%-accurate model → 95%-as-good plan" → error grows with the horizon, (1−γ)⁻². · "PI is always faster" → fewer iterations, but each is an O(S³) solve. · "Stop on the mean change" → the bound needs the max. |
| M3 | "Q-learning is better than SARSA" → better greedy policy, worse online return. · "TD is biased, so MC is better" → TD usually has lower variance, and its bias shrinks as V improves. · "`done` means stop bootstrapping" → only on `terminated`. |
| M4 | "ε = 0.1 is a safe default" → regret grows linearly. · "UCB has no parameters" → it assumes rewards in [0,1]. · "One seed is enough to compare" → call ties. |
| M5 | "A better click model is a better recommender" → it never explores. · "The propensity is bookkeeping" → without it, Day 5 is impossible. · "Narrow widths mean the model is right" → only if the model is linear. |
| M6 | "ε-greedy gets there eventually" → after exponential time. · "Optimistic init is a hack" → it is optimism in tabular form. · "Add the bonus when acting" → it belongs in the backup. |
| M7 | "A bigger network will fix divergence" → Baird diverges even though w = 0 is representable. · "Semi-gradient TD is gradient descent" → there is no objective; A ≠ Aᵀ. · "Features are just preprocessing" → their scale sets the step size. |
| M8 | "My returns went up, so it works" → check Q against MC returns, and the masks. · "Replay is only for data efficiency" → it also decorrelates samples. · "done = terminated or truncated" → bootstrap on truncation. |
| M9 | "One seed is enough" → a correct new seed falls outside a 5-seed band 1/3 of the time. · "Loss down = learning" → the targets move. · "The mean is fine" → IQM is robust. |
| M10 | "A baseline adds bias" → an action-independent baseline does not. · "A good batch means a good gradient" → measure the variance. · "λ is a discount" → λ trades bias against variance; γ defines the objective. |
| M11 | "Clipping bounds the policy change" → it removes the incentive only; epochs can drift; watch the KL. · "PPO works out of the box" → implementation details matter. · "clipfrac 0 means stable" → it can mean stale log-probs. |
| M12 | "A higher RM score is better" → the proxy rises while the gold falls. · "DPO can't hack, it has no RM" → its implicit reward is still a proxy. · "Loss ln 2 means slow learning" → the policy equals the reference. |
| M13 | "Unbiased means accurate" → MSE is what matters. · "DR always wins" → not if both models are wrong. · "Any logs evaluate any policy" → only with overlap. |
| M14 | "Offline RL is off-policy RL with a buffer" → no new data ever corrects the errors. · "BC is weak" → on expert data it matches offline RL (Kumar et al. 2022, §4.2). · "OPE says it's great" → OPE is weakest where coverage is thin. |
| M15 | "Report the best seed" → report all seeds. · "No gain means failure" → a diagnosed tie earns full honesty marks. |

## Appendix C · Running results table (filled in by the room at each wrap-up)
| Day | Lab | Rows | Metric |
|---|---|---|---|
| 1 | 1 Inventory | the room's framings; γ 0.9 and 0.99 | V^π(s₀); iterations vs bound |
| 1 | 2 Inventory | VI, PI; plan in P̂ at 3 data sizes | sweeps; true loss over 5 seeds |
| 1 | 3 Cliff | SARSA, Q-learning; MC vs TD | online return over the last 100 episodes; greedy path length; RMS error at 100 episodes |
| 2 | 4 Bandits | greedy, ε-greedy, UCB1, TS; reward ×10 | regret at T over 20 seeds; verdict or tie |
| 2 | 5 RecSim | LinUCB, LinTS, predict-then-greedy × 3 regimes | regret over 5 seeds; mean logged weight |
| 2 | 6 RiverSwim | ε-greedy, optimistic, MBIE-EB | hitting time; % of seeds reaching the right end; regret over 10 seeds |
| 3 | 7 Baird / CartPole | each leg removed; linear Q | ‖w‖ after k updates; return ÷ random |
| 3 | 8 CartPole | your DQN; DDQN and SB3 bands; room IQM | return at budget; Q(s₀) − MC |
| 3 | 9 Logs | 4 bugs; probes; room IQM | diagnosis tally; probe Q; IQM [CI] + budget |
| 4 | 10 CartPole / gridworld | REINFORCE, + baseline, actor-critic; λ sweep | variance ratio; return; bias and variance by λ |
| 4 | 11 CartPole | clip on vs off × 3; SB3 band | return, approx-KL, clipfrac; verdict |
| 4 | 12 Toy / SmolLM2 | β ∈ {1, 0.25, 0.05}; DPO | proxy vs gold (exact); TV to π\*; \|yours − TRL\| |
| 5 | 13 Logs | 6 estimators × 2 mismatch levels | bias, std, MSE; does the CI cover truth? |
| 5 | 14 Datasets | BC, naive, CQL, IQL | true return; FQE estimate; Q-gap |
| 5 | Capstone | pairs | metric ± spread; budget; one failure |
