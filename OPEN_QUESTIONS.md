# Open questions

This file lists:
- decisions that still need an owner;
- facts we could not verify;
- places where this repo departs from the brief.

Each entry gives our working default, who decides, and when. When an entry is resolved, move it to **Resolved** with the date and the decision.

**Owners:** user (you), AD (academic director), PE (pedagogy expert), TE (technical expert), UI (UI expert), lead (the main session).

Last updated 2026-10-17 (Phase 3, Day 1: Modules 2 and 3).

**Phase 1 merged without its two human Colab smoke runs** (00-setup on Colab CPU and one
`DPOTrainer` step on a T4); their records are not filed. Everything due at "Phase 1 exit"
(A12, A13, B1–B3, B22) is still open and waits on them.

---

## A. Departures from the brief (please confirm)

R10 put the working defaults for A1–A8, A10, A12–A13 and C3–C5 into effect on 2026-10-09.
They stay listed here until you confirm them.

**A1. QUICK mode.** Owner: user. Needed by: Phase 1.
- **Brief:** "Every lab detects a CPU-only runtime and shrinks itself."
- **We propose:** QUICK turns on only when `PRL_QUICK=1` is set, or when a lab designed for a GPU finds no GPU. QUICK changes numbers only, never code paths. A QUICK seed is never overlaid on a full-budget band.
- **Why:** if CPU detection triggered QUICK, every CPU-designed lab would shrink on the very runtime it was designed for.

**A2. Minimum Python version.** Owner: user. Needed by: Phase 1.
- **Brief:** Python ≥ 3.11.
- **We propose:** the repo lock and CI require ≥ 3.12, and notebooks run on 3.13. `prl` itself keeps floors at or below Colab's versions.
- **Why:** SciPy 1.18 needs 3.12, and Colab has run 3.13 since 2026-08-19.

**A3. Installing packages.** Owner: user. Needed by: Phase 1.
- **Brief:** "The first cell installs exactly what it needs", and dependencies are pinned.
- **We propose:**
  - On Colab, keep the preinstalled torch (2.11.0) and every other preinstalled package. Install only what we add, pinned exactly, under constraints taken from every installed distribution.
  - Locally, the setup cell checks the pins and warns, instead of installing.
- **Why:**
  - Reinstalling torch costs about 2.5 GB per lab.
  - A `%pip` install would break the uv lock.
  - The technical review checked this against Colab's real freeze: only gymnasium changes.

**A4. The "solutions bound" variant.** Owner: user. Needed by: Phase 1.
- **Brief:** the build produces a "solutions bound" variant.
- **We propose:** a **run mode** (`PRL_WORKED=1`, also a Colab checkbox) rather than a second committed file. Executed copies are kept as CI artifacts.
- **Why:** one source and no drift.

**A5. Runtimes for Modules 8, 10–12 and 14.** Owner: TE, then user. Needed by: Phase 3.
- **Brief:** Colab T4 "where it matters".
- **We propose:** Colab CPU for every lab, except the Module 12 stretch and the 15c GPU option.
- **Why:** these are small MLPs. The technical review measured about 9.6k DQN steps/s and about 3.6k SB3-PPO steps/s on an M1 Pro (2 threads; *not* a Colab number). Colab's T4 runtime has the same 2 vCPUs.
- **Next step:** Phase 3 times these labs on Colab CPU and T4.

**A6. LunarLander in Lab 8.** Owner: user. Needed by: Phase 3 (Day 3).
- **Brief:** Lab 8 runs DQN on CartPole "then LunarLander".
- **We propose:** LunarLander-v3 moves to the stretch.
- **Why:** installing Box2D on Colab is risky, and the lab has a 52-minute core.
- **Alternative (AD):** a zero-compute core exercise that reads a recorded LunarLander band.

**A7. Harder environments in Lab 11.** Owner: user. Needed by: Phase 3 (Day 4).
- **Brief:** Lab 11 runs PPO on CartPole "then Pendulum/LunarLander".
- **We propose:**
  - The core uses CartPole only.
  - The stretch is SAC on Pendulum, as in the brief.
  - PPO on harder tasks becomes a capstone 15b option.
  - The Gaussian-policy start is the Module 10 stretch.
- **Why:** the compute budget.
- **Alternative (AD):** a core exercise that checks Gaussian `log_prob`/entropy on fixed tensors, plus a recorded Pendulum PPO band.

**A8. Comparing every algorithm with a maintained library.** Owner: user. Needed by: Phase 1.
- **Brief:** compare each core algorithm with "a maintained library".
- **We propose:**
  - **DQN and PPO:** compare with SB3 (recorded bands plus your seed).
  - **DPO:** compare with TRL, matching the loss on the same batch.
  - **LinUCB:** MABWiser parity is an **optional stretch** cell. It is verified to work.
  - **OPE and offline RL:** no library comparison. Ground truth and BC are the baselines, and the pages say why.
- **Why:**
  - **rliable** is archived.
  - **OBP and SCOPE-RL** are unmaintained.
  - **d3rlpy** 2.8.1 imports `gym>=0.26` at load, which would replace Colab's gym, and it pins `gymnasium==1.0.0`. The technical review verified this.
  - **CleanRL's scripts** pin gymnasium 0.29.1, so CleanRL is reading only.

**A9. Module 12 core.** Resolved 2026-10-09 (you).
- **Brief:** a T4 training run is the core.
- **Decision:** the core runs entirely on CPU; the T4 run is the stretch.

**A10. Installing `prl` with its datasets.** Owner: TE. Needed by: Phase 1.
- **Brief:** `pip install git+…@<tag>#subdirectory=prl`.
- **We propose:** datasets and recorded bands ship **inside `prl`** as package data. `data/` holds only the catalog.
- **Why:** a subdirectory install does not include the repo's root `data/` or `runs/`.

**A11. Module 12, B6: checking against TRL on CPU.** Owner: user / TE. Needed by: Phase 3 (Day 4).
- **Brief:** compare with TRL.
- **Problem:** TRL 1.15's `DPOTrainer` always uses a Triton fused LM head, so **it cannot compute a loss on any CPU**.
- **Default:** patch `trl.trainer.utils._ChunkedLogProbFunction` with a 12-line pure-PyTorch stand-in (verified: |Δloss| = 4.8e-7) and record it as a shim. B5 is checked against transformers' own cross-entropy instead.
- **Alternative:** a TRL loss fixture recorded on a T4. Either way, consider reporting the missing CPU path upstream.

**A12. Running more seeds live.** Owner: user / PE. Needed by: Phase 1 exit.
- **Brief:** none.
- **Situation:** "recorded bands + your seed" was decided before any throughput was measured. M1 Pro throughput suggests a 50k-step CartPole run takes about 5–14 s, so labs could run 3+ seeds live within the cap.
- **Default:** keep the decision. Revisit once the `00-setup` throughput cell has a recorded Colab run.

**A13. How Colab installs `prl`.** Owner: user / TE. Needed by: Phase 1.
- **Brief:** `pip install "git+…@<tag>#subdirectory=prl"`.
- **TE suggests:** a `prl` wheel published as a GitHub Release asset with the data inside, installed by URL. It is immutable and fast, and it avoids cloning the whole repo each session.
- **Default:** follow the brief (git + subdirectory) and keep the repo lean. Switch if the Phase 1 Colab install takes more than about 20 s.

**A14. Brief items we restored after the AD review.**
- An n-step core exercise in M3.
- Reward-vs-KL plotted on a KL axis in the M12 core.
- FQE-based OPE in M14's core.
- A named owner for the language-model reward model (TRL `RewardTrainer`, in a provided cell in the M12 stretch).

## B. Facts not yet verified

**Colab and environment**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B1 | Free-tier Colab T4 availability. The official FAQ names no GPU and says access is "not guaranteed". | Phase 1 human smoke run, then the release checklist | user | Phase 1 exit |
| B2 | Colab vCPU count, CPU model and CartPole throughput | `00-setup` throughput cell, recorded | user / TE | Phase 1 exit |
| B3 | Whether Colab honors deep links to a cell id (`#scrollTo=<cell id>`) | Phase 1 smoke run | UI / TE | Phase 1 exit |
| B26 | Colab behaviors Module 0's page relies on: the "not authored by Google" warning and its **Run anyway** button for notebooks opened from GitHub; double-clicking a form cell to show its hidden code; File → Save a copy in Drive; `<details>` hints rendering in Colab | Phase 1 smoke run or the human pilot | user / PE | Before pre-work is sent out |

**Software and tools**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B4 | ~~Whether two figures in one figure div get treated as subfigures~~ **Moot:** there is one figure per div; `filters/figures.lua` doubles only the image at post-render, so the figure keeps one number, caption and alt text (commit 67d2120) | — | — | — |
| B5 | ~~KaTeX self-hosted through the `html-math-method` object form~~ **Resolved:** the object form pastes its URL into every page as is, so it breaks either subdirectory pages or `quarto preview`; `filters/katex.lua` registers the vendored copy as an HTML dependency instead (commit ac42420) | — | — | — |
| B22 | TRL's Triton kernel on a T4 (compute capability 7.5; Triton officially supports 8.0+) | Human T4 smoke run: one `DPOTrainer` step. The fallback is the shim. | user / TE | Phase 1 exit |

**AWS**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B6 | ~~The SSM parameter for the Base DLAMI~~ **Resolved:** `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id` (AWS DLAMI docs; TE review) | — | — | — |
| B7 | ~~g4dn.xlarge on-demand price~~ **Resolved:** $0.526/h in us-east-1, from the public AWS Price List (EC2 offer 20261008184850), checked 2026-10-09 by `infra/aws/price_table.py`; rerun before relying on it | — | — | — |
| B21 | Where `AWS::Budgets::Budget` can be deployed. CloudFormation's per-region schemas (cfn-lint 1.57.2) list it in us-east-1, us-west-2 and eu-west-1 but not eu-north-1; the AWS docs do not say. `prl-aws budget` defaults to us-east-1. | AWS docs or a real deploy | TE | Before release |
| B25 | The AWS user-data on the real AMI, the idle-stop timing, the alarm re-arm service, KMS for the SecureString token, and `deploy --update` keeping parameters (see the Unverified section of `infra/aws/README.md`) | Human end-to-end run | user | Before release |

**Citations: lemma numbers and wording**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B8 | Kakade & Langford 2002, performance-difference lemma. "Lemma 6.1" was inferred from the LaTeX source. | Cite "§6" until the published PDF is read | AD | Phase 3 (Day 4) |
| B9 | Kearns & Singh 2002, the simulation lemma's number in the journal version. Module 2 cites AJKS V3 Lemma 2.2 (opened 2026-10-17) and does not cite Kearns & Singh. | Cite AJKS V3 Lemma 2.2 instead (done) | AD | Phase 5, if the journal version is ever opened |
| B27 | Singh & Yee 1994, the origin of the greedy-loss bound according to AJKS V3 §1.6. Not opened; Module 2 cites AJKS Lemma 1.11 only and does not name Singh & Yee on the page. | Open the paper before adding it to the bib or the page | AD | Phase 5 |
| B28 | Puterman 1994's treatment of modified policy iteration (section number). Not opened; Module 2's collapsed note describes the method without a section citation and the Further reading says "background only". | Open the book, or leave uncited | AD | Phase 5 |
| B23 | ~~The greedy-policy loss lemma~~ **Resolved:** AJKS V3 Lemma 1.11 gives V^{π_Q} ≥ V* − 2‖Q − Q*‖∞/(1 − γ) (no γ in the numerator); PLAN Module 2 corrected | — | — | — |
| B24 | Azar et al. 2024 §4.2, the exact statement about deterministic preferences (we say DPO pushes π(y_l) → 0) | Re-read §4.2 before quoting | AD | Phase 3 (Day 4) |

**Citations: RiverSwim and SARSA**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B10 | ~~The MBIE-EB bonus form~~ **Resolved:** β/√n(s,a), Strehl & Littman 2008 §3.2, Eq. 8, p. 1316 (AD review) | — | — | — |
| B11 | RiverSwim's origin and transition probabilities. The 2008 paper says it is "taken from" Strehl & Littman, ICTAI 2004 (not opened). Fig. 1's probabilities are an image. | Open the 2004 paper and read Fig. 1. Meanwhile our page states its own probabilities. | AD | Phase 3 (Day 2) |
| B12 | Whether Rummery & Niranjan 1994 coined "SARSA". **Partly resolved (2026-10-17, AD):** Sutton & Barto 2018's Chapter 6 bibliographical notes say the report introduced the algorithm as "Modified Connectionist Q-learning" and that the name Sarsa was introduced by Sutton (1996). Module 3 cites the report for the algorithm only. | Read the technical report before claiming anything more. | AD | Phase 5 |
| B29 | Module 3's design numbers came from a scratch design scan on the author's laptop. **Now replaced by records (2026-10-17):** the thresholds come from 100-seed threshold-protocol runs at both budgets (arm64, and Linux x86-64 bit-identical; `runs/2026-10-17-local-threshold-m03-*`), and the figures and the planted-bug returns from three recorded experiments (`runs/2026-10-17-local-m03-cliff-contrast-exp.json`, `runs/2026-10-17-local-m03-nstep-dial-exp.json`, `runs/2026-10-17-local-m03-suspect-returns-exp.json`). The suspect-returns record corrects the scratch scan: the suspect earns more over the first 100 episodes on 10 of 10 seeds, and over the last 500 it is a tie (higher on 4 of 10). The page quotes only exact computations and record numbers. Still open: the colab-sim protocol runs (C11). | TE: colab-sim protocol runs | TE | Phase 3, Day 1 (M3) |
| B30 | Module 3 says every-visit MC is biased on finite data and consistent, with a one-state computation on the page; the consistency proof is credited by Sutton & Barto §5.1 to Singh & Sutton 1996, which we have not opened. | Open Singh & Sutton 1996 if the claim is ever cited to it. | AD | Phase 5 |

**Citations: other sources**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B13 | Page ranges from secondary sources (the Bradley–Terry end page; Sutton et al. 1999; Konda & Tsitsiklis 1999; Christiano et al. 2017; Thrun & Schwartz 1993) | Omitted from the bib until verified | AD | Phase 5 |
| B14 | Equations reconstructed from extracted PDF text (Dudík DR Eq. 1; SNIPS Eq. 7; SAC-v2 target entropy; Ouyang Eq. 2's pretraining term; DeepSeekMath symbols) | Re-check against the typeset PDF before quoting | AD | When quoted |
| B15 | ~~A source for "BC is competitive on expert data"~~ **Resolved:** Kumar, Hong, Singh & Levine 2022, §4.2, Thm 4.3, Practical Observation 4.1. Its venue (ICLR 2022) is still to be confirmed on the proceedings page. | — | AD | Phase 3 (Day 5) |
| B16 | ~~The probe-environments source~~ **Resolved:** Jones 2021 blog post, "Use probe environments" (5 probes) | — | — | — |
| B17 | Section numbers in Agarwal et al. 2021 come from arXiv v4, not the camera-ready | Check the camera-ready | AD | Phase 3 (Day 3) |
| B18 | Gymnasium paper title. The arXiv v4 PDF says "A **Standardized** Interface" (opened 2026-10-10, AD); `references.bib` follows the proceedings BibTeX, "A Standard Interface", which was not re-opened. Which title does the bib cite? | Re-open the NeurIPS 2025 proceedings page | AD | Phase 5 |

**Measurements**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B19 | Whether CPU is as fast as T4 for the small-MLP labs | Timed runs, recorded | TE | Phase 3 |
| B20 | Every lab minute budget in `PLAN.md` | Phase 3 timed runs (compute) plus a human pilot with 2–3 people | PE | Phase 3 and release |

**Module 4 (bandits)**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B31 | Hoeffding 1963 itself was not opened (JSTOR, paywalled), so no theorem number is cited. Module 4 states the inequality from Lattimore and Szepesvári 2020, §5.4, Note 4, Eq. 5.8, and from Auer et al. 2002, Fact 1, and names Hoeffding (1963) only as its origin. | Open the JASA paper (or the 1962 UNC mimeo) before citing a theorem number | AD | Phase 5 |
| B32 | The numbers in Module 4's two After-the-lab figures and knowledge check 4.8 (seeds 0–19) come from the academic director's design scan: scratch code equal to the lab's solutions, run on arm64. They are deterministic, so the technical expert's recorded experiments must reproduce them exactly; the page already names the records, `runs/2026-10-18-local-m04-regret-curves-exp.json` and `runs/2026-10-18-local-m04-scale-exp.json`, which do not exist yet. | TE files both records; the figure script asserts every quoted number; any difference goes back to the AD | TE / AD | Before the Day 2 merge |
| B33 | The `predict_answer` entries for Module 4 quote seat 0 (seeds 1000–1019) from a scratch run of the reference lab, and frequencies over 10 seats (200 seeds) from the design scan. Neither is a filed record. | TE files the worked notebook's run record; a 100-seat scan of the Exercise 4 verdicts and of the Exercise 5 directions | TE | Before the Day 2 merge |
| B34 | How often a seat's own 20-seed run shows UCB1 × 10's mean regret going *down* (checkpoint 5 compares the participant's prediction with the reference directions, not with the seat's run). The design scan found the mean up on 10 of 10 seats. | The 100-seat scan of B33 | TE | Before the Day 2 merge |
| B35 | Module 4's live compute: about 37 s on an Apple M1 Pro for the whole notebook with solutions bound, stretch included (a scratch harness, not a filed run). Not measured on Colab; the estimate there is under 2 minutes. | Worked and QUICK notebook records; a Colab run | TE / user | Before release |

## C. Decisions still open

| # | Question | Default | Owner | Needed by |
|---|---|---|---|---|
| C1 | How do AWS cohort participants reach their instance without long-lived credentials? Options: (a) IAM Identity Center with `ssm:StartSession` scoped by tag; (b) the facilitator runs port forwarding; (c) SageMaker Studio presigned URLs. | (a) for SSO teams; otherwise (c) | user | Phase 1 |
| C2 | Which AWS account, region and G-instance quota for the human end-to-end test? The local `aws` CLI must be reinstalled (it currently fails with "exec format error", probably an x86 binary on arm64), and the Session Manager plugin installed. | us-east-1; quota of 4 vCPUs per participant | user | Before release |
| C3 | Do we need a Genial Labs mirror or branded profile (the reference has one)? | No | user | Phase 1 |
| C4 | License for the synthetic datasets we generate | CC BY 4.0 | user | Phase 1 |
| C5 | Should the brief be committed (for example as `docs/brief.md`)? | No. `PLAN.md` captures it, and a copy is kept outside the repo. | user | Phase 1 |
| C6 | Who records the teaching-eligible Colab runs? The builder can't run Colab. | You or a delegate, using the recorder (about 2–3 h for all notebooks) | user | Before release |
| C7 | Environment for capstone track 15b | `Acrobot-v1` | PE / AD | Phase 4 |
| C8 | The reference repo moved from `3022908` (read in Phase 0) to `af3dabe` (2026-10-09). Should Phase 1 re-read anything we copy? | Yes: the harness semantics, `check_browser.py` and `disclosure.lua` | lead | Phase 1 |
| C9 | Should we report TRL's CPU-incompatible fused LM head upstream (see A11)? | Yes, after Phase 1 confirms it on Linux | TE | Phase 1 |
| C10 | Turn on the repository setting "Allow GitHub Actions to create and approve pull requests" so `health.yml` can open Colab-freeze PRs? It is off today; until then the job pushes the branch, links a compare page in its summary and fails. | Yes | user | Before the first upstream freeze change |
| C11 | The threshold protocol runs in colab-sim and on arm64, but a check holds one threshold per budget. Which platform sets it? | colab-sim sets it; the arm64 record must show zero solution failures and zero passing mutants at that threshold | TE | Phase 3, Day 1 (M3) |
| C12 | Where do participants send their Module 0 run record, and who collects it (email, form or shared folder)? Module 0's objective says "submit", but no channel exists; the page and notebook say "keep the file; your instructor will say where to send it". | — | user | Before pre-work is sent out |
| C16 | Module 4's §3 activity. PLAN §4 says "demo 3"; the draft has Check yourself 3.1 (the UCB1 index by hand) instead, because an OJS demo on the page would request cdn.jsdelivr.net, which `scripts/site_check_allow.yml` allows only for Module 1's page. | Keep the Check yourself; the UI expert may add a demo later, with an allow-list entry from the lead | PE / UI / lead | Day 2 review |
| C17 | Module 4's live split departs from PLAN §4 to match the reading estimate (110 words a minute plus 0.5 minute per display): §1 6, §2 6, §3 10, §4 9, §5 6, §6 3 minutes, still 30 exposition + 10 activity. PLAN has 5, 6, 11, 8, 5, 5. | PE confirms; the lead updates PLAN §4 | PE / lead | Day 2 review |
| C18 | Module 4's checkpoints depart from PLAN §4's table. 1: "greedy is linear on the trap instance" is an exact replay of a recorded greedy run, not a threshold. 3: a third checkpoint, 3.2 (choice shares on fixed posteriors), because no regret threshold separates "play the posterior mean" from the solution (design scan). 4 and 5: the participant writes functions, `verdict` and `scale_directions`, not sentences. 6: an exact replay of UCB1's recorded run, not "regret below threshold"; UCB1 is deterministic given the reward stack, so the replay is strictly stronger. | Default: as drafted; the lead updates PLAN §4 | lead / TE | Day 2 review |
| C19 | Appendix A's checkable answer for M4 ("Up. The bonus is now relatively small, so it behaves nearly greedily") holds for the mean regret only. In the design scan the IQM can fall (seeds 0–19: mean 112.2 → 195.1, IQM 113.8 → 90.1), most single seeds improve, and the 20-seed tie rule calls a tie on 9 of 10 seats. Module 4 defines Exercise 5's "up" on the mean and makes the tie the Explain's point. | The lead updates Appendix A to "Up in the mean; many seeds improve and the tie rule calls a tie" | lead | Day 2 review |

## Resolved

| # | Date | Decision |
|---|---|---|
| R1 | 2026-10-09 | Repo `project-delphi/practical-rl`, public, published to GitHub Pages. |
| R2 | 2026-10-09 | AWS infrastructure as code is CloudFormation with AWS CLI wrapper scripts. The end-to-end run is a human checklist task. |
| R3 | 2026-10-09 | Module 12's core runs on CPU; the T4 training run is the stretch. |
| R4 | 2026-10-09 | Deep-RL labs use "recorded bands + your seed", with at most 10–12 minutes of live compute per lab. |
| R5 | 2026-10-09 | American English. |
| R6 | 2026-10-09 | Upgrade local Quarto to 1.10.19, matching the version pinned in CI. |
| R7 | 2026-10-09 | Persona agents (AD, PE, TE, UI) plus a fresh-context verifier. Git worktrees for parallel work. |
| R8 | 2026-10-09 | d3rlpy dropped; MABWiser kept as an optional parity cell (TE review). |
| R9 | 2026-10-09 | MBIE-EB form, the BC-on-expert source, the probe-environments source and the Base DLAMI SSM parameter verified (AD and TE reviews). |
| R10 | 2026-10-09 | Phase 1 defaults applied for A1–A8, A10, A12–A13 and C3–C5 (you said "go"); revisit any at the Phase 1 gate. Probe environments stay out of `prl` (Module 9 participants write them). |
| R11 | 2026-10-17 | C13 (Module 2's planted bug): keep PLAN's "stop on the mean change" bug. The pedagogy review found the slippery cliff's factor-2 symptom too quiet, so Lab 2's Exercise 5 runs the suspect on three models: the shop (invisible, the lesson), the slippery cliff (4 sweeps early, residual about 2× the guarantee) and a 100-state model with 99 settled states (90 sweeps early, residual about 98× the guarantee). The participant writes the optimality residual (checkpoint 5.1) before fixing the suspect (5.2). |
| R12 | 2026-10-17 | C14 (Module 3 mutants a budgeted RMS cannot separate from the solution): resolved by `exact_only=True` mutants (change B). The threshold protocol leaves them out; verify mode runs them against the exact checkpoints 1.1 and 2a.1. |
| R13 | 2026-10-17 | C15 (Module 3's planted-bug checkpoint): the lead accepts the deviation on the pedagogy expert's verdict. Checkpoint 6 requires the last move's Q to equal its reward (−1) and max Q(start) to equal V*(start), from Q₀ = 10; Q(goal, ·) stays at Q₀ after a correct fix, harmlessly. The lead updates PLAN §4. |
| R14 | 2026-10-18 | `references.bib`: the `thompson1933` annote had been a copy of `watkins1992q`'s (it said Module 3 cited Thompson 1933 for Q-learning's convergence). Corrected by the academic director; the paper itself is still unopened and is cited only as the origin of Thompson sampling. |
