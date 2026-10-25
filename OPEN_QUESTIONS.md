# Open questions

This file lists:
- decisions that still need an owner;
- facts we could not verify;
- places where this repo departs from the brief.

Each entry gives our working default, who decides, and when. When an entry is resolved, move it to **Resolved** with the date and the decision.

**Owners:** user (you), AD (academic director), PE (pedagogy expert), TE (technical expert), UI (UI expert), lead (the main session).

Last updated 2026-10-17 (Phase 3, Day 1: Modules 2 and 3); Module 9 items added 2026-10-24.

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
| B11 | RiverSwim's origin and transition probabilities. The 2008 paper says it is "taken from" Strehl & Littman, ICTAI 2004 (not opened). **Partly resolved (2026-10-18, AD):** Fig. 1 of the 2008 paper was read from the image in its PDF (interior right action 0.3/0.6/0.1, left end 0.3 right, right end holds with 0.3 paying 10,000, left reward 5, start in state 1 or 2), and so was Osband et al. 2013's Fig. 1 (interior 0.35/0.6/0.05 as ours, left end 0.6 right, right end holds with 0.6 paying 1, left reward 5/1000). Ours keeps Osband's interior and rewards and changes the two ends; Module 6's page states all three and says the numbers are ours. Still open: the 2004 paper. | Open the 2004 paper before saying where the chain comes from | AD | Phase 5 |
| B12 | Whether Rummery & Niranjan 1994 coined "SARSA". **Partly resolved (2026-10-17, AD):** Sutton & Barto 2018's Chapter 6 bibliographical notes say the report introduced the algorithm as "Modified Connectionist Q-learning" and that the name Sarsa was introduced by Sutton (1996). Module 3 cites the report for the algorithm only. | Read the technical report before claiming anything more. | AD | Phase 5 |
| B29 | Module 3's design numbers came from a scratch design scan on the author's laptop. **Now replaced by records (2026-10-17):** the thresholds come from 100-seed threshold-protocol runs at both budgets (arm64, and Linux x86-64 bit-identical; `runs/2026-10-17-local-threshold-m03-*`), and the figures and the planted-bug returns from three recorded experiments (`runs/2026-10-17-local-m03-cliff-contrast-exp.json`, `runs/2026-10-17-local-m03-nstep-dial-exp.json`, `runs/2026-10-17-local-m03-suspect-returns-exp.json`). The suspect-returns record corrects the scratch scan: the suspect earns more over the first 100 episodes on 10 of 10 seeds, and over the last 500 it is a tie (higher on 4 of 10). The page quotes only exact computations and record numbers. Still open: the colab-sim protocol runs (C11). | TE: colab-sim protocol runs | TE | Phase 3, Day 1 (M3) |
| B50 | ~~Module 6's statements about its learners rest on a scratch design scan~~ **Resolved (2026-10-18):** `runs/2026-10-18-local-m06-regret-seeds-exp.json` (seeds 0–9) supports every one, `scripts/make_figures_m06.py` asserts them, and the page, the `predict_answer` entries and knowledge check 6.2 now cite the record. The colleague's row, rerun with Exercise 7's second-pass stub, is in `runs/2026-10-18-local-m06-regret-seeds-exp-2.json` (the other rows are identical), which the page and `_variables.yml` now cite. | — | — | — |
| B30 | Module 3 says every-visit MC is biased on finite data and consistent, with a one-state computation on the page; the consistency proof is credited by Sutton & Barto §5.1 to Singh & Sutton 1996, which we have not opened. | Open Singh & Sutton 1996 if the claim is ever cited to it. | AD | Phase 5 |

**Citations: other sources**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B13 | Page ranges from secondary sources (the Bradley–Terry end page; Sutton et al. 1999; Konda & Tsitsiklis 1999; Christiano et al. 2017; Thrun & Schwartz 1993) | Omitted from the bib until verified | AD | Phase 5 |
| B14 | Equations reconstructed from extracted PDF text (Dudík DR Eq. 1; SNIPS Eq. 7; SAC-v2 target entropy; Ouyang Eq. 2's pretraining term; DeepSeekMath symbols) | Re-check against the typeset PDF before quoting | AD | When quoted |
| B15 | ~~A source for "BC is competitive on expert data"~~ **Resolved:** Kumar, Hong, Singh & Levine 2022, §4.2, Thm 4.3, Practical Observation 4.1. Its venue (ICLR 2022) is still to be confirmed on the proceedings page. | — | AD | Phase 3 (Day 5) |
| B16 | ~~The probe-environments source~~ **Resolved:** Jones 2021 blog post, "Use probe environments" (5 probes) | — | — | — |
| B17 | ~~Section numbers in Agarwal et al. 2021 come from arXiv v4, not the camera-ready~~ **Resolved (2026-10-24, AD, Module 9):** the NeurIPS 2021 proceedings PDF has the same numbers: §4.1 stratified bootstrap CIs (pp. 5–6), §4.2 performance profiles, §4.3 robust aggregate metrics and the IQM (pp. 7–8); recorded in `references.bib` | — | — | — |
| B51 | Burda et al. 2019 (RND) was read on arXiv v1 (30 Oct 2018); Module 6 cites its §2.1, §2.2 and §2.2.1 by v1's numbers, not the ICLR 2019 camera-ready's | Open the camera-ready (OpenReview was blocked by a bot check on 2026-10-09) | AD | Phase 5 |
| B52 | Bellemare et al. 2016 was read on arXiv v2 and Osband et al. 2013 on arXiv v5, not the NeurIPS proceedings PDFs; section, equation, figure and table numbers come from those versions. Bellemare et al.'s §4 reports a negative result of Kolter and Ng (2009) on bonuses at most beta/N, which we have not opened; Module 6 attributes it to Bellemare et al.'s report | Compare with the proceedings PDFs; open Kolter and Ng before citing it directly | AD | Phase 5 |
| B18 | Gymnasium paper title. The arXiv v4 PDF says "A **Standardized** Interface" (opened 2026-10-10, AD); `references.bib` follows the proceedings BibTeX, "A Standard Interface", which was not re-opened. Which title does the bib cite? | Re-open the NeurIPS 2025 proceedings page | AD | Phase 5 |
| B40 | Li et al. 2010's Eq. 4 (the LinUCB confidence bound with $\alpha = 1 + \sqrt{\ln(2/\delta)/2}$) is stated with "it can be shown [27]" (Walsh et al. 2009, not opened), for responses independent given the design. Module 5 quotes the value of $\alpha$ and Li et al.'s warning that it may be conservatively large, and does not claim the bound holds for adaptively collected data; for that it cites Lattimore and Szepesvári Thm 20.5. | Open Walsh et al. 2009 before saying more about Eq. 4 | AD | Phase 5 |

**Measurements**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B19 | Whether CPU is as fast as T4 for the small-MLP labs | Timed runs, recorded | TE | Phase 3 |
| B20 | Every lab minute budget in `PLAN.md` | Phase 3 timed runs (compute) plus a human pilot with 2–3 people | PE | Phase 3 and release |
| B41 | Module 5's quoted run numbers are exact runs of the lab's code on the author's machine (macOS arm64, Accelerate), not records: the `predict_answer` entries (seat 0: 486 and 464 clicks, four items shown once, 0.0223 of 20000 draws, 9.3% non-greedy rows, mean weights 1.03 and 1.06, the suspect's 0.62) and the page's After-the-lab fold (verdict tallies over seats 0–29 at 3000 rounds; seat 0's end widths 0.25 and 0.27). Trajectories depend on continuous score margins and `default_rng`, so Linux (OpenBLAS) should reproduce them, but that is not verified. **Now recorded (2026-10-18, TE):** `runs/2026-10-18-local-m05-seats-exp.json` (arm64) re-derives every seat-level number, and `scripts/make_figures_m05.py` asserts them (Exercise 5's are checked but held until the warm-launch pass); seat 0's new_policy weight on the correct log is 1.05 (1.0545), not 1.06, now corrected in the `predict_answer`. Linux x86-64 (`python:3.13-slim` amd64 under Rosetta, the lock's numpy 2.5.3, OpenBLAS 0.3.34): `--reproduce` matches every discrete value of m05-seats, m05-launch and m05-regimes; the floats are not bit-identical (about two thirds are, the rest differ by at most 3.4e-13), and the smallest argmax margin is 3.3e-9, so no choice depends on rounding. Not run on Colab itself. | TE: one recorded experiment over seats 0–29 (Exercise 5's sweep and the seat-0 cells), run in arm64 and in the Linux container; the figure script asserts the quoted numbers | TE | Phase 3 (Day 2, M5) |
| B42 | Module 5's MABWiser 2.7.4 parity stretch ran once, on the AD's machine, in a `uv run --with mabwiser==2.7.4` overlay (Python 3.13.2, NumPy 2.5.3, scikit-learn 1.9.1; not a record): seat 0's stream gave max differences of 1.4e-16 in `A_inv`, 0 in `Xty` against `b`, and 7.1e-15 in the scores. MABWiser is not in `uv.lock`, and the cell skips when the import fails. Read from MABWiser 2.7.4's `linear.py` (installed in a throwaway environment): `_RidgeRegression.init` sets `A_inv = A.copy()`, which equals $A^{-1}$ only when `l2_lambda = 1` (the stretch uses 1); `fit` recomputes `A_inv` with `np.linalg.inv`; LinTS draws with `rng.multivariate_normal(beta, alpha**2 * A_inv)`. The cell reads `mab._imp.arm_to_model`, a private attribute. **Resolved (2026-10-18, TE):** Lab 5 pins `mabwiser==2.7.4` in its front matter. The setup cell installs it on Colab, where colab-compat shows it adds only itself to both freezes (its dependencies are preinstalled), and colab-sim runs the stretch in worked and verify modes on every PR. Elsewhere the cell skips when the import fails. Runs (not records): arm64, `uv run --with mabwiser==2.7.4`, QUICK: 1.4e-16, 0, 5.0e-15; Linux x86-64 container with Colab's versions (numpy 2.1.3, scipy 1.16.3, pandas 2.2.3, scikit-learn 1.6.1, seaborn 0.13.2), QUICK: 1.8e-16, 2.5e-14, 5.6e-15. MABWiser is not in `uv.lock`, so a local run warns and skips; adding it to the `notebooks` group is a request to the lead. | TE: run the stretch with MABWiser 2.7.4 on Python 3.13 and NumPy 2; decide the pin and whether CI runs it | TE | Phase 3 (Day 2, M5) |
| B90 | `scripts/threshold_protocol.py` kept per-seed values by mutant id, so in a check of two functions whose mutants share an id, the second overwrote the first. Module 3's `check_cliff_contrast` has two `random` mutants (of `sarsa` and of `q_learning`), so its two records (`runs/2026-10-17-local-threshold-m03-check_cliff_contrast-{live,quick}-n100-exp.json`) hold only the `q_learning` one. **Checked (2026-10-24, lead; arm64, the live budget, which equals the quick one; not a record):** over seeds 0–99 the overwritten `sarsa` random scores −5062.75 to −4732.71 against the threshold 13.0, and the strongest wrong version is still `max-target` (at most 7.11), so the threshold stands. The protocol now refuses duplicate ids among a check's wrong versions; rerunning this check first needs distinct ids, then fresh records. Log: `prl-wt/_notes/m03_random_sarsa.log` (outside the repo). | Rename the two ids and refile both records the next time Module 3's checks change | TE | When Module 3 next changes |

**Module 4 (bandits)**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B31 | Hoeffding 1963 itself was not opened (JSTOR, paywalled), so no theorem number is cited. Module 4 states the inequality from Lattimore and Szepesvári 2020, §5.4, Note 4, Eq. 5.8, and from Auer et al. 2002, Fact 1, and names Hoeffding (1963) only as its origin. | Open the JASA paper (or the 1962 UNC mimeo) before citing a theorem number | AD | Phase 5 |
| B32 | ~~The numbers in Module 4's After-the-lab figures and knowledge check 4.8 came from the academic director's design scan.~~ **Resolved (2026-10-24):** the technical expert filed `runs/2026-10-18-local-m04-regret-curves-exp.json` and `runs/2026-10-18-local-m04-scale-exp.json`; every quoted number reproduces, `scripts/make_figures_m04.py` asserts each one, and the Linux container reproduced both records exactly. | — | — | — |
| B33 | ~~The `predict_answer` entries quoted seat 0 from a scratch run and frequencies from a 10-seat design scan.~~ **Resolved (2026-10-24):** they now cite the 100-seat record `runs/2026-10-18-local-m04-seats-exp.json` (seats 0–99, the lab's own code at the live budget) and the worked notebook record `runs/2026-10-18-local-04-bandits-27090ede.json`. | — | — | — |
| B34 | ~~How often a seat's 20-seed run shows UCB1 × 10's mean going down.~~ **Resolved (2026-10-24):** on 3 of 100 seats (`runs/2026-10-18-local-m04-seats-exp.json`). Checkpoint 5.2 now grades only "same" against "changes" (lead decision 7), so no seat's own run can contradict it. | — | — | — |
| B35 | Module 4's live compute on Colab is not measured. The worked notebook record took 42.7 s live and 30.4 s QUICK on an Apple M1 Pro under load (`runs/2026-10-18-local-04-bandits-27090ede.json`, `-2.json`, before the second pass). The second pass adds 800,000 steps to Exercise 4 (about 8 s on the same laptop, an estimate). Estimate for Colab: under 3 minutes for the whole notebook, no cell near 3 minutes. | A Colab run of the worked notebook, recorded | TE / user | Before release |

**Module 8 (DQN)**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B70 | ~~Every Module 8 number that comes from a run is pending a record.~~ **Mostly resolved (2026-10-25):** the band records `runs/2026-10-24-local-m08-{dqn,ddqn,sb3}-{live,quick}-exp.json` (seeds 0–4) settle the page's After-the-lab folds and the `predict_answer` entries of Exercises 3, 5 and 6: Q(s₀) below G₀ on every seed at both budgets for all three; Q(s₀) 0.83–2.52 above (sync lag) at 50,000 steps; Double DQN's Q(s₀) below DQN's at the same seed on 5 of 5 seeds live and 4 of 5 QUICK; reference against SB3 a tie at both budgets. `scripts/make_figures_m08.py` asserts them. **Still open:** a per-seat record (`m08-seats`, seats 0–19 at the live budget, DQN and Double DQN) so the debrief and PLAN Appendix A can quote tallies over more than 5 pairs (pedagogy review, finding 19). | TE: `m08-seats` and its assertions | TE / AD | Phase 3 (Day 3) |
| B71 | Module 8's live compute on Colab is not measured. Laptop (Apple M1 Pro, not Colab), the TE's notebook runs before the second pass: worked live 30.2 s of cells (`ex4-run` 11.4 s, checkpoint 4.2 4.1 s, `ex5-run` 8.9 s), QUICK 15.7 s. Estimate for Colab at PLAN §6's assumed 3–5×: 30–60 s per training cell, checkpoint 3.2 (20,000 steps) 12–20 s, about 1.5–2.5 minutes of compute in all. | A recorded Colab run of the worked notebook | TE / user | Before release |
| B72 | Van Hasselt et al. 2016 was read on arXiv v3 (8 Dec 2015), not the AAAI 2016 proceedings PDF; its sections are unnumbered, Theorem 2 is in the arXiv appendix, and Module 8 cites both by v3's titles. | Open the AAAI proceedings PDF and check that Theorem 1's wording, the section titles and the appendix match | AD | Phase 5 |
| B73 | Wang et al. (dueling networks) is cited as arXiv 1511.06581v3; its ICML 2016 publication is not verified (the arXiv record names no venue). | Open the PMLR proceedings page | AD | Phase 5 |
| B74 | ~~SB3 2.9.0 with gymnasium 1.4.0 at the matched configuration ran only in scratch runs, and the differences came from reading the source.~~ **Resolved (2026-10-25, TE):** the SB3 band records ran it at both budgets; their `args.unmatched` lists every difference (Exercise 6 prints it), and `per_seed.truncations` shows SB3 bootstrapping through the 500-step truncation, as the lab does. | — | — | — |
| B75 | Lab 8's stretch has not run on Colab. In the TE's Linux container, `box2d==2.3.10` installs from its cp313 wheel, but `gym.make("LunarLander-v3")` also needs pygame, which Colab preinstalls and `uv.lock` does not; the cell now skips when either import fails, and its local install line names both. 100,000 steps took 53 s under amd64 emulation (not a Colab time). | One run of the stretch on Colab | TE / user | Before release |

**Module 9 (debugging and evaluation)**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B80 | Every Module 9 number from a run is pending a record: the healthy run's signal values (Exercises 1–2), the table of runs A–D (Exercise 3), the probe values (Exercise 4), the sweep (Exercise 5) and the live seed's band (Exercise 6); numbering of the second pass (lead decision 12). The first pass used a scratch DQN close to Module 8's draft configuration (`_notes/m09-ad/`, not a record); the page and the `predict_answer` entries quote only exact arithmetic or say "pending record". | The m09 experiment records and `scripts/make_figures_m09.py` asserting each quoted number | TE | Phase 3 (Day 3, M9) |
| B81 | Whether each signal separates its run on Module 8's final shared DQN. In the scratch scan (5 seeds each, 50,000 steps): run A's relative TD error was about 3× the healthy one (0.030–0.037 against 0.010), run C's Q drift was in the thousands against at most about 170 for every other run, `gather_check` was 0 except on run B (0.97–0.98), and `target_lag` was 0 except on run D (49,500). A second scratch scan (3 seeds) found that × 100 does not reliably break learning when the base learning rate is 2.5e-4 or 1e-4. **TE, 2026-10-25, on Module 8's shared DQN (labs/shared), seeds 0-2, 50,000 steps, every margin asserted on every seed:** all hold except one. Run A (lr × 100), seed 0, ends with Q at the start states 132.67, above 99.34 (above it at every log point from 32,000 steps; seeds 1 and 2: 66.83 and 64.52), so Exercise 3's elimination of "mask" for run A by "Q in range" fails on one seed. What still separates A from C: C's Q drift (last 5) is 1,611.76 to 3,655.13 against at most 94.15 for every other run (A seed 0); probe 1. Also: `gather_check` B 0.958-0.969, others 0.0; `target_lag` D 49,500, others 0; rewards all 1.0; greedy return at 50,000 A-D 9.2-10.4 against healthy 115.1-154.7; the wording check holds (A's lowest relative TD error 0.0529 = 4.95 × the highest of healthy, C, D; B's is 0.105-0.106). **Lead's decision 15 (2026-10-25), applied by the TE:** that margin is replaced by "run C's lowest final Q is at least 10 times every other run's" (1805.99 against 132.67: 13.6 times); every margin then holds. Record `runs/2026-10-25-local-m09-broken-runs-exp.json`; dataset `m09_broken_runs` built. The director rewords the sentences that relied on "<= 99.34" at sign-off (list in `_notes/m09-te-notes.md`). | The generator asserts a margin for each signal on every seed (handoff §3), plus one wording check from the pedagogy review: run A's lowest second-half relative TD error ≥ 1.5 × the highest over the healthy, C and D seeds (scratch: 2.3×). A failed margin goes back to the AD; a failed wording check changes only the Explain, since run A is diagnosed by elimination | TE / AD | Before the datasets ship |
| B82 | Agarwal et al. 2021's Figure A.18 (the single-task bootstrap with few runs) was not opened; Module 9 quotes only §4.1's sentence that cites it. | Open the appendix before saying more | AD | Phase 5 |
| B83 | rliable's archive date (2025-10-15) comes from the Phase 0 entry and was not re-checked; Module 9 says only that the repository is archived. | Re-open the repository page | AD | Phase 5 |

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
| C32 | ~~`tests/test_shared.py` rejected provided code in a shared file.~~ **Resolved (2026-10-24, lead, `b3e02f2`):** a shared file that its origin lab includes is its own source. | — | lead | — |
| C33 | ~~Checkpoint 4.2's budget.~~ **Resolved (2026-10-25, decision 14 and the TE's protocol):** checkpoint 3.2 (numbered 4.2 before the second pass) trains 20,000 steps at both budgets; its metric is the best greedy evaluation minus the first; threshold 91.0 (`runs/2026-10-25-local-threshold-m08-check_train_dqn-{live,quick}-n200-exp.json`). | — | TE / PE | — |
| C34 | ~~Keep `target_sync = 500`, with Q(s₀) below G₀?~~ **Resolved (2026-10-25, decision 14):** keep the configuration. Exercise 5's Predict keeps the sign of Q(s₀) − G₀ and asks, second, whether Double DQN's Q(s₀) is higher or lower than DQN's at the same seed; §5 says which comparison shows which force. The lead rewrites PLAN Appendix A's M8 answer from the records (B70). | — | lead / user | — |
| C35 | ~~Should Exercise 5 train a live Double DQN seed?~~ **Resolved (2026-10-25, decision 14):** keep it (now Exercise 4), on the review's three conditions: the room code records `ddqn_q0`; Exercise 5 asks about the same-seed Q(s₀) difference; the single-seed return plot is gone. | — | PE | — |
| C36 | Run A is "the reference learning rate × 100". It breaks learning only if the reference rate is large enough: in the scratch scans, × 100 broke learning from 1e-3 on 5 of 5 seeds, but not reliably from 2.5e-4 or 1e-4 (3 seeds each). | Keep × 100 while Module 8's rate is 1e-3 (its draft). If Module 8 lowers it, run A uses a fixed large rate and the page says so | AD / lead | When Module 8's configuration is fixed |
| C37 | Exercise 6's band (Exercise 8 in the first pass): a dedicated 10-seed `m09_reference` band (the sweep's × 1 row at the live budget plus a QUICK record), or Module 8's band of the same configuration. | Dedicated, so the band has 10 seeds and the same seeds as the sweep | TE / lead | Before the bands are exported |
| C38 | Exercise 6's checkpoints: 6.1 is exact (`band_position`, written by the participant, pedagogy review finding 11); 6.2 compares the return the live cell already computed with a protocol threshold, or would be a `StochasticCheck` that trains again (doubling the live compute). **TE, 2026-10-25:** QUICK has its threshold, 61.0 (100 seeds, solution p1 92.54, random p99 30.32; `runs/2026-10-25-local-threshold-m09-check_live_return-quick-n100-exp-2.json`; its random-play mutant is in `labs/mutants/protocol/m09.py`, run by `scripts/protocol_m09.py`, since `labs/mutants/` holds only mutants of functions participants write). **The live budget has none:** over seeds 0-99 the protocol refused (solution p1 26.75 below random's p99 30.32): seeds 94 and 78 end at 12.3 and 26.9 after earlier evaluations of up to 286.9 and 224.9, so one run's last greedy evaluation at 50,000 steps can collapse (Module 8 saw the same). Seeds 0-19 alone gave 69.0, which seeds 78 and 94 contradict; not used. A scratch scan of seeds 0-99 (not a record): the best of the run's 10 evaluations is at least 183.0 on every seed, the mean of all 10 at least 126.8. Proposal: 6.2 checks the best evaluation of the live run (a one-line lab change), with new protocol records at both budgets. Until then the live worked and verify modes stop at 6.2. | 6.2 checks the computed value; the threshold comes from the protocol at both budgets | TE | Phase 3 (Day 3, M9) |
| C39 | ~~The diagnosis checkpoints are multiple choice, and the signal names nearly name the cause.~~ **Resolved (2026-10-24, lead decision 12, on the pedagogy review):** the pattern design. Exercise 1 writes the two exact checks and Exercise 2 the two scale signals, both on the healthy run; Exercise 3 diagnoses runs A–D from one table of all four signals (checkpoints 3.1–3.4, one per run). The four per-run diagnosis cells are gone; run A is diagnosed by elimination. | — | — | — |

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
| R15 | 2026-10-24 | C16 and C17 (Module 4's briefing): the pedagogy review keeps Check yourself 3.1 in place of PLAN's "demo 3" (the hand computation prepares Exercises 2 and 6 and needs no CDN allow-list entry; 2 ln 100 ≈ 9.21 is now given in the prompt) and confirms the live split 6 / 6 / 10 / 9 / 6 / 3, 30 + 10 minutes, within ±0.6 minute of the reading estimate in every section. The lead updates PLAN §4. |
| R16 | 2026-10-24 | C18 and C19 (Module 4's checkpoints): the exact replays stay (the review calls them the strongest checkpoint design on Day 2). Checkpoint 5 no longer grades the direction of the mean (lead decision 7): Exercise 5's participant writes `ucb1_for_range` (checkpoint 5.1, exact) and `scale_directions` answers "same" or "changes" per method (checkpoint 5.2); whether UCB1 × 10's regret goes up or down, and the tie, move to the Explain. Over 100 seats the mean rises on 97 and falls on 3, the IQM falls on 41 and the tie rule calls a tie on 96 (`runs/2026-10-18-local-m04-seats-exp.json`). The lead rewords PLAN Appendix A's M4 row and §4's table at integration (lab minutes 11 / 9 / 8 / 9 / 8 / 7 = 52). |
| R17 | 2026-10-24 | C20 and C21 (Module 5's notation), by the lead's Day 2 decisions 1 and 2. Bandits keep $K$ for the number of arms (items), numbered from 0; Module 5 bridges it once as Module 4's $\lvert \mathcal{A} \rvert$, and the GRPO group size takes a subscript when Module 12 is written. Module 5's pseudo-regret is Module 4's $\mathrm{Reg}_T$ over rounds $t = 0, \dots, T - 1$; its $R_T$ row is removed from `notation.qmd`. The learner's linear parameter is $w$ ($w^*_a$, $\hat w_a$, $\tilde w_a$; `w_hat` in code); $\theta_a$ and $\theta^{\text{alt}}_a$ stay the simulator's. |
| R18 | 2026-10-24 | C22 (Module 5's checks). The technical expert built `prl.checks.m05` with fixtures (commits `2d6bbfb`, `63971da`); verify mode rejects all 34 wrong versions. Comparing class methods on the page with the lab is PR #19's change to `tests/test_page_code.py` (by qualified name). |
| R19 | 2026-10-24 | C23 (Module 5's launch), by the lead's Day 2 decisions 4 and 8. Exercise 5 keeps the one-pass launch in three regimes and adds a fourth condition, the well-specified regime after 3 launch passes (30 rounds), chosen from `runs/2026-10-18-local-m05-launch-exp.json`: with 3 passes the well-specified LinUCB call is a tie at 29 of 30 seats, while misspecification still separates the methods at 12 of 30. The lab's `run` takes `passes`; the page's fold and Gotcha now say that greedy's handicap is mostly the one-impression start. |
| R20 | 2026-10-18 | C24 (Module 6's untried pair in MBIE-EB): keep Module 2's self-loop default plus the bonus of one visit, which makes the pair worth at least c/(1 − γ). The pedagogy review endorsed it; Strehl and Littman's own convention, 1/(1 − γ) for such a pair, now sits in a reference block on the page. |
| R21 | 2026-10-18 | C25 (Module 6's Exercise 6): the participant writes `regret(states, actions, R, opt_rewards)`, cumulative regret against the DP optimum, with an exact checkpoint; the run compares the learners with `prl.evaluate.compare`. `verdict` stays Module 4's exercise (pedagogy review; lead, Day 2 decision 3). |
| R22 | 2026-10-18 | C26 (Module 6's re-planning rule, chain length and c): keep re-planning when a count reaches a power of 2, the 10-state river, and c = 1 = r_max, untuned; the page states all three (pedagogy review). |
