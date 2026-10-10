# Open questions

This file lists:
- decisions that still need an owner;
- facts we could not verify;
- places where this repo departs from the brief.

Each entry gives our working default, who decides, and when. When an entry is resolved, move it to **Resolved** with the date and the decision.

**Owners:** user (you), AD (academic director), PE (pedagogy expert), TE (technical expert), UI (UI expert), lead (the main session).

Last updated 2026-10-10 (after Phase 2).

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
| B9 | Kearns & Singh 2002, the simulation lemma's number in the journal version | Cite AJKS V3 Lemma 2.2 instead | AD | Phase 2 |
| B23 | ~~The greedy-policy loss lemma~~ **Resolved:** AJKS V3 Lemma 1.11 gives V^{π_Q} ≥ V* − 2‖Q − Q*‖∞/(1 − γ) (no γ in the numerator); PLAN Module 2 corrected | — | — | — |
| B24 | Azar et al. 2024 §4.2, the exact statement about deterministic preferences (we say DPO pushes π(y_l) → 0) | Re-read §4.2 before quoting | AD | Phase 3 (Day 4) |

**Citations: RiverSwim and SARSA**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B10 | ~~The MBIE-EB bonus form~~ **Resolved:** β/√n(s,a), Strehl & Littman 2008 §3.2, Eq. 8, p. 1316 (AD review) | — | — | — |
| B11 | RiverSwim's origin and transition probabilities. The 2008 paper says it is "taken from" Strehl & Littman, ICTAI 2004 (not opened). Fig. 1's probabilities are an image. | Open the 2004 paper and read Fig. 1. Meanwhile our page states its own probabilities. | AD | Phase 3 (Day 2) |
| B12 | Whether Rummery & Niranjan 1994 coined "SARSA" | Read the technical report. Until then, don't claim it. | AD | Phase 2 |

**Citations: other sources**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B13 | Page ranges from secondary sources (the Bradley–Terry end page; Sutton et al. 1999; Konda & Tsitsiklis 1999; Christiano et al. 2017; Thrun & Schwartz 1993) | Omitted from the bib until verified | AD | Phase 5 |
| B14 | Equations reconstructed from extracted PDF text (Dudík DR Eq. 1; SNIPS Eq. 7; SAC-v2 target entropy; Ouyang Eq. 2's pretraining term; DeepSeekMath symbols) | Re-check against the typeset PDF before quoting | AD | When quoted |
| B15 | ~~A source for "BC is competitive on expert data"~~ **Resolved:** Kumar, Hong, Singh & Levine 2022, §4.2, Thm 4.3, Practical Observation 4.1. Its venue (ICLR 2022) is still to be confirmed on the proceedings page. | — | AD | Phase 3 (Day 5) |
| B16 | ~~The probe-environments source~~ **Resolved:** Jones 2021 blog post, "Use probe environments" (5 probes) | — | — | — |
| B17 | Section numbers in Agarwal et al. 2021 come from arXiv v4, not the camera-ready | Check the camera-ready | AD | Phase 3 (Day 3) |
| B18 | Gymnasium paper title: "Standard" vs "Standardized" in the arXiv v4 PDF | Open the PDF | AD | Phase 1 |

**Measurements**

| # | Item | How it gets checked | Owner | When |
|---|---|---|---|---|
| B19 | Whether CPU is as fast as T4 for the small-MLP labs | Timed runs, recorded | TE | Phase 3 |
| B20 | Every lab minute budget in `PLAN.md` | Phase 3 timed runs (compute) plus a human pilot with 2–3 people | PE | Phase 3 and release |

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
