# CLAUDE.md — working on the Practical RL workshop

Read `PLAN.md` (the plan and phase gates) and `OPEN_QUESTIONS.md` before changing anything.
Work in phases; stop at each phase gate for review. Keep changes scoped to the current phase,
and ask before restructuring anything already approved.

## Hard rules
- Never fabricate citations, numbers, run times, prices or "measured" results. **Measured means
  present in `runs/`.** Everything else is labeled an estimate. Never label a laptop or CI time
  as a Colab or T4 time.
- Cite only what was opened. Every `references.bib` entry records what was checked in `annote`.
  Unverified facts go to `OPEN_QUESTIONS.md`, not onto pages.
- Comparisons: ≥ 3 seeds per side with every seed shown and ties called (no CI below 5 seeds);
  IQM with bootstrap CIs from 5 seeds; one seed is "single seed, not a ranking". State the budget.
- `prl/` holds infrastructure only (envs, data, evaluation, plotting, checks, runtime, records).
  **No learning algorithms in `prl/`.** Participants write them; later labs include reference
  copies via `# %% include labs/shared/*.py`.
- Python only. No API keys or paid services in any lab. No secrets in the repo.
- American English. Plain, short sentences. No hype, no "simply".

## Generated files: never edit by hand
| File(s) | Generator |
|---|---|
| `_includes/*` (except none) | `scripts/gen_includes.py`, `scripts/gen_readiness.py` |
| `notebooks/*.ipynb` | `scripts/build_notebooks.py` from `labs/src/*.py` |
| `prl/src/prl/_palette.py` | `scripts/gen_tokens.py` from `brand/*.yml` |
| `prl/src/prl/_expected/*.npz` | `scripts/make_expected.py` |
CI's drift gate fails if any of these is out of date.

## Everyday commands
```bash
uv sync                                         # dev + CPU torch (default groups)
uv run pytest                                   # prl and repo tests
uv run ruff check . && uv run ruff format --check .
uv run python scripts/variables.py              # validate _variables.yml
uv run python scripts/gen_includes.py           # includes + placeholder module pages
uv run python scripts/build_notebooks.py        # notebooks from labs/src
uv run python scripts/gen_readiness.py          # readiness page from runs/
PRL_QUICK=1 uv run python scripts/check_notebooks.py --mode worked   # or --mode learner
uv run python scripts/check_notebooks.py --mode worked --record && uv run python scripts/add_run_record.py
PATH="$HOME/opt/quarto-1.10.19/bin:$PATH" quarto render && uv run python scripts/link_check.py
```
Quarto is pinned to 1.10.19 (CI uses the same). Locally it lives in `~/opt/quarto-1.10.19`.

## Lab sources (`labs/src/NN-slug.py`, jupytext percent format)
- Front matter: `prl: {module: mNN, pins: {...}}`. Exercise headings come from `_variables.yml`.
- Cell roles: `exN-head`, `exN-predict`, `exN-stub`, `exN-hint`, `exN-sol`, `exN-run`,
  `exN-explain`, `exN-chk` (and `stretch`). Order: head, predict, stub, hint, sol, run,
  explain, chk. Stubs define only top-level `def`/`class` and raise `NotImplementedError("TODO N…")`.
- Solutions use `@lab.solution(N)`; they never overwrite participant code. Checkpoints call
  `lab.check(N, fn, *objs)` with a function from `prl.checks.mNN`, whose expected values are
  fixtures from `scripts/make_expected.py`.
- Budgets come from `prl.runtime.settings(live=..., quick=...)`; lab code never branches on QUICK.

## Roles and parallel work
Persona agents live in `.claude/agents/` (academic-director, pedagogy-expert, technical-expert,
ui-expert, verifier). See `PLAN.md` §11 for who owns what. Parallel work happens in git
worktrees on branches `module/mNN-slug`, `site/<topic>`, `infra/<topic>`; the lead merges after
review and regenerates generated files after the merge. `prl/` changes go through the lead on
the main line of work.

## Commits
Small, well-described commits ending with:
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
