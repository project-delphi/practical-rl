# Practical Reinforcement Learning

A five-day, practical, first-principles reinforcement learning workshop by Genial Labs.
Site: <https://project-delphi.github.io/practical-rl/>

Participants write, run, debug and evaluate RL code all week: every algorithm is derived from
a few ideas, implemented from scratch, then compared with a maintained library or exact ground
truth. Labs open in Google Colab and also run locally (uv) or on AWS.

**Status:** under construction. The scaffold (Phase 1) and the golden Module 1 (Phase 2) are
merged; Phase 3 (the modules, day by day) is next. Phase 1's two Colab smoke runs are not filed
yet, so no lab has a Colab record. See `PLAN.md` for the plan and the
[readiness page](https://project-delphi.github.io/practical-rl/readiness.html) for what has
actually run where.

## Repository
| Path | What it holds |
|---|---|
| `_variables.yml` | Single source of truth: days, clock, modules, exercises, runtimes, pins |
| `modules/`, `*.qmd` | The Quarto site (briefings, schedule, day pages, setup, references, instructor pages) |
| `labs/src/` | Lab sources (jupytext percent format) → `notebooks/` (generated, committed) |
| `prl/` | The shared package: environments, data, evaluation, plotting, checkpoints, runtime, run records |
| `runs/` | Run records: the only evidence for "this lab has run" |
| `scripts/` | Generators, notebook build and checks, readiness, link check |
| `infra/aws/` | Optional GPU path on AWS (CloudFormation) |
| `brand/`, `styles/`, `filters/` | Site theme and Quarto filters |

## Quick start (local)
```bash
uv sync --group notebooks
uv run --group notebooks jupyter lab notebooks/
```
See `CLAUDE.md` for the development commands.

## Licenses
Text, figures and notebook prose: CC BY 4.0 (`LICENSE-CONTENT`). Code: MIT (`LICENSE`).
Third-party datasets and models keep their own licenses.
