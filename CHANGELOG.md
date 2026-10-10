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
- `health.yml` (weekly): the locked-environment run on Linux and macOS arm64 with live budgets; `colab_constraints.py bump`, which refreshes the Colab freeze, runs colab-compat and colab-sim on it and opens a PR with their results; and a `uv lock --upgrade` report. The colab-sim steps moved to a composite action, with the same checks (2026-10-10).
- `scripts/threshold_protocol.py` and `prl.checks.StochasticCheck`: thresholds for stochastic checkpoints, set from at least 100 (QUICK) or 20 (live) seeds of the solution and its mutants, and recorded as experiment records (2026-10-10).

### Fixed
- Colab setup: one pin per package, so `pip` no longer fails with ResolutionImpossible on gymnasium 1.4.0 (2026-10-09).
- A stuck participant is pointed to `lab.use_reference(N)` by the stubs and the solution cells (2026-10-09).
- CI's test job reinstalled torch on every `uv run` despite its dev-only install; `UV_NO_SYNC=1` keeps it dev-only (2026-10-10).
- `add_run_record.py` no longer overwrites a filed record that has the same file name; the new one gets `-2`, `-3`, ... (2026-10-10).
