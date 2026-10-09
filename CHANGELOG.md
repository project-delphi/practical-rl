# Changelog

All notable changes to this workshop. Dates are UTC.

## Unreleased

### Added
- Phase 0 (2026-10-09): `PLAN.md`, `OPEN_QUESTIONS.md`, `references.bib`, persona agents in `.claude/agents/`.
- Phase 1 (in progress, 2026-10-09):
  - `prl` package: runtime, run records, lab harness, checks, five environments, evaluation, data and plotting, with 53 tests.
  - Notebook build from jupytext sources, with a rhythm lint. Worked and learner checks. `00-setup` and a Module 1 stub.
  - `_variables.yml` with a validator, and generated includes for every page. Readiness built from `runs/`.
  - CI: lint, tests and drift gate; colab-compat against Colab's freeze; colab-sim notebook jobs; infra lint. A Pages workflow.
  - `infra/aws` (CloudFormation, `prl-aws`, price table from the AWS Price List).
