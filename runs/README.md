# Run records

One JSON file per notebook run (`kind: notebook`) or scripted experiment
(`kind: experiment`). These files are the **only** evidence the site uses for
"this lab has run". `scripts/gen_readiness.py` builds the readiness page from
them; nothing on the site may claim a lab works on a runtime without a record.

## How a record gets here

1. Run a notebook to the end. Its last cell (`lab.finish()`) prints the record
   between `BEGIN/END PRL RUN RECORD` markers, and:
   - locally, saves it in `runs/inbox/` (gitignored);
   - on Colab, offers it as a download.
2. File it: `uv run python scripts/add_run_record.py` (files everything in the
   inbox), or `... add_run_record.py path/to/record.json`, or paste the printed
   Colab output: `pbpaste | uv run python scripts/add_run_record.py --stdin`.
   The script validates the schema, rejects anything that looks like a
   credential, a private path or a username, checks `env` against
   `_variables.yml`, and writes `runs/<date>-<env>-<notebook>-<sha8>.json`.
3. Commit it. The readiness page updates on the next build.

## What counts as "teaching-eligible"

A notebook run on its **designed runtime** (`env` equals the module's
`runtime`), with solutions bound, the whole notebook, no QUICK settings, no
test doubles, status `pass`, and the **current** `content_sha` and
`prl_version`. Anything else is shown for what it is: a real-path run
elsewhere, a CI run, or a run from before the notebook last changed.

Never edit a record by hand except to add a `note`. Never label a laptop or
CI time as a Colab time.
