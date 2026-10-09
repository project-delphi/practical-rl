#!/usr/bin/env bash
# Lint infra/aws: cfn-lint on the CloudFormation templates, and shellcheck on prl-aws, on this
# script and on the bash user-data embedded in instance.yaml. Tools run through uvx, in
# isolated environments, at pinned versions. Run from anywhere: infra/aws/lint.sh
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CFN_LINT=cfn-lint==1.57.2
SHELLCHECK_PY=shellcheck-py==0.11.0.1

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "== cfn-lint ($CFN_LINT): $(cd "$here" && echo ./*.yaml)"
uvx "$CFN_LINT" "$here"/*.yaml

# The user-data sits between "# BEGIN prl-user-data" and "# END prl-user-data" in a YAML block
# scalar. Strip the YAML indentation and replace the Fn::Sub references with placeholders.
user_data="$tmp/user-data.sh"
# shellcheck disable=SC2016 # the ${...} in the sed patterns are literal Fn::Sub references
{
	echo '#!/bin/bash'
	awk '
		/# BEGIN prl-user-data/ { indent = match($0, /[^ ]/) - 1; on = 1 }
		on { print substr($0, indent + 1) }
		/# END prl-user-data/ { on = 0 }
	' "$here/instance.yaml" |
		sed -e 's/\${AWS::[A-Za-z]*}/cfn-pseudo-parameter/g' -e 's/\${[A-Za-z]*}/cfn-parameter/g'
} >"$user_data"
lines=$(wc -l <"$user_data" | tr -d ' ')
if [ "$lines" -lt 50 ]; then
	echo "lint.sh: could not extract the user-data from instance.yaml (markers missing?)" >&2
	exit 1
fi

echo "== shellcheck ($SHELLCHECK_PY): bin/prl-aws, lint.sh, instance.yaml user-data ($lines lines)"
uvx --from "$SHELLCHECK_PY" shellcheck "$here/bin/prl-aws" "$here/lint.sh" "$user_data"

echo "== all checks passed"
