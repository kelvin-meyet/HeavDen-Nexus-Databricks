#!/usr/bin/env bash
# Run the bundle job model_ops and print its result (the JSON after MODEL_OPS_RESULT in the job's
# output). The job's log goes to stderr. A run that fails, or that prints no result line, fails
# here, so silence is never taken for a change. Run it from the databricks/ folder; needs jq.
#
#   model_ops.sh <action> [version] [context]     target: $TARGET (default prod)
set -euo pipefail

action="$1"
version="${2:-}"
context="${3:-}"
params="action=$action"
[ -z "$version" ] || params="$params,version=$version"
[ -z "$context" ] || params="$params,context=$context"
log="${RUNNER_TEMP:-/tmp}/model_ops.log"

if ! databricks bundle run model_ops -t "${TARGET:-prod}" --params "$params" 2>&1 | tee "$log" >&2; then
  # the job refuses with a plain reason (e.g. the version did not pass its gates)
  reason=$(grep -oE '(ValueError|LookupError|RuntimeError): .*' "$log" | tail -1 || true)
  echo "::error::model_ops $action failed. ${reason:-See the job run in the log.}" >&2
  exit 1
fi
result=$(grep -o 'MODEL_OPS_RESULT {.*}' "$log" | tail -1 | sed 's/^MODEL_OPS_RESULT //' || true)
if ! jq -e .action <<<"$result" >/dev/null 2>&1; then
  echo "::error::the model_ops job printed no result line" >&2
  exit 1
fi
echo "$result"
