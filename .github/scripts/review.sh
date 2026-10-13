#!/usr/bin/env bash
# Who decided what on GitHub, for the audit trail (<catalog>.ml.ops_audit). Needs GH_TOKEN with
# `actions: read`, and jq. The output is one word of base64 JSON, safe to pass as a job parameter
# (heavden.ml.model_ops.decode_context reads it).
#
#   review.sh context <environment> [json]
#       Who approved this run ($GITHUB_RUN_ID) in <environment>, their comment and the run's
#       URL, merged over [json] (e.g. the drift check that led to the request).
#   review.sh rejections
#       Requests rejected in the latest runs of the retrain, promote and rollback workflows.
#       Prints nothing when there are none.
set -euo pipefail

repo="$GITHUB_REPOSITORY"
api="repos/$repo/actions"

case "${1:-}" in
  context)
    environment="$2"
    extra="${3:-}"
    [ -n "$extra" ] || extra='{}'
    review=$(gh api "$api/runs/$GITHUB_RUN_ID/approvals" | jq -c --arg env "$environment" \
      '[.[] | select(.state == "approved" and any(.environments[]; .name == $env))] | last // {}')
    jq -cn --argjson review "$review" --argjson extra "$extra" \
      --arg actor "${GITHUB_TRIGGERING_ACTOR:-}" \
      --arg url "${GITHUB_SERVER_URL:-https://github.com}/$repo/actions/runs/$GITHUB_RUN_ID" \
      '$extra + {
         actor: ($review.user.login // $actor),
         reason: ([$extra.reason, $review.comment] | map(select(. != null and . != "")) | join(" | ")),
         run_url: $url
       }' | base64 -w0
    ;;

  rejections)
    reviews='[]'
    for workflow in retrain promote rollback; do
      # a rejected request ends its run as a failure; requests are told from other runs by title
      runs=$(gh api "$api/workflows/$workflow.yml/runs?status=failure&per_page=15" | jq -c \
        '[.workflow_runs[] | select(.display_title | test("^(Retrain|Promotion|Rollback) request:"))
          | {run_id: .id, title: .display_title, run_url: .html_url, decided_at: .updated_at}]')
      while IFS= read -r run; do
        [ -n "$run" ] || continue
        rejected=$(gh api "$api/runs/$(jq -r .run_id <<<"$run")/approvals" | jq -c \
          '[.[] | select(.state == "rejected")] | last // empty
           | {environment: .environments[0].name, actor: .user.login, comment: (.comment // "")}')
        [ -n "$rejected" ] || continue
        reviews=$(jq -c --argjson run "$run" --argjson rejected "$rejected" --arg workflow "$workflow" \
          '. + [$run + $rejected + {workflow: $workflow}]' <<<"$reviews")
      done < <(jq -c '.[]' <<<"$runs")
    done
    # The newest 10: a job parameter is limited in size, and the job skips what it already holds.
    if [ "$(jq length <<<"$reviews")" -gt 0 ]; then
      jq -c '{reviews: (sort_by(.decided_at) | .[-10:])}' <<<"$reviews" | base64 -w0
    fi
    ;;

  *)
    echo "usage: review.sh context <environment> [json] | rejections" >&2
    exit 2
    ;;
esac
