#!/usr/bin/env bash
# A training result (the JSON after TRAINING_RESULT in the training job's output) as Markdown:
# what a retrain produced and how it did against the champion, gate by gate. Needs jq.
#
#   training_summary.sh '<json>'
set -euo pipefail

result="$1"
jq -e .version <<<"$result" >/dev/null

echo "## Retrained: version $(jq -r .version <<<"$result") ($(jq -r .model_type <<<"$result"))"
echo
if [ "$(jq -r '.aliases | index("champion") != null' <<<"$result")" = "true" ]; then
  echo "**Passed every gate. No model had been put in use here before, so it is \`@champion\` as well as \`@challenger\`:** there is nothing to promote."
elif [ "$(jq -r .passed_gates <<<"$result")" = "true" ]; then
  echo "**Passed every gate: it is \`@challenger\`.** \`@champion\` is unchanged until a promotion is approved."
else
  echo "**Did not pass the gates: no alias.** \`@champion\` stays. The version is registered, so the attempt is on record."
fi
echo
echo "| On the latest days (test), same alert budget | New model | Champion | Gate |"
echo "|---|---|---|---|"
jq -r '.vs_champion | select(. != null) |
  "| Escalations flagged | \(.escalations_flagged * 100 | round)% of \(.escalations) | \(.champion_escalations_flagged * 100 | round)% | at least as many in \(.flagged_at_least_share * 100 | round)% of resamples (needs 75%) |",
  "| AUROC | \(.auroc * 1000 | round / 1000) | \(.champion_auroc * 1000 | round / 1000) | higher in \(.auroc_better_share * 100 | round)% of resamples (needs 75%) |",
  "| AUPRC | \(.auprc * 1000 | round / 1000) | \(.champion_auprc * 1000 | round / 1000) | higher in \(.auprc_better_share * 100 | round)% of resamples (needs 10%: not clearly worse) |",
  "| Brier score | \(.brier * 10000 | round / 10000) | \(.champion_brier * 10000 | round / 10000) | at most 10% worse |",
  "| Alerts per nurse per shift | \(.alerts_per_nurse_shift * 100 | round / 100) | \(.champion_alerts_per_nurse_shift * 100 | round / 100) | |"' <<<"$result"
echo
echo "Resamples are of whole patient stays, the same ones for both models. The champion is given a fresh alert threshold from the same validation days, so the comparison is about the models, not about how old a threshold is."
echo
jq -r '"Validation AUPRC \(.valid_auprc * 1000 | round / 1000) vs NEWS2 \(.news2_valid_auprc * 1000 | round / 1000). Rows: train \(.rows.train), validation \(.rows.valid), test \(.rows.test)."' <<<"$result"
