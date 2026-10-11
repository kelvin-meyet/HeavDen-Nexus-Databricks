#!/usr/bin/env bash
# Creates the GitHub environments and variables the workflows sign in with (Plan.md §16.2).
# Environments are repository settings, not files, so this script is how they are kept as code.
#
#   bash infra/github/setup_environments.sh      # from the repo root, after `terraform apply`
#
# Needs: the GitHub CLI signed in as a repository admin (`gh auth status`), and Terraform state
# (it reads the client IDs from `terraform output`). Writes no secrets: the client and tenant IDs
# only name the service principals; GitHub's OIDC token is what signs in.
# Safe to re-run: PUTs overwrite, and re-adding an existing branch policy is ignored.
set -euo pipefail

REPO=kelvin-meyet/HeavDen-Nexus-Databricks
KELVIN_ID=167044815 # GitHub user id of kelvin-meyet, the required reviewer

cd infra/terraform
TENANT=$(terraform output -raw tenant_id)
IDS=$(terraform output -json ci_client_ids)
client_id() { python -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$IDS" "$1"; }
cd ../..

# dev: any branch (pull requests validate the bundle)
gh api -X PUT "repos/$REPO/environments/dev" --silent

# staging: only main can deploy
gh api -X PUT "repos/$REPO/environments/staging" --silent --input - <<'EOF'
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
EOF
gh api -X POST "repos/$REPO/environments/staging/deployment-branch-policies" -f name=main -f type=branch --silent || true

# prod environments: Kelvin approves every run
for env in prod-deploy retrain-approval production-approval; do
  gh api -X PUT "repos/$REPO/environments/$env" --silent --input - <<EOF
{"reviewers": [{"type": "User", "id": $KELVIN_ID}],
 "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
EOF
done
gh api -X POST "repos/$REPO/environments/prod-deploy/deployment-branch-policies" -f name='v*' -f type=tag --silent || true
gh api -X POST "repos/$REPO/environments/retrain-approval/deployment-branch-policies" -f name=main -f type=branch --silent || true
gh api -X POST "repos/$REPO/environments/production-approval/deployment-branch-policies" -f name=main -f type=branch --silent || true

# Variables (not secrets: they only name the identity)
gh variable set AZURE_TENANT_ID --repo "$REPO" --body "$TENANT"
gh variable set AZURE_CLIENT_ID --repo "$REPO" --env dev --body "$(client_id dev)"
gh variable set AZURE_CLIENT_ID --repo "$REPO" --env staging --body "$(client_id staging)"
for env in prod-deploy retrain-approval production-approval; do
  gh variable set AZURE_CLIENT_ID --repo "$REPO" --env "$env" --body "$(client_id prod)"
done

echo
gh api "repos/$REPO/environments" --jq '.environments[] | "\(.name): \([.protection_rules[]?.type] | join(", "))"'
gh variable list --repo "$REPO"
