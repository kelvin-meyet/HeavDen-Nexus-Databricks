# Deploy paths

Every way code, a model or data reaches a running environment, and who has to approve it. A claim
such as "production only changes from a release tag, after an approval" is only true if it holds
for every row here. `tests/test_project_rules.py` fails when a workflow file exists that this page
does not name.

Step by step for Databricks: [how-code-reaches-databricks.md](how-code-reaches-databricks.md).

## Workflows in this repository

| Workflow | Started by | What it changes | Approval |
|---|---|---|---|
| `ci.yml` | every pull request | Nothing: lint, tests, secret scan, image build, bundle validation. | none |
| `ci.yml` | every merge to `main` | **Staging**: deploys the bundle and runs a smoke data run (pipeline, assistant documents, train, score, drift check, model-ops self-test). When the Databricks assistant's files changed (`.github/scripts/assistant_changed.py`), also deploys it to `heavden-assistant-staging`, red-teams it and runs the quality test. About $1 a merge, about $2.50 more with the assistant. | none |
| `release.yml` | a `v*` tag | **Production**: deploys the bundle to prod; deploys the Databricks assistant to `heavden-assistant-prod` when it changed since the previous tag; exports the website's snapshot, builds and smoke-tests the API image, pushes it to GHCR, points the website at the snapshot and deploys Container Apps. | `prod-deploy`, once per job |
| `prod-run.yml` | by hand | **Production data**: copies Azure SQL, runs the pipeline, can train, scores new hours with `@champion`, checks drift. With `website_snapshot` it also exports a snapshot that the **live website** then serves, without the release's image smoke test. | `prod-run` |
| `staging-run.yml` | by hand | Staging data, same steps as prod-run. | none |
| `data-run.yml` | called by prod-run and staging-run, never by itself | The steps above. | the caller's |
| `retrain.yml` | a drift check that reaches "act", or by hand | Trains and registers a challenger. Changes nothing that is served. | `retrain-approval` (`staging-approval` for staging) |
| `promote.yml` | a retrain that passes its gates, or by hand | **Production model**: moves `@champion`, so the next scoring run uses the new version. | `production-approval` (`staging-approval` for staging) |
| `rollback.yml` | by hand | **Production model**: moves `@champion` back to `@previous_champion`. | `production-approval` (`staging-approval` for staging) |

## Outside the workflows

| Path | What it changes | Approval |
|---|---|---|
| **Vercel** | Publishes `web/` from `main` on every merge: the website's pages change with no tag and no approval. The one exception to "a tag is production". | none |
| **Terraform** (`infra/terraform`) | Azure and Databricks infrastructure, the website's hosting and identities. Applied by hand from a laptop with `az login`; never from CI. | the person applying it |
| **Databricks CLI** | `databricks bundle deploy/run -t dev` from a laptop. Dev only: staging and prod jobs belong to their CI identities. | none |
| **The generator** | `generator.load` / `generator.episode` write source data into Azure SQL and the landing storage of an environment, prod included, from a laptop. | the person running it |

## Settings these claims depend on

Checked against `infra/github/setup_environments.sh` and the project notes, not against GitHub
itself. Check each in the repository's settings after any change there.

- **Branch rules on `main`** (ruleset `protect-main`): a pull request, and six required checks
  (`lint-and-test`, `ml-on-databricks-libraries`, `api-on-serving-libraries`, `api-image`, `web`,
  `bundle-validate`). The secret scan runs inside `lint-and-test`, so it is required too.
- **Required reviewer** on `prod-deploy`, `prod-run`, `retrain-approval` and
  `production-approval`. `prod-deploy` accepts only `v*` tags; the others only `main`.
- **`staging-approval`**: its reviewer is not set by the script. Not verified.
- **Vercel's production branch** is `main`. Not verified from this repository.
