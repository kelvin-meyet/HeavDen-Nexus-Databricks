# How code reaches Databricks

Read this before stepping through the code base. It explains where the code lives, how it gets into the Databricks workspace, and who is allowed to put it there.

## The master copy is GitHub

The repository on GitHub is the master copy. Every other copy is made from it, and changes come back to it through a pull request that CI has checked:

- a laptop clone, where the code is written and the tests run;
- the files the bundle uploads into the workspace, which are what Databricks runs.

Nothing in Databricks is edited by hand.

## Bundle deploy, not Git folders

Code can get into Databricks in two ways. This project uses the first.

| | **Bundle deploy** (used) | **Git folder** (not used) |
|---|---|---|
| What it is | The Databricks CLI reads [`databricks/databricks.yml`](../databricks/databricks.yml). It uploads the repo's files and creates or updates the jobs and pipelines the file describes | A `git clone` of the repo inside the workspace |
| How | `databricks bundle deploy -t <target>` from `databricks/` | Workspace → Create → Git folder |
| Where the files land | `/Workspace/Users/<deploying identity>/.bundle/heavden-nexus/<target>/files/` | Wherever the folder is created |
| Creates jobs and pipelines | Yes (`heavden-sql-ingest`, `heavden-medallion`, …) | No, only files |
| Needs a GitHub account linked in Databricks | No | Yes |

The bundle is the **declarative automation bundle** pattern. One file describes every job and pipeline, for every environment, and one command deploys it. The learning notebooks in `notebooks/databricks/` are uploaded with the code they import, so they open from the bundle's `files/` folder.

## Three routes, one per environment

```
laptop        ── bundle deploy -t dev, as the developer ──────────────────► dev      (heavden_dev)
GitHub main   ── GitHub Actions, as heavden-ci-staging, on every merge ───► staging  (heavden_staging)
GitHub tag v* ── GitHub Actions, as heavden-ci-prod, after approval ──────► prod     (heavden_prod)
```

| Environment | Deployed by | When | What runs |
|---|---|---|---|
| dev | the developer, from a laptop | while building | anything, on small seeded data |
| staging | CI ([`ci.yml`](../.github/workflows/ci.yml), job `staging`) | every merge to `main` | deploy, then Bronze → Silver → Gold on fixed seeded data, then a training smoke test |
| prod | CI ([`release.yml`](../.github/workflows/release.yml)) | a `v*` release tag, after a required reviewer approves | deploy; when the Databricks assistant changed since the last release, also its documents, Genie space and endpoint (red-teamed). Data runs are started on demand with [`prod-run.yml`](../.github/workflows/prod-run.yml), also after approval |

On every pull request, CI also validates the bundle for all three targets, signed in as `heavden-ci-dev`. That identity can't read or write any data.

## What happens automatically, and what needs a person

Merging a pull request reaches **staging only**. Every change to **prod** waits for a person. That's not a first-time setup step; it's how prod always works.

| You do | What runs, without further clicks | Needs your approval |
|---|---|---|
| Open or update a pull request | lint, tests (also on Databricks' library versions), bundle validation as `heavden-ci-dev` | no |
| Merge to `main` | staging: deploy, pipeline, training (a smoke test on 2 days) as `heavden-ci-staging` | no |
| Push a release tag (`git tag -a v0.x.0` and `git push origin v0.x.0`) | **prod deploy**: jobs and pipelines updated to that release's code, no data run. If the Databricks assistant changed since the last release: its documents, patient explanation and Genie space are published, then its endpoint is deployed and red-teamed | **yes**, environment `prod-deploy`; a second time for the assistant's endpoint |
| Start **prod-run** (Actions → prod-run → Run workflow; tick ingestion, pipeline, scoring, drift check and/or training) | **prod run** as `heavden-ci-prod` | **yes**, environment `prod-run` |
| Start **staging-run** (the same steps against staging, to rehearse a drift episode) | **staging run** as `heavden-ci-staging` | no |
| A drift check reaches "act" | a **retrain request** opens and waits | **yes**, `retrain-approval` (rehearsals in staging: `staging-approval`) |
| A retrained model passes its gates | a **promotion request** opens and waits; approving moves `@champion` | **yes**, `production-approval` (staging: `staging-approval`) |
| Start **rollback** with a reason | `@champion` goes back to the previous champion | **yes**, `production-approval` (staging: `staging-approval`) |

**Why prod isn't automatic on merge:**

- **Deploying and running are different risks.** A deploy changes *what code* prod would run. A run changes *prod data and the registered model*. Each has its own gate and its own audit trail.
- **Prod is where the decisions live.** Only prod trains the model that scores patients. A model that changes behaviour should reach production through a deliberate, recorded decision, never as a side effect of merging. This is human-in-the-loop MLOps.
- **Staging has already proven the code.** By the time you tag a release, the same code has run end to end in staging on every merge.
- **Cost and shared limits.** Prod runs process 14 days of data and can wake the free Azure SQL database (about 50 wake-ups a month), so they happen when someone chooses, not on every merge.
- **Audit.** Every approval is recorded in GitHub (who, when, which environment, optional comment) and stays in the repository after the cloud workspace is gone.

In a team, the approver would be someone other than the author (the four-eyes principle). In this solo project the author is the required reviewer, which GitHub allows for environment approvals.

**Typical prod sequence after a change:** merge, staging goes green, tag `v0.x.0`, approve **prod-deploy**, then start **prod-run** with what needs to run (for example only *Train*) and approve it.

## Robot identities: service principals with OIDC, no secrets

GitHub Actions needs its own identity in Databricks, never a person's login. There is one **service principal** per environment, each with **least privilege**:

| Service principal | GitHub environment(s) | May do |
|---|---|---|
| `heavden-ci-dev` | `dev` | validate bundles; no data access |
| `heavden-ci-staging` | `staging` (from `main` only), and `staging-approval` (from `main`, needs a reviewer's approval: rehearsals of retrain, promote and rollback) | deploy and run in `heavden_staging`; read the Azure SQL source |
| `heavden-ci-prod` | `prod-deploy` (from `v*` tags), `prod-run`, `retrain-approval`, `production-approval` (from `main`). Every run needs a reviewer's approval | deploy and run in `heavden_prod`; read the Azure SQL source |

### The same job as "Generate secret", without the secret

The usual setup creates a service principal in the Databricks UI, clicks **Generate secret**, and pastes the client ID and secret into a GitHub environment. This project does the same job with **OpenID Connect (OIDC)**, also called workload identity federation:

| Usual manual step | Here |
|---|---|
| Add a service principal (×3) | `terraform apply` creates them in Entra ID and adds them to the workspace ([`infra/terraform/identities.tf`](../infra/terraform/identities.tf)) |
| Grant access to the catalog | The same Terraform grants each one only its own catalog |
| Generate a secret | Nothing is generated. A **federated credential** tells Entra ID to trust GitHub tokens from this repository's environment X |
| Create the GitHub environments and paste the client ID and secret | [`infra/github/setup_environments.sh`](../infra/github/setup_environments.sh) creates the environments, their approval and branch rules, and only the client ID (`AZURE_CLIENT_ID`) per environment plus one repository variable, `AZURE_TENANT_ID`. Neither is a secret: they name the identity, they don't unlock it |

How a workflow run signs in:

```
1. a job starts in the GitHub environment "staging"
2. GitHub gives the job a signed token: "repo kelvin-meyet/HeavDen-Nexus-Databricks, environment staging"
3. the job presents it to Entra ID (azure/login)
4. Entra ID trusts exactly that statement and returns a short-lived login as heavden-ci-staging
5. the Databricks CLI deploys with it (DATABRICKS_AUTH_TYPE=azure-cli)
```

Nothing reusable is stored anywhere, so there is nothing to leak, expire or rotate. A token from another repository, branch or environment is refused.

The token names the repository with numeric IDs as well as names (`repo:kelvin-meyet@167044815/HeavDen-Nexus-Databricks@1407984379:environment:staging`, GitHub's *immutable subject*). The trust rules match that exact string, so even a deleted and recreated repository with the same name can't sign in. The prefix is the Terraform variable `github_oidc_subject_prefix`.

### Setting it up (once)

From the repository root:

```bash
# 1. The service principals, their permissions and their GitHub trust rules (Entra ID + Databricks)
cd infra/terraform
terraform plan -out tfplan        # review: only additions expected
terraform apply tfplan
cd ../..

# 2. The GitHub environments, their approval and branch rules, and the AZURE_* variables
gh auth status                    # signed in as a repository admin
bash infra/github/setup_environments.sh
```

The script reads the client IDs from `terraform output`, so run it after step 1. It is safe to run again: it overwrites the same settings. Check the result under the repository's **Settings → Environments**. The first pull request then proves the setup works: its `bundle-validate` check signs in as `heavden-ci-dev`.

## Three different "GitHub ↔ Databricks" connections

People often say "connect GitHub and Databricks". There are three separate connections:

| Direction | Purpose | Status here |
|---|---|---|
| Databricks → GitHub, as a person | lets a Git folder pull and push | Not used |
| GitHub → Databricks, as a robot | lets CI deploy bundles (the OIDC setup above) | Set up in Phase 1 |
| Databricks → GitHub, as a robot | lets the drift job request a retrain (`repository_dispatch`) | Phase 3, using a fine-grained token kept in Key Vault |

## Which notebooks run where

| Folder | Runs in | Cost |
|---|---|---|
| `notebooks/local/` | VS Code or Jupyter on a laptop, with the project's `.venv` | free |
| `notebooks/databricks/` | the Databricks workspace in a browser, on Serverless | a few cents per run |

A `notebooks/databricks/` notebook run on a laptop fails with `name 'dbutils' is not defined`, because `dbutils` exists only inside Databricks. These notebooks also find their code from their own workspace path, which only the deployed copy has.
