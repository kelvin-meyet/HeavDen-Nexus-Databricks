# Terraform: Azure resources and the Unity Catalog foundation

Creates everything the platform needs around the existing Azure Databricks workspace, in its resource group. Jobs, pipelines, endpoints and dashboards are defined in the Databricks bundle (`databricks/`), not here.

| Resource | Purpose |
|---|---|
| ADLS Gen2 storage account, containers `landing` and `catalogs` | generator output (one folder per environment); managed storage for the catalogs |
| Databricks access connector (managed identity) + storage role | how Unity Catalog reaches the lake, with no keys |
| Azure SQL server + database on the **free offer** | the hospital's source system; pauses itself if the monthly free allowance runs out, so it never bills |
| Key Vault (RBAC) with the generated SQL admin login | secrets, read by Databricks through the `heavden-kv` secret scope |
| Budget on the subscription | email at 25 / 50 / 75% of $200, and when the forecast passes 100% |
| Storage credential, external locations `heavden-landing` / `heavden-catalogs` | Unity Catalog access to the containers |
| Catalogs `heavden_dev` / `heavden_staging` / `heavden_prod`, each with `bronze`, `silver`, `gold`, `ml`, `monitoring` | the medallion layout, owned by the `heavden-admins` group |
| External volume `heavden_<env>.bronze.landing` | the environment's landing folder, read by Auto Loader |
| Service principals `heavden-ci-dev` / `-staging` / `-prod` (Entra ID + workspace), GitHub OIDC trust, catalog and connection grants | the identities GitHub Actions deploys as (`identities.tf`) |
| Secret scope `heavden-llm` (created empty), serving endpoints `heavden-gpt-5-5` (OpenAI) and `heavden-gpt-5-mini` (Azure OpenAI), query rights for the staging and prod service principals | the Databricks assistant's models on Kelvin's own keys, switched like the models Databricks hosts (`llm_gateway.tf`) |

## Use

Prerequisites: Terraform ≥ 1.6, the Azure CLI signed in (`az login`) as an Owner of the subscription who is also a member of `heavden-admins` (the metastore admin group).

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in; git-ignored
terraform init
terraform plan -out tfplan
terraform apply tfplan
terraform output
```

All providers authenticate with the Azure CLI login, so no keys are stored in files. The SQL admin password is generated and kept in Key Vault. **The state file holds that password**, which is why `*.tfstate` is git-ignored; state is kept locally for this single-developer project.

At the end of the project, `terraform destroy` removes everything except `heavden_prod`, which refuses to be destroyed while it holds tables (drop them first, on purpose).

## Notes

- **The keys behind `heavden-gpt-5-5` and `heavden-gpt-5-mini` never pass through Terraform.** The first apply creates the empty scope (`terraform apply -target=databricks_secret_scope.llm`); the keys are then put in with `databricks secrets put-secret heavden-llm openai-api-key` and `... azure-openai-key` (each prompts for its value); a full apply creates the two endpoints, which read them. Nothing bills while idle; the tokens bill to the OpenAI account and the Azure subscription. Each endpoint allows 60 calls a minute and records its usage (`system.serving.endpoint_usage`).

- Azure SQL accepts connections from Azure services (Databricks serverless runs inside Azure) and from `my_ip`, for loading. The data is synthetic.
- `azurerm` has no setting for the SQL free offer, so the database is created with `azapi` (`useFreeLimit`, `freeLimitExhaustionBehavior = AutoPause`). Only one free database is allowed per subscription.
- External locations skip validation on create, because a new storage role can take a minute to apply; the first pipeline run proves access.

## CI identities (`identities.tf`)

GitHub Actions signs in as one service principal per environment, with no secrets: each Entra ID app trusts the OIDC tokens GitHub issues to jobs in named environments of this repository.

This replaces the usual manual setup (create a service principal in the Databricks UI, click **Generate secret**, paste the client ID and secret into a GitHub environment). Terraform creates the service principals and their grants; instead of a secret, each gets a *federated credential* trusting this repository's GitHub environment; GitHub stores only the client ID and tenant ID, as plain variables.

| GitHub environment | Service principal | Can do | Approval |
|---|---|---|---|
| `dev` | `heavden-ci-dev` | validate bundles (no data access) | none |
| `staging` | `heavden-ci-staging` | deploy and run in `heavden_staging`, read Azure SQL | none (`main` only) |
| `staging-approval` (rehearsals of retrain, promote and rollback) | `heavden-ci-staging` | the same | Kelvin |
| `prod-deploy` (deploy, `v*` tags), `prod-run` (run prod's jobs), `retrain-approval`, `production-approval` | `heavden-ci-prod` | deploy and run in `heavden_prod`, read Azure SQL | Kelvin |

After `terraform apply`, create the GitHub environments and variables once with `bash infra/github/setup_environments.sh` (it reads `terraform output ci_client_ids` and `tenant_id`). Client and tenant IDs aren't secrets: they name the identity, and only a token from this repository's environment can sign in as it.

Each service principal can use only its own catalog. In staging and prod the service principal owns the tables its pipelines create, and owning a catalog doesn't grant reading tables someone else owns, so `heavden-admins` gets read-only access to those two catalogs (`admins_read`); their data changes only through CI. The connection `heavden_sql` still signs in to Azure SQL with the admin login; a read-only login (`db_datareader` + `VIEW CHANGE TRACKING`) is a planned follow-up.
