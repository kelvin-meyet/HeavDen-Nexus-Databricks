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

- Azure SQL accepts connections from Azure services (Databricks serverless runs inside Azure) and from `my_ip`, for loading. The data is synthetic.
- `azurerm` has no setting for the SQL free offer, so the database is created with `azapi` (`useFreeLimit`, `freeLimitExhaustionBehavior = AutoPause`). Only one free database is allowed per subscription.
- External locations skip validation on create, because a new storage role can take a minute to apply; the first pipeline run proves access.
