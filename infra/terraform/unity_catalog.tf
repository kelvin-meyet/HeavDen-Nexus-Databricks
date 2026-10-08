# The Unity Catalog foundation: storage access, one catalog per environment with the
# medallion schemas, the landing volume, and the Key Vault-backed secret scope.
# Jobs, pipelines, endpoints and dashboards live in the Databricks bundle (databricks/), not here.

locals {
  lake_host = "${azurerm_storage_account.lake.name}.dfs.core.windows.net"
  schemas   = ["bronze", "silver", "gold", "ml", "monitoring"]

  catalog_schemas = {
    for pair in setproduct(var.environments, local.schemas) :
    "${pair[0]}.${pair[1]}" => { env = pair[0], schema = pair[1] }
  }
}

resource "databricks_storage_credential" "lake" {
  name    = "heavden-lake"
  owner   = var.admin_group
  comment = "HeavDen lake, through the access connector's managed identity"

  azure_managed_identity {
    access_connector_id = azurerm_databricks_access_connector.uc.id
  }

  depends_on = [azurerm_role_assignment.connector_lake]
}

# Role assignments can take a minute to reach storage; validation would then fail on the
# first apply, so it is skipped (the first pipeline run proves access).
resource "databricks_external_location" "lake" {
  for_each        = azurerm_storage_container.lake
  name            = "heavden-${each.key}"
  url             = "abfss://${each.key}@${local.lake_host}/"
  credential_name = databricks_storage_credential.lake.name
  owner           = var.admin_group
  skip_validation = true
  comment         = "HeavDen ${each.key} container"
}

resource "databricks_catalog" "env" {
  for_each     = toset(var.environments)
  name         = "heavden_${each.key}"
  storage_root = "abfss://catalogs@${local.lake_host}/heavden_${each.key}"
  owner        = var.admin_group
  comment      = "HeavDen Nexus ${each.key}. Synthetic data only."

  # Dev and staging hold only small seeded data; prod is protected from an accidental destroy.
  force_destroy = each.key != "prod"

  depends_on = [databricks_external_location.lake]
}

resource "databricks_schema" "layer" {
  for_each      = local.catalog_schemas
  catalog_name  = databricks_catalog.env[each.value.env].name
  name          = each.value.schema
  owner         = var.admin_group
  force_destroy = each.value.env != "prod"
}

# Each environment reads its own folder of the landing container: /Volumes/heavden_<env>/bronze/landing
resource "databricks_volume" "landing" {
  for_each         = toset(var.environments)
  name             = "landing"
  catalog_name     = databricks_catalog.env[each.key].name
  schema_name      = databricks_schema.layer["${each.key}.bronze"].name
  volume_type      = "EXTERNAL"
  storage_location = "abfss://landing@${local.lake_host}/${each.key}"
  owner            = var.admin_group
  comment          = "Vitals JSON and delayed outcomes written by the generator"
}

resource "databricks_secret_scope" "kv" {
  name = "heavden-kv"

  keyvault_metadata {
    resource_id = azurerm_key_vault.main.id
    dns_name    = azurerm_key_vault.main.vault_uri
  }

  depends_on = [azurerm_role_assignment.databricks_kv]
}
