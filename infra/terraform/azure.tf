# Azure resources around the existing workspace: data lake storage, the Azure SQL source
# database (free offer), Key Vault, the access connector Unity Catalog uses, and a budget.

data "azurerm_client_config" "current" {}

data "azurerm_subscription" "current" {}

data "azurerm_resource_group" "workspace" {
  name = var.resource_group_name
}

# Storage account, SQL server and Key Vault names must be unique across Azure.
resource "random_string" "suffix" {
  length  = 5
  upper   = false
  special = false
}

locals {
  suffix = random_string.suffix.result
  tags = {
    project = "heavden-nexus"
    data    = "synthetic"
  }
}

# --- Data lake -------------------------------------------------------------------------

resource "azurerm_storage_account" "lake" {
  name                            = "heavden${local.suffix}"
  resource_group_name             = data.azurerm_resource_group.workspace.name
  location                        = var.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  is_hns_enabled                  = true # ADLS Gen2
  allow_nested_items_to_be_public = false
  tags                            = local.tags
}

# landing: what the generator writes (vitals JSON, delayed outcomes), one folder per environment.
# catalogs: managed storage for the heavden_<env> catalogs (the metastore has no root of its own).
resource "azurerm_storage_container" "lake" {
  for_each           = toset(["landing", "catalogs"])
  name               = each.key
  storage_account_id = azurerm_storage_account.lake.id
}

# The managed identity Unity Catalog uses to read and write the lake.
resource "azurerm_databricks_access_connector" "uc" {
  name                = "heavden-uc-connector"
  resource_group_name = data.azurerm_resource_group.workspace.name
  location            = var.location
  tags                = local.tags

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_role_assignment" "connector_lake" {
  scope                = azurerm_storage_account.lake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_databricks_access_connector.uc.identity[0].principal_id
}

# Me too, so the generator on my laptop can upload to landing with my `az login`.
resource "azurerm_role_assignment" "me_lake" {
  scope                = azurerm_storage_account.lake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = data.azurerm_client_config.current.object_id
}

# --- Azure SQL: the hospital's source system ----------------------------------------------

resource "random_password" "sql_admin" {
  length           = 24
  special          = true
  override_special = "!#%*-_=+"
  min_upper        = 2
  min_lower        = 2
  min_numeric      = 2
  min_special      = 2
}

resource "azurerm_mssql_server" "source" {
  name                         = "heavden-sql-${local.suffix}"
  resource_group_name          = data.azurerm_resource_group.workspace.name
  location                     = var.location
  version                      = "12.0"
  administrator_login          = "heavden_admin"
  administrator_login_password = random_password.sql_admin.result
  minimum_tls_version          = "1.2"
  tags                         = local.tags
}

# Databricks serverless runs inside Azure, so "allow Azure services" (0.0.0.0) lets it in;
# the data is synthetic, so this simple rule is acceptable here.
resource "azurerm_mssql_firewall_rule" "azure_services" {
  name             = "AllowAzureServices"
  server_id        = azurerm_mssql_server.source.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_mssql_firewall_rule" "me" {
  name             = "MyIP"
  server_id        = azurerm_mssql_server.source.id
  start_ip_address = var.my_ip
  end_ip_address   = var.my_ip
}

# The free offer: serverless General Purpose, 100,000 vCore-seconds and 32 GB a month at no cost.
# AutoPause when the monthly allowance runs out, so it can never bill.
resource "azapi_resource" "source_db" {
  type      = "Microsoft.Sql/servers/databases@2025-01-01"
  name      = "heavden"
  parent_id = azurerm_mssql_server.source.id
  location  = var.location
  tags      = local.tags

  body = {
    sku = {
      name   = "GP_S_Gen5_2"
      tier   = "GeneralPurpose"
      family = "Gen5"
    }
    properties = {
      useFreeLimit                = true
      freeLimitExhaustionBehavior = "AutoPause"
      autoPauseDelay              = 60
      minCapacity                 = 0.5
      maxSizeBytes                = 34359738368
      zoneRedundant               = false
    }
  }
}

# --- Key Vault -----------------------------------------------------------------------------

resource "azurerm_key_vault" "main" {
  name                       = "kv-heavden-${local.suffix}"
  resource_group_name        = data.azurerm_resource_group.workspace.name
  location                   = var.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  rbac_authorization_enabled = true
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  tags                       = local.tags
}

# I manage the secrets.
resource "azurerm_role_assignment" "me_kv" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

# Databricks reads them through the secret scope, as the AzureDatabricks first-party app.
data "azuread_service_principal" "azure_databricks" {
  client_id = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d"
}

resource "azurerm_role_assignment" "databricks_kv" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = data.azuread_service_principal.azure_databricks.object_id
}

resource "azurerm_key_vault_secret" "sql_user" {
  name         = "sql-admin-user"
  value        = azurerm_mssql_server.source.administrator_login
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_role_assignment.me_kv]
}

resource "azurerm_key_vault_secret" "sql_password" {
  name         = "sql-admin-password"
  value        = random_password.sql_admin.result
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_role_assignment.me_kv]
}

# --- Budget --------------------------------------------------------------------------------

resource "azurerm_consumption_budget_subscription" "credits" {
  name            = "heavden-credits"
  subscription_id = data.azurerm_subscription.current.id
  amount          = var.budget_amount
  time_grain      = "Monthly"

  time_period {
    start_date = var.budget_start
  }

  dynamic "notification" {
    for_each = [25, 50, 75]
    content {
      enabled        = true
      threshold      = notification.value
      operator       = "GreaterThanOrEqualTo"
      threshold_type = "Actual"
      contact_emails = [var.alert_email]
    }
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThanOrEqualTo"
    threshold_type = "Forecasted"
    contact_emails = [var.alert_email]
  }
}
