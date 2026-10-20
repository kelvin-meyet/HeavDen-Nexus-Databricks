# The public website's API (Plan.md §13, Phase 5): FastAPI on Azure Container Apps, scaled to
# zero, its image pulled from GitHub's container registry (a public package: no registry login).
#
# The API signs in to Databricks and Azure OpenAI as its own user-assigned managed identity, so
# no secret exists anywhere, and that identity can only read prod:
#   Databricks:   USE CATALOG heavden_prod; USE SCHEMA + SELECT on gold; USE SCHEMA on ml; CAN_USE
#                 on the shared warehouse (identities.tf). EXECUTE on ml.explain_patient_risk and
#                 CAN_RUN on prod's Genie space are granted by the assistant_publish job (bundle
#                 variable assistant_users = the web_client_id output), because that job replaces
#                 both and a grant made here would be lost.
#   Azure OpenAI: Cognitive Services OpenAI User on the resource (call its deployments, nothing else).
#
# The image holds code only. The snapshot the API serves (model, documents, recordings; after
# October also the ward tables) lives in Azure Blob Storage, in the lake account's `website`
# container: an export on Databricks writes it through the volume heavden_prod.ml.website
# (snapshots/<name>/, then current.json naming it), and the API reads what current.json names each
# time it starts, straight from storage with its identity. So starting needs no Databricks: if
# Databricks is off, the site keeps serving the last snapshot. A new snapshot needs no new image or
# deploy; pointing current.json back is the rollback.
#
# The daily cap on live chat answers is kept in the Azure Table `websitecounters` (one row per UTC
# day), because the API is stopped whenever nobody visits and an in-memory count would restart at
# zero. The identity can write that table and nothing else; no visitor data is stored.
#
# Cost: the Consumption plan's free grant (per subscription and month: 180,000 vCPU-seconds,
# 360,000 GiB-seconds, 2 million requests) covers ~100 awake hours at 0.5 vCPU / 1 GiB; scaled to
# zero it costs nothing, and the environment has no fixed fee. Logs are capped at 0.05 GB a day.
#
# Two steps: the identity, its grants, the volume and the environment can be applied at once; the
# container app is created once `api_image` names an image in GHCR and the volume holds a
# snapshot (the API won't start without one). After that, releases deploy new images (CI), and
# Terraform leaves the image alone.

resource "azurerm_user_assigned_identity" "web" {
  name                = "heavden-web-prod"
  resource_group_name = data.azurerm_resource_group.workspace.name
  location            = var.location
  tags                = local.tags
}

# The same identity in the workspace (application id = the managed identity's client id).
# Workspace and SQL access are entitlements, not data: what it can read is the grants below.
resource "databricks_service_principal" "web" {
  application_id        = azurerm_user_assigned_identity.web.client_id
  display_name          = "heavden-web-prod"
  workspace_access      = true
  databricks_sql_access = true
}

resource "databricks_grant" "web_catalog" {
  catalog    = databricks_catalog.env["prod"].name
  principal  = databricks_service_principal.web.application_id
  privileges = ["USE_CATALOG"]
}

resource "databricks_grant" "web_gold" {
  schema     = databricks_schema.layer["prod.gold"].id
  principal  = databricks_service_principal.web.application_id
  privileges = ["USE_SCHEMA", "SELECT"]
}

# ml holds the explain function (EXECUTE comes from assistant_publish); no SELECT, no models
resource "databricks_grant" "web_ml" {
  schema     = databricks_schema.layer["prod.ml"].id
  principal  = databricks_service_principal.web.application_id
  privileges = ["USE_SCHEMA"]
}

# Where Databricks writes the website's snapshots: the prod folder of the `website` container
resource "databricks_volume" "website" {
  name             = "website"
  catalog_name     = databricks_catalog.env["prod"].name
  schema_name      = databricks_schema.layer["prod.ml"].name
  volume_type      = "EXTERNAL"
  storage_location = "abfss://website@${local.lake_host}/prod"
  owner            = var.admin_group
  comment          = "The website's snapshots: snapshots/<name>/ and current.json naming the one the API serves"

  depends_on = [databricks_external_location.lake]
}

# prod's deploy identity publishes (the export runs as it); the website reads from storage below
resource "databricks_grant" "ci_publishes_snapshots" {
  volume     = databricks_volume.website.id
  principal  = databricks_service_principal.ci["prod"].application_id
  privileges = ["READ_VOLUME", "WRITE_VOLUME"]
}

# Read the website container and nothing else in the storage account
resource "azurerm_role_assignment" "web_snapshots" {
  scope                = "${azurerm_storage_account.lake.id}/blobServices/default/containers/${azurerm_storage_container.lake["website"].name}"
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.web.principal_id
  principal_type       = "ServicePrincipal"
}

# The daily cap's count: one row per UTC day (chat/limits.py TableBudget)
resource "azurerm_storage_table" "chat_budget" {
  name               = "websitecounters"
  storage_account_id = azurerm_storage_account.lake.id
}

resource "azurerm_role_assignment" "web_chat_budget" {
  scope                = "${azurerm_storage_account.lake.id}/tableServices/default/tables/${azurerm_storage_table.chat_budget.name}"
  role_definition_name = "Storage Table Data Contributor"
  principal_id         = azurerm_user_assigned_identity.web.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "web_openai" {
  scope                = var.azure_openai_resource_id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_user_assigned_identity.web.principal_id
  principal_type       = "ServicePrincipal"
}

# Needs the subscription's Microsoft.OperationalInsights and Microsoft.App providers registered
# (once per subscription: az provider register --namespace <name>).
# The API's logs, kept 30 days (included in the ingestion price). The daily cap bounds the cost
# if something starts logging in a loop: the website writes a few KB a day.
resource "azurerm_log_analytics_workspace" "web" {
  name                = "heavden-web-logs"
  resource_group_name = data.azurerm_resource_group.workspace.name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  daily_quota_gb      = 0.05
  tags                = local.tags
}

# Consumption only: billed per second while a replica runs. No private endpoints or planned
# maintenance, which would bring a Dedicated plan management charge.
resource "azurerm_container_app_environment" "web" {
  name                       = "heavden-web"
  resource_group_name        = data.azurerm_resource_group.workspace.name
  location                   = var.location
  logs_destination           = "log-analytics"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.web.id
  tags                       = local.tags
}

resource "azurerm_container_app" "api" {
  count                        = var.api_image == "" ? 0 : 1
  name                         = "heavden-api"
  container_app_environment_id = azurerm_container_app_environment.web.id
  resource_group_name          = data.azurerm_resource_group.workspace.name
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.web.id]
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    # Scaled to zero ~5 minutes after the last request. One replica at most: the chat's rate
    # limits and daily cap are counted inside the process.
    min_replicas = 0
    max_replicas = 1

    http_scale_rule {
      name                = "http"
      concurrent_requests = "10"
    }

    container {
      name   = "api"
      image  = var.api_image
      cpu    = 0.5
      memory = "1Gi" # measured 20 Oct (snapshot mode, prod-sized data): peak 588-659 MiB

      env {
        name  = "MODE"
        value = "live"
      }
      env {
        name  = "MANAGED_IDENTITY_CLIENT_ID"
        value = azurerm_user_assigned_identity.web.client_id
      }
      env {
        name  = "DATABRICKS_HOST"
        value = var.databricks_host
      }
      env {
        name  = "LIVE_CATALOG"
        value = databricks_catalog.env["prod"].name
      }
      env {
        name  = "WAREHOUSE_ID"
        value = databricks_sql_endpoint.shared.id
      }
      env {
        name  = "GENIE_TITLE"
        value = "HeavDen Ward Data (prod)"
      }
      env {
        name  = "SNAPSHOT_URL"
        value = "${azurerm_storage_account.lake.primary_blob_endpoint}${azurerm_storage_container.lake["website"].name}/prod"
      }
      env {
        # downloaded here at start-up (the image's own folders are read-only for the API's user)
        name  = "SNAPSHOT_DIR"
        value = "/tmp/snapshot"
      }
      env {
        name  = "CHAT_BUDGET_TABLE_URL"
        value = "${azurerm_storage_account.lake.primary_table_endpoint}${azurerm_storage_table.chat_budget.name}"
      }
      env {
        name  = "LLM_PROVIDER"
        value = "azure"
      }
      env {
        name  = "LLM_MODEL"
        value = var.azure_openai_deployment # listed under app: in chat/models.yaml
      }
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = var.azure_openai_endpoint
      }
      env {
        # Vercel's proxy, then Container Apps' ingress: the visitor is 2 entries from the right
        # of X-Forwarded-For (to confirm on the first deploy)
        name  = "TRUSTED_PROXY_HOPS"
        value = "2"
      }

      # TCP, not GET /health: in live mode /health asks the SQL warehouse for "now", and a probe
      # must never wake it. Start-up (snapshot download included) gets up to 60 s.
      startup_probe {
        transport               = "TCP"
        port                    = 8000
        interval_seconds        = 2
        failure_count_threshold = 30
      }
      liveness_probe {
        transport = "TCP"
        port      = 8000
      }
    }
  }

  lifecycle {
    # releases deploy each new image (CI); Terraform owns everything else
    ignore_changes = [template[0].container[0].image]
  }

  depends_on = [
    databricks_grant.web_catalog,
    databricks_grant.web_gold,
    databricks_grant.web_ml,
    azurerm_role_assignment.web_snapshots,
    azurerm_role_assignment.web_chat_budget,
    databricks_permissions.warehouse,
    azurerm_role_assignment.web_openai,
  ]
}

# Releases deploy new images (release.yml, job website) as prod's deploy identity: it may change
# this app and nothing else in Azure. A revision runs as the website's identity, so updating the
# app also needs the right to assign that identity (to what it may change: this app).
resource "azurerm_role_assignment" "ci_deploys_api" {
  count                = length(azurerm_container_app.api)
  scope                = azurerm_container_app.api[0].id
  role_definition_name = "Contributor"
  principal_id         = azuread_service_principal.ci["prod"].object_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "ci_assigns_web_identity" {
  count                = length(azurerm_container_app.api)
  scope                = azurerm_user_assigned_identity.web.id
  role_definition_name = "Managed Identity Operator"
  principal_id         = azuread_service_principal.ci["prod"].object_id
  principal_type       = "ServicePrincipal"
}
