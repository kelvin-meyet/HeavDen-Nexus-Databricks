# One service principal per environment, used by GitHub Actions (Plan.md §15, §16.2).
#
# GitHub signs in with OpenID Connect: Entra ID trusts the short-lived tokens GitHub issues to a
# job running in a named environment of this repository, so no client secret exists anywhere.
# Each GitHub environment maps to one service principal:
#
#   dev                                                  -> heavden-ci-dev      (validates bundles)
#   staging                                              -> heavden-ci-staging  (deploys and runs staging)
#   prod-deploy, prod-run, retrain-approval,
#   production-approval                                   -> heavden-ci-prod     (behind Kelvin's approval)

locals {
  github_environments = {
    "dev"                 = "dev"
    "staging"             = "staging"
    "prod-deploy"         = "prod"
    "prod-run"            = "prod"
    "retrain-approval"    = "prod"
    "production-approval" = "prod"
  }

  # Environments CI deploys to. Dev is Kelvin's own: CI only validates there, so its service
  # principal gets no data access.
  deploy_environments = toset([for env in var.environments : env if env != "dev"])

  # What a deployed environment's jobs and pipelines need in their own catalog, inherited by its
  # schemas: read and write tables, create tables, materialized views, functions and models, and
  # read the landing volume. No grants on the other catalogs.
  catalog_privileges = [
    "USE_CATALOG",
    "USE_SCHEMA",
    "SELECT",
    "MODIFY",
    "READ_VOLUME",
    "CREATE_TABLE",
    "CREATE_MATERIALIZED_VIEW",
    "CREATE_FUNCTION",
    "CREATE_MODEL",
    "EXECUTE",
  ]
}

resource "azuread_application" "ci" {
  for_each     = toset(var.environments)
  display_name = "heavden-ci-${each.key}"
  owners       = [data.azurerm_client_config.current.object_id]
  notes        = "GitHub Actions identity for HeavDen Nexus ${each.key}. Signs in with OIDC; no secrets."
}

resource "azuread_service_principal" "ci" {
  for_each  = azuread_application.ci
  client_id = each.value.client_id
  owners    = [data.azurerm_client_config.current.object_id]
}

resource "azuread_application_federated_identity_credential" "github" {
  for_each       = local.github_environments
  application_id = azuread_application.ci[each.value].id
  display_name   = "github-${each.key}"
  description    = "GitHub Actions jobs in the ${each.key} environment of ${var.github_repository}"
  issuer         = "https://token.actions.githubusercontent.com"
  audiences      = ["api://AzureADTokenExchange"]
  subject        = "${var.github_oidc_subject_prefix}:environment:${each.key}"
}

# The same identities in the workspace. Workspace access lets them deploy bundle files to their
# home folder and own the jobs and pipelines they deploy (production targets run as the deployer).
resource "databricks_service_principal" "ci" {
  for_each         = azuread_service_principal.ci
  application_id   = each.value.client_id
  display_name     = "heavden-ci-${each.key}"
  workspace_access = true
}

resource "databricks_grant" "catalog" {
  for_each   = local.deploy_environments
  catalog    = databricks_catalog.env[each.key].name
  principal  = databricks_service_principal.ci[each.key].application_id
  privileges = local.catalog_privileges
}

# The ingestion job reads Azure SQL through the shared connection.
resource "databricks_grant" "sql_connection" {
  for_each           = local.deploy_environments
  foreign_connection = databricks_connection.sql_source.name
  principal          = databricks_service_principal.ci[each.key].application_id
  privileges         = ["USE_CONNECTION"]
}

# In staging and prod the pipelines run as the service principal, which then owns the tables it
# creates. Owning the catalog lets the admins manage it but not read tables owned by someone else,
# so they get read access explicitly (to check runs and build dashboards). No write access: changes
# to staging and prod data go through CI.
resource "databricks_grant" "admins_read" {
  for_each   = local.deploy_environments
  catalog    = databricks_catalog.env[each.key].name
  principal  = var.admin_group
  privileges = ["USE_CATALOG", "USE_SCHEMA", "SELECT", "READ_VOLUME", "EXECUTE"]
}

# The staging and prod service principals publish the Clinical Ops dashboard with their own
# credentials, so its queries run as them on the shared SQL warehouse. CAN_USE runs queries; it
# grants no data access (that comes from the catalog grants above). The owner keeps CAN_MANAGE
# and workspace admins keep theirs.
resource "databricks_permissions" "warehouse" {
  sql_endpoint_id = databricks_sql_endpoint.shared.id

  dynamic "access_control" {
    for_each = local.deploy_environments
    content {
      service_principal_name = databricks_service_principal.ci[access_control.value].application_id
      permission_level       = "CAN_USE"
    }
  }
}
