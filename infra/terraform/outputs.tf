output "storage_account" {
  value = azurerm_storage_account.lake.name
}

output "landing_url" {
  value = "abfss://landing@${local.lake_host}/"
}

output "sql_server_fqdn" {
  value = azurerm_mssql_server.source.fully_qualified_domain_name
}

output "sql_database" {
  value = azapi_resource.source_db.name
}

output "key_vault" {
  value = azurerm_key_vault.main.name
}

output "secret_scope" {
  value = databricks_secret_scope.kv.name
}

output "catalogs" {
  value = [for c in databricks_catalog.env : c.name]
}

output "sql_connection" {
  value = databricks_connection.sql_source.name
}

output "sql_warehouse_id" {
  value = databricks_sql_endpoint.shared.id
}

output "tenant_id" {
  value = data.azurerm_client_config.current.tenant_id
}

# GitHub variable AZURE_CLIENT_ID for each environment (not a secret: it only names the identity).
output "ci_client_ids" {
  value = { for env, app in azuread_application.ci : env => app.client_id }
}

# Bundle variable assistant_users for prod (databricks.yml): the website's identity, granted
# EXECUTE on the explain function and CAN_RUN on prod's Genie space by assistant_publish.
output "web_client_id" {
  value = azurerm_user_assigned_identity.web.client_id
}

# Vercel's BACKEND_URL
output "api_url" {
  value = one([for app in azurerm_container_app.api : "https://${app.ingress[0].fqdn}"])
}

# The repository variable WEBSITE_APP_ID: where releases deploy new images (release.yml)
output "api_app_id" {
  value = one(azurerm_container_app.api[*].id)
}
