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
