variable "subscription_id" {
  description = "Azure subscription that holds the Databricks workspace."
  type        = string
}

variable "resource_group_name" {
  description = "Existing resource group of the Databricks workspace; the new resources go next to it."
  type        = string
}

variable "location" {
  description = "Azure region, the same as the workspace."
  type        = string
  default     = "centralus"
}

variable "databricks_host" {
  description = "Workspace URL, e.g. https://adb-1234567890123456.7.azuredatabricks.net"
  type        = string
}

variable "my_ip" {
  description = "My public IP address, allowed through the Azure SQL firewall to load the source tables."
  type        = string
}

variable "alert_emails" {
  description = "Where budget alerts are sent (one or more addresses)."
  type        = list(string)
}

variable "budget_amount" {
  description = "Monthly budget in USD (the free account credit)."
  type        = number
  default     = 200
}

variable "budget_start" {
  description = "First day of the budget's first month (must be the first of a month)."
  type        = string
  default     = "2026-10-01T00:00:00Z"
}

variable "environments" {
  description = "One Unity Catalog catalog per environment: heavden_<env>."
  type        = list(string)
  default     = ["dev", "staging", "prod"]
}

variable "admin_group" {
  description = "Account group that owns the catalogs (it is also the metastore admin)."
  type        = string
  default     = "heavden-admins"
}
