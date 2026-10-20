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

variable "github_repository" {
  description = "GitHub repository (owner/name) whose Actions jobs may sign in as the CI service principals."
  type        = string
  default     = "kelvin-meyet/HeavDen-Nexus-Databricks"
}

# GitHub's OIDC token names the repository with its owner's and its own numeric IDs ("immutable
# subject"), so a deleted-and-recreated repository with the same name can't sign in. Read it with
#   gh api repos/<owner>/<repo>/actions/oidc/customization/sub --jq .sub_claim_prefix
variable "github_oidc_subject_prefix" {
  description = "Start of the subject in this repository's GitHub OIDC tokens."
  type        = string
  default     = "repo:kelvin-meyet@167044815/HeavDen-Nexus-Databricks@1407984379"
}

# The Azure OpenAI deployment behind the heavden-gpt-5-mini endpoint (llm_gateway.tf). None of
# these is a secret: the key goes into the secret scope heavden-llm instead. All three are on the
# deployment's page in the Azure AI Foundry portal ("Target URI").
variable "azure_openai_endpoint" {
  description = "The Azure OpenAI resource's address, e.g. https://<resource>.openai.azure.com"
  type        = string
}

variable "azure_openai_deployment" {
  description = "Name of the gpt-5-mini deployment in that resource."
  type        = string
  default     = "gpt-5-mini"
}

variable "azure_openai_api_version" {
  description = "The api-version in the deployment's Target URI, e.g. 2025-04-01-preview."
  type        = string
}

# The website (website.tf). Neither is a secret.
variable "azure_openai_resource_id" {
  description = "Resource ID of the Azure OpenAI / Foundry resource above (portal: the resource, JSON View, Resource ID). The website's identity gets Cognitive Services OpenAI User on it."
  type        = string
}

variable "api_image" {
  description = "The API image the container app is created with, e.g. ghcr.io/kelvin-meyet/heavden-api:v0.6.1. Empty: no container app yet (the identity and its grants are created anyway). Releases deploy later images."
  type        = string
  default     = ""
}
