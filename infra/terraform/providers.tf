# Every provider signs in with the Azure CLI login (`az login`): no keys or tokens in files.
provider "azurerm" {
  features {
    key_vault {
      # Lets `terraform destroy` remove the vault completely at the end of the project.
      purge_soft_delete_on_destroy = true
    }
  }
  subscription_id = var.subscription_id
}

provider "azapi" {
  subscription_id = var.subscription_id
}

provider "azuread" {}

# A Key Vault-backed secret scope can only be created with Azure authentication, not a
# Databricks token, so the Databricks provider uses the Azure CLI login too.
provider "databricks" {
  host      = var.databricks_host
  auth_type = "azure-cli"
}
