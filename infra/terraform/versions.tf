# Provider versions are pinned to releases at least ~3 weeks old.
terraform {
  required_version = ">= 1.6"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "= 5.6.0"
    }
    # azurerm has no setting for the Azure SQL free offer (useFreeLimit); azapi sets it directly.
    azapi = {
      source  = "Azure/azapi"
      version = "= 2.12.0"
    }
    # Only to look up the AzureDatabricks app, which needs to read Key Vault secrets.
    azuread = {
      source  = "hashicorp/azuread"
      version = "= 3.9.0"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "= 1.132.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "= 3.9.0"
    }
  }
}
