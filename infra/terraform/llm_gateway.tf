# The Databricks assistant's LLMs that run on Kelvin's own keys (decided 17 Oct 2026): one model
# serving endpoint per model, forwarding each request to OpenAI or Azure OpenAI ("external
# models"). They are queried exactly like the models Databricks hosts (databricks-gpt-oss-120b),
# so the Databricks assistant switches between all of them with one setting, LLM_ENDPOINT
# (backend/heavden_api/chat/models.yaml, `databricks:` list).
#
# The keys live in the secret scope heavden-llm, created here empty: Kelvin puts the two keys in
# with `databricks secrets put-secret` (prompted), so they never enter Terraform's state. Only the
# scope's creator and workspace admins can read it. The endpoints read the keys when they are
# created; the service principals may only query the endpoints, never read the keys.
#
# Nothing runs between calls, so nothing bills while idle: the tokens bill to the OpenAI account
# and the Azure subscription. AI Gateway records usage (system.serving.endpoint_usage) and caps
# the calls per minute, a backstop against a runaway loop.
#
# First apply: `terraform apply -target=databricks_secret_scope.llm`, put the two keys, then a
# full apply (an endpoint can't be created before its key exists).

resource "databricks_secret_scope" "llm" {
  name = "heavden-llm"
}

locals {
  llm_rate_limit_per_minute = 60 # calls, all callers together; a question makes 2-6 calls
}

resource "databricks_model_serving" "gpt_5_5" {
  name = "heavden-gpt-5-5"

  config {
    served_entities {
      name = "gpt-5-5"
      external_model {
        name     = "gpt-5.5"
        provider = "openai"
        task     = "llm/v1/chat"
        openai_config {
          openai_api_key = "{{secrets/${databricks_secret_scope.llm.name}/openai-api-key}}"
        }
      }
    }
  }

  ai_gateway {
    usage_tracking_config {
      enabled = true
    }
    rate_limits {
      calls          = local.llm_rate_limit_per_minute
      key            = "endpoint"
      renewal_period = "minute"
    }
  }
}

resource "databricks_model_serving" "gpt_5_mini" {
  name = "heavden-gpt-5-mini"

  config {
    served_entities {
      name = "gpt-5-mini"
      external_model {
        name     = var.azure_openai_deployment
        provider = "openai"
        task     = "llm/v1/chat"
        openai_config {
          openai_api_type        = "azure"
          openai_api_key         = "{{secrets/${databricks_secret_scope.llm.name}/azure-openai-key}}"
          openai_api_base        = var.azure_openai_endpoint
          openai_deployment_name = var.azure_openai_deployment
          openai_api_version     = var.azure_openai_api_version
        }
      }
    }
  }

  ai_gateway {
    usage_tracking_config {
      enabled = true
    }
    rate_limits {
      calls          = local.llm_rate_limit_per_minute
      key            = "endpoint"
      renewal_period = "minute"
    }
  }
}

# The Databricks assistant runs as each environment's service principal (its serving endpoint is
# deployed by the staging / prod job): it may query the two models. CAN_QUERY reads no secret.
resource "databricks_permissions" "llm_query" {
  for_each = {
    gpt_5_5    = databricks_model_serving.gpt_5_5.serving_endpoint_id
    gpt_5_mini = databricks_model_serving.gpt_5_mini.serving_endpoint_id
  }
  serving_endpoint_id = each.value

  dynamic "access_control" {
    for_each = local.deploy_environments
    content {
      service_principal_name = databricks_service_principal.ci[access_control.value].application_id
      permission_level       = "CAN_QUERY"
    }
  }
}
