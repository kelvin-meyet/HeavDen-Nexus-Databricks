"""The Databricks assistant as MLflow logs it ("models from code"): this file is saved with the
model, and Model Serving runs it to build the agent (`heavden_api.chat.assistant_deploy`)."""

import mlflow

from heavden_api.chat.databricks_assistant import DatabricksAssistant

mlflow.openai.autolog()  # each LLM call is a span in the request's trace
mlflow.models.set_model(DatabricksAssistant())
