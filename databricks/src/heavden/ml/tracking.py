"""MLflow tracking and registry helpers.

Locally this uses a SQLite tracking store under `data/mlflow/` (git-ignored), which supports the
model registry and aliases exactly like Unity Catalog does on Databricks. On Databricks the
same calls go to the workspace: the tracking URI is `databricks` and the registry is
`databricks-uc`, with model names like `heavden_prod.ml.deterioration_risk`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from heavden.ml.train import CandidateResult

LOCAL_MODEL_NAME = "deterioration_risk"


@dataclass(frozen=True)
class LoggedCandidate:
    run_id: str
    model_uri: str  # MLflow 3 logged-model URI, e.g. models:/m-...


def use_local_store(root: Path, experiment: str = "heavden-deterioration-risk") -> str:
    """Point MLflow at a local SQLite store (tracking + registry) under `root`."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    mlflow.set_registry_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    return experiment


def log_candidate(
    result: CandidateResult,
    example: pd.DataFrame,
    extra_metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> LoggedCandidate:
    """Log one candidate as an MLflow run (params, validation metrics, calibrated model)."""
    with mlflow.start_run(run_name=result.name) as run:
        mlflow.log_params({"model_type": result.name, **result.params})
        mlflow.log_metrics({f"valid_{k}": v for k, v in result.valid_metrics.items()})
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)
        if tags:
            mlflow.set_tags(tags)
        sample = example.head(50)
        info = mlflow.sklearn.log_model(
            result.model,
            name="model",
            # our CalibratedModel class lives in the installed `heavden` package
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(sample, result.model.predict_proba(sample)[:, 1]),
        )
        return LoggedCandidate(run.info.run_id, info.model_uri)


def register_champion(
    logged: LoggedCandidate, name: str = LOCAL_MODEL_NAME, alias: str = "champion"
) -> int:
    """Register the logged model and point `alias` at the new version. Returns the version."""
    version = mlflow.register_model(logged.model_uri, name)
    mlflow.MlflowClient().set_registered_model_alias(name, alias, version.version)
    return int(version.version)
