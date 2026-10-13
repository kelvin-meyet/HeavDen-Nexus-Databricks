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
    code_paths: list[str] | None = None,
    pip_requirements: list[str] | None = None,
) -> LoggedCandidate:
    """Log one candidate as an MLflow run (params, validation metrics, calibrated model).

    `code_paths` ships source code inside the model (the `heavden` package, whose
    `CalibratedModel` class the pickle refers to), so the model loads wherever its
    `pip_requirements` are installed, without our wheel.
    """
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
            code_paths=code_paths,
            pip_requirements=pip_requirements,
        )
        return LoggedCandidate(run.info.run_id, info.model_uri)


def register_champion(
    logged: LoggedCandidate, name: str = LOCAL_MODEL_NAME, alias: str = "champion"
) -> int:
    """Register the logged model and point `alias` at the new version. Returns the version."""
    version = mlflow.register_model(logged.model_uri, name)
    mlflow.MlflowClient().set_registered_model_alias(name, alias, version.version)
    return int(version.version)


def register_version(
    logged: LoggedCandidate, name: str, *, passed_gates: bool, description: str = ""
) -> tuple[int, list[str]]:
    """Register a trained model as a new version of `name`. Returns (version, aliases set).

    Every training run registers its chosen model, so the registry is the audit trail. Aliases:
    a model that passed the gates becomes `@challenger`; it also becomes `@champion` only if
    there is no champion yet (the first model). Replacing an existing champion is a promotion
    decision for a person, not for the training job (Phase 3: production-approval).
    """
    client = mlflow.MlflowClient()
    version = int(mlflow.register_model(logged.model_uri, name).version)
    if description:
        client.update_model_version(name, str(version), description=description)
    client.set_model_version_tag(name, str(version), "passed_gates", str(passed_gates).lower())
    aliases: list[str] = []
    if passed_gates:
        client.set_registered_model_alias(name, "challenger", version)
        aliases.append("challenger")
        if not has_alias(name, "champion"):
            client.set_registered_model_alias(name, "champion", version)
            aliases.append("champion")
    return version, aliases


def data_blocks(name: str, version: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None:
    """First and last hour of the version's train / validation / test blocks, as its training
    run recorded them (`training.block_windows`). None for versions trained before runs
    recorded their window."""
    client = mlflow.MlflowClient()
    tags = client.get_run(client.get_model_version(name, version).run_id).data.tags
    if "train_from" not in tags:
        return None
    return {
        block: (pd.Timestamp(tags[f"{block}_from"]), pd.Timestamp(tags[f"{block}_to"]))
        for block in ("train", "valid", "test")
    }


def has_alias(name: str, alias: str) -> bool:
    try:
        mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return False
    return True
