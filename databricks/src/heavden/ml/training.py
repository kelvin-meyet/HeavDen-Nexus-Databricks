"""Train, evaluate and register the deterioration-risk model (Plan.md §8).

The steps of notebook 06 as one function, so the Databricks job (`databricks/src/jobs/
train_model.py`) and the tests run the same code:

1. split the labelled patient-hours by time (train / validation / test, with 6 h gaps);
2. fit the candidates (logistic regression, LightGBM), calibrate them on validation, and score
   them and NEWS2 on validation at the same alert budget;
3. choose: the best validation AUPRC among candidates that pass the gates (beat NEWS2 on AUPRC
   and flag at least as many escalations); if none passes, the best one is still registered,
   without an alias, so the registry records every run;
4. score the chosen model and NEWS2 once on test;
5. log every candidate to MLflow and register the chosen one (`tracking.register_version`).

On Databricks `main` reads `<catalog>.gold.training_set` with Spark, logs to the workspace
experiment and registers in Unity Catalog as `<catalog>.ml.deterioration_risk`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import heavden
from heavden.ml import evaluate, tracking, train

# What the logged model needs to load: our code travels inside the model (code_paths), these
# libraries are pinned to the versions it was trained with.
MODEL_LIBRARIES = ("cloudpickle", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy")


@dataclass
class TrainingOutcome:
    chosen: train.CandidateResult
    passed_gates: bool
    news2_valid: dict[str, float]
    test: dict[str, float]
    news2_test: dict[str, float]
    version: int
    aliases: list[str]
    rows: dict[str, int]


def model_requirements() -> list[str]:
    return [f"{lib}=={version(lib)}" for lib in MODEL_LIBRARIES]


def run_training(
    table: pd.DataFrame,
    model_name: str,
    split: train.Split | None = None,
    budget: evaluate.AlertBudget | None = None,
    tags: dict[str, str] | None = None,
) -> TrainingOutcome:
    """Train, evaluate, log and register. The MLflow tracking and registry URIs and the
    experiment must be set by the caller."""
    budget = budget or evaluate.AlertBudget()
    train_df, valid, test = train.time_split(table, split)
    rows = {"train": len(train_df), "valid": len(valid), "test": len(test)}
    if min(rows.values()) == 0 or valid["label"].sum() == 0:
        raise ValueError(f"not enough labelled data for this split: {rows}")

    results, news2_valid = train.train_candidates(train_df, valid, budget)
    chosen = train.select_champion(results, news2_valid)
    passed = chosen is not None
    chosen = chosen or max(results, key=lambda r: r.valid_metrics["auprc"])

    # Test is scored once, for the chosen model, at its validation threshold (and NEWS2 at its own).
    test_scores = chosen.model.predict_proba(test)[:, 1]
    test_metrics = evaluate.evaluate(
        test, test_scores, chosen.valid_metrics["threshold"], True, budget
    )
    news2_test = evaluate.evaluate(
        test, train.news2_scores(test), news2_valid["threshold"], False, budget
    )

    common_tags = (tags or {}) | {f"rows_{k}": str(v) for k, v in rows.items()}
    code = [str(Path(heavden.__file__).parent)]
    requirements = model_requirements()
    logged = {}
    for r in results:
        metrics = {f"news2_valid_{k}": v for k, v in news2_valid.items()}
        if r is chosen:
            metrics |= {f"test_{k}": v for k, v in test_metrics.items()}
            metrics |= {f"news2_test_{k}": v for k, v in news2_test.items()}
        run_tags = common_tags | {
            "chosen": str(r is chosen).lower(),
            "passed_gates": str(train.passes_gates(r.valid_metrics, news2_valid)).lower(),
        }
        logged[r.name] = tracking.log_candidate(
            r, valid, metrics, run_tags, code_paths=code, pip_requirements=requirements
        )

    description = (
        f"{chosen.name}; validation AUPRC {chosen.valid_metrics['auprc']:.3f} "
        f"(NEWS2 {news2_valid['auprc']:.3f}), escalations flagged "
        f"{chosen.valid_metrics['escalations_flagged']:.0%} (NEWS2 "
        f"{news2_valid['escalations_flagged']:.0%}); test AUPRC {test_metrics['auprc']:.3f} "
        f"(NEWS2 {news2_test['auprc']:.3f}). Passed gates: {passed}."
    )
    version_number, aliases = tracking.register_version(
        logged[chosen.name], model_name, passed_gates=passed, description=description
    )
    return TrainingOutcome(
        chosen, passed, news2_valid, test_metrics, news2_test, version_number, aliases, rows
    )


def _utc(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Spark hands timestamps to pandas without a time zone; the session is UTC."""
    for c in columns:
        df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_prod")
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment (bundle)")
    parser.add_argument("--train-days", type=float, default=train.Split.train_days)
    parser.add_argument("--valid-days", type=float, default=train.Split.valid_days)
    args = parser.parse_args(argv)

    import mlflow
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    source = f"{args.catalog}.gold.training_set"
    table = _utc(spark.read.table(source).toPandas(), ("prediction_ts", "label_known_at"))

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_id=args.experiment_id)
    model_name = f"{args.catalog}.ml.deterioration_risk"
    outcome = run_training(
        table,
        model_name,
        train.Split(args.train_days, args.valid_days),
        tags={
            "source_table": source,
            "data_as_of": str(table["label_known_at"].max()),
            "heavden_version": version("heavden-nexus-databricks"),
        },
    )
    print(
        f"{model_name} v{outcome.version} = {outcome.chosen.name}, rows {outcome.rows}, "
        f"passed gates: {outcome.passed_gates}, aliases: {outcome.aliases or 'none'}"
    )
    print(
        f"validation AUPRC {outcome.chosen.valid_metrics['auprc']:.3f} "
        f"vs NEWS2 {outcome.news2_valid['auprc']:.3f}; "
        f"test AUPRC {outcome.test['auprc']:.3f} vs NEWS2 {outcome.news2_test['auprc']:.3f}"
    )
