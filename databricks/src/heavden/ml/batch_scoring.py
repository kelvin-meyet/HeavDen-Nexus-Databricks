"""Batch scoring: score every patient-hour with a registered model and build the Gold tables
that depend on it (Plan.md §7.3, §9).

    gold.risk_scores        one row per patient-hour: risk, band, top factors, model version
    gold.alerts_fact        a stay entering the High band (re-alerts suppressed for 6 h)
    gold.escalations_fact   each recorded escalation, and whether it was flagged beforehand
    gold.site_kpis_hourly   census, bands, alerts and escalations per unit and hour

The same functions build the demo snapshot (`heavden.ml.scoring`, `heavden.analytics.gold`), so
the app shows the same numbers in demo and live mode. Differences, by design:

* **Outcomes are known only once recorded**, about 6 h after the event (`silver.outcomes` holds
  only recorded ones). So an alert's outcome is final when its 6 h window closed at least 6 h
  before the data's as-of time, the same rule as `gold.training_set`; until then it stays null.
* **The alert threshold is the one fixed at training** (the model run's `valid_threshold`), not
  refitted here: scoring must not move the alert budget.

Each run rescores every hour with the chosen model version and replaces the four tables, so it
can be re-run safely. When drift episodes add hours (Phase 3), scoring will append new hours only,
so past scores stay those of the model that was live at the time.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from heavden.analytics import gold
from heavden.ml import scoring, train
from heavden.ml.training import to_utc

RECORDING_DELAY = pd.Timedelta(hours=6)
BACKGROUND_ROWS = 2000  # training rows the explanations are measured against


def score_hours(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    model,
    threshold: float,
    model_version: str,
    as_of: pd.Timestamp,
    background: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """The four Gold tables for every patient-hour in `features`."""
    outcomes = outcomes.sort_values("event_ts").drop_duplicates("encounter_id")
    known_by = as_of - RECORDING_DELAY
    risk = scoring.score_table(
        model, features, scoring.RiskBands(high=threshold), background, model_version
    )
    alerts = gold.alerts_fact(risk, outcomes, known_by)
    return {
        "risk_scores": risk,
        "alerts_fact": alerts,
        "escalations_fact": gold.escalations_fact(risk, outcomes, known_by),
        "site_kpis_hourly": gold.site_kpis_hourly(risk, alerts, outcomes),
    }


def background_sample(training_set: pd.DataFrame, split: train.Split) -> pd.DataFrame:
    """Rows from the model's own training block, the reference population for explanations."""
    rows = train.time_split(training_set, split)[0]
    return rows.sample(min(BACKGROUND_ROWS, len(rows)), random_state=0)


def load_registered_model(name: str, alias: str):
    """(model, alert threshold, version) for `name@alias`, or the newest version if `alias` is
    "latest" (dev and staging, whose 2-day models rarely pass the gates)."""
    import mlflow

    client = mlflow.MlflowClient()
    if alias == "latest":
        versions = client.search_model_versions(f"name = '{name}'")
        if not versions:
            raise LookupError(f"{name} has no versions: run the training job first")
        mv = max(versions, key=lambda v: int(v.version))
    else:
        mv = client.get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
    threshold = client.get_run(mv.run_id).data.metrics["valid_threshold"]
    return model, threshold, str(mv.version)


def to_spark(spark, df: pd.DataFrame):
    """A Spark DataFrame where NaN becomes NULL (pandas marks missing numbers with NaN)."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype("Float64")
    return spark.createDataFrame(df)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_prod")
    parser.add_argument("--model-alias", default="champion", help='an alias, or "latest"')
    parser.add_argument("--train-days", type=float, default=train.Split.train_days)
    parser.add_argument("--valid-days", type=float, default=train.Split.valid_days)
    args = parser.parse_args(argv)

    import mlflow
    from pyspark.sql import SparkSession

    from heavden.pipelines.gold import data_window

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    c = args.catalog

    def read(table: str, times: tuple[str, ...]) -> pd.DataFrame:
        return to_utc(spark.read.table(f"{c}.{table}").toPandas(), times)

    features = read("gold.patient_hour_features", ("prediction_ts",))
    training_set = read("gold.training_set", ("prediction_ts", "label_known_at"))
    outcomes = read("silver.outcomes", ("event_ts", "recorded_ts"))
    window = to_utc(data_window(spark.read.table(f"{c}.silver.vitals")).toPandas(), ("as_of",))
    as_of = window["as_of"].iloc[0]

    mlflow.set_registry_uri("databricks-uc")
    name = f"{c}.ml.deterioration_risk"
    model, threshold, version = load_registered_model(name, args.model_alias)
    split = train.Split(args.train_days, args.valid_days)
    tables = score_hours(
        features, outcomes, model, threshold, version, as_of, background_sample(training_set, split)
    )
    for table, frame in tables.items():
        to_spark(spark, frame).write.mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(f"{c}.gold.{table}")
    risk, alerts = tables["risk_scores"], tables["alerts_fact"]
    print(
        f"{name} v{version} ({args.model_alias}), alert threshold {threshold:.4f}, as of {as_of}: "
        + ", ".join(f"{t} {len(f):,}" for t, f in tables.items())
    )
    print(
        f"High hours {risk['risk_band'].eq('High').mean():.1%}; alerts {len(alerts):,}, "
        f"of which escalated within 6 h {alerts['escalated_within_6h'].sum():.0f} "
        f"(outcome known for {alerts['escalated_within_6h'].notna().sum():,})"
    )
