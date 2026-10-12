"""Batch scoring: score new patient-hours with a registered model and build the Gold tables
that depend on it (Plan.md §7.3, §9).

    gold.risk_scores        the prediction log: one row per patient-hour, scored once (risk, band,
                            top factors, model version, scored_at)
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

* **Each hour is scored once.** A run scores only the hours not yet in `risk_scores` and appends
  them, so past scores stay those of the model that was live at the time (each row keeps its
  model version and its band at that model's threshold). Monitoring reads this log. The other
  three tables are rebuilt from the whole log every run, because outcomes keep arriving.
  `--rescore` replaces the log with the chosen model's scores for every hour, e.g. after a bug.

Re-running is safe: a second run finds no new hours and rebuilds the same three tables.
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


KEYS = ["encounter_id", "prediction_ts"]


def unscored(features: pd.DataFrame, log: pd.DataFrame | None) -> pd.DataFrame:
    """The patient-hours in `features` that the prediction log doesn't hold yet."""
    if log is None or log.empty:
        return features
    seen = features[KEYS].merge(log[KEYS].drop_duplicates(), how="left", indicator=True)
    return features[(seen["_merge"] == "left_only").to_numpy()]


def score_new(
    features: pd.DataFrame,
    model,
    threshold: float,
    model_version: str,
    background: pd.DataFrame,
    scored_at: pd.Timestamp,
) -> pd.DataFrame:
    """Prediction-log rows for `features`, banded at the model's own alert threshold."""
    risk = scoring.score_table(
        model, features, scoring.RiskBands(high=threshold), background, model_version
    )
    return risk.assign(scored_at=scored_at)


def derived_tables(
    log: pd.DataFrame, outcomes: pd.DataFrame, as_of: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """alerts_fact, escalations_fact and site_kpis_hourly from the whole prediction log."""
    outcomes = outcomes.sort_values("event_ts").drop_duplicates("encounter_id")
    known_by = as_of - RECORDING_DELAY
    alerts = gold.alerts_fact(log, outcomes, known_by)
    return {
        "alerts_fact": alerts,
        "escalations_fact": gold.escalations_fact(log, outcomes, known_by),
        "site_kpis_hourly": gold.site_kpis_hourly(log, alerts, outcomes),
    }


def score_hours(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    model,
    threshold: float,
    model_version: str,
    as_of: pd.Timestamp,
    background: pd.DataFrame,
    log: pd.DataFrame | None = None,
    scored_at: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """The four Gold tables after scoring the hours of `features` that `log` doesn't hold.

    `risk_scores` is the whole log (old rows unchanged, new rows appended); `new_scores` is
    just the rows this run added.
    """
    scored_at = scored_at if scored_at is not None else pd.Timestamp.now(tz="UTC")
    new = unscored(features, log)
    if new.empty:  # nothing new since the last run
        new_scores = log.iloc[:0] if log is not None else pd.DataFrame()
    else:
        new_scores = score_new(new, model, threshold, model_version, background, scored_at)
    parts = [df for df in (log, new_scores) if df is not None and not df.empty]
    risk = pd.concat(parts, ignore_index=True) if parts else new_scores
    return {"risk_scores": risk, **derived_tables(risk, outcomes, as_of), "new_scores": new_scores}


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
    parser.add_argument(
        "--rescore", default="false", help='"true" replaces the log: every hour, this model'
    )
    args = parser.parse_args(argv)
    rescore = args.rescore.lower() == "true"

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
    log_table = f"{c}.gold.risk_scores"
    log = None
    if not rescore and spark.catalog.tableExists(log_table):
        log = read("gold.risk_scores", ("prediction_ts",))
        if "scored_at" in log:
            log = to_utc(log, ("scored_at",))
        else:  # rows from before the log kept scoring times
            log["scored_at"] = pd.Series(pd.NaT, index=log.index, dtype="datetime64[ns, UTC]")
    tables = score_hours(
        features,
        outcomes,
        model,
        threshold,
        version,
        as_of,
        background_sample(training_set, split),
        log,
    )
    new_scores = tables.pop("new_scores")
    if log is None:  # first run, or --rescore: the log starts again
        to_spark(spark, tables["risk_scores"]).write.mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(log_table)
    elif not new_scores.empty:  # each hour once: append, never rewrite
        to_spark(spark, new_scores).write.mode("append").option("mergeSchema", "true").saveAsTable(
            log_table
        )
    for table in ("alerts_fact", "escalations_fact", "site_kpis_hourly"):
        to_spark(spark, tables[table]).write.mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(f"{c}.gold.{table}")
    risk, alerts = tables["risk_scores"], tables["alerts_fact"]
    print(
        f"{name} v{version} ({args.model_alias}), alert threshold {threshold:.4f}, as of {as_of}: "
        f"{len(new_scores):,} new hours scored"
        + (" (log replaced)" if log is None else "")
        + "; "
        + ", ".join(f"{t} {len(f):,}" for t, f in tables.items())
    )
    print(
        f"High hours {risk['risk_band'].eq('High').mean():.1%}; alerts {len(alerts):,}, "
        f"of which escalated within 6 h {alerts['escalated_within_6h'].sum():.0f} "
        f"(outcome known for {alerts['escalated_within_6h'].notna().sum():,})"
    )
