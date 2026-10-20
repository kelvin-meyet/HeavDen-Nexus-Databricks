"""The website's snapshot, exported from an environment's Gold on Databricks (Plan.md §13-14).

Run by the bundle job `website_snapshot` (`databricks/resources/website_snapshot.job.yml`) on
serverless environment 6: the libraries the model was trained with, which the website's image
installs too (the `serving` group). It writes, with `heavden_api.snapshot.write_snapshot`:

* the last WINDOW_DAYS days of the Gold tables the website reads, up to "now" (the latest hour
  with a full ward);
* the model in use (the score job's: @champion in prod), its alert threshold and background rows
  from its own training block, as batch scoring uses them (`heavden.ml.batch_scoring`);
* check hours scored here, which the website re-scores when it loads the snapshot.

It refuses a Gold whose scores are behind its features (run the score job first) or whose hours
at "now" another model scored (the board and the what-if would disagree), and re-scores the hours
the prediction log holds from the model: the snapshot must give the log's risks. Then it
publishes to the website's volume (`--volume`; prod: /Volumes/heavden_prod/ml/website) and points
the website at it for its next start; a release publishes it without (`--move-pointer false`) and
points the website at it only after the new image passed its smoke test on it. Without a volume
(staging) it builds and checks only. Its last line is `WEBSITE_SNAPSHOT {json}`.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from heavden_api.snapshot import (
    CHECK_TOLERANCE,
    GOLD_TABLES,
    Snapshot,
    full_ward_hour,
    publish_snapshot,
    write_snapshot,
)

# The website loads the whole copy into memory in snapshot mode (the chat's SQL engine). Measured
# 20 Oct 2026, 1 GiB container, peak after a document search: 24 days 588-659 MiB, 42 days
# 668-708, 84 days 920-941. Every page looks back at most 14 days.
WINDOW_DAYS = 28
TIME_COLUMNS = {
    "patient_hour_features": "prediction_ts",
    "risk_scores": "prediction_ts",
    "alerts_fact": "alert_ts",
    "escalations_fact": "event_ts",
    "site_kpis_hourly": "hour_ts",
    "device_health_daily": "date",
}
KEYS = ["encounter_id", "prediction_ts"]
LOG_CHECK_ROWS = 500


def website_tables(
    tables: dict[str, pd.DataFrame], as_of: pd.Timestamp, days: float = WINDOW_DAYS
) -> dict[str, pd.DataFrame]:
    """The last `days` days of the Gold `tables` up to `as_of`, and the stays in them. Refuses
    patient-hours without a risk score: the pages join the two."""
    start = as_of - pd.Timedelta(days=days)
    out = {
        name: tables[name][tables[name][column].gt(start) & tables[name][column].le(as_of)]
        for name, column in TIME_COLUMNS.items()
    }
    stays = out["patient_hour_features"]["encounter_id"].unique()
    out["encounters"] = tables["encounters"][tables["encounters"]["encounter_id"].isin(stays)]
    seen = out["patient_hour_features"][KEYS].merge(
        out["risk_scores"][KEYS].drop_duplicates(), how="left", indicator=True
    )
    missing = int(seen["_merge"].eq("left_only").sum())
    if missing:
        raise ValueError(
            f"{missing:,} patient-hours up to {as_of} have no risk score: run the score job first"
        )
    return {name: frame.reset_index(drop=True) for name, frame in out.items()}


def log_check(model, version: str, tables: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> str:
    """The board shows the risk logged at "now"; the what-if re-scores that hour with the
    snapshot's model. So the hours at `as_of` must be scored by `version` (each hour keeps the
    model that scored it: after a promotion, until new hours are scored, they aren't), and the
    model must give the risks batch scoring logged for its own hours."""
    log = tables["risk_scores"]
    now = set(log.loc[log["prediction_ts"].eq(as_of), "model_version"].astype(str))
    if now != {str(version)}:
        raise ValueError(
            f"the hours at {as_of} were scored by v{', v'.join(sorted(now))}, not v{version}: "
            f"score new hours with v{version} first, or the board and its what-if disagree"
        )
    logged = log[log["model_version"].astype(str) == str(version)]
    rows = tables["patient_hour_features"].merge(logged[[*KEYS, "risk"]], on=KEYS)
    rows = rows.sample(min(LOG_CHECK_ROWS, len(rows)), random_state=0)
    gap = float(np.max(np.abs(model.predict_proba(rows)[:, 1] - rows["risk"].to_numpy())))
    if gap > CHECK_TOLERANCE:
        raise ValueError(f"v{version} scores {len(rows)} logged hours differently (max {gap:.2e})")
    return f"{len(rows)} hours of v{version} re-scored as logged (max gap {gap:.1e})"


def main(argv: list[str] | None = None) -> None:
    from heavden.ml import train

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_prod")
    parser.add_argument("--volume", default="", help="where to publish; empty: build and check")
    parser.add_argument(
        "--model-alias", default="champion", help="as the score job: the model in use"
    )
    parser.add_argument("--train-days", type=float, default=train.Split.train_days)
    parser.add_argument("--valid-days", type=float, default=train.Split.valid_days)
    parser.add_argument("--window-days", type=float, default=WINDOW_DAYS)
    parser.add_argument(
        "--move-pointer",
        default="true",
        help='"false": publish without pointing the website at it (a release does that once the '
        "new image has passed its smoke test on it)",
    )
    args = parser.parse_args(argv)
    move_pointer = args.move_pointer.lower() == "true"

    import mlflow
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from heavden.ml import batch_scoring, scoring
    from heavden.ml.training import load_exclusions, to_utc
    from heavden.pipelines.gold import require_current

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    c = args.catalog
    require_current(
        spark.read.table(f"{c}.silver.device_readings"), spark.read.table(f"{c}.silver.vitals")
    )

    def read(frame) -> pd.DataFrame:
        times = tuple(name for name, kind in frame.dtypes if kind.startswith("timestamp"))
        return to_utc(frame.toPandas(), times)

    census = read(
        spark.read.table(f"{c}.gold.patient_hour_features").groupBy("prediction_ts").count()
    )
    as_of = full_ward_hour(census.set_index("prediction_ts")["count"].sort_index())
    start = as_of - pd.Timedelta(days=args.window_days)
    tables = {
        name: read(
            spark.read.table(f"{c}.gold.{name}").where(
                (F.col(column) > start.to_pydatetime()) & (F.col(column) <= as_of.to_pydatetime())
            )
        )
        for name, column in TIME_COLUMNS.items()
    }
    tables["encounters"] = read(spark.read.table(f"{c}.gold.encounters"))
    tables = website_tables(tables, as_of, args.window_days)

    mlflow.set_registry_uri("databricks-uc")
    name = f"{c}.ml.deterioration_risk"
    model, threshold, version = batch_scoring.load_registered_model(name, args.model_alias)
    training_set = read(spark.read.table(f"{c}.gold.training_set"))
    split = train.Split(args.train_days, args.valid_days)
    training_rows = batch_scoring.model_blocks(
        training_set, name, version, split, load_exclusions(c)
    )[0]

    out = Path(tempfile.mkdtemp()) / "snapshot"
    bands = scoring.RiskBands(high=threshold)
    manifest = write_snapshot(out, tables, model, training_rows, bands, name, version, as_of)
    Snapshot.load(out)  # it loads, and its model gives its check hours' risks
    logged = log_check(model, version, tables, as_of)

    snapshot = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-v{version}"
    if args.volume:
        publish_snapshot(out, Path(args.volume), snapshot, move_pointer)
    pointed = bool(args.volume) and move_pointer
    result = {
        "snapshot": snapshot,
        "published": bool(args.volume),
        "pointer_moved": pointed,
        "volume": args.volume or None,
        "as_of": manifest.as_of,
        "data_start": manifest.data_start,
        "model_version": manifest.model_version,
        "threshold": threshold,
        "rows": manifest.rows,
        "log_check": logged,
    }
    for table in GOLD_TABLES:
        print(f"  gold.{table}: {manifest.rows[table]:,} rows")
    print(
        f"{name} v{version} (threshold {threshold:.4f}), {manifest.data_start} to {manifest.as_of}"
        f"; check hours re-scored identically; {logged}; "
        + (f"published to {args.volume}/snapshots/{snapshot}" if args.volume else "not published")
        + ("; the website serves it from its next start" if pointed else "")
    )
    print(f"WEBSITE_SNAPSHOT {json.dumps(result)}")
