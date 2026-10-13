"""The drift check: compare the latest data with what the champion was trained on (Plan.md §10).

Run after each load of new data (a drift episode), it measures five kinds of signal, all sites
together and per site, each with a warn and an act level:

* **inputs**: PSI of key features, latest day vs the champion's own data (its training and
  validation blocks: what it was fitted and calibrated on);
* **predictions**: Jensen-Shannon distance of the risk scores, same comparison (both scored
  by the model being judged, not taken from the log, which may hold older versions' scores);
* **alert load**: alerts per nurse per shift in the latest day (known at once, no labels);
* **performance**: AUPRC and Brier score of the model on the latest labelled hours vs its own
  validation scores. Labels arrive ~12 h late, so this window ends earlier;
* **data quality**: share of readings quarantined, and how far each field's missing share has
  moved since training (a firmware update that drops or adds a field).

The check measures and reports; it doesn't decide. A sensor fault (inputs drift at one site)
needs the sensors fixed, not a new model; a change in outcomes (performance drops while the
inputs stay put) is what retraining is for. A person reads the report and chooses (§10.4).

Thresholds start from Plan.md §10.3 and were checked offline (12 Oct) on a rehearsal: prod's
baseline followed by a sensor-bias and a protocol-change episode (`generator.episode`, seeds
3001 and 3002; prod's live episodes use other seeds, so they test these thresholds rather than
repeat the rehearsal), scored with the Phase 0 champion and checked every 12 hours: on normal
days nothing reaches act (at most a warn, Site B's SpO2 PSI up to 0.15); the sensor bias
reaches act on Site B's SpO2 PSI (1.2), prediction JS and alert load (3.3), with performance
unchanged; the protocol change reaches act on AUPRC (-44%) and Brier (+36%) while the inputs
stay stable. Two changes from §10.3: prediction JS
0.08 / 0.12 (normal days reached 0.063 per site), and the performance signals use the
bootstrap bound described in `performance`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from heavden.ml import evaluate
from heavden.ml.features import VITALS
from heavden.monitoring import drift_metrics

FEATURES = (*(f"{v}_mean_1h" for v in VITALS), "news2_total", "readings_1h")
ALL_SITES = "all"
WINDOW = pd.Timedelta(hours=24)  # a whole day, so the time of day doesn't look like drift
LABEL_WINDOW = pd.Timedelta(hours=48)  # escalations are rare: ~10 a day in prod
MIN_ESCALATIONS = 10  # in the labelled window, for AUPRC and Brier


@dataclass(frozen=True)
class Threshold:
    warn: float
    act: float

    def level(self, value: float) -> str:
        if np.isnan(value):
            return "not enough data"
        return "act" if value >= self.act else "warn" if value >= self.warn else "stable"


THRESHOLDS = {
    "feature_psi": Threshold(drift_metrics.PSI_WARN, drift_metrics.PSI_ACT),
    "prediction_js": Threshold(0.08, 0.12),
    "alerts_per_nurse_shift": Threshold(2.5, 3.0),  # the budget is 2
    "auprc_drop": Threshold(0.10, 0.20),  # relative to validation
    "brier_rise": Threshold(0.10, 0.20),  # relative to validation
    "quarantine_rate": Threshold(0.02, 0.05),
    "missing_rate_change": Threshold(0.02, 0.05),  # absolute change in a field's missing share
}


def _row(signal, value, n, feature=None, site=ALL_SITES, reference=np.nan, observed=None):
    """One signal. `value` is what the level is judged on; `observed` the measured quantity
    when they differ (e.g. AUPRC itself, for an AUPRC drop)."""
    return {
        "signal": signal,
        "feature": feature,
        "site_id": site,
        "value": float(value),
        "observed": float(value if observed is None else observed),
        "reference": float(reference),
        "n": int(n),
        "level": THRESHOLDS[signal].level(value),
    }


def _slices(df: pd.DataFrame):
    yield ALL_SITES, df
    yield from df.groupby("site_id")


def _same_site(baseline: pd.DataFrame, site: str) -> pd.DataFrame:
    return baseline if site == ALL_SITES else baseline[baseline["site_id"] == site]


def input_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    """PSI per feature and site, current window vs the champion's data."""
    rows = []
    for site, now in _slices(current):
        then = _same_site(baseline, site)
        for f in FEATURES:
            value = drift_metrics.psi(then[f].to_numpy(), now[f].to_numpy())
            rows.append(_row("feature_psi", value, now[f].notna().sum(), f, site))
    return rows


def prediction_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    """JS distance of the risk scores per site; both frames need `site_id` and `risk`."""
    rows = []
    for site, now in _slices(current):
        then = _same_site(baseline, site)
        value = drift_metrics.js_distance(then["risk"].to_numpy(), now["risk"].to_numpy())
        rows.append(_row("prediction_js", value, len(now), site=site))
    return rows


def alert_load(
    log: pd.DataFrame, start: pd.Timestamp, budget: evaluate.AlertBudget | None = None
) -> list[dict]:
    """Alerts per nurse per shift from the prediction log, for hours from `start` on.

    Counted as on the ward: a stay entering High (each hour's stored band, whichever model
    scored it), 6 h re-alert suppression. Earlier hours are used too, so a stay already High
    before the window doesn't count as entering it."""
    budget = budget or evaluate.AlertBudget()
    onset = evaluate.alert_onsets(log, log["risk_band"].eq("High").to_numpy())
    window = log.assign(alert=onset)[log["prediction_ts"] >= start]
    rows = []
    for site, part in _slices(window):
        value = np.nan
        if len(part) >= drift_metrics.MIN_SAMPLES:
            value = budget.per_nurse_shift(part["alert"].sum(), len(part))
        rows.append(_row("alerts_per_nurse_shift", value, len(part), site=site))
    return rows


def bootstrap_scores(labelled: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """AUPRC and Brier on resamples of whole stays (a stay's hours move together)."""
    y, risk = labelled["label"].to_numpy(), labelled["risk"].to_numpy()
    codes, stays = pd.factorize(labelled["encounter_id"])
    order = np.argsort(codes, kind="stable")
    rows_of = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(stays), len(stays))])
        yb, rb = y[idx], risk[idx]
        if yb.min() < yb.max():
            out.append((average_precision_score(yb, rb), brier_score_loss(yb, rb)))
    return pd.DataFrame(out, columns=["auprc", "brier"])


def performance(
    labelled: pd.DataFrame, reference: dict[str, float], n_boot: int = 200, seed: int = 0
) -> list[dict]:
    """AUPRC drop and Brier rise on labelled hours vs the champion's validation, both relative.

    Two days hold only ~20 escalations, so a window's AUPRC swings by a third on normal days
    (measured offline). The levels therefore use the change we can be 95% sure of: the drop
    from the top of AUPRC's bootstrap interval, the rise from the bottom of Brier's (resampling
    whole stays). `labelled` holds the logged `risk` and the final `label`; `reference` the
    validation `auprc` and `brier`."""
    y, risk = labelled["label"].to_numpy(), labelled["risk"].to_numpy()
    escalations = len(evaluate.warning_times(labelled, risk, 1.0))  # runs of positive hours
    auprc = brier = drop = rise = np.nan
    if escalations >= MIN_ESCALATIONS:
        auprc, brier = average_precision_score(y, risk), brier_score_loss(y, risk)
        boot = bootstrap_scores(labelled, n_boot, seed)
        drop = max(1 - boot["auprc"].quantile(0.95) / reference["auprc"], 0.0)
        rise = max(boot["brier"].quantile(0.05) / reference["brier"] - 1, 0.0)
    return [
        _row("auprc_drop", drop, escalations, reference=reference["auprc"], observed=auprc),
        _row("brier_rise", rise, escalations, reference=reference["brier"], observed=brier),
    ]


@dataclass(frozen=True)
class ReadingQuality:
    """Device readings in the window, and each field's missing share then and now."""

    readings: int
    quarantined: int
    missing_now: pd.Series  # by field
    missing_in_training: pd.Series  # by field, over the champion's training and validation


def data_quality(q: ReadingQuality) -> list[dict]:
    """Share of readings quarantined, and how far each field's missing share has moved (a
    firmware update that drops a field, or starts sending a new one, moves it to ~100%)."""
    total = q.readings + q.quarantined
    enough = total >= drift_metrics.MIN_SAMPLES
    rows = [_row("quarantine_rate", q.quarantined / total if enough else np.nan, total)]
    for field, before in q.missing_in_training.items():
        now = q.missing_now[field] if enough else np.nan
        change = abs(now - before)
        rows.append(
            _row("missing_rate_change", change, q.readings, field, reference=before, observed=now)
        )
    return rows


def run_check(
    model,
    features: pd.DataFrame,
    log: pd.DataFrame,
    training_set: pd.DataFrame,
    baseline: pd.DataFrame,
    reference: dict[str, float],
    quality: ReadingQuality,
    window: pd.Timedelta = WINDOW,
    label_window: pd.Timedelta = LABEL_WINDOW,
) -> pd.DataFrame:
    """Every signal, one row each, for the latest `window` of patient-hours.

    The check judges one model, `model` (its validation scores are `reference`: `auprc` and
    `brier`), and scores everything it compares itself: its reference rows (`baseline`: the
    model's training and validation rows), the latest window of `features`, and the latest
    `label_window` of the labelled `training_set` that comes after `baseline`. The prediction
    log can hold older versions' scores (each hour is scored once), which would compare one
    model's predictions with another's validation. Only the alert load comes from the log
    (`gold.risk_scores`): it's what the nurses received."""

    def scored(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(risk=model.predict_proba(df)[:, 1] if len(df) else [])

    start = features["prediction_ts"].max() - window + pd.Timedelta(hours=1)
    current = scored(features[features["prediction_ts"] >= start])
    reference_rows = scored(baseline)
    # Performance is judged only on hours after the model's own data: on the hours it was fitted
    # and calibrated on, it would be marking its own homework.
    hours = training_set["prediction_ts"]
    unseen = hours > max(hours.max() - label_window, baseline["prediction_ts"].max())
    labelled = scored(training_set[unseen])
    rows = (
        input_drift(reference_rows, current)
        + prediction_drift(reference_rows, current)
        + alert_load(log, start)
        + performance(labelled, reference)
        + data_quality(quality)
    )
    return pd.DataFrame(rows)


def summary(metrics: pd.DataFrame) -> dict:
    """The check's overall level (its worst signal) and the breaches, act first."""
    breaches = metrics[metrics["level"].isin(["warn", "act"])]
    breaches = breaches.assign(rank=breaches["level"].map({"act": 0, "warn": 1}))
    breaches = breaches.sort_values(["rank", "value"], ascending=[True, False])
    n_act = int((breaches["level"] == "act").sum())
    return {
        "status": "act" if n_act else "warn" if len(breaches) else "stable",
        "n_warn": len(breaches) - n_act,
        "n_act": n_act,
        "breaches": [describe(r) for r in breaches.itertuples()],
    }


RESULT_MARKER = "DRIFT_CHECK_RESULT"


def result_line(check: dict, result: dict) -> str:
    """The check on one line, as JSON after `RESULT_MARKER`: what the workflow that started the
    job reads from its output to decide whether to open a retrain request (act)."""
    payload = {k: v if isinstance(v, int | list | type(None)) else str(v) for k, v in check.items()}
    return f"{RESULT_MARKER} {json.dumps(payload | result)}"


def describe(r) -> str:
    """e.g. "feature_psi spo2_mean_1h at SITE_B: 1.199 (act)"."""
    what = f"{r.signal} {r.feature}" if isinstance(r.feature, str) else r.signal
    where = "" if r.site_id == ALL_SITES else f" at {r.site_id}"
    return f"{what}{where}: {r.value:.3f} ({r.level})"


# --- The Databricks job ----------------------------------------------------------------------

# Explicit, so a check with no breaches or no episode still writes typed columns.
CHECKS_SCHEMA = (
    "check_id string, checked_at timestamp, episode string, model_version string, "
    "window_start timestamp, window_end timestamp, labels_through timestamp, status string, "
    "n_warn int, n_act int, breaches array<string>"
)


def validation_metrics(model_name: str, version: str) -> dict[str, float]:
    """The registered version's validation scores, from its MLflow run."""
    import mlflow

    client = mlflow.MlflowClient()
    run = client.get_run(client.get_model_version(model_name, version).run_id)
    return {k: run.data.metrics[f"valid_{k}"] for k in ("auprc", "brier")}


def reading_quality(spark, catalog: str, training: tuple, window: tuple) -> ReadingQuality:
    """Missing shares per field in Silver's readings, over the champion's data and the window
    (each a [start, end) pair), and the quarantined readings in the window."""
    from pyspark.sql import functions as F

    from heavden.pipelines.silver import READING_FIELDS

    fields = [f for f in READING_FIELDS if f not in ("device_id", "ts")]
    readings = spark.read.table(f"{catalog}.silver.device_readings")

    def between(df, start, end):
        ts = F.col("ts")
        return df.where((ts >= F.lit(start.to_pydatetime())) & (ts < F.lit(end.to_pydatetime())))

    def missing(start, end):
        shares = [F.avg(F.col(f).isNull().cast("double")).alias(f) for f in fields]
        row = between(readings, start, end).agg(F.count("*").alias("n"), *shares).toPandas()
        return int(row["n"].iloc[0]), row[fields].iloc[0].astype(float).fillna(0.0)

    quarantine = spark.read.table(f"{catalog}.silver.vitals_quarantine")
    n_now, now = missing(*window)
    return ReadingQuality(
        readings=n_now,
        quarantined=between(quarantine, *window).count(),
        missing_now=now,
        missing_in_training=missing(*training)[1],
    )


def main(argv: list[str] | None = None) -> None:
    import argparse
    import uuid

    import mlflow
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from heavden.ml import batch_scoring, train
    from heavden.ml.training import to_utc

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_prod")
    parser.add_argument("--model-alias", default="champion", help='an alias, or "latest"')
    parser.add_argument("--train-days", type=float, default=train.Split.train_days)
    parser.add_argument("--valid-days", type=float, default=train.Split.valid_days)
    parser.add_argument(  # may arrive empty: the job parameter defaults to ""
        "--episode", nargs="?", default="", const="", help="the drift episode just loaded"
    )
    args = parser.parse_args(argv)

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    c = args.catalog

    def read(table: str, times: tuple[str, ...]) -> pd.DataFrame:
        return to_utc(spark.read.table(f"{c}.{table}").toPandas(), times)

    features = read("gold.patient_hour_features", ("prediction_ts",))
    training_set = read("gold.training_set", ("prediction_ts", "label_known_at"))
    log = read("gold.risk_scores", ("prediction_ts",))

    mlflow.set_registry_uri("databricks-uc")
    name = f"{c}.ml.deterioration_risk"
    model, _, version = batch_scoring.load_registered_model(name, args.model_alias)
    # The model's own training and validation blocks. Training alone is 18 hours in staging,
    # too short a reference: normal days then read as drift.
    split = train.Split(args.train_days, args.valid_days)
    baseline = pd.concat(batch_scoring.model_blocks(training_set, name, version, split))

    hour = pd.Timedelta(hours=1)
    end = features["prediction_ts"].max() + hour
    start = end - WINDOW
    quality = reading_quality(
        spark,
        c,
        training=(baseline["prediction_ts"].min() - hour, baseline["prediction_ts"].max()),
        window=(start - hour, end - hour),
    )
    reference = validation_metrics(name, version)
    metrics = run_check(model, features, log, training_set, baseline, reference, quality)
    result = summary(metrics)

    check = {
        "check_id": str(uuid.uuid4()),
        "checked_at": pd.Timestamp.now(tz="UTC"),
        "episode": args.episode or None,
        "model_version": version,
        "window_start": start,
        "window_end": end,
        "labels_through": training_set["prediction_ts"].max(),
    }
    rows = metrics.assign(**check)[[*check, *metrics.columns]]
    rows = batch_scoring.to_spark(spark, rows)
    rows = rows.withColumn("episode", F.col("episode").cast("string"))  # NULL without episode
    rows.write.mode("append").saveAsTable(f"{c}.monitoring.drift_metrics")
    summary_row = spark.createDataFrame(pd.DataFrame([check | result]), CHECKS_SCHEMA)
    summary_row.write.mode("append").saveAsTable(f"{c}.monitoring.drift_checks")

    print(
        f"{name} v{version} ({args.model_alias}), window {start:%d %b %H:%M} to {end:%d %b %H:%M},"
        f" labels through {check['labels_through']:%d %b %H:%M}: {result['status'].upper()}"
        f" ({result['n_act']} act, {result['n_warn']} warn)"
    )
    for line in result["breaches"]:
        print(" ", line)
    print(result_line(check | {"catalog": c}, result))
