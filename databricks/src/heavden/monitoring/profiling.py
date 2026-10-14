"""Lakehouse Monitoring (data profiling) of the prediction log (Plan.md §10.5).

The monitor profiles `monitoring.inference_log`: one row per scored patient-hour with the drift
check's eight features, the logged risk and band (what the nurses received, under the model
that scored the hour), the risk's decile and, once known, the label. Its baseline,
`monitoring.inference_baseline`, holds each model version's own training and validation rows
scored by that version, so every hour is compared with the reference of the model that
scored it.

* **Written by the score job.** New hours are appended and labels filled in as they become
  known (one MERGE; change data feed is on, so a refresh recomputes only the days that
  changed). The first run backfills the whole log, one model version at a time.
* **Refreshed by the drift check**, which then reads the monitor's numbers. Stage 1 records
  them next to its own as shadow rows that never change a decision, and a monitor that fails
  or lags is reported, not raised: it must not stop the scoring or the check that guard the
  ward (`MONITOR_DECIDES`). Stage 2 switches the performance, alert-load and prediction
  signals once their limits are measured, and from then on a failure stops the check.
* **Text prediction and label** ("1"/"0"), so the monitor counts their values.
* **The custom metrics are computed on `risk`** (Brier, AUPRC, AUROC, alerts per nurse per
  shift), so nothing depends on the monitor's probability column, which the newer "data
  quality" API doesn't have. The monitor is created through the older "quality monitors" API,
  which the dev test proved (14 Oct), with only the settings both APIs share: switching is
  `data_quality_config` (Plan.md §19).
* **Created by job code, not a bundle resource:** a monitor needs its table, which the score
  job writes first. The job's identity (each environment's service principal) owns it.

The monitor's own PSI is not used: it inflates on columns with tied values (NEWS2 read ~7 on
normal days in the dev test). The drift check keeps its tested PSI for the inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from heavden.ml import evaluate
from heavden.monitoring import drift_metrics
from heavden.monitoring.drift_check import ALL_SITES, FEATURES

KEYS = ["encounter_id", "prediction_ts"]
INFERENCE_TABLE = "monitoring.inference_log"
BASELINE_TABLE = "monitoring.inference_baseline"
GRANULARITIES = ("1 day", "1 week")
SLICES = ("site_id",)
COLUMNS = [
    "prediction_ts",
    "encounter_id",
    "site_id",
    "model_version",
    *FEATURES,
    "risk",
    "risk_band",
    "risk_decile",
    "predicted",
    "label",
    "alert_onset",
]
SCHEMA = (
    "prediction_ts timestamp, encounter_id string, site_id string, model_version string, "
    + ", ".join(f"{f} double" for f in FEATURES)
    + ", risk double, risk_band string, risk_decile string, predicted string, label string, "
    "alert_onset int"
)


# --- The rows the monitor reads ---------------------------------------------------------------


def deciles(risk: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each risk's bin among the reference deciles, as text ("d0".."d9"): the same bins as
    `drift_metrics.bin_shares`, so the monitor's JS on it compares with ours."""
    index = np.searchsorted(edges[1:-1], np.asarray(risk, dtype=float), side="right")
    return np.char.add("d", index.astype(str)).astype(object)


def _text(flags: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(flags, dtype=bool), "1", "0").astype(object)


def inference_rows(
    log: pd.DataFrame,
    features: pd.DataFrame,
    training_set: pd.DataFrame,
    alerts: pd.DataFrame,
    edges: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Monitored rows for prediction-log rows `log`: features from Gold, the label where
    `training_set` holds it (final labels only), the alert onsets of `alerts_fact`, and the
    risk decile against the reference of the version that scored the hour (`edges`)."""
    rows = log[[*KEYS, "site_id", "model_version", "risk", "risk_band"]].merge(
        features[[*KEYS, *FEATURES]], on=KEYS, how="left"
    )
    labels = training_set[[*KEYS, "label"]].assign(label=lambda d: _text(d["label"] == 1))
    rows = rows.merge(labels, on=KEYS, how="left")
    onsets = alerts[["encounter_id", "alert_ts"]].rename(columns={"alert_ts": "prediction_ts"})
    rows = rows.merge(onsets.assign(alert_onset=1), on=KEYS, how="left")
    rows["alert_onset"] = rows["alert_onset"].fillna(0).astype(int)
    rows["predicted"] = _text(rows["risk_band"] == "High")
    rows["risk_decile"] = None
    for version, part in rows.groupby("model_version"):
        rows.loc[part.index, "risk_decile"] = deciles(part["risk"], edges[version])
    rows["label"] = rows["label"].astype(object).where(rows["label"].notna(), None)
    return rows[COLUMNS]


def reference_rows(
    reference: pd.DataFrame, risk: np.ndarray, threshold: float, version: str
) -> pd.DataFrame:
    """A version's baseline: its training and validation rows (`reference`, from
    `gold.training_set`) scored by it (`risk`), with its own alert threshold."""
    high = np.asarray(risk) >= threshold
    rows = reference[[*KEYS, "site_id", *FEATURES]].assign(
        model_version=version,
        risk=risk,
        risk_band=np.where(high, "High", "Low"),
        predicted=_text(high),
        label=_text(reference["label"] == 1),
        alert_onset=evaluate.alert_onsets(reference, high).astype(int),
    )
    rows["risk_decile"] = deciles(rows["risk"], drift_metrics.baseline_bins(rows["risk"]))
    return rows[COLUMNS]


def to_write(rows: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """The rows a MERGE has to send: hours not in the table yet, and hours whose label became
    known since (`existing` holds the table's keys and labels; None: no table yet)."""
    if existing is None or existing.empty:
        return rows
    known = rows.merge(existing[[*KEYS, "label"]], on=KEYS, how="left", indicator=True)
    new = known["_merge"] == "left_only"
    labelled = known["label_y"].isna() & known["label_x"].notna() & ~new
    return rows[(new | labelled).to_numpy()]


MERGE = """MERGE INTO {table} t USING {source} s
ON t.encounter_id = s.encounter_id AND t.prediction_ts = s.prediction_ts
WHEN MATCHED AND (t.model_version <> s.model_version OR t.risk <> s.risk
  OR (t.label IS NULL AND s.label IS NOT NULL)) THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *"""


# --- Custom metrics -----------------------------------------------------------------------------

_LABELLED = "CASE WHEN label IS NOT NULL THEN named_struct('s', {s}, 'y', CAST(label AS INT)) END"

# Average precision as sklearn computes it: over distinct scores, highest first, the new true
# positives at that score times the precision there, divided by all positives. One expression
# (no joins or subqueries): sort the labelled rows, walk them once, and close a group of tied
# scores when the score changes.
AUPRC = f"""aggregate(
  array_sort(collect_list({_LABELLED.format(s="-risk")})),
  named_struct('tp', 0, 'fp', 0, 's', CAST(NULL AS DOUBLE), 'total', CAST(0 AS DOUBLE),
    'tp_closed', 0),
  (acc, x) -> CASE WHEN acc.s IS NOT NULL AND x.s <> acc.s
    THEN named_struct('tp', acc.tp + x.y, 'fp', acc.fp + 1 - x.y, 's', x.s,
      'total', acc.total + (acc.tp - acc.tp_closed) * acc.tp / (acc.tp + acc.fp),
      'tp_closed', acc.tp)
    ELSE named_struct('tp', acc.tp + x.y, 'fp', acc.fp + 1 - x.y, 's', x.s,
      'total', acc.total, 'tp_closed', acc.tp_closed) END,
  acc -> CASE WHEN acc.tp > 0
    THEN (acc.total + (acc.tp - acc.tp_closed) * acc.tp / (acc.tp + acc.fp)) / acc.tp END
)"""

# AUROC as the share of (escalated, not escalated) pairs ranked the right way round, ties
# counting half: lowest score first, each group of tied scores adds its positives times the
# negatives below it plus half the negatives tied with it.
AUROC = f"""aggregate(
  array_sort(collect_list({_LABELLED.format(s="risk")})),
  named_struct('s', CAST(NULL AS DOUBLE), 'pos', 0, 'neg', 0, 'below', 0,
    'total', CAST(0 AS DOUBLE), 'p', 0, 'n', 0),
  (acc, x) -> CASE WHEN acc.s IS NOT NULL AND x.s <> acc.s
    THEN named_struct('s', x.s, 'pos', x.y, 'neg', 1 - x.y, 'below', acc.below + acc.neg,
      'total', acc.total + acc.pos * (acc.below + 0.5 * acc.neg), 'p', acc.p + x.y,
      'n', acc.n + 1 - x.y)
    ELSE named_struct('s', x.s, 'pos', acc.pos + x.y, 'neg', acc.neg + 1 - x.y,
      'below', acc.below, 'total', acc.total, 'p', acc.p + x.y, 'n', acc.n + 1 - x.y) END,
  acc -> CASE WHEN acc.p > 0 AND acc.n > 0
    THEN (acc.total + acc.pos * (acc.below + 0.5 * acc.neg)) / (CAST(acc.p AS DOUBLE) * acc.n)
  END
)"""

_BUDGET = evaluate.AlertBudget()
CUSTOM_METRICS = {
    "brier": "avg(CASE WHEN label IS NOT NULL THEN pow(risk - CAST(label AS DOUBLE), 2) END)",
    "auprc": AUPRC,
    "auroc": AUROC,
    # alert onsets per patient-hour x patient-hours per nurse shift (evaluate.AlertBudget)
    "alerts_per_nurse_shift": (
        f"{_BUDGET.patients_per_nurse * _BUDGET.shift_hours} * avg(alert_onset)"
    ),
}
_DOUBLE = '{"name": "output", "type": "double", "nullable": true, "metadata": {}}'


# --- The monitor ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorSpec:
    catalog: str
    assets_dir: str  # workspace folder for the monitor's assets (none without the dashboard)

    @property
    def table(self) -> str:
        return f"{self.catalog}.{INFERENCE_TABLE}"

    @property
    def baseline(self) -> str:
        return f"{self.catalog}.{BASELINE_TABLE}"

    @property
    def output_schema(self) -> str:
        return f"{self.catalog}.monitoring"

    def output_table(self, kind: str) -> str:
        """The monitor's `profile` or `drift` metrics table."""
        return f"{self.table}_{kind}_metrics"


def legacy_settings(spec: MonitorSpec) -> dict:
    """Settings for the "quality monitors" API's create and update (only those the newer API
    also has: no probability column)."""
    from databricks.sdk.service import catalog as c

    return {
        "output_schema_name": spec.output_schema,
        "baseline_table_name": spec.baseline,
        "inference_log": c.MonitorInferenceLog(
            problem_type=c.MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            timestamp_col="prediction_ts",
            granularities=list(GRANULARITIES),
            prediction_col="predicted",
            label_col="label",
            model_id_col="model_version",
        ),
        "slicing_exprs": list(SLICES),
        "custom_metrics": [
            c.MonitorMetric(
                type=c.MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name=name,
                input_columns=[":table"],
                definition=definition,
                output_data_type=_DOUBLE,
            )
            for name, definition in CUSTOM_METRICS.items()
        ],
    }


def data_quality_config(spec: MonitorSpec, output_schema_id: str):
    """The same monitor for the newer "data quality" API (Public Preview), for the day the
    older one is retired: `w.data_quality.create_monitor(Monitor(object_type="table", ...))`."""
    from databricks.sdk.service import dataquality as dq

    granularity = {
        "1 day": dq.AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY,
        "1 week": dq.AggregationGranularity.AGGREGATION_GRANULARITY_1_WEEK,
    }
    return dq.DataProfilingConfig(
        output_schema_id=output_schema_id,
        assets_dir=spec.assets_dir,
        baseline_table_name=spec.baseline,
        skip_builtin_dashboard=True,
        slicing_exprs=list(SLICES),
        inference_log=dq.InferenceLogConfig(
            problem_type=dq.InferenceProblemType.INFERENCE_PROBLEM_TYPE_CLASSIFICATION,
            timestamp_column="prediction_ts",
            granularities=[granularity[g] for g in GRANULARITIES],
            prediction_column="predicted",
            label_column="label",
            model_id_column="model_version",
        ),
        custom_metrics=[
            dq.DataProfilingCustomMetric(
                type=dq.DataProfilingCustomMetricType.DATA_PROFILING_CUSTOM_METRIC_TYPE_AGGREGATE,
                name=name,
                input_columns=[":table"],
                definition=definition,
                output_data_type=_DOUBLE,
            )
            for name, definition in CUSTOM_METRICS.items()
        ],
    )


def _fingerprint(baseline, inference_log, slicing, metrics) -> tuple:
    log = inference_log
    return (
        baseline,
        tuple(log.granularities or ()),
        (log.prediction_col, log.label_col, log.model_id_col, log.timestamp_col),
        tuple(slicing or ()),
        tuple(sorted((m.name, m.definition) for m in metrics or ())),
    )


def settings_differ(info, spec: MonitorSpec) -> bool:
    """Whether an existing monitor (`MonitorInfo`) has other settings than `spec`."""
    want = legacy_settings(spec)
    return _fingerprint(
        info.baseline_table_name, info.inference_log, info.slicing_exprs, info.custom_metrics
    ) != _fingerprint(
        want["baseline_table_name"],
        want["inference_log"],
        want["slicing_exprs"],
        want["custom_metrics"],
    )


def ensure_monitor(w, spec: MonitorSpec) -> str:
    """Create the monitor if it's missing, update it if its settings changed. Returns what was
    done: "created", "updated" or "unchanged"."""
    from databricks.sdk.errors import NotFound

    if not w.tables.exists(spec.table).table_exists:
        raise LookupError(f"{spec.table} doesn't exist yet: run the score job first")
    try:
        info = w.quality_monitors.get(spec.table)
    except NotFound:
        w.quality_monitors.create(
            table_name=spec.table,
            assets_dir=spec.assets_dir,
            skip_builtin_dashboard=True,  # the Model & Drift dashboard replaces it
            **legacy_settings(spec),
        )
        return "created"
    if settings_differ(info, spec):
        w.quality_monitors.update(table_name=spec.table, **legacy_settings(spec))
        return "updated"
    return "unchanged"


ACTIVE = ("PENDING", "RUNNING")


def _state(refresh) -> str:
    return refresh.state.value if refresh.state is not None else "UNKNOWN"


def refresh_and_wait(
    w, table: str, just_created: bool, poll_s: float = 30, timeout_s: float = 2700
):
    """Refresh the monitor and wait for it; raise unless it succeeded.

    Creating a monitor starts a refresh after the data was written: then that one is awaited.
    Any other refresh in progress may have started before this run's data, so it is awaited
    and a new one started."""
    from databricks.sdk.service.catalog import MonitorRefreshInfo

    def active() -> list[MonitorRefreshInfo]:
        listed = w.quality_monitors.list_refreshes(table).refreshes or []
        return [r for r in listed if _state(r) in ACTIVE]

    def wait(refresh: MonitorRefreshInfo) -> MonitorRefreshInfo:
        deadline = time.monotonic() + timeout_s
        while _state(refresh) in ACTIVE:
            if time.monotonic() > deadline:
                raise TimeoutError(f"refresh {refresh.refresh_id} of {table} still running")
            time.sleep(poll_s)
            refresh = w.quality_monitors.get_refresh(table, refresh.refresh_id)
        return refresh

    running = active()
    if just_created:
        for _ in range(4):  # the create's own refresh can take a moment to be listed
            if running:
                break
            time.sleep(poll_s)
            running = active()
    if just_created and running:
        done = wait(running[0])
    else:
        for other in running:
            wait(other)
        done = wait(w.quality_monitors.run_refresh(table))
    if _state(done) != "SUCCESS":
        raise RuntimeError(f"refresh {done.refresh_id} of {table}: {_state(done)} {done.message}")
    return done


# --- Reading the monitor's numbers (stage 1: shadow rows) --------------------------------------


# Stage 1 (Plan.md §10.5): the monitor's numbers are recorded, not judged, so its failures are
# reported and the score job and the drift check carry on. Stage 2, when the numbers decide,
# sets this to True in the same change as their limits: then a failure stops the check.
MONITOR_DECIDES = False


class MonitorBehind(RuntimeError):
    """The monitor's latest day ends before the data: its numbers would describe the past."""


SHADOW = "shadow"  # never warn or act: recorded next to the check's own signals


def _shadow(signal, value, n, site, version=None) -> dict:
    return {
        "signal": signal,
        "feature": None if version is None else f"v{version}",
        "site_id": site,
        "value": float(value),
        "observed": float(value),
        "reference": float("nan"),
        "n": int(n),
        "level": SHADOW,
    }


def _latest(df: pd.DataFrame, granularity: str, end: pd.Timestamp) -> pd.DataFrame:
    """Rows of the latest whole window of `granularity` ending by `end`."""
    part = df[(df["granularity"] == granularity) & (df["window_end"] <= end)]
    return part[part["window_end"] == part["window_end"].max()] if len(part) else part


def _by_site(rows: pd.DataFrame) -> dict[str, pd.Series]:
    site = rows["slice_value"].where(rows["slice_key"] == "site_id", ALL_SITES)
    keep = rows["slice_key"].isna() | (rows["slice_key"] == "site_id")
    return {s: r for s, (_, r) in zip(site[keep], rows[keep].iterrows(), strict=True)}


def monitor_signals(profile: pd.DataFrame, drift: pd.DataFrame, end: pd.Timestamp) -> list[dict]:
    """The monitor's numbers for the day (and week) ending at `end`, as shadow rows.

    `profile`: table-level rows of the profile metrics (`column_name = ':table'`, `INPUT`);
    `drift`: the risk deciles' drift against the baseline. Alert load and performance are read
    over all model versions ("*": what the nurses received); prediction drift for the version
    that scored most of the day, against its own baseline. Raises `MonitorBehind` when no day
    window covers the data's last hour."""
    days = profile[profile["granularity"] == "1 day"]
    last_hour = end - pd.Timedelta(hours=1)
    if not ((days["window_start"] <= last_hour) & (days["window_end"] > last_hour)).any():
        latest = days["window_end"].max() if len(days) else None
        raise MonitorBehind(
            f"the monitor's latest day ends {latest}, the data at {end}: "
            "were the new hours scored (score job) before the check?"
        )

    rows = []
    day = _latest(profile, "1 day", end)
    every = day[day["model_version"] == "*"]
    for site, r in _by_site(every).items():
        value = r["alerts_per_nurse_shift"]
        rows.append(_shadow("lm_alerts_per_nurse_shift", value, r["count"], site))

    versions = day[(day["model_version"] != "*") & day["slice_key"].isna()]
    if len(versions):
        version = versions.loc[versions["count"].idxmax(), "model_version"]
        counts = _by_site(day[day["model_version"] == version])
        js = _latest(drift[drift["model_version"] == version], "1 day", end)
        for site, r in _by_site(js).items():
            n = counts[site]["count"] if site in counts else 0
            rows.append(_shadow("lm_prediction_js", r["js_distance"], n, site, version))

    week = _latest(profile, "1 week", end)
    for site, r in _by_site(week[week["model_version"] == "*"]).items():
        for metric in ("auprc", "brier", "auroc"):
            rows.append(_shadow(f"lm_{metric}_week", r[metric], r["count"], site))
    return rows


def read_monitor_tables(spark, spec: MonitorSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(profile, drift) rows `monitor_signals` reads, from the monitor's output tables."""
    from pyspark.sql import functions as F

    from heavden.ml.training import to_utc

    def read(kind: str, where, columns: list[str]) -> pd.DataFrame:
        df = spark.read.table(spec.output_table(kind)).where(where)
        df = df.select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "granularity",
            "slice_key",
            "slice_value",
            "model_version",
            *columns,
        )
        return to_utc(df.toPandas(), ("window_start", "window_end"))

    profile = read(
        "profile",
        (F.col("column_name") == ":table") & (F.col("log_type") == "INPUT"),
        ["count", *CUSTOM_METRICS],
    )
    drift = read(
        "drift",
        (F.col("column_name") == "risk_decile") & (F.col("drift_type") == "BASELINE"),
        ["js_distance"],
    )
    return profile, drift


def shadow_rows(spark, spec: MonitorSpec, end: pd.Timestamp) -> tuple[list[dict], str]:
    """Refresh the monitor and read its numbers for the window ending at `end`.

    Returns (rows, status). The first row, `lm_status`, records how the monitor did ("ok: ..."
    or "failed: ..." in its `feature`), so every check leaves a trace of it. While the monitor
    doesn't decide (`MONITOR_DECIDES`), a failure is reported here instead of raised."""
    try:
        status = "ok: " + refresh_monitor(spec)
        rows = monitor_signals(*read_monitor_tables(spark, spec), end)
    except Exception as error:
        if MONITOR_DECIDES:
            raise
        status, rows = f"failed: {type(error).__name__}: {error}"[:500], []  # rides in the result
    row = _shadow("lm_status", float("nan"), 0, ALL_SITES) | {"feature": status}
    return [row, *rows], status


def refresh_monitor(spec: MonitorSpec) -> str:
    """Ensure the monitor, refresh it and wait: what the drift check runs first."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    started = time.monotonic()
    action = ensure_monitor(w, spec)
    done = refresh_and_wait(w, spec.table, just_created=action == "created")
    minutes = (time.monotonic() - started) / 60
    return (
        f"Monitor on {spec.table} {action}; refresh {done.refresh_id} succeeded "
        f"in {minutes:.1f} min"
    )


# --- The score job's part ---------------------------------------------------------------------


def update_tables(
    spark,
    catalog: str,
    log: pd.DataFrame,
    features: pd.DataFrame,
    training_set: pd.DataFrame,
    alerts: pd.DataFrame,
    reference_for,
    rescore: bool = False,
) -> str:
    """Bring `monitoring.inference_baseline` and `monitoring.inference_log` up to date with the
    prediction log `log` (the whole log, after this run's appends).

    `reference_for(version)` returns that version's baseline rows (`reference_rows`); it is
    called only for versions the baseline doesn't hold yet. After `rescore`, every hour is sent
    (the MERGE replaces rows whose model or risk changed)."""
    from heavden.ml.training import to_utc

    spec = MonitorSpec(catalog, assets_dir="")
    for table, cdf in ((spec.baseline, False), (spec.table, True)):
        props = " TBLPROPERTIES (delta.enableChangeDataFeed = true)" if cdf else ""
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table} ({SCHEMA}) USING DELTA{props}")

    base = spark.read.table(spec.baseline).select("model_version", "risk").toPandas()
    added = []
    for version in sorted(set(log["model_version"]) - set(base["model_version"])):
        rows = reference_for(version)
        _write(spark, rows).write.format("delta").mode("append").saveAsTable(spec.baseline)
        base = pd.concat([base, rows[["model_version", "risk"]]], ignore_index=True)
        added.append(version)
    edges = {v: drift_metrics.baseline_bins(b["risk"]) for v, b in base.groupby("model_version")}

    existing = None
    if not rescore:
        existing = spark.read.table(spec.table).select(*KEYS, "label").toPandas()
        existing = to_utc(existing, ("prediction_ts",))
    candidates = log  # the hours that may need sending: new ones, and ones waiting for a label
    if existing is not None and len(existing):
        unknown = existing[existing["label"].isna()][KEYS]
        seen = log[KEYS].merge(existing[KEYS], how="left", indicator=True)["_merge"]
        pending = log[KEYS].merge(unknown, how="left", indicator=True)["_merge"]
        candidates = log[((seen == "left_only") | (pending == "both")).to_numpy()]
    rows = to_write(inference_rows(candidates, features, training_set, alerts, edges), existing)
    if len(rows):
        _write(spark, rows).createOrReplaceTempView("heavden_inference_rows")
        spark.sql(MERGE.format(table=spec.table, source="heavden_inference_rows"))
    if existing is None:
        done = f"{len(rows):,} hours sent (every hour: the log was replaced)"
    else:
        filled = len(rows[KEYS].merge(existing[KEYS]))
        done = f"{len(rows) - filled:,} hours added, {filled:,} labels filled in"
    return f"{spec.table}: {done}" + (f"; baseline added for v{', v'.join(added)}" if added else "")


def _write(spark, rows: pd.DataFrame):
    """`rows` as a Spark DataFrame with the table's exact column types."""
    from pyspark.sql import functions as F

    from heavden.ml.batch_scoring import to_spark

    types = dict(part.rsplit(" ", 1) for part in SCHEMA.split(", "))
    df = to_spark(spark, rows[COLUMNS])
    return df.select(*(F.col(c).cast(types[c]).alias(c) for c in COLUMNS))
