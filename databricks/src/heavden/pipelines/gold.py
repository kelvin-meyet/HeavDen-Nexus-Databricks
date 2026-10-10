"""Gold: model features, training labels and device health (Plan.md §7.3).

Plain PySpark, run by the pipeline (`databricks/src/pipelines/medallion/gold.py`) and by the local
tests. The pandas references are `heavden.ml.features` (features, labels) and
`heavden.analytics.gold.device_health_daily`; `tests/test_gold.py` checks these match them.

Differences from the pandas reference, by design:
* **Labels use only recorded outcomes.** An escalation is recorded about 6 h after it happens,
  so a negative label at `t` is final only once `t + 6 h` (the window) `+ 6 h` (the recording
  delay) has passed the data's as-of time. A recorded escalation settles a positive at once.
* **Age and condition flags are taken at admission**, from Silver patients, conditions and
  medications (pandas reads a profile fixed at 1 Nov 2026).
* **Unit history comes from Silver encounters (SCD type 2)**: a transfer inside one load window
  is invisible there, so those hours carry the later unit (CLAUDE.md, known issues).

The data window comes from the readings themselves: the first reading's hour to the hour after
the last reading (`data_window`), so nothing needs to be configured per load.
"""

from __future__ import annotations

from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

VITALS = ("heart_rate", "resp_rate", "spo2", "temp_c", "sbp", "dbp")
HORIZON_H = 6  # label window (t, t + 6 h]
RECORDING_DELAY_H = 6  # outcomes are charted about 6 h after the event
READINGS_PER_HOUR = 12

# Which condition and medication descriptions set each flag: regexes, case-insensitive. Same
# patterns as generator.synthea, which decides how the simulated vitals behave, so the model
# sees exactly the conditions the simulation gives an effect (Data Card §2a).
CONDITION_FLAGS: dict[str, tuple[str, ...]] = {
    "copd": ("chronic obstructive", "pulmonary emphysema"),
    "heart_failure": ("heart failure",),
    "diabetes": ("(?<!pre)diabetes",),  # not "prediabetes"
    "ckd": ("chronic kidney disease",),
    "hypertension": ("hypertension",),
    "atrial_fibrillation": ("atrial fibrillation",),
}
BETA_BLOCKERS = ("metoprolol", "carvedilol", "atenolol", "propranolol", "bisoprolol", "nebivolol")

# device_health_daily (same constants as heavden.analytics.gold)
STEP_MINUTES = 5
OUTAGE_GAP_MINUTES = 25
STUCK_RUN = 6
STUCK_VITALS = ("heart_rate", "sbp", "dbp")


def _pattern(words: tuple[str, ...]) -> str:
    return "(?i)" + "|".join(words)


def _hours(later: Column, earlier: Column) -> Column:
    return (later.cast("double") - earlier.cast("double")) / 3600


def _plus_hours(ts: Column, hours: int) -> Column:
    return ts + F.expr(f"INTERVAL {hours} HOURS")


# ---------- Inputs ----------


def data_window(vitals: DataFrame) -> DataFrame:
    """One row: `data_start` (the first reading's hour) and `as_of` (the hour after the last)."""
    return vitals.agg(
        F.date_trunc("hour", F.min("ts")).alias("data_start"),
        _plus_hours(F.date_trunc("hour", F.max("ts")), 1).alias("as_of"),
    )


def unit_history(encounters_scd2: DataFrame) -> DataFrame:
    """Each stay's unit and when that version was recorded, from Silver encounters (SCD 2)."""
    return encounters_scd2.select(
        "encounter_id", "unit_id", F.col("__START_AT.last_updated").alias("changed_ts")
    )


def profile_flags(
    encounters: DataFrame, conditions: DataFrame, medications: DataFrame
) -> DataFrame:
    """Per stay: the six condition flags and `on_beta_blocker`, active at admission.

    A condition counts if it started on or before the admission date and hadn't stopped by it;
    a medication if it started before admission and hadn't stopped.
    """
    stays = encounters.select("encounter_id", "patient_id", "admit_ts")
    admit_date = F.to_date("admit_ts")
    active_conditions = stays.join(conditions, "patient_id").where(
        (F.col("start_date") <= admit_date)
        & (F.col("stop_date").isNull() | (F.col("stop_date") > admit_date))
    )
    active_meds = stays.join(medications, "patient_id").where(
        (F.col("start_ts") <= F.col("admit_ts"))
        & (F.col("stop_ts").isNull() | (F.col("stop_ts") > F.col("admit_ts")))
    )
    flags = active_conditions.groupBy("encounter_id").agg(
        *[
            F.max(F.col("description").rlike(_pattern(words))).alias(flag)
            for flag, words in CONDITION_FLAGS.items()
        ]
    )
    beta = active_meds.groupBy("encounter_id").agg(
        F.max(F.col("description").rlike(_pattern(BETA_BLOCKERS))).alias("on_beta_blocker")
    )
    out = stays.select("encounter_id").join(flags, "encounter_id", "left")
    out = out.join(beta, "encounter_id", "left")
    names = [*CONDITION_FLAGS, "on_beta_blocker"]
    out = out.fillna(False, subset=names)
    n_conditions = reduce(lambda a, b: a + b, [F.col(f).cast("int") for f in CONDITION_FLAGS])
    return out.withColumn("n_conditions", n_conditions)


def condition_flag_reference(conditions: DataFrame, medications: DataFrame) -> DataFrame:
    """Which source descriptions set each flag, and for how many patients: the audit trail
    for the inclusion rule (Data Card §2a). Descriptions matching no flag aren't listed."""
    rows = [
        conditions.where(F.col("description").rlike(_pattern(words))).select(
            F.lit(flag).alias("flag"),
            F.lit("condition").alias("source"),
            "code",
            "description",
            "patient_id",
        )  # fmt: skip
        for flag, words in CONDITION_FLAGS.items()
    ]
    rows.append(
        medications.where(F.col("description").rlike(_pattern(BETA_BLOCKERS))).select(
            F.lit("on_beta_blocker").alias("flag"),
            F.lit("medication").alias("source"),
            "code",
            "description",
            "patient_id",
        )  # fmt: skip
    )
    return (
        reduce(DataFrame.unionByName, rows)
        .groupBy("flag", "source", "code", "description")
        .agg(F.countDistinct("patient_id").alias("patients"))
        .orderBy("flag", F.desc("patients"), "description")
    )


# ---------- Patient-hour features ----------


def prediction_grid(encounters: DataFrame, window: DataFrame) -> DataFrame:
    """One row per stay per hour on the ward, within the data window.

    The first prediction time is the first whole hour after admission (so at least part of an
    hour of readings exists); times run up to, not including, discharge or `as_of`.
    """
    e = encounters.crossJoin(window)
    first = _plus_hours(F.date_trunc("hour", F.greatest("admit_ts", "data_start")), 1)
    last = F.least(F.coalesce("discharge_ts", "as_of"), F.col("as_of"))
    times = F.when(first < last, F.sequence(first, last, F.expr("INTERVAL 1 HOUR")))
    return (
        e.select("encounter_id", F.explode(times).alias("prediction_ts"), last.alias("_last"))
        .where(F.col("prediction_ts") < F.col("_last"))
        .drop("_last")
    )


def hourly_buckets(vitals: DataFrame) -> DataFrame:
    """Per stay and hour [b, b + 1 h): mean, median, min and max of each vital, and the count."""
    stats = [
        agg(v).alias(f"{v}_{name}")
        for v in VITALS
        for name, agg in (("mean", F.mean), ("median", F.median), ("min", F.min), ("max", F.max))
    ]
    return vitals.groupBy("encounter_id", F.date_trunc("hour", "ts").alias("bucket")).agg(
        *stats, F.count("*").alias("n_readings")
    )


def window_features(grid: DataFrame, buckets: DataFrame) -> DataFrame:
    """Rolling 1/3/6 h features: the row for `t` uses only the six buckets before `t`.

    `lag` is how many hours before `t` a bucket starts (1 = the last hour). Means over 3 and 6
    hours weight each hour by its number of readings, like `features._window_features`.
    """
    g, b = grid.alias("g"), buckets.alias("b")
    near = (
        (F.col("b.encounter_id") == F.col("g.encounter_id"))
        & (F.col("b.bucket") < F.col("g.prediction_ts"))
        & (F.col("b.bucket") >= _plus_hours(F.col("g.prediction_ts"), -6))
    )
    joined = g.join(b, near, "left").select(
        "g.encounter_id",
        "g.prediction_ts",
        F.round(_hours(F.col("g.prediction_ts"), F.col("b.bucket"))).cast("int").alias("lag"),
        *[F.col(f"b.{c}") for c in buckets.columns if c not in ("encounter_id", "bucket")],
    )

    def at(lag: int, column: str) -> Column:
        return F.max(F.when(F.col("lag") == lag, F.col(column)))

    def total(hours: int, column: Column) -> Column:
        return F.sum(F.when(F.col("lag") <= hours, column))

    n_3h, n_6h = total(3, F.col("n_readings")), total(6, F.col("n_readings"))
    aggs = []
    for v in VITALS:
        weighted = F.coalesce(F.col(f"{v}_mean") * F.col("n_readings"), F.lit(0.0))
        aggs += [
            at(1, f"{v}_mean").alias(f"{v}_mean_1h"),
            at(1, f"{v}_median").alias(f"{v}_median_1h"),
            at(1, f"{v}_min").alias(f"{v}_min_1h"),
            at(1, f"{v}_max").alias(f"{v}_max_1h"),
            (total(3, weighted) / n_3h).alias(f"{v}_mean_3h"),
            (total(6, weighted) / n_6h).alias(f"{v}_mean_6h"),
            (at(1, f"{v}_mean") - at(3, f"{v}_mean")).alias(f"{v}_trend_3h"),
        ]
    aggs += [
        F.coalesce(at(1, "n_readings"), F.lit(0)).cast("double").alias("readings_1h"),
        F.coalesce(n_6h, F.lit(0)).alias("_n_6h"),
    ]
    out = joined.groupBy("encounter_id", "prediction_ts").agg(*aggs)
    return out.select(
        "*",
        F.greatest(F.lit(READINGS_PER_HOUR) - F.col("readings_1h"), F.lit(0.0)).alias("missing_1h"),
        F.greatest(F.lit(6 * READINGS_PER_HOUR) - F.col("_n_6h"), F.lit(0))
        .cast("double")
        .alias("missing_6h"),
    ).drop("_n_6h")


def unit_at(grid: DataFrame, units: DataFrame) -> DataFrame:
    """The unit at each prediction time: the latest version recorded before `t`, or the
    earliest one when none was (a stay loaded in one go has only its final version)."""
    joined = grid.join(units, "encounter_id")
    before = F.when(F.col("changed_ts") < F.col("prediction_ts"), F.col("changed_ts"))
    return joined.groupBy("encounter_id", "prediction_ts").agg(
        F.coalesce(F.max_by("unit_id", before), F.min_by("unit_id", "changed_ts")).alias("unit_id")
    )


def latest_observation(grid: DataFrame, observations: DataFrame) -> DataFrame:
    """The last nurse observation strictly before `t`; none yet = alert, on room air."""
    g, o = grid.alias("g"), observations.alias("o")
    before = (F.col("o.encounter_id") == F.col("g.encounter_id")) & (
        F.col("o.obs_ts") < F.col("g.prediction_ts")
    )
    last = (
        g.join(o, before, "left")
        .groupBy("g.encounter_id", "g.prediction_ts")
        .agg(
            F.max_by(
                F.struct("o.obs_ts", "o.acvpu", "o.on_oxygen", "o.o2_flow_lpm"), "o.obs_ts"
            ).alias("o")
        )
    )
    return last.select(
        "encounter_id",
        "prediction_ts",
        F.coalesce("o.acvpu", F.lit("A")).alias("acvpu"),
        F.coalesce(F.col("o.on_oxygen").cast("boolean"), F.lit(False)).alias("on_oxygen"),
        F.coalesce(F.col("o.o2_flow_lpm").cast("double"), F.lit(0.0)).alias("o2_flow_lpm"),
        _hours(F.col("prediction_ts"), F.col("o.obs_ts")).alias("hours_since_obs"),
        (F.coalesce("o.acvpu", F.lit("A")) != "A").alias("new_confusion"),
    )


def _bands(value: Column, edges: list[float], points: list[int]) -> Column:
    """Points by band: `edges` are inclusive upper bounds of every band but the last."""
    out = F.lit(float(points[-1]))
    for edge, p in reversed(list(zip(edges, points, strict=False))):
        out = F.when(value <= edge, float(p)).otherwise(out)
    return F.when(value.isNotNull(), out)


def news2_points(df: DataFrame, suffix: str = "_median_1h") -> DataFrame:
    """NEWS2 components and total, as `heavden.ml.news2.news2` (half-even rounding, like numpy)."""

    def vital(name: str, digits: int = 0) -> Column:
        return F.bround(F.col(f"{name}{suffix}"), digits)

    spo2, oxygen = vital("spo2"), F.col("on_oxygen")
    scale1 = _bands(spo2, [91, 93, 95], [3, 2, 1, 0])
    scale2 = _bands(spo2, [83, 85, 87], [3, 2, 1, 0])
    scale2 = F.when(oxygen & (spo2 >= 93), _bands(spo2, [92, 94, 96], [0, 1, 2, 3])).otherwise(
        scale2
    )
    points = {
        "news2_resp_rate": _bands(vital("resp_rate"), [8, 11, 20, 24], [3, 1, 0, 2, 3]),
        "news2_spo2": F.when(F.col("copd"), scale2).otherwise(scale1),
        "news2_oxygen": F.when(oxygen, 2.0).otherwise(0.0),
        "news2_sbp": _bands(vital("sbp"), [90, 100, 110, 219], [3, 2, 1, 0, 3]),
        "news2_heart_rate": _bands(vital("heart_rate"), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3]),
        "news2_consciousness": F.when(F.col("acvpu") == "A", 0.0).otherwise(3.0),
        "news2_temp": _bands(vital("temp_c", 1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2]),
    }
    total = reduce(lambda a, b: a + b, points.values())  # NULL if any component is NULL
    return df.select("*", *[c.alias(n) for n, c in points.items()], total.alias("news2_total"))


def patient_hour_features(
    vitals: DataFrame,
    encounters: DataFrame,
    units: DataFrame,
    patients: DataFrame,
    flags: DataFrame,
    observations: DataFrame,
) -> DataFrame:
    """`gold.patient_hour_features`: one row per stay and hour, the columns of
    `features.build_patient_hours` without the label.

    `vitals` are Silver readings linked to stays; `encounters` the current stays; `units` the
    unit history (`unit_history`); `patients` current patients (birth_date, sex); `flags` from
    `profile_flags`; `observations` Silver nurse observations.
    """
    window = data_window(vitals)
    grid = prediction_grid(encounters, window)
    feats = window_features(grid, hourly_buckets(vitals))
    stays = encounters.join(patients.select("patient_id", "birth_date", "sex"), "patient_id")
    stay_cols = stays.select(
        "encounter_id",
        "patient_id",
        "site_id",
        "admit_ts",
        F.floor(F.datediff(F.to_date("admit_ts"), F.col("birth_date")) / 365.25).alias("age"),
        "sex",
    )
    unit = unit_at(grid, units)
    df = (
        feats.join(stay_cols, "encounter_id")
        .join(unit, ["encounter_id", "prediction_ts"])
        .join(flags, "encounter_id")
        .join(latest_observation(grid, observations), ["encounter_id", "prediction_ts"])
        .withColumn("unit_type", F.lower(F.element_at(F.split("unit_id", "-"), -1)))
        .withColumn("hours_since_admission", _hours(F.col("prediction_ts"), F.col("admit_ts")))
    )
    df = news2_points(df)
    first = ["encounter_id", "patient_id", "prediction_ts", "site_id", "unit_id", "unit_type"]
    window_cols = [c for c in feats.columns if c not in ("encounter_id", "prediction_ts")]
    profile = ["age", "sex", *CONDITION_FLAGS, "on_beta_blocker", "n_conditions"]
    nurse = ["acvpu", "on_oxygen", "o2_flow_lpm", "hours_since_obs", "new_confusion"]
    scores = [c for c in df.columns if c.startswith("news2_")]
    return df.select(*first, *window_cols, "hours_since_admission", *profile, *nurse, *scores)


def training_set(features: DataFrame, outcomes: DataFrame, window: DataFrame) -> DataFrame:
    """`gold.training_set`: features plus a `label` that is final as of the data.

    `label` = 1 if a recorded escalation falls in (t, t + 6 h]. It is final when that
    escalation is recorded, or (negative) once `t + 12 h` has passed `as_of`, when any
    escalation in the window would have been recorded. Rows whose label isn't final yet are
    left out; `label_known_at` is when it became final.
    """
    f, o = features.alias("f"), outcomes.alias("o")
    in_window = (
        (F.col("o.encounter_id") == F.col("f.encounter_id"))
        & (F.col("o.event_ts") > F.col("f.prediction_ts"))
        & (F.col("o.event_ts") <= _plus_hours(F.col("f.prediction_ts"), HORIZON_H))
    )
    keys = ["encounter_id", "prediction_ts"]
    recorded = (
        f.join(o, in_window)
        .groupBy("f.encounter_id", "f.prediction_ts")
        .agg(F.min("o.recorded_ts").alias("_recorded"))
    )
    settled = _plus_hours(F.col("prediction_ts"), HORIZON_H + RECORDING_DELAY_H)
    labelled = features.join(recorded, keys, "left").crossJoin(window.select("as_of"))
    out = labelled.select(
        *features.columns,
        F.when(F.col("_recorded").isNotNull(), 1.0)
        .when(settled <= F.col("as_of"), 0.0)
        .alias("label"),
        F.coalesce(F.col("_recorded"), settled).alias("label_known_at"),
    )
    return out.where(F.col("label").isNotNull())


# ---------- Device health ----------


def device_health_daily(
    readings: DataFrame, assignments: DataFrame, window: DataFrame
) -> DataFrame:
    """`gold.device_health_daily`, as `heavden.analytics.gold.device_health_daily`.

    `readings` are Silver device readings (linked or not); `assignments` the current device
    assignments. A gap of 25+ minutes before a message counts as an outage; a reading inside a
    run of 6+ identical heart-rate or blood-pressure values (30 minutes) counts as stuck.
    """
    from pyspark.sql import Window

    by_device = Window.partitionBy("device_id").orderBy("ts")
    rd = readings.select("device_id", "ts", *STUCK_VITALS, "spo2", "battery_pct", "firmware")
    gap_minutes = _hours(F.col("ts"), F.lag("ts").over(by_device)) * 60
    rd = rd.withColumn("outage", F.coalesce(gap_minutes >= OUTAGE_GAP_MINUTES, F.lit(False)))
    stuck = F.lit(False)
    for v in STUCK_VITALS:
        same = F.coalesce(F.col(v) == F.lag(v).over(by_device), F.lit(False))
        rd = rd.withColumn(f"_run_{v}", F.sum(F.when(same, 0).otherwise(1)).over(by_device))
        run = Window.partitionBy("device_id", f"_run_{v}")
        stuck = stuck | (F.count("*").over(run) >= STUCK_RUN)
    rd = rd.withColumn("stuck", stuck).withColumn("date", F.date_trunc("day", "ts"))

    daily = rd.groupBy("device_id", "date").agg(
        F.count("*").cast("double").alias("messages_received"),
        F.sum(F.col("outage").cast("int")).cast("double").alias("battery_outages"),
        (F.sum(F.col("stuck").cast("int")) * STEP_MINUTES).cast("double").alias("stuck_minutes"),
        F.min("battery_pct").alias("min_battery_pct"),
        F.array_join(F.array_sort(F.collect_set("firmware")), ",").alias("firmware"),
        F.mean("spo2").alias("mean_spo2"),
    )
    out = daily.join(expected_messages(assignments, window), ["device_id", "date"], "full")
    received = F.coalesce("messages_received", F.lit(0.0))
    expected = F.coalesce("messages_expected", F.lit(0.0))
    return out.select(
        "device_id",
        F.concat(F.lit("SITE_"), F.split("device_id", "-").getItem(1)).alias("site_id"),
        "date",
        received.alias("messages_received"),
        expected.alias("messages_expected"),
        F.when(expected > 0, F.least(100 * received / expected, F.lit(100.0))).alias("uptime_pct"),
        "battery_outages",
        "stuck_minutes",
        "min_battery_pct",
        "firmware",
        "mean_spo2",
    )


def expected_messages(assignments: DataFrame, window: DataFrame) -> DataFrame:
    """Readings each device should have sent per day while worn (one per 5 minutes)."""
    a = assignments.crossJoin(window).select(
        "device_id",
        F.greatest("start_ts", "data_start").alias("s"),
        F.least(F.coalesce("end_ts", "as_of"), F.col("as_of")).alias("e"),
    )
    a = a.where(F.col("e") > F.col("s"))
    days = a.select(
        "device_id",
        "s",
        "e",
        F.explode(
            F.sequence(
                F.date_trunc("day", "s"),
                F.date_trunc("day", F.col("e") - F.expr("INTERVAL 1 MICROSECOND")),
                F.expr("INTERVAL 1 DAY"),
            )
        ).alias("date"),
    )
    overlap = _hours(
        F.least("e", _plus_hours(F.col("date"), 24)), F.greatest("s", F.col("date"))
    ) * (60 / STEP_MINUTES)
    return days.groupBy("device_id", "date").agg(
        F.bround(F.sum(overlap)).alias("messages_expected")
    )
