"""Gold: model features, training labels, device health and the condition-flag reference.

Materialized views over Silver: each refresh recomputes them from the current Silver tables, so
late data and corrections flow through without special handling. The logic is in
`heavden.pipelines.gold`, tested against the pandas references.
"""

from pyspark import pipelines as dp

from heavden.pipelines import gold, silver


def _flags():
    return gold.profile_flags(
        silver.current(spark.read.table("silver.encounters")),
        spark.read.table("silver.conditions"),
        spark.read.table("silver.medications"),
    )


@dp.materialized_view(
    name="gold.patient_hour_features",
    comment="Per stay and hour on the ward: vitals over 1/3/6 h, profile, NEWS2 (synthetic).",
    cluster_by=["prediction_ts", "encounter_id"],
)
@dp.expect("after_admission", "hours_since_admission > 0")
def patient_hour_features():
    encounters = spark.read.table("silver.encounters")
    return gold.patient_hour_features(
        spark.read.table("silver.vitals"),
        silver.current(encounters),
        gold.unit_history(encounters),
        silver.current(spark.read.table("silver.patients")),
        _flags(),
        spark.read.table("silver.nurse_observations"),
    )


@dp.materialized_view(
    name="gold.training_set",
    comment="Patient-hours with a final label: escalation within 6 h, from recorded outcomes.",
    cluster_by=["prediction_ts", "encounter_id"],
)
@dp.expect("label_is_0_or_1", "label IN (0, 1)")
def training_set():
    return gold.training_set(
        spark.read.table("gold.patient_hour_features"),
        spark.read.table("silver.outcomes"),
        gold.data_window(spark.read.table("silver.vitals")),
    )


@dp.materialized_view(
    name="gold.device_health_daily",
    comment="Per device and day: messages vs expected, outages, stuck sensors (synthetic).",
)
@dp.expect("uptime_in_range", "uptime_pct IS NULL OR uptime_pct BETWEEN 0 AND 100")
def device_health_daily():
    return gold.device_health_daily(
        spark.read.table("silver.device_readings"),
        silver.current(spark.read.table("silver.device_assignments")),
        gold.data_window(spark.read.table("silver.vitals")),
    )


@dp.materialized_view(
    name="gold.condition_flag_reference",
    comment="Which diagnoses and drugs set each model flag, and for how many patients.",
)
def condition_flag_reference():
    return gold.condition_flag_reference(
        spark.read.table("silver.conditions"), spark.read.table("silver.medications")
    )
