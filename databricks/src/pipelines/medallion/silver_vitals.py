"""Silver vitals: checked, de-duplicated, linked to patients, and reconciled with Bronze."""

from pyspark import pipelines as dp

from heavden.pipelines import silver


@dp.temporary_view(name="vitals_checked")
@dp.expect_all_or_drop(silver.READING_RULES)
def vitals_checked():
    return silver.typed_readings(spark.readStream.table("vitals_raw"))


@dp.table(
    name="silver.device_readings",
    comment="Valid device readings, one per device and time (synthetic).",
    cluster_by=["device_id", "ts"],
)
def device_readings():
    return silver.device_readings(spark.readStream.table("vitals_checked"))


@dp.table(
    name="silver.vitals_quarantine",
    comment="Readings that failed a check, with the names of the checks they failed.",
)
def vitals_quarantine():
    typed = silver.typed_readings(spark.readStream.table("vitals_raw"))
    return silver.quarantine(typed, silver.READING_RULES)


# A materialized view, not a stream: a reading that lands before its device assignment is
# linked on the next refresh instead of being dropped for good.
@dp.materialized_view(
    name="silver.vitals",
    comment="Device readings linked to the stay and patient wearing the device (synthetic).",
    cluster_by=["encounter_id", "ts"],
)
def vitals():
    assignments = silver.current(spark.read.table("silver.device_assignments"))
    return silver.link_readings(spark.read.table("silver.device_readings"), assignments)


@dp.materialized_view(
    name="silver.vitals_reconciliation",
    comment="Where every Bronze reading went (missing_from_silver > 0: late readings dropped).",
)
@dp.expect("no_missing_readings", "missing_from_silver = 0")
def vitals_reconciliation():
    return silver.reconcile_readings(
        silver.typed_readings(spark.read.table("vitals_raw")),
        spark.read.table("silver.device_readings"),
        spark.read.table("silver.vitals"),
    )
