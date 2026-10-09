"""Silver outcomes: one row per recorded escalation, with the stay's site."""

from pyspark import pipelines as dp

from heavden.pipelines import silver


@dp.temporary_view(name="outcomes_checked")
@dp.expect_all_or_drop(silver.OUTCOME_RULES)
def outcomes_checked():
    return silver.typed_outcomes(spark.read.table("outcomes_raw"))


@dp.materialized_view(
    name="silver.outcomes",
    comment="Escalations (rapid response, ICU transfer) as recorded, de-duplicated (synthetic).",
)
@dp.expect("encounter_known", "site_id IS NOT NULL")
def outcomes():
    encounters = silver.current(spark.read.table("silver.encounters"))
    return silver.outcomes(spark.read.table("outcomes_checked"), encounters)
