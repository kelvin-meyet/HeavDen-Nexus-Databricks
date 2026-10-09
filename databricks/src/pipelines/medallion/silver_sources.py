"""Silver copies of the Azure SQL tables, built with AUTO CDC from what the ingestion job landed.

Bronze `sql_<table>` holds every snapshot and change; AUTO CDC orders them by `last_updated`
(then version, then ingestion time) and keeps the latest row per key (type 1) or its history
with `__START_AT` / `__END_AT` (type 2). Deletes break the source contract and are dropped here
(the ingestion job already failed loudly on them).
"""

from pyspark import pipelines as dp

from heavden.pipelines import silver


def _define(table: str, keys: tuple[str, ...]) -> None:
    changes = f"sql_{table}_changes"

    @dp.temporary_view(name=changes)
    @dp.expect_all_or_drop(silver.SQL_CHANGE_RULES)
    def _changes():
        return spark.readStream.table(f"bronze.sql_{table}")

    scd_type = silver.SCD_TYPE[table]
    dp.create_streaming_table(
        name=f"silver.{table}",
        comment=f"Azure SQL {table} (synthetic), SCD type {scd_type}.",
    )
    dp.create_auto_cdc_flow(
        target=f"silver.{table}",
        source=changes,
        keys=list(keys),
        sequence_by=silver.cdc_sequence(),
        stored_as_scd_type=scd_type,
        track_history_except_column_list=silver.CDC_BOOKKEEPING if scd_type == 2 else None,
    )


for _table, _keys in silver.SQL_KEYS.items():
    _define(_table, _keys)
