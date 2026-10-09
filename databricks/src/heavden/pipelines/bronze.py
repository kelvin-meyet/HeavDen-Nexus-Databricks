"""Bronze: landing files as they arrived, plus where and when (Plan.md §7.2-7.3).

Auto Loader reads `landing/<env>/{vitals,outcomes}/` (the volume `bronze.landing`). Type hints
fix the known fields' types; a field nobody announced (a new firmware payload) is added as a
new column (`addNewColumns`), and a value that doesn't fit its type lands in `_rescued_data`
instead of being lost.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from heavden.pipelines.silver import OUTCOME_FIELDS, READING_FIELDS

AUTO_LOADER_OPTIONS = {
    "cloudFiles.format": "json",
    "cloudFiles.inferColumnTypes": "true",
    "cloudFiles.schemaEvolutionMode": "addNewColumns",
}


def schema_hints(fields: dict[str, str]) -> str:
    """`cloudFiles.schemaHints` text, e.g. "device_id STRING, ts TIMESTAMP"."""
    return ", ".join(f"{name} {kind.upper()}" for name, kind in fields.items())


VITALS_HINTS = schema_hints(READING_FIELDS)
OUTCOMES_HINTS = schema_hints(OUTCOME_FIELDS)


def with_lineage(df: DataFrame) -> DataFrame:
    """Add the source file and the ingestion time to each row."""
    return df.withColumn("_source_file", F.col("_metadata.file_path")).withColumn(
        "_ingest_ts", F.current_timestamp()
    )
