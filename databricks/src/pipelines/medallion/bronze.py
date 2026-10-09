"""Bronze: vitals and outcome files from the landing volume, read with Auto Loader.

`heavden.landing` (pipeline configuration) is the volume `<catalog>.bronze.landing`, which maps
to `landing/<env>/` in ADLS. Each run picks up only the files it hasn't seen.
"""

from pyspark import pipelines as dp

from heavden.pipelines import bronze

LANDING = spark.conf.get("heavden.landing")


def _auto_loader(folder: str, hints: str):
    reader = (
        spark.readStream.format("cloudFiles")
        .options(**bronze.AUTO_LOADER_OPTIONS)
        .option("cloudFiles.schemaHints", hints)
    )
    return bronze.with_lineage(reader.load(f"{LANDING}/{folder}/"))


@dp.table(name="vitals_raw", comment="Device readings as landed (synthetic), plus lineage.")
def vitals_raw():
    return _auto_loader("vitals", bronze.VITALS_HINTS)


@dp.table(name="outcomes_raw", comment="Escalations as recorded (synthetic), plus lineage.")
def outcomes_raw():
    return _auto_loader("outcomes", bronze.OUTCOMES_HINTS)
