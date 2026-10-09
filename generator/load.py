"""Load one simulated period everywhere Databricks reads from: Azure SQL and the landing folder.

One simulation feeds both, so the device ids and times in the vitals files match the encounters
and device assignments in SQL. Run it right before the ingestion job: one database wake-up
then covers the load and the ingestion (Plan.md §14).

    uv run --group azure python -m generator.load --env dev --dry-run  # simulate, write locally
    uv run --group azure python -m generator.load --env dev --replace  # reload both (asks)

`--replace` drops the environment's SQL tables and deletes `landing/<env>/` before loading. The
next ingestion run then takes a full copy of each table, because the recreated tables' change
history starts after the old watermark.
"""

from __future__ import annotations

import argparse

from generator import landing, source_db

LOCAL_ROOT = source_db.ROOT / "data" / "landing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", choices=source_db.ENVIRONMENTS, required=True)
    parser.add_argument("--days", type=int, help="simulated days (default: dev/staging 2, prod 14)")
    parser.add_argument("--replace", action="store_true", help="replace what's there")
    parser.add_argument("--dry-run", action="store_true", help="simulate and write locally only")
    parser.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = parser.parse_args()
    days = args.days or source_db.DEFAULT_DAYS[args.env]

    synthea_tables, reference, ward = source_db.simulate(days)
    as_of = ward.activity.end
    tables = source_db.source_tables(
        synthea_tables, ward.activity, ward.nurse_observations, as_of, reference
    )
    issues = source_db.problems(tables)
    if issues:
        raise SystemExit("These values don't fit the schema:\n  " + "\n  ".join(issues))
    files = landing.landing_files(args.env, ward.readings, ward.truth.outcomes, as_of)
    landing.write_local(files, LOCAL_ROOT)

    vitals = [p for p in files if "/vitals/" in p]
    outcomes = [p for p in files if "/outcomes/" in p]
    recorded = sum(files[p].count("\n") for p in outcomes)
    print(f"{days} simulated days, as of {as_of}:")
    for name, df in tables.items():
        print(f"  sql {name}: {len(df):,} rows")
    print(f"  landing vitals: {len(ward.readings):,} readings in {len(vitals)} files")
    print(
        f"  landing outcomes: {recorded} of {len(ward.truth.outcomes)} escalations recorded "
        f"by {as_of} (label delay {landing.LABEL_DELAY}) in {len(outcomes)} files"
    )
    print(f"  written locally to {LOCAL_ROOT / args.env}")
    if args.dry_run:
        return

    out = source_db.terraform_outputs()
    action = "REPLACE" if args.replace else "load into"
    targets = (
        f"  Azure SQL {out['sql_server_fqdn']} / {out['sql_database']} / schema {args.env}\n"
        f"  {out['landing_url']}{args.env}/"
    )
    question = f"{action}:\n{targets}\n[y/N] "
    if not args.yes and input(question).strip().lower() != "y":
        print("Nothing loaded.")
        return

    connection = source_db.connect(out["sql_server_fqdn"], out["sql_database"], out["key_vault"])
    try:
        source_db.load(connection, args.env, tables, replace=args.replace)
        print("In Azure SQL:", source_db.row_counts(connection, args.env))
    finally:
        connection.close()
    landing.upload(source_db._az(), LOCAL_ROOT, args.env, out["storage_account"], args.replace)
    print(f"Uploaded {len(files)} files to {out['landing_url']}{args.env}/")
    print("Next: databricks bundle run sql_ingest -t <env> within the hour (database is awake).")


if __name__ == "__main__":
    main()
