"""The hospital's source database (Azure SQL): build its tables from the simulation and load them.

One Azure SQL database (the free offer allows one per subscription) holds every environment,
each in its own schema: `dev`, `staging`, `prod`. Databricks reads them with a JDBC job that
follows SQL Server Change Tracking (Plan.md §7.2).

    uv run --group azure python -m generator.source_db --env dev            # load (asks first)
    uv run --group azure python -m generator.source_db --env dev --dry-run  # only build + count

Server, database and Key Vault names come from `terraform output`; the SQL password is read
from Key Vault with the Azure CLI, so it never touches a file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from generator import activity as hospital_activity
from generator import hospital

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = Path(__file__).with_name("sql") / "azure_sql_schema.sql"
ENVIRONMENTS = ("dev", "staging", "prod")

# Load order respects foreign keys; drop in reverse.
TABLE_ORDER = (
    "sites",
    "units",
    "patients",
    "conditions",
    "medications",
    "encounters",
    "nurse_observations",
    "device_assignments",
)

# SQL Server accepts at most 2100 parameters per statement and 1000 rows per VALUES list.
MAX_PARAMS = 2100
MAX_ROWS = 1000


# --- Building the tables -----------------------------------------------------------------


def source_tables(
    synthea_tables: dict[str, pd.DataFrame],
    actual: hospital_activity.HospitalActivity,
    nurse_observations: pd.DataFrame,
    as_of: pd.Timestamp,
    reference_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Every source table as it stands at simulated time `as_of`, columns as in the DDL.

    `actual` is the activity with escalations applied (`WardResult.activity`). A patient (with
    their conditions and medications) appears when first admitted; `last_updated` is the
    simulated time of each row's latest change.
    """
    snapshot = hospital_activity.sql_snapshot(actual, as_of)
    encounters, devices = snapshot["encounters"], snapshot["device_assignments"]
    first_seen = encounters.groupby("patient_id")["admit_ts"].min()
    start = actual.start

    sites = hospital.sites_frame()[["site_id", "name", "city", "beds"]].assign(last_updated=start)
    units = hospital.units_frame()[["unit_id", "site_id", "unit_type", "beds"]].assign(
        last_updated=start
    )

    people = actual.patients[actual.patients["patient_id"].isin(first_seen.index)]
    patients = people[["patient_id", "first_name", "last_name", "birth_date", "sex"]].assign(
        last_updated=people["patient_id"].map(first_seen)
    )

    def history(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["PATIENT"].isin(first_seen.index) & (df["START"] <= reference_date)]
        return df.assign(last_updated=df["PATIENT"].map(first_seen))

    conditions = history(synthea_tables["conditions"])
    conditions = pd.DataFrame(
        {
            "patient_id": conditions["PATIENT"],
            "code": conditions["CODE"],
            "description": conditions["DESCRIPTION"],
            "start_date": conditions["START"].dt.date,
            "stop_date": conditions["STOP"].dt.date,
            "last_updated": conditions["last_updated"],
        }
    ).drop_duplicates(["patient_id", "code", "start_date"])

    # Only medications still active at the reference date: the full Synthea history is ~20x larger.
    meds = history(synthea_tables["medications"])
    meds = meds[meds["STOP"].isna() | (meds["STOP"] > reference_date)]
    medications = pd.DataFrame(
        {
            "patient_id": meds["PATIENT"],
            "code": meds["CODE"],
            "description": meds["DESCRIPTION"],
            "start_ts": meds["START"],
            "stop_ts": meds["STOP"],
            "last_updated": meds["last_updated"],
        }
    ).drop_duplicates(["patient_id", "code", "start_ts"])

    obs = nurse_observations[nurse_observations["obs_ts"] <= as_of]
    obs = obs[obs["encounter_id"].isin(encounters["encounter_id"])]
    nurse = obs[["encounter_id", "patient_id", "obs_ts", "acvpu", "on_oxygen", "o2_flow_lpm"]]
    nurse = nurse.assign(last_updated=nurse["obs_ts"])

    devices = devices[
        ["device_id", "encounter_id", "patient_id", "start_ts", "end_ts", "last_updated"]
    ]

    tables = {
        "sites": sites,
        "units": units,
        "patients": patients,
        "conditions": conditions,
        "medications": medications,
        "encounters": encounters,
        "nurse_observations": nurse,
        "device_assignments": devices,
    }
    return {name: tables[name].reset_index(drop=True) for name in TABLE_ORDER}


# --- SQL text ------------------------------------------------------------------------------


def ddl_batches(schema: str) -> list[str]:
    """The schema file for one environment, split into batches at `GO` (a client-side separator).

    Database-level Change Tracking is switched on separately (`enable_change_tracking`), because
    it fails if it is already on, which it will be after the first environment.
    """
    if schema not in ENVIRONMENTS:
        raise ValueError(f"schema must be one of {ENVIRONMENTS}, not {schema!r}")
    text = SCHEMA_SQL.read_text(encoding="utf-8").replace("dbo.", f"{schema}.")
    batches = [b.strip() for b in re.split(r"^\s*GO\s*$", text, flags=re.MULTILINE)]
    batches = [b for b in batches if b and "ALTER DATABASE" not in b]
    return [f"CREATE SCHEMA {schema}", *batches]


def ddl_columns(table: str) -> list[str]:
    """Column names of a table, in DDL order (read from the schema file)."""
    text = SCHEMA_SQL.read_text(encoding="utf-8")
    match = re.search(rf"CREATE TABLE dbo\.{table} \((.*?)\n\);", text, flags=re.DOTALL)
    if match is None:
        raise KeyError(table)
    names = []
    for line in match.group(1).splitlines():
        token = line.strip().split(" ")[0]
        if token and token.isidentifier() and token.upper() != "CONSTRAINT":
            names.append(token)
    return names


def _sql_value(value):
    """pandas/numpy values to what the driver sends: naive UTC datetimes, None for missing."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return (
            value.tz_convert("UTC").tz_localize(None).to_pydatetime()
            if value.tzinfo
            else value.to_pydatetime()
        )
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def insert_batches(schema: str, table: str, df: pd.DataFrame) -> Iterator[tuple[str, list]]:
    """Multi-row INSERT statements with their parameters, within SQL Server's limits."""
    columns = ddl_columns(table)
    rows_per_batch = min(MAX_ROWS, MAX_PARAMS // len(columns))
    values = df[columns].astype(object).to_numpy()
    placeholder = "(" + ", ".join("?" * len(columns)) + ")"
    for i in range(0, len(values), rows_per_batch):
        chunk = values[i : i + rows_per_batch]
        sql = f"INSERT INTO {schema}.{table} ({', '.join(columns)}) VALUES " + ", ".join(
            [placeholder] * len(chunk)
        )
        yield sql, [_sql_value(v) for row in chunk for v in row]


# --- Talking to Azure SQL ------------------------------------------------------------------


def terraform_outputs(directory: Path = ROOT / "infra" / "terraform") -> dict[str, str]:
    out = subprocess.run(
        ["terraform", f"-chdir={directory}", "output", "-json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {name: item["value"] for name, item in json.loads(out).items()}


def _az() -> str:
    """The Azure CLI, found even when it was installed after this terminal started."""
    found = shutil.which("az")
    if found:
        return found
    default = Path(r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbinz.cmd")
    if default.exists():
        return str(default)
    raise FileNotFoundError("Azure CLI (az) not found; install it and run `az login`.")


def _key_vault_secret(vault: str, name: str) -> str:
    return subprocess.run(
        [_az(), "keyvault", "secret", "show", "--vault-name", vault, "--name", name]
        + ["--query", "value", "-o", "tsv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def connect_with_retry(
    open_connection, attempts: int = 6, wait_seconds: float = 20, sleep=time.sleep
):
    """Open a connection, waiting while a paused serverless database wakes up.

    The first login to an auto-paused database fails with error 40613 ("database ... is not
    currently available") and starts the resume, which usually takes under a minute.
    """
    for attempt in range(1, attempts + 1):
        try:
            return open_connection()
        except Exception as error:  # the driver's error classes vary; match on the code
            if "40613" not in str(error) or attempt == attempts:
                raise
            print(f"  database is waking up (attempt {attempt}/{attempts}); retrying...")
            sleep(wait_seconds)
    raise AssertionError("unreachable")


def connection_string(server: str, database: str, user: str, password: str) -> str:
    """mssql-python accepts only its own keywords (no `Connection Timeout`: use `timeout=`)."""
    return (
        f"Server=tcp:{server},1433;Database={database};Uid={user};Pwd={{{password}}};"
        "Encrypt=yes;TrustServerCertificate=no"
    )


def connect(server: str, database: str, key_vault: str):
    """A connection as the SQL admin, with the login read from Key Vault."""
    import mssql_python  # in the `azure` dependency group

    user = _key_vault_secret(key_vault, "sql-admin-user")
    password = _key_vault_secret(key_vault, "sql-admin-password")
    return connect_with_retry(
        lambda: mssql_python.connect(
            connection_string(server, database, user, password), timeout=60
        )
    )


def enable_change_tracking(cursor) -> None:
    cursor.execute(
        "IF NOT EXISTS (SELECT 1 FROM sys.change_tracking_databases WHERE database_id = DB_ID()) "
        "ALTER DATABASE CURRENT SET CHANGE_TRACKING = ON "
        "(CHANGE_RETENTION = 7 DAYS, AUTO_CLEANUP = ON)"
    )


def drop_schema(cursor, schema: str) -> None:
    for table in reversed(TABLE_ORDER):
        cursor.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
    cursor.execute(f"IF SCHEMA_ID('{schema}') IS NOT NULL EXEC('DROP SCHEMA {schema}')")


def load(connection, schema: str, tables: dict[str, pd.DataFrame], replace: bool = False) -> None:
    """Create the environment's schema and insert every table (in one transaction per table)."""
    cursor = connection.cursor()
    enable_change_tracking(cursor)
    connection.commit()
    if replace:
        drop_schema(cursor, schema)
        connection.commit()
    for batch in ddl_batches(schema):
        cursor.execute(batch)
    connection.commit()
    for table in TABLE_ORDER:
        n = 0
        for sql, params in insert_batches(schema, table, tables[table]):
            cursor.execute(sql, params)
            n += len(params) // len(ddl_columns(table))
        connection.commit()
        print(f"  {schema}.{table}: {n:,} rows")


def row_counts(connection, schema: str) -> dict[str, int]:
    cursor = connection.cursor()
    counts = {}
    for table in TABLE_ORDER:
        cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        counts[table] = cursor.fetchone()[0]
    return counts


# --- Command line --------------------------------------------------------------------------

DEFAULT_DAYS = {"dev": 2, "staging": 2, "prod": 14}


def simulate(days: int, seed: int = 42):
    """The same simulation as the demo snapshot (Synthea seed 42, ward seed 42)."""
    from generator import synthea, truth

    run = synthea.SyntheaRun()
    csv_dir = synthea.run_synthea(run, ROOT / "data" / "synthea" / f"p{run.population}_s{run.seed}")
    tables = synthea.load_tables(csv_dir)
    reference = pd.Timestamp(run.reference_date, tz="UTC")
    profiles = synthea.build_profiles(tables, reference)
    inpatients = hospital.admit_inpatients(profiles)
    plan = hospital_activity.schedule_activity(
        profiles, inpatients, hospital.SIM_START, hours=24 * days, seed=seed
    )
    ward = truth.simulate_ward(inpatients, hospital.SIM_START, 24 * days, seed=seed, activity=plan)
    return tables, reference, ward


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the hospital source tables into Azure SQL.")
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--days", type=int, help="simulated days (default: dev/staging 2, prod 14)")
    parser.add_argument(
        "--replace", action="store_true", help="drop the environment's tables first"
    )
    parser.add_argument("--dry-run", action="store_true", help="build and count, don't connect")
    parser.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = parser.parse_args()
    days = args.days or DEFAULT_DAYS[args.env]

    synthea_tables, reference, ward = simulate(days)
    tables = source_tables(
        synthea_tables, ward.activity, ward.nurse_observations, ward.activity.end, reference
    )
    print(f"{days} simulated days, as of {ward.activity.end}:")
    for name, df in tables.items():
        print(f"  {name}: {len(df):,} rows")
    if args.dry_run:
        return

    out = terraform_outputs()
    target = f"{out['sql_server_fqdn']} / {out['sql_database']} / schema {args.env}"
    if not args.yes and input(f"Load into {target}? [y/N] ").strip().lower() != "y":
        print("Nothing loaded.")
        return
    connection = connect(out["sql_server_fqdn"], out["sql_database"], out["key_vault"])
    try:
        load(connection, args.env, tables, replace=args.replace)
        print("In Azure SQL:", row_counts(connection, args.env))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
