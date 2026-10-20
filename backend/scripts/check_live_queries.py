"""Run every page query of the API on the SQL warehouse, as live mode will (read-only).

The SQL in `heavden_api/queries.py` is written once for DuckDB (demo) and Databricks (live);
this proves the Databricks half on real data before the website uses it, and shows whether the
latest scored hour has a full ward (live mode calls it "now", like the Genie space). It wakes the
warehouse once (~$0.25). Locally, as a CLI profile:

    uv run --group backend python backend/scripts/check_live_queries.py --profile azure-heavden
"""

from __future__ import annotations

import argparse
import os
import time

from heavden.ml import evaluate
from heavden_api import queries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="azure-heavden")
    parser.add_argument("--catalog", default="heavden_prod")
    parser.add_argument("--warehouse", default="0a14d0d391ce8af6")
    args = parser.parse_args(argv)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    from heavden_api.chat.databricks_tools import Warehouse
    from heavden_api.config import databricks_api
    from heavden_api.sources import LiveSource

    source = LiveSource(Warehouse(databricks_api(), args.warehouse, catalog=args.catalog))
    as_of, first = source.now()
    print(f"now = {as_of}, first scored hour = {first}")

    census = source.query(
        "SELECT prediction_ts, count(*) AS stays FROM gold.risk_scores "
        "WHERE prediction_ts > :as_of - INTERVAL 48 HOURS GROUP BY prediction_ts "
        "ORDER BY prediction_ts DESC",
        {"as_of": as_of},
    )
    median = census["stays"].median()
    print(
        f"census, last 48 h: median {median:.0f}; latest hours {census['stays'].head(4).tolist()}"
    )
    if census["stays"].iloc[0] < 0.5 * median:
        print("WARNING: the latest hour is not a full ward; pages and Genie would disagree")

    stay = source.query(
        "SELECT encounter_id FROM gold.risk_scores WHERE prediction_ts = :as_of "
        "ORDER BY risk DESC LIMIT 1",
        {"as_of": as_of},
    )["encounter_id"].iloc[0]
    budget = evaluate.AlertBudget()
    base = {"as_of": as_of, "site_id": None}
    checks = {
        "SUMMARY": base
        | {"patients_per_nurse": budget.patients_per_nurse, "shift_hours": budget.shift_hours},
        "SUMMARY (SITE_B)": base
        | {
            "site_id": "SITE_B",
            "patients_per_nurse": budget.patients_per_nurse,
            "shift_hours": budget.shift_hours,
        },
        "CENSUS_HOURLY": base | {"hours": 72},
        "UNITS_NOW": base,
        "ALERTS_DAILY": base | {"days": 7},
        "DEVICES_DAILY": base | {"days": 7},
        "BOARD": base | {"unit_id": None, "band": None, "limit": 50},
        "BOARD (High)": base | {"unit_id": None, "band": "High", "limit": 400},
        "PATIENT": {"as_of": as_of, "encounter_id": stay},
        "PATIENT_SERIES": {"as_of": as_of, "encounter_id": stay, "hours": 24},
        "PATIENT_ALERTS": {"as_of": as_of, "encounter_id": stay},
        "FEATURE_ROW": {"as_of": as_of, "encounter_id": stay},
    }
    failed = 0
    for label, params in checks.items():
        sql = getattr(queries, label.split(" ")[0])
        started = time.monotonic()
        try:
            frame = source.query(sql, params)
        except Exception as exc:  # report every query, then fail at the end
            failed += 1
            print(f"FAIL {label}: {exc}")
            continue
        first_row = frame.iloc[0].to_dict() if len(frame) else {}
        shown = {k: first_row[k] for k in list(first_row)[:6]}
        print(f"ok   {label}: {len(frame)} rows, {time.monotonic() - started:.1f} s; {shown}")
    if failed:
        raise SystemExit(f"{failed} of {len(checks)} queries failed")
    print(f"all {len(checks)} queries ran on {args.catalog}")


if __name__ == "__main__":
    main()
