"""Build the demo-mode snapshot from the local generator and the registered champion.

    uv run python backend/scripts/build_demo_snapshot.py [--out data/demo_snapshot]

Needs: the Synthea output (notebook 01 / `generator.synthea`), the champion registered in the
local MLflow store (notebook 06), and the RAG index (notebook 08, `data/rag_local/`). Takes a
few minutes: it simulates 14 days of the hospital, builds features, scores every patient-hour
and derives the gold tables.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from generator import activity as hospital_activity  # noqa: E402
from generator import hospital, synthea, truth  # noqa: E402
from heavden.ml import features, scoring, tracking, train  # noqa: E402
from heavden_api.snapshot import build_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DAYS, SEED = 14, 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "demo_snapshot")
    parser.add_argument("--rag", type=Path, default=ROOT / "data" / "rag_local")
    args = parser.parse_args()
    t0 = time.time()

    run = synthea.SyntheaRun()
    csv_dir = synthea.run_synthea(run, ROOT / "data" / "synthea" / f"p{run.population}_s{run.seed}")
    profiles = synthea.build_profiles(
        synthea.load_tables(csv_dir), pd.Timestamp(run.reference_date, tz="UTC")
    )
    inpatients = hospital.admit_inpatients(profiles)
    plan = hospital_activity.schedule_activity(
        profiles, inpatients, hospital.SIM_START, hours=24 * DAYS, seed=SEED
    )
    ward = truth.simulate_ward(inpatients, hospital.SIM_START, 24 * DAYS, seed=SEED, activity=plan)
    print(f"simulated {DAYS} days: {len(ward.readings):,} readings ({time.time() - t0:.0f}s)")

    table = features.build_patient_hours(
        ward.readings,
        ward.activity,
        ward.truth.outcomes,
        nurse_observations=ward.nurse_observations,
    )
    print(f"features: {len(table):,} patient-hours ({time.time() - t0:.0f}s)")

    tracking.use_local_store(ROOT / "data" / "mlflow")
    name = tracking.LOCAL_MODEL_NAME
    version = mlflow.MlflowClient().get_model_version_by_alias(name, "champion").version
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    tr, va, _ = train.time_split(table)
    bands = scoring.RiskBands.fit(model.predict_proba(va)[:, 1])

    rag = args.rag if args.rag.exists() else None
    if rag is None:
        print(f"warning: no RAG index at {args.rag}; run notebook 08 to include document search")
    manifest = build_snapshot(
        args.out,
        table,
        ward.activity,
        ward.truth.outcomes,
        ward.readings,
        model,
        tr,
        bands,
        model_name=name,
        model_version=version,
        rag_dir=rag,
    )
    print(f"snapshot written to {args.out} ({time.time() - t0:.0f}s)")
    print(f"as_of {manifest.as_of} | model {name} v{version} | bands {manifest.bands}")
    for table_name, n in manifest.rows.items():
        print(f"  gold.{table_name}: {n:,} rows")


if __name__ == "__main__":
    main()
