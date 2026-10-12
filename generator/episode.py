"""Drift episodes: carry an environment's hospital on from where it stopped (Plan.md §6).

The baseline load (`generator.load`) simulated each environment's first days. An episode
continues from that moment: patients in bed keep their stays, beds and devices, new admissions
get new encounter ids, and the episode's drift scenarios switch on part-way through. Only what
the episode adds is loaded: new and updated rows in Azure SQL (Change Tracking hands them to the
next ingestion run) and new landing files (Auto Loader picks them up).

The simulator draws its random numbers per period, so a longer run is not the shorter run plus
more hours. Each environment's history is therefore replayed period by period from
`episodes.yaml` (seconds per period), which must only ever be appended to.

The seam: stays that cross into an episode keep their personal normal values (resting heart
rate, blood pressure, ...), but their short-term wander starts afresh, and a deterioration under
way at the episode start stops there.

    uv run --group azure python -m generator.episode --env prod --episode 1 --dry-run
    uv run --group azure python -m generator.episode --env prod --episode 1   # load (asks)
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from generator import drift, landing, source_db, vitals
from generator.activity import HospitalActivity

EPISODES_PATH = Path(__file__).with_name("episodes.yaml")
LOCAL_ROOT = source_db.ROOT / "data" / "landing_episode"
CONTINUING = [
    "encounter_id",
    "patient_id",
    "site_id",
    "bed_id",
    "device_id",
    "admit_ts",
    "planned_discharge_ts",
    "admission_reason",
]


@dataclass(frozen=True)
class Episode:
    name: str
    hours: float
    scenarios: list[str]
    seed: int
    settings: dict[str, dict] = field(default_factory=dict)  # overrides of drift_config.yaml


@dataclass
class Period:
    """The hospital up to `end`: every record so far, and the readings of the latest period."""

    name: str
    start: pd.Timestamp  # of the latest period
    end: pd.Timestamp
    actual: HospitalActivity  # whole history, escalations applied
    nurse_observations: pd.DataFrame  # whole history
    outcomes: pd.DataFrame  # every escalation so far, recorded or not
    readings: pd.DataFrame  # the latest period only
    windows: pd.DataFrame  # scenario start times in the latest period
    baselines: pd.DataFrame  # every stay's personal normal values, by encounter_id


def personal_baselines(stays: pd.DataFrame) -> pd.DataFrame:
    return stays.set_index("encounter_id")[[*vitals.VITALS, "hr_wander_scale"]]


def load_episodes(env: str, path: Path = EPISODES_PATH) -> list[Episode]:
    return [Episode(**e) for e in yaml.safe_load(path.read_text(encoding="utf-8"))[env] or []]


def baseline(profiles: pd.DataFrame, inpatients: pd.DataFrame, days: int) -> Period:
    """The environment's first load, exactly as `generator.load` built it."""
    ward = source_db.baseline_ward(profiles, inpatients, days)
    return Period(
        "baseline",
        ward.activity.start,
        ward.activity.end,
        ward.activity,
        ward.nurse_observations,
        ward.truth.outcomes,
        ward.readings,
        pd.DataFrame(columns=["scenario", "type", "starts_at"]),
        personal_baselines(ward.stays),
    )


def merge_activity(before: HospitalActivity, after: HospitalActivity) -> HospitalActivity:
    """One history from two periods. Stays that cross over take their later state, but keep
    where they were admitted (the later period only knows their bed at its start)."""
    first = before.encounters.set_index("encounter_id")
    enc = after.encounters.copy()
    carried = enc["encounter_id"].isin(first.index)
    for column in ("admit_unit_id", "admit_bed_id"):
        enc.loc[carried, column] = enc.loc[carried, "encounter_id"].map(first[column])
    replaced = before.encounters["encounter_id"].isin(enc["encounter_id"])
    devices = before.device_assignments
    devices = devices[~devices["encounter_id"].isin(enc["encounter_id"])]
    return HospitalActivity(
        encounters=pd.concat([before.encounters[~replaced], enc], ignore_index=True),
        device_assignments=pd.concat([devices, after.device_assignments], ignore_index=True),
        transfers=pd.concat([before.transfers, after.transfers], ignore_index=True),
        patients=pd.concat([before.patients, after.patients]).drop_duplicates("patient_id"),
        start=before.start,
        end=after.end,
    )


def scenario_config(episode: Episode) -> dict[str, dict]:
    config = copy.deepcopy(drift.load_config())
    for scenario, overrides in episode.settings.items():
        config[scenario].update(overrides)
    return config


def continue_hospital(profiles: pd.DataFrame, previous: Period, episode: Episode) -> Period:
    """Run `episode` from the end of `previous`, with the patients who are in bed at that time."""
    encounters = previous.actual.encounters
    in_bed = encounters[encounters["discharge_ts"].isna()][CONTINUING]
    next_no = int(encounters["encounter_id"].str[2:].astype(int).max()) + 1
    result = drift.simulate_episode(
        profiles,
        in_bed,
        previous.end,
        episode.hours,
        episode.scenarios,
        episode.seed,
        scenario_config(episode),
        first_encounter_no=next_no,
        carried_baselines=previous.baselines,
    )
    ward = result.ward
    return Period(
        episode.name,
        previous.end,
        ward.activity.end,
        merge_activity(previous.actual, ward.activity),
        pd.concat([previous.nurse_observations, ward.nurse_observations], ignore_index=True),
        pd.concat([previous.outcomes, ward.truth.outcomes], ignore_index=True),
        ward.readings,
        result.windows,
        pd.concat([previous.baselines, personal_baselines(ward.stays)]).groupby(level=0).last(),
    )


def history(env: str, through: int, episodes: list[Episode] | None = None):
    """Synthea tables, reference date, and the periods: baseline, then episodes 1..`through`."""
    episodes = load_episodes(env) if episodes is None else episodes
    if not 1 <= through <= len(episodes):
        raise SystemExit(f"{env} has {len(episodes)} episodes in {EPISODES_PATH.name}")
    tables, reference, profiles, inpatients = source_db.population()
    periods = [baseline(profiles, inpatients, source_db.DEFAULT_DAYS[env])]
    for episode in episodes[:through]:
        periods.append(continue_hospital(profiles, periods[-1], episode))
    return tables, reference, periods


def source_tables(synthea_tables, reference, period: Period) -> dict[str, pd.DataFrame]:
    return source_db.source_tables(
        synthea_tables, period.actual, period.nurse_observations, period.end, reference
    )


def new_files(env: str, previous: Period, period: Period) -> dict[str, str]:
    """The period's readings, and every outcome recorded since the previous period ended
    (including escalations from before it that were still waiting to be recorded)."""
    recorded = period.outcomes["event_ts"] + landing.LABEL_DELAY
    pending = period.outcomes[recorded > previous.end]
    return landing.landing_files(env, period.readings, pending, period.end, tag=period.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", choices=source_db.ENVIRONMENTS, required=True)
    parser.add_argument("--episode", type=int, required=True, help="1 = the first after baseline")
    parser.add_argument("--dry-run", action="store_true", help="simulate and write locally only")
    parser.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = parser.parse_args()

    tables, reference, periods = history(args.env, args.episode)
    previous, period = periods[-2], periods[-1]
    before = source_tables(tables, reference, previous)
    after = source_tables(tables, reference, period)
    issues = source_db.problems(after)
    if issues:
        raise SystemExit("These values don't fit the schema:\n  " + "\n  ".join(issues))
    changes = source_db.table_changes(before, after)
    files = new_files(args.env, previous, period)
    landing.write_local(files, LOCAL_ROOT)

    print(f"{args.env} episode {args.episode} '{period.name}': {period.start} -> {period.end}")
    for row in period.windows.itertuples():
        print(f"  {row.scenario} ({row.type} drift) from {row.starts_at}")
    if period.windows.empty:
        print("  no drift scenario (a normal stretch)")
    in_bed = previous.actual.encounters["discharge_ts"].isna().sum()
    print(f"  continues {in_bed} stays in bed at {period.start}")
    for name, (inserts, updates) in changes.items():
        if len(inserts) or len(updates):
            print(f"  sql {name}: {len(inserts):,} new, {len(updates):,} updated")
    readings = sum(1 for p in files if "/vitals/" in p)
    outcomes = [p for p in files if "/outcomes/" in p]
    print(f"  landing vitals: {len(period.readings):,} readings in {readings} files")
    print(
        f"  landing outcomes: {sum(files[p].count(chr(10)) for p in outcomes)} recorded "
        f"in {len(outcomes)} files"
    )
    print(f"  written locally to {LOCAL_ROOT / args.env}")
    if args.dry_run:
        return

    out = source_db.terraform_outputs()
    question = (
        f"Load episode {args.episode} '{period.name}' into:\n"
        f"  Azure SQL {out['sql_server_fqdn']} / {out['sql_database']} / schema {args.env}\n"
        f"  {out['landing_url']}{args.env}/\n[y/N] "
    )
    if not args.yes and input(question).strip().lower() != "y":
        print("Nothing loaded.")
        return
    connection = source_db.connect(out["sql_server_fqdn"], out["sql_database"], out["key_vault"])
    try:
        source_db.apply_changes(connection, args.env, changes)
        print("In Azure SQL:", source_db.row_counts(connection, args.env))
    finally:
        connection.close()
    landing.upload(source_db._az(), LOCAL_ROOT, args.env, out["storage_account"], replace=False)
    print(f"Uploaded {len(files)} files to {out['landing_url']}{args.env}/")
    print("Next: ingest + pipeline + score for this environment within the hour (database awake).")


if __name__ == "__main__":
    main()
