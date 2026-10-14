"""The monitor's tables, custom metrics, settings, refresh and shadow numbers
(heavden.monitoring.profiling)."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from heavden.ml import evaluate
from heavden.ml.batch_scoring import to_spark
from heavden.monitoring import drift_check as dc
from heavden.monitoring import drift_metrics
from heavden.monitoring import profiling as pf

T0 = pd.Timestamp("2026-09-20", tz="UTC")
SITES = ("SITE_A", "SITE_B", "SITE_C")


def _log(stays: int, hours: int, seed: int, version: str, start=T0) -> pd.DataFrame:
    """Prediction-log rows at three sites, with tied risks and stored bands."""
    rng = np.random.default_rng(seed)
    df = pd.MultiIndex.from_product(
        [[f"E{seed}-{i}" for i in range(stays)], pd.date_range(start, periods=hours, freq="h")],
        names=["encounter_id", "prediction_ts"],
    ).to_frame(index=False)
    df["site_id"] = np.array(SITES)[df["encounter_id"].str.split("-").str[1].astype(int) % 3]
    df["risk"] = np.round(rng.beta(1, 40, len(df)), 3)
    df["risk_band"] = np.where(df["risk"] >= 0.06, "High", "Low")
    df["model_version"] = version
    return df


def _features(log: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return log[pf.KEYS].assign(**{f: rng.normal(50, 10, len(log)) for f in dc.FEATURES})


def _labels(log: pd.DataFrame, known_before: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """`gold.training_set` rows: final labels only, for hours before `known_before`."""
    rng = np.random.default_rng(seed)
    rows = log[log["prediction_ts"] < known_before]
    label = (rng.random(len(rows)) < rows["risk"] * 3).astype(float)
    return _features(rows).assign(site_id=rows["site_id"].to_numpy(), label=label)


def _alerts(log: pd.DataFrame) -> pd.DataFrame:
    onset = evaluate.alert_onsets(log, log["risk_band"].eq("High").to_numpy())
    alerts = log[onset][["encounter_id", "prediction_ts"]]
    return alerts.rename(columns={"prediction_ts": "alert_ts"})


def test_inference_rows_carry_text_labels_onsets_and_the_reference_deciles():
    log = pd.concat([_log(30, 24, 1, "18"), _log(30, 24, 2, "19", T0 + pd.Timedelta(days=1))])
    known_before = T0 + pd.Timedelta(hours=36)
    labels, alerts = _labels(log, known_before), _alerts(log)
    rng = np.random.default_rng(9)
    reference = {v: np.round(rng.beta(1, 40, 2000), 3) for v in ("18", "19")}
    edges = {v: drift_metrics.baseline_bins(r) for v, r in reference.items()}

    rows = pf.inference_rows(log, _features(log), labels, alerts, edges)

    assert list(rows.columns) == pf.COLUMNS and len(rows) == len(log)
    assert set(rows["predicted"]) == {"0", "1"}
    assert (rows["predicted"].eq("1") == rows["risk_band"].eq("High")).all()
    unknown = rows["prediction_ts"] >= known_before
    assert rows.loc[unknown, "label"].isna().all()
    assert set(rows.loc[~unknown, "label"]) == {"0", "1"}
    assert rows["alert_onset"].sum() == len(alerts)
    for version, part in rows.groupby("model_version"):  # the monitor's bins = our JS's bins
        counts = part["risk_decile"].str[1:].astype(int).value_counts().sort_index()
        shares = np.bincount(counts.index, counts.to_numpy(), len(edges[version]) - 1) / len(part)
        assert shares == pytest.approx(drift_metrics.bin_shares(part["risk"], edges[version]))


def test_a_risk_on_a_decile_edge_lands_in_the_upper_bin_like_numpy():
    edges = drift_metrics.baseline_bins(np.arange(100.0))
    on_edges = edges[1:-1]
    expected = [np.histogram([x], bins=edges)[0].argmax() for x in on_edges]
    assert list(pf.deciles(on_edges, edges)) == [f"d{i}" for i in expected]


def test_reference_rows_use_the_versions_own_threshold():
    log = _log(30, 24, 3, "21")
    reference = _labels(log, T0 + pd.Timedelta(days=2))
    risk = np.linspace(0, 0.1, len(reference))

    rows = pf.reference_rows(reference, risk, threshold=0.05, version="21")

    high = risk >= 0.05
    assert (rows["model_version"] == "21").all() and rows["label"].notna().all()
    assert (rows["predicted"].eq("1") == high).all()
    assert (rows["alert_onset"] == evaluate.alert_onsets(reference, high)).all()
    assert sorted(rows["risk_decile"].unique())[0] == "d0"


def test_a_merge_sends_new_hours_and_labels_that_became_known_only():
    rows = pd.DataFrame(
        {
            "encounter_id": ["A", "B", "C"],
            "prediction_ts": [T0, T0, T0],
            "label": ["0", "1", None],
        }
    )
    existing = pd.DataFrame(
        {"encounter_id": ["A", "B"], "prediction_ts": [T0, T0], "label": ["0", None]}
    )
    assert list(pf.to_write(rows, existing)["encounter_id"]) == ["B", "C"]
    assert len(pf.to_write(rows, None)) == 3


def test_custom_metrics_match_the_python_the_check_uses(spark):
    rng = np.random.default_rng(7)
    n = 6000
    df = pd.DataFrame(
        {"site_id": rng.choice(SITES, n), "risk": np.round(rng.beta(1, 40, n), 3)}  # ties
    )
    df["label"] = np.where(rng.random(n) < df["risk"] * 3, "1", "0").astype(object)
    df.loc[rng.random(n) < 0.15, "label"] = None  # outcomes not known yet
    df.loc[df["site_id"] == "SITE_C", "label"] = "0"  # a site without escalations
    df["alert_onset"] = (rng.random(n) < 0.03).astype(int)
    spark.createDataFrame(df).createOrReplaceTempView("monitored")
    sql = ", ".join(f"{d} AS {k}" for k, d in pf.CUSTOM_METRICS.items())

    got = spark.sql(f"SELECT site_id, {sql} FROM monitored GROUP BY site_id").toPandas()
    empty = spark.sql(f"SELECT {sql} FROM monitored WHERE false").toPandas()

    budget = evaluate.AlertBudget()
    for site, part in df.groupby("site_id"):
        g = got.set_index("site_id").loc[site]
        known = part.dropna(subset=["label"])
        y = known["label"].astype(int)
        assert g["brier"] == pytest.approx(brier_score_loss(y, known["risk"]), abs=1e-12)
        alerts = budget.per_nurse_shift(part["alert_onset"].sum(), len(part))
        assert g["alerts_per_nurse_shift"] == pytest.approx(alerts, abs=1e-12)
        if y.sum():
            assert g["auprc"] == pytest.approx(average_precision_score(y, known["risk"]), abs=1e-12)
            assert g["auroc"] == pytest.approx(roc_auc_score(y, known["risk"]), abs=1e-12)
        else:  # no escalations: no ranking to judge
            assert pd.isna(g["auprc"]) and pd.isna(g["auroc"])
    assert empty.isna().all(axis=None)


SPEC = pf.MonitorSpec("heavden_staging", "/Workspace/x/monitoring")


def test_the_monitor_settings_are_the_same_in_both_apis():
    dq = pytest.importorskip("databricks.sdk.service.dataquality")
    old = pf.legacy_settings(SPEC)
    new = pf.data_quality_config(SPEC, output_schema_id="schema-id")

    assert old["inference_log"].prediction_proba_col is None  # the newer API has none
    assert "prediction_proba_column" not in dq.InferenceLogConfig.__dataclass_fields__
    log = new.inference_log
    assert (log.prediction_column, log.label_column, log.model_id_column, log.timestamp_column) == (
        old["inference_log"].prediction_col,
        old["inference_log"].label_col,
        old["inference_log"].model_id_col,
        old["inference_log"].timestamp_col,
    )
    assert [g.value.removeprefix("AGGREGATION_GRANULARITY_") for g in log.granularities] == [
        g.upper().replace(" ", "_") for g in old["inference_log"].granularities
    ]
    pairs = lambda ms: [(m.name, m.definition, m.input_columns) for m in ms]  # noqa: E731
    assert pairs(new.custom_metrics) == pairs(old["custom_metrics"])
    assert new.baseline_table_name == old["baseline_table_name"] == SPEC.baseline
    assert new.slicing_exprs == old["slicing_exprs"] == ["site_id"]
    assert new.skip_builtin_dashboard


def _info(**changes):
    from databricks.sdk.service import catalog as c

    s = pf.legacy_settings(SPEC) | changes
    return c.MonitorInfo(
        output_schema_name=s["output_schema_name"],
        table_name=SPEC.table,
        status=c.MonitorInfoStatus.MONITOR_STATUS_ACTIVE,
        profile_metrics_table_name=SPEC.output_table("profile"),
        drift_metrics_table_name=SPEC.output_table("drift"),
        monitor_version=0,
        baseline_table_name=s["baseline_table_name"],
        inference_log=s["inference_log"],
        slicing_exprs=s["slicing_exprs"],
        custom_metrics=s["custom_metrics"],
    )


class FakeMonitors:
    """`w.quality_monitors` with a monitor and refreshes that move through given states.

    Time passes between two looks at the monitor (`statuses`, one per `get`); like the service,
    it refuses to list refreshes while the monitor isn't ACTIVE."""

    def __init__(self, info=None, refreshes: dict | None = None, statuses=("ACTIVE",)):
        from databricks.sdk.service import catalog as c

        self.c, self.info, self.calls = c, info, []
        self.states = {k: list(v) for k, v in (refreshes or {}).items()}
        self.statuses = list(statuses)

    def _refresh(self, rid: int, advance: bool):
        states = self.states[rid]
        state = states.pop(0) if advance and len(states) > 1 else states[0]
        return self.c.MonitorRefreshInfo(
            refresh_id=rid, state=self.c.MonitorRefreshInfoState(state), start_time_ms=0
        )

    def get(self, table):
        from databricks.sdk.errors import NotFound

        if self.info is None:
            raise NotFound("no monitor")
        if len(self.statuses) > 1:
            self.statuses.pop(0)
        status = self.c.MonitorInfoStatus(f"MONITOR_STATUS_{self.statuses[0]}")
        return dataclasses.replace(self.info, status=status, latest_monitor_failure_msg="boom")

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))

    def list_refreshes(self, table):
        from databricks.sdk.errors import BadRequest

        if self.statuses[0] != "ACTIVE":
            raise BadRequest(f"not supported when your monitor has status {self.statuses[0]}")
        listed = [self._refresh(rid, advance=False) for rid in self.states]
        return self.c.MonitorRefreshListResponse(refreshes=listed)

    def get_refresh(self, table, rid):
        return self._refresh(rid, advance=True)

    def run_refresh(self, table):
        self.calls.append(("run_refresh", table))
        self.states[99] = ["PENDING", "RUNNING", "SUCCESS"]
        return self._refresh(99, advance=False)


def _client(monitors: FakeMonitors, table_exists: bool = True):
    tables = SimpleNamespace(exists=lambda name: SimpleNamespace(table_exists=table_exists))
    return SimpleNamespace(quality_monitors=monitors, tables=tables)


def test_the_monitor_is_created_updated_or_left_alone():
    created = FakeMonitors()
    assert pf.ensure_monitor(_client(created), SPEC) == "created"
    kind, kwargs = created.calls[0]
    assert kind == "create" and kwargs["skip_builtin_dashboard"] is True
    assert kwargs["inference_log"].prediction_proba_col is None

    assert pf.ensure_monitor(_client(FakeMonitors(_info())), SPEC) == "unchanged"

    old = pf.legacy_settings(SPEC)["custom_metrics"][:-1]  # a metric added since
    changed = FakeMonitors(_info(custom_metrics=old))
    assert pf.ensure_monitor(_client(changed), SPEC) == "updated"
    assert changed.calls[0][0] == "update"

    with pytest.raises(LookupError, match="run the score job first"):
        pf.ensure_monitor(_client(FakeMonitors(), table_exists=False), SPEC)


def test_a_new_monitor_is_awaited_while_pending_then_its_own_first_refresh():
    # 14 Oct, staging: listing the refreshes of a monitor still being set up failed
    first = {1: ["RUNNING", "RUNNING", "SUCCESS"]}
    monitors = FakeMonitors(_info(), first, statuses=["PENDING", "PENDING", "ACTIVE"])
    done = pf.refresh_and_wait(_client(monitors), SPEC.table, just_created=True, poll_s=0)
    assert done.refresh_id == 1 and not monitors.calls  # no second refresh started


def test_a_refresh_already_running_is_awaited_then_a_new_one_started():
    monitors = FakeMonitors(_info(), refreshes={1: ["RUNNING", "SUCCESS"]})
    done = pf.refresh_and_wait(_client(monitors), SPEC.table, just_created=False, poll_s=0)
    assert done.refresh_id == 99 and monitors.calls == [("run_refresh", SPEC.table)]


def test_a_failed_refresh_or_monitor_stops_the_check():
    monitors = FakeMonitors(_info(), refreshes={1: ["RUNNING", "FAILED"]})
    with pytest.raises(RuntimeError, match="FAILED"):
        pf.refresh_and_wait(_client(monitors), SPEC.table, just_created=True, poll_s=0)

    broken = FakeMonitors(_info(), statuses=["PENDING", "ERROR"])
    with pytest.raises(RuntimeError, match="MONITOR_STATUS_ERROR boom"):
        pf.refresh_and_wait(_client(broken), SPEC.table, just_created=True, poll_s=0)


def test_all_the_waiting_shares_one_deadline():
    stuck = FakeMonitors(_info(), statuses=["PENDING"])
    with pytest.raises(TimeoutError, match="still MONITOR_STATUS_PENDING"):
        pf.refresh_and_wait(_client(stuck), SPEC.table, True, poll_s=0, timeout_s=-1)

    slow = FakeMonitors(_info(), refreshes={1: ["RUNNING"]})
    with pytest.raises(TimeoutError, match="refresh 1 of .* still running"):
        pf.refresh_and_wait(_client(slow), SPEC.table, False, poll_s=0, timeout_s=-1)


def _window_rows(granularity: str, starts, length: pd.Timedelta, **values) -> pd.DataFrame:
    """Monitor output rows: every window x (all sites + each site) x model version."""
    rows = []
    for start in starts:
        for version, count in (("*", 600), ("18", 100), ("19", 500)):
            for key, value in ((None, None), *(("site_id", s) for s in SITES)):
                rows.append(
                    {
                        "window_start": start,
                        "window_end": start + length,
                        "granularity": granularity,
                        "slice_key": key,
                        "slice_value": value,
                        "model_version": version,
                        "count": count if key is None else count // 3,
                        **{k: v(start, version, value) for k, v in values.items()},
                    }
                )
    return pd.DataFrame(rows)


def _monitor_output(days: int):
    day, week = pd.Timedelta(days=1), pd.Timedelta(days=7)
    marker = lambda start, version, site: start.day + (0.5 if site else 0)  # noqa: E731
    metrics = {m: marker for m in pf.CUSTOM_METRICS}
    profile = pd.concat(
        [
            _window_rows("1 day", [T0 + i * day for i in range(days)], day, **metrics),
            _window_rows("1 week", [T0, T0 + week], week, **metrics),  # the second is partial
        ]
    )
    drift = _window_rows(
        "1 day",
        [T0 + i * day for i in range(days)],
        day,
        js_distance=lambda start, version, site: 0.01 * int(version) if version != "*" else 0,
    )
    return profile, drift


def test_the_monitor_numbers_come_from_the_days_and_weeks_that_end_with_the_data():
    profile, drift = _monitor_output(days=7)
    end = T0 + pd.Timedelta(days=7)  # data through 26 Sep 23:00

    rows = pd.DataFrame(pf.monitor_signals(profile, drift, end)).set_index(["signal", "site_id"])

    assert (rows["level"] == pf.SHADOW).all()
    alerts = rows.loc["lm_alerts_per_nurse_shift"]
    assert alerts.loc[dc.ALL_SITES, "value"] == 26 and alerts.loc["SITE_B", "value"] == 26.5
    js = rows.loc["lm_prediction_js"]  # the version that scored most of the day, v19
    assert (js["feature"] == "v19").all() and js["value"].tolist() == pytest.approx([0.19] * 4)
    assert js.loc[dc.ALL_SITES, "n"] == 500
    week = rows.loc["lm_auprc_week"]  # 20-27 Sep is whole; 27 Sep-4 Oct is not over yet
    assert week.loc[dc.ALL_SITES, "value"] == 20 and len(week) == 4


def test_a_monitor_behind_the_data_stops_the_check():
    profile, drift = _monitor_output(days=6)  # through 25 Sep
    with pytest.raises(pf.MonitorBehind):
        pf.monitor_signals(profile, drift, T0 + pd.Timedelta(days=7))


def test_while_the_monitor_does_not_decide_its_failure_is_reported_not_raised(monkeypatch):
    def broken(spec):
        raise PermissionError("the service principal can't manage the table")

    monkeypatch.setattr(pf, "refresh_monitor", broken)
    rows, status = pf.shadow_rows(None, SPEC, T0 + pd.Timedelta(days=7))
    assert status.startswith("failed: PermissionError: the service principal")
    assert [(r["signal"], r["feature"], r["level"]) for r in rows] == [
        ("lm_status", status, pf.SHADOW)
    ]

    monkeypatch.setattr(pf, "MONITOR_DECIDES", True)  # stage 2: a failure stops the check
    with pytest.raises(PermissionError):
        pf.shadow_rows(None, SPEC, T0 + pd.Timedelta(days=7))


def test_a_working_monitor_leaves_its_status_and_numbers(monkeypatch):
    monkeypatch.setattr(pf, "refresh_monitor", lambda spec: "refresh 7 succeeded in 12.4 min")
    monkeypatch.setattr(pf, "read_monitor_tables", lambda spark, spec: _monitor_output(days=7))
    rows, status = pf.shadow_rows(None, SPEC, T0 + pd.Timedelta(days=7))
    assert status == "ok: refresh 7 succeeded in 12.4 min"
    assert rows[0]["signal"] == "lm_status" and len(rows) == 1 + 20


def test_shadow_rows_never_change_the_status_or_the_kinds_of_drift():
    judged = [dc._row("feature_psi", 0.15, 900, "spo2_mean_1h", "SITE_B")]  # warn
    status = pf._shadow("lm_status", float("nan"), 0, dc.ALL_SITES) | {"feature": "ok: refresh"}
    shadow = [status, pf._shadow("lm_alerts_per_nurse_shift", 9.0, 900, dc.ALL_SITES)]
    metrics = pd.DataFrame(judged + shadow)

    assert dc.summary(metrics) == dc.summary(pd.DataFrame(judged))
    assert dc.act_kinds(metrics) == []
    assert dc.shadow_line(metrics) == "Monitor (shadow, not judged): alerts_per_nurse_shift 9.000"


# --- On Delta Lake (Linux / CI): the score job's writes and the reader of the monitor's tables


def _delta_tables(spark):
    from conftest import DELTA

    if not DELTA:
        pytest.skip("Delta Lake runs on Linux (CI), not Windows")
    spark.sql("CREATE DATABASE IF NOT EXISTS monitoring")
    return pf.MonitorSpec("spark_catalog", "")


def _version(spark, table: str) -> int:
    return spark.sql(f"DESCRIBE HISTORY {table}").agg({"version": "max"}).first()[0]


def test_the_inference_log_takes_new_hours_late_labels_and_a_rescore(spark):
    spec = _delta_tables(spark)
    loaded = []

    def reference_for(version):
        loaded.append(version)
        rows = _labels(_log(20, 24, 50 + int(version), version), T0 + pd.Timedelta(days=2))
        return pf.reference_rows(rows, np.linspace(0, 0.1, len(rows)), 0.06, version)

    def load(log, labels_before, rescore=False):
        labels = _labels(log, labels_before)
        return pf.update_tables(
            spark, spec.catalog, log, _features(log), labels, _alerts(log), reference_for, rescore
        )

    def table():
        return spark.read.table(spec.table).toPandas()

    # first load: two versions' hours, labels known up to 21 Sep 12:00
    first = pd.concat([_log(30, 24, 1, "18"), _log(30, 24, 2, "19", T0 + pd.Timedelta(days=1))])
    message = load(first, T0 + pd.Timedelta(hours=36))
    before = table()
    assert len(before) == len(first) and not before.duplicated(pf.KEYS).any()
    assert loaded == ["18", "19"] and "baseline added for v18, v19" in message
    properties = spark.sql(f"SHOW TBLPROPERTIES {spec.table}").toPandas()
    assert properties.set_index("key").loc["delta.enableChangeDataFeed", "value"] == "true"

    # second load: 12 more hours of v19, and the labels known by 22 Sep 12:00
    more = _log(30, 36, 2, "19", T0 + pd.Timedelta(days=1))
    second = pd.concat([first, more[more["prediction_ts"] >= T0 + pd.Timedelta(days=2)]])
    version = _version(spark, spec.table)
    message = load(second, T0 + pd.Timedelta(hours=60))
    after = table()
    new = len(second) - len(first)
    waiting = before[before["label"].isna()][pf.KEYS]
    filled = int(after.merge(waiting, on=pf.KEYS)["label"].notna().sum())
    assert filled > 0 and message.startswith(
        f"{spec.table}: {new:,} hours added, {filled:,} labels filled in"
    )
    assert len(after) == len(second) and not after.duplicated(pf.KEYS).any()
    assert loaded == ["18", "19"]  # known versions' baselines aren't rebuilt
    changes = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", version + 1)
        .table(spec.table)
        .groupBy("_change_type")
        .count()
        .toPandas()
        .set_index("_change_type")["count"]
    )
    # what the monitor's next refresh reads: the new hours and the labels filled in
    assert changes["insert"] == new and changes["update_postimage"] == filled

    # the same load again sends nothing
    assert "0 hours added, 0 labels filled in" in load(second, T0 + pd.Timedelta(hours=60))

    # a rescore: every hour now v21's, rows replaced in place
    rescored = second.assign(model_version="21", risk=second["risk"] * 0.9)
    assert "every hour" in load(rescored, T0 + pd.Timedelta(hours=60), rescore=True)
    final = table()
    assert len(final) == len(second) and set(final["model_version"]) == {"21"}
    assert loaded[-1] == "21"
    baseline = spark.read.table(spec.baseline).toPandas()
    assert set(baseline["model_version"]) == {"18", "19", "21"}


def test_the_monitor_tables_are_read_by_their_window_columns(spark):
    spec = _delta_tables(spark)
    profile, drift = _monitor_output(days=7)
    for kind, df in (("profile", profile), ("drift", drift)):
        extra = (
            {"column_name": ":table", "log_type": "INPUT"}
            if kind == "profile"
            else {"column_name": "risk_decile", "drift_type": "BASELINE"}
        )
        written = to_spark(spark, df.assign(**extra)).selectExpr(
            "named_struct('start', window_start, 'end', window_end) AS window", "*"
        )
        written.drop("window_start", "window_end").write.format("delta").mode(
            "overwrite"
        ).saveAsTable(spec.output_table(kind))

    read = pf.read_monitor_tables(spark, spec)
    rows = pd.DataFrame(pf.monitor_signals(*read, T0 + pd.Timedelta(days=7)))

    expected = pd.DataFrame(pf.monitor_signals(profile, drift, T0 + pd.Timedelta(days=7)))
    pd.testing.assert_frame_equal(rows, expected)
