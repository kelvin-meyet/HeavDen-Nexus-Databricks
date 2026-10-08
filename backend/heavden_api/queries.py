"""SQL used by the API, written once for both modes (DuckDB in demo, Databricks SQL in live).

Keep to the common dialect: `INTERVAL n HOURS/DAYS`, `date_trunc`, `CASE`, `COALESCE`, no
engine-specific functions. Parameters use `:name`. `:as_of` is the current hour ("now"):
the snapshot's last hour in demo mode, the latest scored hour in live mode.
"""

SITE_FILTER = "(CAST(:site_id AS VARCHAR) IS NULL OR site_id = :site_id)"

SUMMARY = f"""
WITH now AS (
    SELECT risk_band FROM gold.risk_scores
    WHERE prediction_ts = :as_of AND {SITE_FILTER}
),
last_day AS (
    SELECT * FROM gold.site_kpis_hourly
    WHERE hour_ts > :as_of - INTERVAL 24 HOURS AND hour_ts <= :as_of AND {SITE_FILTER}
),
recent_alerts AS (
    SELECT * FROM gold.alerts_fact
    WHERE alert_ts > :as_of - INTERVAL 7 DAYS AND alert_ts <= :as_of AND {SITE_FILTER}
),
recent_escalations AS (
    SELECT * FROM gold.escalations_fact
    WHERE event_ts > :as_of - INTERVAL 7 DAYS AND event_ts <= :as_of AND {SITE_FILTER}
)
SELECT
    (SELECT count(*) FROM now) AS census,
    (SELECT count(*) FROM now WHERE risk_band = 'High') AS high_risk,
    (SELECT count(*) FROM now WHERE risk_band = 'Medium') AS medium_risk,
    (SELECT COALESCE(sum(alerts_raised), 0) FROM last_day) AS alerts_24h,
    (SELECT sum(alerts_raised) * :patients_per_nurse * :shift_hours / sum(census)
       FROM last_day) AS alerts_per_nurse_shift_24h,
    (SELECT COALESCE(sum(escalations), 0) FROM last_day) AS escalations_24h,
    (SELECT avg(escalated_within_6h) FROM recent_alerts) AS alert_precision_7d,
    (SELECT count(escalated_within_6h) FROM recent_alerts) AS alerts_with_outcome_7d,
    (SELECT count(*) FROM recent_escalations) AS escalations_7d,
    (SELECT avg(CASE WHEN flagged_6h_before THEN 1.0 ELSE 0.0 END)
       FROM recent_escalations) AS escalations_flagged_7d,
    (SELECT median(hours_flagged_before) FROM recent_escalations) AS median_hours_flagged_before_7d
"""

CENSUS_HOURLY = f"""
SELECT hour_ts, site_id,
       sum(census) AS census, sum(n_low) AS n_low, sum(n_medium) AS n_medium,
       sum(n_high) AS n_high, sum(alerts_raised) AS alerts, sum(escalations) AS escalations
FROM gold.site_kpis_hourly
WHERE hour_ts > :as_of - :hours * INTERVAL 1 HOUR AND hour_ts <= :as_of AND {SITE_FILTER}
GROUP BY hour_ts, site_id
ORDER BY hour_ts, site_id
"""

UNITS_NOW = f"""
SELECT site_id, unit_id, census, n_low, n_medium, n_high, round(mean_news2, 2) AS mean_news2
FROM gold.site_kpis_hourly
WHERE hour_ts = :as_of AND {SITE_FILTER}
ORDER BY site_id, unit_id
"""

ALERTS_DAILY = f"""
SELECT date_trunc('day', alert_ts) AS day, site_id,
       count(*) AS alerts,
       sum(CASE WHEN escalated_within_6h = 1 THEN 1 ELSE 0 END) AS followed_by_escalation,
       sum(CASE WHEN escalated_within_6h = 0 THEN 1 ELSE 0 END) AS no_escalation,
       sum(CASE WHEN escalated_within_6h IS NULL THEN 1 ELSE 0 END) AS outcome_pending
FROM gold.alerts_fact
WHERE alert_ts > :as_of - :days * INTERVAL 1 DAY AND alert_ts <= :as_of AND {SITE_FILTER}
GROUP BY 1, 2
ORDER BY 1, 2
"""

DEVICES_DAILY = f"""
SELECT date AS day, site_id,
       count(*) AS devices_worn,
       round(avg(uptime_pct), 2) AS uptime_pct,
       sum(battery_outages) AS battery_outages,
       sum(stuck_minutes) AS stuck_minutes,
       round(avg(mean_spo2), 2) AS mean_spo2,
       count(DISTINCT firmware) AS firmware_versions,
       max(firmware) AS latest_firmware
FROM gold.device_health_daily
WHERE messages_expected > 0
  AND date > :as_of - :days * INTERVAL 1 DAY AND date <= :as_of AND {SITE_FILTER}
GROUP BY 1, 2
ORDER BY 1, 2
"""

BOARD = """
SELECT r.encounter_id, e.patient_label, e.age, e.sex, e.conditions,
       r.site_id, r.unit_id, e.bed_id, e.admit_ts,
       r.risk, r.risk_band, r.news2_total, r.top_factors,
       earlier.risk AS risk_6h_ago
FROM gold.risk_scores r
JOIN gold.encounters e ON e.encounter_id = r.encounter_id
LEFT JOIN gold.risk_scores earlier
       ON earlier.encounter_id = r.encounter_id
      AND earlier.prediction_ts = :as_of - INTERVAL 6 HOURS
WHERE r.prediction_ts = :as_of
  AND (CAST(:site_id AS VARCHAR) IS NULL OR r.site_id = :site_id)
  AND (CAST(:unit_id AS VARCHAR) IS NULL OR r.unit_id = :unit_id)
  AND (CAST(:band AS VARCHAR) IS NULL OR r.risk_band = :band)
ORDER BY r.risk DESC, r.news2_total DESC, r.encounter_id
LIMIT :limit
"""

PATIENT = """
SELECT e.*, r.risk, r.risk_band, r.news2_total, r.top_factors, r.model_version
FROM gold.encounters e
LEFT JOIN gold.risk_scores r
       ON r.encounter_id = e.encounter_id AND r.prediction_ts = :as_of
WHERE e.encounter_id = :encounter_id
"""

PATIENT_SERIES = """
SELECT f.prediction_ts, r.risk, r.risk_band, f.news2_total,
       f.heart_rate_mean_1h, f.resp_rate_mean_1h, f.spo2_min_1h, f.temp_c_mean_1h,
       f.sbp_mean_1h, f.on_oxygen, f.o2_flow_lpm, f.acvpu, f.unit_id
FROM gold.patient_hour_features f
JOIN gold.risk_scores r
  ON r.encounter_id = f.encounter_id AND r.prediction_ts = f.prediction_ts
WHERE f.encounter_id = :encounter_id
  AND f.prediction_ts > :as_of - :hours * INTERVAL 1 HOUR AND f.prediction_ts <= :as_of
ORDER BY f.prediction_ts
"""

PATIENT_ALERTS = """
SELECT alert_ts, risk, news2_total, escalated_within_6h, hours_to_escalation
FROM gold.alerts_fact
WHERE encounter_id = :encounter_id AND alert_ts <= :as_of
ORDER BY alert_ts
"""

FEATURE_ROW = """
SELECT * FROM gold.patient_hour_features
WHERE encounter_id = :encounter_id AND prediction_ts = :as_of
"""
