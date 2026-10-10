---
doc_id: HD-DATA-001
title: Gold Data Dictionary
version: "1.0"
status: current
effective_date: 2026-11-01
owner: HeavDen Data Engineering
audience: analysts, ML team, AI assistant (Genie space)
---

# Gold Data Dictionary

> **Synthetic data.** Tables live in the Unity Catalog schema `gold` of `heavden_dev`, `heavden_staging` and `heavden_prod`. All times are UTC.

## 1. Layers in one paragraph

**Bronze** holds raw data as landed (device JSON, copies of the Azure SQL tables). **Silver** holds cleaned, de-duplicated, typed data with history (patients, encounters, vitals, outcomes). **Gold** holds tables ready for dashboards, the model and the AI assistant. Analysts and the assistant should query **gold** only.

## 2. `gold.patient_hour_features`

One row per patient per hour in hospital: the model's input.

| Column | Meaning |
|---|---|
| `encounter_id`, `patient_id` | the hospital stay and the patient |
| `prediction_ts` | the hour the row describes; features use only data up to this time |
| `site_id`, `unit_id`, `unit_type` | where the patient was at `prediction_ts` |
| `<vital>_mean_1h`, `_median_1h`, `_min_1h`, `_max_1h` | summary of the last hour of readings; `<vital>` is `heart_rate`, `resp_rate`, `spo2`, `temp_c`, `sbp` or `dbp` |
| `<vital>_mean_3h`, `<vital>_mean_6h` | averages over the last 3 and 6 hours |
| `<vital>_trend_3h` | change across the last 3 hours (last hour's mean minus the mean of the hour two hours before it) |
| `readings_1h`, `missing_1h`, `missing_6h` | number of readings received, and expected readings missing (dropouts, battery) |
| `hours_since_admission` | time since the stay began |
| `age`, `sex` | at admission |
| `copd`, `heart_failure`, `diabetes`, `ckd`, `hypertension`, `atrial_fibrillation`, `on_beta_blocker`, `n_conditions` | long-term condition and medication flags, active at admission. Only these six conditions are flagged, and `n_conditions` counts only them (see the *Data Card* and `gold.condition_flag_reference`) |
| `on_oxygen`, `o2_flow_lpm`, `new_confusion`, `hours_since_obs` | from the most recent nurse observation before `prediction_ts` |
| `news2_*`, `news2_total` | the seven NEWS2 component scores and total (see the *NEWS2 Reference Card*) |
| `label` | 1 if escalated in the next 6 hours, 0 if not, null while the 6-hour window is still open (in the demo snapshot; on the data platform labels live in `gold.training_set`) |
| `label_known_at` | when the label became known (`prediction_ts` + 6 hours) |

## 2a. `gold.training_set`

The rows of `gold.patient_hour_features` whose label is final, with the label added: what the model trains on.

| Column | Meaning |
|---|---|
| all columns of `gold.patient_hour_features` | as above |
| `label` | 1 if an escalation was **recorded** in the 6 hours after `prediction_ts`, else 0 |
| `label_known_at` | when the label became final: the escalation's recording time, or `prediction_ts` + 12 hours for a 0 |

Escalations are charted about 6 hours after they happen, so a 0 is only final once the 6-hour window **and** the 6-hour recording delay have passed (12 hours). Hours whose label isn't final yet are left out rather than guessed as 0.

## 3. `gold.risk_scores`

One row per patient per hour, written by the hourly batch scoring job.

| Column | Meaning |
|---|---|
| `encounter_id`, `patient_id`, `site_id`, `unit_id`, `prediction_ts` | as above |
| `risk` | calibrated probability of escalation within 6 hours |
| `risk_band` | `Low` or `High` (High = at or above the alert threshold; see the *Model Card*, section 3) |
| `top_factors` | the 3 main reasons as a JSON array: factor (e.g. "breathing rate"), direction (raises or lowers risk), size in log-odds, and the reading that drove it |
| `news2_total` | NEWS2 at the same hour, for comparison |
| `model_version` | registered model version that produced the score |

## 4. `gold.site_kpis_hourly`

One row per unit per hour.

| Column | Meaning |
|---|---|
| `site_id`, `unit_id`, `hour_ts` | where and when |
| `census` | patients on the unit during the hour |
| `n_low`, `n_high` | patients in each risk band |
| `alerts_raised` | new High alerts |
| `escalations` | rapid response calls and ICU transfers that happened |
| `mean_news2` | average NEWS2 |

## 5. `gold.alerts_fact`

One row per High alert. An alert fires when a patient **enters** the High band. To limit alarm fatigue, a patient who already alerted in the previous **6 hours** does not alert again, even if they leave and re-enter the High band.

| Column | Meaning |
|---|---|
| `alert_id`, `encounter_id`, `patient_id`, `site_id`, `unit_id` | identifiers |
| `alert_ts` | when the patient entered the High band |
| `risk`, `news2_total`, `top_factors` | the score and reasons at alert time |
| `escalated_within_6h` | whether an escalation followed (null until known) |
| `hours_to_escalation` | warning time, if escalated |

## 6. `gold.escalations_fact`

One row per escalation (rapid response call or ICU transfer).

| Column | Meaning |
|---|---|
| `encounter_id`, `patient_id`, `site_id`, `unit_id` | the stay, and where the patient was just before |
| `event_ts`, `event_type` | when, and `rapid_response` or `icu_transfer` |
| `risk_before` | the last risk score before the event |
| `flagged_6h_before` | whether the patient was in the High band at any hour in the 6 hours before |
| `hours_flagged_before` | how many hours before the event they were first flagged High |

## 7. `gold.device_health_daily`

One row per device per day.

| Column | Meaning |
|---|---|
| `device_id`, `site_id`, `date` | identifiers |
| `messages_received`, `messages_expected`, `uptime_pct` | completeness |
| `stuck_minutes`, `battery_outages`, `min_battery_pct` | fault indicators |
| `firmware` | firmware version(s) reported that day |
| `mean_spo2` | daily mean SpO2 across patients on the device; a drop at one site can reveal a sensor fault |

## 7a. `gold.condition_flag_reference`

Which diagnosis and medication descriptions set each model flag, and for how many patients: the audit trail for the condition flags.

| Column | Meaning |
|---|---|
| `flag` | `copd`, `heart_failure`, `diabetes`, `ckd`, `hypertension`, `atrial_fibrillation` or `on_beta_blocker` |
| `source` | `condition` or `medication` |
| `code`, `description` | the SNOMED-CT (conditions) or RxNorm (medications) code and its text |
| `patients` | how many patients have that description |

Descriptions that set no flag (for example prediabetes, anaemia or social findings) are not listed: the simulation gives them no effect on vital signs or risk.

## 8. `gold.encounters`

One row per hospital stay, for display. Names and dates of birth are never included.

| Column | Meaning |
|---|---|
| `encounter_id`, `patient_label` | the stay, and a short display id for the patient (e.g. `P-1042`) |
| `age`, `sex`, `conditions` | at admission; conditions as a readable list |
| `site_id`, `unit_id`, `bed_id` | current (or last) location |
| `admit_ts`, `discharge_ts` | stay start and end (end is null while still in hospital) |

## 9. Common questions and where to look

- *How many high-risk patients are at a site now?* `gold.risk_scores`, latest `prediction_ts`, `risk_band = 'High'`.
- *How many alerts did a unit raise yesterday?* `gold.alerts_fact` or `gold.site_kpis_hourly`.
- *Did alerts lead to escalations?* `gold.alerts_fact.escalated_within_6h`.
- *Were escalations flagged in advance?* `gold.escalations_fact.flagged_6h_before`.
- *Are devices at Site B healthy?* `gold.device_health_daily`.
- *Why is a patient high risk?* `top_factors` in `gold.risk_scores`, or ask the assistant to explain the patient.
