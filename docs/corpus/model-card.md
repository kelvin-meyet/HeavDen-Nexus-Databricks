---
doc_id: HD-ML-001
title: "Model Card: deterioration_risk"
version: "1.0"
status: current
effective_date: 2026-11-01
owner: HeavDen ML Platform Team
audience: clinicians, clinical safety officer, ML and data teams
---

# Model Card: `deterioration_risk`

> **Fictional deployment.** The model is real code trained on fully synthetic patients for the HeavDen-Nexus portfolio project. It has never been validated on real patients and must not be used for care.

## 1. What the model does

Every hour, for every monitored adult ward patient, the model estimates the **probability that the patient will be escalated in the next 6 hours**: a rapid response call or a transfer to intensive care. It returns:
- a **risk** (a calibrated probability),
- a **band** (Low or High; High is an alert), and
- the **top reasons**, from SHAP explanations grouped into clinical factors such as breathing rate or oxygen saturation.

**Intended use:** to help ward nurses notice deteriorating patients earlier, alongside NEWS2 and clinical judgement. **Not intended** for diagnosis, treatment decisions, intensive care, children, or any patient group not represented in training.

## 2. Model details

| Item | Value |
|---|---|
| Registered name | `heavden_prod.ml.deterioration_risk` (Unity Catalog), alias `@champion` |
| Version 1 algorithm | Logistic regression (median imputation, standardisation, C = 0.1) with isotonic calibration |
| Other candidate | LightGBM with isotonic calibration (lower validation AUPRC, 0.33 vs 0.38, and flagged fewer escalations within the alert budget than NEWS2, 74% vs 79%, so it failed a selection gate) |
| Inputs | 69 hourly features (72 model inputs once ward type and sex are one-hot encoded): vital-sign summaries over 1, 3 and 6 hours (mean, median, min, max, 3-hour trend), monitoring gaps, NEWS2 components, nurse observations (oxygen, new confusion), age, sex, long-term conditions, ward type, time since admission |
| Output | probability of escalation within 6 hours |
| Training data | 9 days of synthetic patient-hours (57,257 rows), ~2,000 Synthea patients, 3 sites (see the *Data Card*, HD-ML-002); validated on the next 2 days (11,248 rows) |
| Selection | best validation AUPRC among candidates that beat NEWS2 on AUPRC and flag at least as many escalations within the alert budget |

## 3. Risk bands

Escalation within 6 hours is rare: about **0.9%** of patient-hours. Calibrated probabilities are therefore small numbers, and the bands are set from the **alert budget** rather than round percentages.

An **alert** fires when a patient **enters** the High band. The same patient does not alert again for 6 hours, even if they leave and re-enter High. The High cut-off is the lowest one that keeps alerts within about **2 per nurse per 12-hour shift** (1 nurse per 5 patients), fixed on validation.

| Band | Definition (fixed on validation) | Current risk cut-off | What it means |
|---|---|---|---|
| High (alert) | the alert threshold; about 8% of patient-hours | risk ≥ about 1.5% | per hour: escalation about **8 times** more likely than average (about 1 in 14 High hours is followed by one) |
| Low | everything else, about 92% | risk below about 1.5% | routine monitoring; the ward board still shows the risk and its 6-hour change |

**Why there is no Medium band.** Several definitions of a middle band were tested: the top 10% of risk, risk doubling within 6 hours, recently High, and NEWS2 of 3 or more. None of them was followed by an escalation more often than an average patient-hour (0.5 to 1.1 times the average rate). The model's signal is concentrated above the alert threshold: deteriorating patients cross it. A Medium band would add colour without information, so the ward board shows each patient's risk and its 6-hour change instead.

Responses to each band are set by the escalation protocol (HD-CLIN-001).

## 4. Performance (version 1, test block: 12–14 November, 26 escalations)

**How it was tested.** The patient-hours are split by time: the model learns on the first 9 days, the best candidate is **chosen** on the next 2 days (validation), and the last days (the test block, 16,146 patient-hours) are scored **once**, after the choice. The model never saw the test block. Validation results were used for picking, so they flatter the model; the test results below are the honest ones.

**The comparison is fair.** NEWS2 is scored on the same hours, and each score gets its own alert threshold, fixed on validation for the same budget: at most about 2 alerts per nurse per 12-hour shift.

| Metric | Level | Champion | NEWS2 |
|---|---|---|---|
| **Escalations flagged in advance** (High in the 6 h before) | per escalation | **23 of 26 (88%)** | 19 of 26 (73%) |
| Median warning time, when flagged | per escalation | 4 hours | 4 hours |
| Alerts followed by an escalation within 6 h | per alert | 3.0% (about 1 in 34) | 3.2% (about 1 in 32) |
| Alert load on test | per nurse per shift | 2.1 | 1.8 |
| AUPRC (ranking) | per hour | **0.38** | 0.19 |
| AUROC (ranking) | per hour | 0.79 | 0.72 |

**Reading it:** within the same alert budget, the model alerted in advance for **4 more of the 26** deteriorating patients than NEWS2 (88% vs 73%), with about the same share of alerts followed by an escalation (about 1 in 33 for both). Most alerts are not followed by an escalation within 6 hours, for either score: escalation is rare (about 1% of patient-hours). NEWS2's whole-number scores can't use the budget exactly, so it raises slightly fewer alerts.

**By site (test block):** Site A 8 of 10 escalations flagged, Site B 9 of 9, Site C 6 of 7.

**Rehearsal.** Before Databricks, the same code was trained offline on the same simulation (labels from all outcomes rather than recorded ones, so the test block held 28 escalations): 25 of 28 flagged vs NEWS2 21 of 28, a difference of +14 points with a 95% interval of +3 to +28 (bootstrap over hospital stays). The test block is small, so every interval is wide. The public demo app shows that rehearsal model.

## 5. Performance by group (fairness)

Measured on the offline rehearsal (section 4), with one shared alert threshold for everyone. To be repeated on version 1 as part of monitoring.

| Group | Finding |
|---|---|
| Sites A, B, C | similar: about 2 alerts per nurse per shift each; 83-100% of escalations flagged |
| Sex | similar (86% F, 93% M flagged) |
| Age 85+ | **heaviest alert load (about 4 per nurse per shift), about 2% followed by an escalation**: most alerts on these wards are false alarms |
| Age 50-69 | lowest share flagged (4 of 6 escalations) |
| Age 18-49 | few alerts (about 0.2 per nurse per shift) but over a quarter are followed by an escalation; 3 of 3 flagged |
| COPD | NEWS2 alerts about twice as often on COPD patients as on others; the model less so. 3 of 3 COPD escalations flagged by both (too few to judge) |
| Ward type | step-down units have the highest alert load and lowest alert precision (about 2%) |

Slices with fewer than about 20 positive hours (e.g. age 18-49) are too small to draw conclusions from.

## 6. What the model relies on

Grouped SHAP importance, largest first: patient background (age, long-term conditions), blood pressure, oxygen saturation, breathing rate, heart rate. Oxygen therapy and confusion contribute little. This matches the hidden rule used by the synthetic data generator, which is only checkable because the data is synthetic.

## 7. Known limitations

- **Synthetic data.** Simulated deteriorations are smoother than real ones, so real-world gains over NEWS2 would likely be smaller.
- **Uneven alert load by age.** Wards with very old patients carry most of the alerts, mostly false alarms; younger patients rarely alert (section 5). Clinicians should not be reassured by a Low band in a young patient who looks unwell.
- **Only six long-term conditions.** The model knows COPD, heart failure, diabetes, chronic kidney disease, hypertension and atrial fibrillation (plus beta-blockers), and `n_conditions` counts only these. Conditions that matter in real wards, such as ischaemic heart disease, anaemia, dementia or alcohol and drug dependence, are not represented, because the synthetic data gives them no effect.
- **Sudden events.** Escalations without a gradual change in vital signs beforehand cannot be predicted.
- **Measurement faults.** The model trusts the monitor. Faulty sensors (e.g. FSN-2026-11, SpO2 under-reading at Site B) produce wrong risks.
- **Protocol dependence.** The model learned when staff escalated under protocol v1.0. After protocol v2.0 (2026-11-09) it under-predicts escalation until retrained (see the *Drift Incident Runbook*, scenario 5.3).
- **Alert-load drift.** On the test block the model raised 2.1 alerts per nurse per shift against a budget of 2 (fixed on validation, where it raised 1.9). The alert load needs no outcomes, so it is monitored continuously.

## 8. Monitoring and updates

Checked after each new batch of data for feature drift (PSI), prediction drift (Jensen-Shannon), alert load, data quality, and performance and calibration once outcomes are known (about 12 hours later). Retraining and promotion each need a human approval on GitHub. See the *Drift Incident Runbook*, HD-OPS-002.
