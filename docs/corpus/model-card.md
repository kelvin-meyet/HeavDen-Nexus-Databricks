---
doc_id: HD-ML-001
title: "Model Card: deterioration_risk"
version: "2.0"
status: current
effective_date: 2026-10-14
supersedes: HD-ML-001 v1.0 (model version 1)
model_version: "3"
owner: HeavDen ML Platform Team
audience: clinicians, clinical safety officer, ML and data teams
---

# Model Card: `deterioration_risk` (card v2.0, model version 3)

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
| Version 3 algorithm | LightGBM (gradient-boosted trees, 29 rounds chosen by early stopping) with isotonic calibration |
| Other candidate | Logistic regression (C = 0.1) with isotonic calibration: it also beat NEWS2, but had the lower validation AUPRC (0.15 vs 0.16) and flagged fewer validation escalations within the alert budget (65% vs 85%) |
| Inputs | 69 hourly features (72 model inputs once ward type and sex are one-hot encoded): vital-sign summaries over 1, 3 and 6 hours (mean, median, min, max, 3-hour trend), monitoring gaps, NEWS2 components, nurse observations (oxygen, new confusion), age, sex, long-term conditions, ward type, time since admission. Unchanged from version 1. |
| Output | probability of escalation within 6 hours |
| Training data | 20 September to 9 October (123,566 synthetic patient-hours), ~2,000 Synthea patients, 3 sites (see the *Data Card*, HD-ML-002). The 3,592 patient-hours at Northshore during the SpO2 sensor fault (field safety notice FSN-2026-10, 4 to 6 October) were left out. |
| Validation and test | validation 9 October 18:00 to 11 October 11:00 (11,487 patient-hours); test 11 October 18:00 to 13 October 17:00 (11,307), both entirely under escalation protocol v2.0 |
| Selection | best validation AUPRC among candidates that beat NEWS2 on AUPRC and flag at least as many escalations within the alert budget |
| Promotion | beat version 1 at the same alert budget on resampled patient stays (section 4), then approved by a person; `@champion` since 16 October 2026. New patient-hours from 14 October on are scored by version 3; earlier hours keep the version that scored them. |

**Version history**

| Version | Algorithm | Trained on | Outcome |
|---|---|---|---|
| 1 | Logistic regression | 20 to 29 September (protocol v1.0) | first champion; tested on 1 to 3 October (card v1.0). Its accuracy fell after protocol v2.0 |
| 2 | Logistic regression | to 3 October, validated 4 to 5 October (protocol v1.0 only) | passed its gates, but its retrain was approved before enough data existed under protocol v2.0; **promotion rejected** |
| 3 | LightGBM | to 9 October, validated and tested under protocol v2.0 | **current champion**; version 1 kept as `@previous_champion` for rollback |

## 3. Risk bands

Escalation within 6 hours is rare, so calibrated probabilities are small numbers and the bands are set from the **alert budget** rather than round percentages. Escalations became more frequent under protocol v2.0: about 12 a day from 8 to 13 October, against about 8 a day before it. On the test days **about 1.5%** of patient-hours were followed by one within 6 hours (about 0.9% before the change).

An **alert** fires when a patient **enters** the High band. The same patient does not alert again for 6 hours, even if they leave and re-enter High. The High cut-off is the lowest one that keeps alerts within about **2 per nurse per 12-hour shift** (1 nurse per 5 patients), fixed on validation.

| Band | Definition (fixed on validation) | Current risk cut-off | What it means |
|---|---|---|---|
| High (alert) | the alert threshold; about 8% of patient-hours on validation (9% on test) | risk ≥ about 1.6% | about 1 alert in 25 is followed by an escalation within 6 hours (section 4) |
| Low | everything else | risk below about 1.6% | routine monitoring; the ward board still shows the risk and its 6-hour change |

**Why there is no Medium band.** On the offline rehearsal, several definitions of a middle band were tested: the top 10% of risk, risk doubling within 6 hours, recently High, and NEWS2 of 3 or more. None of them was followed by an escalation more often than an average patient-hour (0.5 to 1.1 times the average rate). The model's signal is concentrated above the alert threshold: deteriorating patients cross it. A Medium band would add colour without information, so the ward board shows each patient's risk and its 6-hour change instead.

Responses to each band are set by the escalation protocol (HD-CLIN-001).

## 4. Performance (version 3, test block: 11 to 13 October, 28 escalations)

**How it was tested.** The patient-hours are split by time with 6-hour gaps: the model learns on everything up to 9 October, the best candidate is **chosen** on the next 2 days (validation), and the last 2 days (the test block) are scored **once**, after the choice. The model never saw the test block. Validation results were used for picking, so they flatter the model; the test results below are the honest ones.

**The comparison is fair.** Version 1 (the champion it replaced) and NEWS2 are scored on the same hours, and each score gets its own alert threshold, fixed on the same validation days for the same budget: at most about 2 alerts per nurse per 12-hour shift.

| Metric | Level | Version 3 | Version 1 | NEWS2 |
|---|---|---|---|---|
| **Escalations flagged in advance** (High in the 6 h before) | per escalation | **22 of 28 (79%)** | 20 of 28 (71%) | 17 of 28 (61%) |
| Median warning time, when flagged | per escalation | 4 hours | | 3 hours |
| Alerts followed by an escalation within 6 h | per alert | 4.0% (about 1 in 25) | | 4.1% (about 1 in 24) |
| Alert load on test | per nurse per shift | 2.15 | 2.10 | 1.79 |
| AUPRC (ranking) | per hour | 0.21 | 0.22 | 0.15 |
| AUROC (ranking) | per hour | **0.76** | 0.70 | 0.74 |
| Brier score (calibration; lower is better) | per hour | 0.013 | 0.013 | |

**Against version 1, on 300 resamples of the test block's patient stays:** version 3 flagged at least as many escalations in 89% of them, had the higher AUROC in 97%, and the higher AUPRC in 52%.

**Reading it:** within the same alert budget, version 3 alerted in advance for **2 more of the 28** deteriorating patients than version 1 and **5 more** than NEWS2, for about the same number of alerts per nurse. Its overall ranking (AUROC) is clearly better than version 1's under the new protocol. Its very top of the list (AUPRC) and its calibration (Brier) are about the same as version 1's: the gain is in the middle of the ranking, where protocol v2.0 now escalates patients earlier, with milder vital signs. Most alerts are still not followed by an escalation within 6 hours, for either score: escalation is rare.

**Version 1, for comparison over time** (card v1.0): on its own test block (1 to 3 October, before protocol v2.0, 26 escalations) it flagged 23 of 26 against NEWS2's 19. Those were different days, before protocol v2.0, so the two tables can't be compared directly.

**Rehearsal.** Before Databricks, the same code was trained offline on the same simulation (labels from all outcomes rather than recorded ones, so the test block held 28 escalations): 25 of 28 flagged vs NEWS2 21 of 28, a difference of +14 points with a 95% interval of +3 to +28 (bootstrap over hospital stays). The test blocks are small, so every interval is wide. The public demo app shows that rehearsal model.

## 5. Performance by group (fairness)

Measured on the offline rehearsal (section 4), with one shared alert threshold for everyone. **Not yet repeated on version 3**; it is part of monitoring.

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

On the offline rehearsal model, grouped SHAP importance, largest first: patient background (age, long-term conditions), blood pressure, oxygen saturation, breathing rate, heart rate. Oxygen therapy and confusion contribute little. This matches the hidden rule used by the synthetic data generator, which is only checkable because the data is synthetic. **Not yet re-measured for version 3**: a tree model can weigh features differently. The top reasons shown for each patient are computed from version 3 itself.

## 7. Known limitations

- **Synthetic data.** Simulated deteriorations are smoother than real ones, so real-world gains over NEWS2 would likely be smaller.
- **Small test block.** 28 escalations: the comparison with version 1 rests on resampling, and a different 2 days could shift every figure by several points.
- **Uneven alert load by age.** Wards with very old patients carry most of the alerts, mostly false alarms; younger patients rarely alert (section 5). Clinicians should not be reassured by a Low band in a young patient who looks unwell.
- **Only six long-term conditions.** The model knows COPD, heart failure, diabetes, chronic kidney disease, hypertension and atrial fibrillation (plus beta-blockers), and `n_conditions` counts only these. Conditions that matter in real wards, such as ischaemic heart disease, anaemia, dementia or alcohol and drug dependence, are not represented, because the synthetic data gives them no effect.
- **Sudden events.** Escalations without a gradual change in vital signs beforehand cannot be predicted.
- **Measurement faults.** The model trusts the monitor. Faulty sensors (e.g. FSN-2026-10, SpO2 under-reading at Northshore) produce wrong risks. Hours of known faults are left out of training and of the drift checks' reference, not of scoring.
- **Protocol dependence.** Version 3 was retrained after protocol v2.0 (2026-10-06) and chosen and tested under it. Version 1 had learned the old protocol: by 13 October the drift check put its ranking (AUPRC) at least 21% and its calibration (Brier) at least 48% worse than at validation, which is why it was replaced. Most of version 3's training days are still from before the change; another change in escalation practice would need the same retrain.
- **Alert-load drift.** On the test block version 3 raised 2.15 alerts per nurse per shift against a budget of 2 (fixed on validation, where it raised 1.97). The alert load needs no outcomes, so it is monitored continuously.

## 8. Monitoring and updates

Checked after each new batch of data for feature drift (PSI), prediction drift (Jensen-Shannon), alert load, data quality, and performance and calibration once outcomes are known (about 12 hours later). Retraining and promotion each need a human approval on GitHub. See the *Drift Incident Runbook*, HD-OPS-002.

**This card is reissued for every model version promoted to production** (the version it describes is in its header). A production release is refused while the card describes another version than the one in use.
