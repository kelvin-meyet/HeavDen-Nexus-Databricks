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
| Other candidate | LightGBM with isotonic calibration (lost on validation AUPRC: 0.32 vs 0.38) |
| Inputs | 69 hourly features (72 model inputs once ward type and sex are one-hot encoded): vital-sign summaries over 1, 3 and 6 hours (mean, median, min, max, 3-hour trend), monitoring gaps, NEWS2 components, nurse observations (oxygen, new confusion), age, sex, long-term conditions, ward type, time since admission |
| Output | probability of escalation within 6 hours |
| Training data | 9 days of synthetic patient-hours, ~2,000 Synthea patients, 3 sites (see the *Data Card*, HD-ML-002) |
| Selection | best validation AUPRC among candidates that beat NEWS2 on AUPRC and flag at least as many escalations within the alert budget |

## 3. Risk bands

Escalation within 6 hours is rare: about **0.9%** of patient-hours. Calibrated probabilities are therefore small numbers, and the bands are set from the **alert budget** rather than round percentages.

An **alert** fires when a patient **enters** the High band. The same patient does not alert again for 6 hours, even if they leave and re-enter High. The High cut-off is the lowest one that keeps alerts within about **2 per nurse per 12-hour shift** (1 nurse per 5 patients), fixed on validation.

| Band | Definition (fixed on validation) | Current risk cut-off | What it means |
|---|---|---|---|
| High (alert) | the alert threshold; about 8% of patient-hours | risk ≥ about 1.7% | per hour: escalation about **8 times** more likely than average (about 1 in 14 High hours is followed by one) |
| Low | everything else, about 92% | risk below about 1.7% | routine monitoring; the ward board still shows the risk and its 6-hour change |

**Why there is no Medium band.** Several definitions of a middle band were tested: the top 10% of risk, risk doubling within 6 hours, recently High, and NEWS2 of 3 or more. None of them was followed by an escalation more often than an average patient-hour (0.5 to 1.1 times the average rate). The model's signal is concentrated above the alert threshold: deteriorating patients cross it. A Medium band would add colour without information, so the ward board shows each patient's risk and its 6-hour change instead.

Responses to each band are set by the escalation protocol (HD-CLIN-001).

## 4. Performance (test block, 2 days, 28 escalations)

Each score uses its own alert threshold, fixed on validation for at most 2 alerts per nurse per shift.

| Metric | Level | Champion | NEWS2 | 95% interval of the difference |
|---|---|---|---|---|
| **Escalations flagged in advance** (High in the 6 h before) | per escalation | **25 of 28 (89%)** | 21 of 28 (75%) | +3 to +28 points |
| Flagged at least 3 hours ahead | per escalation | 19 | 12 | |
| Alerts followed by an escalation within 6 h | per alert | 2.9% (about 1 in 34) | 3.1% | -1 to +1 points |
| Alert load on test | per nurse per shift | 2.1 | 1.8 | |
| AUPRC (ranking) | per hour | 0.40 | 0.22 | +0.08 to +0.26 |
| AUROC (ranking) | per hour | 0.79 | 0.74 | -0.01 to +0.13 |

**Reading it:** within the same alert budget, the model flags **14 percentage points more** deteriorating patients in advance (89% vs 75%: 4 more of the 28); both scores are followed by an escalation after about 1 in 30 alerts. Most alerts are not followed by an escalation within 6 hours, for either score. NEWS2's whole-number scores can't use the budget exactly, so it raises slightly fewer alerts.

Intervals come from a bootstrap that resamples whole hospital stays. The test block holds only 28 escalations, so every interval is wide.

## 5. Performance by group (fairness)

One shared alert threshold for everyone.

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
- **Alert-load drift.** On the test block the model raised 2.1 alerts per nurse per shift against a budget of 2. The alert load needs no outcomes, so it is monitored continuously.

## 8. Monitoring and updates

Monitored hourly for feature drift (PSI), prediction drift (Jensen-Shannon), performance once outcomes arrive (6-hour delay), calibration and alert load. Retraining and promotion each need a human approval on GitHub. See the *Drift Incident Runbook*, HD-OPS-002.
