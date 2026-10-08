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
- a **band** (Low, Medium, High), and
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
| Selection | best validation AUPRC among candidates that beat NEWS2 on AUPRC and on precision at the alert budget |

## 3. Risk bands

Escalation within 6 hours is rare: about **0.9%** of patient-hours. Calibrated probabilities are therefore small numbers, and the bands are set from the **alert budget** rather than round percentages:

| Band | Definition (fixed on validation) | Current risk cut-off | What it means |
|---|---|---|---|
| High (alert) | top 3.3% of patient-hours | risk ≥ about 1.7% | escalation about **14 times** more likely than average; roughly 1 in 8 High hours is followed by escalation |
| Medium | next 6.7% (up to the top 10%) | risk about 0.7-1.7% | above-average risk; watch the trend |
| Low | the remaining 90% | risk below about 0.7% | routine monitoring |

The 3.3% comes from the staffing model: 2 alerts per nurse per 12-hour shift with 1 nurse per 5 patients. Responses to each band are set by the escalation protocol (HD-CLIN-001).

## 4. Performance (test block, 2 days, 28 escalations)

| Metric | Champion | NEWS2 | 95% interval of the difference |
|---|---|---|---|
| AUPRC (main metric) | 0.40 | 0.22 | +0.08 to +0.26 |
| AUROC | 0.79 | 0.74 | -0.01 to +0.13 |
| Precision at the same number of alerts | 0.15 | 0.11 | +0.01 to +0.07 |
| Recall at the same number of alerts | 0.54 | 0.41 | +0.05 to +0.22 |
| Escalations caught (of 28) | 24 | 21 | |
| Caught at least 3 hours ahead | 17 | 12 | |

Intervals come from a bootstrap that resamples whole hospital stays. The champion's own AUPRC interval is wide (about 0.26-0.54), because the test block holds few escalations.

## 5. Performance by group (fairness)

One shared alert threshold for everyone.

| Group | Finding |
|---|---|
| Sites A, B, C | similar (AUPRC 0.38-0.42, recall 0.51-0.62) |
| Sex | similar (AUPRC 0.41 F, 0.39 M) |
| Age 70+ | most alerts go to this group; recall about 0.65-0.70; for 85+ most alerts are false alarms (precision about 0.09) |
| Age under 70 | **few alerts and lower recall (about 0.25-0.35)**: deterioration in younger patients is more likely to be missed |
| COPD | equal recall to NEWS2 with far fewer alerts (6.5% vs 22% of COPD patient-hours) |
| Ward type | step-down units have the lowest precision (about 0.07) |

Slices with fewer than about 20 positive hours (e.g. age 18-49) are too small to draw conclusions from.

## 6. What the model relies on

Grouped SHAP importance, largest first: patient background (age, long-term conditions), blood pressure, oxygen saturation, breathing rate, heart rate. Oxygen therapy and confusion contribute little. This matches the hidden rule used by the synthetic data generator, which is only checkable because the data is synthetic.

## 7. Known limitations

- **Synthetic data.** Simulated deteriorations are smoother than real ones, so real-world gains over NEWS2 would likely be smaller.
- **Age gap.** Younger patients' deteriorations are caught less often (section 5). Clinicians should not be reassured by a Low band in a young patient who looks unwell.
- **Sudden events.** Escalations without a gradual change in vital signs beforehand cannot be predicted.
- **Measurement faults.** The model trusts the monitor. Faulty sensors (e.g. FSN-2026-11, SpO2 under-reading at Site B) produce wrong risks.
- **Protocol dependence.** The model learned when staff escalated under protocol v1.0. After protocol v2.0 (2026-11-09) it under-predicts escalation until retrained (see the *Drift Incident Runbook*, scenario 5.3).
- **Alert-rate drift.** On the test block the alert rate was 3.9% against the 3.3% budget. The alert rate is monitored.

## 8. Monitoring and updates

Monitored hourly for feature drift (PSI), prediction drift (Jensen-Shannon), performance once outcomes arrive (6-hour delay), calibration and alert rate. Retraining and promotion each need a human approval on GitHub. See the *Drift Incident Runbook*, HD-OPS-002.
