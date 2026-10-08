---
doc_id: HD-OPS-002
title: Drift Incident Runbook
version: "1.0"
status: current
effective_date: 2026-11-01
owner: HeavDen ML Platform Team
audience: ML platform on-call, data engineering, clinical safety officer
---

# Drift Incident Runbook

> **Fictional document.** Written for the HeavDen-Nexus portfolio project, which uses fully synthetic data.

## 1. Purpose

The deterioration risk model was trained on past data. When the data or the world changes, its predictions can quietly get worse. This runbook says how a drift alert is raised, how to diagnose it, and what the correct response is for each kind of drift. **Retraining is not always the right answer.**

## 2. How an alert is raised

1. The `drift_check` task runs at the end of every scoring job in production.
2. It compares recent data with the champion model's training data, and recent predictions with outcomes once their 6-hour label window has closed.
3. If a threshold below is breached, Databricks sends an **email alert** with a summary and a link to the *Model and Drift* dashboard, and calls GitHub to start the `retrain.yml` workflow.
4. `retrain.yml` **waits for a human approval** in the GitHub `retrain-approval` environment. Nothing is retrained automatically.

## 3. Thresholds

| Signal | Warn | Act |
|---|---|---|
| PSI on a key feature (any slice, at least 500 samples) | > 0.10 | > 0.25 |
| Jensen-Shannon distance on predictions | > 0.05 | > 0.10 |
| AUPRC drop against the champion's validation AUPRC | > 10% relative | > 20% relative |
| Brier score (calibration) worsening | > 10% | > 20% |
| Quarantine or null rate in silver vitals | > 2% | > 5% |
| Alert rate against the 3.3% budget | > 4.5% | > 6% |

**Why PSI and not a KS test?** With thousands of readings a KS test flags tiny, harmless differences as "significant". PSI measures *how much* a distribution moved, and the minimum of 500 samples stops small slices from raising false alarms.

**Why the alert rate?** It needs no outcomes, so it is known immediately, while AUPRC must wait 6 hours for labels.

## 4. Triage checklist

1. Open the *Model and Drift* dashboard. Which signal fired: features, predictions, performance, or data quality?
2. Is it **one site** or **all sites**? One site points to devices or local practice. All sites points to patients or protocol.
3. Which **features** moved? Only SpO2? All respiratory features? None (only outcomes)?
4. Check the pipeline event log and `_rescued_data` for schema problems.
5. Check the document library for recent **field safety notices** and **protocol changes**.
6. Decide using section 5, record the decision in `ml.ops_audit`, and approve or reject the retrain request on GitHub with a comment.

## 5. Scenarios and correct responses

### 5.1 SpO2 sensor bias (data drift, gradual)
- **Symptoms:** PSI on `spo2_*` features rises at **one site** (Site B), growing day by day. Other vitals unchanged. Alert rate at that site rises. Calibration worsens there.
- **Cause:** a device fault (see device manual, FSN-2026-11): SpO2 reads up to 3 points low.
- **Correct response:** **do not retrain.** The patients haven't changed; the measurements are wrong. Retraining would teach the model to accept the faulty readings. Ask Clinical Engineering to fix the devices, add a temporary note on the ward board, and **reject** the retrain request with a reference to the notice. Consider excluding affected readings from future training data.

### 5.2 Respiratory outbreak (abrupt population drift)
- **Symptoms:** at all sites, a sudden rise in respiratory admissions. PSI rises on breathing rate, SpO2, temperature and oxygen use. Prediction distribution shifts up.
- **Cause:** a new mix of patients (e.g. a pneumonia outbreak), not a fault.
- **Correct response:** wait for labelled outcomes (at least 6 hours, ideally a day). If AUPRC and calibration on the new patients hold up, **no retrain** is needed: the model is seeing sicker patients and correctly scoring them higher. If performance drops, **approve a retrain** that includes the new cohort, and check the slice gates.

### 5.3 Protocol change (concept drift)
- **Symptoms:** feature distributions look **normal**, but more patients are escalated than predicted. AUPRC may hold while **calibration** gets worse (predicted risks too low). Starts on the same day at all sites.
- **Cause:** the escalation protocol changed (HD-CLIN-001 v2.0, effective 2026-11-09). The same vital signs now lead to escalation more often.
- **Correct response:** this is the one case where **retraining is required**: the meaning of the label has changed. Approve the retrain once enough post-change outcomes exist (at least 48 hours), and train only on data from after the change, or weight it heavily. Check the alert budget with the wards, because escalations will rise.

### 5.4 Firmware schema change (schema drift)
- **Symptoms:** a new column `etco2` appears in `_rescued_data`; `motion` becomes null; expectations on `motion` fail or rows are dropped; the quarantine rate rises.
- **Cause:** firmware 3.2.0 rollout (see device manual, section 6).
- **Correct response:** a **data engineering** fix, not a model fix. Update the silver schema and expectations, make `motion` optional, and decide whether `etco2` should become a feature in a later model version. **Reject** any retrain request until the pipeline is fixed.

## 6. Promotion after a retrain

A retrained model is registered as `@challenger`. It is promoted only if it beats the current `@champion` on a recent labelled window, its calibration is acceptable, no large slice gets more than 5% worse, and it still beats NEWS2. Promotion needs a second approval in the GitHub `production-approval` environment. Rollback is the `rollback.yml` workflow, which points `@champion` back at the previous version.

## 7. Communication

For any incident lasting more than one shift, inform the clinical safety officer and the nurse leads at the affected sites, and add a banner to the ward board.
