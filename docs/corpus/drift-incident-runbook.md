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

1. The `drift_check` job runs in production after each new batch of data has been scored.
2. It compares the latest day of data and predictions with the data the champion was trained and calibrated on, and the latest two days of predictions with their outcomes once known (about 12 hours later: the 6-hour window, then 6 hours until escalations are recorded). It checks all sites together and each site on its own.
3. If any signal reaches the **act** level below, the workflow that ran the check opens a **retrain request** on GitHub (the `retrain.yml` workflow), which lists the breaches. A **warn** is only recorded in the monitoring tables. While a request is open (it may be held for days), later checks that reach **act** open no second one.
4. `retrain.yml` **waits for a human approval** in the GitHub `retrain-approval` environment, and GitHub **emails** the approver. Nothing is retrained automatically, and rejecting the request is a normal outcome (see section 5).

## 3. Thresholds

| Signal | Warn | Act |
|---|---|---|
| PSI on a key feature (any slice, at least 500 samples) | > 0.10 | > 0.25 |
| Jensen-Shannon distance on predictions | > 0.08 | > 0.12 |
| AUPRC drop against the champion's validation AUPRC | > 10% relative | > 20% relative |
| Brier score (calibration) worsening | > 10% | > 20% |
| Share of readings quarantined in silver | > 2% | > 5% |
| Change in a reading field's missing share since training | > 2 points | > 5 points |
| Alert load against the budget of 2 alerts per nurse per shift | > 2.5 | > 3 |

**Why PSI and not a KS test?** With thousands of readings a KS test flags tiny, harmless differences as "significant". PSI measures *how much* a distribution moved, and the minimum of 500 samples stops small slices from raising false alarms.

**Why the alert load?** It needs no outcomes, so it is known immediately, while AUPRC and escalations flagged must wait about 12 hours for labels.

**Why a margin for chance on AUPRC and Brier?** Two days hold only about 20 escalations, so on normal days the AUPRC of a two-day window swings by a third either way. The check therefore uses the change it can be 95% sure of (a bootstrap over patient stays), and needs at least 10 escalations in the window; with fewer it reports "not enough data". The prediction thresholds were set so that normal days stay below warn (they reached 0.063 at one site).

## 4. Triage checklist

1. Open the *Model and Drift* dashboard. Which signal fired: features, predictions, performance, or data quality?
2. Is it **one site** or **all sites**? One site points to devices or local practice. All sites points to patients or protocol.
3. Which **features** moved? Only SpO2? All respiratory features? None (only outcomes)?
4. Check the pipeline event log and `_rescued_data` for schema problems.
5. Check the document library for recent **field safety notices** and **protocol changes**.
6. Decide using section 5, and approve or reject the retrain request on GitHub **with a comment giving the reason**. The decision, the reviewer and the comment are recorded in `ml.ops_audit`: an approved retrain, promotion or rollback at once, a rejection at the next production run (a rejected request is given no access to production, so it cannot record itself).

## 5. Scenarios and correct responses

### 5.1 SpO2 sensor bias (data drift, gradual)
- **Symptoms:** PSI on `spo2_*` features rises at **one site** (Site B), growing day by day. Other vitals unchanged. Prediction distribution and alert load at that site rise. Performance on outcomes changes little, because the patients haven't changed.
- **Cause:** a device fault (see device manual, FSN-2026-11): SpO2 reads up to 3 points low.
- **Correct response:** **do not retrain.** The patients haven't changed; the measurements are wrong. Retraining would teach the model to accept the faulty readings. Ask Clinical Engineering to fix the devices, add a temporary note on the ward board, and **reject** the retrain request with a reference to the notice. Consider excluding affected readings from future training data.

### 5.2 Respiratory outbreak (abrupt population drift)
- **Symptoms:** at all sites, a sudden rise in respiratory admissions. PSI rises on breathing rate, SpO2, temperature and oxygen use. Prediction distribution shifts up.
- **Cause:** a new mix of patients (e.g. a pneumonia outbreak), not a fault.
- **Correct response:** wait for labelled outcomes (at least 6 hours, ideally a day). If AUPRC and calibration on the new patients hold up, **no retrain** is needed: the model is seeing sicker patients and correctly scoring them higher. If performance drops, **approve a retrain** that includes the new cohort, and check the slice gates.

### 5.3 Protocol change (concept drift)
- **Symptoms:** feature distributions look **normal**, but more patients are escalated than predicted. **AUPRC drops and calibration gets worse** (predicted risks too low), once the outcomes are known, about 12 hours after the change. Starts on the same day at all sites.
- **Cause:** the escalation protocol changed (HD-CLIN-001 v2.0, effective 2026-11-09). The same vital signs now lead to escalation more often.
- **Correct response:** this is the one case where **retraining is required**: the meaning of the label has changed. Approve the retrain only once enough post-change outcomes exist: **about a week**. With one or two days there is nothing to validate a new model on. Retraining uses the whole history, with the most recent days kept for calibration, the alert threshold and the comparison with the current champion; a model trained only on post-change data was usually worse than one trained on everything. If the new model does not pass the gates, the champion stays and the retrain is repeated when more outcomes exist. Check the alert budget with the wards, because escalations will rise.

### 5.4 Firmware schema change (schema drift)
- **Symptoms:** a new column `etco2` appears in `_rescued_data`; `motion` becomes null; expectations on `motion` fail or rows are dropped; the quarantine rate rises.
- **Cause:** firmware 3.2.0 rollout (see device manual, section 6).
- **Correct response:** a **data engineering** fix, not a model fix. Update the silver schema and expectations, make `motion` optional, and decide whether `etco2` should become a feature in a later model version. **Reject** any retrain request until the pipeline is fixed.

## 6. Promotion after a retrain

A retrained model is registered as a new version. It becomes `@challenger` only if it beats NEWS2 on validation and beats the current `@champion` on the most recent days, with both models held to the same alert budget. A few days hold too few escalations to trust a single comparison, so both are scored on the same resamples of patient stays, and the new model must flag at least as many escalations in at least 75% of them, have the higher AUROC in at least 75%, not be clearly worse on AUPRC, and have a Brier score no more than 10% worse. AUPRC alone is not the gate: after a protocol change it barely moves, because it is set by the highest-ranked patients, while the retrained model's gain is in the escalations further down the ranking.

**Promotion.** A model that passes opens a promotion request (the `promote.yml` workflow), which needs a second approval, in the GitHub `production-approval` environment. Approving points `@champion` at the new version. From the next scoring run, new patient-hours are scored by it, at its own alert threshold; hours already scored keep the version that scored them. The promotion is refused for a version that did not pass its gates, or that was compared with a champion other than the one in use.

**Rollback.** The `rollback.yml` workflow, started by hand with a reason and approved in the same environment, points `@champion` back at the previous champion, which is kept as `@previous_champion`. Running it again undoes the rollback.

## 7. Communication

For any incident lasting more than one shift, inform the clinical safety officer and the nurse leads at the affected sites, and add a banner to the ward board.
