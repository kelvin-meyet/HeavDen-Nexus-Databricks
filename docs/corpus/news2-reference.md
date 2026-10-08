---
doc_id: HD-CLIN-002
title: NEWS2 Reference Card
version: "1.2"
status: current
effective_date: 2026-03-01
owner: HeavDen Network Clinical Governance Committee
audience: ward nurses, ward doctors, data and ML teams
---

# NEWS2 Reference Card

> **Fictional document.** Written for the HeavDen-Nexus portfolio project, which uses fully synthetic patients. The scoring tables follow the published NEWS2 (Royal College of Physicians, 2017), but this card is not clinical guidance.

## 1. What NEWS2 is

The **National Early Warning Score 2** turns seven routine observations into a single number. Each observation earns 0-3 points depending on how far it is from normal, and the points are added up. A higher total means a higher risk of serious deterioration. NEWS2 is the clinical baseline that the deterioration risk model must beat.

The seven parameters are: respiration rate, oxygen saturation (SpO2), use of supplemental oxygen, systolic blood pressure, pulse (heart rate), level of consciousness, and temperature.

## 2. Scoring table

| Parameter | 3 | 2 | 1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| Respiration rate (per min) | ≤8 | | 9-11 | 12-20 | | 21-24 | ≥25 |
| SpO2 scale 1 (%) | ≤91 | 92-93 | 94-95 | ≥96 | | | |
| Air or oxygen | | oxygen | | air | | | |
| Systolic BP (mmHg) | ≤90 | 91-100 | 101-110 | 111-219 | | | ≥220 |
| Pulse (per min) | ≤40 | | 41-50 | 51-90 | 91-110 | 111-130 | ≥131 |
| Consciousness | | | | Alert | | | new Confusion, Voice, Pain, Unresponsive |
| Temperature (°C) | ≤35.0 | | 35.1-36.0 | 36.1-38.0 | 38.1-39.0 | ≥39.1 | |

## 3. SpO2 scale 2 (patients with hypercapnic respiratory failure, e.g. COPD)

Some patients with chronic lung disease, most often **COPD**, normally run a lower oxygen saturation, and too much oxygen can be harmful to them. A doctor may prescribe a target of **88-92%** and record that the patient uses **scale 2**.

| SpO2 scale 2 (%) | Points |
|---|---|
| ≤83 | 3 |
| 84-85 | 2 |
| 86-87 | 1 |
| 88-92, or ≥93 on room air | 0 |
| 93-94 on oxygen | 1 |
| 95-96 on oxygen | 2 |
| ≥97 on oxygen | 3 |

Use scale 2 only when it has been prescribed. Everyone else uses scale 1. At HeavDen, scale 2 is applied automatically to patients with a recorded COPD diagnosis.

## 4. Consciousness: ACVPU

Consciousness is assessed with **ACVPU**: **A**lert, new **C**onfusion, responds to **V**oice, responds to **P**ain, **U**nresponsive. Alert scores 0. Anything else scores 3, including *new* confusion, which is often an early sign of sepsis.

## 5. Oxygen scores even when SpO2 looks fine

A patient on supplemental oxygen scores **2 points** for the oxygen alone. Oxygen raises SpO2 and can hide a worsening breathing problem, so the extra points keep these patients visible. Ward oxygen targets are in the *Site and Unit Handbook*.

## 6. Clinical response thresholds

The response to each score is set by the local escalation protocol (HD-CLIN-001). In outline, a total of 5 or more, or 3 in any single parameter, needs urgent review, and 7 or more needs an emergency response. Check the **current** version of the protocol, because the thresholds changed on 2026-11-09.

## 7. How NEWS2 is calculated in the data platform

- Respiration rate, SpO2, blood pressure, pulse and temperature come from the wearable vital-signs monitor (hourly summaries of 5-minute readings).
- Consciousness (ACVPU) and oxygen come from the **nurse observations** charted in the patient record, using the most recent observation before the scoring time.
- If no nurse observation exists yet, the patient is scored as alert and on room air.
- The columns are `news2_resp_rate`, `news2_spo2`, `news2_oxygen`, `news2_sbp`, `news2_heart_rate`, `news2_consciousness`, `news2_temp` and `news2_total` in `gold.patient_hour_features` (see the *Gold Data Dictionary*).
