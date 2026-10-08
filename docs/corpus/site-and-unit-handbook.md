---
doc_id: HD-OPS-001
title: Site and Unit Handbook
version: "2.1"
status: current
effective_date: 2026-10-01
owner: HeavDen Network Operations
audience: all ward staff, analysts
---

# Site and Unit Handbook

> **Fictional organisation and document.** Written for the HeavDen-Nexus portfolio project. The HeavDen network, its hospitals, phone extensions and staff do not exist.

## 1. The HeavDen network

The HeavDen network has three acute hospitals in Massachusetts with **360 monitored adult ward beds** in total. Each site has three kinds of ward: **general** medical, **step-down** (higher-dependency patients, e.g. after ICU) and **respiratory**.

| Site ID | Hospital | City | Monitored beds | General | Step-down | Respiratory |
|---|---|---|---|---|---|---|
| SITE_A | HeavDen General Hospital | Boston | 160 | 88 | 40 | 32 |
| SITE_B | HeavDen Northshore Medical Center | Salem | 110 | 61 | 28 | 22 |
| SITE_C | HeavDen Valley Community Hospital | Worcester | 90 | 50 | 22 | 18 |

Unit IDs combine the site and ward type, e.g. `SITE_B-RESPIRATORY`. Typical occupancy is around 80-85% (about 300 patients across the network), and the median length of stay is about 4.5 days.

## 2. Rapid response team (RRT) contacts

| Site | RRT extension | Outreach nurse bleep | Cover |
|---|---|---|---|
| SITE_A General | 2222 | 4410 | 24/7, two teams |
| SITE_B Northshore | 3333 | 5520 | 24/7, one team |
| SITE_C Valley | 4444 | 6630 | 24/7, one team; ICU transfers go to SITE_A General |

Valley (SITE_C) has a small ICU without capacity for long stays. Patients needing prolonged intensive care are transferred to General (SITE_A).

## 3. Staffing and shifts

- Shifts are **12 hours**: day 07:00-19:00, night 19:00-07:00 (local time).
- Ward nurse-to-patient ratio is **1 nurse to 5 patients** on general and respiratory wards, and 1 to 4 on step-down.
- The **alert budget** for the deterioration risk score is set from this staffing: about **2 High alerts per nurse per 12-hour shift**, which is about 3.3% of patient-hours. More alerts than that cause alarm fatigue.

## 4. Oxygen targets

| Patient group | SpO2 target |
|---|---|
| Most adult patients | 92-96% |
| COPD and other patients at risk of CO2 retention, on NEWS2 scale 2 | 88-92% |

Nurses titrate oxygen in steps (2, 4, 6, then 10 litres/min) to keep patients in their target range, and wean it when saturation is well above target. Oxygen use is recorded at every nurse observation, because it adds 2 NEWS2 points.

## 5. Nurse observation rounds

Nurses chart a full set of observations, including consciousness (ACVPU) and current oxygen, at least every **4 hours**, every **2 hours** for a NEWS2 of 3-4, and **hourly** for a NEWS2 of 5 or more. Vital signs between rounds come from the VitalBand monitor (see HD-DEV-001).

## 6. Site differences worth knowing

- **General (SITE_A)** is the regional referral centre: older patients, more heart failure and step-down transfers.
- **Northshore (SITE_B)** received the first batch of VitalBand units with the November 2026 sensor-module update (see the device manual's field safety notice FSN-2026-11).
- **Valley (SITE_C)** is the smallest site, with the fewest escalations in absolute terms. Per-site metrics there are noisier.

## 7. Who to contact about the data platform

- Ward board, risk score or alert questions: the **ML platform team** (ticket queue "HeavDen-Nexus").
- Device faults: **Clinical Engineering**, extension 7100.
- Incorrect patient details: the site admissions office.
