---
doc_id: HD-DEV-001
title: VitalBand VB-200 Wearable Monitor, User and Integration Manual
version: "3.2"
status: current
effective_date: 2026-11-01
owner: HeavDen Clinical Engineering
audience: ward nurses, clinical engineering, data engineering
---

# VitalBand VB-200 Wearable Monitor: User and Integration Manual

> **Fictional device and document.** Written for the HeavDen-Nexus portfolio project. The VitalBand VB-200 does not exist.

## 1. Overview

The **VitalBand VB-200** is a wireless wearable monitor used on all general, step-down and respiratory beds at the three HeavDen sites. Each device measures six vital signs every **5 minutes** and sends one message per reading over the hospital Wi-Fi.

| Vital | Field | Unit | Typical noise per reading |
|---|---|---|---|
| Heart rate | `heart_rate` | beats/min | ±1.5 |
| Respiration rate | `resp_rate` | breaths/min | ±0.8 |
| Oxygen saturation | `spo2` | % | ±0.5 |
| Temperature | `temp_c` | °C | ±0.05 |
| Systolic blood pressure | `sbp` | mmHg | ±3 |
| Diastolic blood pressure | `dbp` | mmHg | ±2 |

## 2. Devices are reused between patients

Each device has a fixed ID such as `DEV-A-0042` (site letter and number). When a patient is discharged the device is cleaned, the battery is swapped and it is given to the **next** patient. A device ID therefore does **not** identify a patient. To know who was wearing a device at a given moment, join on the **device assignment** records (`device_assignments` in Azure SQL), which hold the encounter, device, start time and end time of every assignment.

## 3. Message format

Firmware 3.1.x sends one JSON message per reading:

```json
{
  "device_id": "DEV-B-0107",
  "ts": "2026-11-03T14:35:00Z",
  "heart_rate": 92.4,
  "resp_rate": 21.0,
  "spo2": 94.1,
  "temp_c": 37.62,
  "sbp": 118.0,
  "dbp": 71.0,
  "motion": 0.12,
  "battery_pct": 63,
  "firmware": "3.1.4"
}
```

- `ts` is the measurement time in UTC, rounded to the 5-minute grid.
- `motion` is a 0-1 activity index from the accelerometer. Readings taken during high motion (above about 0.6) are less reliable.
- Messages are landed as JSON files in the ADLS landing zone and ingested by Auto Loader into `bronze.vitals_raw`.

## 4. Known measurement problems

Clinical engineering has characterised four kinds of fault. The data platform must expect all of them.

| Problem | What you see in the data | How often |
|---|---|---|
| **Motion artefact** | Short spikes, especially in heart rate and SpO2, while `motion` is high | several times a day per patient |
| **Dropped message** | A missing 5-minute reading | about 1.5% of messages |
| **Stuck sensor** | The same value repeated for 30-60 minutes | about 1% chance per device per hour |
| **Battery outage** | No messages for 30-90 minutes while a flat battery is swapped | batteries drain 1-2% per hour, so roughly every 2-4 days |

Silver-layer rules quarantine physically impossible values (for example heart rate outside 20-250 or SpO2 above 100) in `silver.vitals_quarantine`.

## 5. Alarms on the device

The VB-200 shows a local amber light for low battery (below 15%) and sensor-off. It does **not** raise clinical alarms. Clinical alerting is done centrally by NEWS2 and the deterioration risk score on the ward board.

## 6. Firmware release notes

### 3.1.4 (current baseline, all sites)
- Stability fixes for Wi-Fi roaming between wards.
- Message schema as in section 3.

### 3.2.0 (staged rollout from November 2026)
- **New field `etco2`**: end-tidal CO2 in mmHg from the optional nasal sampling line. It is null when no sampling line is attached.
- **Removed field `motion`**: the activity index moves to a separate diagnostics stream and is no longer sent with vital-sign messages.
- `firmware` reports `"3.2.0"`.
- **Integration impact:** this is a **schema change**. Pipelines that require `motion` will fail or drop rows; `etco2` will appear in `_rescued_data` until the schema is updated. See the *Drift Incident Runbook*, scenario "firmware schema change".

## 7. Field safety notice FSN-2026-11 (SpO2 under-reading)

**Affects:** a batch of VB-200 units at **HeavDen Northshore Medical Center (Site B)** after a sensor-module update in November 2026.

**Problem:** affected units gradually **under-read SpO2**, by about 1.5 percentage points per day, levelling off at about **3 points** low. Heart rate, respiration and the other vitals are not affected. The patients are not actually less well oxygenated.

**Consequences:**
- NEWS2 SpO2 points and the deterioration risk score are overstated for affected patients, so false alerts increase at Site B.
- Nurses may give supplemental oxygen that is not needed.

**Action:**
1. Clinical engineering recalibrates affected units (field fix 3.1.5).
2. Until fixed, check SpO2 with a hand-held oximeter before acting on a low reading at Site B.
3. Data and ML teams: this is a **measurement problem, not a change in patients**. Do **not** retrain the model on affected data. See the *Drift Incident Runbook*, scenario "SpO2 sensor bias".
