---
doc_id: HD-CLIN-001
title: Deteriorating Patient Escalation Protocol
version: "1.0"
status: superseded
effective_date: 2026-03-01
superseded_by: HD-CLIN-001 v2.0 (effective 2026-11-09)
owner: HeavDen Network Clinical Governance Committee
audience: ward nurses, ward doctors, rapid response team
---

# Deteriorating Patient Escalation Protocol (v1.0)

> **Fictional document.** Written for the HeavDen-Nexus portfolio project, which uses fully synthetic patients. It is not clinical guidance and must not be used to care for real patients.

> **Superseded.** This version applied until 2026-11-08. From **2026-11-09** use version 2.0, which escalates earlier. See *Escalation Protocol v2.0, section 1 "What changed"*.

## 1. Purpose and scope

This protocol tells ward staff at all three HeavDen sites (General, Northshore, Valley) when and how to escalate care for an adult inpatient whose condition is getting worse. It applies to general, step-down and respiratory wards. It does not apply to intensive care, theatres or the emergency department.

Escalation means asking for a more senior or more specialised review: the ward doctor, the critical care outreach nurse, or the **rapid response team (RRT)**.

## 2. The two signals staff use

Staff use two complementary signals:

1. **NEWS2 score.** Calculated from every full set of observations (see the *NEWS2 Reference*, HD-CLIN-002). It is the primary trigger.
2. **Deterioration risk score.** The predicted probability that the patient will need escalation in the next 6 hours, shown on the ward board with a band (Low or High), its change over the last 6 hours and the main reasons. It is produced hourly by the `deterioration_risk` model (see the *Model Card*, HD-ML-001).

The risk score **supports** clinical judgement and NEWS2. It never replaces them. A patient who worries the nurse is escalated whatever the scores say.

## 3. Escalation by NEWS2 score

| NEWS2 | Observation frequency | Response |
|---|---|---|
| 0 | at least every 12 hours | routine care |
| 1-4 | at least every 4-6 hours | inform the nurse in charge, who decides whether more frequent monitoring is needed |
| 3 in any single parameter | at least hourly | urgent review by the ward doctor |
| 5-6 | at least hourly | **urgent** review by the ward doctor or outreach nurse within 60 minutes |
| 7 or more | continuous monitoring | **emergency**: call the rapid response team immediately |

## 4. Escalation by risk band

| Band | Response |
|---|---|
| Low | routine care; no action from the score alone. A risk that is rising over 6 hours is a prompt for clinical judgement at the next observation round, not an alert |
| High (alert) | nurse reviews the patient **within 60 minutes**, repeats a full set of observations and recalculates NEWS2; escalate if NEWS2 or clinical concern warrants it |

A High risk alert on its own does **not** trigger a rapid response call in v1.0. It triggers a bedside review.

## 5. How to call the rapid response team

- Dial the site RRT extension (see the *Site and Unit Handbook*, HD-OPS-001).
- Use **SBAR**: Situation, Background, Assessment, Recommendation.
- State the NEWS2 score, the risk band and the top reasons shown on the ward board.
- The RRT aims to attend within **15 minutes**.

## 6. Documentation

Every escalation is recorded in the patient record with the time, the NEWS2 score, the risk band, who was called and the outcome. These records become the **escalation events** the risk model learns from: a `rapid_response` call or an `icu_transfer` within 6 hours of a prediction counts as a positive outcome.

## 7. Review

The Clinical Governance Committee reviews this protocol at least yearly, or earlier if audit shows late escalations.
