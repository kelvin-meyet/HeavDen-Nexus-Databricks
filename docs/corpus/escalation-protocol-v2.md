---
doc_id: HD-CLIN-001
title: Deteriorating Patient Escalation Protocol
version: "2.0"
status: current
effective_date: 2026-11-09
supersedes: HD-CLIN-001 v1.0
owner: HeavDen Network Clinical Governance Committee
audience: ward nurses, ward doctors, rapid response team
---

# Deteriorating Patient Escalation Protocol (v2.0)

> **Fictional document.** Written for the HeavDen-Nexus portfolio project, which uses fully synthetic patients. It is not clinical guidance and must not be used to care for real patients.

## 1. What changed from v1.0

Version 2.0 takes effect on **2026-11-09** at all three sites. An audit of 2026 found that deteriorating patients were often escalated late, after several hours of worsening observations. Version 2.0 therefore asks staff to **escalate earlier**:

1. **Lower trigger for an urgent response.** A NEWS2 of **5 or more** (previously 7) now triggers a call to the rapid response team.
2. **Earlier action on a single red score.** A score of 3 in any single parameter now triggers a review by the outreach nurse within 30 minutes (previously the ward doctor within 60 minutes).
3. **Proactive review rounds.** The outreach nurse visits every ward each shift and reviews any patient with a NEWS2 of 1 or more whose observations are trending the wrong way, even if they look stable.
4. **High risk alerts get a senior review.** A High band on the deterioration risk score now triggers a review by the outreach nurse within 30 minutes.

**What this means for the data.** Patients are now escalated earlier in a deterioration, and some patients who would previously have stayed on the ward are escalated during proactive rounds. The *same* observations are therefore followed by escalation more often than before. For the deterioration risk model this is **concept drift**: the relationship between vital signs and the outcome has changed. See the *Drift Incident Runbook*, HD-OPS-002, scenario "protocol change".

## 2. Purpose and scope

As v1.0: adult inpatients on general, step-down and respiratory wards at HeavDen General, Northshore and Valley. Not intensive care, theatres or the emergency department.

## 3. Escalation by NEWS2 score

| NEWS2 | Observation frequency | Response |
|---|---|---|
| 0 | at least every 12 hours | routine care |
| 1-4 | at least every 4 hours | nurse in charge informed; included in the proactive review round if trending up |
| 3 in any single parameter | at least hourly | **outreach nurse review within 30 minutes** |
| 5-6 | at least hourly | **call the rapid response team** |
| 7 or more | continuous monitoring | **emergency**: rapid response team immediately, consider ICU referral |

## 4. Escalation by risk band

| Band | Response |
|---|---|
| Low | routine care |
| Medium | included in the proactive review round; nurse reviews the trend each observation round |
| High (alert) | **outreach nurse review within 30 minutes**; repeat full observations and NEWS2; call the RRT if NEWS2 is 5 or more or there is clinical concern |

The model's alert budget is unchanged: about 2 High alerts per nurse per 12-hour shift. If alerts consistently exceed that, report it to the ML platform team (see the *Drift Incident Runbook*).

## 5. How to call the rapid response team

Unchanged from v1.0: dial the site RRT extension (*Site and Unit Handbook*, HD-OPS-001), use **SBAR**, state the NEWS2 score, risk band and top reasons. The RRT aims to attend within **15 minutes**.

## 6. Documentation

As v1.0. Proactive-round escalations are recorded like any other escalation, with the reason "proactive review".

## 7. Review

First review 3 months after go-live (February 2027), with an audit of RRT call volumes and outcomes.
