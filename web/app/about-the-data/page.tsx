import type { Metadata } from "next";
import Link from "next/link";

import styles from "./about.module.css";

export const metadata: Metadata = {
  title: "The hospital and its data · HeavDen Nexus",
  description: "How the fictional HeavDen hospitals run, and how their synthetic data is generated.",
};

// Figures below come from the generator (generator/hospital.py, activity.py, vitals.py, care.py,
// truth.py, synthea.py). Update them here when the simulation changes.

const SITES = [
  { name: "HeavDen General Hospital", city: "Boston", beds: 160 },
  { name: "HeavDen Northshore Medical Center", city: "Salem", beds: 110 },
  { name: "HeavDen Valley Community Hospital", city: "Worcester", beds: 90 },
];

const UNITS = [
  { name: "General medicine", share: "55%", who: "Most admissions" },
  { name: "Step-down", share: "25%", who: "More often heart failure or atrial fibrillation" },
  { name: "Respiratory", share: "20%", who: "More often COPD" },
];

const CONDITIONS = [
  {
    name: "COPD (chronic lung disease)",
    effect:
      "Normal oxygen saturation around 91% instead of 97%, faster breathing. Scored on the NEWS2 scale for COPD; oxygen target 88–92%. Deteriorates more often through breathing problems.",
  },
  {
    name: "Heart failure",
    effect: "Heart rate about 8 beats higher. Deteriorates more often in a cardiac way; higher risk overall.",
  },
  {
    name: "Atrial fibrillation",
    effect: "Heart rate about 10 beats higher, and more irregular.",
  },
  { name: "High blood pressure", effect: "Systolic about 14 higher, diastolic about 8 higher." },
  { name: "Diabetes, chronic kidney disease", effect: "Systolic about 5 higher." },
  {
    name: "On a beta-blocker (medicine)",
    effect: "Heart rate about 10 beats lower, which can hide a rising heart rate.",
  },
];

const FAULTS = [
  "Sensor noise and motion artefacts when the patient moves",
  "About 1 message in 70 lost on the way",
  "Stuck sensors: about once every 100 patient-hours, a sensor repeats its last value for 30 to 60 minutes",
  "Batteries drain 1–2% an hour; a flat one takes the monitor offline for 30 to 90 minutes",
];

const DRIFT = [
  {
    name: "SpO2 sensor fault",
    body: "A firmware bug makes oxygen readings at one site read low. Patients are no sicker: the right response is to fix the sensors, not to retrain the model.",
  },
  {
    name: "Respiratory outbreak",
    body: "Suddenly most admissions have pneumonia: a new kind of patient the model has seen little of.",
  },
  {
    name: "Protocol change",
    body: "Staff start escalating earlier. The vitals look the same but the outcomes change, so the model's idea of risk goes out of date.",
  },
  {
    name: "Firmware update",
    body: "Monitors start sending a new measurement (exhaled CO2) and stop sending motion. The data pipeline has to cope with the new shape.",
  },
];

const LEFT_OUT = [
  "Conditions that matter in real hospitals but have no effect here: heart disease and previous heart attacks, anaemia, dementia, alcohol or drug dependence.",
  "Social circumstances that patient records list among conditions (work, education, housing): never used, because they have no clinical meaning here and would raise fairness problems.",
  "Treatments other than oxygen. Who deteriorates is decided first, so care doesn't change outcomes.",
  "Deteriorations are smoother and more regular than real ones, so real-world results would likely be weaker.",
];

export default function AboutTheDataPage() {
  return (
    <div className="page">
      <header className="pageHead">
        <h1>The hospital and its data</h1>
        <p className="lede">
          Everything in this app comes from a simulation. This page describes the fictional hospital network, what
          its staff need, and how each piece of data is made, so you can read the charts and numbers with the right
          expectations.
        </p>
      </header>

      <nav aria-label="On this page" className={styles.toc}>
        <a href="#hospital">The hospital</a>
        <a href="#need">What the wards need</a>
        <a href="#patients">The patients</a>
        <a href="#monitors">Monitors and nurses</a>
        <a href="#deterioration">Who deteriorates</a>
        <a href="#records">Where the data goes</a>
        <a href="#drift">Planned changes</a>
        <a href="#limits">What it leaves out</a>
      </nav>

      <section className="section" id="hospital" aria-labelledby="hospital-title">
        <h2 id="hospital-title">The hospital</h2>
        <p className={styles.prose}>
          HeavDen Health is an invented network of three hospitals in Massachusetts with 360 monitored ward beds. Each
          hospital has the same three units, sized as a share of its beds.
        </p>
        <div className={styles.pair}>
          <div className="tableWrap">
            <table className="table">
              <caption className="visually-hidden">Hospitals</caption>
              <thead>
                <tr>
                  <th scope="col">Hospital</th>
                  <th scope="col">City</th>
                  <th scope="col" className="num">
                    Beds
                  </th>
                </tr>
              </thead>
              <tbody>
                {SITES.map((s) => (
                  <tr key={s.name}>
                    <td>{s.name}</td>
                    <td>{s.city}</td>
                    <td className="num">{s.beds}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="tableWrap">
            <table className="table">
              <caption className="visually-hidden">Units in every hospital</caption>
              <thead>
                <tr>
                  <th scope="col">Unit</th>
                  <th scope="col" className="num">
                    Beds
                  </th>
                  <th scope="col">Patients</th>
                </tr>
              </thead>
              <tbody>
                {UNITS.map((u) => (
                  <tr key={u.name}>
                    <td>{u.name}</td>
                    <td className="num">{u.share}</td>
                    <td>{u.who}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p className={styles.prose}>
          The wards run about 83% full (roughly 300 of 360 beds). A typical stay lasts about 4 days; when a patient goes
          home, the bed is cleaned and filled again within hours to two days. About 1 stay in 7 moves once to another
          unit. Every patient wears a monitor from admission to discharge, and the monitor then goes to the next
          patient.
        </p>
      </section>

      <section className="section" id="need" aria-labelledby="need-title">
        <h2 id="need-title">What the wards need</h2>
        <p className={styles.prose}>
          On a general ward, a patient getting worse often shows it in their vital signs hours before a crisis. Nurses
          today use <strong>NEWS2</strong>, a points score from seven bedside measurements. The question this
          project answers is: <em>which patients are likely to need urgent help (a rapid-response call or a move to
          intensive care) in the next 6 hours?</em> Alerts have a cost: a nurse can act on about 2 per 12-hour shift
          before they start being ignored, so the model has to beat NEWS2 within that budget.
        </p>
      </section>

      <section className="section" id="patients" aria-labelledby="patients-title">
        <h2 id="patients-title">The patients</h2>
        <p className={styles.prose}>
          About 2,000 adults (18 to 95) are created by <strong>Synthea</strong>, an open-source tool that writes
          realistic lifetime medical histories: diagnoses, medicines, birth dates. They're invented people. Hospital
          inpatients are older and sicker than the general population, so the people admitted are drawn with extra
          weight on age and long-term illness.
        </p>
        <p className={styles.prose}>
          Their records list over 200 different conditions. The simulation gives an effect to six of them, plus
          beta-blockers, chosen because each changes bedside vital signs or the risk of deteriorating in a
          well-known way, each is common enough to learn from, and each is a classic trap for early-warning scores (a
          reading that is normal for one patient is alarming for another):
        </p>
        <div className="tableWrap" style={{ maxWidth: 820 }}>
          <table className="table">
            <caption className="visually-hidden">Conditions that shape the simulation</caption>
            <thead>
              <tr>
                <th scope="col">Condition</th>
                <th scope="col">What it does in the simulation</th>
              </tr>
            </thead>
            <tbody>
              {CONDITIONS.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>{c.effect}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${styles.prose} muted small`}>
          Age matters too: blood pressure rises and heart rate falls after 50, and risk grows with age. Every other
          condition has no effect here, so the model doesn't use it: it would only add noise. The model&rsquo;s
          &ldquo;number of conditions&rdquo; counts these six.
        </p>
      </section>

      <section className="section" id="monitors" aria-labelledby="monitors-title">
        <h2 id="monitors-title">Monitors and nurses</h2>
        <p className={styles.prose}>
          Each monitor sends six vital signs every 5 minutes: heart rate, breathing rate, oxygen saturation (SpO2),
          temperature and blood pressure (systolic and diastolic). They are made in two layers, kept apart on purpose:
        </p>
        <ul className={styles.list}>
          <li>
            <strong>The body:</strong> each patient&rsquo;s own normal values, drifting slowly, with a day and night
            rhythm, and faster heart and breathing rates during a fever.
          </li>
          <li>
            <strong>The device:</strong> what the monitor actually reports, with real-world faults:
            <ul>
              {FAULTS.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </li>
        </ul>
        <p className={styles.prose}>
          A monitor knows its own ID, never the patient. Readings are matched to a patient later, from the record of
          which monitor was on whom and when.
        </p>
        <p className={styles.prose}>
          Two parts of NEWS2 can&rsquo;t come from a monitor, so nurses add them. They chart{" "}
          <strong>consciousness</strong> (alert, new confusion, or responding only to voice or pain) and{" "}
          <strong>oxygen</strong> every 4 hours, every 2 hours when NEWS2 is 3–4, and hourly when it is 5 or more or
          the patient isn&rsquo;t alert. When oxygen saturation falls below target (92–96%, or 88–92% with COPD), they
          start or turn up oxygen; this hides a falling saturation, just as on a real ward.
        </p>
      </section>

      <section className="section" id="deterioration" aria-labelledby="deterioration-title">
        <h2 id="deterioration-title">Who deteriorates</h2>
        <p className={styles.prose}>
          A <strong>hidden rule</strong> gives every patient, every hour, a small chance of starting to deteriorate. The
          chance rises with abnormal vital signs, age, long-term illness and a personal frailty nobody can see. The rule
          lives only in the simulator: the model never sees it and has to learn the pattern from the data.
        </p>
        <ul className={styles.list}>
          <li>
            A deterioration builds over 2 to 12 hours, as a breathing problem, an infection spreading into the blood
            (sepsis) or a heart problem, each changing the vital signs in its own way. About 1 in 10 comes on in under
            2 hours.
          </li>
          <li>
            About a quarter recover on their own. The rest end in an escalation: a rapid-response call (about 70%) or
            a move to intensive care (about 30%), after which the patient leaves the ward.
          </li>
          <li>The result: about 3 to 4 escalations per 100 patient-days.</li>
          <li>
            An escalation is <strong>recorded 6 hours after</strong> it happens, as outcomes are in practice. So the
            most recent hours can&rsquo;t be judged yet, and the model is only trained on hours whose outcome is
            known.
          </li>
        </ul>
      </section>

      <section className="section" id="records" aria-labelledby="records-title">
        <h2 id="records-title">Where the data goes</h2>
        <p className={styles.prose}>Like a real hospital, the data arrives from two kinds of system:</p>
        <ul className={styles.list}>
          <li>
            <strong>Hospital records</strong> (a SQL database): hospitals, units, patients, their conditions and
            medicines, stays (admission, transfers, discharge), nurse observations and which monitor is on which
            patient. Records are updated as things happen, and a copy job picks up only what changed.
          </li>
          <li>
            <strong>Monitor readings and outcomes</strong> (files): one file per hour of readings, as a device gateway
            would write them, and the escalations as they are recorded.
          </li>
        </ul>
        <p className={styles.prose}>
          Both flow into Azure Databricks, where they are checked, cleaned and joined in layers: raw (bronze), cleaned
          and linked to patients (silver), and hourly features, risk scores and ward summaries (gold). The{" "}
          <Link href="/behind-the-scenes">behind the scenes</Link> page shows the rest of the journey, and how well the
          model does.
        </p>
      </section>

      <section className="section" id="drift" aria-labelledby="drift-title">
        <h2 id="drift-title">Planned changes</h2>
        <p className={styles.prose}>
          To show how the platform notices when the world changes, four changes can be switched on in the
          simulation. Each calls for a different response.
        </p>
        <div className={styles.cards}>
          {DRIFT.map((d) => (
            <article key={d.name}>
              <h3>{d.name}</h3>
              <p>{d.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section" id="limits" aria-labelledby="limits-title">
        <h2 id="limits-title">What this world leaves out</h2>
        <p className={styles.prose}>A simulation is simpler than a hospital. The main gaps:</p>
        <ul className={styles.list}>
          {LEFT_OUT.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <p className={`${styles.prose} muted small`}>
          All data is synthetic. HeavDen Nexus is a portfolio project, not a medical device.
        </p>
      </section>
    </div>
  );
}
