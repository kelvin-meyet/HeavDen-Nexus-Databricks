"use client";

import Link from "next/link";

import { useApi } from "@/lib/api";
import { riskPct, when } from "@/lib/format";
import type { Health } from "@/lib/types";

import styles from "./behind.module.css";

const REPO = "https://github.com/kelvin-meyet/HeavDen-Nexus-Databricks";

const FLOW = [
  {
    title: "Invent the patients",
    body: "About 2,000 people with realistic medical histories (Synthea), admitted to three fictional hospitals with 360 monitored beds. Nobody here is real.",
  },
  {
    title: "Simulate the ward",
    body: "Wearable monitors send six vital signs every 5 minutes, with real-world faults: noise, motion, dropped messages, flat batteries, stuck sensors. A hidden rule decides who deteriorates; nurses chart oxygen and alertness. The simulation runs on its own calendar: 14 days from 1 November 2026, paused at the last hour, which is the ward time shown in the sidebar.",
  },
  {
    title: "Build the data platform",
    body: "Raw readings and hospital records flow into Azure Databricks through bronze, silver and gold layers, with data-quality checks at each step.",
  },
  {
    title: "Train and test the model",
    body: "Hourly features from the last 1 to 6 hours predict escalation within 6 hours. Two candidates are compared on later days than they were trained on, and must beat the NEWS2 score nurses use today.",
  },
  {
    title: "Watch it, with a human in charge",
    body: "Drift monitoring compares new data with what the model learned. When something changes, an email goes out, and retraining or promotion only happens after an approval on GitHub.",
  },
  {
    title: "Serve this app",
    body: "Next.js on Vercel and a FastAPI service on Render. Without cloud credits it runs from a saved snapshot of the gold tables, the model and the document index.",
  },
];

const LESSONS = [
  {
    title: "Count alerts, not hours",
    body: "The alert budget was first applied as \"3.3% of hours in the High band\". But a score that hovers around its threshold re-alerts every time it crosses, so the model raised about 1.6 times as many alerts as NEWS2 at the same share of hours. Thresholds are now set from the number of alerts nurses can handle.",
  },
  {
    title: "Drop the Medium band",
    body: "Four definitions of a middle band were tested, including \"risk rising\" and \"recently High\". None was followed by an escalation more often than an average hour. Instead of a colour that means nothing, the ward list shows each patient's risk and whether it is rising.",
  },
  {
    title: "Never claim certainty",
    body: "The calibration step once turned the top 5 validation hours (all escalated) into a 100% risk. Two anchor points now keep every prediction strictly between 0% and 100%, without changing the ranking.",
  },
  {
    title: "Split by time, never at random",
    body: "A random split lets the model see neighbouring hours of the same patient in training and testing, which inflated its main score from 0.40 to 0.51. Every result here comes from later days than the model saw.",
  },
  {
    title: "Pick the assistant's model by checking its answers",
    body: "The cheapest model wrote fluent but wrong answers: it called the gap between 89% and 75% \"4 points\" instead of 14, and accepted a false premise about a sensor fault. Every recorded answer is now checked against the SQL and documents behind it.",
  },
];

export default function BehindPage() {
  const health = useApi<Health>("/health");
  const h = health.data;

  return (
    <div className="page">
      <header className="pageHead">
        <h1>Behind the scenes</h1>
        <p className="lede">
          How HeavDen Nexus is built, how well the model does, and what testing changed along the way.
        </p>
      </header>

      <section className="section" aria-labelledby="flow-title">
        <h2 id="flow-title">From invented patients to this page</h2>
        <ol className={styles.flow}>
          {FLOW.map((step) => (
            <li key={step.title}>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </li>
          ))}
        </ol>
        <p className="small muted">
          How the hospitals run and how each piece of data is made:{" "}
          <Link href="/about-the-data">the hospital and its data</Link>.
        </p>
      </section>

      <section className="section" aria-labelledby="results-title">
        <h2 id="results-title">How the model compares with NEWS2</h2>
        <p className="muted">
          Two test days with 28 escalations, each score allowed about 2 alerts per nurse per 12-hour shift.
        </p>
        <div className="tableWrap" style={{ maxWidth: 760 }}>
          <table className="table">
            <thead>
              <tr>
                <th scope="col">What was measured</th>
                <th scope="col" className="num">Model</th>
                <th scope="col" className="num">NEWS2</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Escalations flagged in the 6 hours before</td>
                <td className="num"><strong>25 of 28</strong></td>
                <td className="num">21 of 28</td>
              </tr>
              <tr>
                <td>Flagged at least 3 hours ahead</td>
                <td className="num"><strong>19</strong></td>
                <td className="num">12</td>
              </tr>
              <tr>
                <td>Alerts followed by an escalation within 6 hours</td>
                <td className="num">about 1 in 34</td>
                <td className="num">about 1 in 32</td>
              </tr>
              <tr>
                <td>Ranking quality (AUPRC; higher is better)</td>
                <td className="num"><strong>0.40</strong></td>
                <td className="num">0.22</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="muted small" style={{ maxWidth: 760 }}>
          The model catches more deteriorating patients earlier; it does not raise fewer false alarms. With only
          28 escalations, the gain could plausibly be anywhere from 3 to 28 percentage points. Simulated
          deteriorations are smoother than real ones, so real-world gains would likely be smaller.
        </p>
        {h && (
          <p className="small muted">
            Serving model: {h.model.name} version {h.model.version} ({h.model.type === "Pipeline" ? "logistic regression" : h.model.type}).
            Alert threshold: risk of {riskPct(h.risk_bands.high)} or more. Data as of {when(h.as_of)}.
          </p>
        )}
      </section>

      <section className="section" aria-labelledby="lessons-title">
        <h2 id="lessons-title">What testing changed</h2>
        <div className={styles.lessons}>
          {LESSONS.map((l) => (
            <article key={l.title}>
              <h3>{l.title}</h3>
              <p>{l.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="code-title">
        <h2 id="code-title">The code</h2>
        <p>
          Everything, from the patient generator to this page, is in the{" "}
          <a href={REPO}>HeavDen-Nexus repository on GitHub</a>, with learning notebooks for each step.
        </p>
      </section>
    </div>
  );
}
