"use client";

import Link from "next/link";

import { PageHero } from "@/components/PageHero";
import { ProofTiles } from "@/components/ProofTiles";
import { TechStack } from "@/components/TechStack";
import { useApi } from "@/lib/api";
import { riskPct, when } from "@/lib/format";
import { CODE_PAGE } from "@/lib/site";
import { PROJECT_STACK } from "@/lib/stack";
import type { Health } from "@/lib/types";

import styles from "./behind.module.css";

const FLOW = [
  {
    title: "Invent the patients",
    body: "About 2,000 people with realistic medical histories (Synthea), admitted to three fictional hospitals with 360 monitored beds. Nobody here is real.",
  },
  {
    title: "Simulate the ward",
    body: "Wearable monitors send six vital signs every 5 minutes, with real-world faults: noise, motion, dropped messages, flat batteries, stuck sensors. A hidden rule decides who deteriorates; nurses chart oxygen and alertness. The hospital keeps its own simulated clock, shown in the top bar.",
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
    title: "Add the AI assistant",
    body: "Nexus AI answers questions in plain English with three tools: the wards' numbers, a search of the hospital's documents and a patient's risk explanation. Every answer shows the queries and sources behind it.",
  },
  {
    title: "Serve this app",
    body: "Next.js on Vercel and a FastAPI service on Azure Container Apps that sleeps when nobody is visiting. It reads production's gold tables, with a saved snapshot of them, the model and the document index to fall back on.",
  },
];

// How each part was built. Every statement here can be checked in the repository (docs/Phase-5-README.md
// walks through each one); keep them true when the build changes.
const PARTS = [
  {
    name: "Analytics",
    body: "Each scoring run appends to four gold tables: the prediction log, alerts, escalations and hourly figures per hospital. An alert is a patient entering the High band, not repeated for the next 6 hours. Two Databricks AI/BI dashboards, Clinical Ops and Model & Drift, are defined as code and deployed to each environment. This site's Analytics page runs fixed SQL on the same gold tables.",
  },
  {
    name: "Machine learning",
    body: "Each patient-hour is described only from what was known by then: vital-sign levels and trends over the last 1 to 6 hours and the patient's background. Labels come from recorded outcomes: an intensive care transfer or rapid response call within 6 hours. Logistic regression and LightGBM train on older days, are chosen on the next ones and tested once on the most recent; probabilities are calibrated, and the alert threshold comes from the alert budget. MLflow tracks every run, Unity Catalog holds the models, and only an approved promotion changes the one in use.",
  },
  {
    name: "Nexus AI",
    body: "One tool-calling loop on gpt-5-mini (Azure OpenAI) with three tools. Numbers: on live data, Databricks Genie writes and runs the SQL on production's gold tables; on the saved snapshot, the model's own SQL runs on a locked, read-only copy. Documents: hybrid search (FAISS vectors and BM25 keywords) over nine fictional hospital documents, cited in each answer. Patients: the top reasons behind a patient's latest score. Guardrails screen each question, fix the assistant's scope and check every answer for leaks.",
  },
  {
    name: "The website",
    body: "Next.js on Vercel, and a FastAPI service in one container on Azure Container Apps that scales to zero. Pages read production's gold tables through the SQL warehouse, capped at 2 hours of warehouse time a day; past that they read the saved snapshot from Azure Blob Storage. Only the website can call the API, Vercel's network caches pages, and chat has a limit per visitor and a daily cap. The API's image is built and tested in GitHub Actions and reaches production only through an approved release.",
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
      <PageHero
        title="Behind the scenes"
        aside={<ProofTiles />}
        actions={
          <>
            <a href="#results-title" className="buttonBright">
              See the results
            </a>
            <Link href={CODE_PAGE} className="buttonGhost">
              Request the code
            </Link>
          </>
        }
      >
        <p className="lede">
          How HeavDen Nexus is built, how well the model does in production, and what testing changed along the
          way.
        </p>
      </PageHero>

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

      <div className={styles.resultsAndParts}>
        <section className="section" aria-labelledby="results-title">
          <h2 id="results-title">How the model compares with NEWS2</h2>
          <div className={styles.news2}>
            <p>
              <strong>NEWS2</strong>, the National Early Warning Score 2 (Royal College of Physicians, 2017), is the
              bedside score many hospitals use to decide when a patient needs help. Nurses give points for seven
              observations: breathing rate, oxygen saturation, extra oxygen, systolic blood pressure, heart rate,
              level of consciousness and temperature. A total of 5 or more, or 3 for any single observation, calls
              for an urgent review; 7 or more calls for an emergency response.
            </p>
            <p>
              <strong>Why it is the bar to beat:</strong> it is what nurses already use. A model that alerts them is
              only worth having if, for the same number of alerts a nurse can handle, it finds more of the patients
              who are getting worse. Here NEWS2 is computed in full, including the second oxygen scale for patients
              with COPD, on the same hours as the model, and each score gets its own alert threshold for the same
              budget.
            </p>
          </div>
          <p className="muted">
            The production model (version 3) on its test block: the 2 most recent days of its data, which it never
            saw in training, with 28 escalations. Each score is allowed about 2 alerts per nurse per 12-hour shift.
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
                  <td className="num"><strong>22 of 28</strong></td>
                  <td className="num">17 of 28</td>
                </tr>
                <tr>
                  <td>Median warning, when flagged</td>
                  <td className="num"><strong>4 hours</strong></td>
                  <td className="num">3 hours</td>
                </tr>
                <tr>
                  <td>Alerts followed by an escalation within 6 hours</td>
                  <td className="num">about 1 in 25</td>
                  <td className="num">about 1 in 24</td>
                </tr>
                <tr>
                  <td>Alerts per nurse per shift</td>
                  <td className="num">2.15</td>
                  <td className="num">1.79</td>
                </tr>
                <tr>
                  <td>Ranking quality (AUROC; higher is better)</td>
                  <td className="num"><strong>0.76</strong></td>
                  <td className="num">0.74</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="muted small" style={{ maxWidth: 760 }}>
            The model warns about 5 more of the 28 deteriorating patients, an hour earlier at the median, for a
            slightly higher alert load; it does not raise fewer false alarms. With only 28 escalations, small
            differences are within noise. Simulated deteriorations are smoother than real ones, so real-world gains
            would likely be smaller. Full figures are in the model card, which comes with the code.
          </p>
          {h && (
            <p className="small muted">
              Serving model: {h.model.name} version {h.model.version} (
              {h.model.type === "Pipeline" ? "logistic regression" : h.model.type.startsWith("LGBM") ? "LightGBM" : h.model.type}).
              Alert threshold: risk of {riskPct(h.risk_bands.high)} or more. Data as of {when(h.as_of)}.
            </p>
          )}
        </section>

        <section className="section" aria-labelledby="parts-title">
          <h2 id="parts-title">How each part was built</h2>
          <div className={styles.parts}>
            {PARTS.map((part) => (
              <article key={part.name}>
                <h3>{part.name}</h3>
                <p>{part.body}</p>
              </article>
            ))}
          </div>
        </section>
      </div>

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

      <section className={`whiteBand ${styles.stackBand}`} aria-labelledby="stack-title">
        <div className={styles.stackHead}>
          <h2 id="stack-title">Tech stack</h2>
          <p className="muted">Everything HeavDen Nexus runs on, from the cloud to the chat.</p>
        </div>
        <TechStack groups={PROJECT_STACK} />
      </section>

      <section className="section" aria-labelledby="code-title">
        <h2 id="code-title">The code</h2>
        <p>
          Everything, from the patient generator to this page, with learning notebooks for each step, is shared on
          request with recruiters and hiring teams: <Link href={CODE_PAGE}>request access</Link>.
        </p>
      </section>
    </div>
  );
}
