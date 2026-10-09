import type { Metadata } from "next";
import Link from "next/link";

import { BUILDER, MAILTO } from "@/lib/site";

import styles from "./work.module.css";

export const metadata: Metadata = {
  title: `Work with me · ${BUILDER.name}`,
  description: `${BUILDER.name}, ${BUILDER.title}: data platforms on Databricks and beyond, predictive models and MLOps, and AI assistants, for teams in any industry.`,
};

// Proof figures come from this project's own measurements (README results, the dev pipeline run
// of 9 Oct 2026, the assistant's red-team run). Keep them true: update them when those change.
const PROOF = [
  { figure: "25 of 28", label: "escalations flagged in advance by the model, against 21 for the clinical standard" },
  { figure: "153,311", label: "readings checked and reconciled through the data platform, none lost" },
  { figure: "19 of 19", label: "adversarial tests passed by the AI assistant" },
  { figure: "< $25", label: "a month to keep the whole platform running, by design" },
];

const SERVICES = [
  {
    name: "Data platforms and pipelines",
    body: "Ingestion from databases and files, change data capture, medallion pipelines with data-quality checks, governance with Unity Catalog, infrastructure as code. On Databricks, or with plain SQL, Python and DuckDB when that fits better.",
    proof: "Azure SQL and device files into bronze, silver and gold, every reading reconciled.",
    href: "/about-the-data",
  },
  {
    name: "Predictive models that earn their place",
    body: "Framing the decision first, a baseline to beat, time-based splits, calibrated probabilities, explanations people can act on, and evaluation that shows uncertainty rather than hiding it.",
    proof: "A risk model compared with NEWS2 at the same alert budget, with bootstrap intervals.",
    href: "/behind-the-scenes",
  },
  {
    name: "MLOps and monitoring",
    body: "Experiment tracking and a model registry with MLflow, CI/CD, drift detection, and retraining and promotion behind human approval, so models stay trustworthy after launch.",
    proof: "Tests on every change; drift monitoring and approval-gated retraining being built now.",
    href: "/behind-the-scenes",
  },
  {
    name: "AI assistants on your data",
    body: "Tool-calling assistants that combine text-to-SQL, document search (RAG) and your own functions, with guardrails, citations, and evaluation and red-teaming before anyone relies on them.",
    proof: "The ward assistant: queries, cited documents and patient explanations.",
    href: "/assistant",
  },
  {
    name: "Data products people use",
    body: "APIs and web apps that put data and models in front of the people who decide: FastAPI, Next.js, clear charts and plain-language explanations.",
    proof: "This app: ward board, patient charts with a what-if panel, analytics.",
    href: "/patients",
  },
  {
    name: "Cloud cost and governance",
    body: "Serverless and scale-to-zero designs, budgets and alerts, least-privilege access, and secrets kept out of code, so a platform is affordable to keep running.",
    proof: "Built on a $200 credit; designed to run for under $25 a month.",
    href: "/behind-the-scenes",
  },
];

const PRINCIPLES = [
  {
    name: "Start from the decision",
    body: "What will someone do differently because of this work? That sets the metric, the data and the model.",
  },
  {
    name: "Beat the baseline first",
    body: "A simple rule or a SQL query often does the job. A model has to earn its place against it.",
  },
  {
    name: "Test before the cloud",
    body: "Check data and code against the real rules offline, so the first cloud run is a confirmation, not a debugging session.",
  },
  {
    name: "Measure, don't guess",
    body: "Reconcile every record, measure what each run costs, and report uncertainty honestly.",
  },
  {
    name: "Leave it understandable",
    body: "Decisions written down, tests in place, and documentation your team can pick up without me.",
  },
];

export default function WorkWithMePage() {
  return (
    <div className="page">
      <header className={`pageHead ${styles.head}`}>
        <p className={styles.eyebrow}>
          {BUILDER.name} · {BUILDER.title}
        </p>
        <h1>I build data platforms, predictive models and AI assistants that people can trust.</h1>
        <p className="lede">
          I&rsquo;m the developer behind {BUILDER.brand}. HeavDen Nexus, this app, is my worked example: synthetic
          hospital data, a cloud data platform, a model that has to beat the clinical standard, an AI assistant and
          this web app, built and tested end to end. I work with teams in any industry, as a hands-on engineer on
          Databricks or anywhere else.
        </p>
        <div className={styles.actions}>
          <a href={MAILTO} className="button">
            Email me
          </a>
          <a href={BUILDER.linkedin} className="buttonQuiet" rel="noopener noreferrer" target="_blank">
            LinkedIn<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
          <a href={BUILDER.github} className="buttonQuiet" rel="noopener noreferrer" target="_blank">
            GitHub<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
        </div>
      </header>

      <section className="section" aria-labelledby="proof-title">
        <h2 id="proof-title" className="visually-hidden">
          Proof from this project
        </h2>
        <dl className={styles.proof}>
          {PROOF.map((p) => (
            <div key={p.figure}>
              <dt>{p.label}</dt>
              <dd>{p.figure}</dd>
            </div>
          ))}
        </dl>
        <p className="muted small">Measured on this project&rsquo;s synthetic data; details on each linked page.</p>
      </section>

      <section className="section" aria-labelledby="services-title">
        <h2 id="services-title">What I can build for you</h2>
        <ul className={styles.services}>
          {SERVICES.map((s) => (
            <li key={s.name}>
              <h3>{s.name}</h3>
              <p>{s.body}</p>
              <p className={styles.serviceProof}>
                <span className={styles.proofLabel}>In this app:</span> {s.proof}{" "}
                <Link href={s.href}>See it</Link>
              </p>
            </li>
          ))}
        </ul>
      </section>

      <section className="section" aria-labelledby="how-title">
        <h2 id="how-title">How I work</h2>
        <ol className={styles.principles}>
          {PRINCIPLES.map((p) => (
            <li key={p.name}>
              <h3>{p.name}</h3>
              <p>{p.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className={`section ${styles.contact}`} aria-labelledby="contact-title">
        <h2 id="contact-title">Let&rsquo;s talk</h2>
        <p>
          Open to full-time roles, contracts and consulting projects. Tell me about the decision your data should
          support, and I&rsquo;ll tell you how I&rsquo;d approach it.
        </p>
        <div className={styles.actions}>
          <a href={MAILTO} className="button">
            {BUILDER.email}
          </a>
          <a href={BUILDER.linkedin} className="buttonQuiet" rel="noopener noreferrer" target="_blank">
            LinkedIn<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
          <a href={BUILDER.repo} className="buttonQuiet" rel="noopener noreferrer" target="_blank">
            This project&rsquo;s code<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
        </div>
      </section>
    </div>
  );
}
