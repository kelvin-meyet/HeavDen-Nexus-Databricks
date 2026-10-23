import type { Metadata } from "next";
import Link from "next/link";

import { PageHero } from "@/components/PageHero";
import { TechStack } from "@/components/TechStack";
import { BUILDER, CODE_REQUEST, MAILTO } from "@/lib/site";
import { EXPERIENCE_STACK } from "@/lib/stack";

import styles from "./work.module.css";

export const metadata: Metadata = {
  title: `About me · ${BUILDER.name}`,
  description: `${BUILDER.name}, ${BUILDER.title}: data engineering, analytics engineering, full-stack data science, MLOps and AI engineering. The engineer who built HeavDen Nexus.`,
};

// A personal page: what I do, what I've built and how to reach me. It offers no services (Vercel's
// Hobby plan is for non-commercial personal use). The project's own results and stack are on Behind
// the scenes; this page is about the person.

const GLANCE = [
  { label: "Platforms", items: ["Azure", "Databricks", "AWS", "Google Cloud"] },
  { label: "Languages", items: ["Python", "SQL", "R"] },
  {
    label: "Focus",
    items: ["Data engineering", "Analytics engineering", "Data science", "MLOps", "AI engineering"],
  },
];

const SKILLS = [
  {
    name: "Data engineering and pipelines",
    body: "Ingestion from databases and files, change data capture, medallion pipelines with data-quality checks, governance with Unity Catalog, infrastructure as code. On Databricks, or with plain SQL, Python and DuckDB when that fits better.",
    proof: "Azure SQL and device files into bronze, silver and gold, every reading reconciled.",
    href: "/about-the-data",
  },
  {
    name: "Analytics engineering and dashboards",
    body: "Modelled, governed data layers with metric definitions people agree on, and enterprise-scale dashboards that stay fast, accurate and access-controlled as the questions and the users grow.",
    proof: "Clinical Ops and Model & Drift dashboards on Databricks AI/BI, over the gold tables; the Analytics page here.",
    href: "/analytics",
  },
  {
    name: "Full-stack data science",
    body: "From the question to a model people use: framing the decision, a baseline to beat, features, time-based validation, calibrated probabilities, explanations people can act on, and the API and app that put it in front of them.",
    proof: "A risk model that beats NEWS2 at the same alert budget, served here with a what-if panel.",
    href: "/patients",
  },
  {
    name: "MLOps and monitoring",
    body: "Experiment tracking and a model registry with MLflow, CI/CD, drift detection, and retraining and promotion behind human approval, so models stay trustworthy after launch.",
    proof: "Drift monitoring and approval-gated retraining, proven on three drift episodes in production.",
    href: "/behind-the-scenes",
  },
  {
    name: "AI assistants grounded in data",
    body: "Tool-calling assistants that combine text-to-SQL, document search (RAG) and custom functions, with guardrails, citations, and evaluation and red-teaming before anyone relies on them.",
    proof: "The ward assistant: queries, cited documents and patient explanations.",
    href: "/assistant",
  },
  {
    name: "Cloud cost and governance",
    body: "Serverless and scale-to-zero designs, budgets and alerts, least-privilege access, and secrets kept out of code, so a platform is affordable to keep running.",
    proof: "Built on a $200 credit; designed to run for under $25 a month.",
    href: "/behind-the-scenes",
  },
];

// Credentials. Paste each one's verification link into `url` and its "View credential" link appears;
// until then the card shows without one, so there is never a dead link.
const CERTIFICATIONS: { name: string; issuer: string; url?: string }[] = [
  { name: "End-to-end MLOps with Databricks", issuer: "Maven", url: "https://maven.com/certificate/WgcEcOVq" },
  {
    name: "Machine Learning in the Enterprise",
    issuer: "Google Cloud",
    url: "https://www.coursera.org/account/accomplishments/verify/PLQF9JSJ63GH",
  },
  {
    name: "Production Machine Learning Systems",
    issuer: "Google Cloud",
    url: "https://www.coursera.org/account/accomplishments/verify/BN6QJKJ9RM5Y",
  },
  {
    name: "AWS Certified Solutions Architect – Associate (SAA)",
    issuer: "Amazon Web Services",
    url: "https://www.credly.com/badges/976f8472-d3ea-4c32-ad8c-3f394949417e/public_url",
  },
  {
    name: "AWS Certified Cloud Practitioner (CCP)",
    issuer: "Amazon Web Services",
    url: "https://www.credly.com/badges/35f0d01a-abc7-4848-8407-8e54f8b4cd9c/public_url",
  },
  { name: "Python for Data Science, AI & Development", issuer: "IBM" },
  { name: "Microsoft Azure AI Fundamentals", issuer: "Opsgility" },
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
    body: "Decisions written down, tests in place, and documentation a team can pick up without me.",
  },
];

export default function AboutMePage() {
  return (
    <div className="page">
      <PageHero
        lockup={
          <p className={styles.lockup}>
            <span className={styles.name}>{BUILDER.name}</span>
            <span className={styles.role}>{BUILDER.title}</span>
          </p>
        }
        title="I build data platforms, predictive models and AI assistants that people can trust."
        actions={
          <>
            <a href="#contact" className="buttonBright">
              Get in touch
            </a>
            <a href="#code" className={styles.codeLink}>
              Request the code
            </a>
          </>
        }
        aside={
          <section className={styles.glance} aria-labelledby="glance-title">
            <h2 id="glance-title">At a glance</h2>
            <dl>
              {GLANCE.map((row) => (
                <div key={row.label}>
                  <dt>{row.label}</dt>
                  <dd>
                    <ul>
                      {row.items.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        }
      >
        <p className="lede">
          My work spans the whole data lifecycle: data engineering, analytics engineering with enterprise-scale
          dashboards, full-stack data science from the first question to a model in production, MLOps and AI
          engineering, across Azure, AWS and Google Cloud. HeavDen Nexus, this app, is my worked example, built and
          tested end to end.
        </p>
      </PageHero>

      <div className={styles.builtAndHow}>
        <section className={styles.built} aria-labelledby="built-title">
          <h2 id="built-title">What I&rsquo;ve built</h2>
          <ul className={styles.skills}>
            {SKILLS.map((s) => (
              <li key={s.name}>
                <h3>{s.name}</h3>
                <p>{s.body}</p>
                <p className={styles.skillProof}>
                  <span className={styles.proofLabel}>In this app:</span> {s.proof} <Link href={s.href}>See it</Link>
                </p>
              </li>
            ))}
          </ul>
        </section>

        <aside className={styles.how} aria-labelledby="how-title">
          <h2 id="how-title">How I work</h2>
          <ol className={styles.principles}>
            {PRINCIPLES.map((p) => (
              <li key={p.name}>
                <h3>{p.name}</h3>
                <p>{p.body}</p>
              </li>
            ))}
          </ol>
        </aside>
      </div>

      <section className={`whiteBand ${styles.stackBand}`} aria-labelledby="stack-title">
        <div className={styles.stackHead}>
          <h2 id="stack-title">Tech stack</h2>
          <p className="muted">The tools and techniques I work with, across projects and clouds.</p>
        </div>
        <TechStack groups={EXPERIENCE_STACK} />
      </section>

      <section className={`section ${styles.certs}`} aria-labelledby="certs-title">
        <h2 id="certs-title">Certifications and training</h2>
        <ul className={styles.certList}>
          {CERTIFICATIONS.map((c) => (
            <li key={c.name}>
              <span className={styles.certIssuer}>{c.issuer}</span>
              <span className={styles.certName}>{c.name}</span>
              {c.url && (
                <a href={c.url} className={styles.certLink} rel="noopener noreferrer" target="_blank">
                  View credential<span className="visually-hidden"> (opens in a new tab)</span>
                </a>
              )}
            </li>
          ))}
        </ul>
      </section>

      <section id="contact" className={`nightBand ${styles.contact}`} aria-labelledby="contact-title">
        <h2 id="contact-title">Get in touch</h2>
        <p>
          Open to full-time, contract and freelance opportunities in data platform, analytics, ML and AI
          engineering. If you&rsquo;re hiring, I&rsquo;d like to hear from you.
        </p>
        <div className={styles.contactActions}>
          <a href={MAILTO} className="buttonBright">
            Email me
          </a>
          <a href={BUILDER.linkedin} className="buttonGhost" rel="noopener noreferrer" target="_blank">
            LinkedIn<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
          <a href={CODE_REQUEST} className="buttonGhost">
            Request the code
          </a>
        </div>
        <p id="code" className={styles.codeNote}>
          The full source code, from the data platform and the model to Nexus AI and this site, is shared on
          request with recruiters and hiring teams.
        </p>
      </section>
    </div>
  );
}
