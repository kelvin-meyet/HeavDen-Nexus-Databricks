"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useApi } from "@/lib/api";
import { BUILDER } from "@/lib/site";
import { whenFull } from "@/lib/format";
import type { Health } from "@/lib/types";

import styles from "./Shell.module.css";

const NAV = [
  { href: "/", label: "Overview", exact: true },
  { href: "/patients", label: "Patients" },
  { href: "/analytics", label: "Analytics" },
  { href: "/assistant", label: "Assistant" },
  { href: "/behind-the-scenes", label: "Behind the scenes" },
];

const CTA = { href: "/work-with-me", label: "Work with me" };

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  // Calling /health on every page load also wakes the backend if Render has put it to sleep.
  const health = useApi<Health>("/health");
  // On narrow screens the links sit behind a Menu button.
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (e: KeyboardEvent) => e.key === "Escape" && setMenuOpen(false);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [menuOpen]);

  const isActive = (href: string, exact?: boolean) => (exact ? path === href : path.startsWith(href));
  const note =
    health.data?.mode === "demo"
      ? "The wards are a 14-day simulation, paused at this hour. Every date here is simulated."
      : "The wards run on a simulated clock, moved forward by each simulation run.";

  return (
    <div className={styles.app}>
      <a className={styles.skip} href="#main">
        Skip to content
      </a>
      <header className={styles.topbar}>
        <div className={styles.bar}>
          <Link href="/" className={styles.brand}>
            <span className={styles.brandMark} aria-hidden="true">
              <svg viewBox="0 0 32 32" width="22" height="22">
                <polyline
                  points="2,20 9,20 12,9 16,26 20,14 23,20 30,20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.6"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              </svg>
            </span>
            <span>
              HeavDen <span className={styles.brandProduct}>Nexus</span>
            </span>
          </Link>

          <button
            type="button"
            className={styles.menuButton}
            aria-expanded={menuOpen}
            aria-controls="main-nav"
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? "Close" : "Menu"}
          </button>

          <nav id="main-nav" aria-label="Main" className={styles.nav} data-open={menuOpen}>
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={styles.navLink}
                onClick={() => setMenuOpen(false)}
                aria-current={isActive(item.href, item.exact) ? "page" : undefined}
              >
                {item.label}
              </Link>
            ))}
            <Link
              href={CTA.href}
              className={styles.cta}
              onClick={() => setMenuOpen(false)}
              aria-current={isActive(CTA.href) ? "page" : undefined}
            >
              {CTA.label}
            </Link>
          </nav>

          <div className={styles.status} title={health.data ? note : undefined}>
            {health.data ? (
              <>
                <span className={styles.liveDot} aria-hidden="true" />
                <span className={styles.statusLabel}>Simulated ward time</span>
                <span className={styles.time}>{whenFull(health.data.as_of)}</span>
              </>
            ) : health.error ? (
              <span className={styles.offline}>Data service waking up&hellip;</span>
            ) : (
              <span className={styles.statusLabel}>Connecting&hellip;</span>
            )}
          </div>
        </div>
      </header>

      <p className={styles.synthetic}>
        <strong>Synthetic data.</strong> Every patient and hospital here is invented. Not a medical device.
        {health.data && <span className={styles.syntheticNote}> {note}</span>}{" "}
        <Link href="/about-the-data">How the data is made</Link>
      </p>

      <div className={styles.content}>
        <main id="main">{children}</main>
        <footer className={styles.footer}>
          HeavDen Nexus is a portfolio project. It is not a medical device and must not be used for real patient
          care. <Link href="/about-the-data">About the hospital and its data</Link>
          <span className={styles.credit}>
            Built by <Link href="/work-with-me">{BUILDER.name}</Link>, {BUILDER.title} ·{" "}
            <a href={BUILDER.linkedin} rel="noopener noreferrer" target="_blank">
              LinkedIn
            </a>{" "}
            ·{" "}
            <a href={BUILDER.github} rel="noopener noreferrer" target="_blank">
              GitHub
            </a>
          </span>
        </footer>
      </div>
    </div>
  );
}
