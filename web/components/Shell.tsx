"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useApi } from "@/lib/api";
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

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  // Calling /health on every page load also wakes the backend if Render has put it to sleep.
  const health = useApi<Health>("/health");

  return (
    <div className={styles.app}>
      <a className={styles.skip} href="#main">
        Skip to content
      </a>
      <aside className={styles.sidebar}>
        <Link href="/" className={styles.brand}>
          <span className={styles.brandMark} aria-hidden="true">
            <svg viewBox="0 0 32 32" width="28" height="28">
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

        <nav aria-label="Main" className={styles.nav}>
          {NAV.map((item) => {
            const active = item.exact ? path === item.href : path.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={styles.navLink}
                aria-current={active ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className={styles.status}>
          {health.data ? (
            <>
              <p className={styles.live}>
                <span className={styles.liveDot} aria-hidden="true" />
                Simulated ward time
              </p>
              <p className={styles.time}>{whenFull(health.data.as_of)}</p>
              <p className={styles.statusNote}>
                {health.data.mode === "demo"
                  ? "The wards are a 14-day simulation, paused at this hour. Every date here is simulated."
                  : "The wards run on a simulated clock, moved forward by each simulation run."}
              </p>
            </>
          ) : health.error ? (
            <p className={styles.offline}>The data service isn&rsquo;t answering yet; it may be waking up.</p>
          ) : (
            <p className={styles.statusNote}>Connecting to the data service&hellip;</p>
          )}
        </div>

        <p className={styles.synthetic}>
          <strong>Synthetic data.</strong> Every patient and hospital here is invented. Not a medical device.{" "}
          <Link href="/about-the-data" className={styles.aboutLink}>
            How the data is made
          </Link>
        </p>
      </aside>

      <div className={styles.content}>
        <main id="main">{children}</main>
        <footer className={styles.footer}>
          HeavDen Nexus is a portfolio project. It is not a medical device and must not be used for real patient
          care. <Link href="/about-the-data">About the hospital and its data</Link>
        </footer>
      </div>
    </div>
  );
}
