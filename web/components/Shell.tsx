"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useApi } from "@/lib/api";
import { when } from "@/lib/format";
import type { Health } from "@/lib/types";

import styles from "./Shell.module.css";

const NAV = [
  { href: "/analytics", label: "Analytics" },
  { href: "/patients", label: "Patients" },
  { href: "/assistant", label: "Assistant" },
  { href: "/behind-the-scenes", label: "Behind the scenes" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  // Calling /health on every page load also wakes the backend if Render has put it to sleep.
  const health = useApi<Health>("/health");

  return (
    <>
      <a className={styles.skip} href="#main">
        Skip to content
      </a>
      <header className={styles.header}>
        <div className={styles.bar}>
          <Link href="/" className={styles.brand}>
            HeavDen <span className={styles.brandProduct}>Nexus</span>
          </Link>
          <nav aria-label="Main" className={styles.nav}>
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={styles.navLink}
                aria-current={path.startsWith(item.href) ? "page" : undefined}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
        <p className={styles.status}>
          <strong>Synthetic data.</strong> Every patient and hospital here is invented.{" "}
          {health.data ? (
            <>
              Ward time is {when(health.data.as_of)}
              {health.data.mode === "demo" ? ", from a saved snapshot." : "."}
            </>
          ) : health.error ? (
            <span className={styles.offline}>The data service isn&rsquo;t answering yet; it may be waking up.</span>
          ) : (
            <>Connecting to the data service&hellip;</>
          )}
        </p>
      </header>
      <main id="main">{children}</main>
      <footer className={styles.footer}>
        <p>
          HeavDen Nexus is a portfolio project. It is not a medical device and must not be used for real patient
          care.
        </p>
      </footer>
    </>
  );
}
