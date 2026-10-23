"use client";

import { useEffect, useRef, useState } from "react";

import type { StackGroup } from "@/lib/stack";

import styles from "./TechStack.module.css";

/** A tech stack as cards of chips in columns. The cards, then their chips, cascade in once when the
 *  stack scrolls into view; without JavaScript, or with reduced motion, everything is simply there. */
export function TechStack({ groups }: { groups: StackGroup[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [reveal, setReveal] = useState<"none" | "pending" | "shown">("none");

  useEffect(() => {
    const el = ref.current;
    if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (el.getBoundingClientRect().top < window.innerHeight) return; // already on screen: no show
    setReveal("pending");
    const seen = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setReveal("shown");
        seen.disconnect();
      },
      { threshold: 0.15 },
    );
    seen.observe(el);
    return () => seen.disconnect();
  }, []);

  return (
    <div ref={ref} className={styles.stack} data-reveal={reveal}>
      {groups.map(({ group, tools }, g) => (
        <div key={group} className={styles.group} style={{ "--delay": `${g * 70}ms` } as React.CSSProperties}>
          <h3>
            {group} <span className={styles.count}>{tools.length}</span>
          </h3>
          <ul>
            {tools.map((tool, i) => (
              <li key={tool} style={{ "--delay": `${g * 70 + 160 + Math.min(i, 12) * 24}ms` } as React.CSSProperties}>
                {tool}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
