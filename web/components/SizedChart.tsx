"use client";

import { useLayoutEffect, useRef, useState } from "react";

/**
 * Measures its own width and renders the chart at exact pixel sizes.
 * (Recharts' ResponsiveContainer sometimes measured too early inside grid layouts and drew
 * the chart tiny; measuring ourselves is reliable.)
 */
export function SizedChart({
  height,
  children,
}: {
  height: number;
  children: (width: number, height: number) => React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setWidth(Math.floor(el.getBoundingClientRect().width));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} style={{ width: "100%", height }}>
      {width > 0 && children(width, height)}
    </div>
  );
}

/** Legend labels in text colour (Recharts paints them in each series' colour by default). */
export function legendText(value: string) {
  return <span style={{ color: "var(--graphite)" }}>{value}</span>;
}
