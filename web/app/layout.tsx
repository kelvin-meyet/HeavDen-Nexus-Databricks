import type { Metadata } from "next";
import { Atkinson_Hyperlegible, Bricolage_Grotesque } from "next/font/google";

import { Shell } from "@/components/Shell";

import "./globals.css";

// Body and numbers: Atkinson Hyperlegible, designed so characters can't be confused
// (1/7, 8/3, 0/O), which matters when a number is a vital sign.
const body = Atkinson_Hyperlegible({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-body",
  display: "swap",
});

// Headings: Bricolage Grotesque, a lively grotesque with character, for titles and figures.
const display = Bricolage_Grotesque({
  subsets: ["latin"],
  weight: ["500", "700", "800"],
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: "HeavDen Nexus",
  description:
    "Deterioration risk, ward analytics and an assistant for a fictional hospital network. All data is synthetic.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${body.variable} ${display.variable}`}>
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
