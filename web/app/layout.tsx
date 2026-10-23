import type { Metadata } from "next";
import { Atkinson_Hyperlegible, Schibsted_Grotesk } from "next/font/google";

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

// Headings and figures: Schibsted Grotesk, a newsroom grotesk built for confident headlines.
const display = Schibsted_Grotesk({
  subsets: ["latin"],
  weight: ["500", "700", "800"],
  variable: "--font-display",
  display: "swap",
});

const DESCRIPTION =
  "AI early warning for hospital wards: a predictive model that must beat the clinical standard, a data " +
  "platform on Azure Databricks and an AI assistant that shows its working. Built end to end by Kelvin " +
  "Ofori-Minta. All data is synthetic.";

export const metadata: Metadata = {
  title: "HeavDen Nexus",
  description: DESCRIPTION,
  // What LinkedIn, Slack and others show when the link is shared.
  openGraph: {
    title: "HeavDen Nexus: AI early warning for hospital wards",
    description: DESCRIPTION,
    type: "website",
  },
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
