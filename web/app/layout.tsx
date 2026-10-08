import type { Metadata } from "next";
import { Atkinson_Hyperlegible } from "next/font/google";

import { Shell } from "@/components/Shell";

import "./globals.css";

// Atkinson Hyperlegible: designed for low-vision readers, so every character is unambiguous,
// which suits a tool where 1 vs 7 or 8 vs 3 on a vital sign matters.
const body = Atkinson_Hyperlegible({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "HeavDen Nexus",
  description:
    "Deterioration risk, ward analytics and an assistant for a fictional hospital network. All data is synthetic.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={body.variable}>
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
