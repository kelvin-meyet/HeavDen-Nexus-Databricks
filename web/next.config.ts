import type { NextConfig } from "next";

// The browser only ever calls /api/*; Next forwards it to the FastAPI backend.
// Locally that's uvicorn on :8000; on Vercel, BACKEND_URL points at Render.
// The backend URL (and anything secret) never reaches the browser.
const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/:path*` }];
  },
};

export default config;
