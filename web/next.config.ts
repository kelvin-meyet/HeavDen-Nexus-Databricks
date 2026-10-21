import type { NextConfig } from "next";

// The browser only ever calls /api/*; the routes in app/api/ forward it to the FastAPI backend
// (lib/backend.ts), so the backend's address and its secret never reach the browser.
const config: NextConfig = {};

export default config;
