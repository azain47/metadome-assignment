import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// changeOrigin:false keeps the browser's Host header, so ADK's origin check
// (Origin must equal scheme://Host) passes through the proxy.
// AGENT_URL points the proxy elsewhere (e.g. the mock on another port while the real agent holds 8000).
const target = process.env.AGENT_URL || "http://127.0.0.1:8000";
const proxy = Object.fromEntries(
  ["/apps", "/run_sse", "/img", "/api", "/health"].map((p) => [p, { target, changeOrigin: false }]),
);

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
});
