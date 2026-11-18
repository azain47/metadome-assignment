import { useEffect, useState } from "react";
import { getHealth, type Health } from "../api";
import { LEVELS, type Level, type RunSettings } from "../settings";
import { SessionsMenu } from "./Sessions";
import type { PanelProps } from "./types";

const CHOICES: [key: keyof RunSettings, label: string][] = [
  ["image_quality", "quality"],
  ["reasoning_effort", "thinking"],
];

const LIGHTS: [key: string, label: string][] = [
  ["blender", "blender"],
  ["storage", "s3"],
  ["db", "db"],
  ["image_backend", "img"],
  ["llm", "llm"],
];

// 3-row block glyphs (▀ █ ▄); joined column-wise so rows always line up.
const GLYPHS: Record<string, string[]> = {
  t: ["▀█▀", " █ ", " ▀ "],
  r: ["█▀█", "█▀▄", "▀ ▀"],
  u: ["█ █", "█ █", "▀▀▀"],
  e: ["█▀▀", "█▀▀", "▀▀▀"],
  s: ["█▀▀", "▀▀█", "▀▀▀"],
  h: ["█ █", "█▀█", "▀ ▀"],
  o: ["█▀█", "█ █", "▀▀▀"],
};
const WORDMARK = [0, 1, 2].map((row) => [..."trueshot"].map((c) => GLYPHS[c][row]).join(" ")).join("\n");

/** Wordmark, session id, New, Sessions list, Open-by-id, run settings, and health from /api/health (polled every 10 s). */
export function Header(props: PanelProps) {
  const { sessionId, settings, streaming, actions } = props;
  const [health, setHealth] = useState<Health | null>(null);
  const [draft, setDraft] = useState("");
  useEffect(() => {
    const poll = () => getHealth().then(setHealth, () => setHealth(null));
    poll();
    const t = setInterval(poll, 10_000);
    return () => clearInterval(t);
  }, []);
  return (
    <header className="header">
      <pre className="wordmark" aria-label="trueshot">
        {WORDMARK}
      </pre>
      <div className="meta">
        <div className="row">
          <span className="dim">session</span>
          <span className="sid" title="session id">
            {sessionId || "none"}
          </span>
          <button onClick={actions.newSession} disabled={streaming} title="blank view; the first message creates the session">
            new
          </button>
          <SessionsMenu {...props} />
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (draft.trim()) actions.open(draft.trim());
              setDraft("");
            }}
          >
            <span className="dim">open&nbsp;</span>
            <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="session id ⏎" aria-label="open session id" />
          </form>
        </div>
        <div className="row settings">
          <span className="dim">applies to next message</span>
          {CHOICES.map(([k, label]) => (
            <label key={k}>
              <span className="dim">{label}&nbsp;</span>[
              <select value={settings[k]} onChange={(e) => actions.setSettings({ ...settings, [k]: e.target.value as Level })}>
                {LEVELS.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
              ]
            </label>
          ))}
        </div>
        <div className="row lights">
          {LIGHTS.map(([k, label]) => {
            const h = health?.[k];
            const detail = h ? [h.name, h.model, h.detail].filter(Boolean).join(" · ") : "unreachable";
            const cls = h ? (h.ok ? "ok" : "bad") : "unknown";
            return (
              <span key={k} className={`light ${cls}`} title={`${k}: ${detail}`}>
                <span className="mark">[{h ? (h.ok ? "ok" : "!!") : "??"}]</span> {label}
                {k === "image_backend" && h?.name ? ` ${h.name}` : ""}
              </span>
            );
          })}
        </div>
      </div>
    </header>
  );
}
