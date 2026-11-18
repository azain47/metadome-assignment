// ADK api_server client: sessions, /run_sse stream (fetch + ReadableStream), health.
export const APP = "trueshot";
export const USER = "demo";
const SESS = `/apps/${APP}/users/${USER}/sessions`;

export type Json = unknown;
export type State = Record<string, Json>;

export interface Part {
  text?: string;
  thought?: boolean;
  functionCall?: { id?: string; name: string; args?: Json };
  functionResponse?: { id?: string; name: string; response?: Json };
}

/** A user message for /run_sse: text, or a functionResponse answering a RequestInput pause. */
export interface Content {
  role: "user";
  parts: Part[];
}

export interface TsStep {
  id: string;
  node: string;
  name: string;
  status: "running" | "ok" | "error";
  args?: Json;
  summary?: string;
  duration_ms?: number;
}

export interface AdkEvent {
  id?: string;
  invocationId?: string;
  author?: string;
  timestamp?: number;
  partial?: boolean;
  content?: { role?: string; parts?: Part[] };
  actions?: { stateDelta?: State };
  nodeInfo?: { path?: string };
  customMetadata?: { ts_step?: TsStep; [key: string]: Json };
  longRunningToolIds?: string[];
  errorCode?: string;
  errorMessage?: string;
  error?: string; // mid-stream server error line
}

export interface Session {
  id: string;
  state: State;
  events: AdkEvent[];
  /** Epoch seconds. */
  lastUpdateTime?: number;
}

async function check(res: Response): Promise<Response> {
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 500)}`);
  return res;
}

/** The server's reason from a `check` error ("409 {"detail": ...}"), else the message itself. */
export function refusal(message: string): string {
  try {
    return String(JSON.parse(message.slice(message.indexOf(" ") + 1)).detail ?? message);
  } catch {
    return message;
  }
}

export async function getSession(id: string): Promise<Session> {
  return (await check(await fetch(`${SESS}/${encodeURIComponent(id)}`))).json();
}

/** All of the user's sessions; `events` is empty in the list (getSession fetches them). */
export async function listSessions(): Promise<Session[]> {
  return (await check(await fetch(SESS))).json();
}

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `shoot-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

/** Creates `shoot-YYYYMMDD-HHMM`, suffixing -2, -3… if that id is taken. */
export async function createSession(): Promise<Session> {
  const base = stamp();
  let last: unknown;
  for (let i = 1; i <= 5; i++) {
    const sessionId = i === 1 ? base : `${base}-${i}`;
    const res = await fetch(SESS, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, state: {} }),
    });
    if (res.ok) return res.json();
    last = `${res.status} ${await res.text()}`;
    if (res.status >= 500) break;
  }
  throw new Error(`create session failed: ${last}`);
}

/** POST /run_sse and feed each `data:` JSON line to onEvent. Resolves when the stream ends
 * (run finished, or suspended at a RequestInput pause). `invocationId` alone resumes an interrupted run.
 * `stateDelta` is merged into session state before the run (ADK `state_delta`). */
export async function runSse(
  sessionId: string,
  input: { newMessage: Content } | { invocationId: string },
  stateDelta: State,
  onEvent: (ev: AdkEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const body = { appName: APP, userId: USER, sessionId, streaming: false, ...input, stateDelta };
  const res = await check(
    await fetch("/run_sse", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    }),
  );
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value.replace(/\r\n/g, "\n");
    let cut: number;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      const data = chunk
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

export interface Health {
  [k: string]: { ok: boolean; detail?: string; name?: string; model?: string };
}

export async function getHealth(): Promise<Health> {
  return (await check(await fetch("/api/health"))).json();
}

export const imgUrl = (key: string) => `/img/${key.split("/").map(encodeURIComponent).join("/")}`;
