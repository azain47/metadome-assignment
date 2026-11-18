// Pure ADK event → ViewState reducer. No React, no HTTP: panels only read ViewState.
import type { AdkEvent, Content, Json, Session } from "./api";

/** ADK's RequestInput pause arrives as a call to this pseudo-tool; the answer is its functionResponse. */
export const REQUEST_INPUT = "adk_request_input";
export type Decision = "approve" | "revise" | "reject";

export interface Msg {
  id: string;
  role: "user" | "agent" | "error";
  author: string;
  text: string;
  json?: Json; // structured agent output (e.g. planner JSON), shown as a card
}

export interface FeedItem {
  id: string;
  kind: "llm" | "step" | "approval";
  node: string; // graph node name without @N
  attempt: number; // @N run count of the node (1 if unknown)
  name: string;
  args?: Json;
  result?: Json;
  summary?: string;
  status: "running" | "ok" | "error";
  t0?: number; // seconds
  durationMs?: number;
}

export interface FeedGroup {
  key: string;
  node: string;
  attempt: number;
  items: FeedItem[];
}

// Session-state keys from the UI contract (written by the agent's state.py).
export interface RunInfo {
  status: "idle" | "running" | "awaiting_approval" | "done" | "failed";
  node?: string;
  job_id?: string;
  location?: string;
  attempt?: number;
  /** Why the run failed (set by the agent's RunFailurePlugin). */
  error?: string;
}
export interface GalleryItem {
  id: string;
  job_id: string;
  location?: string;
  attempt?: number;
  kind: "preview" | "guide" | "plate" | "car" | "composite" | "final";
  key: string;
  scores?: Record<string, number | boolean>;
  verdict?: "pass" | "fail" | "best_below_bar" | null;
  reason?: string | null;
}
export interface Spend {
  llm_usd?: number;
  image_usd?: number;
  total_usd?: number;
  cap_usd?: number;
}
export interface SessionState {
  run?: RunInfo;
  gallery?: GalleryItem[];
  spend?: Spend;
  "user:approved_shots"?: Json[];
  [key: string]: Json;
}

export interface ApprovalPayload {
  /** "guide": the Blender guide before any image is made; absent: the final still. */
  stage?: "guide";
  job_id?: string;
  location?: string;
  final_key?: string;
  scores?: Record<string, number | boolean>;
  verdict?: "pass" | "best_below_bar" | null;
  reason?: string | null;
  attempt?: number;
  below_threshold?: boolean;
}
/** An adk_request_input call with no matching functionResponse yet. */
export interface PendingApproval {
  id: string; // interrupt id = functionCall id
  message: string;
  payload: ApprovalPayload;
}

export interface ViewState {
  msgs: Msg[];
  feed: FeedItem[];
  state: SessionState;
  lastInvocationId?: string;
  pending?: PendingApproval;
}

export const emptyView: ViewState = { msgs: [], feed: [], state: {} };

/** `trueshot@1/compose@2` → {node: "compose", attempt: 2}. */
export function parseNodePath(path?: string): { node: string; attempt: number } {
  const last = (path || "").split("/").filter(Boolean).pop() || "";
  const m = /^(.*?)(?:@(\d+))?$/.exec(last)!;
  return { node: m[1] || "agent", attempt: m[2] ? Number(m[2]) : 1 };
}

/** Planner-style structured text: a JSON object/array, optionally fenced. */
export function parseJsonText(text: string): Json | undefined {
  const t = text.trim().replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/, "$1");
  if (!/^[[{]/.test(t)) return undefined;
  try {
    return JSON.parse(t);
  } catch {
    return undefined;
  }
}

function summarize(resp: Json): string {
  if (resp == null) return "";
  if (typeof resp !== "object") return String(resp);
  const obj = resp as Record<string, unknown>; // narrowed to non-null object above
  for (const k of ["summary", "error", "message", "result"]) {
    const val = obj[k];
    if (typeof val === "string") return (k === "error" ? "error: " : "") + val;
  }
  return JSON.stringify(resp).slice(0, 160);
}

function upsert(feed: FeedItem[], id: string, patch: (old?: FeedItem) => FeedItem): FeedItem[] {
  const i = feed.findIndex((f) => f.id === id);
  if (i < 0) return [...feed, patch()];
  const next = feed.slice();
  next[i] = patch(feed[i]);
  return next;
}

export function reduce(v: ViewState, ev: AdkEvent): ViewState {
  if (ev.partial) return v;
  let { msgs, feed, state } = v;
  const evId = ev.id || `${ev.timestamp ?? ""}-${v.msgs.length}-${v.feed.length}`;
  const err = ev.error || ev.errorMessage || ev.errorCode;
  if (err) msgs = [...msgs, { id: `${evId}:err`, role: "error", author: "error", text: err }];
  const delta = ev.actions?.stateDelta;
  if (delta && Object.keys(delta).length) state = { ...state, ...delta };

  const { node, attempt } = parseNodePath(ev.nodeInfo?.path);
  let pending = v.pending;
  (ev.content?.parts || []).forEach((p, i) => {
    const pid = `${evId}:${i}`;
    // Who wrote it is `author`, never content.role (graph function nodes send role "user").
    if (p.text && !p.thought) {
      if (ev.author === "user") {
        const m: Msg = { id: pid, role: "user", author: "user", text: p.text };
        const local = msgs.findIndex((x) => x.id.startsWith("local-") && x.text === p.text);
        msgs = local >= 0 ? msgs.map((x, j) => (j === local ? m : x)) : [...msgs, m];
      } else if (p.text.trim()) {
        msgs = [...msgs, { id: pid, role: "agent", author: ev.author || "agent", text: p.text, json: parseJsonText(p.text) }];
      }
    }
    const fc = p.functionCall;
    if (fc?.name === REQUEST_INPUT) {
      const args: { message?: string; payload?: ApprovalPayload; interruptId?: string } =
        fc.args && typeof fc.args === "object" ? fc.args : {};
      const id = fc.id || args.interruptId || pid;
      const message = args.message || "Approval requested";
      pending = { id, message, payload: args.payload || {} };
      feed = upsert(feed, id, () => ({
        id, kind: "approval", node, attempt, name: "awaiting approval", args: args.payload, summary: message, status: "running", t0: ev.timestamp,
      }));
    } else if (fc) {
      const id = fc.id || pid;
      feed = upsert(feed, id, (old) => ({
        ...old, id, kind: "llm", node, attempt, name: fc.name, args: fc.args, status: old?.status ?? "running", t0: ev.timestamp,
      }));
    }
    const fr = p.functionResponse;
    if (fr?.name === REQUEST_INPUT) {
      const id = fr.id || pid;
      const r: { decision?: string; note?: string } = fr.response && typeof fr.response === "object" ? fr.response : {};
      const text = [r.decision || "answered", r.note].filter(Boolean).join(": ");
      if (pending?.id === id) pending = undefined;
      feed = upsert(feed, id, (old) => ({
        ...(old || { id, kind: "approval", node, attempt, name: "awaiting approval" }),
        name: "approval", result: fr.response, summary: text, status: "ok",
        durationMs: old?.t0 != null && ev.timestamp != null ? Math.round((ev.timestamp - old.t0) * 1000) : undefined,
      }));
      if (!msgs.some((m) => m.id === `answer:${id}`)) msgs = [...msgs, { id: `answer:${id}`, role: "user", author: "user", text }];
    } else if (fr) {
      const id = fr.id || pid;
      const failed = !!fr.response && typeof fr.response === "object" && "error" in fr.response;
      feed = upsert(feed, id, (old) => ({
        ...(old || { id, kind: "llm", node, attempt, name: fr.name }),
        result: fr.response,
        summary: summarize(fr.response),
        status: failed ? "error" : "ok",
        durationMs: old?.t0 != null && ev.timestamp != null ? Math.round((ev.timestamp - old.t0) * 1000) : undefined,
      }));
    }
  });

  const step = ev.customMetadata?.ts_step;
  if (step?.id) {
    const id = `step:${step.id}`;
    feed = upsert(feed, id, (old) => ({
      id,
      kind: "step",
      node: step.node ? parseNodePath(step.node).node : node,
      attempt: old?.attempt ?? attempt,
      name: step.name,
      args: step.args ?? old?.args,
      result: old?.result,
      summary: step.summary ?? old?.summary,
      status: step.status,
      t0: old?.t0 ?? ev.timestamp,
      durationMs: step.duration_ms ?? old?.durationMs,
    }));
  }
  return { msgs, feed, state, pending, lastInvocationId: ev.invocationId || v.lastInvocationId };
}

/** Rebuild the view from a stored session; the server's state is authoritative (includes user:/app:). */
export function fromSession(s: Pick<Session, "events" | "state">): ViewState {
  return { ...s.events.reduce(reduce, emptyView), state: s.state };
}

/** Optimistic user message, replaced when the server echoes the user event. */
export function addLocalMessage(v: ViewState, text: string): ViewState {
  return { ...v, msgs: [...v.msgs, { id: `local-${v.msgs.length}`, role: "user", author: "user", text }] };
}

/** Consecutive feed items of the same node run form one group. */
export function feedGroups(feed: FeedItem[]): FeedGroup[] {
  const groups: FeedGroup[] = [];
  for (const it of feed) {
    const g = groups[groups.length - 1];
    if (g && g.node === it.node && g.attempt === it.attempt) g.items.push(it);
    else groups.push({ key: it.id, node: it.node, attempt: it.attempt, items: [it] });
  }
  return groups;
}

/** The run was interrupted (page reload / stop) if state says running but no stream is open, or it failed;
 * either way its last invocation can be resumed. A RequestInput pause (`awaiting_approval`) is answered
 * through the approval card instead. */
export function canResume(v: ViewState, streaming: boolean): boolean {
  const status = v.state.run?.status;
  return !streaming && !v.pending && (status === "running" || status === "failed") && !!v.lastInvocationId;
}

export function textMessage(text: string): Content {
  return { role: "user", parts: [{ text }] };
}

/** The `newMessage` that answers a pending approval; the same invocation then continues. */
export function answerMessage(interruptId: string, decision: Decision, note = ""): Content {
  return {
    role: "user",
    parts: [{ functionResponse: { id: interruptId, name: REQUEST_INPUT, response: { decision, note } } }],
  };
}
