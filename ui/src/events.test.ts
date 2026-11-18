// Run: npm test (node --test with built-in TypeScript type stripping).
import { test } from "node:test";
import assert from "node:assert/strict";
import type { AdkEvent } from "./api.ts";
import { addLocalMessage, answerMessage, canResume, emptyView, feedGroups, fromSession, parseNodePath, reduce } from "./events.ts";

const P = (node: string) => ({ path: `trueshot@1/${node}` });

test("parseNodePath strips @N and returns it as attempt", () => {
  assert.deepEqual(parseNodePath("trueshot@1/compose@2"), { node: "compose", attempt: 2 });
  assert.deepEqual(parseNodePath(""), { node: "agent", attempt: 1 });
});

test("pairs functionCall with functionResponse by id, with duration", () => {
  let v = reduce(emptyView, {
    id: "e1", author: "setup", timestamp: 10, nodeInfo: P("setup@1"),
    content: { role: "model", parts: [{ functionCall: { id: "c1", name: "render_preview", args: { w: 768 } } }] },
  });
  assert.equal(v.feed[0].status, "running");
  v = reduce(v, {
    id: "e2", author: "setup", timestamp: 11.5, nodeInfo: P("setup@1"),
    content: { role: "user", parts: [{ functionResponse: { id: "c1", name: "render_preview", response: { summary: "ok!" } } }] },
  });
  assert.equal(v.feed.length, 1);
  assert.deepEqual([v.feed[0].status, v.feed[0].summary, v.feed[0].durationMs], ["ok", "ok!", 1500]);
  assert.equal(v.msgs.length, 0);
});

test("ts_step events upsert by id and group by node run", () => {
  const step = (status: "running" | "ok", extra = {}): AdkEvent => ({
    author: "trueshot", nodeInfo: P("compose@2"),
    customMetadata: { ts_step: { id: "s1", node: "compose", name: "plates", status, ...extra } },
  });
  let v = reduce(emptyView, step("running", { args: { n: 2 } }));
  v = reduce(v, step("ok", { summary: "2 plates", duration_ms: 2100 }));
  assert.equal(v.feed.length, 1);
  assert.deepEqual(v.feed[0], {
    id: "step:s1", kind: "step", node: "compose", attempt: 2, name: "plates", args: { n: 2 }, result: undefined,
    summary: "2 plates", status: "ok", t0: undefined, durationMs: 2100,
  });
  assert.deepEqual(feedGroups(v.feed).map((g) => [g.node, g.attempt]), [["compose", 2]]);
});

test("chat author comes from `author`, not content.role; JSON text becomes a card", () => {
  let v = addLocalMessage(emptyView, "hello");
  v = reduce(v, { id: "u", author: "user", content: { role: "user", parts: [{ text: "hello" }] } });
  v = reduce(v, { id: "g", author: "trueshot", content: { role: "user", parts: [{ text: "retrying" }] } });
  v = reduce(v, { id: "p", author: "plan", content: { role: "model", parts: [{ text: '```json\n{"shots":[1]}\n```' }] } });
  assert.deepEqual(v.msgs.map((m) => [m.role, m.author]), [["user", "user"], ["agent", "trueshot"], ["agent", "plan"]]);
  assert.equal(v.msgs[0].id, "u:0");
  assert.deepEqual(v.msgs[2].json, { shots: [1] });
});

test("contentless events still merge stateDelta; partials and errors", () => {
  let v = reduce(emptyView, { invocationId: "inv1", nodeInfo: { path: "" }, actions: { stateDelta: { run: { status: "running" } } } });
  v = reduce(v, { partial: true, content: { parts: [{ text: "x" }] } });
  v = reduce(v, { error: "boom" });
  assert.equal(v.state.run?.status, "running");
  assert.deepEqual(v.msgs.map((m) => [m.role, m.text]), [["error", "boom"]]);
  assert.equal(canResume(v, false), true);
  assert.equal(canResume(v, true), false);
});

test("a failed run is resumable; done or never-run is not", () => {
  const v = fromSession({ state: { run: { status: "failed" } }, events: [{ id: "a", author: "user", invocationId: "i", content: { parts: [{ text: "go" }] } }] });
  assert.equal(canResume(v, false), true);
  assert.equal(canResume({ ...v, state: { run: { status: "done" } } }, false), false);
  assert.equal(canResume({ ...v, lastInvocationId: undefined }, false), false);
});

test("fromSession rebuilds history and keeps server state authoritative", () => {
  const v = fromSession({
    state: { "user:looks": ["dusk"], run: { status: "done" } },
    events: [{ id: "a", author: "user", invocationId: "i", content: { parts: [{ text: "go" }] }, actions: { stateDelta: { run: { status: "running" } } } }],
  });
  assert.equal(v.state.run?.status, "done");
  assert.equal(v.msgs[0].text, "go");
  assert.equal(v.lastInvocationId, "i");
});

const pauseEvent: AdkEvent = {
  id: "pause", invocationId: "inv", author: "trueshot", timestamp: 100, nodeInfo: P("approval@2"),
  longRunningToolIds: ["int-1"],
  content: { role: "model", parts: [{ functionCall: { id: "int-1", name: "adk_request_input", args: {
    interruptId: "int-1", message: "Approve job-1?",
    payload: { job_id: "job-1", final_key: "jobs/job-1/final.png", scores: { spokes: 5 }, verdict: "pass", below_threshold: false },
  } } }] },
  actions: { stateDelta: { run: { status: "awaiting_approval" } } },
};

test("adk_request_input with no response is pending, shown as one approval row, not resumable", () => {
  const v = reduce(emptyView, pauseEvent);
  assert.deepEqual(v.pending, {
    id: "int-1", message: "Approve job-1?",
    payload: { job_id: "job-1", final_key: "jobs/job-1/final.png", scores: { spokes: 5 }, verdict: "pass", below_threshold: false },
  });
  assert.deepEqual(v.feed.map((f) => [f.kind, f.name, f.status]), [["approval", "awaiting approval", "running"]]);
  assert.equal(canResume(v, false), false);
});

test("a matching functionResponse resolves the pause (live echo and stored session dedupe)", () => {
  const answer: AdkEvent = { id: "ans", invocationId: "inv", author: "user", content: answerMessage("int-1", "revise", "lower camera") };
  let v = reduce(reduce(emptyView, pauseEvent), { author: "user", content: answer.content }); // local echo
  v = reduce(v, answer); // server echo of the same answer
  assert.equal(v.pending, undefined);
  assert.deepEqual(v.feed.map((f) => [f.kind, f.status, f.summary]), [["approval", "ok", "revise: lower camera"]]);
  assert.deepEqual(v.msgs.map((m) => [m.role, m.text]), [["user", "revise: lower camera"]]);
  // A response for a different interrupt leaves the pause pending.
  const other = reduce(reduce(emptyView, pauseEvent), { author: "user", content: answerMessage("int-0", "approve") });
  assert.equal(other.pending?.id, "int-1");
  // Rebuilt from a stored session: resolved.
  assert.equal(fromSession({ state: {}, events: [pauseEvent, answer] }).pending, undefined);
});

test("answerMessage builds the functionResponse newMessage", () => {
  assert.deepEqual(answerMessage("int-1", "approve", "ship it"), {
    role: "user",
    parts: [{ functionResponse: { id: "int-1", name: "adk_request_input", response: { decision: "approve", note: "ship it" } } }],
  });
});
