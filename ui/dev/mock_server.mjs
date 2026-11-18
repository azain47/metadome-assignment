// Dev mock of the agent HTTP API (ADK api_server shape + /img + /api/health). No deps.
// Usage: node dev/mock_server.mjs [port=8000] [stepMs=600]
import http from "node:http";
import { randomUUID } from "node:crypto";

const PORT = Number(process.argv[2] || 8000);
const STEP_MS = Number(process.argv[3] || 600);
const SESS = /^\/apps\/trueshot\/users\/demo\/sessions(?:\/([^/]+))?$/;
const sessions = new Map(); // id -> {id, appName, userId, state, events, lastUpdateTime, cursor}

const now = () => Date.now() / 1000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const json = (res, code, body) => {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
};
const readBody = (req) =>
  new Promise((ok) => {
    let b = "";
    req.on("data", (c) => (b += c));
    req.on("end", () => ok(b ? JSON.parse(b) : {}));
  });
const publicSession = ({ cursor, ...s }) => s;

// ---- scripted run -------------------------------------------------------
// Steps are partial events; author/nodeInfo follow ADK 2.14 (graph function nodes author
// "trueshot", LLM nodes author = agent name). A step with `pause: <interruptId>` is an
// adk_request_input call: the stream ends there and the run waits for a functionResponse.
function makeShot(state) {
  const job = `job-${Math.floor(Math.random() * 900 + 100)}`;
  const loc = "coastal_road";
  const steps = [];
  let attempt = 1;
  let gallery = [...(state.gallery || [])];
  let spend = { llm_usd: 0, image_usd: 0, total_usd: 0, cap_usd: 5, ...(state.spend || {}) };
  let final = null;
  const path = (node) => `trueshot@1/${node}@${attempt}`;
  const run = (node, status = "running") => ({ run: { status, node, job_id: job, location: loc, attempt } });
  const addSpend = (llm, img) => {
    spend = { ...spend, llm_usd: spend.llm_usd + llm, image_usd: spend.image_usd + img };
    spend.total_usd = +(spend.llm_usd + spend.image_usd).toFixed(4);
    return { spend };
  };
  const addImg = (kind, scores = {}, verdict = null, reason = null) => {
    const id = `${job}-${kind}-${attempt}-${gallery.length}`;
    const item = { id, job_id: job, location: loc, attempt, kind, key: `jobs/${job}/${loc}/a${attempt}/${kind}-${gallery.length}.png`, scores, verdict, reason };
    if (kind === "final") final = item;
    gallery = [...gallery, item];
    return { gallery };
  };
  const tool = (agent, node, name, args, response) => {
    const id = `adk-${randomUUID()}`;
    steps.push({ author: agent, nodeInfo: { path: path(node) }, content: { role: "model", parts: [{ functionCall: { id, name, args } }] } });
    steps.push({ author: agent, nodeInfo: { path: path(node) }, content: { role: "user", parts: [{ functionResponse: { id, name, response } }] }, delay: 2 });
  };
  const tsStep = (node, name, args, summary, delta = {}, status = "ok") => {
    const id = randomUUID();
    steps.push({ author: "trueshot", nodeInfo: { path: path(node) }, customMetadata: { ts_step: { id, node, name, status: "running", args } } });
    steps.push({ author: "trueshot", nodeInfo: { path: path(node) }, customMetadata: { ts_step: { id, node, name, status, args, summary, duration_ms: 1800 + Math.round(Math.random() * 900) } }, actions: { stateDelta: delta }, delay: 3 });
  };
  const noise = (delta) => steps.push({ nodeInfo: { path: "" }, actions: { stateDelta: delta, agentState: { x: 1 } }, delay: 0.2 });
  const say = (author, node, text, delta) => steps.push({ author, nodeInfo: { path: path(node) }, content: { role: author === "trueshot" ? "user" : "model", parts: [{ text }] }, actions: { stateDelta: delta } });
  const compose = () => {
    noise(run("compose"));
    tsStep("compose", "render_guides", { kinds: ["grey", "lines", "depth"] }, "3 guides rendered", addImg("guide"));
    tsStep("compose", "generate_plates", { n: 2, backend: "fake" }, "2 plates, best layout 0.91", addImg("plate", { layout: 0.91, horizon_err: 0.006 }));
    tool("compose", "compose", "render_car", { samples: 128, motion_blur: true }, { ok: true, summary: "car rendered 1536×1024 in 48 s" });
    steps.push({ actions: { stateDelta: addImg("car") }, nodeInfo: { path: "" }, delay: 0.2 });
  };
  const pause = () => {
    const id = `adk-${randomUUID()}`;
    const below = final.verdict !== "pass";
    const payload = { job_id: job, location: loc, final_key: final.key, scores: final.scores, verdict: final.verdict, reason: final.reason, attempt, below_threshold: below };
    const args = {
      interruptId: id,
      message: below ? `Best result for ${job} is below the bar (${final.reason}). Approve anyway, revise, or reject?` : `${job} at ${loc} passed all checks. Approve to export?`,
      payload,
      response_schema: { type: "object", properties: { decision: { enum: ["approve", "revise", "reject"] }, note: { type: "string" } }, required: ["decision"] },
    };
    steps.push({ author: "trueshot", nodeInfo: { path: path("approval") }, content: { role: "model", parts: [{ functionCall: { id, name: "adk_request_input", args } }] }, longRunningToolIds: [id], actions: { stateDelta: run("approval", "awaiting_approval") }, pause: id });
  };

  return {
    steps,
    start() {
      noise(run("plan"));
      say("plan", "plan", JSON.stringify({ job_id: job, shots: [{ location: loc, framing: "3/4 front, low", time: "golden hour", motion: true }], notes: "road wet near shoulder" }), { ...addSpend(0.01, 0), "app:product_facts": { paint: "from brief", wheels: "5-spoke" } });
      noise(run("setup"));
      tool("setup", "setup", "scene_load_car", { asset: "car.glb" }, { ok: true, summary: "imported 2.1M tris in 14.2 s" });
      tool("setup", "setup", "scene_set_camera", { focal_mm: 50, height_m: 0.9, yaw_deg: 35 }, { ok: true, summary: "camera placed" });
      tool("setup", "setup", "render_preview", { width: 768 }, { ok: true, key: "preview.png", summary: "preview rendered" });
      steps.push({ actions: { stateDelta: addImg("preview") }, nodeInfo: { path: "" }, delay: 0.2 });
      compose();
      tsStep("check", "checks", { checks: ["spokes", "ghost", "halo"] }, "spoke count 4 ≠ 5", addImg("composite", { spokes: 4, ghost: true, halo: 0.88 }, "fail", "spoke count 4, expected 5"), "error");
      say("trueshot", "check", "Attempt 1 failed the spoke check; retrying with a faster shutter.");
      attempt = 2;
      compose();
      tsStep("check", "checks", { checks: ["spokes", "ghost", "halo"] }, "all checks pass", { ...addImg("final", { spokes: 5, ghost: true, halo: 0.95 }, "pass"), ...addSpend(0.02, 0) });
      pause();
    },
    answer(decision, note) {
      if (decision === "approve") {
        const approved = [...(state["user:approved_shots"] || []), { job_id: job, key: final.key, note, at: new Date().toISOString() }];
        tsStep("finalize", "export_still", { job_id: job, key: final.key }, `exported ${job} with passes`, { "user:approved_shots": approved, ...run("finalize", "done") });
        say("report", "report", `Approved ${job}: stills and passes exported.`);
      } else if (decision === "revise") {
        attempt += 1;
        tsStep("repair", "apply_note", { note }, `replanned from note: ${note || "(none)"}`, run("repair"));
        compose();
        tsStep("check", "checks", { checks: ["spokes", "ghost", "halo"] }, "halo 0.81 below bar", addImg("final", { spokes: 5, ghost: true, halo: 0.81 }, "best_below_bar", "halo 0.81 < 0.9"));
        pause();
      } else {
        say("report", "report", `Rejected ${job}; nothing exported.`, run("report", "done"));
      }
    },
  };
}

async function runSse(req, res) {
  // ADK origin check: Origin must equal scheme://Host (verifies Vite changeOrigin:false).
  const origin = req.headers.origin;
  if (origin && origin !== `http://${req.headers.host}`) return json(res, 403, { detail: "origin not allowed" });
  const body = await readBody(req);
  const s = sessions.get(body.sessionId);
  if (!s) return json(res, 404, { detail: "Session not found" });
  const answer = body.newMessage?.parts?.find((p) => p.functionResponse?.name === "adk_request_input")?.functionResponse;
  let invocationId;
  if (answer) {
    // Resolving a RequestInput pause continues the SAME invocation.
    if (!s.cursor?.awaiting || s.cursor.awaiting !== answer.id) return json(res, 400, { detail: `no pending interrupt ${answer.id}` });
    invocationId = s.cursor.invocationId;
    s.events.push({ id: randomUUID(), invocationId, author: "user", timestamp: now(), content: body.newMessage });
    s.cursor.awaiting = null;
    s.cursor.shot.answer(answer.response?.decision, answer.response?.note || "");
  } else if (body.newMessage) {
    if (s.cursor?.awaiting) return json(res, 409, { detail: "run is awaiting approval" });
    invocationId = `e-${randomUUID()}`;
    s.events.push({ id: randomUUID(), invocationId, author: "user", timestamp: now(), content: body.newMessage });
    const shot = makeShot(s.state);
    shot.start();
    s.cursor = { invocationId, shot, i: 0, awaiting: null };
  } else if (body.invocationId && s.cursor?.invocationId === body.invocationId && !s.cursor.awaiting) {
    invocationId = body.invocationId;
  } else {
    return json(res, 400, { detail: "nothing to resume" });
  }
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
  let closed = false;
  res.on("close", () => {
    if (!res.writableFinished) {
      closed = true;
      console.log(`client disconnected; aborting run ${invocationId}`);
    }
  });
  const cur = s.cursor;
  const steps = cur.shot.steps;
  while (cur.i < steps.length && !closed) {
    const { delay = 1, pause, ...step } = steps[cur.i];
    await sleep(STEP_MS * delay);
    if (closed) break;
    const ev = { id: randomUUID(), invocationId, timestamp: now(), author: "trueshot", ...step, actions: { stateDelta: {}, ...step.actions } };
    Object.assign(s.state, ev.actions.stateDelta);
    s.events.push(ev);
    s.lastUpdateTime = ev.timestamp;
    cur.i++;
    if (process.env.MOCK_FAIL_AT && cur.i === Number(process.env.MOCK_FAIL_AT)) {
      res.write(`data: ${JSON.stringify({ error: "mock failure" })}\n\n`);
      break;
    }
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
    if (pause) {
      cur.awaiting = pause; // run suspended; the stream ends here
      break;
    }
  }
  if (cur.i >= steps.length && !cur.awaiting) s.cursor = null;
  res.end();
}

// ---- images ------------------------------------------------------------
const COLORS = { preview: "#4a5568", guide: "#777", plate: "#2f6f8f", car: "#7a2f2f", composite: "#8f6f2f", final: "#2f8f4f" };
function svg(key) {
  const kind = Object.keys(COLORS).find((k) => key.includes(`/${k}`)) || "preview";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1536" height="1024" viewBox="0 0 1536 1024">
<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#9cc3e6"/><stop offset=".45" stop-color="${COLORS[kind]}"/><stop offset="1" stop-color="#222"/></linearGradient></defs>
<rect width="1536" height="1024" fill="url(#g)"/><path d="M0 1024 L700 470 L836 470 L1536 1024Z" fill="#333"/>
<rect x="520" y="560" width="500" height="170" rx="60" fill="#c33"/><circle cx="640" cy="740" r="62" fill="#111"/><circle cx="910" cy="740" r="62" fill="#111"/>
<text x="40" y="110" font-size="96" font-family="sans-serif" fill="#fff">${kind}</text>
<text x="40" y="990" font-size="34" font-family="monospace" fill="#ddd">${key.replace(/[<&]/g, "")}</text></svg>`;
}

// ---- router ------------------------------------------------------------
http
  .createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const p = decodeURIComponent(url.pathname);
    try {
      const m = SESS.exec(url.pathname);
      if (m && req.method === "POST" && !m[1]) {
        const body = await readBody(req);
        const id = body.sessionId || randomUUID();
        if (sessions.has(id)) return json(res, 409, { detail: `Session already exists: ${id}` });
        const s = { id, appName: "trueshot", userId: "demo", state: body.state || {}, events: [], lastUpdateTime: now(), cursor: null };
        sessions.set(id, s);
        return json(res, 200, publicSession(s));
      }
      if (m && req.method === "GET" && !m[1]) return json(res, 200, [...sessions.values()].map(({ events, ...s }) => publicSession(s)));
      if (m && req.method === "GET") {
        const s = sessions.get(decodeURIComponent(m[1]));
        return s ? json(res, 200, publicSession(s)) : json(res, 404, { detail: "Session not found" });
      }
      if (p === "/run_sse" && req.method === "POST") return await runSse(req, res);
      if (p.startsWith("/img/_blob/")) {
        res.writeHead(200, { "Content-Type": "image/svg+xml", "Cache-Control": "max-age=3600" });
        return res.end(svg(p.slice(11)));
      }
      if (p.startsWith("/img/")) {
        res.writeHead(302, { Location: `/img/_blob/${url.pathname.slice(5)}` });
        return res.end();
      }
      if (p === "/api/health")
        return json(res, 200, {
          blender: { ok: true, detail: "mock blender 5.3" },
          storage: { ok: true, detail: "mock s3" },
          db: { ok: true, detail: "mock" },
          image_backend: { ok: true, name: "fake", model: "fake", detail: "offline" },
          llm: { ok: false, model: "none", detail: "no key (mock)" },
        });
      if (p === "/health") return json(res, 200, { status: "ok" });
      json(res, 404, { detail: "not found" });
    } catch (e) {
      json(res, 500, { detail: String(e) });
    }
  })
  .listen(PORT, "127.0.0.1", () => console.log(`mock agent on http://127.0.0.1:${PORT} (step ${STEP_MS} ms)`));
