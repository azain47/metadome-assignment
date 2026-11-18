// Run: npm test (node --test with built-in TypeScript type stripping).
import { test } from "node:test";
import assert from "node:assert/strict";
import { sessionRows, shortId, shortTime } from "./sessions.ts";

// Local-time dates so the assertions hold in any TZ.
const now = new Date(2026, 10, 17, 15, 30);
const at = (...a: [number, number, number, number, number]) => new Date(...a).getTime() / 1000;

test("shortTime: HH:MM today, MM-DD HH:MM this year, date only for older years", () => {
  assert.equal(shortTime(at(2026, 10, 17, 9, 5), now), "09:05");
  assert.equal(shortTime(at(2026, 10, 16, 23, 59), now), "11-16 23:59");
  assert.equal(shortTime(at(2025, 0, 2, 8, 0), now), "2025-01-02");
  assert.equal(shortTime(undefined, now), "–");
});

test("shortId keeps the first UUID block and readable ids whole", () => {
  assert.equal(shortId("cff64521-1c2c-4527-b41e-e0e56c87e4fd"), "cff64521…");
  assert.equal(shortId("shoot-20261117-1504-2"), "shoot-20261117-1504-2");
  assert.equal(shortId("x".repeat(30)), `${"x".repeat(21)}…`);
});

test("sessionRows: newest first, never-run sessions last as empty, locations from shots then run, current marker", () => {
  const rows = sessionRows(
    [
      { id: "old", lastUpdateTime: at(2026, 10, 15, 12, 0), state: { run: { status: "done", location: "alps" }, shots: { alps: {} } } },
      { id: "fresh", lastUpdateTime: at(2026, 10, 17, 15, 4), state: {} },
      {
        id: "mid",
        lastUpdateTime: at(2026, 10, 17, 14, 20),
        state: { run: { status: "awaiting_approval", location: "norway" }, shots: { alps: {}, ladakh: {} } },
      },
    ],
    "mid",
    now,
  );
  assert.deepEqual(
    rows.map((r) => [r.id, r.time, r.status, r.empty, r.locations.join(","), r.current]),
    [
      ["mid", "14:20", "awaiting approval", false, "alps,ladakh,norway", true],
      ["old", "11-15 12:00", "done", false, "alps", false],
      ["fresh", "15:04", "empty", true, "", false],
    ],
  );
});
