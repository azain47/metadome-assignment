// Pure session-list rows for the header's sessions dropdown. No React, no HTTP.
import type { Session } from "./api";
import type { SessionState } from "./events";

export interface SessionRow {
  id: string;
  /** Display id: UUIDs cut to their first block, long ids ellipsized. */
  short: string;
  /** HH:MM today, MM-DD HH:MM this year, else YYYY-MM-DD. */
  time: string;
  /** state.run.status with spaces, or "empty" for a session that never ran. */
  status: string;
  /** Never ran: no state.run and no state.shots. Listed last, dimmed. */
  empty: boolean;
  /** state.shots keys, then state.run.location if not among them. */
  locations: string[];
  current: boolean;
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_ID = 22;

export function shortId(id: string): string {
  if (UUID.test(id)) return `${id.slice(0, 8)}…`;
  return id.length > MAX_ID ? `${id.slice(0, MAX_ID - 1)}…` : id;
}

const p2 = (n: number) => String(n).padStart(2, "0");

/** `seconds` is epoch seconds (ADK lastUpdateTime); formatted in local time relative to `now`. */
export function shortTime(seconds: number | undefined, now: Date): string {
  if (!seconds) return "–";
  const d = new Date(seconds * 1000);
  const hm = `${p2(d.getHours())}:${p2(d.getMinutes())}`;
  if (d.toDateString() === now.toDateString()) return hm;
  const md = `${p2(d.getMonth() + 1)}-${p2(d.getDate())}`;
  return d.getFullYear() === now.getFullYear() ? `${md} ${hm}` : `${d.getFullYear()}-${md}`;
}

export function sessionLocations(state: SessionState | undefined): string[] {
  const shots = state?.shots;
  const keys = shots && typeof shots === "object" && !Array.isArray(shots) ? Object.keys(shots) : [];
  const loc = state?.run?.location;
  return loc && !keys.includes(loc) ? [...keys, loc] : keys;
}

/** Sessions that ran, newest first by lastUpdateTime, then never-run ones; `current` marks the open session. */
export function sessionRows(list: Pick<Session, "id" | "state" | "lastUpdateTime">[], current: string, now: Date): SessionRow[] {
  return list
    .map((s) => {
      const state = s.state as SessionState | undefined;
      const locations = sessionLocations(state);
      const empty = !state?.run && !locations.length;
      const row: SessionRow = {
        id: s.id,
        short: shortId(s.id),
        time: shortTime(s.lastUpdateTime, now),
        status: empty ? "empty" : (state?.run?.status || "idle").replace(/_/g, " "),
        empty,
        locations,
        current: s.id === current,
      };
      return { row, t: s.lastUpdateTime || 0 };
    })
    .sort((a, b) => Number(a.row.empty) - Number(b.row.empty) || b.t - a.t)
    .map(({ row }) => row);
}
