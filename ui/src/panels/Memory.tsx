import type { Json, State } from "../api";
import type { PanelProps } from "./types";

/** Long-term memory: the `user:*` and `app:*` state keys. */
export function memoryState(state: State): Record<string, Json> {
  return Object.fromEntries(Object.entries(state).filter(([k]) => k.startsWith("user:") || k.startsWith("app:")));
}

/** ASCII tree lines (`├─ key: value`) for nested state; scalars print as JSON. */
function tree(obj: object, prefix = ""): string[] {
  const entries = Object.entries(obj);
  return entries.flatMap(([k, v]: [string, Json], i) => {
    const last = i === entries.length - 1;
    const head = `${prefix}${last ? "└─ " : "├─ "}${k}`;
    if (v && typeof v === "object") {
      const kids = tree(v, prefix + (last ? "   " : "│  "));
      return kids.length ? [`${head}:`, ...kids] : [`${head}: ${Array.isArray(v) ? "[]" : "{}"}`];
    }
    return [`${head}: ${JSON.stringify(v)}`];
  });
}

/** Long-term memory as a tree (drawer tab). */
export function Memory({ view }: PanelProps) {
  const mem = memoryState(view.state);
  return <pre className="pane memory">{Object.keys(mem).length ? ["memory", ...tree(mem)].join("\n") : "memory\n└─ (empty)"}</pre>;
}
