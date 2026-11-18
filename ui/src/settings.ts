// Run settings chosen in the header; sent as `user:settings` in every run's state delta.
import type { State } from "./api";

export const SETTINGS_KEY = "user:settings";
export const LEVELS = ["low", "medium", "high"] as const;
export type Level = (typeof LEVELS)[number];

export interface RunSettings {
  image_quality: Level;
  reasoning_effort: Level;
}

export const DEFAULT_SETTINGS: RunSettings = { image_quality: "high", reasoning_effort: "medium" };

const level = (v: unknown, fallback: Level): Level => (LEVELS.includes(v as Level) ? (v as Level) : fallback);

/** The session's stored settings, defaults for missing or invalid values. */
export function settingsFrom(state: State): RunSettings {
  const s = (state[SETTINGS_KEY] || {}) as Record<string, unknown>;
  return {
    image_quality: level(s.image_quality, DEFAULT_SETTINGS.image_quality),
    reasoning_effort: level(s.reasoning_effort, DEFAULT_SETTINGS.reasoning_effort),
  };
}
