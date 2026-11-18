import { useEffect, useRef } from "react";

/** Keeps a scroll pane at its newest (bottom) content while the user hasn't scrolled up,
 * including when the pane itself shrinks (e.g. the approval card appears above it). */
export function usePinToBottom<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  useEffect(() => {
    const el = ref.current;
    const content = el?.firstElementChild;
    if (!el || !content) return;
    let pinned = true;
    const onScroll = () => {
      pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    };
    const stick = () => {
      if (pinned) el.scrollTop = el.scrollHeight;
    };
    const ro = new ResizeObserver(stick);
    ro.observe(el);
    ro.observe(content);
    el.addEventListener("scroll", onScroll);
    return () => {
      ro.disconnect();
      el.removeEventListener("scroll", onScroll);
    };
  }, []);
  return ref;
}
