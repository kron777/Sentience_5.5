# --- ADD near top-level (new) ---
MAX_PRIORITY_DELTA = 0.25
MIN_FOCUS_HOLD_SECONDS = 0.5


# --- ADD inside __init__ ---
self.last_attention_change_ts = 0.0
self.last_focus_signature = None
self.attention_metrics = {
    "llm_invocations": 0,
    "fallback_used": 0,
    "cycles": 0,
    "blocked_shifts": 0
}
