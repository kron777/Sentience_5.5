async def analyze_attention_async(self, event: Any = None):
    self._prune_history()
    self.attention_metrics["cycles"] += 1

    prev = self.current_attention_state.copy()
    focus_type, focus_target, priority_score = (
        prev["focus_type"],
        prev["focus_target"],
        prev["priority_score"]
    )

    llm_reasoning = "Rule-based attention."

    # ---- LLM AS ADVISOR ONLY ----
    if self.cumulative_attention_salience >= self.llm_attention_threshold_salience:
        self.attention_metrics["llm_invocations"] += 1
        proposal = await self._infer_attention_state_llm(
            self._compile_llm_context_for_attention()
        )
        if proposal:
            focus_type = proposal.get("focus_type", focus_type)
            focus_target = proposal.get("focus_target", focus_target)
            proposed_priority = float(proposal.get("priority_score", priority_score))
            llm_reasoning = proposal.get("llm_reasoning", "")

            # ---- GOVERNOR ----
            delta = proposed_priority - priority_score
            if abs(delta) > MAX_PRIORITY_DELTA:
                proposed_priority = priority_score + MAX_PRIORITY_DELTA * (1 if delta > 0 else -1)

            priority_score = max(0.0, min(1.0, proposed_priority))
        else:
            self.attention_metrics["fallback_used"] += 1
    else:
        self.attention_metrics["fallback_used"] += 1
        focus_type, focus_target, priority_score = self._apply_simple_attention_rules()

    # ---- FOCUS STABILITY CHECK ----
    now = self._get_current_time()
    signature = f"{focus_type}:{focus_target}"
    if signature != self.last_focus_signature:
        if now - self.last_attention_change_ts < MIN_FOCUS_HOLD_SECONDS:
            self.attention_metrics["blocked_shifts"] += 1
            return  # reject rapid oscillation
        self.last_attention_change_ts = now
        self.last_focus_signature = signature

    self.current_attention_state = {
        "timestamp": str(now),
        "focus_type": focus_type,
        "focus_target": focus_target,
        "priority_score": priority_score
    }

    self.save_attention_log(
        id=str(uuid.uuid4()),
        timestamp=self.current_attention_state["timestamp"],
        focus_type=focus_type,
        focus_target=focus_target,
        priority_score=priority_score,
        llm_reasoning=llm_reasoning,
        context_snapshot_json=json.dumps(self._compile_llm_context_for_attention()),
        sensory_snapshot_json=json.dumps(self.sensory_data)
    )

    self.publish_attention_state(None)
    self.cumulative_attention_salience = 0.0
