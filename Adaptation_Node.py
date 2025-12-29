#!/usr/bin/env python3
import logging
import json
import os
import sys
import time
import threading
from typing import Dict, Any, Optional, Callable
from collections import deque

# Optional ROS Integration
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("adaptation_node")

def _log(level, msg):
    getattr(logger, level.lower())(msg)

# ---------------- Node ----------------
class AdaptationNode:
    def __init__(
        self,
        config_path: Optional[str] = None,
        ros_enabled: bool = False,
        decision_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.node_name = "adaptation_node"
        self.ros_enabled = ros_enabled

        # --- Config ---
        self.config = self._load_config(config_path)
        self.current_strategy = self.config["default_strategy"]
        self.adjustment_factor = self.config["default_adjustment_factor"]

        # --- Stability controls (NEW) ---
        self.strategy_inertia = 0.6                 # <<< MOD >>>
        self.confidence_decay_rate = 0.05           # <<< MOD >>>
        self.severity_decay_rate = 0.05             # <<< MOD >>>
        self.last_strategy_change = time.time()

        self.strategy_history = deque(maxlen=20)
        self.outcome_history = deque(maxlen=30)     # <<< MOD >>>

        self.confidence_threshold = self.config["confidence_threshold"]
        self.low_confidence_threshold = self.config["low_confidence_threshold"]
        self.execution_interval = self.config["execution_interval"]

        # --- Sensory ---
        self.sensory_data = {
            "vision": {"data": None, "timestamp": 0.0},
            "sound": {"data": None, "timestamp": 0.0},
            "instructions": {"data": None, "timestamp": 0.0},
        }

        self.decision_callback = decision_callback
        self.incoming_queue = deque()

        # --- Runtime ---
        self._shutdown_flag = threading.Event()
        self._poll_thread = threading.Thread(
            target=self._dynamic_poll_loop, daemon=True
        )
        self._poll_thread.start()

        _log("info", f"AdaptationNode online (strategy='{self.current_strategy}')")

    # ---------------- Config ----------------
    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        if path:
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "default_strategy": "balanced",
            "default_adjustment_factor": 0.1,
            "confidence_threshold": 0.6,
            "low_confidence_threshold": 0.3,
            "execution_interval": 0.2,
            "strategies": {
                "optimized": {"adjustment_factor": 0.05, "priority_shift": "high"},
                "balanced": {"adjustment_factor": 0.1, "priority_shift": "medium"},
                "conservative": {"adjustment_factor": 0.2, "priority_shift": "low"},
            },
        }

    # ---------------- Decay & Learning ----------------
    def _decay_inputs(self, confidence: float, severity: float):
        # <<< MOD >>> prevent persistent pressure
        confidence = max(0.0, confidence - self.confidence_decay_rate)
        severity = max(0.0, severity - self.severity_decay_rate)
        return confidence, severity

    def _strategy_success_rate(self) -> float:
        if not self.outcome_history:
            return 0.5
        return sum(o["success"] for o in self.outcome_history) / len(self.outcome_history)

    # ---------------- Adaptation Logic ----------------
    def _trigger_adaptation(self, input_data: Dict[str, Any], source: str):
        try:
            confidence = input_data.get("confidence", 0.5)
            severity = input_data.get("severity", 0.0)

            confidence, severity = self._decay_inputs(confidence, severity)

            sensory_influence = self._assess_sensory_influence()
            adjusted_confidence = min(1.0, confidence + sensory_influence)

            now = time.time()
            time_since_change = now - self.last_strategy_change

            # <<< MOD >>> inertia guard
            if time_since_change < 2.0:
                return

            success_rate = self._strategy_success_rate()

            if severity > 0.7:
                new_strategy = "conservative"
            elif adjusted_confidence > self.confidence_threshold and success_rate > 0.6:
                new_strategy = "optimized"
            elif adjusted_confidence < self.low_confidence_threshold:
                new_strategy = "conservative"
            else:
                new_strategy = "balanced"

            if new_strategy != self.current_strategy:
                if random.random() > self.strategy_inertia:
                    self._update_strategy(new_strategy, input_data)

        except Exception as e:
            _log("error", f"Adaptation error: {e}")

    def _update_strategy(self, strategy: str, input_data: Dict[str, Any]):
        strategies = self.config["strategies"]
        if strategy not in strategies:
            return

        self.current_strategy = strategy
        self.adjustment_factor = strategies[strategy]["adjustment_factor"]
        self.last_strategy_change = time.time()

        self.strategy_history.append(
            {
                "timestamp": time.time(),
                "strategy": strategy,
                "confidence": input_data.get("confidence", 0.5),
            }
        )

        _log("info", f"Strategy → {strategy}")

    # ---------------- Sensory ----------------
    def _assess_sensory_influence(self) -> float:
        influence = 0.0
        now = time.time()
        for info in self.sensory_data.values():
            if info["timestamp"] > now - 5.0:
                data = info["data"]
                if isinstance(data, dict):
                    influence += min(0.2, data.get("severity", 0.0) * 0.1)
        return influence

    # ---------------- Apply ----------------
    def apply_adaptation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        try:
            strategies = self.config["strategies"]
            shift = strategies[self.current_strategy]["priority_shift"]

            if decision.get("ethical_score", 1.0) < 0.7:
                decision["priority"] = "low"
            elif decision.get("priority") != "critical":
                decision["priority"] = shift

            decision["confidence"] = max(
                0.0,
                min(1.0, decision.get("confidence", 0.5) + self.adjustment_factor),
            )

            decision["adaptation"] = {
                "strategy": self.current_strategy,
                "factor": self.adjustment_factor,
                "timestamp": time.time(),
            }

            return decision
        except Exception as e:
            _log("error", f"Apply adaptation failed: {e}")
            return decision

    # ---------------- Runtime ----------------
    def _dynamic_poll_loop(self):
        while not self._shutdown_flag.is_set():
            if self.incoming_queue:
                decision = self.incoming_queue.popleft()
                adapted = self.apply_adaptation(decision)
                if self.decision_callback:
                    self.decision_callback(adapted)
            time.sleep(self.execution_interval)

    def record_outcome(self, success: bool):
        # <<< MOD >>> learning hook
        self.outcome_history.append({"success": success})

    def shutdown(self):
        self._shutdown_flag.set()
        _log("info", "AdaptationNode shutdown")

# ---------------- Main ----------------
if __name__ == "__main__":
    node = AdaptationNode()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
