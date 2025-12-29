#!/usr/bin/env python3
import json
import time
import threading
from typing import Dict, List, Optional
from collections import deque

import rospy
from std_msgs.msg import String

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VisualizationNode")


class VisualizationNode:
    """
    VisualizationNode
    ------------------
    Purpose:
    - Aggregate monitoring + prediction + safety outputs
    - Maintain a rolling time window of metrics
    - Publish compact visualization-ready summaries (no heavy plotting in-node)
    - Optional background plotting hook (off by default)

    Design principles:
    - Non-blocking callbacks
    - Bounded memory
    - ROS-safe (no matplotlib calls in callback thread)
    """

    def __init__(self):
        self.node_name = "visualization_node"

        # --- Parameters ---
        self.max_points: int = rospy.get_param("~max_points", 300)
        self.publish_interval: float = rospy.get_param("~publish_interval", 1.0)
        self.enable_plotting: bool = rospy.get_param("~enable_plotting", False)

        # --- Internal buffers ---
        self.data_points: deque = deque(maxlen=self.max_points)
        self.last_publish_time: float = 0.0

        # --- Publishers ---
        self.pub_summary = rospy.Publisher(
            "/visualization/summary", String, queue_size=10
        )
        self.pub_stream = rospy.Publisher(
            "/visualization/stream", String, queue_size=10
        )

        # --- Subscribers ---
        rospy.Subscriber("/monitoring_output", String, self.callback)
        rospy.Subscriber("/prediction_output", String, self.callback)
        rospy.Subscriber("/system_safety_output", String, self.callback)

        # --- Optional plotting thread ---
        self._shutdown_flag = threading.Event()
        if self.enable_plotting:
            self._plot_thread = threading.Thread(
                target=self._plotting_loop, daemon=True
            )
            self._plot_thread.start()

        logger.info(f"{self.node_name}: initialized")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            node = payload.get("node", "unknown")
            confidence = float(payload.get("confidence", 0.0))
            status = payload.get("status", "unknown")

            entry = {
                "timestamp": time.time(),
                "node": node,
                "status": status,
                "confidence": confidence,
                "raw": payload,
            }

            self.data_points.append(entry)
            self.pub_stream.publish(json.dumps(entry))

            self._maybe_publish_summary()

        except Exception as e:
            logger.error(f"{self.node_name}: callback error: {e}")

    # ------------------------------------------------------------------
    # Summary logic
    # ------------------------------------------------------------------
    def _maybe_publish_summary(self) -> None:
        now = time.time()
        if now - self.last_publish_time < self.publish_interval:
            return

        self.last_publish_time = now
        summary = self._build_summary()

        if summary:
            self.pub_summary.publish(json.dumps(summary))
            logger.debug(f"{self.node_name}: summary published")

    def _build_summary(self) -> Optional[Dict]:
        if not self.data_points:
            return None

        latest = self.data_points[-1]
        confidences = [dp["confidence"] for dp in self.data_points]

        summary = {
            "timestamp": time.time(),
            "latest_node": latest["node"],
            "latest_status": latest["status"],
            "latest_confidence": latest["confidence"],
            "confidence_avg": sum(confidences) / len(confidences),
            "confidence_min": min(confidences),
            "confidence_max": max(confidences),
            "sample_count": len(self.data_points),
        }
        return summary

    # ------------------------------------------------------------------
    # Optional plotting (off by default)
    # ------------------------------------------------------------------
    def _plotting_loop(self):
        import matplotlib.pyplot as plt

        logger.info(f"{self.node_name}: plotting thread started")
        while not self._shutdown_flag.is_set():
            try:
                if len(self.data_points) < 2:
                    time.sleep(2.0)
                    continue

                timestamps = [dp["timestamp"] for dp in self.data_points]
                confidences = [dp["confidence"] for dp in self.data_points]

                plt.figure(figsize=(10, 5))
                plt.plot(timestamps, confidences, marker="o")
                plt.xlabel("Time")
                plt.ylabel("Confidence")
                plt.title("System Confidence Over Time")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("/tmp/system_confidence.png")
                plt.close()

            except Exception as e:
                logger.error(f"{self.node_name}: plotting error: {e}")

            time.sleep(5.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self):
        self._shutdown_flag.set()
        logger.info(f"{self.node_name}: shutdown complete")

    def run(self):
        rospy.spin()
        self.shutdown()


if __name__ == "__main__":
    try:
        rospy.init_node("visualization_node", anonymous=False)
        node = VisualizationNode()
        node.run()
    except Exception as e:
        rospy.logerr(f"visualization_node: fatal error: {e}")
