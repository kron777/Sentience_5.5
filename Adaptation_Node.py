```python
#!/usr/bin/env python3
import logging
import json
import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from collections import deque

# Optional ROS Integration (for compatibility)
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    rospy = None
    String = None
    class StringFallback:
        def __init__(self, data: str = ""):
            self.data = data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptation_node.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def _log_info(name: str, msg: str):
    logger.info(f"{name}: {msg}")

def _log_warn(name: str, msg: str):
    logger.warning(f"{name}: {msg}")

def _log_error(name: str, msg: str):
    logger.error(f"{name}: {msg}")

def _log_debug(name: str, msg: str):
    logger.debug(f"{name}: {msg}")


class AdaptationNode:
    def __init__(
        self,
        config_path: Optional[str] = None,
        ros_enabled: bool = False,
        decision_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.node_name = 'adaptation_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # Load config (fallback to defaults)
        self.config = self._load_config(config_path)
        self.current_strategy = self.config.get('default_strategy', 'balanced')
        self.adjustment_factor = self.config.get('default_adjustment_factor', 0.1)
        self.strategy_history = deque(maxlen=10)  # Track recent strategies for analysis
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.low_confidence_threshold = self.config.get('low_confidence_threshold', 0.3)
        self.execution_interval = self.config.get('execution_interval', 0.1)

        # Sensory/decision integration placeholders (dynamic)
        self.sensory_data: Dict[str, Any] = {
            'vision': {'data': None, 'timestamp': 0.0},
            'sound': {'data': None, 'timestamp': 0.0},
            'instructions': {'data': None, 'timestamp': 0.0}
        }
        self.decision_callback = decision_callback  # Optional external callback for adapted decisions

        # Publishers/Subscribers (ROS or dynamic queues)
        self.pub_adaptation_output = None
        self.sub_learning = None
        self.sub_monitoring = None
        self.sub_decisions = None  # New: For applying adaptations to incoming decisions
        self.incoming_queue: deque = deque()  # Dynamic queue for non-ROS inputs

        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_adaptation_output = rospy.Publisher('adaptation_output', String, queue_size=10)
            self.sub_learning = rospy.Subscriber('learning_output', String, self._ros_callback_learning)
            self.sub_monitoring = rospy.Subscriber('monitoring_output', String, self._ros_callback_monitoring)
            self.sub_decisions = rospy.Subscriber('decision_input', String, self._ros_callback_decision)  # New subscriber
            rospy.Timer(rospy.Duration(self.execution_interval), self._poll_and_adapt)
        else:
            # Dynamic mode: Thread for polling
            self._shutdown_flag = threading.Event()
            self._poll_thread = threading.Thread(target=self._dynamic_poll_loop, daemon=True)
            self._poll_thread.start()

        _log_info(self.node_name, f"AdaptationNode initialized with strategy '{self.current_strategy}' (ROS: {self.ros_enabled})")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load config from file or env vars, with fallbacks."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                _log_warn(self.node_name, f"Failed to load config from {config_path}: {e}")
        # Fallback defaults
        return {
            'default_strategy': 'balanced',
            'default_adjustment_factor': 0.1,
            'confidence_threshold': 0.5,
            'low_confidence_threshold': 0.3,
            'execution_interval': 0.1,
            'strategies': {
                'optimized': {'adjustment_factor': 0.05, 'priority_shift': 'high'},
                'balanced': {'adjustment_factor': 0.1, 'priority_shift': 'medium'},
                'conservative': {'adjustment_factor': 0.2, 'priority_shift': 'low'}
            }
        }

    def _create_sensory_callback(self, sensor_type: str) -> Callable:
        """Dynamic placeholder for sensory inputs influencing adaptation."""
        def callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            self.sensory_data[sensor_type] = {'data': processed, 'timestamp': timestamp}
            # Sensory anomalies could trigger conservative strategy
            if sensor_type == 'vision' and 'anomaly' in processed:
                self._trigger_adaptation({'type': 'sensory_alert', 'severity': processed['anomaly']})
        return callback

    # ROS Callbacks (wrappers)
    def _ros_callback_learning(self, data: String):
        try:
            suggestion = json.loads(data.data)
            self.receive_learning_suggestion(suggestion)
        except Exception as e:
            _log_error(self.node_name, f"Error processing learning data: {e}")

    def _ros_callback_monitoring(self, data: String):
        try:
            alert = json.loads(data.data)
            self.receive_monitoring_alert(alert)
        except Exception as e:
            _log_error(self.node_name, f"Error processing monitoring data: {e}")

    def _ros_callback_decision(self, data: String):
        try:
            decision = json.loads(data.data)
            adapted = self.apply_adaptation(decision)
            if self.decision_callback:
                self.decision_callback(adapted)
            else:
                # Publish back or log
                self._publish_adaptation({'type': 'adapted_decision', 'data': adapted})
        except Exception as e:
            _log_error(self.node_name, f"Error processing decision: {e}")

    # Dynamic Input Methods
    def receive_input(self, input_type: str, data: Dict[str, Any]):
        """Dynamic method to feed inputs (learning/monitoring/decisions)."""
        if input_type == 'learning':
            self.receive_learning_suggestion(data)
        elif input_type == 'monitoring':
            self.receive_monitoring_alert(data)
        elif input_type == 'decision':
            adapted = self.apply_adaptation(data)
            if self.decision_callback:
                self.decision_callback(adapted)
            else:
                self._publish_adaptation({'type': 'adapted_decision', 'data': adapted})
        else:
            _log_warn(self.node_name, f"Unknown input type: {input_type}")

    def receive_learning_suggestion(self, suggestion: Dict[str, Any]) -> None:
        _log_info(self.node_name, "Received learning suggestion")
        self._trigger_adaptation(suggestion, source='learning')

    def receive_monitoring_alert(self, alert: Dict[str, Any]) -> None:
        _log_info(self.node_name, "Received monitoring alert")
        self._trigger_adaptation(alert, source='monitoring')

    def _trigger_adaptation(self, input_data: Dict[str, Any], source: str = 'unknown') -> None:
        """Central trigger for strategy updates based on input."""
        try:
            confidence = input_data.get('confidence', 0.5)
            recommendation = input_data.get('recommendation', '')
            status = input_data.get('status', '')
            severity = input_data.get('severity', 0.0)

            # Incorporate sensory data if relevant
            sensory_influence = self._assess_sensory_influence(input_data)
            adjusted_confidence = min(1.0, confidence + sensory_influence)

            if status == 'alert' or severity > 0.7:
                self._update_strategy('conservative', input_data)
            elif recommendation == 'retrain_or_restart' or adjusted_confidence > self.confidence_threshold:
                self._update_strategy('optimized', input_data)
            elif adjusted_confidence < self.low_confidence_threshold:
                self._update_strategy('conservative', input_data)
            else:
                self._update_strategy('balanced', input_data)

            _log_info(self.node_name, f"Triggered adaptation from {source} (confidence: {adjusted_confidence:.2f})")
        except Exception as e:
            _log_error(self.node_name, f"Error in trigger_adaptation: {e}")

    def _assess_sensory_influence(self, input_data: Dict[str, Any]) -> float:
        """Assess if current sensory data warrants caution (e.g., anomalies)."""
        influence = 0.0
        current_time = time.time()
        for sensor, info in self.sensory_data.items():
            if info['timestamp'] > current_time - 5.0:  # Recent data
                data = info['data']
                if isinstance(data, dict) and ('anomaly' in data or 'risk' in data):
                    influence += data.get('severity', 0.0) * 0.2
        return influence

    def _update_strategy(self, new_strategy: str, input_data: Dict[str, Any]) -> None:
        """Update strategy with smoothing and history tracking."""
        try:
            strategies = self.config.get('strategies', {})
            if new_strategy in strategies:
                self.current_strategy = new_strategy
                self.adjustment_factor = strategies[new_strategy].get('adjustment_factor', self.adjustment_factor)
                self.strategy_history.append({
                    'strategy': new_strategy,
                    'timestamp': time.time(),
                    'trigger': input_data.get('type', 'unknown'),
                    'confidence': input_data.get('confidence', 0.5)
                })

            output = {
                'status': 'adapted',
                'strategy': self.current_strategy,
                'adjustment_factor': self.adjustment_factor,
                'history_length': len(self.strategy_history),
                'trigger': input_data.get('type', 'manual')
            }
            self._publish_adaptation(output)
            _log_info(self.node_name, f"Strategy updated to '{new_strategy}' (factor: {self.adjustment_factor:.2f})")
        except Exception as e:
            _log_error(self.node_name, f"Error updating strategy: {e}")

    def apply_adaptation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Apply current strategy to a decision, incorporating ethics/safety."""
        try:
            # Ethical/Dharma alignment: Bias toward non-harm
            if decision.get('ethical_score', 1.0) < 0.7:
                decision['priority'] = 'low'  # Conservative for low ethics
                decision['adjustment'] = max(self.adjustment_factor, 0.15)

            # Strategy-specific shifts
            strategies = self.config.get('strategies', {})
            shift = strategies.get(self.current_strategy, {}).get('priority_shift', 'medium')
            if decision.get('priority') != 'critical':  # Avoid overriding critical
                decision['priority'] = shift

            # Adjustment factor application (e.g., scale confidence or speed)
            if 'confidence' in decision:
                decision['confidence'] = max(0.0, min(1.0, decision['confidence'] + self.adjustment_factor))
            decision['adaptation_applied'] = {
                'strategy': self.current_strategy,
                'factor': self.adjustment_factor,
                'timestamp': time.time()
            }

            # Log adapted decision
            _log_info(self.node_name, f"Adapted decision with strategy '{self.current_strategy}': {json.dumps({k: v for k, v in decision.items() if k not in ['full_context']})}")
            return decision
        except Exception as e:
            _log_error(self.node_name, f"Error applying adaptation: {e}")
            return decision

    def _publish_adaptation(self, output: Dict[str, Any]) -> None:
        """Publish adaptation output (ROS or log/queue)."""
        try:
            output_str = json.dumps(output)
            if ROS_AVAILABLE and self.ros_enabled and self.pub_adaptation_output:
                self.pub_adaptation_output.publish(String(data=output_str))
            else:
                # Dynamic: Could emit to a callback or queue
                _log_info(self.node_name, f"Dynamic adaptation output: {output_str}")
        except Exception as e:
            _log_error(self.node_name, f"Error publishing adaptation: {e}")

    def _poll_and_adapt(self, event):
        """ROS timer callback: Poll for decisions in queue."""
        if self.incoming_queue:
            decision = self.incoming_queue.popleft()
            adapted = self.apply_adaptation(decision)
            if self.decision_callback:
                self.decision_callback(adapted)
            else:
                self._publish_adaptation({'type': 'adapted_decision', 'data': adapted})

    def _dynamic_poll_loop(self):
        """Dynamic polling loop for non-ROS mode."""
        while not self._shutdown_flag.is_set():
            self._poll_and_adapt(None)
            time.sleep(self.execution_interval)

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down AdaptationNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("AdaptationNode shutdown.")

    def run(self):
        """Run the node."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sentience Adaptation Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = AdaptationNode(config_path=args.config, ros_enabled=args.ros_enabled)
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
