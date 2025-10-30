```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

# --- Asyncio Imports (for potential future async operations) ---
import asyncio
import threading
from collections import deque

# Optional ROS Integration (for compatibility)
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
    # Placeholder for custom messages - use String or dict fallbacks
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    TheoryOfMindUpdate = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    TheoryOfMindUpdate = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts/utils.py' exists and contains parse_message_data and load_config
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    # Fallback implementations
    def parse_message_data(msg: Any, fields_map: Dict[str, tuple], node_name: str = "unknown_node") -> Dict[str, Any]:
        """
        Generic parser for messages (ROS String/JSON or plain dict). 
        """
        data: Dict[str, Any] = {}
        if hasattr(msg, 'data') and isinstance(getattr(msg, 'data', None), str):
            try:
                parsed_json = json.loads(msg.data)
                for key_in_msg, (default_val, target_key) in fields_map.items():
                    data[target_key] = parsed_json.get(key_in_msg, default_val)
            except json.JSONDecodeError:
                _log_error(node_name, f"Could not parse message data as JSON: {msg.data}")
                for key_in_msg, (default_val, target_key) in fields_map.items():
                    data[target_key] = default_val
        elif isinstance(msg, dict):
            for key_in_msg, (default_val, target_key) in fields_map.items():
                data[target_key] = msg.get(key_in_msg, default_val)
        else:
            # Fallback: treat as object with attributes
            for key_in_msg, (default_val, target_key) in fields_map.items():
                data[target_key] = getattr(msg, key_in_msg, default_val)
        return data

    def load_config(node_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Fallback config loader: returns hardcoded defaults.
        """
        _log_warn(node_name, f"Using hardcoded default configuration as '{config_path}' could not be loaded.")
        return {
            'db_root_path': '/tmp/sentience_db',
            'default_log_level': 'INFO',
            'ros_enabled': False,
            'theory_of_mind_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate ToM (e.g., empathetic agent models)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            }
        }.get(node_name, {})  # Return node-specific or empty dict


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class TheoryOfMindNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'theory_of_mind_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "theory_of_mind_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing ToM compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.tom_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for ToM logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS theory_of_mind_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                agent_id TEXT,
                beliefs_json TEXT,
                desires_json TEXT,
                intentions_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Theory of Mind Node online, modeling others with compassionate and empathetic insight.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_tom_update = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_tom_update = rospy.Publisher('/theory_of_mind_update', TheoryOfMindUpdate, queue_size=10)
            rospy.Subscriber('/add_agent', String, self.add_or_update_agent_callback)
            rospy.Subscriber('/predict_behavior', String, self.predict_behavior_callback)
            rospy.Timer(rospy.Duration(4.0), self._periodic_update_and_publish)  # Periodic broadcast
        else:
            # Dynamic mode: Start polling thread for simulated inputs
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing ToM compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory input of other agents (e.g., vision detects person -> update agent model)
            if sensor_type == 'vision':
                if 'person' in str(processed).lower():
                    self.pending_updates.append({
                        'type': 'update_agent', 'data': {
                            'agent_id': 'person1', 'beliefs': ['believes robot is friendly'], 'desires': ['wants assistance'], 'intentions': ['will ask for help']
                        }
                    })
            elif sensor_type == 'sound':
                if 'voice' in str(processed).lower():
                    self.pending_updates.append({
                        'type': 'predict_behavior', 'data': {'agent_id': 'speaker', 'query': 'predict based on speech'}
                    })
            elif sensor_type == 'instructions':
                self.pending_updates.append({
                    'type': 'add_agent', 'data': {'agent_id': 'instructor', 'beliefs': [], 'desires': [], 'intentions': []}
                })
            # Compassionate bias: If distress in sound, bias toward empathetic agent model
            if 'distress' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated ToM context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_agent_update()
            self._simulate_behavior_prediction()
            self._periodic_update_and_publish()
            time.sleep(4.0)

    def _simulate_agent_update(self):
        """Simulate agent update in non-ROS mode."""
        agent_id = 'sim_person'
        self.pending_updates.append({
            'type': 'update_agent', 'data': {
                'agent_id': agent_id, 'beliefs': ['believes in cooperation'], 'desires': ['wants peace'], 'intentions': ['will collaborate']
            }
        })
        _log_debug(self.node_name, f"Simulated agent update for {agent_id}")

    def _simulate_behavior_prediction(self):
        """Simulate behavior prediction in non-ROS mode."""
        self.pending_updates.append({'type': 'predict_behavior', 'data': {'agent_id': 'sim_person'}})
        _log_debug(self.node_name, "Simulated behavior prediction")

    # --- Core Theory of Mind Logic ---
    def add_or_update_agent(self, agent_id: str, beliefs: List[str], desires: List[str], intentions: List[str]):
        """Add or update agent model with compassionate bias toward empathy."""
        # Compassionate bias: Ensure models include empathetic elements
        if self.ethical_compassion_bias > 0.1 and 'belief' in str(beliefs) and 'empathy' not in beliefs:
            beliefs.append("believes in mutual understanding")
        self.agents[agent_id] = {
            "beliefs": beliefs,
            "desires": desires,
            "intentions": intentions
        }
        _log_info(self.node_name, f"Updated agent '{agent_id}' model.")
        self._log_agent_update(agent_id, self.agents[agent_id])

    def get_agent_model(self, agent_id: str) -> Dict[str, Any]:
        """Get agent model."""
        return self.agents.get(agent_id, {})

    def predict_agent_behavior(self, agent_id: str) -> str:
        """Predict agent behavior with compassionate bias toward positive outcomes."""
        agent = self.agents.get(agent_id)
        if not agent:
            return "Unknown behavior"
        # Simple rule: if strong intention exists, predict that action
        if agent["intentions"]:
            prediction = f"Likely to {agent['intentions'][0]}"
        elif agent["desires"]:
            prediction = f"Probably seeking {agent['desires'][0]}"
        else:
            prediction = "Behavior unclear"
        # Compassionate bias: Add note for empathetic interpretation
        if self.ethical_compassion_bias > 0.1:
            prediction += f" (Empathetic note: Approach with compassion. Bias: {self.ethical_compassion_bias})"
        _log_debug(self.node_name, f"Predicted behavior for '{agent_id}': {prediction}")
        return prediction

    def _log_agent_update(self, agent_id: str, agent_data: Dict[str, Any]):
        """Log agent update to DB with sensory snapshot."""
        try:
            sensory_snapshot = json.dumps(self.sensory_data)
            self.cursor.execute('''
                INSERT INTO theory_of_mind_log (id, timestamp, agent_id, beliefs_json, desires_json, intentions_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), agent_id,
                json.dumps(agent_data['beliefs']), json.dumps(agent_data['desires']), json.dumps(agent_data['intentions']), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log agent update for '{agent_id}': {e}")

    def add_or_update_agent_callback(self, msg: Any):
        """Handle incoming add/update agent directive."""
        fields_map = {'data': ('', 'agent_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        agent_data = json.loads(data.get('agent_data', '{}'))
        agent_id = agent_data.get('agent_id', '')
        beliefs = agent_data.get('beliefs', [])
        desires = agent_data.get('desires', [])
        intentions = agent_data.get('intentions', [])
        self.add_or_update_agent(agent_id, beliefs, desires, intentions)

    def predict_behavior_callback(self, msg: Any):
        """Handle incoming predict behavior directive."""
        fields_map = {'data': ('', 'prediction_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        prediction_data = json.loads(data.get('prediction_data', '{}'))
        agent_id = prediction_data.get('agent_id', '')
        prediction = self.predict_agent_behavior(agent_id)
        self.publish_prediction(agent_id, prediction)

    def _periodic_update_and_publish(self):
        """Periodic update and publishing of agent models."""
        if self.agents:
            summary = self.summary()
            self.publish_summary(summary)

    def publish_prediction(self, agent_id: str, prediction: str):
        """Publish prediction (ROS or log)."""
        event = {
            'agent_id': agent_id,
            'prediction': prediction,
            'timestamp': time.time()
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_tom_update:
            if hasattr(TheoryOfMindUpdate, 'data'):
                self.pub_tom_update.publish(String(data=json.dumps(event)))
            else:
                update_msg = TheoryOfMindUpdate(data=json.dumps(event))
                self.pub_tom_update.publish(update_msg)
        else:
            # Dynamic mode: Log
            _log_info(self.node_name, f"Predicted for '{agent_id}': {prediction}")

    def publish_summary(self, summary: Dict[str, Any]):
        """Publish summary (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_tom_update:
                if hasattr(TheoryOfMindUpdate, 'data'):
                    self.pub_tom_update.publish(String(data=json.dumps(summary)))
                else:
                    summary_msg = TheoryOfMindUpdate(data=json.dumps(summary))
                    self.pub_tom_update.publish(summary_msg)
            else:
                _log_info(self.node_name, f"ToM Summary: {json.dumps(summary)}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish ToM summary: {e}")

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary with compassionate insights."""
        compassionate_summary = {}
        for agent_id, agent in self.agents.items():
            compassionate_summary[agent_id] = agent.copy()
            # Compassionate bias: Add empathetic note to model
            if self.ethical_compassion_bias > 0.1 and agent['desires']:
                compassionate_summary[agent_id]['compassionate_note'] = f"Empathize with desires: {agent['desires'][0]}. Bias: {self.ethical_compassion_bias}."
        return compassionate_summary

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down TheoryOfMindNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with simulated or actual ROS."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    self._simulate_agent_update()
                    self._simulate_behavior_prediction()
                    self._periodic_update_and_publish()
                    time.sleep(4.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Theory of Mind Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = TheoryOfMindNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate agent
            node.add_or_update_agent("Alice", ["thinks it will rain"], ["wants to stay dry"], ["will carry an umbrella"])
            prediction = node.predict_agent_behavior("Alice")
            print("Prediction for Alice:", prediction)
            time.sleep(1)
            print("Theory of Mind simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
