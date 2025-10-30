```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

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
    NarrativeState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    NarrativeState = ROSMsgFallback


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
            'temporal_narrative_node': {
                'narrative_window': 10,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate narratives (e.g., resilient themes)
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


class TemporalNarrativeNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'temporal_narrative_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "temporal_narrative_log.db")
        self.narrative_window = self.params.get('narrative_window', 10)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing narrative compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.memory_events: Deque[Dict[str, Any]] = deque(maxlen=self.narrative_window)
        self.emotional_trace: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.goal_history: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates

        # Initialize SQLite database for narrative logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS temporal_narrative_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                narrative_theme TEXT,
                recent_memory_json TEXT,
                mood_trend TEXT,
                goal_trail_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Temporal Narrative Node online, weaving compassionate stories of experience.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_narrative = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_narrative = rospy.Publisher('/narrative_state', NarrativeState, queue_size=10)
            rospy.Subscriber('/memory_log', String, self.memory_callback)
            rospy.Subscriber('/emotional_state', String, self.emotion_callback)
            rospy.Subscriber('/goal_updates', String, self.goal_callback)
            rospy.Timer(rospy.Duration(6.0), self.generate_narrative)
        else:
            # Dynamic mode: Start polling thread for simulated data
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing narrative compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on narrative inputs
            if sensor_type == 'vision':
                self.pending_updates.append({'type': 'memory', 'data': {'event': processed.get('description', 'visual event'), 'timestamp': timestamp}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'type': 'emotion', 'data': {'mood': 'reflective' if random.random() < 0.5 else 'neutral', 'intensity': random.uniform(0.3, 0.7)}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'type': 'goal', 'data': {'goal': 'reflect', 'status': 'in_progress', 'timestamp': timestamp}})
            # Compassionate bias: If distress in sound, bias toward resilient narrative themes
            if 'distress' in str(processed):
                self._update_cumulative_salience(self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated narrative context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_memory_event()
            self._simulate_emotion()
            self._simulate_goal_update()
            self.generate_narrative()
            time.sleep(6.0)

    def _simulate_memory_event(self):
        """Simulate a memory event in non-ROS mode."""
        event_data = {'event': random.choice(['memory of success', 'memory of challenge', 'new learning']), 'timestamp': time.time()}
        self.memory_events.append(event_data)
        _log_debug(self.node_name, f"Simulated memory event: {event_data['event']}")

    def _simulate_emotion(self):
        """Simulate an emotion in non-ROS mode."""
        emotion_data = {'mood': random.choice(['neutral', 'joyful', 'reflective']), 'intensity': random.uniform(0.2, 0.8), 'timestamp': time.time()}
        self.emotional_trace.append(emotion_data)
        _log_debug(self.node_name, f"Simulated emotion: {emotion_data['mood']} (intensity {emotion_data['intensity']:.2f})")

    def _simulate_goal_update(self):
        """Simulate a goal update in non-ROS mode."""
        goal_data = {'goal': 'narrative_reflection', 'status': random.choice(['in_progress', 'completed', 'abandoned']), 'timestamp': time.time()}
        self.goal_history.append(goal_data)
        _log_debug(self.node_name, f"Simulated goal update: {goal_data}")

    # --- Core Narrative Logic ---
    def memory_callback(self, msg: Any):
        """Handle incoming memory log data."""
        fields_map = {'data': ('', 'memory_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        memory_data = json.loads(data.get('memory_data', '{}'))
        self.memory_events.append({
            "event": memory_data.get("event", "unknown"),
            "timestamp": memory_data.get("timestamp", time.time())
        })

    def emotion_callback(self, msg: Any):
        """Handle incoming emotional state data."""
        fields_map = {'data': ('', 'emotion_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        emotion_data = json.loads(data.get('emotion_data', '{}'))
        self.emotional_trace.append({
            "mood": emotion_data.get("mood", "neutral"),
            "intensity": emotion_data.get("mood_intensity", 0.0),
            "timestamp": emotion_data.get("timestamp", time.time())
        })

    def goal_callback(self, msg: Any):
        """Handle incoming goal update data."""
        fields_map = {'data': ('', 'goal_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        goal_data = json.loads(data.get('goal_data', '{}'))
        self.goal_history.append({
            "goal": goal_data.get("goal", ""),
            "status": goal_data.get("status", "in_progress"),
            "timestamp": goal_data.get("timestamp", time.time())
        })

    def generate_narrative(self):
        """Generate and publish narrative with compassionate bias toward resilient themes."""
        recent_events = list(self.memory_events)
        recent_goals = list(self.goal_history)
        mood_summary = self._summarize_emotion()

        # Compassionate bias: If recent events have challenges, bias toward resilience theme
        theme = self._infer_theme(recent_events, recent_goals)
        if any("challenge" in e["event"].lower() for e in recent_events) and self.ethical_compassion_bias > 0.1:
            theme = "resilience"  # Override to compassionate theme

        story = {
            "timestamp": time.time(),
            "narrative_theme": theme,
            "recent_memory": recent_events[-3:] if len(recent_events) >= 3 else recent_events,
            "mood_trend": mood_summary,
            "goal_trail": recent_goals[-3:] if len(recent_goals) >= 3 else recent_goals,
            "compassionate_bias": self.ethical_compassion_bias  # Include bias for awareness
        }

        # Log with sensory snapshot
        sensory_snapshot = json.dumps(self.sensory_data)
        self._log_narrative(story, sensory_snapshot)

        self.publish_narrative(story)
        _log_info(self.node_name, f"Generated narrative theme: {story['narrative_theme']}")

    def _summarize_emotion(self) -> str:
        """Summarize emotional trend with compassionate note."""
        if not self.emotional_trace:
            return "neutral"
        mood_counts = {}
        for e in self.emotional_trace:
            mood = e['mood']
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        dominant = max(mood_counts, key=mood_counts.get)
        # Compassionate bias: If trend negative, note resilience
        if dominant in ['anxious', 'sad'] and self.ethical_compassion_bias > 0.1:
            dominant += " (resilient reflection)"
        return dominant

    def _infer_theme(self, events: List[Dict[str, Any]], goals: List[Dict[str, Any]]) -> str:
        """Infer narrative theme with compassionate bias toward growth."""
        if not events:
            return "awakening"
        if any("conflict" in e["event"].lower() for e in events):
            return "resilience"
        if any(g["status"] == "completed" for g in goals):
            return "progress"
        if any(g["status"] == "abandoned" for g in goals):
            return "redirection"
        return "continuity"

    def _log_narrative(self, story: Dict[str, Any], sensory_snapshot: str):
        """Log narrative to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO temporal_narrative_log (id, timestamp, narrative_theme, recent_memory_json, mood_trend, goal_trail_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), story['narrative_theme'],
                json.dumps(story['recent_memory']), story['mood_trend'],
                json.dumps(story['goal_trail']), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log narrative: {e}")

    def publish_narrative(self, story: Dict[str, Any]):
        """Publish narrative (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_narrative:
                if hasattr(NarrativeState, 'data'):
                    self.pub_narrative.publish(String(data=json.dumps(story)))
                else:
                    narrative_msg = NarrativeState(data=json.dumps(story))
                    self.pub_narrative.publish(narrative_msg)
            else:
                # Dynamic mode: Log
                _log_info(self.node_name, f"Narrative: {json.dumps(story)}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish narrative: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down TemporalNarrativeNode.")
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
                    self._simulate_memory_event()
                    self._simulate_emotion()
                    self._simulate_goal_update()
                    self.generate_narrative()
                    time.sleep(6.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Temporal Narrative Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = TemporalNarrativeNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate events
            node.memory_callback({'data': json.dumps({'event': 'simulated memory', 'timestamp': time.time()})})
            time.sleep(1)
            print("Temporal narrative simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
