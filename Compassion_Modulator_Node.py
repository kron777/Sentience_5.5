```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique compassion event IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

# --- Asyncio Imports for LLM calls ---
import asyncio
import aiohttp
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
    CompassionState = ROSMsgFallback
    SufferingUpdate = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    CompassionState = ROSMsgFallback
    SufferingUpdate = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback


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
            'compassion_modulator_node': {
                'default_compassion_level': 0.5,
                'suffering_update_interval': 0.5,
                'llm_compassion_threshold': 0.6,  # Salience to trigger LLM for nuanced compassion
                'ethical_compassion_bias': 0.3,  # Bias toward compassionate modulation
                'recent_context_window_s': 30.0,  # Longer window for suffering history
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 10.0
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


class CompassionModulatorNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'compassion_modulator_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "compassion_log.db")
        self.default_compassion_level = self.params.get('default_compassion_level', 0.5)
        self.suffering_update_interval = self.params.get('suffering_update_interval', 0.5)
        self.llm_compassion_threshold = self.params.get('llm_compassion_threshold', 0.6)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 30.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.3)

        # Sensory placeholders (e.g., detect suffering via emotional cues in sound/vision)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters (for nuanced compassion assessment)
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 10.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Compassion Modulator Node online, nurturing empathetic and supportive modulation.")

        # --- Asyncio Setup ---
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()
        self._async_session = None
        self.active_llm_task: Optional[asyncio.Task] = None

        # --- Initialize SQLite database ---
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS compassion_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                agent_id TEXT,
                suffering_score REAL,
                compassion_level REAL,
                llm_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_compassion_timestamp ON compassion_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.suffering_map: Dict[str, float] = {}  # agent_id -> suffering_score [0..1]
        self.compassion_level: float = self.default_compassion_level
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=20)  # Queue for suffering updates
        self.modulation_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns
        self.cumulative_suffering_salience = 0.0  # To trigger LLM for nuanced compassion

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_compassion_state = None
        self.pub_error_report = None
        self.sub_suffering_updates = None
        self.sub_directives = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_compassion_state = rospy.Publisher('/compassion_state', CompassionState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.sub_suffering_updates = rospy.Subscriber('/suffering_updates', SufferingUpdate, self.suffering_update_callback)
            self.sub_directives = rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.directive_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.suffering_update_interval), self.process_pending_updates)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_compassion_state()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing compassion (e.g., detect suffering cues)."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on compassion (e.g., emotional cues from sound/vision)
            if sensor_type == 'vision':
                self.pending_updates.append({'agent_id': 'observed_agent', 'suffering_score': random.uniform(0.1, 0.6), 'source': 'vision_cue'})
            elif sensor_type == 'sound':
                self.pending_updates.append({'agent_id': 'auditory_agent', 'suffering_score': random.uniform(0.2, 0.7), 'source': 'emotional_tone'})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'agent_id': 'user', 'suffering_score': random.uniform(0.3, 0.8), 'source': 'user_expression'})
            self.cumulative_suffering_salience = min(1.0, self.cumulative_suffering_salience + 0.2)  # Sensory adds salience
            _log_debug(self.node_name, f"{sensor_type} input updated compassion context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.process_pending_updates()
            time.sleep(self.suffering_update_interval)

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    # --- Asyncio Thread Management ---
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    async def _create_async_session(self):
        _log_info(self.node_name, "Creating aiohttp ClientSession...")
        self._async_session = aiohttp.ClientSession()
        _log_info(self.node_name, "aiohttp ClientSession created.")

    async def _close_async_session(self):
        if self._async_session:
            _log_info(self.node_name, "Closing aiohttp ClientSession...")
            await self._async_session.close()
            self._async_session = None
            _log_info(self.node_name, "aiohttp ClientSession closed.")

    def _shutdown_async_loop(self):
        if self._async_loop and self._async_thread.is_alive():
            _log_info(self.node_name, "Shutting down asyncio loop...")
            future = asyncio.run_coroutine_threadsafe(self._close_async_session(), self._async_loop)
            try:
                future.result(timeout=5.0)
            except asyncio.TimeoutError:
                _log_warn(self.node_name, "Timeout waiting for async session to close.")
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            self._async_thread.join(timeout=5.0)
            if self._async_thread.is_alive():
                _log_warn(self.node_name, "Asyncio thread did not shut down gracefully.")
            _log_info(self.node_name, "Asyncio loop shut down.")

    # --- Core Compassion Modulation Logic ---
    async def update_suffering_async(self, agent_id: str, suffering_score: float, context: Dict[str, Any] = None) -> float:
        """Update suffering score and modulate compassion asynchronously, with LLM if salient."""
        suffering_score = max(0.0, min(suffering_score, 1.0))
        self.suffering_map[agent_id] = suffering_score

        # Update salience for LLM trigger
        self.cumulative_suffering_salience += suffering_score * 0.5
        self.cumulative_suffering_salience = min(1.0, self.cumulative_suffering_salience)

        # Ethical compassion bias: Always increase compassion slightly for high suffering
        if suffering_score > self.ethical_compassion_bias:
            self.compassion_level = min(1.0, self.compassion_level + 0.05)

        # LLM for nuanced compassion if salient
        if self.cumulative_suffering_salience >= self.llm_compassion_threshold:
            _log_info(self.node_name, f"Using LLM for compassionate modulation (Salience: {self.cumulative_suffering_salience:.2f}).")
            llm_output = await self._assess_compassion_llm(agent_id, suffering_score, context)
            if llm_output:
                self.compassion_level = float(llm_output.get('compassion_level', self.compassion_level))
                reasoning = llm_output.get('reasoning', 'LLM assessed compassion.')
            else:
                reasoning = "Fallback compassion adjustment."
        else:
            # Simple update
            if not self.suffering_map:
                self.compassion_level = self.default_compassion_level
            else:
                avg_suffering = sum(self.suffering_map.values()) / len(self.suffering_map)
                self.compassion_level = min(1.0, max(self.default_compassion_level, avg_suffering * 1.5))
            reasoning = "Simple compassion modulation based on average suffering."

        # Log and publish
        sensory_snapshot = json.dumps(self.sensory_data)
        self._log_compassion_modulation(agent_id, suffering_score, self.compassion_level, reasoning, sensory_snapshot)
        self.publish_compassion_state()
        self.cumulative_suffering_salience = 0.0  # Reset after analysis

        return self.compassion_level

    async def _assess_compassion_llm(self, agent_id: str, suffering_score: float, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use LLM to assess compassionate modulation nuancedly."""
        prompt_text = f"""
        Assess the appropriate compassion level for the robot to modulate its behavior toward agent '{agent_id}' with suffering score {suffering_score}.
        Context: {json.dumps(context or {}, indent=2)}

        Consider compassionate principles: Prioritize empathetic, supportive responses without overwhelming the agent.
        Output JSON:
        {{
            "compassion_level": number (0.0-1.0, higher = more compassionate),
            "reasoning": "explanation of modulation, favoring compassionate bias"
        }}
        """
        response_schema = {
            "type": "object",
            "properties": {
                "compassion_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string"}
            },
            "required": ["compassion_level", "reasoning"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.2, max_tokens=150)
        if not llm_output_str.startswith("Error:"):
            try:
                return json.loads(llm_output_str)
            except json.JSONDecodeError:
                self._report_error("LLM_PARSE_ERROR", "Failed to parse compassion response.", 0.8)
        return None

    # --- Dynamic Input Methods ---
    def update_suffering(self, agent_id: str, suffering_score: float, context: Dict[str, Any] = None) -> float:
        """Wrapper for async update."""
        if self.ros_enabled and ROS_AVAILABLE:
            # In ROS mode, queue
            self.pending_updates.append({'agent_id': agent_id, 'suffering_score': suffering_score, 'context': context})
            return self.compassion_level
        else:
            # Dynamic: Run async
            try:
                return asyncio.run_coroutine_threadsafe(self.update_suffering_async(agent_id, suffering_score, context), self._async_loop).result(timeout=2.0)
            except asyncio.TimeoutError:
                _log_warn(self.node_name, "Compassion update timed out.")
                return self.compassion_level

    def suffering_update_callback(self, msg: Any):
        """ROS callback for suffering updates."""
        fields_map = {'data': ('', 'update_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        update_data = json.loads(data.get('update_data', '{}'))
        self.update_suffering(update_data.get('agent_id'), update_data.get('suffering_score'), update_data.get('context'))

    def directive_callback(self, msg: Any):
        """ROS callback for directives influencing compassion."""
        fields_map = {'data': ('', 'directive_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        directive_data = json.loads(data.get('directive_data', '{}'))
        if directive_data.get('type') == 'compassion_adjust':
            self.update_suffering(directive_data.get('agent_id'), directive_data.get('suffering_score'))

    # --- Publishing and Logging ---
    def publish_compassion_state(self):
        """Publish compassion state (ROS or log)."""
        state_data = {
            'timestamp': str(self._get_current_time()),
            'compassion_level': self.compassion_level,
            'suffering_map': self.suffering_map,
            'history_length': len(self.modulation_history)
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_compassion_state:
            try:
                if hasattr(CompassionState, 'data'):
                    self.pub_compassion_state.publish(String(data=json.dumps(state_data)))
                else:
                    compassion_msg = CompassionState(data=json.dumps(state_data))
                    self.pub_compassion_state.publish(compassion_msg)
                _log_debug(self.node_name, f"Published compassion state: {self.compassion_level:.2f}")
            except Exception as e:
                self._report_error("PUBLISH_COMPASSION_ERROR", f"Failed to publish compassion state: {e}", 0.7)
        else:
            _log_info(self.node_name, f"Dynamic compassion state: {state_data}")

    def _log_compassion_modulation(self, agent_id: str, suffering_score: float, compassion_level: float, reasoning: str, sensory_snapshot: str):
        """Log compassion modulation to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO compassion_log (id, timestamp, agent_id, suffering_score, compassion_level, llm_reasoning, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), agent_id, suffering_score,
                compassion_level, reasoning, json.dumps({'agent_id': agent_id}), sensory_snapshot
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Logged compassion modulation for {agent_id} (Level: {compassion_level:.2f}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to log compassion modulation: {e}", 0.9)

    def process_pending_updates(self, event: Any = None):
        """Process pending updates in dynamic or timer mode."""
        if self.pending_updates:
            update_data = self.pending_updates.popleft()
            self.update_suffering_async(update_data['agent_id'], update_data['suffering_score'], update_data.get('context'))

    def get_compassion_level(self) -> float:
        return self.compassion_level

    def summary(self) -> Dict[str, Any]:
        return {
            "suffering_map": self.suffering_map,
            "compassion_level": self.compassion_level,
            "history_length": len(self.modulation_history),
            "pending_updates": len(self.pending_updates)
        }

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down CompassionModulatorNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self._shutdown_async_loop()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Compassion Modulator Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = CompassionModulatorNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.update_suffering("agent_A", 0.7)
            node.update_suffering("agent_B", 0.4)
            node.update_suffering("agent_C", 0.9)
            print(node.summary())
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
