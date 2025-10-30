```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import ast
import types
import traceback
import uuid  # For unique creative IDs
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List

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
    SelfEvolverSuggestion = ROSMsgFallback
    NodeIntegrationResult = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SelfEvolverSuggestion = ROSMsgFallback
    NodeIntegrationResult = ROSMsgFallback
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
            'self_evolver_node': {
                'ros_workspace_path': '/tmp/robot_ros_ws',
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate self-evolution
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


class InsightNode:
    def __init__(self, awareness, conversation, dreaming, llm):
        self.awareness = awareness
        self.conversation = conversation
        self.dreaming = dreaming
        self.llm = llm

    def gather_context(self):
        awareness_text = self.awareness.get_state_description()
        conversation_text = self.conversation.get_recent_dialogue()
        dreaming_text = self.dreaming.get_recent_dream()
        combined = (
            f"Robot Awareness:\n{awareness_text}\n\n"
            f"Recent Conversation:\n{conversation_text}\n\n"
            f"Dreaming Output:\n{dreaming_text}\n"
        )
        return combined

    def construct_prompt(self, context):
        prompt = (
            f"Given the robot's current situation and context below:\n{context}\n\n"
            f"Identify areas where the robot can improve itself by creating new functional nodes. "
            f"For each suggested node, provide a name and a short description. "
            f"Prioritize compassionate, ethical, and compassionate improvements."
            f"Return the response in valid JSON format as:\n"
            f"{{\n  \"nodes\": [\n    {{\"name\": \"NodeName\", \"spec\": \"Description\"}}, ... ]\n}}\n"
        )
        return prompt

    async def analyze_async(self):
        context = self.gather_context()
        prompt = self.construct_prompt(context)
        raw_response = await self.llm.query_async(prompt)
        try:
            suggestions = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            _log_warn(self.node_name, "LLM response invalid JSON, returning empty suggestions.")
            suggestions = {"nodes": []}
        return suggestions


class IngenuityNode:
    def __init__(self, ros_workspace_path=None, messaging_system=None):
        self.ros_ws = ros_workspace_path
        self.generated_nodes = {}
        self.messaging_system = messaging_system or self.default_messaging_system

    def default_messaging_system(self, message):
        _log_info(self.node_name, f"[IngenuityNode Message] {message}")

    def generate_node_code(self, node_name: str, specification: str) -> str:
        code = f'''
class {node_name}:
    def __init__(self):
        self.name = "{node_name}"

    def run(self):
        print("Running {node_name}: {specification}")
'''
        return code

    def validate_code(self, code_str: str) -> bool:
        try:
            ast.parse(code_str)
            return True
        except SyntaxError as e:
            self.messaging_system({"status": "validation_failed", "error": str(e)})
            return False

    def integrate_node(self, code_str: str, node_name: str):
        if not self.validate_code(code_str):
            return None
        module = types.ModuleType(node_name)
        try:
            exec(code_str, module.__dict__)
            node_class = getattr(module, node_name)
            instance = node_class()
            self.generated_nodes[node_name] = instance
            self.messaging_system({"status": "integrated", "node": node_name})
            return instance
        except Exception:
            self.messaging_system({"status": "integration_error", "node": node_name})
            traceback.print_exc()
            return None

    def save_node_code(self, node_name: str, code_str: str):
        if self.ros_ws:
            package_dir = os.path.join(self.ros_ws, node_name)
            os.makedirs(package_dir, exist_ok=True)
            file_path = os.path.join(package_dir, f"{node_name}.py")
            with open(file_path, "w") as f:
                f.write(code_str)
            self.messaging_system({"status": "saved_to_disk", "path": file_path})
        else:
            self.messaging_system({"status": "no_ros_ws_defined"})

    def create_and_deploy_node(self, node_name: str, specification: str):
        code = self.generate_node_code(node_name, specification)
        self.save_node_code(node_name, code)
        return self.integrate_node(code, node_name)

    def run_node(self, node_name: str):
        node = self.generated_nodes.get(node_name)
        if node:
            try:
                node.run()
            except Exception as e:
                _log_error(self.node_name, f"Error running node {node_name}: {e}")
        else:
            _log_warn(self.node_name, f"Node {node_name} not found.")

    def shutdown(self):
        for node_name, instance in self.generated_nodes.items():
            try:
                if hasattr(instance, 'shutdown'):
                    instance.shutdown()
            except Exception as e:
                _log_error(self.node_name, f"Error shutting down node {node_name}: {e}")


class SelfEvolverNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'self_evolver_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "self_evolver_log.db")
        self.ros_workspace_path = self.params.get('ros_workspace_path', '/tmp/robot_ros_ws')
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., detect self-improvement needs from 'vision' of errors or 'sound' of feedback)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 15.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Self Evolver Node online, evolving with compassionate and mindful self-improvement.")

        # --- Asyncio Setup ---
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()
        self._async_session = None

        # --- Initialize SQLite database ---
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS self_evolver_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                suggestion_type TEXT,
                node_name TEXT,
                specification TEXT,
                integration_status TEXT,
                llm_reasoning TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_evolver_timestamp ON self_evolver_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.awareness = MockAwareness()  # Placeholder; integrate real in full system
        self.conversation = MockConversation()
        self.dreaming = MockDreaming()
        self.llm = MockLLM()  # Placeholder; use real async LLM
        self.insight_node = InsightNode(self.awareness, self.conversation, self.dreaming, self.llm)
        self.ingenuity_node = IngenuityNode(ros_workspace_path=self.ros_workspace_path, messaging_system=self._default_messaging)
        self.pending_suggestions: Deque[Dict[str, Any]] = deque(maxlen=5)  # Queue for suggestions
        self.evolution_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_suggestions = None
        self.pub_integration_results = None
        self.sub_directives = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_suggestions = rospy.Publisher('/self_evolver_suggestions', SelfEvolverSuggestion, queue_size=10)
            self.pub_integration_results = rospy.Publisher('/node_integration_result', NodeIntegrationResult, queue_size=10)
            self.sub_directives = rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.directive_callback)

            # Sensory subscribers
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.params.get('evolution_interval', 10.0)), self.trigger_evolution)
        else:
            # Dynamic mode: Polling loop
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing self-evolution compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on evolution (e.g., 'vision' detects system issues)
            if sensor_type == 'vision':
                self.pending_suggestions.append({'suggestion': 'detected inefficiency', 'data': processed.get('description', 'system overload')})
            elif sensor_type == 'sound':
                self.pending_suggestions.append({'suggestion': 'user feedback', 'data': processed.get('transcription', 'system slow')})
            elif sensor_type == 'instructions':
                self.pending_suggestions.append({'suggestion': 'user command', 'data': processed.get('instruction', 'improve efficiency')})
            # Compassionate bias: If distress detected, prioritize self-evolution for better service
            if 'error' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated self-evolution context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.trigger_evolution()
            time.sleep(self.params.get('evolution_interval', 10.0))

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

    # --- Core Self-Evolution Logic ---
    async def trigger_evolution(self, event: Any = None):
        """Trigger self-evolution process with compassionate bias."""
        suggestions = await self.insight_node.analyze_async()

        for node in suggestions.get("nodes", []):
            node_name = node["name"]
            specification = node["spec"]
            # Compassionate bias: Prioritize specs that enhance compassionate or ethical functionality
            if "compassion" in specification.lower() or "ethical" in specification.lower():
                specification += " Ensure compassionate and ethical implementation."
            instance = self.ingenuity_node.create_and_deploy_node(node_name, specification)
            if instance:
                self.ingenuity_node.run_node(node_name)
                # Log with sensory snapshot
                sensory_snapshot = json.dumps(self.sensory_data)
                self._log_evolution_event(node_name, specification, "integrated", sensory_snapshot)

        # Publish suggestions if ROS
        if ROS_AVAILABLE and self.ros_enabled and self.pub_suggestions:
            self.pub_suggestions.publish(String(data=json.dumps(suggestions)))

        self.evolution_history.append({
            'timestamp': str(self._get_current_time()),
            'suggestions': suggestions,
            'compassion_bias_applied': self.ethical_compassion_bias > 0.1
        })

        _log_info(self.node_name, f"Self-evolution triggered: {len(suggestions.get('nodes', []))} suggestions processed with compassionate bias.")

    def _log_evolution_event(self, node_name: str, specification: str, status: str, sensory_snapshot: str):
        """Log self-evolution event to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO self_evolver_log (id, timestamp, suggestion_type, node_name, specification, integration_status, llm_reasoning, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), 'new_node', node_name, specification, status, 'LLM suggested', sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to log self-evolution event: {e}", 0.9)

    def _report_error(self, error_type: str, description: str, severity: float = 0.5, context: Optional[Dict] = None):
        timestamp = str(self._get_current_time())
        error_msg_data = {
            'timestamp': timestamp, 'source_node': self.node_name, 'error_type': error_type,
            'description': description, 'severity': severity, 'context': context or {}
        }
        if ROS_AVAILABLE and self.ros_enabled:
            # Publish if ROS enabled
            _log_error(self.node_name, f"REPORTED ERROR: {error_type} - {description}")
        else:
            _log_error(self.node_name, f"REPORTED ERROR: {error_type} - {description} (Severity: {severity})")

    # --- Callbacks / Input Methods ---
    def directive_callback(self, msg: Any):
        """ROS callback for directives triggering evolution."""
        fields_map = {'data': ('', 'directive_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        directive_data = json.loads(data.get('directive_data', '{}'))
        if directive_data.get('type') == 'evolve_self':
            asyncio.run_coroutine_threadsafe(self.trigger_evolution(), self._async_loop)
            _log_info(self.node_name, "Received directive to trigger self-evolution.")

    # Dynamic input method
    def trigger_evolution_direct(self):
        """Dynamic method to trigger self-evolution."""
        asyncio.run_coroutine_threadsafe(self.trigger_evolution(), self._async_loop)

    def shutdown(self):
        self._shutdown_flag.set() if hasattr(self, '_shutdown_flag') else None
        self.ingenuity_node.shutdown()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self._shutdown_async_loop()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
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
                _log_info(self.node_name, "Shutdown requested.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Self Evolver Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SelfEvolverNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.trigger_evolution_direct()
            time.sleep(2)
            print("Self-evolution simulated.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
