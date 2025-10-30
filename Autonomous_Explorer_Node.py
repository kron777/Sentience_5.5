```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique exploration log IDs
import sys
import argparse
from datetime import datetime
from typing import Set, Dict, Any, Optional, Callable, List, Tuple
from collections import deque

# --- Asyncio Imports for LLM calls (if needed for novelty assessment) ---
import asyncio
import aiohttp
import threading

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
    ExplorationDecision = ROSMsgFallback
    CandidateStates = ROSMsgFallback
    SensoryQualia = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    ExplorationDecision = ROSMsgFallback
    CandidateStates = ROSMsgFallback
    SensoryQualia = ROSMsgFallback


# --- Import shared utility functions (renamed for generality) ---
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
        default_config = {
            'db_root_path': '/tmp/sentience_db',
            'default_log_level': 'INFO',
            'ros_enabled': False,  # Default to non-ROS for dynamic mode
            'autonomous_explorer_node': {
                'novelty_threshold': 0.5,
                'exploration_update_interval': 1.0,
                'llm_novelty_threshold': 0.7,  # Salience to trigger LLM for novelty
                'recent_context_window_s': 30.0,  # Longer window for exploration history
                'ethical_bias': 0.2,  # Bias toward safe/non-harmful states
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 15.0
            }
        }
        if node_name == "global":
            return default_config
        return default_config.get(node_name, {})


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class AutonomousExplorerNode:
    """
    Drives curiosity and autonomous exploration by seeking novelty, aligned with compassionate, non-harmful principles.
    Integrates sensory data, ethical checks, and optional LLM for sophisticated novelty assessment.
    """

    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'autonomous_explorer_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "exploration_log.db")
        self.novelty_threshold = self.params.get('novelty_threshold', 0.5)
        self.exploration_update_interval = self.params.get('exploration_update_interval', 1.0)
        self.llm_novelty_threshold = self.params.get('llm_novelty_threshold', 0.7)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 30.0)
        self.ethical_bias = self.params.get('ethical_bias', 0.2)  # Penalize potentially harmful states
        self.sensory_sources = self.params.get('sensory_inputs', {})

        # LLM Parameters (optional for advanced novelty)
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 15.0)
        self.use_llm_for_novelty = self.llm_novelty_threshold > 0  # Enable if threshold set

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Autonomous Explorer Node online, seeking novel horizons with mindful curiosity.")

        # --- Dynamic Sensory Placeholders ---
        self.sensory_data: Dict[str, Any] = {
            'vision': {'data': None, 'timestamp': 0.0},
            'sound': {'data': None, 'timestamp': 0.0},
            'instructions': {'data': None, 'timestamp': 0.0}
        }
        self.vision_callback = self._create_sensory_callback('vision')
        self.sound_callback = self._create_sensory_callback('sound')
        self.instructions_callback = self._create_sensory_callback('instructions')

        # --- Asyncio Setup (for LLM calls) ---
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
            CREATE TABLE IF NOT EXISTS exploration_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                chosen_state TEXT,
                novelty_score REAL,
                ethical_score REAL,
                llm_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_exploration_timestamp ON exploration_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.known_states: Set[str] = set()  # Use strings for hashability (JSON-serialized states)
        self.exploration_history: Dict[str, float] = {}  # state -> novelty score
        self.exploration_queue: deque = deque()  # Pending candidate states
        self.cumulative_novelty_salience = 0.0  # To trigger LLM

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_exploration_decision = None
        self.pub_error_report = None
        self.sub_candidates = None
        self.sub_sensory = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_exploration_decision = rospy.Publisher('/exploration_decision', ExplorationDecision, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.sub_candidates = rospy.Subscriber('/candidate_states', CandidateStates, self.candidate_states_callback)
            self.sub_sensory = rospy.Subscriber('/sensory_qualia', SensoryQualia, self.sensory_qualia_callback)  # For dynamic salience
            rospy.Timer(rospy.Duration(self.exploration_update_interval), self._run_exploration_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_callback(self, sensor_type: str):
        def callback(data: Any):
            timestamp = time.time()
            processed_data = data if isinstance(data, dict) else {'raw': str(data)}
            self.sensory_data[sensor_type] = {'data': processed_data, 'timestamp': timestamp}
            # Sensory novelty adds to salience
            if 'novel' in processed_data or random.random() < 0.3:  # Simulate
                self.cumulative_novelty_salience = min(1.0, self.cumulative_novelty_salience + 0.2)
            _log_debug(self.node_name, f"{sensor_type} input updated at {timestamp}")
        return callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_exploration_wrapper(None)
            time.sleep(self.exploration_update_interval)

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    # --- Asyncio Thread Management ---
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_until_complete(self._create_async_session())
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

    def _run_exploration_wrapper(self, event):
        """Wrapper to run exploration from timer or loop."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM novelty task active. Skipping cycle.")
            return

        if self.exploration_queue:
            candidates = self.exploration_queue.popleft()
            self.active_llm_task = asyncio.run_coroutine_threadsafe(
                self.explore_async(candidates), self._async_loop
            )
        else:
            _log_debug(self.node_name, "No candidate states in queue.")

    # --- Error Reporting Utility ---
    def _report_error(self, error_type: str, description: str, severity: float = 0.5, context: Optional[Dict] = None):
        timestamp = str(self._get_current_time())
        error_msg_data = {
            'timestamp': timestamp, 'source_node': self.node_name, 'error_type': error_type,
            'description': description, 'severity': severity, 'context': context or {}
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_error_report:
            try:
                self.pub_error_report.publish(String(data=json.dumps(error_msg_data)))
                rospy.logerr(f"{self.node_name}: REPORTED ERROR: {error_type} - {description}")
            except Exception as e:
                _log_error(self.node_name, f"Failed to publish error report: {e}")
        else:
            _log_error(self.node_name, f"REPORTED ERROR: {error_type} - {description} (Severity: {severity})")

    # --- LLM Call Function (for advanced novelty) ---
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.4, max_tokens: int = 150) -> str:
        if not self.use_llm_for_novelty:
            return "LLM disabled."
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Higher for creative novelty
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {'Content-Type': 'application/json'}

        if response_schema:
            prompt_text += "\n\nProvide the response in JSON format according to this schema:\n" + json.dumps(response_schema, indent=2)
            payload["messages"] = [{"role": "user", "content": prompt_text}]

        api_url = self.llm_base_url

        try:
            async with self._async_session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=self.llm_timeout), headers=headers) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                    return result['choices'][0]['message']['content']
                
                self._report_error("LLM_RESPONSE_EMPTY", "LLM response empty.", 0.5, {'prompt_snippet': prompt_text[:100]})
                return "Error: LLM response empty."
        except Exception as e:
            self._report_error("LLM_API_ERROR", f"LLM request failed: {e}", 0.9, {'url': api_url})
            return f"Error: {e}"

    # --- Utility to accumulate novelty salience ---
    def _update_cumulative_salience(self, score: float):
        self.cumulative_novelty_salience += score
        self.cumulative_novelty_salience = min(1.0, self.cumulative_novelty_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        current_time = self._get_current_time()
        # Prune exploration history (keep recent)
        to_remove = [state for state, ts in self.exploration_history.items() if current_time - ts > self.recent_context_window_s]
        for state in to_remove:
            del self.exploration_history[state]

    # --- Callbacks / Input Methods ---
    def candidate_states_callback(self, msg: Any):
        """ROS or dynamic callback for candidate states."""
        fields_map = {'timestamp': (str(self._get_current_time()), 'timestamp'), 'candidates_json': ('[]', 'candidates_json')}
        data = parse_message_data(msg, fields_map, self.node_name)
        try:
            candidates = json.loads(data.get('candidates_json', '[]'))
            self.exploration_queue.append(candidates)
            self._update_cumulative_salience(0.3)  # Candidates add salience
            _log_debug(self.node_name, f"Received {len(candidates)} candidate states.")
        except json.JSONDecodeError as e:
            self._report_error("JSON_DECODE_ERROR", f"Failed to parse candidates: {e}", 0.5)

    def sensory_qualia_callback(self, msg: Any):
        """Integrate sensory qualia for novelty hints."""
        fields_map = {'timestamp': (str(self._get_current_time()), 'timestamp'), 'salience_score': (0.0, 'salience_score')}
        data = parse_message_data(msg, fields_map, self.node_name)
        self._update_cumulative_salience(data.get('salience_score', 0.0) * 0.4)
        _log_debug(self.node_name, f"Sensory qualia added salience: {data.get('salience_score', 0.0)}.")

    # --- Core Exploration Logic (Async with optional LLM) ---
    async def explore_async(self, candidate_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Choose the most novel, ethically sound state to explore next.
        Uses LLM if salience high for nuanced assessment.
        """
        self._prune_history()

        # Serialize states for hashing (assume dicts with hashable keys/values)
        serialized_candidates = [json.dumps(state, sort_keys=True) for state in candidate_states]

        # Ethical pre-filter: Bias away from harmful states
        ethical_candidates = await self._ethical_filter_async(serialized_candidates)

        if not ethical_candidates:
            _log_warn(self.node_name, "No ethical candidates; falling back to random safe choice.")
            chosen_serialized = random.choice(serialized_candidates)
            chosen_state = candidate_states[serialized_candidates.index(chosen_serialized)]
            novelty_score = 0.5  # Neutral
            ethical_score = 0.8
            llm_reasoning = "Fallback due to ethical constraints."
        else:
            # Novelty assessment
            if self.cumulative_novelty_salience >= self.llm_novelty_threshold and self.use_llm_for_novelty:
                _log_info(self.node_name, f"Using LLM for novelty assessment (Salience: {self.cumulative_novelty_salience:.2f}).")
                novelty_output = await self._assess_novelty_llm(ethical_candidates)
                if novelty_output:
                    chosen_serialized = novelty_output.get('chosen_state', ethical_candidates[0])
                    novelty_score = float(novelty_output.get('novelty_score', 0.5))
                    llm_reasoning = novelty_output.get('reasoning', 'LLM assessed novelty.')
                else:
                    novelty_score, llm_reasoning = self._simple_novelty(ethical_candidates), "Fallback novelty."
            else:
                novelty_score, llm_reasoning = self._simple_novelty(ethical_candidates), "Simple novelty rules."

            chosen_serialized = max(ethical_candidates, key=lambda s: self.exploration_history.get(s, 1.0))
            chosen_state = candidate_states[serialized_candidates.index(chosen_serialized)]
            ethical_score = 0.9  # Assumed safe

        # Update knowledge
        self.known_states.add(chosen_serialized)
        self.exploration_history[chosen_serialized] = novelty_score

        # Log and publish
        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_exploration_log(
            id=str(uuid.uuid4()),
            timestamp=str(self._get_current_time()),
            chosen_state=json.dumps(chosen_state),
            novelty_score=novelty_score,
            ethical_score=ethical_score,
            llm_reasoning=llm_reasoning,
            context_snapshot_json=json.dumps({'candidates': candidate_states}),
            sensory_snapshot_json=sensory_snapshot
        )
        self.publish_exploration_decision(chosen_state, novelty_score, ethical_score, llm_reasoning)
        self.cumulative_novelty_salience = 0.0

        return {
            'chosen_state': chosen_state,
            'novelty_score': novelty_score,
            'ethical_score': ethical_score,
            'reasoning': llm_reasoning
        }

    async def _ethical_filter_async(self, serialized_candidates: List[str]) -> List[str]:
        """Simple ethical filter; extend with LLM if needed."""
        # Placeholder: Filter out 'harmful' states (e.g., based on keywords)
        safe_candidates = [s for s in serialized_candidates if 'harm' not in s.lower()]
        # Bias: Penalize low-ethics with probability
        for s in safe_candidates[:]:
            if random.random() < self.ethical_bias:
                safe_candidates.remove(s)
        return safe_candidates

    async def _assess_novelty_llm(self, candidates: List[str]) -> Optional[Dict[str, Any]]:
        """Use LLM to rank novelty if salience high."""
        prompt_text = f"""
        Assess novelty of these serialized states for a compassionate robot explorer. Prioritize safe, enriching discoveries.
        States: {json.dumps(candidates, indent=2)}

        Output JSON: {{
            "chosen_state": "the most novel state string",
            "novelty_score": number (0.0-1.0),
            "reasoning": "explanation, favoring ethical/novel over risky"
        }}
        """
        response_schema = {
            "type": "object",
            "properties": {
                "chosen_state": {"type": "string"},
                "novelty_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string"}
            },
            "required": ["chosen_state", "novelty_score", "reasoning"]
        }

        llm_output = await self._call_llm_api(prompt_text, response_schema, temperature=0.4, max_tokens=150)
        if not llm_output.startswith("Error:"):
            try:
                return json.loads(llm_output)
            except json.JSONDecodeError:
                self._report_error("LLM_PARSE_ERROR", "Failed to parse novelty response.", 0.8)
        return None

    def _simple_novelty(self, candidates: List[str]) -> float:
        """Fallback novelty scoring."""
        scores = []
        for state in candidates:
            if state not in self.known_states:
                scores.append(1.0)
            else:
                visits = sum(1 for s in self.exploration_history if s == state)
                scores.append(max(0.0, 1.0 - 0.1 * visits))
        return max(scores) if scores else 0.5

    # --- Database and Publishing Functions ---
    def save_exploration_log(self, **kwargs: Any):
        try:
            self.cursor.execute('''
                INSERT INTO exploration_log (id, timestamp, chosen_state, novelty_score, ethical_score, llm_reasoning, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['chosen_state'], kwargs['novelty_score'],
                kwargs['ethical_score'], kwargs['llm_reasoning'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved exploration log (ID: {kwargs['id']}, State: {kwargs['chosen_state'][:50]}...).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save exploration log: {e}", 0.9)

    def publish_exploration_decision(self, chosen_state: Dict[str, Any], novelty_score: float, ethical_score: float, reasoning: str):
        decision_data = {
            'timestamp': str(self._get_current_time()),
            'chosen_state': chosen_state,
            'novelty_score': novelty_score,
            'ethical_score': ethical_score,
            'reasoning': reasoning
        }
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_exploration_decision:
                if hasattr(ExplorationDecision, 'data'):
                    self.pub_exploration_decision.publish(String(data=json.dumps(decision_data)))
                else:
                    decision_msg = ExplorationDecision(**decision_data)
                    self.pub_exploration_decision.publish(decision_msg)
            _log_info(self.node_name, f"Exploration decision published: {chosen_state.get('id', 'unknown')} (Novelty: {novelty_score:.2f}).")
        except Exception as e:
            self._report_error("PUBLISH_EXPLORATION_ERROR", f"Failed to publish decision: {e}", 0.7)

    def summary(self) -> Dict[str, Any]:
        return {
            "known_states_count": len(self.known_states),
            "exploration_history_size": len(self.exploration_history),
            "pending_candidates": len(self.exploration_queue),
            "cumulative_salience": self.cumulative_novelty_salience
        }

    def shutdown(self):
        self._shutdown_flag.set() if hasattr(self, '_shutdown_flag') else None
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
    parser = argparse.ArgumentParser(description='Sentience Autonomous Explorer Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = AutonomousExplorerNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            candidates = [{"id": f"state{i}", "description": f"Novel area {i}"} for i in range(5)]
            for _ in range(3):
                decision = asyncio.run(node.explore_async(candidates))
                print(f"Dynamic exploration: {decision}")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
