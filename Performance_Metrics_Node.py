```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique report IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Any, Optional, Deque

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
    PerformanceReport = ROSMsgFallback
    SystemMetric = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MotivationState = ROSMsgFallback
    WorldModelState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    PerformanceReport = ROSMsgFallback
    SystemMetric = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MotivationState = ROSMsgFallback
    WorldModelState = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts/utils.py' exists and contains parse_message_data and load_config
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    # Fallback implementations
    def parse_message_data(msg: Any, fields_map: Dict[str, tuple], node_name: str = "unknown_node") -> Dict[str, Any]:
        """
        Generic parser for communications (ROS String/JSON or plain dict). 
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
            'performance_metrics_node': {
                'report_interval': 1.0,
                'llm_analysis_threshold_salience': 0.6,
                'recent_context_window_s': 15.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate performance assessments (e.g., growth-oriented insights)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 30.0
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


class PerformanceMetricsNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'performance_metrics_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "performance_log.db")
        self.report_interval = self.params.get('report_interval', 1.0)
        self.llm_analysis_threshold_salience = self.params.get('llm_analysis_threshold_salience', 0.6)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 15.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing performance compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 30.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's performance metrics system online, assessing with compassionate and mindful evaluation.")

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
            CREATE TABLE IF NOT EXISTS performance_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                overall_score REAL,
                suboptimal_flag BOOLEAN,
                kpis_json TEXT,
                llm_analysis_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_performance_report = {
            'timestamp': str(time.time()),
            'overall_score': 1.0,
            'suboptimal_flag': False,
            'kpis': {
                'task_completion_rate': 1.0,
                'latency_avg_ms': 50,
                'resource_utilization_avg_percent': 0.2,
                'error_rate': 0.0
            }
        }

        # History deques
        self.recent_system_metrics = deque(maxlen=20)  # More granular data needed for performance
        self.recent_cognitive_directives = deque(maxlen=5)
        self.recent_motivation_states = deque(maxlen=5)
        self.recent_world_model_states = deque(maxlen=5)

        self.cumulative_performance_salience = 0.0

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_performance_report = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_performance_report = rospy.Publisher('/performance_report', PerformanceReport, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/system_metrics', SystemMetric, self.system_metric_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/motivation_state', MotivationState, self.motivation_state_callback)
            rospy.Subscriber('/world_model_state', WorldModelState, self.world_model_state_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.report_interval), self._run_performance_analysis_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_performance_report(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing performance compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on performance metrics
            if sensor_type == 'vision':
                self.recent_system_metrics.append({'timestamp': timestamp, 'metric_name': 'vision_latency', 'value': random.uniform(30, 70), 'unit': 'ms'})
            elif sensor_type == 'sound':
                self.recent_system_metrics.append({'timestamp': timestamp, 'metric_name': 'audio_processing_time', 'value': random.uniform(20, 60), 'unit': 'ms'})
            elif sensor_type == 'instructions':
                self.recent_system_metrics.append({'timestamp': timestamp, 'metric_name': 'command_error_rate', 'value': random.uniform(0.0, 0.1), 'unit': '%'})
            # Compassionate bias: If distress in sound, simulate lower performance compassionately
            if 'distress' in str(processed):
                self._update_cumulative_salience(self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated performance context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_performance_analysis_wrapper(None)
            time.sleep(self.report_interval)

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

    def _run_performance_analysis_wrapper(self, event: Any = None):
        """Wrapper to run the async performance analysis from a timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM performance analysis task already active. Skipping new cycle.")
            return
        
        # Schedule the async task
        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self.analyze_performance_async(event), self._async_loop
        )

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

    # --- LLM Call Function ---
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.1, max_tokens: int = 300) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. Low temperature for factual analysis.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Low temperature for factual analysis
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
                
                self._report_error("LLM_RESPONSE_EMPTY", "LLM response had no content from local server.", 0.5, {'prompt_snippet': prompt_text[:100]})
                return "Error: LLM response empty."
        except aiohttp.ClientError as e:
            self._report_error("LLM_API_ERROR", f"LLM API request failed (aiohttp ClientError to local server): {e}", 0.9, {'url': api_url})
            return f"Error: LLM API request failed: {e}"
        except asyncio.TimeoutError:
            self._report_error("LLM_TIMEOUT", f"LLM API request timed out after {self.llm_timeout} seconds (local server).", 0.8, {'prompt_snippet': prompt_text[:100]})
            return "Error: LLM API request timed out."
        except json.JSONDecodeError:
            self._report_error("LLM_JSON_PARSE_ERROR", "Failed to parse local LLM response JSON.", 0.7)
            return "Error: Failed to parse LLM response."
        except Exception as e:
            self._report_error("UNEXPECTED_LLM_ERROR", f"An unexpected error occurred during local LLM call: {e}", 0.9, {'prompt_snippet': prompt_text[:100]})
            return f"Error: An unexpected error occurred: {e}"

    # --- Utility to accumulate input salience ---
    def _update_cumulative_salience(self, score: float):
        """Accumulates salience from new inputs for triggering LLM analysis."""
        self.cumulative_performance_salience += score
        self.cumulative_performance_salience = min(1.0, self.cumulative_performance_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        # Note: system_metrics deque has a larger maxlen as it needs more granular data
        while self.recent_system_metrics and (current_time - float(self.recent_system_metrics[0].get('timestamp', 0.0))) > self.recent_context_window_s:
            self.recent_system_metrics.popleft()
        
        for history_deque in [
            self.recent_cognitive_directives, self.recent_motivation_states,
            self.recent_world_model_states
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct) ---
    def system_metric_callback(self, msg: Any):
        """Handle incoming system metrics."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'metric_name': ('', 'metric_name'),
            'value': (0.0, 'value'), 'unit': ('', 'unit'), 'source_node': ('unknown', 'source_node')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_system_metrics.append(data)
        # High or critical metric values might indicate performance issues
        if "error" in data.get('metric_name', '').lower() and data.get('value', 0.0) > 0:
            self._update_cumulative_salience(0.5)
        elif "latency" in data.get('metric_name', '').lower() and data.get('value', 0.0) > 200:  # Example threshold
            self._update_cumulative_salience(0.3)
        _log_debug(self.node_name, f"Received System Metric: {data.get('metric_name', 'N/A')}: {data.get('value', 'N/A')}.")

    def cognitive_directive_callback(self, msg: Any):
        """Handle incoming cognitive directives."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name and data.get('directive_type') == 'AuditPerformance':
            try:
                payload = json.loads(data.get('command_payload', '{}'))
                self._update_cumulative_salience(data.get('urgency', 0.0) * 1.0)  # Explicit audit request is high salience
                _log_info(self.node_name, f"Received directive to audit performance based on reason: '{data.get('reason', 'N/A')}'.")
            except json.JSONDecodeError as e:
                self._report_error("JSON_DECODE_ERROR", f"Failed to decode command_payload in CognitiveDirective: {e}", 0.5, {'payload': data.get('command_payload')})
            except Exception as e:
                self._report_error("DIRECTIVE_PROCESSING_ERROR", f"Error processing CognitiveDirective for performance: {e}", 0.7, {'directive': data})
        
        self.recent_cognitive_directives.append(data)  # Store all directives for context
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    def motivation_state_callback(self, msg: Any):
        """Handle incoming motivation state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'dominant_goal_id': ('none', 'dominant_goal_id'),
            'overall_drive_level': (0.0, 'overall_drive_level'), 'active_goals_json': ('{}', 'active_goals_json')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('active_goals_json'), str):
            try:
                data['active_goals'] = json.loads(data['active_goals_json'])
            except json.JSONDecodeError:
                data['active_goals'] = {}
        self.recent_motivation_states.append(data)
        # Current goals define what 'good' performance means (e.g., fast vs. accurate)
        if data.get('overall_drive_level', 0.0) > 0.5:
            self._update_cumulative_salience(data.get('overall_drive_level', 0.0) * 0.2)
        _log_debug(self.node_name, f"Received Motivation State. Goal: {data.get('dominant_goal_id', 'N/A')}.")

    def world_model_state_callback(self, msg: Any):
        """Handle incoming world model state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'num_entities': (0, 'num_entities'),
            'changed_entities_json': ('[]', 'changed_entities_json'),
            'significant_change_flag': (False, 'significant_change_flag'),
            'consistency_score': (1.0, 'consistency_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('changed_entities_json'), str):
            try:
                data['changed_entities'] = json.loads(data['changed_entities_json'])
            except json.JSONDecodeError:
                data['changed_entities'] = []
        self.recent_world_model_states.append(data)
        # Environmental complexity influences expected performance
        if data.get('significant_change_flag', False) and data.get('num_entities', 0) > 5:
            self._update_cumulative_salience(0.3)
        _log_debug(self.node_name, f"Received World Model State. Significant Change: {data.get('significant_change_flag', False)}.")

    # --- Core Performance Analysis Logic (Async with LLM) ---
    async def analyze_performance_async(self, event: Any = None):
        """
        Asynchronously analyzes recent system metrics and cognitive context to generate
        a comprehensive performance report, using LLM for nuanced assessment with compassionate bias.
        """
        self._prune_history()  # Keep context history fresh

        overall_score = self.current_performance_report.get('overall_score', 1.0)
        suboptimal_flag = self.current_performance_report.get('suboptimal_flag', False)
        kpis = self.current_performance_report.get('kpis', {})
        llm_analysis_reasoning = "Not evaluated by LLM."
        
        if self.cumulative_performance_salience >= self.llm_analysis_threshold_salience or \
           (self.current_performance_report.get('suboptimal_flag', False) and self.current_performance_report.get('overall_score', 1.0) < 0.7):
            _log_info(self.node_name, f"Triggering LLM for performance analysis (Salience: {self.cumulative_performance_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_performance()
            llm_performance_output = await self._assess_performance_llm(context_for_llm)

            if llm_performance_output:
                overall_score = max(0.0, min(1.0, llm_performance_output.get('overall_score', overall_score)))
                suboptimal_flag = llm_performance_output.get('suboptimal_flag', suboptimal_flag)
                kpis = llm_performance_output.get('kpis', kpis)
                llm_analysis_reasoning = llm_performance_output.get('llm_analysis_reasoning', 'LLM provided no specific reasoning.')
                _log_info(self.node_name, f"LLM Performance Report. Score: {overall_score:.2f}. Suboptimal: {suboptimal_flag}.")
            else:
                _log_warn(self.node_name, "LLM performance analysis failed. Applying simple fallback.")
                overall_score, suboptimal_flag, kpis = self._apply_simple_performance_rules()
                llm_analysis_reasoning = "Fallback to simple rules due to LLM failure."
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_performance_salience:.2f}) for LLM performance analysis. Applying simple rules.")
            overall_score, suboptimal_flag, kpis = self._apply_simple_performance_rules()
            llm_analysis_reasoning = "Fallback to simple rules due to low salience."

        self.current_performance_report = {
            'timestamp': str(self._get_current_time()),
            'overall_score': overall_score,
            'suboptimal_flag': suboptimal_flag,
            'kpis': kpis
        }

        # Sensory snapshot for logging
        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_performance_log(
            id=str(uuid.uuid4()),
            timestamp=self.current_performance_report['timestamp'],
            overall_score=overall_score,
            suboptimal_flag=suboptimal_flag,
            kpis_json=json.dumps(kpis),
            llm_analysis_reasoning=llm_analysis_reasoning,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_performance()),
            sensory_snapshot_json=sensory_snapshot
        )
        self.publish_performance_report(None)  # Publish updated report
        self.cumulative_performance_salience = 0.0  # Reset after report generation

    async def _assess_performance_llm(self, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to assess the robot's overall performance and identify areas of suboptimality with compassionate bias.
        """
        prompt_text = f"""
        You are the Performance Metrics Module of a robot's cognitive architecture, powered by a large language model. Your crucial role is to analyze raw system metrics and contextual information to generate a comprehensive performance report. You must assess the robot's `overall_score`, identify if `suboptimal_flag` is true, and provide detailed `kpis`, with a bias toward compassionate, growth-oriented interpretations.

        Robot's Recent System Metrics:
        --- Raw System Metrics ---
        {json.dumps(context_for_llm.get('recent_system_metrics', []), indent=2)}

        Robot's Current Cognitive Context (for interpreting performance):
        --- Cognitive Context ---
        {json.dumps(context_for_llm.get('cognitive_context', {}), indent=2)}

        Sensory Snapshot:
        --- Sensory Data ---
        {json.dumps(context_for_llm.get('sensory_snapshot', {}), indent=2)}

        Based on this data, provide:
        1.  `overall_score`: number (0.0 to 1.0, where 1.0 is optimal performance. Aggregate all metrics and context into a single score.)
        2.  `suboptimal_flag`: boolean (True if performance is significantly below expected or desired levels, False otherwise.)
        3.  `kpis`: object (A JSON object containing key performance indicators, e.g., 'task_completion_rate', 'latency_avg_ms', 'resource_utilization_avg_percent', 'error_rate', 'goal_attainment_score'.)
        4.  `llm_analysis_reasoning`: string (Detailed explanation for your assessment, referencing specific metrics, goals, and environmental factors that influenced the performance, with compassionate, encouraging tone.)

        Consider:
        -   **System Metrics**: Analyze `value` for `metric_name` (e.g., high `error_rate`, high `latency_avg_ms`, low `resource_utilization_avg_percent` if idle is expected).
        -   **Motivation State**: Is the robot pursuing a `dominant_goal_id`? How well is it progressing toward it given its `overall_drive_level`? (This implies a 'goal_attainment_score' KPI).
        -   **World Model State**: Is the environment `complexity` high? This might justify lower performance scores. Are there `significant_change_flag`s that explain temporary dips?
        -   **Cognitive Directives**: Was there a directive to `AuditPerformance` or `OptimizePerformance`? What was the reason?
        -   **Ethical Compassion Bias**: Prioritize compassionate, growth-oriented insights (threshold: {self.ethical_compassion_bias}).

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'overall_score': number
        3.  'suboptimal_flag': boolean
        4.  'kpis': object
        5.  'llm_analysis_reasoning': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "suboptimal_flag": {"type": "boolean"},
                "kpis": {"type": "object"},  # Flexible JSON structure for KPIs
                "llm_analysis_reasoning": {"type": "string"}
            },
            "required": ["timestamp", "overall_score", "suboptimal_flag", "kpis", "llm_analysis_reasoning"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.1, max_tokens=600)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure boolean/numerical fields are correctly typed
                if 'overall_score' in llm_data:
                    llm_data['overall_score'] = float(llm_data['overall_score'])
                if 'suboptimal_flag' in llm_data:
                    llm_data['suboptimal_flag'] = bool(llm_data['suboptimal_flag'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for performance: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_PERFORMANCE_ANALYSIS_FAILED", f"LLM call failed for performance analysis: {llm_output_str}", 0.9)
            return None

    def _apply_simple_performance_rules(self) -> tuple[float, bool, Dict[str, Any]]:
        """
        Fallback mechanism to generate a simple performance report using rule-based logic
        if LLM is not triggered or fails.
        """
        current_time = self._get_current_time()
        
        # Calculate simple KPIs from recent system metrics
        total_latency = 0.0
        error_count = 0
        cpu_util_sum = 0.0
        num_metrics = 0
        
        for metric in list(self.recent_system_metrics)[-10:]:  # Recent metrics
            if current_time - float(metric.get('timestamp', 0.0)) < 10.0:  # Last 10 seconds
                if metric.get('metric_name') == 'latency_ms':
                    total_latency += metric.get('value', 0.0)
                elif metric.get('metric_name') == 'error_count':
                    error_count += metric.get('value', 0.0)
                elif metric.get('metric_name') == 'cpu_util_percent':
                    cpu_util_sum += metric.get('value', 0.0)
                num_metrics += 1

        avg_latency = total_latency / num_metrics if num_metrics > 0 else 0.0
        avg_cpu_util = cpu_util_sum / num_metrics if num_metrics > 0 else 0.0
        
        # Simple task completion rate (hypothetical for fallback)
        task_completion_rate = 1.0  # Assume perfect unless an error flag indicates otherwise
        if error_count > 0 or avg_latency > 150:
            task_completion_rate = 0.8  # Reduced if errors or high latency

        kpis = {
            'task_completion_rate': task_completion_rate,
            'latency_avg_ms': avg_latency,
            'resource_utilization_avg_percent': avg_cpu_util,
            'error_rate': error_count
        }

        # Determine overall score and suboptimal flag
        overall_score = 1.0
        suboptimal_flag = False

        if kpis['latency_avg_ms'] > 100:
            overall_score -= 0.2
            suboptimal_flag = True
        if kpis['error_rate'] > 0:
            overall_score -= 0.3
            suboptimal_flag = True
        if kpis['task_completion_rate'] < 0.9:
            overall_score -= 0.4
            suboptimal_flag = True
        
        # Clamp score between 0 and 1
        overall_score = max(0.0, min(1.0, overall_score))

        _log_warn(self.node_name, f"Simple rule: Generated fallback performance report. Score: {overall_score:.2f}.")
        return overall_score, suboptimal_flag, kpis

    def _compile_llm_context_for_performance(self) -> Dict[str, Any]:
        """
        Gathers and formats all relevant data for the LLM's performance assessment.
        """
        context = {
            "current_time": self._get_current_time(),
            "last_performance_report": self.current_performance_report,
            "recent_system_metrics": list(self.recent_system_metrics),
            "cognitive_context": {
                "latest_motivation_state": self.recent_motivation_states[-1] if self.recent_motivation_states else "N/A",
                "latest_world_model_state": self.recent_world_model_states[-1] if self.recent_world_model_states else "N/A",
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name]
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Deep parse any nested JSON strings in context for better LLM understanding
        for category_key in context["cognitive_context"]:
            item = context["cognitive_context"][category_key]
            if isinstance(item, dict):
                for field, value in item.items():
                    if isinstance(value, str) and field.endswith('_json'):
                        try:
                            item[field] = json.loads(value)
                        except json.JSONDecodeError:
                            pass

        return context

    # --- Database and Publishing Functions ---
    def save_performance_log(self, **kwargs: Any):
        """Saves a performance report entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO performance_log (id, timestamp, overall_score, suboptimal_flag, kpis_json, llm_analysis_reasoning, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['overall_score'], kwargs['suboptimal_flag'],
                kwargs['kpis_json'], kwargs['llm_analysis_reasoning'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved performance log (ID: {kwargs['id']}, Score: {kwargs['overall_score']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save performance log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_performance_log: {e}", 0.9)

    def publish_performance_report(self, event: Any = None):
        """Publishes the robot's current performance report."""
        timestamp = str(self._get_current_time())
        # Update timestamp before publishing
        self.current_performance_report['timestamp'] = timestamp
        
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_performance_report:
                if hasattr(PerformanceReport, 'data'):  # String fallback
                    self.pub_performance_report.publish(String(data=json.dumps(self.current_performance_report)))
                else:
                    report_msg = PerformanceReport()
                    report_msg.timestamp = timestamp
                    report_msg.overall_score = self.current_performance_report['overall_score']
                    report_msg.suboptimal_flag = self.current_performance_report['suboptimal_flag']
                    report_msg.kpis_json = json.dumps(self.current_performance_report['kpis'])
                    self.pub_performance_report.publish(report_msg)
            _log_debug(self.node_name, f"Published Performance Report. Score: '{self.current_performance_report['overall_score']}'.")
        except Exception as e:
            self._report_error("PUBLISH_PERFORMANCE_REPORT_ERROR", f"Failed to publish performance report: {e}", 0.7)

    def publish_cognitive_directive(self, directive_type: str, target_node: str, command_payload: str, urgency: float, reason: str = ""):
        """Helper to publish a CognitiveDirective message."""
        timestamp = str(self._get_current_time())
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_cognitive_directive:
                if hasattr(CognitiveDirective, 'data'):  # String fallback
                    directive_data = {
                        'timestamp': timestamp,
                        'directive_type': directive_type,
                        'target_node': target_node,
                        'command_payload': command_payload,
                        'urgency': urgency,
                        'reason': reason
                    }
                    self.pub_cognitive_directive.publish(String(data=json.dumps(directive_data)))
                else:
                    directive_msg = CognitiveDirective()
                    directive_msg.timestamp = timestamp
                    directive_msg.directive_type = directive_type
                    directive_msg.target_node = target_node
                    directive_msg.command_payload = command_payload
                    directive_msg.urgency = urgency
                    directive_msg.reason = reason
                    self.pub_cognitive_directive.publish(directive_msg)
            _log_debug(self.node_name, f"Issued Cognitive Directive '{directive_type}' to '{target_node}'.")
        except Exception as e:
            _log_error(self.node_name, f"Failed to issue cognitive directive from Performance Metrics Node: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down PerformanceMetricsNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self._shutdown_async_loop()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with asynchronous integration."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    self._run_performance_analysis_wrapper(None)
                    time.sleep(1)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Performance Metrics Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = PerformanceMetricsNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate inputs
            node.system_metric_callback({'data': json.dumps({'metric_name': 'latency_ms', 'value': 120.0})})
            node.system_metric_callback({'data': json.dumps({'metric_name': 'error_rate', 'value': 0.1})})
            node.motivation_state_callback({'data': json.dumps({'dominant_goal_id': 'task_completion', 'overall_drive_level': 0.7})})
            time.sleep(2)
            print("Performance metrics simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
