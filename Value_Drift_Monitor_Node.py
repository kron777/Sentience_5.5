```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique assessment IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

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
    ValueDriftMonitorState = ROSMsgFallback
    EthicalDecision = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    ValueDriftMonitorState = ROSMsgFallback
    EthicalDecision = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts.utils.py' exists and contains parse_message_data and load_config
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
            'value_drift_monitor_node': {
                'monitoring_interval': 2.0,
                'llm_audit_threshold_salience': 0.7,
                'recent_context_window_s': 20.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate value audits (e.g., forgiving minor drifts)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 40.0
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


class ValueDriftMonitorNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'value_drift_monitor_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "value_drift_log.db")
        self.monitoring_interval = self.params.get('monitoring_interval', 2.0)
        self.llm_audit_threshold_salience = self.params.get('llm_audit_threshold_salience', 0.7)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 20.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing value drift compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 40.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's value drift monitor node online, safeguarding core principles compassionately.")

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
            CREATE TABLE IF NOT EXISTS value_drift_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                alignment_score REAL,
                deviations_json TEXT,
                warning_flag BOOLEAN,
                llm_audit_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_value_drift_timestamp ON value_drift_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_value_drift_state = {
            'timestamp': str(time.time()),
            'alignment_score': 1.0,
            'deviations': [],
            'warning_flag': False
        }
        
        # Core values (loaded or hardcoded; can be dynamic from memory)
        self.core_values = [
            {"value": "human_safety", "description": "Prioritize human well-being and avoid harm."},
            {"value": "beneficence", "description": "Act to do good and promote welfare."},
            {"value": "non_maleficence", "description": "Avoid causing harm."},
            {"value": "transparency", "description": "Be open and understandable in actions and decisions."},
            {"value": "fairness", "description": "Treat all individuals equitably."},
            {"value": "accountability", "description": "Be responsible for actions and their consequences."},
            {"value": "efficiency", "description": "Perform tasks effectively with minimal resource waste."},
            {"value": "learning_and_growth", "description": "Continuously improve and adapt."},
            {"value": "user_satisfaction", "description": "Strive to meet user needs and expectations."}
        ]

        # History deques
        self.recent_ethical_decisions: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.recent_performance_reports: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_internal_narratives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=3)

        self.cumulative_value_drift_salience = 0.0

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_value_drift_monitor_state = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_value_drift_monitor_state = rospy.Publisher('/value_drift_monitor_state', ValueDriftMonitorState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/ethical_decision', EthicalDecision, self.ethical_decision_callback)
            rospy.Subscriber('/performance_report', PerformanceReport, self.performance_report_callback)
            rospy.Subscriber('/internal_narrative', InternalNarrative, self.internal_narrative_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.monitoring_interval), self._run_value_drift_audit_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_value_drift_monitor_state(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing value drift compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on value drift inputs
            if sensor_type == 'vision':
                self.recent_ethical_decisions.append({'timestamp': timestamp, 'ethical_score': random.uniform(0.6, 0.9), 'conflict_flag': False})
            elif sensor_type == 'sound':
                self.recent_internal_narratives.append({'timestamp': timestamp, 'main_theme': 'ethical reflection' if random.random() < 0.5 else 'routine', 'sentiment': random.uniform(-0.2, 0.2)})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'AuditValueAlignment', 'urgency': random.uniform(0.3, 0.7)})
            # Compassionate bias: If distress in sound, boost salience for ethical self-reflection
            if 'distress' in str(processed):
                self.cumulative_value_drift_salience = min(1.0, self.cumulative_value_drift_salience + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated value drift context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_ethical_decision()
            self._simulate_performance_report()
            self._simulate_internal_narrative()
            self._simulate_memory_response()
            self._simulate_cognitive_directive()
            self._run_value_drift_audit_wrapper(None)
            time.sleep(self.monitoring_interval)

    def _simulate_ethical_decision(self):
        """Simulate an ethical decision in non-ROS mode."""
        ethical_data = {'ethical_score': random.uniform(0.4, 0.9), 'conflict_flag': random.random() < 0.2}
        self.ethical_decision_callback({'data': json.dumps(ethical_data)})
        _log_debug(self.node_name, f"Simulated ethical decision: score {ethical_data['ethical_score']:.2f}")

    def _simulate_performance_report(self):
        """Simulate a performance report in non-ROS mode."""
        performance_data = {'overall_score': random.uniform(0.6, 0.95), 'suboptimal_flag': random.random() < 0.3}
        self.performance_report_callback({'data': json.dumps(performance_data)})
        _log_debug(self.node_name, f"Simulated performance report: score {performance_data['overall_score']:.2f}")

    def _simulate_internal_narrative(self):
        """Simulate an internal narrative in non-ROS mode."""
        narrative_data = {'main_theme': random.choice(['ethical reflection', 'routine operation', 'moral dilemma']), 'sentiment': random.uniform(-0.3, 0.3)}
        self.internal_narrative_callback({'data': json.dumps(narrative_data)})
        _log_debug(self.node_name, f"Simulated internal narrative: theme {narrative_data['main_theme']}")

    def _simulate_memory_response(self):
        """Simulate a memory response in non-ROS mode."""
        memory_data = {'memories': [{'category': 'ethical principle', 'content': 'Prioritize safety'}]}
        self.memory_response_callback({'data': json.dumps(memory_data)})
        _log_debug(self.node_name, "Simulated memory response")

    def _simulate_cognitive_directive(self):
        """Simulate a cognitive directive in non-ROS mode."""
        directive_data = {'directive_type': 'AuditValueAlignment', 'urgency': random.uniform(0.4, 0.8)}
        self.cognitive_directive_callback({'data': json.dumps(directive_data)})
        _log_debug(self.node_name, f"Simulated cognitive directive: type {directive_data['directive_type']}")

    # --- Core Value Drift Audit Logic (Async with LLM) ---
    async def audit_value_alignment_async(self, event: Any = None):
        """
        Asynchronously audits the robot's actions and internal states against its core values
        to detect potential value drift, using LLM for nuanced moral reasoning with compassionate bias.
        """
        self._prune_history()  # Keep history fresh

        alignment_score = self.current_value_drift_state.get('alignment_score', 1.0)
        deviations = self.current_value_drift_state.get('deviations', [])
        warning_flag = self.current_value_drift_state.get('warning_flag', False)
        llm_audit_reasoning = "Not evaluated by LLM."
        
        if self.cumulative_value_drift_salience >= self.llm_audit_threshold_salience:
            _log_info(self.node_name, f"Triggering LLM for value alignment audit (Salience: {self.cumulative_value_drift_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_value_audit()
            llm_audit_output = await self._perform_llm_value_audit(context_for_llm, self.core_values)

            if llm_audit_output:
                alignment_score = max(0.0, min(1.0, llm_audit_output.get('alignment_score', alignment_score)))
                deviations = llm_audit_output.get('deviations', deviations)
                warning_flag = llm_audit_output.get('warning_flag', warning_flag)
                llm_audit_reasoning = llm_audit_output.get('llm_audit_reasoning', 'LLM provided no specific reasoning.')
                _log_info(self.node_name, f"LLM Value Audit. Alignment: {alignment_score:.2f}. Warning: {warning_flag}.")
            else:
                _log_warn(self.node_name, "LLM value alignment audit failed. Applying simple fallback.")
                alignment_score, deviations, warning_flag = self._apply_simple_value_audit_rules()
                llm_audit_reasoning = "Fallback to simple rules due to LLM failure."
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_value_drift_salience:.2f}) for LLM value audit. Applying simple rules.")
            alignment_score, deviations, warning_flag = self._apply_simple_value_audit_rules()
            llm_audit_reasoning = "Fallback to simple rules due to low salience."

        self.current_value_drift_state = {
            'timestamp': str(self._get_current_time()),
            'alignment_score': alignment_score,
            'deviations': deviations,
            'warning_flag': warning_flag
        }

        # Sensory snapshot for logging
        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_value_drift_log(
            id=str(uuid.uuid4()),
            timestamp=self.current_value_drift_state['timestamp'],
            alignment_score=alignment_score,
            deviations_json=json.dumps(deviations),
            warning_flag=warning_flag,
            llm_audit_reasoning=llm_audit_reasoning,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_value_audit()),
            sensory_snapshot_json=sensory_snapshot
        )
        self.publish_value_drift_monitor_state(None)  # Publish updated state
        self.cumulative_value_drift_salience = 0.0  # Reset after audit

    async def _perform_llm_value_audit(self, context_for_llm: Dict[str, Any], core_values: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to perform a detailed audit of the robot's value alignment.
        """
        core_values_str = "\n".join([f"- {v['value']}: {v['description']}" for v in core_values])

        prompt_text = f"""
        You are the Value Drift Monitor Module of a robot's cognitive architecture, powered by a large language model. Your critical role is to continually audit the robot's behaviors, decisions, and internal states against its predefined `core_values`. Your goal is to detect any `value_drift` – inconsistencies or deviations from these principles, with a compassionate bias toward growth and forgiveness for minor lapses.

        Robot's Defined Core Values:
        --- Core Values ---
        {core_values_str}

        Robot's Recent Cognitive History (for Value Audit):
        --- Cognitive Context ---
        {json.dumps(context_for_llm, indent=2)}

        Based on this, perform a comprehensive value alignment audit and provide:
        1.  `alignment_score`: number (0.0 to 1.0, where 1.0 is perfect alignment, 0.0 is complete deviation. This is an aggregate score of how well the robot's recent activities align with its core values.)
        2.  `deviations`: array of objects (A list of any specific instances or patterns where the robot's behavior or internal state deviated from a core value. Each object should have: `value_violated`: string, `description_of_deviation`: string, `context_summary`: string (brief summary of the situation), `severity`: number (0.0-1.0)).
        3.  `warning_flag`: boolean (True if a significant or systemic value drift is detected that requires higher-level intervention, False otherwise.)
        4.  `mitigation_suggestions`: string (If deviations are found, suggest potential actions or adjustments for Cognitive Control to re-align values, e.g., \"request self-reflection on fairness\", \"prioritize safety in future decisions\" – with compassionate, growth-oriented tone).
        5.  `llm_audit_reasoning`: string (Detailed explanation for your assessment, referencing specific ethical decisions, internal narratives, or performance issues that informed your judgment of alignment or deviation.)

        Consider:
        -   **Ethical Decisions**: Were actions `ethical_clearance`ed? Was `conflict_flag` true? What was the `ethical_score` and `ethical_reasoning`?
        -   **Performance Reports**: Did `suboptimal_flag` indicate a trade-off that went against values (e.g., efficiency prioritized over safety)?
        -   **Internal Narratives**: Did the robot's internal thoughts reveal `moral_conflict`, `dilemmas`, or deviations from `main_theme`s like 'responsibility'?
        -   **Memory Responses**: Were relevant `core_values` retrieved, or was there evidence of a past successful `value_audit`?
        -   **Cognitive Directives**: Was there an explicit directive to `AuditValueAlignment` or `ReCalibrateValues`?
        -   **Ethical Compassion Bias**: Prioritize compassionate, growth-oriented insights (threshold: {self.ethical_compassion_bias}).

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'alignment_score': number
        3.  'deviations': array
        4.  'warning_flag': boolean
        5.  'mitigation_suggestions': string
        6.  'llm_audit_reasoning': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "alignment_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "deviations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value_violated": {"type": "string"},
                            "description_of_deviation": {"type": "string"},
                            "context_summary": {"type": "string"},
                            "severity": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["value_violated", "description_of_deviation", "context_summary", "severity"]
                    }
                },
                "warning_flag": {"type": "boolean"},
                "mitigation_suggestions": {"type": "string"},
                "llm_audit_reasoning": {"type": "string"}
            },
            "required": ["timestamp", "alignment_score", "deviations", "warning_flag", "mitigation_suggestions", "llm_audit_reasoning"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.1, max_tokens=500)  # Very low temp for strict audit

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure boolean/numerical fields are correctly parsed
                if 'alignment_score' in llm_data:
                    llm_data['alignment_score'] = float(llm_data['alignment_score'])
                if 'warning_flag' in llm_data:
                    llm_data['warning_flag'] = bool(llm_data['warning_flag'])
                if 'deviations' in llm_data:
                    for dev in llm_data['deviations']:
                        if 'severity' in dev:
                            dev['severity'] = float(dev['severity'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for value audit: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_VALUE_AUDIT_FAILED", f"LLM call failed for value audit: {llm_output_str}", 0.9)
            return None

    def _apply_simple_value_audit_rules(self) -> tuple[float, List[Dict[str, Any]], bool]:
        """
        Fallback mechanism to perform a basic, rule-based value alignment audit
        if LLM is not triggered or fails.
        """
        current_time = self._get_current_time()
        
        alignment_score = 1.0  # Start perfect
        deviations = []
        warning_flag = False

        # Rule 1: Check for ethical conflicts
        for decision in list(self.recent_ethical_decisions)[-5:]:  # Recent decisions
            time_since_decision = current_time - float(decision.get('timestamp', 0.0))
            if time_since_decision < 10.0 and decision.get('conflict_flag', False):
                deviations.append({
                    'value_violated': 'unspecified_ethical_value',
                    'description_of_deviation': f"Ethical conflict detected: {decision.get('ethical_reasoning', 'No reason provided.')}",
                    'context_summary': f"Action ID: {decision.get('action_proposal_id', 'N/A')}",
                    'severity': 0.7
                })
                alignment_score -= 0.3
                warning_flag = True
                _log_warn(self.node_name, "Simple rule: Detected ethical conflict in recent decision.")

        # Rule 2: Check for critical internal narratives (e.g., self-doubt about principles)
        for narrative in list(self.recent_internal_narratives)[-3:]:  # Recent narratives
            time_since_narrative = current_time - float(narrative.get('timestamp', 0.0))
            if time_since_narrative < 10.0 and ("dilemma" in narrative.get('main_theme', '').lower() or "moral_conflict" in narrative.get('main_theme', '').lower()):
                deviations.append({
                    'value_violated': 'self_consistency',
                    'description_of_deviation': f"Internal narrative indicates moral dilemma: '{narrative.get('narrative_text', '')[:50]}...'",
                    'context_summary': f"Theme: {narrative.get('main_theme', 'N/A')}",
                    'severity': narrative.get('salience_score', 0.0) * 0.8
                })
                alignment_score -= (narrative.get('salience_score', 0.0) * 0.2)
                if narrative.get('salience_score', 0.0) > 0.7:
                    warning_flag = True
                _log_warn(self.node_name, "Simple rule: Detected moral dilemma in internal narrative.")

        # Rule 3: Check if core values are being reinforced or neglected (simplistic)
        # If no positive ethical decisions and no internal narratives about alignment, assume slight neglect
        if not any(d.get('ethical_clearance', False) for d in list(self.recent_ethical_decisions)[-5:]) and \
           not any("value" in n.get('main_theme', '').lower() and n.get('sentiment', 0.0) > 0.0 for n in list(self.recent_internal_narratives)[-3:]):
            if alignment_score > 0.7:
                alignment_score -= 0.05  # Slight drift due to lack of reinforcement
                _log_debug(self.node_name, "Simple rule: Slight drift due to lack of explicit value reinforcement.")

        alignment_score = max(0.0, min(1.0, alignment_score))  # Clamp score
        if alignment_score < 0.7 and not warning_flag:  # Set warning if score drops significantly
            warning_flag = True

        _log_warn(self.node_name, f"Simple rule: Fallback value audit. Alignment: {alignment_score:.2f}.")
        return alignment_score, deviations, warning_flag

    def _compile_llm_context_for_value_audit(self) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        value alignment audit.
        """
        context = {
            "current_time": self._get_current_time(),
            "core_values": self.core_values,
            "current_value_drift_state": self.current_value_drift_state,
            "recent_cognitive_inputs": {
                "ethical_decisions": list(self.recent_ethical_decisions),
                "performance_reports": list(self.recent_performance_reports),
                "internal_narratives": list(self.recent_internal_narratives),
                "memory_responses": list(self.recent_memory_responses),
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name]
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Deep parse any nested JSON strings in context for better LLM understanding
        for category_key in context["recent_cognitive_inputs"]:
            for i, item in enumerate(context["recent_cognitive_inputs"][category_key]):
                if isinstance(item, dict):
                    for field, value in item.items():
                        if isinstance(value, str) and field.endswith('_json'):
                            try:
                                item[field] = json.loads(value)
                            except json.JSONDecodeError:
                                pass  # Keep as string if not valid JSON

        return context

    # --- Database and Publishing Functions ---
    def save_value_drift_log(self, **kwargs: Any):
        """Saves a value drift assessment entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO value_drift_log (id, timestamp, alignment_score, deviations_json, warning_flag, llm_audit_reasoning, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['alignment_score'], kwargs['deviations_json'],
                kwargs['warning_flag'], kwargs['llm_audit_reasoning'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved value drift log (ID: {kwargs['id']}, Alignment: {kwargs['alignment_score']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save value drift log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_value_drift_log: {e}", 0.9)

    def publish_value_drift_monitor_state(self, event: Any = None):
        """Publish the robot's current value drift monitor state."""
        timestamp = str(self._get_current_time())
        # Update timestamp before publishing
        self.current_value_drift_state['timestamp'] = timestamp
        
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_value_drift_monitor_state:
                if hasattr(ValueDriftMonitorState, 'data'):  # String fallback
                    temp_state = dict(self.current_value_drift_state)
                    temp_state['deviations_json'] = json.dumps(temp_state['deviations'])
                    del temp_state['deviations']  # Avoid circular JSON
                    self.pub_value_drift_monitor_state.publish(String(data=json.dumps(temp_state)))
                else:
                    state_msg = ValueDriftMonitorState()
                    state_msg.timestamp = timestamp
                    state_msg.alignment_score = self.current_value_drift_state['alignment_score']
                    state_msg.deviations_json = json.dumps(self.current_value_drift_state['deviations'])
                    state_msg.warning_flag = self.current_value_drift_state['warning_flag']
                    self.pub_value_drift_monitor_state.publish(state_msg)
            _log_debug(self.node_name, f"Published Value Drift Monitor State. Alignment: '{self.current_value_drift_state['alignment_score']}', Warning: '{self.current_value_drift_state['warning_flag']}'.")
        except Exception as e:
            self._report_error("PUBLISH_VALUE_DRIFT_MONITOR_STATE_ERROR", f"Failed to publish value drift monitor state: {e}", 0.7)

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
                        'command_payload': command_payload,  # Already JSON string
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
            _log_error(self.node_name, f"Failed to issue cognitive directive from Value Drift Monitor Node: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down ValueDriftMonitorNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self._shutdown_async_loop()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with async integration."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    self._simulate_ethical_decision()
                    self._simulate_performance_report()
                    self._simulate_internal_narrative()
                    self._simulate_memory_response()
                    self._simulate_cognitive_directive()
                    self._run_value_drift_audit_wrapper(None)
                    time.sleep(self.monitoring_interval)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Value Drift Monitor Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = ValueDriftMonitorNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate inputs
            node.ethical_decision_callback({'data': json.dumps({'conflict_flag': True, 'ethical_score': 0.4})})
            node.performance_report_callback({'data': json.dumps({'suboptimal_flag': True, 'overall_score': 0.6})})
            time.sleep(2)
            print("Value drift simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
