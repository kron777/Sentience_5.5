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
    CognitiveDirective = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    ValueDriftState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    CognitiveDirective = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    ValueDriftState = ROSMsgFallback


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
            'system_integration_node': {
                'update_interval': 1.0,
                'max_event_history': 50,
                'batch_size': 50,
                'log_flush_interval_s': 10.0,
                'trait_update_interval_s': 60.0,
                'llm_endpoint': 'http://localhost:8080/phi2',
                'learning_rate': 0.01,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate system directives (e.g., empathetic error handling)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 5.0
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


class SystemIntegrationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'system_integration_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "system_log.db")
        self.update_interval = self.params.get('update_interval', 1.0)
        self.max_event_history = self.params.get('max_event_history', 50)
        self.batch_size = self.params.get('batch_size', 50)
        self.log_flush_interval_s = self.params.get('log_flush_interval_s', 10.0)
        self.trait_update_interval_s = self.params.get('trait_update_interval_s', 60.0)
        self.llm_endpoint = self.params.get('llm_endpoint', 'http://localhost:8080/phi2')
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing integration compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Load character traits
        self.character_traits = self._load_character_traits()

        # Internal state
        self.node_status = {
            'sensory_qualia_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'world_model_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'cognitive_reasoning_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'cognitive_control_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'behavior_execution_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'attention_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'emotion_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'motivation_node': {'status': 'unknown', 'last_updated': str(time.time())},
            'value_drift_monitor_node': {'status': 'unknown', 'last_updated': str(time.time())},
        }
        self.event_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_event_history)
        self.log_buffer: Deque[Dict[str, Any]] = deque(maxlen=self.batch_size)
        self.trait_update_buffer: Deque[Dict[str, Any]] = deque(maxlen=self.batch_size)
        self.latest_states = {
            'emotion': None,
            'motivation': None,
            'value_drift': None,
        }

        # Initialize SQLite database
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                system_status TEXT,
                directive_type TEXT,
                target_node TEXT,
                confidence_score REAL,
                contributing_factors TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Async setup
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()
        self._async_session = None

        # Simulated ROS Compatibility: Conditional Setup
        self.pub_directive = None
        self.pub_system_health = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)
            self.pub_system_health = rospy.Publisher('/system_health', String, queue_size=10)

            # Subscribers for status
            rospy.Subscriber('/sensory_qualia_node/status', String, lambda msg: self.status_callback('sensory_qualia_node', msg))
            rospy.Subscriber('/world_model_node/status', String, lambda msg: self.status_callback('world_model_node', msg))
            rospy.Subscriber('/cognitive_reasoning_node/status', String, lambda msg: self.status_callback('cognitive_reasoning_node', msg))
            rospy.Subscriber('/cognitive_control_node/status', String, lambda msg: self.status_callback('cognitive_control_node', msg))
            rospy.Subscriber('/behavior_execution_node/status', String, lambda msg: self.status_callback('behavior_execution_node', msg))
            rospy.Subscriber('/attention_node/status', String, lambda msg: self.status_callback('attention_node', msg))
            rospy.Subscriber('/emotion_node/status', String, lambda msg: self.status_callback('emotion_node', msg))
            rospy.Subscriber('/motivation_node/status', String, lambda msg: self.status_callback('motivation_node', msg))
            rospy.Subscriber('/value_drift_monitor_node/status', String, lambda msg: self.status_callback('value_drift_monitor_node', msg))
            # State subscribers
            rospy.Subscriber('/emotion_state', EmotionState, self.emotion_state_callback)
            rospy.Subscriber('/motivation_state', MotivationState, self.motivation_state_callback)
            rospy.Subscriber('/value_drift_state', ValueDriftState, self.value_drift_state_callback)

            rospy.Timer(rospy.Duration(self.update_interval), self.monitor_system)
            rospy.Timer(rospy.Duration(self.log_flush_interval_s), self.flush_log_buffer)
            rospy.Timer(rospy.Duration(self.trait_update_interval_s), self.flush_trait_updates)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        _log_info(self.node_name, "System Integration Node initialized with compassionate system health monitoring.")

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing integration compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on status or states
            if sensor_type == 'vision':
                self.pending_updates.append({'type': 'status', 'data': {'node_name': 'sensory_qualia_node', 'status': 'running'}})
            elif sensor_type == 'sound':
                self.latest_states['emotion'] = {'mood': 'reactive' if random.random() < 0.5 else 'normal'}
            elif sensor_type == 'instructions':
                self.latest_states['motivation'] = {'dominant_goal_id': 'system_maintenance', 'overall_drive_level': 0.7}
            # Compassionate bias: If distress in sound, bias toward health check
            if 'distress' in str(processed):
                self.pending_updates.append({'type': 'health', 'data': {'severity': 'medium', 'reason': 'environmental stress'}})
            _log_debug(self.node_name, f"{sensor_type} input updated integration context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_status_update()
            self._simulate_state_update()
            self.monitor_system()
            time.sleep(self.update_interval)

    def _simulate_status_update(self):
        """Simulate a status update in non-ROS mode."""
        node_name = random.choice(list(self.node_status.keys()))
        self.status_callback(node_name, {'data': json.dumps({'status': random.choice(['running', 'paused', 'error'])})})
        _log_debug(self.node_name, f"Simulated status for {node_name}")

    def _simulate_state_update(self):
        """Simulate a state update in non-ROS mode."""
        self.emotion_state_callback({'data': json.dumps({'mood': 'normal', 'sentiment_score': 0.5})})
        self.motivation_state_callback({'data': json.dumps({'dominant_goal_id': 'integration', 'overall_drive_level': 0.4})})
        self.value_drift_state_callback({'data': json.dumps({'drift_score': 0.1})})
        _log_debug(self.node_name, "Simulated state updates")

    # --- Core Integration Logic ---
    def status_callback(self, node_name: str, msg: Any):
        """Handle node status updates."""
        fields_map = {'data': ('', 'status_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        status_data = json.loads(data.get('status_data', '{}'))
        status = status_data.get('status', 'unknown')
        self.node_status[node_name]['status'] = status
        self.node_status[node_name]['last_updated'] = str(self._get_current_time())
        _log_debug(self.node_name, f"Updated status for {node_name}: {status}")

    def emotion_state_callback(self, msg: Any):
        """Handle incoming emotion state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'mood': ('normal', 'mood'),
            'sentiment_score': (0.0, 'sentiment_score'),
        }
        self.latest_states['emotion'] = parse_message_data(msg, fields_map, self.node_name)

    def motivation_state_callback(self, msg: Any):
        """Handle incoming motivation state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'dominant_goal_id': ('none', 'dominant_goal_id'),
            'overall_drive_level': (0.0, 'overall_drive_level'),
        }
        self.latest_states['motivation'] = parse_message_data(msg, fields_map, self.node_name)

    def value_drift_state_callback(self, msg: Any):
        """Handle incoming value drift state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'drift_score': (0.0, 'drift_score'),
        }
        self.latest_states['value_drift'] = parse_message_data(msg, fields_map, self.node_name)

    async def _async_monitor_system(self):
        """Monitor system health and generate directives with compassionate bias."""
        emotion = self.latest_states.get('emotion', {'mood': 'normal', 'sentiment_score': 0.0})
        motivation = self.latest_states.get('motivation', {'dominant_goal_id': 'none', 'overall_drive_level': 0.0})
        value_drift = self.latest_states.get('value_drift', {'drift_score': 0.0})

        # Calculate system health
        healthy_nodes = sum(1 for status in self.node_status.values() if status['status'] == 'running')
        total_nodes = len(self.node_status)
        system_health = healthy_nodes / total_nodes if total_nodes > 0 else 0.0
        failed_nodes = [name for name, status in self.node_status.items() if status['status'] != 'running']

        # Simplified LLM input for system directive with compassionate bias
        llm_input = {
            'system_health': system_health,
            'failed_nodes': failed_nodes,
            'mood': emotion['mood'],
            'goal_id': motivation['dominant_goal_id'],
            'drift_score': value_drift['drift_score'],
            'recent_events': [{'type': e['type'], 'target_node': e['target_node']} for e in list(self.event_history)[-5:]],  # Limit history
            'traits': {
                'reliability': self.character_traits['personality'].get('reliability', {}).get('value', 0.7),
                'empathy': self.character_traits['emotional_tendencies'].get('empathy', {}).get('value', 0.8),
            },
            'ethical_compassion_bias': self.ethical_compassion_bias  # Guide toward compassionate directives
        }

        # Query LLM
        try:
            async with self._async_session.post(self.llm_endpoint, json=llm_input, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                if response.status == 200:
                    llm_output = await response.json()
                    directive_type = llm_output.get('directive_type', 'none')
                    target_node = llm_output.get('target_node', 'none')
                    command_payload = llm_output.get('command_payload', {})
                    confidence_score = llm_output.get('confidence', 0.5)
                    contributing_factors = llm_output.get('factors', {})
                else:
                    _log_warn(self.node_name, f"LLM request failed with status {response.status}")
                    directive_type, target_node, command_payload, confidence_score, contributing_factors = 'none', 'none', {}, 0.3, {}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            _log_error(self.node_name, f"LLM request failed: {e}")
            directive_type, target_node, command_payload, confidence_score, contributing_factors = 'none', 'none', {}, 0.3, {}

        # Adjust directive based on traits and inputs with compassionate bias
        reliability = self.character_traits['personality'].get('reliability', {}).get('value', 0.7)
        empathy = self.character_traits['emotional_tendencies'].get('empathy', {}).get('value', 0.8)

        if failed_nodes and reliability > 0.6:
            directive_type = 'restart_node' if directive_type == 'none' else directive_type
            target_node = failed_nodes[0] if failed_nodes else 'none'
            command_payload = {'action': 'restart', 'priority': 0.9} if not command_payload else command_payload
            self._update_character_traits('personality', 'reliability', min(1.0, reliability + 0.05), 'node_failure')
        if system_health > 0.9 and reliability > 0.7:
            self._update_character_traits('personality', 'reliability', min(1.0, reliability + 0.03), 'stable_system')
        if value_drift['drift_score'] > 0.5:
            directive_type = 'ethical_check' if directive_type == 'none' else directive_type
            target_node = 'value_drift_monitor_node' if target_node == 'none' else target_node
            command_payload = {'action': 'reassess_values', 'priority': 0.8} if not command_payload else command_payload
        if emotion['mood'] == 'positive' and empathy > 0.7:
            command_payload['tone'] = 'friendly'

        # Compassionate bias: If low health, prioritize self-care directive
        if system_health < 0.5 and self.ethical_compassion_bias > 0.1:
            directive_type = 'self_care'
            target_node = 'motivation_node'
            command_payload = {'action': 'reflect', 'priority': 1.0}

        command_payload_json = json.dumps(command_payload)
        system_status = {'health': system_health, 'failed_nodes': failed_nodes}
        return directive_type, target_node, command_payload_json, confidence_score, contributing_factors, system_status

    def monitor_system(self, event: Any = None):
        """Periodic system monitoring."""
        timestamp = str(self._get_current_time())
        directive_type, target_node, command_payload_json, confidence_score, contributing_factors, system_status = asyncio.run_coroutine_threadsafe(
            self._async_monitor_system(), self._async_loop
        ).result()

        self.event_history.append({
            'type': directive_type,
            'target_node': target_node,
            'command_payload': command_payload_json,
            'timestamp': timestamp
        })

        # Publish directive
        self.publish_directive(timestamp, directive_type, target_node, command_payload_json, confidence_score, contributing_factors)

        # Publish system health
        self.publish_system_health(timestamp, system_status, confidence_score)

        # Log to buffer with sensory snapshot
        sensory_snapshot = json.dumps(self.sensory_data)
        self.log_buffer.append((
            timestamp,
            json.dumps(system_status),
            directive_type,
            target_node,
            confidence_score,
            json.dumps(contributing_factors),
            sensory_snapshot
        ))

        _log_info(self.node_name, f"System Health: {system_status['health']:.2f}, Directive: {directive_type}, Target: {target_node}")

    def publish_directive(self, timestamp: str, directive_type: str, target_node: str, command_payload_json: str, confidence_score: float, contributing_factors: Dict[str, Any]):
        """Publish cognitive directive."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_directive:
                if hasattr(CognitiveDirective, 'data'):  # String fallback
                    directive_data = {
                        'timestamp': timestamp,
                        'directive_type': directive_type,
                        'target_node': target_node,
                        'command_payload': command_payload_json,
                        'confidence_score': confidence_score,
                        'contributing_factors': json.dumps(contributing_factors),
                    }
                    self.pub_directive.publish(String(data=json.dumps(directive_data)))
                else:
                    msg = CognitiveDirective()
                    msg.timestamp = timestamp
                    msg.directive_type = directive_type
                    msg.target_node = target_node
                    msg.command_payload = command_payload_json
                    msg.confidence_score = confidence_score
                    msg.contributing_factors_json = json.dumps(contributing_factors)
                    self.pub_directive.publish(msg)
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish directive: {e}")

    def publish_system_health(self, timestamp: str, system_status: Dict[str, Any], confidence_score: float):
        """Publish system health status."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_system_health:
                health_data = {
                    'timestamp': timestamp,
                    'system_health': system_status['health'],
                    'failed_nodes': system_status['failed_nodes'],
                    'confidence_score': confidence_score,
                }
                self.pub_system_health.publish(String(data=json.dumps(health_data)))
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish system health: {e}")

    def flush_log_buffer(self, event: Any = None):
        """Flush log buffer to database with sensory snapshot."""
        if not self.log_buffer:
            return
        entries = list(self.log_buffer)
        self.log_buffer.clear()
        try:
            self.cursor.executemany('''
                INSERT INTO system_log (timestamp, system_status, directive_type, target_node, confidence_score, contributing_factors, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', [(e[0], e[1], e[2], e[3], e[4], e[5], e[6]) for e in entries])
            self.conn.commit()
            _log_info(self.node_name, f"Flushed {len(entries)} system log entries to DB.")
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to flush log buffer: {e}")
            for entry in entries:
                self.log_buffer.append(entry)

    def flush_trait_updates(self, event: Any = None):
        """Flush trait updates to JSON file."""
        if not self.trait_update_buffer:
            return
        try:
            with open(self.traits_path, 'w') as f:
                json.dump(self.character_traits, f, indent=2)
            entries = list(self.trait_update_buffer)
            self.trait_update_buffer.clear()
            _log_info(self.node_name, f"Flushed {len(entries)} trait updates to {self.traits_path}")
        except (IOError, json.JSONDecodeError) as e:
            _log_error(self.node_name, f"Failed to flush trait updates: {e}")
            for entry in entries:
                self.trait_update_buffer.append(entry)

    def _load_character_traits(self) -> Dict[str, Any]:
        """Load default character traits from JSON."""
        try:
            with open(self.traits_path, 'r') as f:
                traits = json.load(f)
            _log_info(self.node_name, f"Loaded character traits from {self.traits_path}")
            return traits
        except (FileNotFoundError, json.JSONDecodeError) as e:
            _log_warn(self.node_name, f"Failed to load character traits: {e}")
            return {
                "personality": {"reliability": {"value": 0.7, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "emotional_tendencies": {"empathy": {"value": 0.8, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "behavioral_preferences": {},
                "metadata": {"version": "1.0", "created": datetime.utcnow().isoformat() + 'Z', "last_modified": datetime.utcnow().isoformat() + 'Z', "update_history": []}
            }

    def _update_character_traits(self, trait_category: str, trait_key: str, new_value: float, source: str):
        """Update character traits and log changes with compassionate alignment."""
        try:
            if trait_category not in self.character_traits:
                self.character_traits[trait_category] = {}
            if trait_key not in self.character_traits[trait_category]:
                self.character_traits[trait_category][trait_key] = {"value": 0.5, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}
            # Compassionate bias: Cap empathy/reliability to prevent over-adaptation
            new_value = max(0.0, min(1.0, new_value)) if trait_key in ['reliability', 'empathy'] else new_value
            self.character_traits[trait_category][trait_key]['value'] = new_value
            self.character_traits[trait_category][trait_key]['weight'] = min(1.0, self.character_traits[trait_category][trait_key]['weight'] + self.learning_rate)
            self.character_traits[trait_category][trait_key]['last_updated'] = datetime.utcnow().isoformat() + 'Z'
            self.character_traits[trait_category][trait_key]['update_source'] = source
            self.character_traits['metadata']['last_modified'] = datetime.utcnow().isoformat() + 'Z'
            self.character_traits['metadata']['update_history'].append({
                'trait': f"{trait_category}.{trait_key}",
                'new_value': new_value,
                'source': source,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
            self.trait_update_buffer.append((trait_category, trait_key, new_value, source))
            _log_info(self.node_name, f"Updated trait {trait_category}.{trait_key} to {new_value} from {source}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to update trait {trait_category}.{trait_key}: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down SystemIntegrationNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        self.flush_log_buffer()
        self.flush_trait_updates()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if hasattr(self, '_async_loop') and self._async_thread.is_alive():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            self._async_thread.join(timeout=5.0)
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with asyncio integration."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    self._simulate_status_update()
                    self._simulate_state_update()
                    self.monitor_system()
                    time.sleep(self.update_interval)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience System Integration Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SystemIntegrationNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a status
            node.status_callback('sensory_qualia_node', {'data': json.dumps({'status': 'running'})})
            node.emotion_state_callback({'data': json.dumps({'mood': 'normal', 'sentiment_score': 0.5})})
            time.sleep(2)
            print("System integration simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
