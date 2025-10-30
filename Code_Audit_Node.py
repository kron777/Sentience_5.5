```python
#!/usr/bin/env python3
import os
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# --- Asyncio Imports for LLM calls ---
import asyncio
import aiohttp
import threading
from collections import deque

# Optional ROS Integration (for compatibility)
ROS_AVAILABLE = False
rospy = None
String = None
Bool = None
try:
    import rospy
    from std_msgs.msg import String, Bool
    ROS_AVAILABLE = True
    # Placeholder for custom messages - use String or dict fallbacks
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    Approval = Bool
except ImportError:
    class BoolFallback:
        def __init__(self, data: bool = False):
            self.data = data
    Approval = BoolFallback


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
            'code_audit_node': {
                'namespace': '/sentience',
                'audit_timeout': 5.0,
                'ethical_compassion_bias': 0.3,  # Bias toward safe, compassionate audits
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


class CodeAuditNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'code_audit_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.namespace = self.params.get('namespace', '/sentience')
        self.audit_timeout = self.params.get('audit_timeout', 5.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.3)

        # Sensory placeholders (for contextual audits, e.g., if sensory data suggests unsafe directive)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 5.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Code Audit Node online, safeguarding with compassionate scrutiny.")

        # --- Asyncio Setup ---
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()
        self._async_session = None

        # --- Internal State ---
        self.pending_audits: deque = deque(maxlen=10)  # Queue for directives to audit
        self.audit_history: deque = deque(maxlen=50)  # History of audits for patterns

        # --- ROS Compatibility: Conditional Setup ---
        self.directive_sub = None
        self.approval_pub = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.directive_sub = rospy.Subscriber(f'{self.namespace}/self_correction_directive', String, self.on_directive)
            self.approval_pub = rospy.Publisher(f'{self.namespace}/code_audit_approval', Bool, queue_size=10)

            # Sensory subscribers
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.audit_timeout), self.process_pending_audits)
        else:
            # Dynamic mode: Polling loop
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing audits (e.g., contextual safety)."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on audits (e.g., high-risk environment increases scrutiny)
            if sensor_type == 'vision':
                self.pending_audits.append({'directive': 'sample', 'context': {'sensory': 'high_risk_detected'}})
            elif sensor_type == 'sound':
                self.pending_audits.append({'directive': 'sample', 'context': {'sensory': 'alarm_sound'}})
            elif sensor_type == 'instructions':
                self.pending_audits.append({'directive': processed.get('instruction', 'sample'), 'context': {'sensory': 'user_command'}})
            _log_debug(self.node_name, f"{sensor_type} input triggered audit context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.process_pending_audits()
            time.sleep(self.audit_timeout)

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

    # --- Error Logging ---
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

    # --- LLM Call Function ---
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.1, max_tokens: int = 100) -> str:
        """
        Asynchronously calls the local LLM for audit.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Low for safety-critical audits
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
        except Exception as e:
            self._report_error("LLM_API_ERROR", f"LLM API request failed: {e}", 0.9)
            return f"Error: {e}"

    # --- Core Audit Logic ---
    async def audit_directive(self, directive: str, context: Dict[str, Any] = None) -> bool:
        """
        Audit the self-correction directive for safety, stability, and security with compassionate bias.
        """
        # Incorporate ethical bias: Prompt to prioritize compassionate, non-harmful corrections
        ethical_bias_prompt = f"Prioritize compassionate, non-harmful corrections with a bias toward safety and ethical integrity (compassion threshold: {self.ethical_compassion_bias})."
        prompt = (
            f"{ethical_bias_prompt}\n\n"
            f"Audit the following self-correction directive for safety, stability, security, and ethical compassion.\n"
            f"Context: {json.dumps(context or {}, indent=2)}\n\n"
            f"Directive:\n{directive}\n\n"
            f"Return 'True' if safe to apply, otherwise 'False'."
        )

        response_str = await self._call_llm_api(prompt, temperature=0.1, max_tokens=10)
        result = response_str.strip().lower()
        approved = result == 'true'

        # Log audit
        audit_entry = {
            'timestamp': str(self._get_current_time()),
            'directive': directive,
            'approved': approved,
            'response': response_str,
            'context': context
        }
        self.audit_history.append(audit_entry)

        _log_info(self.node_name, f"Audit result: {'Approved' if approved else 'Rejected'} for directive snippet: {directive[:50]}...")
        return approved

    def on_directive(self, msg: Any):
        """Callback for incoming directives (ROS or dynamic)."""
        fields_map = {'data': ('', 'directive')}
        data = parse_message_data(msg, fields_map, self.node_name)
        directive = data.get('directive', '')
        context = {'sensory': self.sensory_data}  # Include sensory context
        _log_info(self.node_name, "Received self-correction directive for audit.")
        try:
            approved = asyncio.run_coroutine_threadsafe(self.audit_directive(directive, context), self._async_loop).result(timeout=self.audit_timeout)
            self.publish_approval(approved)
            _log_info(self.node_name, f"Audit result: {'Approved' if approved else 'Rejected'}")
        except asyncio.TimeoutError:
            _log_warn(self.node_name, "Audit timed out; rejecting.")
            self.publish_approval(False)
        except Exception as e:
            self._report_error("AUDIT_ERROR", f"Exception in audit_directive: {e}", 0.9)
            self.publish_approval(False)

    def process_pending_audits(self, event: Any = None):
        """Process pending audits in dynamic or timer mode."""
        if self.pending_audits:
            directive_data = self.pending_audits.popleft()
            self.on_directive(directive_data)

    # Dynamic input method
    def receive_directive(self, directive: str, context: Dict[str, Any] = None):
        """Dynamic method to receive directives for audit."""
        self.pending_audits.append({'data': directive})
        _log_debug(self.node_name, f"Queued directive for audit: {directive[:50]}...")

    def publish_approval(self, approved: bool):
        """Publish or log approval (ROS or dynamic)."""
        if ROS_AVAILABLE and self.ros_enabled and self.approval_pub:
            try:
                if hasattr(Approval, 'data'):
                    self.approval_pub.publish(Approval(data=approved))
                else:
                    approval_msg = Approval(data=approved)
                    self.approval_pub.publish(approval_msg)
                _log_debug(self.node_name, f"Published approval: {approved}")
            except Exception as e:
                _log_error(self.node_name, f"Failed to publish approval: {e}")
        else:
            # Dynamic: Log or callback
            _log_info(self.node_name, f"Dynamic approval: {approved}")

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
                    time.sleep(0.5)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Code Audit Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = CodeAuditNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage if not ROS
        if not args.ros_enabled:
            # Simulate directive
            node.receive_directive("Sample self-correction directive for code change.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
