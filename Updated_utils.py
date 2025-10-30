```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import yaml
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
    UtilityService = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    UtilityService = ROSMsgFallback


# --- Import shared utility functions ---
# This is the utility module itself, so self-referential; use fallback for config
try:
    from sentience.scripts.utils import load_config  # Circular, so fallback below
except ImportError:
    # Fallback for load_config (defined below in this file)
    pass

def parse_message_data(msg: Any, fields_map: Dict[str, tuple], node_name: str = "unknown_node") -> Dict[str, Any]:
    """
    Parses a message into a dictionary, applying default values and validating types.
    This function handles both native ROS message objects and std_msgs/String messages
    that contain JSON payloads. Adapted for general use with compassionate logging.

    Args:
        msg: The incoming message object.
        fields_map (dict): A dictionary defining expected fields, their default values,
                           and expected types.
                           Format: {'desired_key': (default_value, 'msg_field_name', expected_type_or_tuple_of_types)}
                           'msg_field_name' is the actual field name in the message/JSON.
                           If msg_field_name is missing, desired_key will be used as field name.
        node_name (str): The name of the node calling this utility, for logging purposes.

    Returns:
        dict: A dictionary containing the parsed data. Includes 'parse_success': False
              and 'error_reason' if validation fails.
    """
    parsed_data = {'parse_success': True, 'error_reason': None}
    raw_data_source = {}

    if hasattr(msg, 'data') and isinstance(getattr(msg, 'data', None), str):
        try:
            raw_data_source = json.loads(msg.data)
        except json.JSONDecodeError as e:
            _log_warn(node_name, f"JSON decode error in String message: {e}. Raw: '{msg.data[:100]}...'")
            parsed_data['parse_success'] = False
            parsed_data['error_reason'] = f"JSON decode error: {e}"
            # Populate with defaults for all fields to avoid KeyError downstream
            for key, (default_value, _, _) in fields_map.items():
                parsed_data[key] = default_value
            return parsed_data
    elif hasattr(msg, '__dict__'):  # For native ROS messages
        raw_data_source = msg.__dict__
    else:
        _log_warn(node_name, f"Unsupported message type: {type(msg)}. Expected String or native ROS message.")
        parsed_data['parse_success'] = False
        parsed_data['error_reason'] = f"Unsupported message type: {type(msg)}"
        # Populate with defaults for all fields
        for key, (default_value, _, _) in fields_map.items():
            parsed_data[key] = default_value
        return parsed_data

    for desired_key, (default_value, msg_field_name, expected_type) in fields_map.items():
        actual_field_name = msg_field_name if msg_field_name else desired_key  # Use msg_field_name if provided

        if actual_field_name in raw_data_source:
            value = raw_data_source[actual_field_name]
            # Validate type
            if not isinstance(value, expected_type):
                _log_warn(node_name, f"Type mismatch for '{desired_key}' (field '{actual_field_name}'). Expected {expected_type}, got {type(value)}. Using default.")
                parsed_data[desired_key] = default_value
                parsed_data['parse_success'] = False
                parsed_data['error_reason'] = f"Type mismatch for '{desired_key}' (expected {expected_type}, got {type(value)})"
            else:
                parsed_data[desired_key] = value
        else:
            _log_warn(node_name, f"Field '{desired_key}' (or '{actual_field_name}') not found in message. Using default: {default_value}")
            parsed_data[desired_key] = default_value
            parsed_data['parse_success'] = False
            parsed_data['error_reason'] = f"Missing field '{desired_key}'"

    # Compassionate bias: If parse failed, add encouraging note
    if not parsed_data['parse_success']:
        parsed_data['compassionate_note'] = "Parse error noted; system will learn from this with compassion."

    return parsed_data

def load_config(node_name: str, config_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration parameters from a centralized YAML file for a specific node.

    Args:
        node_name (str): The name of the current ROS node (e.g., 'attention_node').
        config_file_path (str): The absolute path to the sentience_config.yaml file.

    Returns:
        dict: A dictionary of parameters relevant to the given node,
              merged with global parameters. Returns an empty dict on error.
    """
    config = {}
    try:
        # Expand user home directory if '~' is used
        expanded_path = os.path.expanduser(config_file_path)
        with open(expanded_path, 'r') as f:
            full_config = yaml.safe_load(f)

        if not isinstance(full_config, dict):
            _log_error(node_name, f"Config file '{expanded_path}' is not a valid YAML dictionary.")
            return {}

        # Load global parameters
        global_params = full_config.get('global', {})
        
        # Initialize node_params with global LLM settings
        node_params = global_params.get('llm_api', {}).copy()
        
        # Override with node-specific parameters
        node_specific_params = full_config.get(node_name, {})
        
        # Merge global LLM parameters with node-specific LLM overrides
        if 'llm_api' in node_specific_params:
            # Update global LLM params with node-specific ones
            node_params.update(node_specific_params.pop('llm_api'))

        # Add remaining node-specific parameters
        node_params.update(node_specific_params)

        # Handle db_root_path globally and apply to node's db_path
        db_root_path = os.path.expanduser(global_params.get('db_root_path', "~/.sentience"))
        node_params['db_root_path'] = db_root_path  # Make available to all nodes

        # Set default log level
        node_params['default_log_level'] = global_params.get('default_log_level', 'INFO')

        _log_info(node_name, f"Successfully loaded configuration for '{node_name}' from '{expanded_path}'.")
        return node_params

    except FileNotFoundError:
        _log_warn(node_name, f"Configuration file not found at '{config_file_path}'. Using fallback parameters.")
    except yaml.YAMLError as e:
        _log_error(node_name, f"Error parsing YAML configuration from '{config_file_path}': {e}")
    except Exception as e:
        _log_error(node_name, f"An unexpected error occurred while loading config for '{node_name}': {e}")
    return {}


# Example usage and testing
if __name__ == "__main__":
    # Simulate a message for parsing
    class MockMsg:
        def __init__(self):
            self.data = '{"field1": "value1", "field2": 42}'

    msg = MockMsg()
    fields_map = {
        'desired_key1': ('default1', 'field1', str),
        'desired_key2': ('default2', 'field2', int),
        'missing_key': ('default3', 'missing', str)
    }
    result = parse_message_data(msg, fields_map, "test_node")
    print("Parsed data:", result)

    # Simulate config loading
    config = load_config("test_node", "~/.sentience/config.yaml")
    print("Loaded config:", config)
```
