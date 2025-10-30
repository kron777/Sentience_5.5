```python:disable-run
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
    EthicalDecision = ROSMsgFallback
    CognitionInput = ROSMsgFallback
    EmotionInput = ROSMsgFallback
    HardwareStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    EthicalDecision = ROSMsgFallback
    CognitionInput = ROSMsgFallback
    EmotionInput = ROSMsgFallback
    HardwareStatus = ROSMsgFallback


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
        if hasattr(msg
```
