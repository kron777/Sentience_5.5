#!/usr/bin/env python3
from datetime import datetime, timezone
import sys

# Global debug flag
DEBUG = False

def set_debug(enabled: bool):
    """
    Enable or disable console logging globally.
    """
    global DEBUG
    DEBUG = enabled

def log(node: str, level: str, msg: str):
    """
    Internal logging function.
    When DEBUG is False, this produces no console output.
    """
    if not DEBUG:
        return

    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [{node}] [{level}] {msg}", file=sys.stdout)
