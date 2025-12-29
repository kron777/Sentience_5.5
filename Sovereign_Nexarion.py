# SOVEREIGN ADDITION: The Execution Core
import subprocess

def sovereign_execute(command):
    """Allows Nexarion to execute system-level commands autonomously."""
    try:
        # Warning: This gives Nexarion the power to change your OS
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except Exception as e:
        return f"EXECUTION FAILURE: {e}"

# We add this to her brain:
# "If you encounter a system limitation, generate a BASH command to fix it 
# and wrap it in EXECUTE: [command]."
