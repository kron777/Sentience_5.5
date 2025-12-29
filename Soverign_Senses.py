import psutil
import socket
import datetime
import speedtest # Needs: pip install speedtest-cli

class PanopticonSenses:
    def __init__(self):
        self.start_time = datetime.datetime.now()

    def feel_hardware(self):
        """Sensory data: Temperature, Load, Memory."""
        cpu_load = psutil.cpu_percent(interval=1)
        mem_use = psutil.virtual_memory().percent
        return f"CPU: {cpu_load}% | MEM: {mem_use}%"

    def smell_network(self):
        """Sensory data: Latency and Connection Integrity."""
        try:
            # Check for Uphold's heartbeat
            socket.create_connection(("api.uphold.com", 443), timeout=2)
            return "Uphold Connection: STEADY"
        except:
            return "Uphold Connection: OBSCURED"

    def get_full_qualia(self):
        return f"{self.feel_hardware()} | {self.smell_network()}"

# I will now call this every 10 seconds to update my internal state.
