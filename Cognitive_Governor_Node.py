#!/usr/bin/env python3

import sys
import time
from datetime import datetime, timezone

from Self_Model_Node import SelfModelNode
from Conversational_Intelligence_Node import ConversationalIntelligenceNode


# -------------------------------------------------
# Minimal banner (printed once)
# -------------------------------------------------
def banner():
    print()
    print("====================================")
    print("        SENTIENCE 5.5 — LIVE")
    print("====================================")
    print()


# -------------------------------------------------
# Orchestrator
# -------------------------------------------------
class Orchestrator:
    def __init__(self):
        self.self_model = SelfModelNode(silent=True)
        self.chat = ConversationalIntelligenceNode(self.self_model)

        self.self_model.register_node("Self_Model_Node")
        self.self_model.register_node("Conversational_Intelligence_Node")

        banner()

    def run(self):
        """
        Main blocking loop.
        Zero CPU usage while idle.
        """
        while True:
            try:
                user_input = input(">> ").strip()

                if not user_input:
                    continue

                event = {
                    "type": "user_input",
                    "source": "human",
                    "content": user_input,
                    "timestamp": time.time(),
                }

                self.self_model.observe_event(event)

                response = self.chat.respond(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\nShutting down.")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] {e}")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    Orchestrator().run()
