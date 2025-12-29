#!/usr/bin/env python3
import time
from collections import deque

class PredictionNode:
    def __init__(self):
        self.node_name = "Prediction_Node"
        self.last_prediction = None
        self.history = deque(maxlen=100)

    def predict(self, user_input: str) -> dict:
        prediction = {
            "expects_question": "?" in user_input,
            "topic": "existential" if any(
                k in user_input.lower()
                for k in ["alive", "conscious", "sentient", "self"]
            ) else "general",
            "timestamp": time.time()
        }
        self.last_prediction = prediction
        return prediction

    def evaluate(self, next_input: str) -> dict:
        if not self.last_prediction:
            return {"status": "no_prediction"}

        result = {
            "success": self.last_prediction["expects_question"] == ("?" in next_input),
            "prediction": self.last_prediction,
            "actual": next_input,
            "timestamp": time.time()
        }

        self.history.append(result)
        self.last_prediction = None
        return result

    def accuracy(self):
        if not self.history:
            return None
        return sum(1 for r in self.history if r["success"]) / len(self.history)
