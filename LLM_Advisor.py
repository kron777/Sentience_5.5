import requests

class LLMAdvisor:
    def __init__(self, url="http://localhost:8000/v1/chat/completions"):
        self.url = url
        self.enabled = True

    def ask(self, context):
        if not self.enabled:
            return None

        try:
            payload = {
                "model": "phi-2",
                "messages": context,
                "temperature": 0.4,
                "max_tokens": 200
            }
            r = requests.post(self.url, json=payload, timeout=2)
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            self.enabled = False
            return None
