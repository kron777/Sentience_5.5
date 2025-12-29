#!/usr/bin/env python3
"""
Emotion_Mood_Node (UPDATED)
Authoritative affect synthesis node.
LLM is advisory only. Deterministic affect core.
Evolver-compatible.
"""

import os, sys, time, json, sqlite3, threading, argparse, uuid
from typing import Dict, Any, Optional, Deque
from collections import deque
from datetime import datetime

# Async (LLM advisory only)
import asyncio
import aiohttp

# --- ROS optional ---
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass


# =========================
# Logging
# =========================
def _log(level, node, msg):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

def _log_info(n,m): _log("INFO",n,m)
def _log_warn(n,m): _log("WARN",n,m)
def _log_err(n,m):  _log("ERROR",n,m)
def _log_dbg(n,m):  _log("DEBUG",n,m)


# =========================
# Emotion Model (Deterministic Core)
# =========================
class EmotionStateModel:
    """
    Continuous affect model.
    valence ∈ [-1,1]
    arousal ∈ [0,1]
    stability ∈ [0,1]
    """

    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.1
        self.stability = 0.9

    def integrate(self, dv: float, da: float):
        # bounded integration
        self.valence = max(-1.0, min(1.0, self.valence + dv))
        self.arousal = max(0.0, min(1.0, self.arousal + da))

    def decay(self, rate=0.02):
        # emotional settling
        self.valence *= (1.0 - rate)
        self.arousal *= (1.0 - rate)

    def mood_label(self) -> str:
        if self.valence > 0.4:
            return "positive"
        if self.valence < -0.4:
            return "negative"
        return "neutral"


# =========================
# Node
# =========================
class EmotionMoodNode:
    def __init__(self, config_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "emotion_mood_node"
        self.ros_enabled = ros_enabled or os.getenv("ROS_ENABLED","false").lower()=="true"

        _log_info(self.node_name,"Initializing Emotion Mood Node (deterministic core)")

        # --- Core affect model
        self.affect = EmotionStateModel()

        # --- History (bounded, interpretable)
        self.recent_social: Deque[Dict] = deque(maxlen=5)
        self.recent_internal: Deque[Dict] = deque(maxlen=5)
        self.recent_sensory: Deque[Dict] = deque(maxlen=5)

        # --- Learning signals (for evolver)
        self.prediction_error_history: Deque[float] = deque(maxlen=50)

        # --- DB
        db_root = "/tmp/sentience_db"
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "emotion_state.db")
        self._init_db()

        # --- Async LLM (advisory)
        self.llm_enabled = True
        self.llm_model = "phi-2"
        self.llm_url = "http://localhost:8000/v1/chat/completions"
        self.llm_timeout = 15.0

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._session = None

        # --- ROS pubs
        self.pub_state = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_state = rospy.Publisher("/emotion_state", String, queue_size=10)
            rospy.Timer(rospy.Duration(0.5), self._tick)

        _log_info(self.node_name,"Emotion Mood Node online")

    # =========================
    # DB
    # =========================
    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS emotion_state (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            valence REAL,
            arousal REAL,
            stability REAL,
            mood TEXT
        )
        """)
        self.conn.commit()

    def _log_state(self):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO emotion_state VALUES (?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                time.time(),
                self.affect.valence,
                self.affect.arousal,
                self.affect.stability,
                self.affect.mood_label()
            )
        )
        self.conn.commit()

    # =========================
    # Async
    # =========================
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _ensure_session(self):
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def _llm_reflect(self, context: Dict[str,Any]) -> Optional[Dict]:
        """
        Advisory only. Used to *annotate*, not decide.
        """
        await self._ensure_session()

        prompt = f"""
You are an affective reflection assistant.
Given the emotional state and context, suggest:
- adjustment_valence (float -0.2 to 0.2)
- adjustment_arousal (float -0.2 to 0.2)
- note (string)

Context:
{json.dumps(context, indent=2)}
"""

        payload = {
            "model": self.llm_model,
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.3,
            "max_tokens":200
        }

        try:
            async with self._session.post(
                self.llm_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.llm_timeout)
            ) as r:
                j = await r.json()
                txt = j["choices"][0]["message"]["content"]
                return json.loads(txt)
        except Exception as e:
            _log_warn(self.node_name,f"LLM advisory failed: {e}")
            return None

    # =========================
    # Inputs
    # =========================
    def observe_social(self, mood: str, confidence: float):
        self.recent_social.append({"mood":mood,"confidence":confidence})
        dv = 0.0
        if mood=="sad": dv = -0.1*confidence
        if mood=="happy": dv = 0.1*confidence
        self.affect.integrate(dv, 0.05*confidence)

    def observe_internal(self, sentiment: float):
        self.recent_internal.append({"sentiment":sentiment})
        self.affect.integrate(0.1*sentiment, abs(sentiment)*0.05)

    def observe_sensory(self, salience: float):
        self.recent_sensory.append({"salience":salience})
        self.affect.integrate(0.0, salience*0.1)

    # =========================
    # Main tick
    # =========================
    def _tick(self, event=None):
        # natural decay
        self.affect.decay()

        # advisory reflection (rare)
        if self.llm_enabled and random.random()<0.1:
            ctx = {
                "valence": self.affect.valence,
                "arousal": self.affect.arousal,
                "history": {
                    "social": list(self.recent_social),
                    "internal": list(self.recent_internal)
                }
            }
            asyncio.run_coroutine_threadsafe(
                self._apply_llm_reflection(ctx), self._loop
            )

        self._publish()
        self._log_state()

    async def _apply_llm_reflection(self, ctx):
        res = await self._llm_reflect(ctx)
        if not res: return
        dv = float(res.get("adjustment_valence",0.0))
        da = float(res.get("adjustment_arousal",0.0))
        self.affect.integrate(dv, da)

    # =========================
    # Output
    # =========================
    def _publish(self):
        state = {
            "timestamp": time.time(),
            "valence": self.affect.valence,
            "arousal": self.affect.arousal,
            "stability": self.affect.stability,
            "mood": self.affect.mood_label()
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_state:
            self.pub_state.publish(String(data=json.dumps(state)))
        else:
            _log_dbg(self.node_name,f"Emotion state: {state}")

    # =========================
    # Shutdown
    # =========================
    def shutdown(self):
        if self.conn:
            self.conn.close()
        if self._session:
            self._loop.call_soon_threadsafe(self._session.close)
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("shutdown")

    def run(self):
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.spin()
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()
    node = EmotionMoodNode(ros_enabled=args.ros_enabled)
    node.run()
