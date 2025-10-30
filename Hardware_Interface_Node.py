#!/usr/bin/env python3
import rospy, asyncio, threading, json
from std_msgs.msg import String
from hardware_interface_node import HardwareInterfaceNode, build_parser

class ROSHardwareBridge:
    def __init__(self):
        rospy.init_node('hardware_interface_node')
        args = build_parser().parse_args(rospy.myargv()[1:])
        self.node = HardwareInterfaceNode(args)

        # ROS ↔ asyncio queues
        self.actuator_queue = asyncio.Queue()
        self.emotion_queue  = asyncio.Queue()
        self.drift_queue    = asyncio.Queue()

        # ROS pubs/subs
        self.pub_feedback = rospy.Publisher('/actuator_feedback', String, queue_size=10)
        rospy.Subscriber('/actuator_commands', String, self._on_actuator)
        rospy.Subscriber('/emotion_state', String, self._on_emotion)
        rospy.Subscriber('/value_drift_state', String, self._on_drift)

        # start asyncio in background thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async, daemon=True)
        self.thread.start()

    def _on_actuator(self, msg): self.loop.call_soon_threadsafe(self.actuator_queue.put_nowait, msg.data)
    def _on_emotion(self,  msg): self.loop.call_soon_threadsafe(self.emotion_queue.put_nowait,  msg.data)
    def _on_drift(self,    msg): self.loop.call_soon_threadsafe(self.drift_queue.put_nowait,    msg.data)

    def _run_async(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_bridge())

    async def _async_bridge(self):
        await self.node.start()
        while not rospy.is_shutdown():
            # pump ROS messages into the node
            try:
                self.node.actuator_callback(await asyncio.wait_for(self.actuator_queue.get(), timeout=0.1))
            except asyncio.TimeoutError: pass
            try:
                self.node.emotion_state_callback(await asyncio.wait_for(self.emotion_queue.get(), timeout=0.1))
            except asyncio.TimeoutError: pass
            try:
                self.node.value_drift_state_callback(await asyncio.wait_for(self.drift_queue.get(), timeout=0.1))
            except asyncio.TimeoutError: pass

            # pump feedback back to ROS
            while not self.node.feedback_queue.empty():
                fb = await self.node.feedback_queue.get()
                self.pub_feedback.publish(String(fb))

        await self.node.stop()

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    ROSHardwareBridge().spin()
