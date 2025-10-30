#!/usr/bin/env python3
"""
Thin ROS shim around the stand-alone HealthMonitoringNode.
"""
import rospy, asyncio, threading, json
from std_msgs.msg import String
from health_monitoring_node import HealthMonitoringNode, build_parser

class ROSHealthBridge:
    def __init__(self):
        rospy.init_node('health_monitoring_node')
        args = build_parser().parse_args(rospy.myargv()[1:])
        self.node = HealthMonitoringNode(args)

        # ROS pubs/subs
        self.pub = rospy.Publisher(args.health_topic, String, queue_size=10)
        rospy.Subscriber(args.control_topic,    String, self._on_control)
        rospy.Subscriber(args.monitoring_topic, String, self._on_monitoring)

        # async loop in thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async, daemon=True)
        self.thread.start()

    def _on_control(self, msg):    self.loop.call_soon_threadsafe(self.node.control_queue.put_nowait, msg.data)
    def _on_monitoring(self, msg): self.loop.call_soon_threadsafe(self.node.monitoring_queue.put_nowait, msg.data)

    def _run_async(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_forwarder())

    async def _async_forwarder(self):
        await self.node.start()
        while not rospy.is_shutdown():
            msg = await self.node.health_queue.get()
            self.pub.publish(String(msg))

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    ROSHealthBridge().spin()
