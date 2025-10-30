#!/usr/bin/env python3
"""
ROS 2 bridge that forwards topics to the stand-alone Integration HTTP service
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import aiohttp
import asyncio
import threading
import json

class IntegrationROS2Bridge(Node):
    def __init__(self):
        super().__init__('integration_node')
        self.declare_parameter("service_url", "http://localhost:8087")
        self.url = self.get_parameter("service_url").value

        # publishers
        self.pub = self.create_publisher(String, 'integration_output', 10)

        # subscribers
        self.subs = [
            self.create_subscription(String, 'decision_making_output', self._forward("decision_making_output"), 10),
            self.create_subscription(String, 'learning_output', self._forward("learning_output"), 10),
            self.create_subscription(String, 'communication_output', self._forward("communication_output"), 10),
            self.create_subscription(String, 'monitoring_output', self._forward("monitoring_output"), 10),
            self.create_subscription(String, 'adaptation_output', self._forward("adaptation_output"), 10),
        ]

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def _forward(self, topic: str):
        def cb(msg: String):
            payload = json.loads(msg.data)
            asyncio.run_coroutine_threadsafe(self._post(topic, payload), self.loop)
        return cb

    async def _post(self, topic: str, payload: dict):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/{topic}", json=payload) as resp:
                pass  # fire-and-forget

        # long-poll for integrated result
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/integration_output") as resp:
                integrated = await resp.json()
                self.pub.publish(String(data=json.dumps(integrated)))

def main(args=None):
    rclpy.init(args=args)
    node = IntegrationROS2Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
