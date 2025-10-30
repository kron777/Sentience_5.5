#!/usr/bin/env python3
"""
ROS 2 bridge that forwards topics to the stand-alone Interaction HTTP service
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import aiohttp
import asyncio
import threading
import json

class InteractionROS2Bridge(Node):
    def __init__(self):
        super().__init__('interaction_interface_node')
        self.declare_parameter("service_url", "http://localhost:8088")
        self.url = self.get_parameter("service_url").value

        # publishers
        self.publisher_ = self.create_publisher(String, 'interaction_response', 10)

        # subscribers
        self.sub = self.create_subscription(String, 'control_output', self._forward, 10)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def _forward(self, msg: String):
        payload = json.loads(msg.data)
        asyncio.run_coroutine_threadsafe(self._post_and_poll(payload), self.loop)

    async def _post_and_poll(self, payload: dict):
        # POST to /control_output
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/control_output", json=payload) as resp:
                pass  # fire-and-forget

        # long-poll for response
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/interaction_response") as resp:
                response = await resp.json()
                self.publisher_.publish(String(data=json.dumps(response)))

def main(args=None):
    rclpy.init(args=args)
    node = InteractionROS2Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
