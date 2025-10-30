#!/usr/bin/env python3
"""
ROS 2 bridge that talks to the stand-alone Ingenuity HTTP service
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import Trigger
import aiohttp
import asyncio
import threading
import json

class IngenuityROS2Bridge(Node):
    def __init__(self):
        super().__init__('ingenuity_node')
        self.declare_parameter("service_url", "http://localhost:8085")
        self.url = self.get_parameter("service_url").value

        self.pub = self.create_publisher(String, 'ingenuity/status', 10)
        self.srv = self.create_service(Trigger, 'ingenuity/create_node', self.handle_create_node)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def handle_create_node(self, request, response):
        future = asyncio.run_coroutine_threadsafe(self._call_create(), self.loop)
        result = future.result()
        response.success = result["success"]
        response.message = result["message"]

        msg = String()
        msg.data = response.message
        self.pub.publish(msg)
        return response

    async def _call_create(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/create_node", json={
                "node_name": "DemoNode",
                "specification": "Demo node created by ROS 2 bridge."
            }) as resp:
                return await resp.json()

def main(args=None):
    rclpy.init(args=args)
    node = IngenuityROS2Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
