#!/usr/bin/env python3
"""
ROS 2 bridge that talks to the stand-alone Insight HTTP service
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import Trigger
import aiohttp
import asyncio
import threading
import json

class InsightROS2Bridge(Node):
    def __init__(self):
        super().__init__('insight_node')
        self.declare_parameter("service_url", "http://localhost:8086")
        self.url = self.get_parameter("service_url").value

        self.publisher_ = self.create_publisher(String, 'insight/suggestions', 10)
        self.srv = self.create_service(Trigger, 'insight/analyze', self.handle_analyze_request)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def handle_analyze_request(self, request, response):
        future = asyncio.run_coroutine_threadsafe(self._call_analyze(), self.loop)
        result = future.result()
        response.success = True
        response.message = json.dumps(result)

        msg = String()
        msg.data = response.message
        self.publisher_.publish(msg)
        return response

    async def _call_analyze(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/analyze") as resp:
                return await resp.json()

def main(args=None):
    rclpy.init(args=args)
    node = InsightROS2Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
