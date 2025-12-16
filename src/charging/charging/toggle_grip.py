import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import Trigger

class ToggleGrip(Node):
    def __init__(self):
        super().__init__('toggle_grip')
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')


    def execute(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        # wait for 2 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')

def main(args=None):
    rclpy.init(args=args)
    toggle_grip = ToggleGrip()
    toggle_grip.execute()
    rclpy.shutdown()