import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import numpy as np
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from charging.keyhole import find_median_position
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class CV(Node):
    def __init__(self, constant=False):
        super().__init__('cv_node')

        self.cb_group = ReentrantCallbackGroup()

        # Publisher for result
        self.usbc_pub = self.create_publisher(PointStamped, '/usbc_pose', 1)
        
        # Service to trigger detection
        self.detect_srv = self.create_service(Trigger, '/detect_usbc', self.detect_callback, callback_group=self.cb_group)
        
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10, callback_group=self.cb_group)
        self.frame = None
        # self.collecting = False
        # self.target_frames = 20

        if constant:
            self.get_logger().info("CV node started in constant detection mode")
            self.create_timer(1.0, self.constant_detection)

        self.get_logger().info("CV service ready")

    def image_callback(self, msg):
        # if self.collecting:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame = cv_image
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def constant_detection(self):
        self.get_logger().info("Detection requested, running...")
        if self.frame is None:
            self.get_logger().error("No image frame available")
            return 
    
        x, y = find_median_position(self.frame)
        self.get_logger().info(f"Detection result: x={x}, y={y}")
        
        point_msg = PointStamped()
        point_msg.header = Header()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        
        if x is not None and y is not None:
            point_msg.point.x = float(x)
            point_msg.point.y = float(y)
            point_msg.point.z = 0.0  # Assuming z=0 for 2D detection
            self.usbc_pub.publish(point_msg)
            self.get_logger().info(f"Published cube pose: ({x}, {y})")

    def detect_callback(self, request, response):
        self.get_logger().info("Detection requested, running...")
        if self.frame is None:
            self.get_logger().error("No image frame available")
            response.success = False
            response.message = "No image frame available"
            return response
        
        # self.frames = []
        # self.collecting = True
        
        # # Wait for frames
        # timeout = 5.0 # seconds
        # start_time = time.time()
        # while len(self.frames) < self.target_frames:
        #     if time.time() - start_time > timeout:
        #         self.get_logger().warning("Timeout waiting for frames")
        #         break
        #     self.get_logger().info(f"Collected {len(self.frames)}/{self.target_frames} frames")
        #     time.sleep(0.1)

            
        # self.collecting = False
        
        # if len(self.frames) == 0:
        #      self.get_logger().error("No frames captured")
        #      response.success = False
        #      response.message = "No frames captured"
        #      return response

        x, y = find_median_position(self.frame)
        self.get_logger().info(f"Detection result: x={x}, y={y}")
        
        point_msg = PointStamped()
        point_msg.header = Header()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        
        if x is not None and y is not None:
            point_msg.point.x = float(x)
            point_msg.point.y = float(y)
            point_msg.point.z = 0.0  # Assuming z=0 for 2D detection
            self.usbc_pub.publish(point_msg)
            self.get_logger().info(f"Published cube pose: ({x}, {y})")
            response.success = True
            response.message = f"Detected at ({x}, {y})"
        else:
            # Publish failure message with NaN to trigger callback
            point_msg.point.x = float('nan')
            point_msg.point.y = float('nan')
            point_msg.point.z = float('nan')
            self.usbc_pub.publish(point_msg)
            self.get_logger().warning("Detection failed, published NaN")
            response.success = False
            response.message = "Failed to detect USB-C position"
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CV(constant=False)
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()