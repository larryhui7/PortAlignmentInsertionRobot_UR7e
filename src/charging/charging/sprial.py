# ROS Libraries
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, WrenchStamped, Pose
from moveit_msgs.msg import RobotTrajectory
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from std_srvs.srv import Trigger
from std_msgs.msg import String 
import numpy as np
import time
import threading

class SpiralSearchNode(Node):
    def __init__(self):
        super().__init__('spiral_search_node')

        # 1. THREADING CONFIGURATION
        # Allows nested blocking calls without deadlocks
        self.cb_group = ReentrantCallbackGroup()
        self.motion_lock = threading.Lock()
        
        # 2. SUBSCRIBERS
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1, callback_group=self.cb_group)
        
        self.declare_parameter('force_topic', '/force_torque_sensor_broadcaster/wrench')
        force_topic = self.get_parameter('force_topic').get_parameter_value().string_value
        self.create_subscription(WrenchStamped, force_topic, self.wrench_callback, 10, callback_group=self.cb_group)

        # 3. CLIENTS (MoveIt)
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path', callback_group=self.cb_group)
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory', callback_group=self.cb_group)
        
        # 4. PUBLISHERS (URScript)
        # *** CORRECT TOPIC ***
        self.script_publisher = self.create_publisher(String, '/urscript_interface/script_command', 1)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State
        self.joint_state = None
        self.current_force_z = 0.0

        # Service
        self.service = self.create_service(Trigger, '/spiral_search', self.spiral_search_callback, callback_group=self.cb_group)
        self.get_logger().info('Spiral search node ready (URScript Mode - FIXED).')

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def wrench_callback(self, msg: WrenchStamped):
        self.current_force_z = msg.wrench.force.z

    def spiral_search_callback(self, request, response):
        self.get_logger().info("Spiral search requested")
        
        if self.joint_state is None:
            response.success = False
            response.message = "FAIL: No joint state received"
            return response

        try:
            success, msg = self._perform_spiral_search()
            response.success = success
            response.message = msg
        except Exception as e:
            self.get_logger().error(f"Critical Crash: {e}")
            response.success = False
            response.message = f"CRASH: {str(e)}"

        return response

    def _perform_spiral_search(self, max_radius=0.01, target_force_n=5.0, search_speed=0.02):
        """
        CORRECTED SPIRAL SCRIPT
        Force: 10.0 N (Breaks friction at 100% slider)
        Speed: 0.02 m/s (Visible movement)
        """
        
        # 1. Wait for MoveIt (Sanity check)
        if not self.cartesian_client.wait_for_service(timeout_sec=1.0):
             self.get_logger().warn("MoveIt service not active (Warning only)")

        # 2. GENERATE URSCRIPT - MOVE DOWN UNTIL RESISTANCE, THEN SPIRAL
        ur_script = f"""
def spiral_search():
    textmsg("Starting Spiral Search")
    start = get_actual_tcp_pose()
    zero_ftsensor()
    sleep(0.1)
    force_mode(p[0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,-{target_force_n},0,0,0], 2, [0.1, 0.1, 0.05, 0.5, 0.5, 0.5])
    sleep(0.5)
    ref_pose = get_actual_tcp_pose()
    ref_z = ref_pose[2]
    max_steps = 500
    step_size = 0.001  # 0.5mm - smallest reliable movement
    textmsg("Spiraling with force...")
    spiral_center = ref_pose
    step_count = 0
    side_length = 1
    direction = 0  # 0=right(+x), 1=up(+y), 2=left(-x), 3=down(-y)
    moves_in_direction = 0
    x_offset = 0.0
    y_offset = 0.0
    while step_count < max_steps:
        curr = get_actual_tcp_pose()
        drop = ref_z - curr[2]
        if drop > 0.003:
            textmsg("HOLE FOUND!")
            break
        end
        dx = 0.0
        dy = 0.0
        if direction == 0:
            dx = step_size  # Right
            x_offset = x_offset + step_size
        elif direction == 1:
            dy = step_size  # Up
            y_offset = y_offset + step_size
        elif direction == 2:
            dx = -step_size  # Left
            x_offset = x_offset - step_size
        elif direction == 3:
            dy = -step_size  # Down
            y_offset = y_offset - step_size
        end
        target = pose_trans(spiral_center, p[x_offset, y_offset, 0, 0, 0, 0])
        movel(target, a=1.2, v=0.03, r=0.0)
        moves_in_direction = moves_in_direction + 1
        step_count = step_count + 1
        if moves_in_direction >= side_length:
            moves_in_direction = 0
            direction = (direction + 1) % 4
            if direction == 0 or direction == 2:
                side_length = side_length + 1
            end
        end
    end
    end_force_mode()
    textmsg("Spiral Complete")
end
spiral_search()
"""

        self.get_logger().info("Sending clean URScript to robot...")
        
        # Check subscriber count to verify connection
        sub_count = self.script_publisher.get_subscription_count()
        if sub_count == 0:
            self.get_logger().warn(f"Warning: No listeners on /urscript_interface/script_command. Is the driver running?")
        else:
            self.get_logger().info(f"Verified {sub_count} listener(s) on topic.")

        msg = String()
        msg.data = ur_script
        self.script_publisher.publish(msg)
        
        # 4. MONITOR EXECUTION
        start_time = time.time()
        
        # Get initial Z (approximate)
        start_z = 0.0
        try:
            if self.tf_buffer.can_transform('base_link', 'tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                tf = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                start_z = tf.transform.translation.z
        except Exception as e:
            self.get_logger().warn(f"TF Init Error: {e}")

        self.get_logger().info(f"Monitoring spiral... Start Z: {start_z:.4f}")
        
        # Monitor Loop (120 sec timeout - spiral can take time!)
        iteration = 0
        while (time.time() - start_time) < 120.0:
            time.sleep(0.1)
            iteration += 1
            
            try:
                if self.tf_buffer.can_transform('base_link', 'tool0', rclpy.time.Time()):
                    tf = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                    current_z = tf.transform.translation.z
                    diff = start_z - current_z
                    
                    # Log every 2 seconds
                    if iteration % 20 == 0:
                        elapsed = time.time() - start_time
                        self.get_logger().info(f"t={elapsed:.1f}s | Z={current_z:.4f}m | Drop={diff*1000:.1f}mm | Force={abs(self.current_force_z):.1f}N")

                    # Detection Threshold (3mm drop)
                    if diff > 0.003: 
                        self.get_logger().info(f"✓ HOLE DETECTED! Z-Drop: {diff*1000:.1f}mm")
                        return True, "SUCCESS: Hole Detected via URScript"
            except Exception as e:
                if iteration % 50 == 0:
                    self.get_logger().warn(f"TF error: {e}")

        self.get_logger().warn("Spiral search timed out after 120 seconds")
        return False, "FAIL: Timeout or Spiral finished without insertion"

def main(args=None):
    rclpy.init(args=args)
    node = SpiralSearchNode()
    
    # CRITICAL: MultiThreadedExecutor is required
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


