# ROS Libraries
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, PoseStamped
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np
import tf2_geometry_msgs
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
import cv2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

# from moveit_commander import MoveGroupCommander

class Move(Node):
    def __init__(self):
        super().__init__('move_node')

        self.usbc_sub = self.create_subscription(PointStamped, '/usbc_pose', self.usbc_callback, 1)
        self.chessboard_pose_sub = self.create_subscription(PoseStamped, '/calibration/chessboard_pose', self.chessboard_pose_callback, 1)
        
        self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        self.joint_state = None 
        self.chessboard_pose = None
        self.requested_height = False
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path') # Action Client for Execution 
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory') 

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        self.detect_cli = self.create_client(Trigger, '/detect_usbc')
        self.height_cli = self.create_client(Trigger, '/calibrate_height')
        self.spiral_cli = self.create_client(Trigger, '/spiral_search')

        self.current_plan = None
        self.joint_state = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state = "START" 
        self.is_executing = False  # Track if a job is currently executing

        # CV Setup
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_matrix = None
        self.current_usbc_pose = None

        self.create_timer(1, self.run)

        self.x_offset = 0
        self.y_offset = 0

        self.block_height = 0.0274  # height above the chessboard

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_model.fromCameraInfo(msg)
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.camera_model:
            cx, cy = int(self.camera_model.cx()), int(self.camera_model.cy())
            cx_c, cy_c = int(1920/2), int(1080/2)
            cx_2, cy_2 = int(self.camera_model.cx() + self.x_offset), int(self.camera_model.cy() + self.y_offset)
            cv2.circle(img, (cx_c, cy_c), 5, (255, 0, 0), -1) # Blue for principal point
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1) # Blue for principal point
            cv2.circle(img, (cx_2, cy_2), 5, (0, 255, 0), -1) # Green for principal point
        
        if self.current_usbc_pose:
            ux, uy = int(self.current_usbc_pose[0]), int(self.current_usbc_pose[1])
            cv2.circle(img, (ux, uy), 5, (0, 0, 255), -1) # Red for USB-C
        
        cv2.imshow("Camera View", img)
        cv2.waitKey(1)

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def chessboard_pose_callback(self, msg: PoseStamped):
        self.chessboard_pose = msg

    def _request_detection(self):
        """Request USB-C detection from CV service"""
        if not self.detect_cli.wait_for_service(timeout_sec=0.1):
            self.get_logger().warning('Detection service not available', throttle_duration_sec=5.0)
            return
        
        req = Trigger.Request()
        future = self.detect_cli.call_async(req)
        # Don't wait for response, it will publish to topic when ready

    def _request_height(self):
        if not self.height_cli.wait_for_service(timeout_sec=0.1):
            self.get_logger().warning('Height detection service not available', throttle_duration_sec=5.0)
            return
        
        req = Trigger.Request()
        future = self.height_cli.call_async(req)
        # Don't wait for response, it will publish to topic when ready

    def usbc_callback(self, usbc_pose: PointStamped):
        # Check if detection failed (NaN values indicate failure)
        if np.isnan(usbc_pose.point.x) or np.isnan(usbc_pose.point.y):
            self.get_logger().warning("Detection failed (received NaN), back to align")
            self.state = "ALIGN"
            self.current_usbc_pose = None
            return
        
        self.current_usbc_pose = (usbc_pose.point.x, usbc_pose.point.y)
        # self.get_logger().info(f"Received usbc pose: {usbc_pose.point.x}, {usbc_pose.point.y}")

        try:
            # Get transform from base_link to camera_link
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return

        z_depth = -(self.chessboard_pose.pose.position.z + self.block_height - tf.transform.translation.z)
        
        x, y = self.move_pixel(usbc_pose.point.x - self.x_offset, usbc_pose.point.y - self.y_offset, z_depth=z_depth)

        self.get_logger().info(f"Computed move x: {x}, y: {y} at depth z: {z_depth}")
        
        tol = 0.001
        if abs(z_depth) - 0.12 < tol:
            self.get_logger().info("At preshift depth, reducing tolerance")
            tol = 0.0007
        if abs(x) < 0.001 and abs(y) < tol:
            self.get_logger().info("Already aligned within 1mm, skipping move")
            self.state = "DIVE"
            return
        
        self.get_logger().info(f"OFF BY: x: {x:.4f} m, y: {y:.4f} m")

        # Trigger the move
        # self.get_logger().info(f"Moving Robot x: {x}, y: {y}, z: {-0.0}")
        # self.move_robot_relative(dz=0.05, dx=0.05, dy=0.05)
        self.move_robot_relative(dz=-0.0, dx=x, dy=-y)
        self.state = "ALIGN"
    
    def dive(self):
        try:
            # Get transform from base_link to camera_link
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return

        z_depth = -(self.chessboard_pose.pose.position.z + self.block_height - tf.transform.translation.z)
        
        if abs(z_depth) - 0.12 < 0.01:
            self.get_logger().info("Within 1 cm of preshift position, skipping dive")
            self.state = "SHIFT1"
            return

        if z_depth < 0.3:
            self.get_logger().info(f"Diving to preshift position which is some amount above {z_depth:.3f} meters")
            self.move_robot_relative(dz=-z_depth + 0.12, dx=0.0, dy=0.0)
            self.state = "ALIGN"
        else:
            self.get_logger().info(f"Diving {z_depth/2:.3f} meters towards base frame")
            self.move_robot_relative(dz=-z_depth / 2, dx=0.0, dy=0.0)
            self.state = "ALIGN"        

    def shift(self):
        try:
            # Get transform from base_link to camera_link
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_link', rclpy.time.Time())

            camera_to_tool0 = self.tf_buffer.lookup_transform(
                'tool0', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return
        
        if self.chessboard_pose is None:
            return
        
        self.get_logger().info("Shifting above insert_pose")

        z_depth = -(self.chessboard_pose.pose.position.z + self.block_height - tf.transform.translation.z)
        
        self.get_logger().info(f"translations: {camera_to_tool0.transform.translation.x}, {camera_to_tool0.transform.translation.y}")
        if self.state == "SHIFTING2":
            d_x = camera_to_tool0.transform.translation.x # + 0.01 # - 0.058 # + 0.0045 # + 0.0637 
            d_y = camera_to_tool0.transform.translation.y # + 0.01 # + 0.0 # - 0.0045 # + 0.009 
            d_z = -z_depth + 0.1
        elif self.state == "SHIFTING1":
            d_x = 0.01
            d_y = -0.01
            d_z = 0
        self.move_robot_relative(dz=d_z, dx=d_x, dy=d_y, aligned=False) 
        # ee -> usbc, ee -> circle, ee to hole
        # USBC: 0.1, -0.07, 0.063; BIG HOLE: 0.1, 0.003, 0.095; LITTLE HOLE: 0.1, 0.0032, 0.048
        # NEW LITTLE HOLE: z: -d + 0.25, x:0.0048, y:0.033
        if self.state == "SHIFTING1":
            self.state = "SHIFT2"
        elif self.state == "SHIFTING2":
            self.state = "INSERT"

    def insert(self):
        try:
            # Get transform from base_link to camera_link
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return
        
        if self.chessboard_pose is None:
            return

        z_depth = -(self.chessboard_pose.pose.position.z + self.block_height - tf.transform.translation.z)

        self.move_robot_relative(dz=-z_depth + 0.0115, aligned=False) # 0.1, -0.07, 0.063
        self.state = "SPIRAL"
    
    def start(self):
        try:
            # Get transform from base_link to camera_link
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return
        
        if self.chessboard_pose is None:
            return

        self.move_robot_relative(dx=(self.chessboard_pose.pose.position.x - tf.transform.translation.x), 
            dy=(self.chessboard_pose.pose.position.y - tf.transform.translation.y), 
            dz=0
        )

        self.state = "ALIGN"

    def run(self):
        # if self.requested_height is False:
        #     self._request_height()
        #     self.requested_height = True
        #     return

        if self.is_executing:
            return  # Don't queue new jobs while one is executing
        
        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return
        
        if self.chessboard_pose is None:
            self.get_logger().info("No calibration pose yet, cannot proceed")
            return
        
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
             self.get_logger().error("Service /compute_cartesian_path not available! Is the MoveIt server running?")
             return
        
        if self.state == "START":
            self.get_logger().info("Starting alignment process")
            self.start()
        elif self.state == "ALIGN":
            self.state = "ALIGNING" # waiting for usbc callback
            self.get_logger().info("REQUEST USB-C detection, aligning tool")
            self._request_detection()
        
        elif self.state == "DIVE":
            self.state = "DIVING"
            self.dive()
        elif self.state == "SHIFT1":
            self.state = "SHIFTING1"
            self.shift()
        elif self.state == "SHIFT2":
            self.state = "SHIFTING2"
            self.shift()
        elif self.state == "INSERT":
            self.state = "INSERTING"
            self.insert()
        elif self.state == "SPIRAL":
            self.state = "SPIRALING"
            
            self.spiral_cli.wait_for_service(timeout_sec=2.0)
            req = Trigger.Request()
            future = self.spiral_cli.call_async(req)
            self.get_logger().info("Spiral search service called")
            self.state = "DONE"

    def move_pixel(self, u, v, z_depth):
        rectified_uv = self.camera_model.rectifyPoint((u, v))

        ray = self.camera_model.projectPixelTo3dRay(rectified_uv)

        x_metric = ray[0] * z_depth
        y_metric = ray[1] * z_depth
        return x_metric, y_metric
            
    def move_robot_relative(self, dx=0.0, dy=0.0, dz=0.0, aligned=True):
        """
        Moves the robot relative to the BASE frame using a Cartesian path.
        Example: move_robot_relative(dz=0.3048) moves 1 foot up/forward in Z.
        """
        self.is_executing = True
        self.get_logger().info(f"Moving robot relative: dx={dx}, dy={dy}, dz={dz}")
        # 1. Get Current Pose (We need this to calculate the start and end)
        try:
            # Get transform from base_link to tool0
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time())
            
            tf_cam = self.tf_buffer.lookup_transform(
                'base_link', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return

        # 2. Define Waypoints
        # We construct a pose that is offset from the current one
        import copy
        from geometry_msgs.msg import Pose
        
        # Start at current
        current_pose = Pose()
        current_pose.position.x = tf.transform.translation.x
        current_pose.position.y = tf.transform.translation.y
        current_pose.position.z = tf.transform.translation.z
        current_pose.orientation = tf.transform.rotation
        
        # Target Pose (Relative Offset)
        target_pose = copy.deepcopy(current_pose)
        target_pose.position.x += dx + 0.0005
        target_pose.position.y += dy - 0.0005
        target_pose.position.z += dz
        
        if aligned:
            try:
                t_stamped = self.tf_buffer.lookup_transform(
                    'tool0',         # Target frame (End Effector)
                    'camera_link',   # Source frame (Camera)
                    rclpy.time.Time()
                )
                
                # Convert TF to Matrix 
                T_c_e = np.eye(4)
                t = t_stamped.transform.translation
                r = t_stamped.transform.rotation
                T_c_e[:3, 3] = [t.x, t.y, t.z]
                T_c_e[:3, :3] = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
                
                # 2. Convert desired Camera Pose to Matrix (T_base_cam)
                T_base_cam = np.eye(4)
                T_base_cam[:3, 3] = [tf_cam.transform.translation.x, tf_cam.transform.translation.y, tf_cam.transform.translation.z]
                T_base_cam[:3, :3] = R.from_quat([
                    self.chessboard_pose.pose.orientation.x, 
                    self.chessboard_pose.pose.orientation.y, 
                    self.chessboard_pose.pose.orientation.z, 
                    self.chessboard_pose.pose.orientation.w
                ]).as_matrix()

                T_base_ee_matrix = T_base_cam @ np.linalg.inv(T_c_e)
                
                # 4. Convert back to Pose message
                result_pose = Pose()
                result_pose.position.x = T_base_ee_matrix[0, 3]
                result_pose.position.y = T_base_ee_matrix[1, 3]
                result_pose.position.z = T_base_ee_matrix[2, 3]
                
                quat = R.from_matrix(T_base_ee_matrix[:3, :3]).as_quat()
                result_pose.orientation.x = quat[0]
                result_pose.orientation.y = quat[1]
                result_pose.orientation.z = quat[2]
                result_pose.orientation.w = quat[3]
                
                target_pose.orientation = result_pose.orientation
                
            except Exception as e:
                self.get_logger().warning(f"Could not transform chessboard orientation: {e}")
                return
        else:
            try:
                # Get current tool0 to tip transform
                T_e_tip = np.eye(4)
                T_e_tip[:3, 3] = [0, 0, 0.19]  # tip is 19cm below tool0 in +Z direction
                
                # Get current base to tool0 transform
                T_base_e_current = np.eye(4)
                T_base_e_current[:3, 3] = [current_pose.position.x, current_pose.position.y, current_pose.position.z]
                T_base_e_current[:3, :3] = R.from_quat([
                    current_pose.orientation.x,
                    current_pose.orientation.y,
                    current_pose.orientation.z,
                    current_pose.orientation.w
                ]).as_matrix()
                
                # Get current tip position in base frame
                T_base_tip_current = T_base_e_current @ T_e_tip
                current_tip_pos = T_base_tip_current[:3, 3]
                
                # Get desired chessboard orientation (with 180° flip around Z)
                chess_rot = R.from_quat([
                    self.chessboard_pose.pose.orientation.x,
                    self.chessboard_pose.pose.orientation.y,
                    self.chessboard_pose.pose.orientation.z,
                    self.chessboard_pose.pose.orientation.w
                ])
                z_rotation = R.from_euler('z', np.pi)  # 180° rotation
                desired_tip_rot = chess_rot * z_rotation
                
                # Build desired tip transform in base frame
                # Position: current tip position + dx/dy/dz offsets
                T_base_tip_desired = np.eye(4)
                T_base_tip_desired[:3, :3] = desired_tip_rot.as_matrix()
                T_base_tip_desired[:3, 3] = [
                    current_tip_pos[0] + dx + 0.0005,
                    current_tip_pos[1] + dy - 0.0005,
                    current_tip_pos[2] + dz
                ]
                
                # Compute tool0 pose that achieves this tip pose
                # T_base_tip = T_base_e @ T_e_tip
                # So: T_base_e = T_base_tip @ T_e_tip^(-1)
                T_base_e_new = T_base_tip_desired @ np.linalg.inv(T_e_tip)
                
                # Convert back to Pose message
                target_pose.position.x = T_base_e_new[0, 3]
                target_pose.position.y = T_base_e_new[1, 3]
                target_pose.position.z = T_base_e_new[2, 3]
                
                quat = R.from_matrix(T_base_e_new[:3, :3]).as_quat()
                target_pose.orientation.x = quat[0]
                target_pose.orientation.y = quat[1]
                target_pose.orientation.z = quat[2]
                target_pose.orientation.w = quat[3]
                
            except Exception as e:
                self.get_logger().warning(f"Could not transform chessboard orientation: {e}")
                return
            # # Rotate chessboard orientation 180° around Z-axis
            # chess_quat = [
            #     self.chessboard_pose.pose.orientation.x,
            #     self.chessboard_pose.pose.orientation.y,
            #     self.chessboard_pose.pose.orientation.z,
            #     self.chessboard_pose.pose.orientation.w
            # ]
            # chess_rot = R.from_quat(chess_quat)
            
            # # 180° rotation around Z-axis
            # z_rotation = R.from_euler('z', np.pi)  # 180 degrees = π radians
            
            # # Combine rotations
            # rotated = chess_rot * z_rotation
            # rotated_quat = rotated.as_quat()
            
            # target_pose.orientation.x = rotated_quat[0]
            # target_pose.orientation.y = rotated_quat[1]
            # target_pose.orientation.z = rotated_quat[2]
            # target_pose.orientation.w = rotated_quat[3]
        
        # self.get_logger().info(f"curret pose orient: {R.from_quat([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]).as_matrix()}")
        # self.get_logger().info(f"target pose orient: {R.from_quat([target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]).as_matrix()}")
        # self.get_logger().info(f"chessboard pose orient: {R.from_quat([self.chessboard_pose.pose.orientation.x, self.chessboard_pose.pose.orientation.y, self.chessboard_pose.pose.orientation.z, self.chessboard_pose.pose.orientation.w]).as_matrix()}")
        # self.get_logger().info(f"tf_cam orient: {R.from_quat([tf_cam.transform.rotation.x, tf_cam.transform.rotation.y, tf_cam.transform.rotation.z, tf_cam.transform.rotation.w]).as_matrix()}")
        # target_pose.orientation = tf_cam.transform.rotation

        # 3. Call Compute Cartesian Path Service
        req = GetCartesianPath.Request()
        req.header.frame_id = "base_link"
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = "ur_manipulator"
        req.start_state.joint_state = self.joint_state
        req.waypoints = [target_pose] # List of waypoints
        req.max_step = 0.01  # 1 cm resolution
        req.jump_threshold = 0.0 # Disable jump check for simple linear moves
        req.avoid_collisions = True

        self.get_logger().info("Computing Cartesian Path...")

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self.cartesian_path_callback)

    def cartesian_path_callback(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return

        if response.error_code.val!= 1: # 1 is SUCCESS
            self.get_logger().error(f"Path planning failed: {response.error_code.val}")
            return

        # 4. Execute the Trajectory
        fraction = response.fraction
        self.get_logger().info(f"Path computed! Fraction: {fraction:.2f} (1.0 = 100%)")
        
        if fraction > 0.9: # Only move if we found > 90% of the path
            self.execute_trajectory(response.solution)
        else:
            self.get_logger().warn("Path incomplete, aborting move.")

    def execute_trajectory(self, robot_trajectory):
        """Helper to send the computed path to the controller"""
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = robot_trajectory
        
        self.get_logger().info("Sending trajectory to controller...")
        self.execute_client.wait_for_server()
        
        self.send_goal_future = self.execute_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Move complete!')
        self.is_executing = False


def main(args=None):
    rclpy.init(args=args)
    node = Move()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()