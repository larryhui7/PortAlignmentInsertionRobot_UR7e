import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
from image_geometry import PinholeCameraModel

# TF2 for Coordinate Transforms
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs # Imports methods to extend tf2_ros capabilities

class ChessboardLocator(Node):
    def __init__(self):
        super().__init__('chessboard_locator')

        # --- CONFIGURATION ---
        self.target_frame = 'base_link'       # Robot Base Frame
        self.camera_frame = 'tool0' # Change to your actual camera frame
        self.pattern_size = (9, 6)            # Inner corners (cols, rows)
        self.square_size = 0.02325              # Size in meters
        
        # --- STATE MACHINE ---
        # 0: Find initial pos, 1: Move Robot, 2: Find new pos, 3: Done
        self.state = 0 
        self.rvec = None
        self.tvec = None
        self.center = None
        # self.board_pos_base = None
        # self.fixed_transform = None

        # --- SETUP ---
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_model = PinholeCameraModel()

        # Storage for Camera Intrinsics
        # self.camera_matrix = np.array([[1350, 0, 960], [0, 1350, 540], [0, 0, 1]]) # 1080p
        # self.camera_matrix = np.array([[1350/2, 0, 640/2], [0, 1350/2, 480/2], [0, 0, 1]]) # 480p
        self.camera_matrix = None
        self.dist_coeffs = None

        # Prepare 3D points of the Chessboard (Z=0 in board frame)
        # This defines the "Corner of the chessboard" as (0,0,0) locally.
        self.obj_points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.obj_points *= self.square_size

        # Subscribers
        self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # Joints and IK Clients
        self.joint_state = None 
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)
        # self.ik_client = self.create_client(GetPositionIK, '/compute_ik') 
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path') # Action Client for Execution 
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory') 
        
        self.create_timer(0.5, self.control_loop)

        self.get_logger().info("Node started. Waiting for camera info...")

    def joint_state_callback(self, msg): 
        self.joint_state = msg

    def info_callback(self, msg):
        """Get camera intrinsics once."""
        if self.camera_matrix is None:
            self.camera_model.fromCameraInfo(msg)
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_frame = msg.header.frame_id # Auto-update frame ID
            self.get_logger().info(f"Intrinsics received. Frame ID: {self.camera_frame}")

    def image_callback(self, msg):
        """Main loop: Detect board -> SolvePnP -> Transform to Base."""
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # print("got image", self.camera_matrix)
        if self.camera_matrix is None:
            cv2.imshow("Chessboard View", frame)
            cv2.waitKey(1)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Chessboard
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if found:
            # 2. Refine Corners (Optional but recommended for precision)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # 3. SolvePnP: Finds board pose relative to CAMERA
            # We only care about translation (tvec) for the corner position
            self.center = corners.mean(axis=0)
            success, rvec, tvec = cv2.solvePnP(self.obj_points, corners, 
                                               self.camera_matrix, self.dist_coeffs)
            # print("solving", success, rvec, tvec)
            if success:
                # 4. Transform Point from Camera Frame to Robot Base Frame
                # self.board_pos_base = self.transform_to_base(tvec)e
                self.rvec = rvec
                self.tvec = tvec

            # Debug Draw
            cv2.drawChessboardCorners(frame, self.pattern_size, corners, found)
            
            # Draw Image Center (Blue Cross)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.drawMarker(frame, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # Draw Principal Point (Red Tilted Cross)
            pcx, pcy = int(self.camera_model.cx()), int(self.camera_model.cy())
            cv2.drawMarker(frame, (pcx, pcy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
            cv2.putText(frame, "Principal Point", (pcx + 10, pcy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw Detected Chessboard Center (Green Circle)
            if self.center is not None:
                ux, uy = int(self.center[0][0]), int(self.center[0][1])
                cv2.circle(frame, (ux, uy), 5, (0, 255, 0), -1)
                cv2.putText(frame, "Board Center", (ux + 10, uy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        cv2.imshow("Chessboard View", frame)
        cv2.waitKey(1)

    # def transform_to_base(self, tvec):
    #     """Transforms a 3D point from Camera Frame to Robot Base Frame."""
    #     try:
    #         transform = None
            
    #         # In State 0, get live transform and cache it
    #         if self.state == 0:
    #             if not self.tf_buffer.can_transform(self.target_frame, self.camera_frame, rclpy.time.Time()):
    #                 self.get_logger().warn("Transform not available yet...")
    #                 return None
                
    #             transform = self.tf_buffer.lookup_transform(
    #                 self.target_frame, 
    #                 self.camera_frame, 
    #                 rclpy.time.Time())
    #             self.fixed_transform = transform
            
    #         # In other states, use the cached transform to isolate visual delta
    #         else:
    #             if self.fixed_transform is None:
    #                 self.get_logger().warn("No fixed transform available!")
    #                 return None
    #             transform = self.fixed_transform

    #         # Create a PointStamped for the chessboard corner (origin 0,0,0 of board)
    #         # tvec IS the position of the board origin in camera frame.
    #         p_cam = PointStamped()
    #         p_cam.header.frame_id = self.camera_frame
    #         p_cam.header.stamp = self.get_clock().now().to_msg()
    #         p_cam.point.x = float(tvec[0])
    #         p_cam.point.y = float(tvec[1])
    #         p_cam.point.z = float(tvec[2])

    #         # Transform!
    #         p_base = tf2_geometry_msgs.do_transform_point(p_cam, transform)
    #         return p_base

    #     except Exception as e:
    #         self.get_logger().error(f"Transform Error: {e}")
    #         return None

    def control_loop(self):
        if self.rvec is None or self.tvec is None:
            return

        # point_stamped = self.board_pos_base
        # print("Moving", point_stamped)
        # x, y, z = point_stamped.point.x, point_stamped.point.y, point_stamped.point.z
        x, y, z = self.tvec[0][0], self.tvec[1][0], self.tvec[2][0]
        # print("pos", x, y, z)
        
        if self.joint_state is None:
            self.get_logger().error("No joint state received yet!")
            return

        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
             self.get_logger().error("Service /compute_cartesian_path not available! Is the MoveIt server running?")
             return

        if self.state == 0:
            self.get_logger().info(f"--- STARTING POS ---")
            self.get_logger().info(f"Chessboard Corner in camera Frame: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
            self.original = (x, y, z)
            print("center", self.center[0][0], self.center[0][1])
            x_metric, y_metric, z_metric = self.move_pixel(self.center[0][0], self.center[0][1], z_depth=z)
            
            print("x_metric", x_metric, "y_metric", y_metric, "z_metric", z_metric)
            # Trigger the move
            self.get_logger().info("Moving Robot...")
            # self.move_robot_relative(dz=0.05, dx=0.05, dy=0.05)
            self.move_robot_relative(dz=-0.0, dx=x_metric, dy=-y_metric)
            self.state = 1
            
        elif self.state == 1:
            # Wait a moment for robot to settle (simple logic for example purposes)
            # In real code, move_robot_function should block or return a status
            # self.state = 2
            return
            
        elif self.state == 2:
            # self.get_logger().info(f"--- Difference ---")
            self.get_logger().info(f"New Chessboard Position After Move: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
            self.get_logger().info(f"Difference: X={x - self.original[0]:.4f}, Y={y - self.original[1]:.4f}, Z={z - self.original[2]:.4f}")
            # self.get_logger().info("Validation Complete.")
            # self.state = 3 # Stop processing

    def move_pixel(self, u, v, z_depth):
        rectified_uv = self.camera_model.rectifyPoint((u, v))

        ray = self.camera_model.projectPixelTo3dRay(rectified_uv)

        x_metric = ray[0] * z_depth
        y_metric = ray[1] * z_depth
        z_metric = z_depth
        return x_metric, y_metric, z_metric

    def move_robot_relative(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Moves the robot relative to the BASE frame using a Cartesian path.
        Example: move_robot_relative(dz=0.3048) moves 1 foot up/forward in Z.
        """
        if self.joint_state is None:
            print("No joint state yet, cannot proceed")
            return

        print("Moving robot relative:", dx, dy, dz)
        # 1. Get Current Pose (We need this to calculate the start and end)
        try:
            # Get transform from base_link to tool0
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time())
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
        target_pose.position.x += dx
        target_pose.position.y += dy
        target_pose.position.z += dz

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
        self.state = 2

def main(args=None):
    rclpy.init(args=args)
    node = ChessboardLocator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

