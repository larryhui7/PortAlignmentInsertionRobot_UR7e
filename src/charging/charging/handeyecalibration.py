import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from image_geometry import PinholeCameraModel
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation as R
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import time

class HandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration_node')

        # --- CONFIGURATION ---
        self.pattern_size = (9, 6)
        self.square_size = 0.02325
        self.base_frame = 'base_link'       # Robot Base
        self.ee_frame = 'tool0'             # Robot End Effector (Gripper/Flange)
        
        # Strategy: "Super-Sphere" (24 Steps)
        # Maximizes rotational diversity for high-accuracy calibration.
        # Max Reach: 0.12m (Safe). Max Drop: 0.15m.
        # Includes Counter-Moves for 5cm Camera Y-Offset.


        self.motion_sequence = [
#             # --- START ---
            [0.0, 0.0, 0.0,  0.0,  0.0,  0.0],  # 1. Home Capture
            [0.0, 0.0, 0.0,  0.0,  0.0,  0.0],  # 1. Home Capture

            # 26. Twist Top-Left
            # Action: Small Tilt Down/Right + Heavy Roll
            # Compensation: Robot moves +X slightly to counter the camera roll swing.
            [0.015, 0.0, 0.0,  0.08, 0.08,  0.3],


            # 27. Return Home
            [-0.015, 0.0, 0.0, -0.08, -0.08, -0.3],


            # 28. Twist Bottom-Right
            # Action: Small Tilt Up/Left + Heavy Roll
            [0.015, 0.0, 0.0, -0.08, -0.08,  0.3],


            # 29. Return Home
            [-0.015, 0.0, 0.0,  0.08,  0.08, -0.3],


            # 30. Twist Top-Right
            # Action: Small Tilt Down/Left + Heavy Opposite Roll (-0.3)
            # Compensation: Robot moves -X slightly to counter the clockwise roll swing.
            [-0.015, 0.0, 0.0,  0.08, -0.08, -0.3],


            # 31. Return Home
            [0.015, 0.0, 0.0, -0.08,  0.08,  0.3],


            # 32. Twist Bottom-Left
            # Action: Small Tilt Up/Right + Heavy Opposite Roll (-0.3)
            [-0.015, 0.0, 0.0, -0.08,  0.08, -0.3],


            # 33. Final Home (Reset)
            [0.015, 0.0, 0.0,  0.08, -0.08,  0.3],



            # --- SET 1: PURE ROLLS (The "Up" Vector) ---
            # 2. Roll +45 deg (Counter-Clockwise)
            # Compensate: Move Right (+X) & Out (+Y)
            [0.035, 0.015, 0.0,  0.0,  0.0,  0.78],
            
            # 3. Return Home (Undo)
            [-0.035, -0.015, 0.0,  0.0,  0.0, -0.78],
            
            # 4. Roll -45 deg (Clockwise)
            # Compensate: Move Left (-X) & Out (+Y)
            [-0.035, 0.015, 0.0,  0.0,  0.0, -0.78],
            
            # 5. Return Home (Undo)
            [0.035, -0.015, 0.0,  0.0,  0.0,  0.78], 


            # --- SET 2: CARDINALS (N, S, E, W) ---
            # 6. East: Move Right 12cm, Tilt "Left" (Ry)
            [0.12, 0.0, 0.0,  0.0, -0.20,  0.0],
            
            # 7. Return Home
            [-0.12, 0.0, 0.0,  0.0,  0.20,  0.0],
            
            # 8. West: Move Left 12cm, Tilt "Right" (Ry)
            [-0.12, 0.0, 0.0,  0.0,  0.20,  0.0],
            
            # 9. Return Home
            [0.12, 0.0, 0.0,  0.0, -0.20,  0.0],


            # 10. North: Move Up (Y+) 12cm, Tilt "Down" (Rx)
            # (Previously failed at 20cm, safe at 12cm)
            [0.0, 0.12, 0.0,  0.20,  0.0,  0.0],
            
            # 11. Return Home
            [0.0, -0.12, 0.0, -0.20,  0.0,  0.0],
            
            # 12. South: Move Down (Y-) 12cm, Tilt "Up" (Rx)
            [0.0, -0.12, 0.0, -0.20,  0.0,  0.0],
            
            # 13. Return Home
            [0.0, 0.12, 0.0,  0.20,  0.0,  0.0],


            # --- SET 3: CORNERS (Diagonal Constraints) ---
            # 14. North-East: Move Right & Up, Tilt Down-Left
            [0.08, 0.08, 0.0,  0.15, -0.15, 0.0],
            
            # 15. Return Home
            [-0.08, -0.08, 0.0, -0.15,  0.15, 0.0],


            # 16. South-West: Move Left & Down, Tilt Up-Right
            [-0.08, -0.08, 0.0, -0.15,  0.15, 0.0],
            
            # 17. Return Home
            [0.08, 0.08, 0.0,  0.15, -0.15, 0.0],


            # 18. North-West: Move Left & Up, Tilt Down-Right
            [-0.08, 0.08, 0.0,  0.15,  0.15, 0.0],
            
            # 19. Return Home
            [0.08, -0.08, 0.0, -0.15, -0.15, 0.0],


            # 20. South-East: Move Right & Down, Tilt Up-Left
            [0.08, -0.08, 0.0, -0.15, -0.15, 0.0],
            
            # 21. Return Home
            [-0.08, 0.08, 0.0,  0.15,  0.15, 0.0],


            # --- SET 4: DEPTH & TWIST (The "Screw") ---
            # 22. Dive: Drop Z 15cm, Roll Camera 30 deg
            # Helps separate Depth from Focal Length in the solver
            [0.02, 0.02, -0.05,  0.0,  0.0,  0.5],
            
            # 23. Return Home
            [-0.02, -0.02, 0.05,  0.0,  0.0, -0.5],
            
            # 24. Lift: Raise Z 10cm, Opposite Roll
            [0.0, 0.0, 0.10,  0.0,  0.0, -0.3],


            # 25. Final Home Return (Must be perfect 0,0,0)
            [0.0, 0.0, -0.10,  0.0,  0.0,  0.3],

#             # --- PHASE 5: THE Z-AXIS LOCKER (The "Clock Face") ---
#             # Goal: Put the chessboard in the extreme corners of the image 
#             # using ROTATION, not translation. This decouples Intrinsics from Extrinsics.
            

# # --- PHASE 5: THE "SAFE" Z-AXIS LOCKER ---
#             # Goal: Gently place target in corners using Rotation to lock intrinsics.
#             # Reduced magnitudes to prevent "Out of Frame" errors.

        ]

        # Merge pairs of motions (move + return home) into single combined motions
        merged_sequence = [self.motion_sequence[0]]  # Keep the initial home position
        for i in range(1, len(self.motion_sequence), 2):
            if i + 1 < len(self.motion_sequence):
                # Add elements element-wise for each pair
                merged_motion = [
                    self.motion_sequence[i][j] + self.motion_sequence[i + 1][j] 
                    for j in range(6)
                ]
                merged_sequence.append(merged_motion)
            else:
                # If odd number of motions, keep the last one as-is
                merged_sequence.append(self.motion_sequence[i])
        
        self.motion_sequence = merged_sequence


        # Calibration Motions: [x, y, z, roll, pitch, yaw] INCREMENTAL relative to previous pose
        # Strategy: Move laterally and tilt back toward chessboard for rotation diversity
        # Each move is relative to the PREVIOUS position, not the start
        # self.motion_sequence = [
        #     [0.0, 0.0, 0.0,  0.0,  0.0,  0.0],      # 1. Start pose (capture)
        #     [0.0, 0.15, 0.0,  0.2,  0.0,  0.0],     # 2. Tilt roll +15°
        #     [0.0, -0.4, -0.2, -0.65,  0.0,  0.0],     # 3. Tilt roll to -15° (undo +15, add -15)
        #     [0.0, 0.25, 0.2,  0.45,  0.15,  0.0],    # 4. Back to center roll, tilt pitch +15°
        #     [0.0, 0.0, 0.0,  0.0, -0.40,  0.0],     # 5. Tilt pitch to -15°
        #     [0.0, 0.0, 0.1,  0.0, 0.25,  0.3],     # 5. Tilt pitch to -15°
        #     [0.0, 0.0, -0.2,  0.0, 0.0,  -0.6],     # 6. Tilt pitch to -15°
        #     [0.15, 0.0, 0.1,  0.0,  -0.2, 0.3],   # 7. Back to center pitch, move right 10cm, yaw left 20°
        #     [0.0, 0.1, 0.0,  0.15,  0.0,  0.0],     # 8. Same position, add roll tilt +15°
        #     [0.0, -0.2, 0.0, -0.30,  0.0,  0.0],     # 9. Change roll to -15°
        #     [-0.3, 0.1, 0.0,  0.15,  0.4,  0.0],    # 10. Back to center roll, pitch up +15°
        #     [0.0, 0.1, 0.0,  0.15,  0.0,  0.2],     # 8. Same position, add roll tilt +15°
        #     [0.0, -0.2, 0.0, -0.30,  0.0,  -0.4],     # 9. Change roll to -15°
        #     [0.15, 0.1, 0.0, 0.15,  -0.2,  0.2],     # 9. Change roll to -15°
        # ]
        
        # --- STATE ---
        self.state = "IDLE" # IDLE, MOVING, WAITING, WAITING_NEW_IMAGE, CAPTURING, CALIBRATING, DONE
        self.current_step = 0
        self.latest_image_msg = None
        self.joint_state = None  # To be filled from TF or another source
        self.settle_start_time = None  # For timing robot settle period
        self.motion_end_time = None  # Time when robot motion completed
        
        # --- DATA STORAGE ---
        self.R_gripper2base = [] # Robot Rotation
        self.t_gripper2base = [] # Robot Translation
        self.R_target2cam = []   # Chessboard Rotation
        self.t_target2cam = []   # Chessboard Translation
        self.image_corners = []  # Store detected corners for final reprojection error

        # --- SETUP ---
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Prepare 3D points
        self.obj_points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.obj_points *= self.square_size

        # Subscribers
        self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path') # Action Client for Execution 
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory') 
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.calibration_client = self.create_client(Trigger, '/calibrate_height')
        
        # Publisher for chessboard pose
        self.chessboard_pose_pub = self.create_publisher(PoseStamped, '/calibration/chessboard_pose', 10)
        self.chessboard_pose_msg = None  # Store computed pose for continuous publishing
        
        # Store camera transform for continuous publishing
        self.camera_transform_msg = None
        
        # Timer to continuously publish chessboard pose (10Hz)
        self.pose_publish_timer = self.create_timer(0.1, self.publish_chessboard_pose)
        
        # Timer to republish static camera transform (1Hz)
        self.tf_republish_timer = self.create_timer(1.0, self.republish_camera_transform)

        # Service
        self.srv = self.create_service(Trigger, '/handeye_calibration', self.trigger_callback)
        
        # Control Loop Timer (1Hz)
        self.timer = self.create_timer(1.0, self.control_loop)
        
        # Auto-start timer
        self.start_timer = self.create_timer(2.0, self.auto_start)

        self.get_logger().info("Hand-Eye Node Ready. Will auto-start in 2 seconds.")

    def auto_start(self):
        self.start_timer.cancel()
        if self.state == "IDLE":
            self.state = "MOVING"
            self.current_step = 0
            self.R_gripper2base = []
            self.t_gripper2base = []
            self.R_target2cam = []
            self.t_target2cam = []
            self.get_logger().info("Auto-starting calibration sequence...")

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg
        
    def trigger_callback(self, request, response):
        if self.state == "IDLE":
            self.state = "MOVING"
            self.current_step = 0
            self.R_gripper2base = []
            self.t_gripper2base = []
            self.R_target2cam = []
            self.t_target2cam = []
            response.success = True
            response.message = "Starting calibration sequence..."
            self.get_logger().info("Starting Sequence...")
        else:
            response.success = False
            response.message = f"Busy. Current state: {self.state}"
        return response

    def control_loop(self):
        """State Machine for the Calibration Process"""
        
        if self.state == "MOVING":
            if self.current_step >= len(self.motion_sequence):
                self.state = "CALIBRATING"
                return

            # Execute Move
            move_cmd = self.motion_sequence[self.current_step]
            self.move_robot_relative(move_cmd)
            self.state = "WAITING"

        elif self.state == "WAITING":
            # Don't do anything, wait for robot motion to complete
            pass

        elif self.state == "WAITING_NEW_IMAGE":
            # Ensure we have a fresh image after robot stopped
            if self.settle_start_time is None:
                self.settle_start_time = self.get_clock().now()
            
            elapsed = (self.get_clock().now() - self.settle_start_time).nanoseconds / 1e9
            if elapsed >= 2.5:  # Wait 2.5 seconds for robot to fully settle
                self.settle_start_time = None
                self.state = "CAPTURING"

        elif self.state == "CAPTURING":
            success = self.process_capture()
            if success:
                self.get_logger().info(f"Captured Sample {self.current_step + 1}/{len(self.motion_sequence)}")
                self.current_step += 1
                self.state = "MOVING"
            else:
                self.get_logger().warn("Chessboard not found or quality too low! Retrying same pose...")
                self.state = "WAITING_NEW_IMAGE"
                # Optional: Add logic to abort if too many failures
        
        elif self.state == "CALIBRATING":
            self.perform_calibration()
            self.state = "DONE"

    def process_capture(self):
        """1. Get TF (Base->Gripper), 2. Detect Board (Cam->Target)"""
        if self.latest_image_msg is None or self.camera_matrix is None:
            return False

        # --- 1. Get Robot Pose (Base -> End Effector) ---
        try:
            # Get the latest transform (after robot has settled)
            # We use Time(0) to get the most recent available transform
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, 
                self.ee_frame, 
                rclpy.time.Time(),  # Time(0) means "get latest available"
                timeout=Duration(seconds=1.0))
            
            # Verify the transform is from AFTER the motion ended
            # This ensures we're using the robot's settled position, not an old transform
            if self.motion_end_time is None:
                self.get_logger().error("motion_end_time not set!")
                return False
                
            tf_time = rclpy.time.Time.from_msg(trans.header.stamp)
            time_since_motion_end = (tf_time - self.motion_end_time).nanoseconds / 1e9
            
            if time_since_motion_end < 1.0:
                self.get_logger().warn(f"Transform is from {time_since_motion_end:.2f}s after motion ended (need >1s), retrying...")
                return False
            
            current_time = self.get_clock().now()
            age = (current_time - tf_time).nanoseconds / 1e9
            self.get_logger().info(f"Got TF at {time_since_motion_end:.2f}s after motion ended (age: {age:.3f}s) ✓")
            
            # Extract Rotation Matrix and Translation Vector
            q = [trans.transform.rotation.x, trans.transform.rotation.y, 
                 trans.transform.rotation.z, trans.transform.rotation.w]
            rmat = R.from_quat(q).as_matrix()
            tvec = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]

            temp_R_grip = rmat
            temp_t_grip = np.array(tvec).reshape(3,1)

        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return False

        # --- 2. Get Camera Pose (Camera -> Chessboard) ---
        frame = self.bridge.imgmsg_to_cv2(self.latest_image_msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if not found:
            return False

        # Refine corner detection with stricter criteria for sub-millimeter accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        success, rvec_cam, tvec_cam = cv2.solvePnP(self.obj_points, corners, 
                                                   self.camera_matrix, self.dist_coeffs,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            return False

        # Convert rvec to Rotation Matrix
        rmat_cam, _ = cv2.Rodrigues(rvec_cam)
        
        # Ensure chessboard Z-axis points SAME direction as camera Z-axis (both pointing outward/forward)
        # The third column of rmat_cam is the chessboard's Z-axis in camera frame
        # Camera Z points out of the camera (+Z forward), chessboard Z should also point forward (+Z)
        z_axis = rmat_cam[:, 2]
        if z_axis[2] < 0:  # If Z points toward camera (negative Z), flip 180° around chessboard's Z
            self.get_logger().info("Flipping chessboard orientation for consistency (Z-axis now points same direction as camera)")
            flip_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            rmat_cam = rmat_cam @ flip_matrix
            # Update rvec for reprojection check
            rvec_cam, _ = cv2.Rodrigues(rmat_cam)

        # Verify detection quality via reprojection error
        projected_corners, _ = cv2.projectPoints(
            self.obj_points, rvec_cam, tvec_cam,
            self.camera_matrix, self.dist_coeffs)
        reprojection_error = cv2.norm(corners, projected_corners, cv2.NORM_L2) / len(projected_corners)
        
        if reprojection_error > 0.5:  # Reject if error > 0.5 pixels
            self.get_logger().warn(f"High reprojection error: {reprojection_error:.3f}px, rejecting sample")
            return False
        
        self.get_logger().info(f"Reprojection error: {reprojection_error:.3f}px ✓")

        # Note: rmat_cam already computed and orientation-corrected above
        # Live feed in image_callback() already shows the chessboard detection

        # Store Data
        self.R_gripper2base.append(temp_R_grip)
        self.t_gripper2base.append(temp_t_grip)
        self.R_target2cam.append(rmat_cam)
        self.t_target2cam.append(tvec_cam)
        self.image_corners.append(corners)  # Store for final reprojection error
        
        return True

    def perform_calibration(self):
        self.get_logger().info("Computing Hand-Eye Calibration...")

        # cv2.calibrateHandEye expects lists of 3x3 R matrices and 3x1 t vectors
        # Method: CALIB_HAND_EYE_TSAI is robust standard
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                self.R_gripper2base,
                self.t_gripper2base,
                self.R_target2cam,
                self.t_target2cam,
                method=cv2.CALIB_HAND_EYE_DANIILIDIS
            )

            self.get_logger().info("\n" + "="*30)
            self.get_logger().info("CALIBRATION SUCCESSFUL")
            self.get_logger().info("="*30)
            
            # Construct 4x4 Homogeneous Matrix
            # H = np.eye(4)
            # H[:3, :3] = R_cam2gripper
            # H[:3, 3] = t_cam2gripper.flatten()
            
            # Print Python-ready numpy array
            np.set_printoptions(suppress=True, precision=6)
            self.get_logger().info("\n# Copy this into your code:")
            self.get_logger().info(f"R_camera_mount = np.array({np.array2string(R_cam2gripper, separator=',')})")
            self.get_logger().info(f"t_camera_mount = np.array({np.array2string(t_cam2gripper.flatten(), separator=',')})")
            # print(f"\nFull Matrix:\n{H}")

            # Publish transform
            transform_msg = TransformStamped()
            transform_msg.header.stamp = self.get_clock().now().to_msg()
            transform_msg.header.frame_id = self.ee_frame
            transform_msg.child_frame_id = 'camera_link'
            
            t_flat = t_cam2gripper.flatten()
            transform_msg.transform.translation.x = float(t_flat[0])
            transform_msg.transform.translation.y = float(t_flat[1])
            transform_msg.transform.translation.z = float(t_flat[2])
            
            quat = R.from_matrix(R_cam2gripper).as_quat() # [x, y, z, w]
            transform_msg.transform.rotation.x = float(quat[0])
            transform_msg.transform.rotation.y = float(quat[1])
            transform_msg.transform.rotation.z = float(quat[2])
            transform_msg.transform.rotation.w = float(quat[3])
            
            # Store for continuous publishing
            self.camera_transform_msg = transform_msg
            self.tf_broadcaster.sendTransform(transform_msg)

            self.get_logger().info("Published static transform from camera_link to tool0 (will republish continuously)")

            # Compute and publish chessboard pose in base frame
            chessboard_R, chessboard_t = self.compute_chessboard_in_base(R_cam2gripper, t_cam2gripper)
            
            if chessboard_R is not None:
                # Create and store pose message for continuous publishing
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'base_link'
                
                pose_msg.pose.position.x = float(chessboard_t[0])
                pose_msg.pose.position.y = float(chessboard_t[1])
                pose_msg.pose.position.z = float(chessboard_t[2])
                
                chess_quat = R.from_matrix(chessboard_R).as_quat()
                pose_msg.pose.orientation.x = float(chess_quat[0])
                pose_msg.pose.orientation.y = float(chess_quat[1])
                pose_msg.pose.orientation.z = float(chess_quat[2])
                pose_msg.pose.orientation.w = float(chess_quat[3])
                
                # Store for continuous publishing by timer
                self.chessboard_pose_msg = pose_msg
                
                self.get_logger().info("Chessboard pose computed - will publish continuously to /calibration/chessboard_pose")
                
                # # Also publish as TF for visualization
                # chess_tf = TransformStamped()
                # chess_tf.header.stamp = self.get_clock().now().to_msg()
                # chess_tf.header.frame_id = 'base_link'
                # chess_tf.child_frame_id = 'chessboard_origin'
                
                # chess_tf.transform.translation.x = float(chessboard_t[0])
                # chess_tf.transform.translation.y = float(chessboard_t[1])
                # chess_tf.transform.translation.z = float(chessboard_t[2])
                # chess_tf.transform.rotation.x = float(chess_quat[0])
                # chess_tf.transform.rotation.y = float(chess_quat[1])
                # chess_tf.transform.rotation.z = float(chess_quat[2])
                # chess_tf.transform.rotation.w = float(chess_quat[3])
                
                # self.tf_broadcaster.sendTransform(chess_tf)
                # self.get_logger().info("Published chessboard TF frame for RViz visualization")

            # Compute final reprojection error across all calibration samples
            self.compute_final_reprojection_error(R_cam2gripper, t_cam2gripper)

            # Trigger next calibration step
            # if self.calibration_client.wait_for_service(timeout_sec=1.0):
            #     req = Trigger.Request()
            #     self.future = self.calibration_client.call_async(req)
            #     self.get_logger().info("Triggered /calibrate_height")
            # else:
            #     self.get_logger().warn("/calibrate_height service not available")

        except Exception as e:
            self.get_logger().error(f"Calibration Calculation Failed: {e}")

    def compute_final_reprojection_error(self, R_cam2gripper, t_cam2gripper):
        """
        Compute the overall reprojection error across all calibration samples.
        This verifies the quality of the hand-eye calibration by re-projecting
        the detected chessboard corners and comparing to the original observations.
        """
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info("FINAL REPROJECTION ERROR ANALYSIS")
        self.get_logger().info("="*50)
        
        total_error_squared = 0.0
        total_points = 0
        sample_errors = []
        max_error = 0.0
        
        for i in range(len(self.R_gripper2base)):
            # Use the detected camera-to-chessboard transform to reproject
            rvec_cam, _ = cv2.Rodrigues(self.R_target2cam[i])
            tvec_cam = self.t_target2cam[i]
            
            # Project chessboard 3D points to image plane
            projected_corners, _ = cv2.projectPoints(
                self.obj_points, rvec_cam, tvec_cam,
                self.camera_matrix, self.dist_coeffs)
            
            # Compute error against stored observed corners
            observed_corners = self.image_corners[i]
            
            # Calculate per-point errors
            diff = observed_corners - projected_corners
            errors_squared = np.sum(diff**2, axis=2).flatten()
            errors = np.sqrt(errors_squared)
            
            sample_error = np.mean(errors)
            sample_errors.append(sample_error)
            
            total_error_squared += np.sum(errors_squared)
            total_points += len(errors)
            
            max_error = max(max_error, np.max(errors))
            
        # Compute overall RMS error
        rms_error = np.sqrt(total_error_squared / total_points)
        mean_sample_error = np.mean(sample_errors)
        std_sample_error = np.std(sample_errors)
        
        self.get_logger().info(f"\nTotal samples: {len(self.R_gripper2base)}")
        self.get_logger().info(f"Total points: {total_points} ({len(self.obj_points)} per sample)")
        self.get_logger().info(f"\n{'='*50}")
        self.get_logger().info(f"Overall RMS Reprojection Error: {rms_error:.4f} pixels")
        self.get_logger().info(f"Mean per-sample error: {mean_sample_error:.4f} ± {std_sample_error:.4f} pixels")
        self.get_logger().info(f"Maximum point error: {max_error:.4f} pixels")
        self.get_logger().info(f"{'='*50}")
        
        if rms_error < 0.3:
            self.get_logger().info("✓ EXCELLENT! RMS error < 0.3 pixels")
        elif rms_error < 0.5:
            self.get_logger().info("✓ GOOD! RMS error < 0.5 pixels")
        elif rms_error < 1.0:
            self.get_logger().info("⚠ ACCEPTABLE. RMS error < 1.0 pixel")
        else:
            self.get_logger().warn("⚠ WARNING! High RMS error - consider recalibrating")
        
        # Log per-sample errors
        self.get_logger().info("\nPer-sample reprojection errors:")
        for i, err in enumerate(sample_errors):
            self.get_logger().info(f"  Sample {i+1}: {err:.4f} pixels")

    def compute_chessboard_in_base(self, R_cam2gripper, t_cam2gripper):
        """
        Compute the chessboard pose in base frame using all captured data.
        This averages the transform across all observations for robustness.
        
        Transform chain: Base -> Gripper -> Camera -> Chessboard
        """
        if len(self.R_gripper2base) == 0:
            self.get_logger().error("No calibration data available")
            return None, None
        
        self.get_logger().info("\nComputing chessboard pose in base frame...")
        
        # For each observation, compute: T_base_chessboard = T_base_gripper * T_gripper_cam * T_cam_chessboard
        positions_base = []
        orientations_base = []
        
        for i in range(len(self.R_gripper2base)):
            # T_base_gripper (from robot)
            T_base_gripper = np.eye(4)
            T_base_gripper[:3, :3] = self.R_gripper2base[i]
            T_base_gripper[:3, 3:4] = self.t_gripper2base[i]
            
            # T_gripper_cam (from hand-eye calibration)
            T_gripper_cam = np.eye(4)
            T_gripper_cam[:3, :3] = R_cam2gripper
            T_gripper_cam[:3, 3:4] = t_cam2gripper.reshape(3, 1)
            
            # T_cam_chessboard (from PnP)
            T_cam_target = np.eye(4)
            T_cam_target[:3, :3] = self.R_target2cam[i]
            T_cam_target[:3, 3:4] = self.t_target2cam[i]
            
            # Chain the transforms
            T_base_target = T_base_gripper @ T_gripper_cam @ T_cam_target
            
            positions_base.append(T_base_target[:3, 3])
            orientations_base.append(R.from_matrix(T_base_target[:3, :3]))
        
        # Average the positions (simple mean)
        mean_position = np.mean(positions_base, axis=0)
        self.get_logger().info(f"Mean Position (m): [{mean_position[0]:.6f}, {mean_position[1]:.6f}, {mean_position[2]:.6f}]")
        
        # Align all orientations to a reference before averaging (handle 180° flips)
        reference_rotation = orientations_base[0]
        aligned_orientations = []
        
        for i, orientation in enumerate(orientations_base):
            # Check if this orientation is flipped relative to reference
            quat = orientation.as_quat()
            ref_quat = reference_rotation.as_quat()
            
            # Quaternions q and -q represent the same rotation
            # If dot product is negative, they're on opposite hemispheres
            dot_product = np.dot(quat, ref_quat)
            if dot_product < 0:
                quat = -quat
                self.get_logger().info(f"Sample {i+1}: Aligned quaternion (was flipped)")
            
            aligned_orientations.append(R.from_quat(quat))
        
        # Average the aligned orientations (using quaternion averaging)
        quats = np.array([r.as_quat() for r in aligned_orientations])
        mean_quat = self.average_quaternions(quats)
        mean_rotation = R.from_quat(mean_quat).as_matrix()
        
        # Enforce desired orientation in base frame: +X, -Y, -Z
        # Extract the chessboard's axes in base frame
        x_axis = mean_rotation[:, 0]  # First column = chessboard X in base
        y_axis = mean_rotation[:, 1]  # Second column = chessboard Y in base
        z_axis = mean_rotation[:, 2]  # Third column = chessboard Z in base
        
        self.get_logger().info(f"Chessboard X-axis in base: [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
        self.get_logger().info(f"Chessboard Y-axis in base: [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")
        self.get_logger().info(f"Chessboard Z-axis in base: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
        
        # Check if we need to flip the chessboard 180° to match desired orientation
        # We want: +X (x_axis[0] > 0), -Y (y_axis[1] < 0), -Z (z_axis[2] < 0)
        needs_flip = False
        
        # Primary check: X should point in +X direction
        if x_axis[0] < 0:
            self.get_logger().info("Chessboard X pointing in -X, flipping 180° around Z-axis")
            needs_flip = True
        
        if needs_flip:
            # Flip 180° around the chessboard's Z-axis (which is currently pointing -Z)
            # This inverts X and Y axes while keeping Z the same
            flip_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            mean_rotation = mean_rotation @ flip_matrix
            
            # Log new axes
            x_axis = mean_rotation[:, 0]
            y_axis = mean_rotation[:, 1]
            z_axis = mean_rotation[:, 2]
            self.get_logger().info(f"After flip - X-axis: [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
            self.get_logger().info(f"After flip - Y-axis: [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")
            self.get_logger().info(f"After flip - Z-axis: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
        
        # Compute standard deviation to assess consistency
        position_std = np.std(positions_base, axis=0)
        
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info("CHESSBOARD POSE IN BASE FRAME")
        self.get_logger().info("="*50)
        self.get_logger().info(f"Position (m): [{mean_position[0]:.6f}, {mean_position[1]:.6f}, {mean_position[2]:.6f}]")
        self.get_logger().info(f"Position Std Dev (mm): [{position_std[0]*1000:.3f}, {position_std[1]*1000:.3f}, {position_std[2]*1000:.3f}]")
        
        # Log individual observations to see spread
        self.get_logger().info("\nIndividual observations (position only):")
        for i, pos in enumerate(positions_base):
            deviation = np.linalg.norm(pos - mean_position) * 1000  # mm
            self.get_logger().info(f"  Sample {i+1}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}] - deviation: {deviation:.3f}mm")
        
        max_deviation = np.max([np.linalg.norm(p - mean_position) for p in positions_base]) * 1000
        self.get_logger().info(f"\nMaximum deviation from mean: {max_deviation:.3f}mm")
        
        if max_deviation < 1.0:
            self.get_logger().info("✓ Excellent! All observations within 1mm")
        elif max_deviation < 2.0:
            self.get_logger().info("✓ Good! All observations within 2mm")
        else:
            self.get_logger().warn(f"⚠ High deviation! Consider re-calibrating")
        
        return mean_rotation, mean_position

    def average_quaternions(self, quats):
        """
        Average quaternions using the method from:
        Markley, F. Landis, et al. "Averaging quaternions." 
        Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
        """
        # Build the matrix M = sum(q * q^T) for all quaternions
        M = np.zeros((4, 4))
        for q in quats:
            M += np.outer(q, q)
        M /= len(quats)
        
        # The eigenvector corresponding to the largest eigenvalue is the average
        eigenvalues, eigenvectors = np.linalg.eig(M)
        max_idx = np.argmax(eigenvalues)
        avg_quat = eigenvectors[:, max_idx].real
        
        # Ensure unit quaternion
        avg_quat /= np.linalg.norm(avg_quat)
        
        return avg_quat

    def publish_chessboard_pose(self):
        """Continuously publish the chessboard pose if available"""
        if self.chessboard_pose_msg is not None:
            # Update timestamp
            self.chessboard_pose_msg.header.stamp = self.get_clock().now().to_msg()
            self.chessboard_pose_pub.publish(self.chessboard_pose_msg)

    def republish_camera_transform(self):
        """Republish camera transform to ensure it's always available"""
        if self.camera_transform_msg is not None:
            # Update timestamp for static transform
            self.camera_transform_msg.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(self.camera_transform_msg)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_model.fromCameraInfo(msg)
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        # Just update the buffer, processing happens in control_loop
        self.latest_image_msg = msg
        if self.state == "WAITING_NEW_IMAGE":
            self.state = "CAPTURING"
        
        # Continuously display live feed
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try to detect chessboard for visual feedback
            if self.camera_matrix is not None:
                found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
                
                if found:
                    # Draw the corners
                    cv2.drawChessboardCorners(frame, self.pattern_size, corners, found)
                    cv2.putText(frame, "Chessboard Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Chessboard Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display state information
            cv2.putText(frame, f"State: {self.state}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Step: {self.current_step}/{len(self.motion_sequence)}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Hand-Eye Calibration - Live Feed", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error displaying image: {e}")

    def move_robot_relative(self, twist):
        """
        Moves the robot relative to the BASE frame using a Cartesian path.
        Example: move_robot_relative(dz=0.3048) moves 1 foot up/forward in Z.
        """
        self.is_executing = True
        self.get_logger().info(f"Moving robot relative: {twist}")
        # 1. Get Current Pose (We need this to calculate the start and end)
        try:
            # Get transform from base_link to tool0 (latest available)
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
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
        current_euler = R.from_quat([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]).as_euler('xyz')
        
        # Target Pose (Relative Offset)
        target_pose = copy.deepcopy(current_pose)
        target_pose.position.x += twist[0]
        target_pose.position.y += twist[1]
        target_pose.position.z += twist[2]
        target_quat = R.from_euler('xyz', [
            current_euler[0] + twist[3], current_euler[1] + twist[4], current_euler[2] + twist[5]
        ]).as_quat()  # Convert to quaternion
        target_pose.orientation.x = target_quat[0]
        target_pose.orientation.y = target_quat[1]
        target_pose.orientation.z = target_quat[2]
        target_pose.orientation.w = target_quat[3]

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
        self.get_logger().info('Move complete! Waiting for robot to settle...')
        # Record when motion ended
        self.motion_end_time = self.get_clock().now()
        # Transition to waiting state, control_loop will handle timing
        self.state = "WAITING_NEW_IMAGE"
        self.settle_start_time = None  # Reset for fresh timing

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


