import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

class CameraBaseCalibrator(Node):
    def __init__(self):
        super().__init__('camera_base_calibrator')

        # --- CONFIGURATION ---
        self.aruco_id = 10              # The ID of your ArUco tag
        self.aruco_size = 0.15          # Size of ArUco tag in meters (e.g., 5cm)
        self.pattern_size = (9, 6)      # Chessboard inner corners
        self.square_size = 0.02325      # Chessboard square size in meters
        
        # Frames
        self.marker_frame = f'ar_marker_{self.aruco_id}'
        self.camera_frame = 'camera_link'
        self.board_frame = 'ar_chessboard_frame'
        self.base_frame = 'base_link'

        # --- CV SETUP ---
        self.bridge = CvBridge()
        
        # ArUco Dictionary - CHANGE THIS IF NEEDED (e.g. DICT_4X4_50, DICT_6X6_250)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera Intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None

        # Prepare 3D points for Chessboard
        self.board_obj_points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.board_obj_points[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.board_obj_points *= self.square_size

        # --- ROS SETUP ---
        self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Storage for transforms
        self.T_cam_aruco = None # Transform matrix 4x4
        self.T_cam_board = None # Transform matrix 4x4

        self.get_logger().info("Calibrator Node Started. Waiting for camera info...")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera Intrinsics Received.")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 1. Detect ArUco Marker ---
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and self.aruco_id in ids:
            index = np.where(ids == self.aruco_id)[0][0]
            # Estimate Pose of ArUco
            rvec_ar, tvec_ar, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[index], self.aruco_size, self.camera_matrix, self.dist_coeffs)
            
            # Draw for debug
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec_ar, tvec_ar, 0.03)

            # Store as 4x4 Matrix
            self.T_cam_aruco = self.get_transform_matrix(rvec_ar, tvec_ar)
            
            # PUBLISH TF: ar_marker -> camera
            # We detected Cam->ArUco. TF expects Parent->Child.
            # Ideally, the Marker is the fixed anchor. So we publish Marker->Camera.
            # T_aruco_cam = inverse(T_cam_aruco)
            T_aruco_cam = np.linalg.inv(self.T_cam_aruco)
            self.broadcast_transform(T_aruco_cam, self.marker_frame, self.camera_frame)

        # --- 2. Detect Chessboard ---
        found, board_corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if found:
            # Refine
            cv2.cornerSubPix(gray, board_corners, (11,11), (-1,-1), 
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # Estimate Pose
            success, rvec_cb, tvec_cb = cv2.solvePnP(
                self.board_obj_points, board_corners, self.camera_matrix, self.dist_coeffs)
            
            if success:
                cv2.drawChessboardCorners(frame, self.pattern_size, board_corners, found)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec_cb, tvec_cb, 0.05)
                
                # Store as 4x4 Matrix
                self.T_cam_board = self.get_transform_matrix(rvec_cb, tvec_cb)

                # PUBLISH TF: camera -> chessboard
                self.broadcast_transform(self.T_cam_board, self.camera_frame, self.board_frame)

        # --- 3. Compute Final Transform (Base -> Board) ---
        # If we see both, we can compute the chain locally for verification
        if self.T_cam_aruco is not None and self.T_cam_board is not None:
            # self.get_logger().info("Both ArUco and Chessboard detected. Computing Base->Board transform...")
            self.compute_base_to_board()

        cv2.imshow("Board Pose", frame)
        cv2.waitKey(1)

    def get_transform_matrix(self, rvec, tvec):
        """Convert rvec/tvec to 4x4 Homogeneous Matrix"""
        mat = np.eye(4)
        R, _ = cv2.Rodrigues(rvec)
        mat[:3, :3] = R
        mat[:3, 3] = tvec.flatten()
        return mat

    def broadcast_transform(self, mat, parent_frame, child_frame):
        """Helper to publish TFs"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Extract translation
        t.transform.translation.x = mat[0, 3]
        t.transform.translation.y = mat[1, 3]
        t.transform.translation.z = mat[2, 3]

        # Extract rotation (convert Matrix to Quaternion)
        quat = Rotation.from_matrix(mat[:3, :3]).as_quat() # x, y, z, w
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def compute_base_to_board(self):
        """
        Look up T_aruco_base from TF (published by your other script)
        and calculate T_base_board = inv(T_aruco_base) * inv(T_cam_aruco) * T_cam_board
        """
        try:
            # 1. Get T_aruco_base (Provided by your 'ConstantTransformPublisher')
            # Note: Your script publishes Header: ar_marker, Child: base_link
            # That is T_aruco_base.
            tf_msg = self.tf_buffer.lookup_transform(
                self.marker_frame, self.base_frame, rclpy.time.Time())
            
            # Convert Msg to Matrix
            t = tf_msg.transform.translation
            r = tf_msg.transform.rotation
            q = [r.x, r.y, r.z, r.w]
            
            T_aruco_base = np.eye(4)
            T_aruco_base[:3, :3] = Rotation.from_quat(q).as_matrix()
            T_aruco_base[:3, 3] = [t.x, t.y, t.z]

            # 2. Compute T_base_board
            # We want: Base -> Board
            # Chain: Base -> Aruco -> Camera -> Board
            
            # T_base_aruco = inv(T_aruco_base)
            T_base_aruco = np.linalg.inv(T_aruco_base)

            # T_aruco_cam = inv(T_cam_aruco)
            T_aruco_cam = np.linalg.inv(self.T_cam_aruco)

            # Multiply: Base -> Aruco -> Cam -> Board
            T_base_board = T_base_aruco @ T_aruco_cam @ self.T_cam_board

            # Print result
            x, y, z = T_base_board[:3, 3]
            self.get_logger().info(str(T_base_board))
            self.get_logger().info(f"CALCULATED BASE->BOARD: X={x:.3f}, Y={y:.3f}, Z={z:.3f}", throttle_duration_sec=2.0)

        except Exception as e:
            # This is normal on startup while waiting for transforms
            self.get_logger().info(f"Waiting for transforms: {e}", throttle_duration_sec=2.0)
            pass

def main(args=None):
    rclpy.init(args=args)
    node = CameraBaseCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


