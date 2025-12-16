import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from image_geometry import PinholeCameraModel
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
from scipy.spatial.transform import Rotation as R

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')

        # --- CONFIGURATION ---
        self.pattern_size = (9, 6)            # Inner corners (cols, rows)
        self.square_size = 0.02325            # Size in meters
        self.target_frame = 'base_link'       # Robot Base Frame
        self.fixed_pose = None

        # --- STATE ---
        self.running = False

        # --- SETUP ---
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Prepare 3D points of the Chessboard (Z=0 in board frame)
        self.obj_points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.obj_points *= self.square_size

        # Subscribers
        self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # Publisher for the result
        self.result_pub = self.create_publisher(PoseStamped, '/calibration/chessboard_pose', 10)
        
        # Service to trigger calibration
        self.srv = self.create_service(Trigger, '/calibrate_height', self.trigger_callback)
        
        self.create_timer(0.1, self.publish_fixed_pose)

        self.get_logger().info("Calibration Node started. Waiting for service call...")

    def publish_fixed_pose(self):
        return
        if self.fixed_pose is not None:
            self.result_pub.publish(self.fixed_pose)

    def trigger_callback(self, request, response):
        self.get_logger().info("Calibration triggered!")
        self.running = True
        self.fixed_pose = None
        
        response.success = True
        response.message = "Calibration started. Watch /calibration/chessboard_pose"
        return response

    def info_callback(self, msg):
        """Get camera intrinsics once."""
        if self.camera_matrix is None:
            self.camera_model.fromCameraInfo(msg)
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info(f"Intrinsics received. Frame ID: {msg.header.frame_id}")

    def image_callback(self, msg):
        """Main loop: Detect board -> SolvePnP -> Collect Z samples."""
        if not self.running:
            # Just show feed if not running
            # if self.camera_matrix is not None:
            #     frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            #     cv2.imshow("Calibration View", frame)
            #     cv2.waitKey(1)
            cv2.destroyAllWindows()
            return

        # if self.fixed_pose is not None:
        #     self.fixed_pose.header.stamp = msg.header.stamp
        #     self.result_pub.publish(self.fixed_pose)
            
        #     if self.camera_matrix is not None:
        #         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        #         cv2.imshow("Chessboard View", frame)
        #         cv2.waitKey(1)
        #     return

        # if not self.handeyecalibration.wait_for_service(timeout_sec=0.1):
        #     self.get_logger().warning('Detection service not available', throttle_duration_sec=5.0)
        #     return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow("Chessboard View", frame)
        cv2.waitKey(1)

        if self.camera_matrix is None:
            return

        # self.get_logger().info("Processing frame for chessboard detection...")

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
                # 4. Transform Pose from Camera Frame to Robot Base Frame
                self.rvec = rvec
                self.tvec = tvec

                self.get_logger().info(f"Chessboard detected. rvec: {rvec.flatten()} tvec: {tvec.flatten()}")

                # Create PoseStamped in Camera Frame
                pose_cam = PoseStamped()
                pose_cam.header.frame_id = "camera_link"
                pose_cam.header.stamp = msg.header.stamp

                # Position
                pose_cam.pose.position.x = float(tvec[0])
                pose_cam.pose.position.y = float(tvec[1])
                pose_cam.pose.position.z = float(tvec[2])

                # Orientation (Rotation Vector -> Quaternion)
                rot = R.from_rotvec(rvec.flatten())
                # rot = rot * R.from_euler('z', 180, degrees=True)
                quat = rot.as_quat() # [x, y, z, w]
                # self.get_logger().info(f"rodges rot: {cv2.Rodrigues(rvec)[0]}")
                # self.get_logger().info(f"rot: {rot.as_matrix()}")
                pose_cam.pose.orientation.x = quat[0]
                pose_cam.pose.orientation.y = quat[1]
                pose_cam.pose.orientation.z = quat[2]
                pose_cam.pose.orientation.w = quat[3]

                try:
                    # Transform Pose using buffer interface
                    pose_base = self.tf_buffer.transform(pose_cam, self.target_frame, timeout=Duration(seconds=0.1))
                    
                    # Publish
                    self.fixed_pose = pose_base
                    self.get_logger().info(f"Pose Locked in {self.target_frame}: {pose_base.pose.position.x}, {pose_base.pose.position.y}, {pose_base.pose.position.z}")
                    
                    self.running = False  # Stop after one successful detection

                except Exception as e:
                    self.get_logger().warn(f"Transform Error: {e}")

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
        

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
