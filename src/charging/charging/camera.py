# import cv2
# import sys

# def main():
#     # Default camera ID is 0, can be passed as argument
#     camera_id = 0
#     if len(sys.argv) > 1:
#         try:
#             camera_id = int(sys.argv[1])
#         except ValueError:
#             print("Usage: python3 camera.py [camera_id]")
#             return

#     cap = cv2.VideoCapture(camera_id)
    
#     # Set resolution to 1080p (1920x1080)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
#     # Set FPS to 30
#     cap.set(cv2.CAP_PROP_FPS, 30)

#     if not cap.isOpened():
#         print(f"Cannot open camera {camera_id}")
#         return

#     # Verify settings
#     actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
#     print(f"Camera {camera_id} opened.")
#     print(f"Resolution: {actual_w}x{actual_h}")
#     print(f"FPS: {actual_fps}")
#     print("Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
            
#         cv2.imshow('Camera Stream', frame)
        
#         if cv2.waitKey(1) == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()