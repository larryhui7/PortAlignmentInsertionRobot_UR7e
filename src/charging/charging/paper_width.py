#!/usr/bin/env python3
"""
Quick script to detect a piece of paper in frame and print its width in pixels.
"""

import cv2
import numpy as np

# --- Configuration ---
CAMERA_INDEX = 6        # Webcam index (change as needed)

def main():
    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
        
        cv2.rectangle(frame, (200, 200), (400, 300), (0, 255, 0), 3)
        # 484-65, 393-65
        # 419 x 328
        cv2.imshow('Paper Detection', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
