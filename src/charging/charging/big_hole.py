import cv2
import numpy as np
import sys
import os

# === Configuration ===
DISPLAY_FRAMES = True  # Set to False to disable visualization

# Hough Circle parameters
CANNY_THRESHOLD1 = 10
CANNY_THRESHOLD2 = 150
HOUGH_PARAM1 = 100  # Higher threshold for Canny edge detection in Hough
HOUGH_PARAM2 = 50   # Accumulator threshold (lower = more circles detected)
MIN_RADIUS = 20     # Minimum circle radius in pixels
MAX_RADIUS = 200    # Maximum circle radius in pixels
MIN_DISTANCE = 50   # Minimum distance between circle centers

code_path = os.path.abspath(__file__)
code_dir = os.path.dirname(code_path)


def find_median_position(camera_id=6, num_frames=20):
    """
    Run video capture for a specified number of frames and return the median position
    of detected circles.
    
    Args:
        camera_id: Camera device ID
        num_frames: Number of frames to process (default 20)
    
    Returns:
        tuple: (median_x, median_y) coordinates, or (None, None) if no circles found
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None, None
    
    all_x_positions = []
    all_y_positions = []
    
    for frame_idx in range(num_frames):
        ret, image = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        if image is None:
            print("Error: Could not load image.")
            continue

        # if frame_idx == 0:
        #     cv2.imwrite("debug.png", image)
    
        
        # Convert to grayscale
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        blur = cv2.medianBlur(s_channel, 9)

        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        purple_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Apply Otsu's thresholding
        # threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Detect edges using Canny
        edges = cv2.Canny(purple_mask, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,  # Inverse ratio of accumulator resolution
            minDist=MIN_DISTANCE,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=MIN_RADIUS,
            maxRadius=MAX_RADIUS
        )
        
        # Create visualization
        result_image = image.copy()
        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Filter circles by circularity (check how close to perfect circle)
            valid_circles = []
            for (x, y, r) in circles:
                # Create a mask for this circle
                mask = np.zeros(purple_mask.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, 2)
                
                # Get edge pixels that overlap with circle perimeter
                circle_edges = cv2.bitwise_and(edges, edges, mask=mask)
                
                # Calculate circularity score (edge pixels / expected perimeter)
                edge_pixels = np.count_nonzero(circle_edges)
                expected_perimeter = 2 * np.pi * r
                circularity = edge_pixels / expected_perimeter if expected_perimeter > 0 else 0
                
                # Only keep circles with good circularity (close to perfect)
                if circularity > 0.3:  # At least 30% of expected perimeter has edges
                    valid_circles.append((x, y, r, circularity))
            
            # Sort by circularity score (best first)
            valid_circles.sort(key=lambda c: c[3], reverse=True)
            
            if len(valid_circles) > 0:
                # Store the best circle's center
                best_circle = valid_circles[0]
                x, y, r, circularity = best_circle
                all_x_positions.append(x)
                all_y_positions.append(y)
                
                # Draw all valid circles (green for best, yellow for others)
                for idx, (cx, cy, cr, circ) in enumerate(valid_circles):
                    if idx == 0:
                        color = (0, 255, 0)  # Green for best match
                        thickness = 3
                    else:
                        color = (0, 255, 255)  # Yellow for others
                        thickness = 2
                    
                    # Draw the circle outline
                    cv2.circle(result_image, (cx, cy), cr, color, thickness)
                    # Draw the center
                    cv2.circle(result_image, (cx, cy), 5, color, -1)
                    
                    # Add label
                    label = f"#{idx+1}: c={circ:.2f}"
                    cv2.putText(result_image, label, (cx - 50, cy - cr - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw on edge display too
                    cv2.circle(edge_display, (cx, cy), cr, color, 2)
                    cv2.circle(edge_display, (cx, cy), 5, color, -1)
        
        # Display frames
        if DISPLAY_FRAMES:
            cv2.imshow("1. Original Image", image)
            cv2.imshow("2. Detected Edges (Canny)", edge_display)
            cv2.imshow("3. Detected Circles", result_image)
            cv2.imshow("4. thres", purple_mask)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate and return median position
    if len(all_x_positions) < 6 or len(all_y_positions) < 6:
        print(f"Found circles in {len(all_x_positions)} out of {num_frames} frames.")
        return None, None
    
    # Get frame dimensions from last processed image
    frame_width = purple_mask.shape[1]
    frame_height = purple_mask.shape[0]
    
    # Calculate median and normalize (similar to keyhole.py)
    median_x = (float(np.median(all_x_positions)) - frame_width / 2) / 15
    median_y = (float(np.median(all_y_positions)) - frame_height / 2) / 15
    
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    print(f"Processed {num_frames} frames, found circles in {len(all_x_positions)} frames")
    print(f"Median position: ({median_x}, {median_y})")
    
    return median_x, median_y


# Example usage:
if __name__ == "__main__":
    x, y = find_median_position(camera_id=6, num_frames=1000)
    if x is not None and y is not None:
        print(f"Final median coordinates: x={x}, y={y}")
