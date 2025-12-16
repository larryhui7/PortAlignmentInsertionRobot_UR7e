import cv2
import numpy as np

# --- Input Mode ---
USE_IMAGE = False       # Set to True to use an image file, False to use webcam
IMAGE_PATH = "far.jpeg"  # Path to the image file (only used if USE_IMAGE is True)

# --- You can tune these parameters ---
MIN_AREA = 100         # The minimum size of the port (in pixels) to be considered
MIN_OVAL_RATIO = 0.2    # The "oval-ness". 1.0 is a perfect circle, 0.1 is a thin line.
MAX_OVAL_RATIO = 0.5    # We don't want perfect circles, so we set a max.
CANNY_LOW = 20          # Low threshold for edge detection (lowered for better low-light detection)
CANNY_HIGH = 60         # High threshold for edge detection (lowered for better low-light detection)
MAX_ELLIPSE_FIT_ERROR_RATIO = 0.3  # Maximum average distance as ratio of ellipse size (e.g., 0.15 = 15% of perimeter)

# --- Distance estimation parameters ---
# You MUST calibrate these for your specific setup!
KNOWN_WIDTH = 2.56     # The real-world width of the oval port in cm (CHANGE THIS!)
KNOWN_HEIGHT = .865     # The real-world height of the oval port in cm (CHANGE THIS!)
FOCAL_LENGTH = 1884.11      # Focal length in pixels (calibrate by measuring at known distance)
# To calibrate FOCAL_LENGTH:
# 1. Measure the real width of your oval port (KNOWN_WIDTH)
# 2. Place it at a known distance (e.g., 50 cm)
# 3. Run the code and note the major_axis in pixels
# 4. FOCAL_LENGTH = (major_axis * distance) / KNOWN_WIDTH
# --- End of tuneable parameters ---

# Initialize video capture or load image
if USE_IMAGE:
    # Load image from file
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        exit()
    print(f"Loaded image: {IMAGE_PATH}")
    use_video = False
else:
    # Start the webcam
    # vid capture 6: suvans cam
    # vid capture 4: realsense rgb cam 
    cap = cv2.VideoCapture(6)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    use_video = True

while True:
    # Read a frame from the webcam or use the loaded image
    if use_video:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break
    # If using image, frame is already loaded

    # 1. Pre-processing: Grayscale and Blur
    # Convert to grayscale for shape detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a stronger blur to smooth out small edges and preserve large continuous shapes
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # 2. Robust Thresholding
    # We use THRESH_BINARY_INV to make the dark holes (shadows) 
    # white, and the lighter block black.
    # THRESH_OTSU automatically finds the best threshold value,
    # which makes this robust to lighting changes.

    # equalized_image = cv2.equalizeHist(blur)
    # blur = equalized_image

    # (T, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 2)

    # 3. Morphological Opening (Cleanup)
    # This removes small white specks (noise) that aren't holes.
    # You can increase the kernel size to remove larger noise.
    morph_kernel_size = 5
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

    # 4. Edge Detection
    # Use Canny edge detection to find the outlines of shapes
    edges = cv2.Canny(morphed, CANNY_LOW, CANNY_HIGH)

    # 3. Find Contours (Shapes)
    # Find all the continuous outlines in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black background to draw all contours
    contour_frame = np.zeros_like(frame)
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 255), 2)  # Draw all contours in yellow

    # 4. Loop over contours and filter for ovals
    for contour in contours:
        # Filter 1: Check if the contour is large enough
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        # Filter 2: Check if the contour has enough points to fit an ellipse
        # fitEllipse requires at least 5 points
        if len(contour) < 5:
            continue

        try:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (minor_axis, major_axis), angle = ellipse

            # Filter 3: Check for valid ellipse data
            if minor_axis <= 0 or major_axis <= 0:
                continue

            # Filter 4: Check the "oval-ness" (aspect ratio)
            ratio = minor_axis / major_axis

            # Filter 5: Check goodness of fit between ellipse and contour
            # Sample points along the ellipse and check distance to original contour
            num_samples = 50  # Number of points to sample along the ellipse
            total_distance = 0
            
            for i in range(num_samples):
                # Calculate angle for this sample point
                theta = 2 * np.pi * i / num_samples
                
                # Get point on the ellipse
                # Ellipse equation: x = cx + (major/2)*cos(theta)*cos(angle) - (minor/2)*sin(theta)*sin(angle)
                #                   y = cy + (major/2)*cos(theta)*sin(angle) + (minor/2)*sin(theta)*cos(angle)
                angle_rad = np.deg2rad(angle)
                x_ellipse = cx + (major_axis/2) * np.cos(theta) * np.cos(angle_rad) - (minor_axis/2) * np.sin(theta) * np.sin(angle_rad)
                y_ellipse = cy + (major_axis/2) * np.cos(theta) * np.sin(angle_rad) + (minor_axis/2) * np.sin(theta) * np.cos(angle_rad)
                
                # Calculate distance from this ellipse point to the original contour
                # pointPolygonTest returns negative distance if outside, positive if inside, 0 if on edge
                distance = abs(cv2.pointPolygonTest(contour, (x_ellipse, y_ellipse), True))
                total_distance += distance
            
            # Calculate average distance
            avg_distance = total_distance / num_samples
            
            # Scale the error threshold based on the size of the ellipse
            # Use the average of major and minor axes as a measure of size
            ellipse_size = (major_axis + minor_axis) / 2
            max_allowed_error = ellipse_size * MAX_ELLIPSE_FIT_ERROR_RATIO
            
            # Skip this contour if the fit is too poor
            if avg_distance > max_allowed_error:
                continue
            
            if MIN_OVAL_RATIO < ratio < MAX_OVAL_RATIO:
                # --- Success! We found a likely oval port ---
                
                # A. Get the simple (axis-aligned) bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # B. Estimate distance using the major axis (larger dimension)
                # Distance = (Known_Width * Focal_Length) / Perceived_Width_in_Pixels
                perceived_width_pixels = major_axis
                distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width_pixels
                distance_inches = distance_cm / 2.54
                
                # C. Draw the green bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # D. Draw labels with distance information
                cv2.putText(frame, "Oval Port", (x, y - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Error: {avg_distance / max_allowed_error:.2f} Distance: {distance_cm:.1f} cm ({distance_inches:.1f} in)", 
                            (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # E. (Optional) Draw the fitted ellipse in blue to see the fit
                cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
                
                # F. (Optional) Draw center point
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        except cv2.error:
            # fitEllipse can sometimes fail, so we catch the error
            pass

    # Display the resulting frame
    cv2.imshow('Oval Port Detector', frame)
    
    # Display the blurred image (helpful for tuning blur parameters)
    cv2.imshow('Blurred', blur)
    
    # Display the threshold image (helpful for seeing binary separation)
    cv2.imshow('Threshold', thresh)
    
    # Display the morphed image (helpful for seeing noise cleanup)
    cv2.imshow('Morphed', morphed)
    
    # Display all detected contours (helpful for understanding what's being detected)
    cv2.imshow('Contours', contour_frame)
    
    # Display the edge detection (helpful for tuning CANNY thresholds)
    cv2.imshow('Edges', edges)

    # Press 'q' to quit the program
    # For images, wait indefinitely; for video, check every 1ms
    wait_time = 0 if not use_video else 1
    if cv2.waitKey(wait_time) == ord('q'):
        break

# Clean up
if use_video:
    cap.release()
cv2.destroyAllWindows()