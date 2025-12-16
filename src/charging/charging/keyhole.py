import cv2
import numpy as np
import sys
import os

# === Configuration ===
# Match threshold: 0.0 to 1.0 (higher = stricter)
MATCH_THRESHOLD = 0.8

# Template scale range as percentage of search area
MIN_TEMPLATE_SIZE = 0.0005 * 0.75  # 0.1% of search area
MAX_TEMPLATE_SIZE = 0.015 * 0.75  # 1% of search area
NUM_SCALES = 20

DISPLAY_FRAMES = False  # Set to False to disable visualization

# === 1. Load Images ===
# image = cv2.imread("far.jpeg")

code_path = os.path.abspath(__file__)
code_dir = os.path.dirname(code_path)

# Simple NMS: remove overlapping boxes
def non_max_suppression(matches, overlap_thresh=0.3):
    if len(matches) == 0:
        return []
    
    boxes = []
    for m in matches:
        x, y = m['location']
        w, h = m['size']
        boxes.append([x, y, x + w, y + h, m['score']])
    
    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / areas[order[1:]]
        order = order[np.concatenate([[0], np.where(overlap <= overlap_thresh)[0] + 1])[1:]]
    
    return [matches[i] for i in keep]

def find_median_position(image, template_path=os.path.join(code_dir, "usbc_template.png")):
    """
    Process a list of images and return the median position.
    
    Args:
        image: A numpy array representing the image
        template_path: Path to the template image
    
    Returns:
        tuple: (median_x, median_y) coordinates, or (None, None) if no matches found
    """
    all_x_positions = []
    all_y_positions = []
    
    template = cv2.imread(template_path)
    if template is None:
        print("Error: Could not load template.")
        return None, None
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    if image is None:
        print("Error: Could not load image.")
        return None, None

    # Convert to grayscale (preserves light/dark contrast)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Search the entire image
    search_gray = image_gray

    # Store all matches
    all_matches = []

    # === 3. Single-Scale Template Matching with Correlation Map ===
    tH, tW = template_gray.shape[:2]
    search_area = search_gray.shape[0] * search_gray.shape[1]
    template_area = tH * tW

    # Calculate scale range
    min_scale = np.sqrt((MIN_TEMPLATE_SIZE * search_area) / template_area)
    max_scale = np.sqrt((MAX_TEMPLATE_SIZE * search_area) / template_area)

    # Use middle scale for visualization
    display_scale = (min_scale + max_scale) / 2

    # print(f"Template area: {template_area} pixels")
    # print(f"Displaying correlation map at scale {display_scale:.3f}")

    # Resize template to display scale
    resized_template = cv2.resize(template_gray, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    rH, rW = resized_template.shape[:2]

    # Perform template matching at this scale
    correlation_map = cv2.matchTemplate(search_gray, resized_template, cv2.TM_CCOEFF_NORMED)

    # print(f"Correlation map range: {correlation_map.min():.3f} to {correlation_map.max():.3f}")

    # Create heatmap visualization
    # Normalize correlation map to 0-255 for display
    correlation_normalized = ((correlation_map - correlation_map.min()) / 
                            (correlation_map.max() - correlation_map.min()) * 255).astype(np.uint8)

    # Apply colormap (hot = red/yellow for high correlation, blue/black for low)
    heatmap = cv2.applyColorMap(correlation_normalized, cv2.COLORMAP_JET)

    # Resize heatmap to match search region size (correlation map is smaller)
    heatmap_resized = cv2.resize(heatmap, (search_gray.shape[1], search_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Create overlay on search region
    search_region_color = cv2.cvtColor(search_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(search_region_color, 0.4, heatmap_resized, 0.6, 0)

    # Mark threshold line on heatmap
    threshold_locations = np.where(correlation_map >= MATCH_THRESHOLD)
    for pt in zip(*threshold_locations[::-1]):
        cv2.circle(overlay, pt, 2, (255, 255, 255), -1)

    # print(f"Found {len(threshold_locations[0])} points above threshold {MATCH_THRESHOLD}")

    # Now do multi-scale for finding all matches
    # print(f"\nSearching at scales {min_scale:.3f} to {max_scale:.3f}...")

    for scale in np.linspace(min_scale, max_scale, NUM_SCALES)[::-1]:
        # Resize template
        resized = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rH, rW = resized.shape[:2]
        
        # Skip if too large
        if rH > search_gray.shape[0] or rW > search_gray.shape[1]:
            continue
        
        # Template matching (TM_CCOEFF_NORMED is good for contrast matching)
        result = cv2.matchTemplate(search_gray, resized, cv2.TM_CCOEFF_NORMED)
        
        # Find all matches above threshold
        locations = np.where(result >= MATCH_THRESHOLD)
        for pt in zip(*locations[::-1]):
            all_matches.append({
                'score': result[pt[1], pt[0]],
                'location': pt,
                'scale': scale,
                'size': (rW, rH)
            })

    # === 4. Remove Overlapping Matches (Non-Maximum Suppression) ===
    # print(f"\nFound {len(all_matches)} raw matches above threshold {MATCH_THRESHOLD}")

    # === 5. Draw Results ===
    result_image = image.copy()

    # Draw min and max template sizes for visualization
    min_w = int(tW * min_scale)
    min_h = int(tH * min_scale)
    max_w = int(tW * max_scale)
    max_h = int(tH * max_scale)
    
    # Draw in top-left with some padding
    start_x, start_y = 50, 50
    
    # Max size (Yellow)
    cv2.rectangle(result_image, (start_x, start_y), (start_x + max_w, start_y + max_h), (0, 255, 255), 2)
    cv2.putText(result_image, "Max Size", (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Min size (Red)
    cv2.rectangle(result_image, (start_x, start_y), (start_x + min_w, start_y + min_h), (0, 0, 255), 2)
    cv2.putText(result_image, "Min", (start_x + min_w + 5, start_y + min_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(all_matches) == 0:
        # print("No matches found in this frame.")
        if DISPLAY_FRAMES:
            cv2.imshow("2. Correlation Heatmap (white dots = above threshold)", overlay)
            cv2.imshow("3. Matches Found", result_image)
        cv2.waitKey(1)
        return None, None
    
    # Sort by score
    all_matches.sort(key=lambda x: x['score'], reverse=True)

    filtered_matches = non_max_suppression(all_matches, overlap_thresh=0.3)
    # print(f"After removing overlaps: {len(filtered_matches)} unique matches")
    
    # Store the best match position (center of the bounding box)
    if len(filtered_matches) > 0:
        best_match = filtered_matches[0]
        x = best_match['location'][0]
        y = best_match['location'][1]
        w, h = best_match['size']
        center_x = x + w // 2
        center_y = y + h // 2
        # all_x_positions.append(center_x)
        # all_y_positions.append(center_y)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for idx, match in enumerate(filtered_matches):
        x = match['location'][0]
        y = match['location'][1]
        w, h = match['size']
        scale = match['scale']

        resized = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search_gray, resized, cv2.TM_CCOEFF_NORMED)
        result_norm = ((result - result.min()) / 
                            (result.max() - result.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(result_norm, cv2.COLORMAP_JET)
        heatmap_padded = cv2.copyMakeBorder(heatmap, int(np.floor((h-1)/2)), int(np.ceil((h-1)/2)), 
                                            int(np.floor((w-1)/2)), int(np.ceil((w-1)/2)), cv2.BORDER_DEFAULT)
        # heatmap_resized = cv2.resize(heatmap, (search_gray.shape[1], search_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        search_region_color = cv2.cvtColor(search_gray, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(search_region_color, 0.4, heatmap_padded, 0.6, 0)
        

        
        color = colors[idx % len(colors)]
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
        
        label = f"#{idx+1}: {match['score']:.2f}"
        cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # print(f"Match #{idx+1}: Score={match['score']:.3f}, Scale={match['scale']:.2f}, Pos=({x},{y})")

    # Display
    if DISPLAY_FRAMES:
        cv2.imshow("1. Original", image)
        cv2.imshow("2. Correlation Heatmap (white dots = above threshold)", overlay)
        cv2.imshow("3. Matches Found", result_image)
        cv2.waitKey(1)

    # print("\nHeatmap colors:")
    # print("  RED/YELLOW = High correlation (good match)")
    # print("  GREEN = Medium correlation")
    # print("  BLUE/BLACK = Low correlation (poor match)")
    # print("  WHITE DOTS = Above threshold")

    return center_x, center_y
    # cv2.destroyAllWindows()

    # cv2.destroyAllWindows()
    
    # Calculate and return median position
    # if len(all_x_positions) < 6 or len(all_y_positions) < 6:
    #     print(f"Found {len(all_x_positions)}, {len(all_y_positions)} across 1 frame.")
    #     return None, None
    
    # frame_width = image_gray.shape[1]
    # frame_height = image_gray.shape[0]result_image

    # median_x = (float(np.median(all_x_positions)) - frame_width / 2) / 15
    # median_y = (float(np.median(all_y_positions)) - frame_height / 2) / 15
    
    # print(f"Frame dimensions: {frame_width}x{frame_height}")
    # print(f"Processed {len(images)} frames, found positions in {len(all_x_positions)} frames")
    # print(f"Median position: ({median_x}, {median_y})")
    
    # return median_x, median_y


# Example usage:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        
        ret, frame = cap.read()

        x, y = find_median_position(frame, template_path=os.path.join(code_dir, "usbc_template.png"))
        if x is not None and y is not None:
            print(f"Final median coordinates: x={x}, y={y}")