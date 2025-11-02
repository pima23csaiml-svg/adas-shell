import cv2
import numpy as np

def get_roi(img):
    """Defines a trapezoidal Region of Interest (ROI) for lane detection."""
    height, width = img.shape[:2]
    # Adjust these vertices based on your camera's mounting angle
    vertices = np.array([
        [(int(width * 0.2), int(height * 0.9)),  # Bottom-left
         (int(width * 0.45), int(height * 0.6)), # Top-left
         (int(width * 0.55), int(height * 0.6)), # Top-right
         (int(width * 0.9), int(height * 0.9))]  # Bottom-right
    ], dtype=np.int32)
    
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def process_frame_ldw(frame):
    """
    Processes a single video frame to detect lanes and warn of departure.
    This function now returns the original frame with overlays.
    """
    height, width = frame.shape[:2]
    
    # Create a copy to draw on
    output_frame = frame.copy()
    
    # 1. Convert to grayscale and apply blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    
    # 3. Define Region of Interest (ROI)
    roi_edges = get_roi(edges)
    
    # 4. Hough Line Transform
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=100
    )
    
    # --- Lane averaging and departure logic ---
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Avoid division by zero
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.5:  # Left lane (negative slope)
                left_lines.append(line[0])
            elif slope > 0.5: # Right lane (positive slope)
                right_lines.append(line[0])

    warning = ""
    lane_color = (0, 255, 0) # Green
    
    # Simple logic: Check if we've lost one of the lanes
    if not left_lines and lines is not None:
        warning = "WARNING: Drifting Left!"
        lane_color = (0, 0, 255) # Red
    if not right_lines and lines is not None:
        warning = "WARNING: Drifting Right!"
        lane_color = (0, 0, 255) # Red

    # Draw detected lines (for visualization)
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw left lane
    if left_lines:
        avg_left_line = np.mean(left_lines, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = avg_left_line
        cv2.line(line_img, (x1, y1), (x2, y2), lane_color, 10)

    # Draw right lane
    if right_lines:
        avg_right_line = np.mean(right_lines, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = avg_right_line
        cv2.line(line_img, (x1, y1), (x2, y2), lane_color, 10)

    # Combine line image with original frame
    # We use addWeighted to make the lines semi-transparent
    output_frame = cv2.addWeighted(output_frame, 0.8, line_img, 1.0, 0.0)

    # Add warning text
    if warning:
        cv2.putText(output_frame, warning, (width // 2 - 200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    return output_frame