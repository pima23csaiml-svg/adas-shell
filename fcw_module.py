import cv2
import torch
import numpy as np

# --- These values are for demonstration and must be calibrated! ---
# Assume an average car width is 1.8 meters
KNOWN_WIDTH_METERS = 1.8 
# This is found through calibration (focal length in pixels)
FOCAL_LENGTH_PIXELS = 800 # !! GUESS - YOU MUST CALIBRATE THIS !!

# Safe Time-to-Collision threshold (in seconds)
TTC_THRESHOLD = 2.0 

def load_fcw_model():
    """Loads the YOLOv5s model from PyTorch Hub."""
    try:
        # --- THIS LINE IS UPDATED ---
        # Removed ':v6.0' to load the latest compatible version
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    except Exception as e:
        print(f"Error loading model from PyTorch Hub: {e}")
        print("Attempting to load from local cache if available...")
        # --- THIS LINE IS UPDATED ---
        # Removed ':v6.0' to load the latest compatible version
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)

    
    # Filter for objects we care about: car, truck, bus
    # 2: car, 5: bus, 7: truck
    model.classes = [2, 5, 7] 
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"FCW model loaded and using device: {device}")
    return model, device

def process_frame_fcw(frame, model, device):
    """
    Processes a single video frame for Forward Collision Warning.
    This function now returns the original frame with overlays.
    """
    
    # Create a copy to draw on
    output_frame = frame.copy()
    
    # 1. Run model inference
    # Convert BGR (OpenCV) to RGB (YOLO)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    
    # Get detections as a pandas DataFrame
    detections = results.pandas().xyxy[0] 
    
    vehicle_detected = False
    
    for _, det in detections.iterrows():
        # Get bounding box coordinates
        xmin = int(det['xmin'])
        ymin = int(det['ymin'])
        xmax = int(det['xmax'])
        ymax = int(det['ymax'])
        label = det['name']
        confidence = det['confidence']
        
        if confidence > 0.5: # Filter weak detections
            vehicle_detected = True
            
            # --- Inaccurate Distance Estimation (for demo only) ---
            # d = (W * F) / w
            # W = KNOWN_WIDTH_METERS
            # F = FOCAL_LENGTH_PIXELS
            # w = width of bounding box in pixels
            bbox_width_pixels = xmax - xmin
            
            if bbox_width_pixels > 0:
                distance_meters = (KNOWN_WIDTH_METERS * FOCAL_LENGTH_PIXELS) / bbox_width_pixels
            else:
                distance_meters = float('inf')

            # --- Simple TTC Calculation ---
            # NOTE: This assumes 'our' speed. We don't have it, so we'll
            # just show a warning if the *distance* is too close.
            # A real system would use `TTC = distance / relative_velocity`
            
            # FAKE TTC (assuming we travel at 10 m/s ~ 36 kph)
            our_speed_mps = 10 
            ttc = distance_meters / our_speed_mps 
            
            color = (0, 255, 0) # Green
            warning = f"{label.upper()}: {distance_meters:.1f}m"

            if ttc < TTC_THRESHOLD: # Check Time-to-Collision
                color = (0, 0, 255) # Red
                warning = f"!! BRAKE !! {label.upper()}: {distance_meters:.1f}m"
                
                # Draw a prominent warning
                cv2.putText(output_frame, "FORWARD COLLISION WARNING!", 
                            (output_frame.shape[1] // 2 - 300, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

            # Draw bounding box
            cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label
            label_bg_color = (color[0] // 2, color[1] // 2, color[2] // 2) # Darker bg
            cv2.rectangle(output_frame, (xmin, ymin - 20), (xmin + len(warning) * 10, ymin), label_bg_color, -1)
            cv2.putText(output_frame, warning, (xmin, ymin - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    return output_frame