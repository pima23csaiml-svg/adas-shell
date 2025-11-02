import cv2
import torch
from ldw_module import process_frame_ldw
from fcw_module import load_fcw_model, process_frame_fcw

def main():
    # --- Configuration ---
    # !! Replace with your video file. Or use 0 for webcam. !!
    VIDEO_SOURCE = "test_video.mp4" 
    
    # --- Initialization ---
    print("Loading video capture...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {width}x{height}")

    print("Loading FCW model (YOLOv5s)... This may take a moment.")
    fcw_model, device = load_fcw_model()
    print("Model loaded.")

    # --- Main Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
        
        # Create a copy to draw on
        display_frame = frame.copy()

        # 1. Run FCW processing
        # This function will draw FCW boxes and warnings on the frame
        display_frame = process_frame_fcw(display_frame, fcw_model, device)
        
        # 2. Run LDW processing
        # This function will draw LDW lines and warnings on the *same* frame
        display_frame = process_frame_ldw(display_frame)
        
        # Now 'display_frame' has both sets of overlays
        cv2.imshow("ADAS Prototype", display_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()