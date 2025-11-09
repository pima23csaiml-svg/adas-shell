Real-Time ADAS Prototype (LDW & FCW)

This project is a proof-of-concept for an Intelligent Driver Assistance System (ADAS) built using Python, OpenCV, and PyTorch. It processes a video feed from a file or webcam to provide two critical, real-time safety alerts:

Lane Departure Warning (LDW): Detects lane lines on the road and issues a visual warning if the vehicle drifts out of its lane.

Forward Collision Warning (FCW): Uses a YOLO AI model to detect vehicles, estimate their distance, and warn the driver of a potential collision.

Features

Real-Time Processing: Analyzes video frames with minimal latency.

Modular Design: ldw_module.py and fcw_module.py handle their tasks independently.

Object Detection: Uses a pre-trained YOLOv5 model from PyTorch Hub to detect cars, trucks, and buses.

Lane Detection: Uses classic computer vision (Canny edge detection and Hough Transform) to find lane lines.

Combined HMI: Both alerts are overlaid onto a single, clear video display for the driver.

How It Works

The main_adas.py script reads a video stream frame by frame. Each frame is processed by both modules:

fcw_module.py: The frame is passed to the YOLO model, which returns a list of detected objects (cars, etc.) and their bounding box coordinates. The script then estimates the distance to these objects and draws red boxes and a "BRAKE!" warning for any vehicle that is too close.

ldw_module.py: The frame is converted to grayscale, blurred, and masked to a "Region of Interest" (the road ahead). Canny edge detection and a Hough Transform find the lane lines, which are then drawn on the frame. A "Drifting Left/Right" warning appears if a lane line is lost.

Main: The final frame, now with all overlays, is displayed to the user.

Installation

This project requires Python 3.

Clone or Download:
Download the three .py files (main_adas.py, ldw_module.py, fcw_module.py) into a new folder.

Install Python Libraries:
The YOLOv5 model requires several dependencies. You can install all necessary libraries using pip. Open your terminal or command prompt and run:

# Main libraries for vision and AI
pip install opencv-python-headless numpy torch torchvision

# Dependencies required by the YOLOv5 model
pip install pandas tqdm seaborn


opencv-python-headless: For all image processing and video reading.

numpy: For numerical calculations.

torch & torchvision: The deep learning framework that runs YOLO.

pandas, tqdm, seaborn: Helper libraries that YOLO uses for processing results and showing progress bars.

How to Run

Add a Test Video:

Download a dashcam video (e.g., from Pexels, YouTube) and save it in the same folder as the Python files.

For this example, let's assume you named it test_video.mp4.

Edit main_adas.py:

Open main_adas.py in a text editor.

Find the line VIDEO_SOURCE = "test_video.mp4" near the top.

Change "test_video.mp4" to the name of your video file.

(Optional) To use your webcam: Change the line to VIDEO_SOURCE = 0.

Run the Program:

Open your terminal or command prompt.

Navigate to the project folder:

cd path/to/your/ADAS-Project


Run the main script:

python main_adas.py


First-Time Run:

The first time you run it, the program will need an internet connection to download the pre-trained yolov5s.pt model (about 14MB). This is a one-time setup.

To Quit:

A window titled "ADAS Prototype" will open, showing the video.

Press the 'q' key on your keyboard to stop the script and close the window.

Future Improvements

Train a Custom Model: The pre-trained YOLO model is good, but you can train your own model on custom dashcam footage for much higher accuracy.

Improve Distance Estimation: The current method is an approximation. A more robust solution would involve a stereo camera (for depth) or calibrating the focal length of the specific camera being used.

Sensor Fusion: Combine the two modules. For example, only trigger FCW for a car that is inside the detected lane lines.