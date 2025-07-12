import cv2
from ultralytics import YOLO
import tkinter as tk

# Path to the model and video file
MODEL_PATH = 'human_pose_detection.pt'  # Change if you want to use a different model
VIDEO_PATH = 'video-1.mp4'  # Path to the video file

# Get screen size using tkinter
root = tk.Tk()
root.withdraw()  # Hide the main window
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

# Load the YOLO pose model
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Resize for large preview (full screen)
    preview = cv2.resize(annotated_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Show the live preview
    cv2.imshow('YOLO Pose Detection - Live Preview', preview)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 