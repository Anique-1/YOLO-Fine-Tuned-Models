import os
from glob import glob
from ultralytics import YOLO
import cv2

# Paths
MODEL_PATH = 'human_pose_detection.pt'  # Change if you want to use a different model
TEST_IMAGES_DIR = 'dataset/test/images'
OUTPUT_DIR = 'runs/pose/test/'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Get all image paths
image_paths = glob(os.path.join(TEST_IMAGES_DIR, '*'))

# Run inference and save results
for img_path in image_paths:
    results = model(img_path)
    # Save annotated image
    base_name = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, base_name)
    annotated_img = results[0].plot()  # Get annotated image as numpy array
    cv2.imwrite(save_path, annotated_img)  # Save using OpenCV
    print(f"Processed {img_path} -> {save_path}")

print(f"All test images processed. Results saved to {OUTPUT_DIR}") 