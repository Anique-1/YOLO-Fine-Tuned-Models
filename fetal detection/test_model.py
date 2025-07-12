from ultralytics import YOLO
import os
from glob import glob

# Path to the trained model (use the fine-tuned model if available, else fallback to pretrained)
MODEL_PATH = 'fetal_segmentation_model.pt' if os.path.exists('fetal_segmentation_model.pt') else 'yolo11n-seg.pt'

# Directory containing test images
TEST_IMAGES_DIR = 'dataset/test/images/'
# Directory to save results
RESULTS_DIR = 'test_results/'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the model
model = YOLO(MODEL_PATH)

# Get all test image paths
image_paths = glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))

# Run inference and save results
for img_path in image_paths:
    results = model(img_path, save=True, project=RESULTS_DIR, name='', exist_ok=True)
    print(f"Processed: {img_path}")

print(f"All results saved to {RESULTS_DIR}") 