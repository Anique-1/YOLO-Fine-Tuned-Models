import os
from ultralytics import YOLO

# Path to the trained model
MODEL_PATH = 'circuit_boards_detection_model.pt'
# Path to test images
TEST_IMAGES_DIR = os.path.join('dataset', 'test', 'images')
# Path to save prediction images
PRED_DIR = os.path.join('dataset', 'test', 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Get all test image paths
image_files = [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} test images.")

for img_path in image_files:
    results = model(img_path)
    print(f"\nResults for {os.path.basename(img_path)}:")
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
            conf = float(box.conf[0]) if hasattr(box, 'conf') else None
            xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
            print(f"  Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")
        # Save the image with predictions drawn
        save_path = os.path.join(PRED_DIR, os.path.basename(img_path))
        result.save(filename=save_path) 