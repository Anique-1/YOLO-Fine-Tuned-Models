from ultralytics import YOLO
import os

# Path to the trained model (update if needed)
MODEL_PATH = 'basketball_detection_model.pt'  # or 'teeth_detection_model.pt' if that's your latest
TEST_IMAGES_DIR = 'dataset/test/images'
OUTPUT_DIR = 'test_results'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

# Get all image files in the test images directory
image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in image_files:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    # Run inference
    results = model(img_path)
    # Save the annotated image to the output directory
    results[0].save(filename=os.path.join(OUTPUT_DIR, img_name))

print(f"Processed {len(image_files)} images. Results saved to '{OUTPUT_DIR}' directory.") 