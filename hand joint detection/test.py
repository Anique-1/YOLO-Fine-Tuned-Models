from ultralytics import YOLO
import os
from PIL import Image

# Path to the trained model (update if needed)
MODEL_PATH = 'hand_joint_detection_model.pt'  # or 'teeth_detection_model.pt' if that's your latest
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
    # Plot with custom line width (thickness)
    annotated_img = results[0].plot(line_width=1)  # Change 4 to your desired thickness
    # Convert numpy array to PIL Image and save
    annotated_img_pil = Image.fromarray(annotated_img)
    # Save the annotated image
    output_path = os.path.join(OUTPUT_DIR, img_name)
    annotated_img_pil.save(output_path)

print(f"Processed {len(image_files)} images. Results saved to '{OUTPUT_DIR}' directory.") 