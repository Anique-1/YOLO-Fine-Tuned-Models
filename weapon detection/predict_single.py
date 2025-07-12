from ultralytics import YOLO
import os
import sys

# Path to the trained model (update if needed)
MODEL_PATH = 'weapon_detection.pt'  # or 'teeth_detection_model.pt' if that's your latest
OUTPUT_DIR = 'single_image_result'

if len(sys.argv) != 2:
    print(f"Usage: python {os.path.basename(__file__)} <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

# Run inference
results = model(image_path)

# Save the annotated image to the output directory
output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
results[0].save(filename=output_path)

print(f"Processed image: {image_path}\nResult saved to: {output_path}") 