from ultralytics import YOLO
import os

# Load the trained model
model = YOLO('skin_disease_detection_model.pt')

# Path to the single image for prediction
image_path = 'pexels-karolina-grabowska-6643064.jpg'

# Run prediction on the single image
# The result will be saved in a 'runs/detect/predict' directory.
results = model.predict(image_path, save=True)

print("Prediction finished.")
# The path to the saved image will be in the results
# By default, it's in a directory like 'runs/detect/predict'
# The result object also contains information about the detections.
for r in results:
    if r.save_dir:
        print(f"Results saved to {r.save_dir}")
        break 