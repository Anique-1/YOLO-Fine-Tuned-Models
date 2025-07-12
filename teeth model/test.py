from ultralytics import YOLO
import os

# Load the trained model
model = YOLO('teeth_detection_model.pt')

# Path to the single image for prediction
image_path = 'teeth model/dataset/test/images/000dc27f-NAJIB_MARDANLOO_MASUME_2020-07-12185357_jpg.rf.8a45b187ef17ffeb7bd70850c94e7a39.jpg'

# Run prediction on the single image
# The result will be saved in a 'runs/detect/predict' directory.
results = model.predict(image_path, save=True, device='gpu')

print("Prediction finished.")
# The path to the saved image will be in the results
# By default, it's in a directory like 'runs/detect/predict'
# The result object also contains information about the detections.
for r in results:
    if r.save_dir:
        print(f"Results saved to {r.save_dir}")
        break 