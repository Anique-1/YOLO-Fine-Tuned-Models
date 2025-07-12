import os
from ultralytics import YOLO

# Path to the trained model
MODEL_PATH = 'German_Traffic_Sign_Detection_Benchmark.pt'

# Ask user for image path
image_path = input('Enter the path to the image you want to test: ').strip()

if not os.path.isfile(image_path):
    print(f'Image not found: {image_path}')
    exit(1)

# Directory to save the prediction
upload_dir = 'upload'
os.makedirs(upload_dir, exist_ok=True)

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Run inference
results = model(image_path)

# Print and save results
for result in results:
    print(f'Predictions for {os.path.basename(image_path)}:')
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0]) if hasattr(box, 'cls') else None
        conf = float(box.conf[0]) if hasattr(box, 'conf') else None
        xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
        print(f'  Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}')
    # Save the image with predictions drawn
    save_path = os.path.join(upload_dir, 'predicted_image.jpg')
    result.save(filename=save_path)
    print(f'Saved prediction image to: {save_path}') 