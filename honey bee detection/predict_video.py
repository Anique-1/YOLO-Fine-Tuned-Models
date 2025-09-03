from ultralytics import YOLO
import os
import cv2

# Path to the trained model (update if needed)
MODEL_PATH = 'honeybee_detection_model.pt'
OUTPUT_DIR = 'video_results'
VIDEO_PATH = 'd42cdc7f97ff2322ba97349c8af34c81-720p-preview_R5hwg78t.mp4'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

# Define colors for each class (BGR format)
CLASS_COLORS = {
    'Honeybee': (0, 255, 0),  # Green
    # Add more classes and their colors if needed
}
# Default color for unknown classes
DEFAULT_COLOR = (128, 128, 128)  # Gray

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
output_path = os.path.join(OUTPUT_DIR, 'annotated', 'annotated_video.mp4')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Increased box and text parameters for better visibility
BOX_THICKNESS = 1  # Increased from 1 to 3
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5 # Increased from 0.7 to 1.2
TEXT_THICKNESS = 1  # Increased from 2 to 3
TEXT_PADDING = 8 # Increased padding around text

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference on the frame
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results[0].names[cls] if hasattr(results[0], 'names') else str(cls)
        
        # Get color based on class name
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        
        # Draw rectangle with increased thickness
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        
        # Draw label with larger, more visible text
        label_text = f"{label} {conf:.2f}"
        
        # Calculate text size with larger scale
        text_size = cv2.getTextSize(label_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
        text_width, text_height = text_size
        
        # Create a larger background rectangle for the text
        bg_x1 = x1
        bg_y1 = y1 - text_height - TEXT_PADDING
        bg_x2 = x1 + text_width + 10
        bg_y2 = y1
        
        # Ensure background rectangle stays within frame bounds
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(width, bg_x2)
        
        # Draw background rectangle with same color as bounding box
        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # Add a slight border to the text background for better visibility
        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 2)
        
        # Draw the text with larger font
        text_x = x1 + 5
        text_y = y1 - 8
        cv2.putText(annotated_frame, label_text, (text_x, text_y), 
                   TEXT_FONT, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)
        
        # Add a subtle shadow effect for even better visibility
        cv2.putText(annotated_frame, label_text, (text_x + 2, text_y + 2), 
                   TEXT_FONT, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS)
        cv2.putText(annotated_frame, label_text, (text_x, text_y), 
                   TEXT_FONT, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)
    
    # Resize for live preview
    preview_frame = cv2.resize(annotated_frame, (1920, 1080))
    
    # Show live preview
    cv2.imshow('Live Preview', preview_frame)
    
    # Write frame to output video
    out.write(annotated_frame)
    
    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video: {VIDEO_PATH}")
print(f"Annotated video saved at: {output_path}")
print("\nClass Colors:")
for class_name, color in CLASS_COLORS.items():
    print(f"  {class_name}: {color} (BGR)")
print(f"\nText settings:")
print(f"  Font scale: {TEXT_SCALE}")
print(f"  Text thickness: {TEXT_THICKNESS} (normal thickness)")
print(f"  Box thickness: {BOX_THICKNESS} (thick for visibility)")