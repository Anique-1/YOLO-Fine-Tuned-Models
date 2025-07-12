from ultralytics import YOLO
import os
import cv2
import numpy as np

# Path to the trained model (update if needed)
MODEL_PATH = 'fetal_segmentation_model.pt'  # or 'teeth_detection_model.pt' if that's your latest
OUTPUT_DIR = 'video_results'
VIDEO_PATH = '3099938-hd_1920_1080_30fps.mp4'  # Set your video path here

# Color customization options
# You can change these colors to customize the segmentation appearance
SEGMENTATION_COLORS = {
    'Baby': (0, 255, 0),  # Green for baby segmentation (BGR format)
    # Add more classes if you have them: 'class_name': (B, G, R)
}

# Visualization options
SHOW_BOXES = True      # Show bounding boxes
SHOW_LABELS = True     # Show class labels
SHOW_CONF = True       # Show confidence scores
MASK_ALPHA = 0.3       # Transparency of segmentation mask (0.0 = transparent, 1.0 = opaque)
LINE_WIDTH = 2         # Width of bounding box lines

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

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

def create_custom_annotated_frame(frame, results):
    """Create custom annotated frame with specified colors and settings"""
    annotated_frame = frame.copy()
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for i, (mask, box, class_id, conf) in enumerate(zip(masks, boxes, class_ids, confidences)):
            # Get class name
            class_name = model.names[int(class_id)]
            
            # Get color for this class
            color = SEGMENTATION_COLORS.get(class_name, (0, 255, 0))  # Default to green
            
            # Apply mask with custom color and transparency
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # Create colored mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask_bool] = color
            
            # Blend with original frame
            annotated_frame = cv2.addWeighted(annotated_frame, 1 - MASK_ALPHA, colored_mask, MASK_ALPHA, 0)
            
            # Draw bounding box if enabled
            if SHOW_BOXES:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, LINE_WIDTH)
            
            # Draw label if enabled
            if SHOW_LABELS:
                label = class_name
                if SHOW_CONF:
                    label += f' {conf:.2f}'
                
                # Calculate text position
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1]
                
                # Draw text background
                cv2.rectangle(annotated_frame, 
                            (text_x, text_y - text_size[1] - 5),
                            (text_x + text_size[0], text_y + 5),
                            color, -1)
                
                # Draw text
                cv2.putText(annotated_frame, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference on the frame
    results = model(frame, verbose=False)
    
    # Use custom annotation instead of default plot
    annotated_frame = create_custom_annotated_frame(frame, results)
    
    # Resize for live preview
    preview_frame = cv2.resize(annotated_frame, (1280, 720))
    
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
print(f"Segmentation color: {SEGMENTATION_COLORS}")
print(f"Mask transparency: {MASK_ALPHA}") 