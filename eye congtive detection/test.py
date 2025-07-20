from ultralytics import YOLO
import os
import cv2
import numpy as np

# Path to the trained model (update if needed)
MODEL_PATH = 'eye_conjuntiva_detection_model.pt'  # Eye conjunctiva detection model
TEST_IMAGES_DIR = 'dataset/test/images'
OUTPUT_DIR = 'test_results'

# Class names and dark colors (from dataset/data.yaml)
CLASS_NAMES = ['forniceal', 'forniceal_palpebral', 'palpebral']
# Dark colors in BGR format for OpenCV
CLASS_COLORS = [
    (0, 0, 139),    # Dark red/blue for forniceal
    (0, 100, 0),    # Dark green for forniceal_palpebral
    (139, 0, 139)   # Dark magenta for palpebral
]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

# Get all image files in the test images directory
image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} test images")
print("Processing images for eye conjunctiva segmentation...")
print(f"Categories: {CLASS_NAMES}")

def create_segmentation_visualization(image, results):
    """Create segmentation-only visualization with dark colors and external labels with arrows"""
    annotated_image = image.copy()
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()  # shape: (num_instances, H, W)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # class indices for each mask
        for mask, cls_idx in zip(masks, classes):
            if cls_idx < len(CLASS_COLORS):
                color = CLASS_COLORS[cls_idx]
                mask = (mask > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(annotated_image, dtype=np.uint8)
                for c in range(3):
                    colored_mask[:, :, c] = mask_resized * color[c]
                # Blend the colored mask with the original image (dark overlay, higher alpha)
                annotated_image = cv2.addWeighted(annotated_image, 1.0, colored_mask, 0.6, 0)
                # Find centroid
                moments = cv2.moments(mask_resized)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    label = CLASS_NAMES[cls_idx]
                    # Place label outside the mask (above and to the right)
                    offset_x, offset_y = 60, -40
                    label_x = min(cx + offset_x, annotated_image.shape[1] - 10)
                    label_y = max(cy + offset_y, 20)
                    # Draw arrow from label to centroid
                    cv2.arrowedLine(
                        annotated_image,
                        (label_x, label_y),
                        (cx, cy),
                        color,
                        2,
                        tipLength=0.2
                    )
                    # Draw label background (light highlight)
                    font_scale = 4
                    font_thickness = 4
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    highlight_color = (255, 255, 200)  # Light yellow
                    padding_x, padding_y = 12, 16
                    cv2.rectangle(
                        annotated_image,
                        (label_x - padding_x, label_y - text_size[1] - padding_y),
                        (label_x + text_size[0] + padding_x, label_y + padding_y),
                        highlight_color, -1
                    )
                    # Draw label text (dark color)
                    cv2.putText(
                        annotated_image,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),  # Black text
                        font_thickness,
                        cv2.LINE_AA
                    )
    return annotated_image

for i, img_name in enumerate(image_files, 1):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    print(f"Processing {i}/{len(image_files)}: {img_name}")
    
    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_name}")
        continue
    
    # Run inference
    results = model(image, verbose=False)
    
    # Create segmentation-only visualization
    annotated_image = create_segmentation_visualization(image, results[0])
    
    # Save the annotated image to the output directory
    output_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(output_path, annotated_image)

print(f"\nProcessed {len(image_files)} images successfully!")
print(f"Results saved to '{OUTPUT_DIR}' directory.")
print("Segmentation masks applied with dark colors:")
for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    print(f"  {name}: Dark color (BGR: {color})") 