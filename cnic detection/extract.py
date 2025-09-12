import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict

class YOLOTextExtractor:
    def __init__(self, model_path: str = "cnic_detection_model.pt"):
        """
        Initialize the YOLO Text Extractor
        
        Args:
            model_path: Path to your trained YOLO model. If None, will use YOLOv8n
        """
        # Load YOLO model
        if model_path and os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
            print(f"Loaded custom YOLO model from {model_path}")
        else:
            # Use pretrained model for general object detection
            # You should replace this with your trained text detection model
            self.yolo_model = YOLO('cnic_detection_model.pt')
            print("Using YOLOv8n pretrained model. Please train a custom model for better text detection.")
        
        # Initialize EasyOCR reader for Urdu and English
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        print("OCR Reader initialized for English and Urdu")
    
    def detect_text_regions(self, image_path: str, confidence_threshold: float = 0.25) -> List[Dict]:
        """
        Detect text regions using YOLO
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of detected regions with bounding boxes
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run YOLO detection
        results = self.yolo_model(image, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.yolo_model.names[class_id] if class_id < len(self.yolo_model.names) else 'unknown'
                    })
        
        return detections
    
    def extract_text_from_regions(self, image_path: str, detections: List[Dict]) -> List[Dict]:
        """
        Extract text from detected regions using OCR
        
        Args:
            image_path: Path to the input image
            detections: List of detected regions
        
        Returns:
            List of regions with extracted text
        """
        image = cv2.imread(image_path)
        results = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Crop the detected region
            cropped_region = image[y1:y2, x1:x2]
            
            if cropped_region.size == 0:
                continue
            
            # Apply OCR to the cropped region
            try:
                ocr_results = self.ocr_reader.readtext(cropped_region)
                
                extracted_texts = []
                for (bbox_ocr, text, conf) in ocr_results:
                    if conf > 0.3:  # Filter low confidence text
                        extracted_texts.append({
                            'text': text,
                            'confidence': conf,
                            'bbox_in_crop': bbox_ocr
                        })
                
                results.append({
                    'detection_id': i,
                    'yolo_bbox': detection['bbox'],
                    'yolo_confidence': detection['confidence'],
                    'yolo_class': detection['class_name'],
                    'extracted_texts': extracted_texts
                })
                
            except Exception as e:
                print(f"Error processing region {i}: {e}")
                continue
        
        return results
    
    def process_image(self, image_path: str, output_dir: str = "output") -> Dict:
        """
        Complete pipeline: detect regions and extract text
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
        
        Returns:
            Dictionary containing all results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing image: {image_path}")
        
        # Step 1: Detect text regions
        print("Detecting text regions...")
        detections = self.detect_text_regions(image_path)
        print(f"Found {len(detections)} regions")
        
        # Step 2: Extract text from regions
        print("Extracting text from regions...")
        text_results = self.extract_text_from_regions(image_path, detections)
        
        # Step 3: Visualize results
        self.visualize_results(image_path, text_results, output_dir)
        
        # Step 4: Compile final results
        final_results = {
            'image_path': image_path,
            'total_detections': len(detections),
            'text_extractions': text_results,
            'all_extracted_text': self.compile_all_text(text_results)
        }
        
        return final_results
    
    def compile_all_text(self, text_results: List[Dict]) -> List[str]:
        """
        Compile all extracted text into a simple list
        """
        all_texts = []
        for result in text_results:
            for text_item in result['extracted_texts']:
                all_texts.append(text_item['text'])
        return all_texts
    
    def visualize_results(self, image_path: str, text_results: List[Dict], output_dir: str):
        """
        Visualize detection and text extraction results
        """
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Original image with YOLO detections
        ax1.imshow(image_rgb)
        ax1.set_title("YOLO Detections", fontsize=16)
        ax1.axis('off')
        
        for result in text_results:
            x1, y1, x2, y2 = result['yolo_bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x1, y1-5, f"Conf: {result['yolo_confidence']:.2f}", 
                    color='red', fontsize=10, weight='bold')
        
        # Plot 2: Image with extracted text overlay
        ax2.imshow(image_rgb)
        ax2.set_title("Extracted Text", fontsize=16)
        ax2.axis('off')
        
        for result in text_results:
            x1, y1, x2, y2 = result['yolo_bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='blue', linewidth=2)
            ax2.add_patch(rect)
            
            # Add extracted text
            y_offset = 0
            for text_item in result['extracted_texts']:
                ax2.text(x1, y2 + 15 + y_offset, 
                        f"{text_item['text']} ({text_item['confidence']:.2f})",
                        color='blue', fontsize=8, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                y_offset += 20
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save results to file
        """
        import json
        
        # Save JSON results
        with open(os.path.join(output_dir, 'extraction_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save text file with all extracted text
        with open(os.path.join(output_dir, 'extracted_text.txt'), 'w', encoding='utf-8') as f:
            f.write("Extracted Text from Image:\n")
            f.write("="*50 + "\n")
            for text in results['all_extracted_text']:
                f.write(f"{text}\n")

# Main execution example
if __name__ == "__main__":
    # Initialize the text extractor (OCR-based, no YOLO needed)
    extractor = YOLOTextExtractor()
    
    # Process an image
    image_path = "dataset/train/images/IMG_4223_jpg.rf.60b24ce6c14d87771a651bb8e170e9d2.jpg"  # Replace with your image path
    
    try:
        # Run the complete pipeline
        results = extractor.process_image(image_path, output_dir="output")
        
        # Print results
        print("\n" + "="*50)
        print("TEXT EXTRACTION RESULTS")
        print("="*50)
        print(f"Total text regions found: {results['total_detections']}")
        print(f"Total extracted texts: {len(results['all_extracted_text'])}")
        
        # Print categorized text
        print("\nCategorized Text:")
        for category, texts in results['categorized_text'].items():
            print(f"\n{category.upper()}:")
            for text_info in texts:
                print(f"  - {text_info['text']} (confidence: {text_info['confidence']:.2f})")
        
        print("\nAll Extracted Text:")
        for i, text in enumerate(results['all_extracted_text'], 1):
            print(f"{i}. {text}")
        
        # Save results
        extractor.save_results(results, "output")
        
        print(f"\nResults saved to 'output' directory")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install easyocr opencv-python matplotlib pillow")
        
    print("\n" + "="*50)
    print("USAGE INSTRUCTIONS")
    print("="*50)
    print("1. Place your image file in the same directory")
    print("2. Update 'image_path' variable with your image filename")
    print("3. Run the script: python text_extractor.py")
    print("4. Check the 'output' folder for results")
    print("5. The script will automatically detect and extract:")
    print("   - CNIC numbers")
    print("   - Names and father names")
    print("   - Dates")
    print("   - Address information")
    print("   - General text in both Urdu and English")