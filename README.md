# AI Model Detection Collection

This repository contains a comprehensive collection of pre-trained AI models for various computer vision and detection tasks. Each folder contains a specific model along with its corresponding test and prediction scripts.

## 📁 Project Structure

### 🏀 Basketball Detection
- **Model**: `basketball_detection_model.pt` (5.2MB)
- **Scripts**:
  - `test.py` - Test script for basketball detection
  - `predict_video.py` - Video prediction script for basketball detection
- **Description**: Detects basketballs in images and videos

### 🧠 Brain Tumor Detection
- **Model**: `brain_tumor_segmentation_model.pt` (6.0MB)
- **Scripts**:
  - `test_model.py` - Test script for brain tumor segmentation
- **Description**: Performs brain tumor segmentation on medical images

### ☕ Coffee Box Detection
- **Model**: `coffee_box_detection_model.pt`
- **Scripts**:
  - `test.py` - Test script for coffee box detection
  - `predict_video.py` - Video prediction script for coffee box detection
- **Description**: Detects coffee boxes in images and videos

### 👶 Fetal Detection
- **Model**: `fetal_segmentation_model.pt`
- **Scripts**:
  - `test_model.py` - Test script for fetal segmentation
  - `predict_video.py` - Video prediction script for fetal detection
- **Description**: Performs fetal segmentation on ultrasound images

### 🚦 German Traffic Sign Detection
- **Model**: `German_Traffic_Sign_Detection_Benchmark.pt` (5.3MB)
- **Scripts**:
  - `test_one_image.py` - Test script for single image traffic sign detection
- **Description**: Detects German traffic signs in images

### 🍃 Leave Detection
- **Model**: `leave_disease_detection_model.pt`
- **Scripts**:
  - `test.py` - Test script for leave disease detection
  - `predict_video.py` - Video prediction script for leave detection
- **Description**: Detects diseases in plant leaves

### 👥 Multi People Tracking
- **Model**: `yolov8n.pt` (6.2MB)
- **Scripts**:
  - `main.py` - Main tracking script (289 lines)
- **Description**: Tracks multiple people in videos using YOLOv8

### 🧘 Pose Detection
- **Model**: `human_pose_detection.pt`
- **Scripts**:
  - `test_yolo_pose.py` - Test script for pose detection
  - `test_yolo_pose_video.py` - Video prediction script for pose detection
- **Description**: Detects human poses in images and videos

### 🦺 PPE Detection
- **Model**: `PPE_detection_model.pt`
- **Scripts**:
  - `test.py` - Test script for PPE detection
  - `video_test.py` - Video test script for PPE detection
- **Description**: Detects Personal Protective Equipment (PPE) in images and videos

### 🔌 Printed Boards Detection
- **Model**: `circuit_boards_detection_model.pt`
- **Scripts**:
  - `main.py` - Main script for circuit board detection
  - `test.py` - Test script for circuit board detection
- **Description**: Detects printed circuit boards in images

### 🩺 Skin Disease Detection
- **Model**: `skin_disease_detection_model.pt`
- **Scripts**:
  - `test.py` - Test script for skin disease detection
  - `test_folder.py` - Batch testing script for multiple images
- **Description**: Detects various skin diseases in images

### ☀️ Solar Detection
- **Model**: `solar_detection_model.pt`
- **Scripts**:
  - `test.py` - Test script for solar detection
  - `predict_video.py` - Video prediction script for solar detection
- **Description**: Detects solar panels and related objects

### 🦷 Teeth Model
- **Model**: `teeth_detection_model.pt`
- **Scripts**:
  - `test.py` - Test script for teeth detection
- **Description**: Detects and analyzes teeth in dental images

### 🔫 Weapon Detection
- **Model**: `weapon_detection.pt` (5.3MB)
- **Scripts**:
  - `test.py` - Test script for weapon detection
  - `predict_video.py` - Video prediction script for weapon detection
  - `predict_single.py` - Single image prediction script
- **Description**: Detects weapons in images and videos

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- OpenCV
- YOLOv8 (for pose detection and multi-people tracking)

### Installation
```bash
# Install required packages
pip install torch torchvision
pip install opencv-python
pip install ultralytics  # For YOLOv8 models
```

### Usage
Each folder contains specific scripts for testing and prediction. Generally:

1. **For testing**: Run the `test.py` or `test_model.py` script
2. **For video prediction**: Run the `predict_video.py` script
3. **For single image prediction**: Run the `predict_single.py` script (where available)

### Example Usage
```bash
# Test basketball detection
cd "basketball detection"
python test.py

# Run video prediction for weapon detection
cd "weapon detection"
python predict_video.py

# Test pose detection
cd "pose detection"
python test_yolo_pose.py
```

## 📊 Model Information

- **Total Models**: 14 different detection models
- **Model Types**: Object detection, segmentation, pose estimation, tracking
- **File Formats**: `.pt` (PyTorch models)
- **Total Size**: Approximately 50+ MB of model files

## 🔧 Customization

Each model can be customized by:
1. Modifying the test scripts to work with your specific data
2. Adjusting confidence thresholds in the prediction scripts
3. Training on your own dataset for domain-specific applications

## 📝 Notes

- All models are pre-trained and ready to use
- Some models require specific input formats (check individual scripts)
- Video prediction scripts typically work with common video formats (MP4, AVI, etc.)
- Test scripts usually work with common image formats (JPG, PNG, etc.)

## 🤝 Contributing

Feel free to contribute by:
- Adding new detection models
- Improving existing scripts
- Adding documentation for specific use cases
- Reporting issues or bugs

## 📄 License

This project contains various pre-trained models. Please check individual model licenses and terms of use before commercial deployment.

---

**Note**: This collection is for research and educational purposes. Some models may have specific licensing requirements for commercial use. 