from ultralytics import YOLO

# Load the trained model from the project's root directory.
model = YOLO('PPE_detection_model.pt')

# Specify the directory containing the test images.
test_images_path = 'PPE_Dataset/test/images'

# Run the prediction on the test images.
# The 'save=True' argument will automatically save the images with bounding boxes.
# The 'project' and 'name' arguments are used to define the output directory.
# Results will be saved in 'runs/detect/test_run'.
# `exist_ok=True` will overwrite existing directory.
model.predict(source=test_images_path, save=True, project='runs/detect', name='test_run', exist_ok=True)

print("Testing complete. The results are saved in the 'runs/detect/test_run' directory.") 