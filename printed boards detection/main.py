from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    # Train the model using the 'data.yaml' dataset for 100 epochs with GPU
    results = model.train(data='dataset\data.yaml', epochs=50, imgsz=640, device=0)

    # Save the trained model
    model.save('circuit_boards_detection_model.pt')

    # After training, you can run validation
    # results = model.val()

    # You can also export the model to ONNX format
    # success = model.export(format='onnx')
