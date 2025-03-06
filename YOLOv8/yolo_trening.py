from ultralytics import YOLO

#Load a model
model = YOLO('yolov8n-cls.pt')

#Train the model
results = model.train(data='zdj2', epochs=50, imgsz=64)