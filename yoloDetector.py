from ultralytics import YOLO

# load a model
model = YOLO("yolov8n.yaml")

# training model
results = model.train(data = "/Users/arhan.sheth/Documents/Codes/DX/Detection/objectDetection/configs.yaml", epochs = 10)
