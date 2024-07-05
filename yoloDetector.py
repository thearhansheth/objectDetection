from ultralytics import YOLO

# load a model
model = YOLO("yolov8s.yaml")

# training model
results = model.train(data = "/Users/arhan.sheth/Documents/Codes/DX/Detection/objectDetection/configs.yaml", 
                      epochs = 200,
                      learning_rate = 0.001,
                      augmentation = True)

