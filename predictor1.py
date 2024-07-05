from ultralytics import YOLO
import cv2 as cv
import matplotlib.pyplot as plt

# Loading the trained model
model = YOLO("/Users/arhan.sheth/Documents/Codes/DX/Detection/objectDetection/runs/detect/train7/weights/best.pt")

# Loading the input image
image_path = "/Users/arhan.sheth/Documents/Codes/DX/Detection/objectDetection/data/prediction/alpaca.jpeg"
image = cv.imread(image_path)
# Convert BGR to RGB
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  

# Making predictions
results = model.predict(image_rgb)

# Drawing the boxes and labels on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy)
        label = result.names[box.cls]
        confidence = box.conf

        # Drawing the rectangle
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Putting label and confidence score
        cv.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Saving the output image
cv.imwrite("/Users/arhan.sheth/Documents/Codes/DX/Detection/objectDetection/data/prediction/predicted_image.jpg", image)
