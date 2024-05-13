import cv2

import numpy as np

from ultralytics import YOLO

# For this - at least at start - we will use YOLO object detection
model = YOLO("yolo-Weights/yolov8n.pt")

# Init for classification task
class_names = model.names

# And for capture cv2 with videocapturing
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 960)

while True:
    # Capturing camera with cv2
    not_fail, image = capture.read()
    results = model(image, stream=True)

    # Unpacking results
    for result in results: 
        for box in result.boxes:
            # Getting boxes for predictions
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Marking objects
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Confidence score for objects
            conf_score = np.round(int(box.conf[0])*100)/100

            # Predicted object
            pred_class = int(box.cls[0])

            # Marking object on camera
            cv2.putText(image, class_names[pred_class], [np.round(x1, x2), y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Webcam', image)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()