import cv2
import numpy as np

# Define camera calibration parameters (replace with your calibration results)
focal_length = 1000  # Replace with your focal length
known_width = 10.0   # Replace with the known width of the object in centimeters

# Camera calibration matrix (replace with your calibration matrix)
camera_matrix = np.array([[focal_length, 0, 320], [0, focal_length, 240], [0, 0, 1]])
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"
# Assuming you have loaded the model earlier in your code
model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLabels = []
file_name = 'labels.txt'
with open(file_name) as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Set the input size for the model
model.setInputSize(320, 320)

# Other settings
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture('pexels-geoge-morina-5330833.mp4')

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('Cannot open the video')

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    # Set the input size for the model
    model.setInputSize(320, 320)

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                # Calculate object distance in centimeters
                width_in_pixels = boxes[2] - boxes[0]
                depth_cm = (known_width * focal_length) / max(width_in_pixels, 1)  # Avoid division by zero

                # Check bounding box area before processing
                if width_in_pixels * boxes[3] < 2000:
                    continue

                # Annotate the detected objects with depth information
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                label = f"{classLabels[ClassInd - 1]} - {depth_cm:.2f} cm"
                cv2.putText(frame, label, (boxes[0] + 10, boxes[1] + 40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
