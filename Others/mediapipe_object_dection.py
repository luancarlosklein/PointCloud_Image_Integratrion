import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

model = "efficientdet_lite2_uint8.tflite"
image_path = os.path.join("image300.jpg")
img = cv2.imread(image_path)
#plt.imshow(img)

# Display the frame in the "Webcam" window
#cv2.imshow("Webcam", img)

#use Mediapipe Tasks API
base_options = python.BaseOptions(model_asset_path=model)
options = vision.ObjectDetectorOptions(base_options=base_options,score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#using Mediapipe Image Attribute initialize the input image path. 
image = mp.Image.create_from_file(image_path)
detect_image = detector.detect(image)
image = image.numpy_view()

for detection in detect_image.detections:

    # mAP score and the Detected image label
    target = detection.categories[0]
    category_name = target.category_name

    if category_name == "Person":

        # Insert bounding_box
        bbox = detection.bounding_box
        # the bounding box contains four parameters: 
        #x, y, width and height
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, (0,255,0), 3)

        score = round(target.score, 2)
        label = f"{category_name}:{score}"
        loc = (bbox.origin_x+15,bbox.origin_y+25)
        cv2.putText(image, label, loc, cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1)
cv2.imshow("Webcam", image)

print("foi")
# Release the VideoCapture object and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
