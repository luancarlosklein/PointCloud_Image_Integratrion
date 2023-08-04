from cvlib.object_detection import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import matplotlib.pyplot as plt
import math

## Define the image
image_path = "image301_2.jpg"


## Read the image
frame = cv2.imread(image_path)
## Get the sizes of the image
width_image, height_image = frame.shape[:2]
## Rotate the image to identify the person
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov4-tiny')


## Define the size of the person
width_person = bbox[0][2]
height_person  = bbox[0][3]
## Define the initial point of the person (uper left)
initial_point = (bbox[0][0], bbox[0][1])
## Draw the box around the person
frame = draw_bbox(frame, bbox, label, conf)
## Rotate the image to the original
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

## Do the rotation of the upper left point
rotate_initial_point = (initial_point[1], width_image - initial_point[0])


print(f"Initial Point: {rotate_initial_point}")
print(f"Height: {height_person}")
print(f"Widht: {width_person}")

#####################
### Just to show the point
# Definir o raio do círculo
radius = 50

# Definir a cor do círculo (BGR)
color = (0, 0, 255)  # Vermelho: (B, G, R)


# Desenhar o círculo na imagem
cv2.circle(frame, rotate_initial_point, radius, color, thickness=2)

###############################################

# Display the frame in the "Webcam" window
cv2.imshow("Webcam", frame)

# Release the VideoCapture object and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()