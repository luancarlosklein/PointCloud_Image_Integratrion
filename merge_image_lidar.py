#!/usr/bin/env python3

import rosbag
import os
import numpy
import cvlib as cv
import cv2
import matplotlib.pyplot as plt
import math
import keypoint_detection
import os

import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from mpl_toolkits.mplot3d import Axes3D
from cvlib.object_detection import YOLO
from cvlib.object_detection import draw_bbox
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


### Auxiliar Functions

## Function to calculate the angle between three points
def calc_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return round(np.degrees(angle))

## Function that receives a image, 
## three lists of the points from Lidar (adapted to the image)
## The original points
## The name of the joint
def generate_points_lidar(image_path, x, y, z, points, joint_name):    
    ## Call a function to detect the key_point. Return the (X, Y) of the joint
    joint = keypoint_detection.detect(image_path, joint_name)
    
    ## Define the point list, only with Y and Z
    uni_points = []
    for cont, value in enumerate(y):
        uni_points.append( (y[cont], z[cont]))

    ## Find the near points to joint
    near_points = []
    near_points_index = []
    ratio = 1
    ## Finf the first 5 points. If the ratio is bigger than 20, just stop
    while len(near_points) < 5:
        new_points, new_points_index = points_in_the_ratio(joint, uni_points, ratio)
        ratio += 1

        for i in new_points:
            if i not in near_points:
                near_points.append(i)

        for i in new_points_index:
            if i not in near_points_index:
                near_points_index.append(i)
        if ratio > 20:
            break

    ## Gets the original points from the LiDar
    x_original = points[:, 0]## Deep
    y_original = points[:, 1]## Widht
    z_original = points[:, 2]## Height

    ### Calculate the average between the points of the LiDar
    avg_lidar = [0, 0, 0]
    for i in near_points_index:
        avg_lidar[0] += x_original[i]
        avg_lidar[1] += y_original[i]
        avg_lidar[2] += z_original[i]
    
    avg_lidar[0] /= len(near_points_index)
    avg_lidar[1] /= len(near_points_index)
    avg_lidar[2] /= len(near_points_index)

    ## Return tbe average of the points and the respective index
    return(joint, avg_lidar, near_points_index)


## Calculate the distance between two points in 2D
def distance_two_points(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

## Return the points inside of a ratio related to one specific points, give a points list
def points_in_the_ratio(point, points_list, ration):
    near_points = []
    near_points_index = []
    for count, other_point in enumerate(points_list):
        distance = distance_two_points(point, other_point)
        if distance <= ration:
            near_points.append(other_point)
            near_points_index.append(count)
    return (near_points, near_points_index)

## Function to draw a specific point in a image
def draw_point(img, x, y, color):
    size = 3
    cv2.circle(img, (x, y), size, color, -1)

## Function to detect the person and return the initial and end point, and the size of the person
def person_detection(img):
    ## Model to identify the person
    model = "efficientdet_lite2_uint8.tflite"
    ## Use Mediapipe Tasks API
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    ## Using Mediapipe Image Attribute initialize the input image path. 
    image = mp.Image.create_from_file(image_path)
    detect_image = detector.detect(image)
    image = image.numpy_view()

    for detection in detect_image.detections:
        # mAP score and the Detected image label
        target = detection.categories[0]
        category_name = target.category_name
        if category_name == "person":
            # Insert bounding_box
            bbox = detection.bounding_box
            # the bounding box contains four parameters: 
            #x, y, width and height
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(frame, start_point, end_point, (0,255,0), 3)

            score = round(target.score, 2)
            label = f"{category_name}:{score}"
            loc = (bbox.origin_x+15,bbox.origin_y+25)
        break
    return (start_point, end_point, bbox.width, bbox.height)

## Define the image
image_path = os.path.join("Data_normal", "camera_data", "image282.jpg")
lidar_path = os.path.join('Data_normal', 'lidar_data', 'point_cloud280.bag')

## Read the image
frame = cv2.imread(image_path)
## Get the sizes of the image
width_image, height_image = frame.shape[:2]

## Get the informations related to the person in the image
key_point_image, final_point, width_person, height_person = person_detection(frame)

## Read the file with the cloudpoint
bag = rosbag.Bag(lidar_path)

points = []

## read the data and put in arrays
for topic, msg, t in bag.read_messages(topics=['/rslidar_points']):
    point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(point_cloud))

    # Create the 3D Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ## Get the sizes of the data in the lidar
    deep_lidar = max(points[:, 0]) - min(points[:, 0])
    widht_lidar = max(points[:, 1]) - min(points[:, 1])
    height_lidar = max(points[:, 2]) - min(points[:, 2])
    
    ## Calculate the ratios 
    ratio_x = 1
    ratio_y = width_person / widht_lidar
    ratio_z = height_person / height_lidar
    
    points_height = []
    points_deep = []
    points_widht = []

    ## Normalize the points to fit in the image
    x = points[:, 0] * ratio_x## Deep
    y = points[:, 1] * ratio_y## Widht
    z = points[:, 2] * ratio_z## Height
    
    ## The values can be negative. In this case, just sum the minimum value to all the other values to put the minimum in zero
    min_x = min(x)
    min_y = min(y)
    min_z = min(z)

    if min_x < 0:
        x = x + abs(min_x)
    if min_y < 0:
        y = y + abs(min_y)
    if min_z < 0:
        z = z + abs(min_z)
    

    ## The minimum point of the image (0,0) in the image is different of the LiDar points.
    ## So, it is necessary make the conversion and adaptation
    max_y = max(y)
    max_z = max(z)
    y = (y*(-1)) + max_y
    z = (z*(-1)) + max_z


    ## Now, it is necessary move the clound point to the correct point of the image (to match both)
    
    ## Define the key poin in the lidar
    key_point_lidar = (min(y), min(z))

    ## Calculate the difference the key point of the image (top left) and the same point of the image
    desl_y = key_point_image[0] - key_point_lidar[0]
    desl_z = key_point_image[1] - key_point_lidar[1]

    ## Do the move
    y = y + desl_y
    z = z + desl_z

    ## Show the cloud points
    if len(points) > 0:
        ax.scatter(x, y, z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D CloudPoint')

    # # Exibe o gráfico
    plt.show()

    ## Define the point (x, y, z) related to one specific part of the body
    mediapipe_left_shoulder, left_shoulder, points_left_shoulder = generate_points_lidar(image_path, x, y, z, points, "LEFT_SHOULDER")
    mediapipe_left_elbow, left_elbow, points_left_elbow = generate_points_lidar(image_path, x, y, z, points, "LEFT_ELBOW")
    mediapipe_left_hip, left_hip, points_left_hip = generate_points_lidar(image_path, x, y, z, points, "LEFT_HIP")

    ## Define the adjusted hip (straigt)
    left_hip_straight = [left_hip[0], left_shoulder[1], left_hip[2]]
    mediapipe_left_hip_straight = [mediapipe_left_shoulder[0], mediapipe_left_hip[1], 0]
    ## Put the deep in the keypoints fo the image 
    mediapipe_left_shoulder.append(0)
    mediapipe_left_elbow.append(0)
    mediapipe_left_hip.append(0)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    ## OBS: In lidar the order is: Deep - Widht - Height
    ##      In image the order is: Widht - Height - Deep
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    
    ## Calculate the RA angle (to the left part of the body) using the points from the Lidar
    angle_ra_left = calc_angle(left_elbow, left_shoulder, left_hip_straight)

    ## Calculate the RA angle (to the left part of the body) using the points from keypoint detection
    mediapipe_angle_ra_left = calc_angle(mediapipe_left_elbow, mediapipe_left_shoulder, mediapipe_left_hip_straight)

    print(f"RA angle calculated by LiDar: {angle_ra_left}")
    print(f"RA angle calculated by MediaPipe: {mediapipe_angle_ra_left}")

bag.close()

## Print the points of the LiDar in the image
for count, value in enumerate(x):
    ## If the point belong to a specific part, prints with a different colot
    if count in points_left_shoulder:
        draw_point(frame, int(y[count]), int(z[count]), (0, 255, 0))
    elif count in points_left_elbow:
        draw_point(frame, int(y[count]), int(z[count]), (0, 0, 255))
    elif count in points_left_hip:
        draw_point(frame, int(y[count]), int(z[count]), (255, 0, 0))
    else:
        draw_point(frame, int(y[count]), int(z[count]), (255, 255, 255))

# Display the frame in the "Webcam" window
cv2.imshow("Image", frame)

# Release the VideoCapture object and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()