#!/usr/bin/env python3

import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os



from cvlib.object_detection import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import matplotlib.pyplot as plt
import math


# Função para desenhar pontos específicos em uma imagem
def desenhar_ponto(img, x, y):
    # Definir a cor do ponto (verde no exemplo)
    cor = (0, 255, 0)
    # Definir o tamanho do ponto
    tamanho = 2
    # Desenhar o ponto na imagem
    cv2.circle(img, (x, y), tamanho, cor, -1)


## Define the image
image_path = "image301_teste.jpg"
## Read the image
frame = cv2.imread(image_path)
## Get the sizes of the image
width_image, height_image = frame.shape[:2]
print("SIZE OF THE IMAGE")
print(width_image)
print(height_image )
## Rotate the image to identify the person
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov4-tiny')

## Define the initial point of the person (uper left)
initial_point = (bbox[0][0], bbox[0][1])
final_point = (bbox[0][2], bbox[0][3])

## Define the size of the person
width_person = final_point[0] - initial_point[0]
height_person  = final_point[1] - initial_point[1]

## Draw the box around the person
frame = draw_bbox(frame, bbox, label, conf)
## Rotate the image to the original
#frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

## Do the rotation of the upper left point
rotate_initial_point = initial_point#(initial_point[1], width_image - initial_point[0])


print(f"Initial Point: {rotate_initial_point}")
print(f"Height: {height_person}")
print(f"Widht: {width_person}")


#size_person_image = (width_person, height_person)


############################################################################################################################################
## Variables to be used from image
## rotate_initial_point: Left upper point
## size_person_image: size of the person based on the image (widht and height)
################

## Since the image is inverted, invert the sizes
#size_to_image = (height_person, width_person)
#print("SIZE OF THE PERSON IN IMAGE")
#print(size_person_image)

## 0: X | 1: Y | Z: 2
## Z é a altura
## X é a largura
## Y é a profundidade

## Read the file
bag = rosbag.Bag( os.path.join('Data_inverted', 'lidar_data', 'point_cloud298.bag'))

points = []

for topic, msg, t in bag.read_messages(topics=['/rslidar_points']):
    point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(point_cloud))

    # Create the 3D Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    height_lidar = max(points[:, 0]) - min(points[:, 0])
    widht_lidar = max(points[:, 2]) - min(points[:, 2])
    print("SIZE OF THE LIDARRRR")
    print(height_lidar)
    print(widht_lidar)

    ## Calculate the ratios 
    ratio_height = 300#height_person / height_lidar
    ratio_widht = 300#width_person / widht_lidar
   


    ## Ratio to put the points in the same size
    #############################
    #x_size = max(points[:, 0]) - min(points[:, 0]) ## Widht of the person
    #z_size = max(points[:, 2]) - min(points[:, 2]) ## Height of the person
    #print("SIZE OF THE LIDARRRR")
    #print(x_size)
    #print(z_size)
    #ratio_widht = size_person_image[0] / x_size
    #ratio_height = size_person_image[1] / z_size
    #print("RATIOS")
    #print(ratio_widht)
    #print(ratio_height)
    ##############################


    ## RESIZE THE POINTS ON THE LIDAR

    points_height = []
    points_deep = []
    points_widht = []

    points_height_original = points[:, 0]
    points_deep_original = points[:, 1]
    points_widht_original = points[:, 2]

    for cont, value in enumerate(points_height_original):
        points_height.append( points_height_original[cont] * ratio_height)
        points_deep.append( points_deep_original[cont] * 1)
        points_widht.append( points_widht_original[cont] * ratio_widht)

    height_size_new = max(points_height) - min(points_height) ## Widht of the person
    widht_size_new = max(points_widht) - min(points_widht) ## Height of the person
    
    print("NEW SIZES OF THE LIDAR")
    print(height_size_new)
    print(widht_size_new)

    ## Find the left down point
    main_point_lidar = ( min(points_widht), min(points_height))
    
    print("Main Point of Image")
    print(rotate_initial_point)
    print("Main Point of LiDar")
    print(main_point_lidar)

    #######
    translation_height = rotate_initial_point[1] - main_point_lidar[1]
    translation_widht = rotate_initial_point[0] - main_point_lidar[0]
    print("OFF SETS")
    print(f"X: {translation_height}")
    print(f"Z: {translation_widht}")
    new_main_point_lidar = (int(main_point_lidar[0] +  translation_widht), int(main_point_lidar[1] + translation_height))

    ## Do the translation for all the points

    new_points_height = []
    for i in points_height:
        new_points_height.append( int(i + translation_height) )


    new_points_widht = []
    for i in points_widht:
        new_points_widht.append(int(i + translation_widht))

    # Plota a nuvem de pontos
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    
    # Configurações adicionais do gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nuvem de Pontos 3D')

    # Exibe o gráfico
    plt.show()

bag.close()

#print(f"Size: {x_size}, {z_size}")


print(new_main_point_lidar)
desenhar_ponto(frame, new_main_point_lidar[0], new_main_point_lidar[1])


for count, value in enumerate(new_points_widht):
    print(count)
    
    print(f"X: {new_points_widht[count]}")
    print(f"Y: {new_points_height[count]}")
    desenhar_ponto(frame, new_points_widht[count], new_points_height[count])

#####################
### Just to show the point
# Definir o raio do círculo
radius = 25

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







