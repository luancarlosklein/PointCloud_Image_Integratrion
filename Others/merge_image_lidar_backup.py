#!/usr/bin/env python3

import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy

from cvlib.object_detection import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import matplotlib.pyplot as plt
import math

import keypoint_detection


def calc_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return round(np.degrees(angle))

def generate_points_lidar(image_path, x, y, z, points, joint_name):    
    joint = keypoint_detection.detect(image_path, joint_name)
    uni_points = []

    for cont, value in enumerate(y):
        uni_points.append( (y[cont], z[cont]))

    near_points = []
    near_points_index = []
    ratio = 1
    while len(near_points) < 5:
        new_points, new_points_index = pontos_dentro_do_raio(joint, uni_points, ratio)
        ratio += 1

        for i in new_points:
            if i not in near_points:
                near_points.append(i)

        for i in new_points_index:
            if i not in near_points_index:
                near_points_index.append(i)
        print(ratio)
        if ratio > 20:
            break

    ## Calculate the average position
    x_original = points[:, 0]## Deep
    y_original = points[:, 1]## Widht
    z_original = points[:, 2]## Height

    avg_lidar = [0, 0, 0]
    for i in near_points_index:
        avg_lidar[0] += x_original[i]
        avg_lidar[1] += y_original[i]
        avg_lidar[2] += z_original[i]
    
    avg_lidar[0] /= len(near_points_index)
    avg_lidar[1] /= len(near_points_index)
    avg_lidar[2] /= len(near_points_index)

    return(avg_lidar, near_points_index)

def distancia_pontos(ponto1, ponto2):
    """Calcula a distância entre dois pontos 2D."""
    return math.sqrt((ponto2[0] - ponto1[0]) ** 2 + (ponto2[1] - ponto1[1]) ** 2)

def pontos_dentro_do_raio(ponto, lista_pontos, raio):
    """Retorna os pontos dentro do raio dado de um ponto dado em uma lista de pontos."""
    pontos_proximos = []
    pontos_proximos_index = []
    for count, outro_ponto in enumerate(lista_pontos):
        distancia = distancia_pontos(ponto, outro_ponto)
        if distancia <= raio:
            pontos_proximos.append(outro_ponto)
            pontos_proximos_index.append(count)
    return (pontos_proximos, pontos_proximos_index)



# Função para desenhar pontos específicos em uma imagem
def desenhar_ponto(img, x, y, cor):
    # Definir o tamanho do ponto
    tamanho = 3
    # Desenhar o ponto na imagem
    cv2.circle(img, (x, y), tamanho, cor, -1)


## Define the image
image_path = os.path.join("Data_normal", "camera_data", "image200.jpg")
## Read the image
frame = cv2.imread(image_path)
## Get the sizes of the image
width_image, height_image = frame.shape[:2]
print("SIZE OF THE IMAGE")
print(width_image)
print(height_image )
## Rotate the image to identify the person
bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov4-tiny')

## Define the initial point of the person (uper left)
key_point_image = (bbox[0][0], bbox[0][1])
final_point = (bbox[0][2], bbox[0][3])

## Define the size of the person
width_person = final_point[0] - key_point_image[0]
height_person  = final_point[1] - key_point_image[1]

## Draw the box around the person
frame = draw_bbox(frame, bbox, label, conf)
## Rotate the image to the original
#frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

## Do the rotation of the upper left point
#rotate_initial_point = initial_point#(initial_point[1], width_image - initial_point[0])

print(f"Initial Point: {key_point_image}")
print(f"Height: {height_person}")
print(f"Widht: {width_person}")


#############################################################################

# # Função para desenhar pontos específicos em uma imagem
# def desenhar_ponto(img, x, y):
#     # Definir a cor do ponto (verde no exemplo)
#     cor = (0, 255, 0)
#     # Definir o tamanho do ponto
#     tamanho = 4
#     # Desenhar o ponto na imagem
#     cv2.circle(img, (x, y), tamanho, cor, -1)



## Read the file
bag = rosbag.Bag( os.path.join('Data_normal', 'lidar_data', 'point_cloud198.bag'))

points = []

for topic, msg, t in bag.read_messages(topics=['/rslidar_points']):
    point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(point_cloud))

    # Create the 3D Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    deep_lidar = max(points[:, 0]) - min(points[:, 0])
    widht_lidar = max(points[:, 1]) - min(points[:, 1])
    height_lidar = max(points[:, 2]) - min(points[:, 2])
    
    
    print("SIZE OF THE LIDARRRR")
    print(height_lidar)
    print(widht_lidar)
    print(deep_lidar)

    ## Calculate the ratios 
    ratio_x = 1
    ratio_y = width_person / widht_lidar
    ratio_z = height_person / height_lidar
    

    points_height = []
    points_deep = []
    points_widht = []

    x = points[:, 0] * ratio_x## Deep
    y = points[:, 1] * ratio_y## Widht
    z = points[:, 2] * ratio_z## Height
    
    min_x = min(x)
    min_y = min(y)
    min_z = min(z)

    if min_x < 0:
        x = x + abs(min_x)
    
    if min_y < 0:
        y = y + abs(min_y)

    if min_z < 0:
        z = z + abs(min_z)
    

    ## Precisa converter a medida do lidar para ficar adequada ao eixo da image (com o 0,0 no canto esquerdo superior)

    max_y = max(y)
    max_z = max(z)

    y = (y*(-1)) + max_y
    z = (z*(-1)) + max_z

     
    ####################
    key_point_lidar = (min(y), min(z))

    desl_y = key_point_image[0] - key_point_lidar[0]
    desl_z = key_point_image[1] - key_point_lidar[1]

    y = y + desl_y
    z = z + desl_z

    # # Plota a nuvem de pontos
    if len(points) > 0:
         ax.scatter(x, y, z)

    #######################################

    left_shoulder, points_left_shoulder = generate_points_lidar(image_path, x, y, z, points, "LEFT_SHOULDER")
    left_elbow, points_left_elbow = generate_points_lidar(image_path, x, y, z, points, "LEFT_ELBOW")
    left_hip, points_left_hip = generate_points_lidar(image_path, x, y, z, points, "LEFT_HIP")

#  ###    "shoulder_left_rotation_ra" : (left_elbow, left_shoulder, left_hip_straight),
    print("LEFT SHOULDER")
    print(left_shoulder)

    print("LEFT ELBOW")
    print(left_elbow)

    print("LEFT HIP")
    print(left_hip)


    #x - Deep
    #y - Widht
    #z - Height

    left_hip_straight = [left_hip[0], left_shoulder[1], left_hip[2]]

    angle_ra_left = calc_angle(left_elbow, left_shoulder, left_hip_straight)

    print("RA ANGLE")
    print(angle_ra_left)
    # import keypoint_detection
    # left_shoulder = keypoint_detection.detect(image_path, "LEFT_SHOULDER")
    # print("LEFT SHOULDER")
    # print(left_shoulder)

    # uni_points = []

    # for cont, value in enumerate(y):
    #     uni_points.append( (y[cont], z[cont]))

    # near_points = []
    # near_points_index = []
    # ratio = 1
    # while len(near_points) < 5:
    #     new_points, new_points_index = pontos_dentro_do_raio(left_shoulder, uni_points, ratio)
    #     ratio += 1

    #     for i in new_points:
    #         if i not in near_points:
    #             near_points.append(i)

    #     for i in new_points_index:
    #         if i not in near_points_index:
    #             near_points_index.append(i)
    #     print(ratio)
    #     if ratio > 20:
    #         break
    
    # print("NEAR POINTS")
    # print(near_points)
    # print(near_points_index)

    # ## Calculate the average position

    # x_original = points[:, 0]## Deep
    # y_original = points[:, 1]## Widht
    # z_original = points[:, 2]## Height

    # left_shoulder_avg_lidar = [0, 0, 0]
    # for i in near_points_index:
    #     left_shoulder_avg_lidar[0] += x_original[i]
    #     left_shoulder_avg_lidar[1] += y_original[i]
    #     left_shoulder_avg_lidar[2] += z_original[i]
    
    # left_shoulder_avg_lidar[0] /= len(near_points_index)
    # left_shoulder_avg_lidar[1] /= len(near_points_index)
    # left_shoulder_avg_lidar[2] /= len(near_points_index)

    # print("AVERAGE LIDAR")
    # print(left_shoulder_avg_lidar)
        


  


    #######################################




    # # Configurações adicionais do gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nuvem de Pontos 3D')

    # # Exibe o gráfico
    plt.show()

bag.close()

#print(f"Size: {x_size}, {z_size}")
#print(new_main_point_lidar)
#desenhar_ponto(image, new_main_point_lidar[0], new_main_point_lidar[1])


for count, value in enumerate(x):
    #if count in near_points_index:
    if count in points_left_shoulder:
        desenhar_ponto(frame, int(y[count]), int(z[count]), (0, 255, 0))
    elif count in points_left_elbow:
        desenhar_ponto(frame, int(y[count]), int(z[count]), (0, 0, 255))
    elif count in points_left_hip:
        desenhar_ponto(frame, int(y[count]), int(z[count]), (255, 0, 0))
    else:
        desenhar_ponto(frame, int(y[count]), int(z[count]), (255, 255, 255))
#####################
### Just to show the point
# Definir o raio do círculo
radius = 25

# Definir a cor do círculo (BGR)
color = (0, 0, 255)  # Vermelho: (B, G, R)



###############################################

# Display the frame in the "Webcam" window
cv2.imshow("Webcam", frame)

# Release the VideoCapture object and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()







