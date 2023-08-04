#!/usr/bin/env python3

# # import rospy
# # from sensor_msgs.msg import PointCloud2

# # import pcl #sudo apt install python3-pcl
# # import sensor_msgs.point_cloud2 as pc2
# # #import numpy as np
# # #import ros_numpy #pip3 install rosnumpy

# # from pyntcloud import PyntCloud
# # import pandas as pd

# # import numpy as np
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots

# # # Cria a figura e os subplots
# # fig = make_subplots(rows=1, cols=1)
# # scatter = go.Scatter3d()


# # def lidar_callback(data):
# #     # Função de retorno chamada quando uma nova mensagem do LiDAR é recebida
# #     # Aqui você pode processar os dados do LiDAR conforme necessário
# #     # Neste exemplo, estamos apenas imprimindo as leituras mínima e máxima do LiDAR
# #     #print("min_range")
# #     #min_range = min(data.ranges)
# #     #max_range = max(data.ranges)
# #     #rospy.loginfo("Leitura mínima do LiDAR: %f", min_range)
# #     #rospy.loginfo("Leitura máxima do LiDAR: %f", max_range)

# #     # Exemplo de impressão das informações do PointCloud2
# #     #rospy.loginfo("Header: %s", data.header)
# #     #rospy.loginfo("Número de pontos: %s", len(data.data))
# #     #rospy.loginfo("Campos: %s", data.fields)

# #     #rospy.loginfo("Leitura mínima do LiDAR")


# #     #pc = ros_numpy.numpify(data)
# #     #points=np.zeros((pc.shape[0],3))
# #     #points[:,0]=pc['x']
# #     #points[:,1]=pc['y']
# #     #points[:,2]=pc['z']


# #     print("received")
    
# #     global scatter

# #     # Extrai os campos relevantes da mensagem PointCloud2
# #     point_cloud = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)

# #     # Cria um DataFrame a partir dos pontos
# #     point_cloud_df = pd.DataFrame(point_cloud, columns=["x", "y", "z"])

# #     # Atualiza os dados do scatter
# #     scatter.x = point_cloud_df["x"]
# #     scatter.y = point_cloud_df["y"]
# #     scatter.z = point_cloud_df["z"]

# #     # Atualiza o layout do gráfico
# #     layout = go.Layout(scene=dict(aspectmode="data"))

# #     # Atualiza a figura
# #     fig.data = [scatter]
# #     fig.layout = layout

# #     # Atualiza a visualização
# #     fig.update_layout(scene=dict(aspectmode="data"))
# #     fig.show(renderer="notebook")

# #     ##### CONVERTE PARA PCL
# #     #pc = ros_numpy.numpify(data)
# #     #height = pc.shape[0]
# #     #width = pc.shape[1]
# #     #np_points = np.zeros((height * width, 3), dtype=np.float32)
# #     #np_points[:, 0] = np.resize(pc['x'], height * width)
# #     #np_points[:, 1] = np.resize(pc['y'], height * width)
# #     #np_points[:, 2] = np.resize(pc['z'], height * width)
# #     #p = pcl.PointCloud(np.array(np_points, dtype=np.float32))
# #     #rospy.loginfo(p)
    
# # rospy.init_node('lidar_subscriber', anonymous=True)
# # rospy.Subscriber('/rslidar_points', PointCloud2, lidar_callback)


# # # Mantém o nó em execução até que seja encerrado
# # rospy.spin()


# ################### CODIGO 2
# import rospy
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# from cv_bridge import CvBridge
# import cv2
# import time
# from sensor_msgs.msg import Image

# # Cria uma figura 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Lista vazia para armazenar os pontos
# points = []

# ############ DEFINE THE CALLBACKS ####################
# def callback_lidar_data(msg):
#     global points

#     # Extrai os campos relevantes da mensagem PointCloud2
#     point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

#     # Converte os pontos para um array numpy
#     points = np.array(list(point_cloud))


# def callback_webcam_receiver(msg):
#     bridge = CvBridge()
#     frame = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
#     cv2.imshow('Webcam', frame)
#     cv2.waitKey(1)

# #########################################################

# def update_plot(frame):
#     global points

#     # Limpa o gráfico
#     ax.cla()

#     # Plota a nuvem de pontos atualizada
#     if len(points) > 0:
#         ax.scatter(points[:, 0], points[:, 1], points[:, 2])

#     # Configurações adicionais do gráfico
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Nuvem de Pontos 3D')

# if __name__ == '__main__':

#     ## Subscribe in the LiDar topic
#     rospy.init_node('point_cloud_viewer', anonymous=True)
#     rospy.Subscriber('/rslidar_points', PointCloud2, callback_lidar_data)

#     # Update the LidarGraph
#     ani = FuncAnimation(fig, update_plot, interval=100)

#     # Exibe o gráfico
#     plt.show()

#     ########################################

#     rospy.init_node('webcam_receiver', anonymous=True)
#     rospy.Subscriber('webcam_image', Image, callback_webcam_receiver)

#     # Espera um breve momento antes de criar a janela
#     time.sleep(1)

#     cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Webcam', 640, 480)
#     rospy.spin()
#     cv2.destroyAllWindows()

############ CÓDIGO 3


# import rospy
# import rosbag
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# # Cria uma figura 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Lista vazia para armazenar os pontos
# points = []
# cont = 0
# def callback(msg):
#     global points, cont


#     bag = rosbag.Bag(f'point_cloud{cont}.bag', 'w')
#     bag.write('/rslidar_points', msg)
#     bag.close()
#     cont += 1

#     # Extrai os campos relevantes da mensagem PointCloud2
#     point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

#     # Converte os pontos para um array numpy
#     points = np.array(list(point_cloud))

# def update_plot(frame):
#     global points

#     # Limpa o gráfico
#     ax.cla()

#     # Plota a nuvem de pontos atualizada
#     if len(points) > 0:
#         ax.scatter(points[:, 0], points[:, 1], points[:, 2])

#     # Configurações adicionais do gráfico
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Nuvem de Pontos 3D')

# if __name__ == '__main__':
#     # rospy.init_node('point_cloud_viewer', anonymous=True)
#     # rospy.Subscriber('/rslidar_points', PointCloud2, callback)

#     # # Atualiza o gráfico continuamente
#     # ani = FuncAnimation(fig, update_plot, interval=100)

#     # # Exibe o gráfico
#     # plt.show()


import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bag = rosbag.Bag('point_cloud145.bag')

# Lista vazia para armazenar os pontos
points = []

for topic, msg, t in bag.read_messages(topics=['/rslidar_points']):
    point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(point_cloud))

    # Cria uma figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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