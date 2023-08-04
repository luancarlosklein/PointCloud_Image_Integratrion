#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def webcam_publisher():
    rospy.init_node('webcam_publisher', anonymous=True)
    image_pub = rospy.Publisher('webcam_image', Image, queue_size=10)
    rate = rospy.Rate(10)  # Taxa de publicação (10 Hz)

    # Inicializa o objeto CvBridge
    bridge = CvBridge()

    # Inicializa a captura de vídeo da webcam
    ## RealSense Cam, the code is 6
    cap = cv2.VideoCapture(4)

    while not rospy.is_shutdown():
        # Captura um frame da webcam
        ret, frame = cap.read()

        if ret:
            # Converte a imagem do formato BGR para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Cria a mensagem Image a partir do frame
            msg = bridge.cv2_to_imgmsg(frame_rgb, "rgb8")

            # Publica a mensagem no tópico "webcam_image"
            image_pub.publish(msg)

        rate.sleep()

    # Libera os recursos
    cap.release()

if __name__ == '__main__':
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        pass
