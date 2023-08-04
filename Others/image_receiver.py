#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

def webcam_receiver(msg):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('webcam_receiver', anonymous=True)
    rospy.Subscriber('webcam_image', Image, webcam_receiver)

    # Espera um breve momento antes de criar a janela
    time.sleep(1)

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', 640, 480)
    rospy.spin()
    cv2.destroyAllWindows()
