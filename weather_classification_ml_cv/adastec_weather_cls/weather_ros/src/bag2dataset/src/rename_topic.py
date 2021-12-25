#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import sys
import cv2
import rospy    
import random
import numpy as np

from sensor_msgs.msg import Image

rename_topic_pub = rospy.Publisher('/dash_rain/image_rect', Image, queue_size=10)

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding")
    dtype = np.dtype("uint8") 
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data) 
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()

    return image_opencv


def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height 

    return img_msg

def create_video(ros_goruntu):
    img_bgr = np.frombuffer(ros_goruntu.data, dtype=np.uint8).reshape(ros_goruntu.height, ros_goruntu.width, -1)

    height,width,layers = img_bgr.shape

    img_sgmnt = cv2_to_imgmsg(img_bgr)
    img_sgmnt.header.stamp = rospy.Time.now()
    rename_topic_pub.publish(img_sgmnt)


def main(args): 
    rospy.init_node('rename_topic_node', anonymous=True)
    rospy.Subscriber("/dash_center/image_rect", Image, create_video)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main(sys.argv)
