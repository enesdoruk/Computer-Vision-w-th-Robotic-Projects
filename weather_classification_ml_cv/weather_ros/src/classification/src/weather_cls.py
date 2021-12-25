#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import sys
import cv2
import rospy    
import random
import numpy as np

import message_filters

from sensor_msgs.msg import Image

import os
import pickle
import argparse

import time

import torch
from torch import nn, optim

from process.feature_extractor import feature_extract
from process.visualization import plotting
from process.autoencoder_model import Autoencoder

import warnings
warnings.filterwarnings("ignore")


adastec_cls_pub = rospy.Publisher('/adas_cls_pub', Image, queue_size=10)



def test(img, size_x, size_y, suncloud_wght, sunrain_wght, cloudrain_wght, 
        sun_cloud_rain_wght, encoder_wght, in_shape, enc_shape):
    rainy_symbol = cv2.imread(os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/weather_symbols/rain.png')
    rainy_symbol = cv2.resize(rainy_symbol, (160,160))
    cloudy_symbol = cv2.imread(os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/weather_symbols/cloud.png')
    cloudy_symbol = cv2.resize(cloudy_symbol, (160,160))
    sunny_symbol = cv2.imread(os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/weather_symbols/sun.jpg')
    sunny_symbol = cv2.resize(sunny_symbol, (160,160))
    
    img_show = cv2.resize(img, (640,480))
    
    
    filename_cloudRain = cloudrain_wght
    model_cloudRain = pickle.load(open(filename_cloudRain, 'rb'))
    
    filename_sunRain = sunrain_wght
    model_sunRain = pickle.load(open(filename_sunRain, 'rb'))
    
    filename_sunCloud = suncloud_wght
    model_sunCloud = pickle.load(open(filename_sunCloud, 'rb'))
    
    filename_sunCloudRain = sun_cloud_rain_wght
    model_sunCloudRain = pickle.load(open(filename_sunCloudRain, 'rb'))
    
    
    features = feature_extract()
    img_rsz = features.resize(img, size_x, size_y)
    
    hga_feat = features.calc_hga(img_rsz)
    hga_feat_norm = hga_feat / max(hga_feat)
    
    hsv_feat = features.calc_HSV(img_rsz)
    hsv_feat_norm = hsv_feat / max(hsv_feat)
    
    features_gabor = features.calc_gabor(img_rsz)
    
    global_feature = np.hstack([hga_feat_norm, hsv_feat_norm])
    global_feature = global_feature.reshape(1, -1)
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    tensor_feature = torch.from_numpy(global_feature).to(device)
    
    
    encoder = Autoencoder(in_shape = in_shape, enc_shape = enc_shape).double().to(device)
    encoder.load_state_dict(torch.load(encoder_wght))
    encoder.eval()
    
    
    with torch.no_grad():
        loss = nn.MSELoss()
        
        encoded = encoder.encode(tensor_feature)
        decoded = encoder.decode(encoded)
        
        mse = loss(decoded, tensor_feature).item()
        
        enc = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()

    predict_sunCloudRain = model_sunCloudRain.predict(enc)


    predict_sunRain = model_sunRain.predict(enc)

    rain = 0 
    cloud = 0
    sun = 0

    weather = ''
    
    
    if predict_sunRain == -1:
        predict_sunCloud = model_sunCloud.predict(enc)
        
        if predict_sunCloud == 1:
            print('Class output = Cloud')
            weather = 'Cloudy'
            
            cloud += 1
            th_min, th_max = 240, 255
            b, g, r = 255, 229, 204
            orgx, orgy = 270, 180
            plotting(img_show, cloudy_symbol, weather, 
                     th_min, th_max, b, g, r, orgx, orgy,
                     rain, sun, cloud)

        elif predict_sunCloud == -1:
            print('Class output = Sun')
            weather = 'Sunny'
            
            sun +=1
            th_min, th_max = 230, 255
            b, g, r = 51, 153, 255
            orgx, orgy = 260, 180
            plotting(img_show, sunny_symbol, weather, 
                     th_min, th_max, b, g, r, orgx, orgy,
                     rain, sun, cloud)
            
        
    elif predict_sunRain == 1:
        predict_cloudRain = model_cloudRain.predict(enc)
        
        if predict_cloudRain == 1:
            print('Class output = Cloud')
            weather = 'Cloudy'
            
            cloud += 1
            th_min, th_max = 240, 255
            b, g, r = 255, 229, 204
            orgx, orgy = 270, 180
            plotting(img_show, cloudy_symbol, weather, 
                     th_min, th_max, b, g, r, orgx, orgy,
                     rain, sun, cloud)
            
        elif predict_cloudRain == -1:
            print('Class output = Rain')
            weather = 'Rainy'
            
            rain += 1
            th_min, th_max = 230, 255
            b, g, r = 255, 229, 204
            orgx, orgy = 270, 180
            plotting(img_show, rainy_symbol, weather, 
                     th_min, th_max, b, g, r, orgx, orgy,
                     rain, sun, cloud)
    
    img_show = cv2.resize(img_show, (980, 720))

    return img_show

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



def adastec_cls(dash_img):
    new_frame_time = time.time()

    image = np.frombuffer(dash_img.data, dtype=np.uint8).reshape(dash_img.height, dash_img.width, -1)

    size_x = int(rospy.get_param("~size_x"))
    size_y = int(rospy.get_param("~size_y"))
    suncloud_wght = rospy.get_param("~sc_wght")
    sunrain_wght = rospy.get_param("~sr_wght")
    cloudrain_wght = rospy.get_param("~cr_wght")
    sun_cloud_rain_wght = rospy.get_param("~scr_wght")
    encoder_wght = rospy.get_param("~encoder_wght")
    in_shape = int(rospy.get_param("~in_shape"))
    enc_shape = int(rospy.get_param("~enc_shape"))

    img_show = test(image, size_x, size_y, suncloud_wght, sunrain_wght,
                     cloudrain_wght, sun_cloud_rain_wght, encoder_wght, in_shape, enc_shape)
    

    fps = 1/(new_frame_time)
    
    cv2.putText(img_show, 'FPS: {}'.format(fps), (840, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,255,0), 1, cv2.LINE_AA)

    output_pub = cv2_to_imgmsg(img_show)
    output_pub.header.stamp = rospy.Time.now()
    adastec_cls_pub.publish(output_pub)

    


def main(args): 
    rospy.init_node('classification_node', anonymous=True)
    rospy.Subscriber("{}".format(rospy.get_param("~topic")), Image, adastec_cls)

    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main(sys.argv)
