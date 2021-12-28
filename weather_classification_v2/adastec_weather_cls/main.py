#/usr/bin/python3

import os
import sys
import cv2
import pickle
import argparse
import numpy as np

import torch
from torch import nn, optim

from process.feature_extractor import feature_extract
from process.visualization import plotting
from process.autoencoder_model import Autoencoder

import warnings
warnings.filterwarnings("ignore")


def test(img, size_x, size_y, suncloud_wght, sunrain_wght, cloudrain_wght, sun_cloud_rain_wght):
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
    
    #filename_sunCloudRain = sun_cloud_rain_wght
    #model_sunCloudRain = pickle.load(open(filename_sunCloudRain, 'rb'))
    
    
    features = feature_extract()
    #img_rsz = features.resize(img, size_x, size_y)
    
    hga_feat = features.calc_hga(img)
    #hga_feat_norm = hga_feat / max(hga_feat)
    hsv_feat = features.calc_HSV(img)
    #hsv_feat_norm = hsv_feat / max(hsv_feat)
    #gabor_feat = features.calc_gabor(img)
    roi_feat = features.calc_roi(img)

    global_feature = np.hstack([roi_feat, hga_feat, hsv_feat])
    global_feature = global_feature.reshape(1,-1)
    
    '''
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    tensor_feature = torch.from_numpy(global_feature).to(device)
    
    encoder = Autoencoder(in_shape = 48000, enc_shape = 3000).double().to(device)
    encoder.load_state_dict(torch.load(os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/autoencoder_best.pth'))
    encoder.eval()
    
    
    with torch.no_grad():
        loss = nn.MSELoss()
        
        encoded = encoder.encode(tensor_feature)
        decoded = encoder.decode(encoded)
        
        mse = loss(decoded, tensor_feature).item()
        
        enc = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()
    '''
    
    #predict_sunCloudRain = model_sunCloudRain.predict(global_feature)
    #print("sun_cloud_rain output = ", predict_sunCloudRain)
    
    predict_sunRain = model_sunRain.predict(global_feature)

    
    weather = ''
    rain = 0 
    cloud = 0
    sun = 0
    
    if predict_sunRain == -1:
        predict_sunCloud = model_sunCloud.predict(global_feature)
        
        if predict_sunCloud == 1:
            print('Class output = Cloud')
            print('from suncloud')
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
        predict_cloudRain = model_cloudRain.predict(global_feature)
        
        if predict_cloudRain == 1:
            print('Class output = Cloud')
            print('from cloudrain')
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
            plotting(img_show, sunny_symbol, weather, 
                     th_min, th_max, b, g, r, orgx, orgy,
                     rain, sun, cloud)
            
   
    return img_show

def get_args():
    parser = argparse.ArgumentParser(description='Adverse Weather Classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sx', '--size_x', type=int, default=120,
                        help='size x of image')
    parser.add_argument('-sy', '--size_y', type=int, default=80,
                        help='size y of image')
    parser.add_argument('-sr', '--sunrain', type=str, default = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/sun_rain_wghts_adabostDT.sav',
                        help='sun rain weight file')
    parser.add_argument('-sc', '--suncloud', type=str, default = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/sun_cloud_wghts_adabostDT.sav',
                        help='sun cloud weight file')
    parser.add_argument('-cr', '--cloudrain', type=str, 
                        default = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/cloud_rain_wghts_adabostDT.sav',
                        help='cloud rain weight file')
    parser.add_argument('-scr', '--suncloudrain', type=str, 
                        default = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/sun_cloud__rain_wghts_adabostDT.sav',
                        help='cloud rain weight file')

    return parser.parse_args()
    


if __name__ == '__main__':
    args = get_args()
    
    img = cv2.imread(os.path.expanduser('~') + "/Desktop/weather_classification/adastec_weather_cls/dataset/img/Sunny/69798.jpg")
    
    if img is not None:
        result = test(img, args.size_x, args.size_y,
             args.suncloud, args.sunrain, args.cloudrain, args.suncloudrain)
        
        #cv2.imshow("ADASTEC Adverse Weather Classification", result)
        #cv2.waitKey(5000)
        #cv2.destroyAllWindows()
        