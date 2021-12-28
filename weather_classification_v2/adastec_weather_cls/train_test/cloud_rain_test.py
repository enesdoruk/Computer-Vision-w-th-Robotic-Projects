import os
import sys
import cv2
import pickle
import pandas as pd
import numpy as np

import torch 
from torch import nn, optim

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

sys.path.append(os.path.expanduser('~') +  '/Desktop/weather_classification/adastec_weather_cls')

from process.feature_extractor import feature_extract
from process.autoencoder_model import Autoencoder


import warnings
warnings.filterwarnings('ignore')

def cloud_rain_test():
    categories = {1: 'cloud',  -1: 'rainy'}
    
    img = cv2.imread(os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/img/Rainy/88430.jpg')
                    
    filename = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/cloud_rain_wghts_adabostDT.sav'
    model = pickle.load(open(filename, 'rb'))
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
    #encoder = Autoencoder(in_shape= 48000, enc_shape= 3000).double().to(device)
    #encoder.load_state_dict(torch.load(os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/autoencoder_best.pth'))
    #encoder.eval()
    
    features = feature_extract()
    #img_rsz = features.resize(img, 120, 80)
    
    
    hga_feat = features.calc_hga(img)
    #hga_feat_norm = hga_feat / max(hga_feat)
    hsv_feat = features.calc_HSV(img)
    #hsv_feat_norm = hsv_feat / max(hsv_feat)
    #gabor_feat = features.calc_gabor(img)
    roi_feat = features.calc_roi(img)
    
    global_feature = np.hstack([roi_feat, hga_feat, hsv_feat])
    global_feature = global_feature.reshape(1, -1)
        
    ''' 
    tensor_feature = torch.from_numpy(global_feature).to(device)
        
    with torch.no_grad():
        loss = nn.MSELoss()
            
        encoded = encoder.encode(tensor_feature)
        decoded = encoder.decode(encoded)
            
        mse = loss(decoded, tensor_feature).item()
            
        enc = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()
    '''
    
    predict = model.predict(global_feature)
    print(predict)
    
    
if __name__ == '__main__':
    cloud_rain_test()