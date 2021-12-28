import cv2
import numpy as np 
import os 
import pandas as pd
import random
import time

import warnings
warnings.filterwarnings("ignore")

class feature_extract:
    
    def resize(self, image, size_x, size_y):
        image = cv2.resize(image, (size_x,size_y))
        
        return image


    def calc_hga(self, image):
        hga_list = []
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        
        grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
           
        hga_hist = cv2.calcHist([grad],[0],None,[256],[0,256])
          
        for i in range(hga_hist.shape[0]):
            hga_list.append(int(hga_hist[i,0]))
        
        return hga_list  


    def calc_HSV(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        
        features_flatten = []
        h_list = []
        s_list = []
        v_list = []
        
        h, s, v = cv2.split(image)
        
        hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
        hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
        hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
                
        for i in range(hist_h.shape[0]):
            h_list.append(int(hist_h[i,0]))
            s_list.append(int(hist_s[i,0]))
            v_list.append(int(hist_v[i,0]))
            
        features_flatten = np.hstack([h_list, s_list, v_list])

        
        return features_flatten
    
    
    def calc_roi(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        roi_x = [k for k in range(540, 650, 10)]
        roi_y = [j for j in range(790, 820 ,10)]

        roi_ort = []
        
        for i in range(len(roi_x)-1):
            for j in range(len(roi_y)-1):
                ort = int(img[roi_x[i]:roi_x[i+1], roi_y[j]:roi_y[j+1]].mean())
                roi_ort.append(ort)

                
        return roi_ort
    
    
    def list2csv(self, total_list, file_name):
        df = pd.DataFrame(total_list)
        df.to_csv('{}'.format(file_name), index=False, header=False)


    def shuffle_list(self, total_list):
        total_list = random.shuffle(total_list)





def run(dataset_path, size_x, size_y, filename):
    dataset_folder = os.listdir(dataset_path)
    total_list = []

    for j in range(len(dataset_folder)):
        image_dir = os.listdir(dataset_path + '/' + dataset_folder[j])

        for i in range(len(image_dir)):
            start = time.time()
            
            img = cv2.imread(dataset_path + '/' + dataset_folder[j] + '/' + '{}'.format(image_dir[i]))
            
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            if img != 'None':
                extractor = feature_extract()
                features_hga = extractor.calc_hga(img)
                features_hsv = extractor.calc_HSV(img)
                features_roi = extractor.calc_roi(img)
                
                if dataset_folder[j] == 'Sunny':
                    global_feature = np.hstack([features_roi, features_hga, features_hsv, 0])
                elif dataset_folder[j] == 'Cloudy':
                    global_feature = np.hstack([features_roi, features_hga, features_hsv, 1])
                elif dataset_folder[j] == 'Rainy':
                    global_feature = np.hstack([features_roi, features_hga, features_hsv, 2])
                
                total_list.append(global_feature)
                
                stop = time.time()
                
                print("Step {}: {}/{}".format(j, i, len(image_dir)), 
                      " image_name = ", image_dir[i], " label = ", 
                      dataset_folder[j], "time = {:.4f}".format((stop - start)), 
                      "len_hga = ", len(features_hga), 
                      " len_hsv = ", len(features_hsv), 
                      " len roi = ", len(features_roi))
        
    extractor.shuffle_list(total_list)
    extractor.list2csv(total_list, filename) 
    
    return total_list        




if __name__ == '__main__':
    run(dataset_path = os.path.expanduser('~') + '/Desktop//weather_classification/adastec_weather_cls/dataset/img', 
        size_x = 120, size_y = 80, filename = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/csv/dataset.csv')