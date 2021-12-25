import cv2
import numpy as np 
import os 
import pandas as pd
import random

class feature_extract:
    
    def resize(self, image, size_x, size_y):
        image = cv2.resize(image, (size_x,size_y))
        
        return image


    def calc_hga(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

        edges_x = cv2.filter2D(image,cv2.CV_8U,kernelx)
        edges_y = cv2.filter2D(image,cv2.CV_8U,kernely)

            
        features = np.zeros([edges_x.shape[0], edges_x.shape[1]])
        feature_list = []
        features_flatten = []
        
        for j in range(edges_x.shape[0]):
            for k in range(edges_x.shape[1]):
                features[j,k] = int(np.sqrt(np.power(edges_x[j,k], 2) + np.power(edges_y[j,k], 2)))
                features_flatten.append(features[j,k])
        
        #features_flatten = features_flatten / max(features_flatten)
        
        return features_flatten


    def calc_HSV(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
        image_hsv = np.zeros([image.shape[0], image.shape[1]])
        
        features_flatten = []
        mean_hsv = []

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image_hsv[i,j] = (image[i,j,0] + image[i,j,1] + image[i,j,2]) // 3
                mean_hsv.append(image_hsv[i,j])
        
        h, s, v = cv2.split(image)
        
        features_h = []
        features_s = []
        features_v = []
        
        
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                features_s.append(s[j,k])
                features_h.append(h[j,k])
                features_v.append(v[j,k])
        
        
        #features_h = features_h / max(features_h)
        #features_s = features_s / max(features_s)
        #features_v = features_v / max(features_v)
        
        features_flatten = np.hstack([features_s, features_v, features_h, mean_hsv])


        return features_flatten
    
    
    def calc_gabor(self, image):
        features_flatten = []
        
        g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        
        filtered_img = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)
        
        h, w = g_kernel.shape[:2]
        
        g_kernel = cv2.resize(filtered_img, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
        
        r, g, b = cv2.split(g_kernel)
        
        features_flatten = np.hstack([r, g, b])
        
        features_flatten = features_flatten.flatten()
        
        
        return features_flatten 
    
    
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
            image = cv2.imread(dataset_path + '/' + dataset_folder[j] + '/' + '{}'.format(image_dir[i]))

            if image != 'None':
                extractor = feature_extract()
                img = extractor.resize(image, size_x, size_y)
                features_hga = extractor.calc_hga(img)                
                features_hsv = extractor.calc_HSV(img)

                if dataset_folder[j] == 'Sunny':
                    global_feature = np.hstack([features_hga, features_hsv, 0])
                elif dataset_folder[j] == 'Cloudy':
                    global_feature = np.hstack([features_hga, features_hsv, 1])
                elif dataset_folder[j] == 'Rainy':
                    global_feature = np.hstack([features_hga, features_hsv, 2])
                
                total_list.append(global_feature)

                print("Step: {}:{}/{}".format(j, i, len(image_dir)), " image_name = ", image_dir[i], " label = ", dataset_folder[j], " len_hga = ", len(features_hga), " len_hsv = ", len(features_hsv))
        
    extractor.shuffle_list(total_list)
    extractor.list2csv(total_list, filename) 
    
    return total_list        


if __name__ == '__main__':
    run(dataset_path = os.path.expanduser('~') + '/Desktop//weather_classification/adastec_weather_cls/dataset/img', 
        size_x = 120, size_y = 80, filename = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/csv/dataset.csv')