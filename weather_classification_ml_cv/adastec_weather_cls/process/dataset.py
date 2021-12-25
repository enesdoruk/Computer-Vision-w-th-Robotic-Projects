import os 
import cv2
import time
import random
import argparse
import numpy as np 
import pandas as pd

from feature_extractor import feature_extract

class Dataset:
    def __init__(self, path, x_size, y_size, save_path):
        self.path = path
        self.x_size = x_size
        self.y_size = y_size
        self.save_path = save_path


    def cloud_rain_sun(self):
        dataset_folder = os.listdir(self.path)
        save = "weather_cls.csv"
        total_list = []
        start = time.time()

        for j in range(len(dataset_folder)):
            image_dir = os.listdir(self.path + '/' + dataset_folder[j])

            for i in range(len(image_dir)):
                image = cv2.imread(self.path + '/' + dataset_folder[j] + '/' + '{}'.format(image_dir[i]))

                if image != 'None':
                    extractor = feature_extract()
                    img = extractor.resize(image, self.x_size, self.y_size)
                    
                    features_hga = extractor.calc_hga(img)
                    features_hsv = extractor.calc_HSV(img)

                    if dataset_folder[j] == 'Sunny':
                        global_feature = np.hstack([features_hga, features_hsv, 0])
                    elif dataset_folder[j] == 'Cloudy':
                        global_feature = np.hstack([features_hga, features_hsv, 1])
                    elif dataset_folder[j] == 'Rainy':
                        global_feature = np.hstack([features_hga, features_hsv, 2])
                    
                    total_list.append(global_feature)

                    print("Step: {}: {}/{}".format(j, i, len(image_dir)), " image_name = ", image_dir[i], " label = ", dataset_folder[j], " len_hga = ", len(features_hga), " len_hsv = ", len(features_hsv))
            
        extractor.shuffle_list(total_list)
        extractor.list2csv(total_list, self.save_path + "/" + save) 

        stop = time.time()

        print("Create Dataset {} seconds".format(stop - start))

def run_dataset(path, x_size, y_size, save_path):
    dataset = Dataset(path, x_size, y_size, save_path)
    dataset.cloud_rain_sun()


def get_args():
    parser = argparse.ArgumentParser(description='Create Dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--path', type=str, 
                        default=os.path.expanduser('~') + "/Desktop/weather_classification/adastec_weather_cls/dataset/img",
                        help='dataset destination')
    parser.add_argument('-sx', '--size_x', type=int, default=80,
                        help='size x of image')
    parser.add_argument('-sy', '--size_y', type=int, default=60,
                        help='size y of image')
    parser.add_argument('-sp', '--save_path', type=str, 
                        default=os.path.expanduser('~') + "/Desktop/weather_classification/adastec_weather_cls/dataset/csv",
                        help='save path destination')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    run_dataset(args.path, args.size_x, args.size_y, args.save_path)