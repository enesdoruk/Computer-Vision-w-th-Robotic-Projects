import os
import sys
import cv2
import time
import pickle
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.expanduser('~') +  '/Desktop/weather_classification/adastec_weather_cls')

from sklearn.svm import SVC  
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from process.feature_extractor import feature_extract
from process.preprocessing import data_preprocess
from process.perform_metrics import performance_metrics
from process.model import Adaboost

import warnings
warnings.filterwarnings("ignore")

def train(dataset_path, size_x, size_y, preprocess, test_size,
          save_data_path, save_model):
    
    dataset_folder = os.listdir(dataset_path)
    
    #total_list = []
    
    categories = {'sun': 0,  'rainy': 2}
    
    '''
    #create Dataset
    print('='*50)
    
    for j in range(len(dataset_folder)):
        image_dir = os.listdir(dataset_path + '/' + dataset_folder[j])
        
        
        for i in range(len(image_dir)):
            image = cv2.imread(dataset_path + '/' + dataset_folder[j] + '/' + '{}'.format(image_dir[i]))
    
            if image != 'None':
                image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    
                extractor = feature_extract()
                img = extractor.resize(image, size_x, size_y)
                
                features_hga = extractor.calc_hga(img)
                features_hsv = extractor.calc_HSV(img)                
                features_gabor = extractor.calc_gabor(img)

                
                if dataset_folder[j] == 'Sunny':
                    global_feature = np.hstack([features_hga, features_hsv, 0])
                elif dataset_folder[j] == 'Cloudy':
                    global_feature = np.hstack([features_hga, features_hsv, 1])
                elif dataset_folder[j] == 'Rainy':
                    global_feature = np.hstack([features_hga, features_hsv, 2])
                    
                total_list.append(global_feature)
    
                print("Step: {}:{}/{}".format(j, i, len(image_dir)), " image_name = ", image_dir[i], " label = ", dataset_folder[j], " len_hga = ", len(features_hga), " len_hsv = ", len(features_hsv))
                
    print('='*50)
    '''
    
    filename = os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/csv/dataset.csv'
    total_list = pd.read_csv(filename)
    total_list = pd.DataFrame(total_list)
    
    #Process Dataset
    #extractor.shuffle_list(total_list)
    #total_list = pd.DataFrame(total_list)
    
    if preprocess == True: 
        dataset = data_preprocess(total_list)
        dataset_pre = dataset.preprocess(total_list)
        
        sun_rain = dataset.sun_rain_dataset(dataset_pre)
        #dataset.list2csv(cloud_rain, file_name= '/home/enesdrk/Desktop/xx.csv')
    
    if preprocess == False:
        dataset = data_preprocess(total_list)
        sun_rain = dataset.sun_rain_dataset(total_list)
        
    
    x = sun_rain.iloc[:,:-1]
    y = sun_rain.iloc[:,-1]
    
    #Labels should be {-1, 1} for adaboost 
    for i in range(len(y)):
        if y.iloc[i] == 0:
            y.iloc[i] = -1
        elif y.iloc[i] == 2:
            y.iloc[i] = 1
    
    
    x = x.to_numpy()
    y = y.to_numpy()
    
    
    #Split Dataset for training and testing
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 101, test_size= test_size)
    
    
    #Define DT and RAdaboost and training
    N = len(train_y)
    w = np.array([1/N for i in range(N)])
    
    
    dt_stump = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, 
                                      criterion='entropy',
                                      splitter='best', min_samples_split= 5,
                                      max_features= 'log2',
                                      class_weight= 'balanced')
    
    dt_stump.fit(train_x, train_y, sample_weight=w)
    
    param_grid_dt = {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 75, 100]}
    
    ada_dt_start = time.time()
    
    ada_dt = AdaBoostClassifier(base_estimator=dt_stump, algorithm="SAMME.R")
    ada_dt_clf = GridSearchCV(ada_dt, param_grid_dt, scoring='f1', verbose=1)
    ada_dt_clf.fit(train_x, train_y)
    
    y_predict_ada_dt = ada_dt_clf.predict(test_x)
    
    ada_dt_stop = time.time()
    
    print("True label = ", test_y)
    print("RAdaboost label = ", y_predict_ada_dt)
    print("RAdaboost train time = ", ada_dt_stop - ada_dt_start)
    
    
    '''
    #Real Adaboost classifier using python scratch 
    adaboost_scratch_start = time.time()
    
    adabost_scratch = Adaboost()
    adabost_scratch.fit(train_x, train_y)
    
    y_predict_scratch = adabost_scratch.predict(test_x)
    
    adaboost_scratch_stop = time.time()
    
    print('='*50)
    print("True label = ", test_y)
    print("RealAdaboost label = ", y_predict_scratch)
    print("RealAdaboost train time = ", adaboost_scratch_stop - adaboost_scratch_start)
    print('='*50)
    '''
    
    #Calculate performance metrics
    measure_metrics = performance_metrics(y_true= test_y, y_predict = y_predict_ada_dt)
    tp = measure_metrics.find_TP(test_y, y_predict_ada_dt) 
    tn = measure_metrics.find_TN(test_y, y_predict_ada_dt)
    fp = measure_metrics.find_FP(test_y, y_predict_ada_dt)
    fn = measure_metrics.find_FN(test_y, y_predict_ada_dt)
    
    
    #Outputs
    #print("model params = ", clf.get_params())
    print('='*50)
    print("RAdaboost train score = ", ada_dt_clf.score(train_x,  train_y))
    print("RAdaboost test score = ", ada_dt_clf.score(test_x, test_y))
    print("Accuracy = ", measure_metrics.accuracy(tp, tn, fp, fn))
    print("Precision = ", measure_metrics.precision(tp, fp))
    print("Recall = ", measure_metrics.recall(tp, fn))
    print("f1 score =", measure_metrics.f1_score(measure_metrics.precision(tp, fp), measure_metrics.recall(tp, fn)))
    print('='*50)
    
    
    #Saving ML model
    if save_model == True:
        filename_ada_dt_clf = save_data_path
        pickle.dump(ada_dt_clf, open(filename_ada_dt_clf, 'wb'))
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the sun rain cls',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, 
                        default=os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/dataset/img',
                        help='dataset destination')
    parser.add_argument('-sx', '--size_x', type=int, default=160,
                        help='size x of image')
    parser.add_argument('-sy', '--size_y', type=int, default=120,
                        help='size y of image')
    parser.add_argument('-f', '--save', type=bool, default=True,
                        help='save model')
    parser.add_argument('-p', '--preprocess', type=bool, default=False,
                        help='preprocess')
    parser.add_argument('-sp', '--save_path', type=str, 
                        default=os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls/weights/sun_rain_wghts_adabostDT.sav',
                        help='save path destination')
    parser.add_argument('-t', '--test_size', type=float, default=0.15,
                        help='Percent of the data that is used as test (0-1)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    train( dataset_path= args.dataset, 
           size_x = args.size_x,
           size_y = args.size_y,
           preprocess = args.preprocess,
           test_size = args.test_size,
           save_data_path = args.save_path,
           save_model = args.save)