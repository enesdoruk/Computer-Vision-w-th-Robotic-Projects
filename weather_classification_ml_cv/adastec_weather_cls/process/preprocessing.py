import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer

class data_preprocess:
    def __init__(self, data):
        self.data = data


    def preprocess(self, data):
        imputer = SimpleImputer(fill_value=np.nan, strategy='mean')
        data = imputer.fit_transform(data)

        data_pd = pd.DataFrame(data)

        x_data = data_pd.iloc[:,:-1]

        y_data = data_pd.iloc[:,-1]
        y_data = y_data.astype(int)

        mean = np.mean(x_data.values)
        std = np.std(x_data.values) 
        data_norm = (x_data.values - mean) / std
        
        data_norm = pd.DataFrame(data_norm)
        data_norm =  pd.concat([data_norm, y_data], axis=1)

        return data_norm

    
    def list2csv(self, df, file_name):
        return df.to_csv('{}'.format(file_name), index=False, header=False)


    def cloud_rain_dataset(self, data):
        label = data.iloc[:,-1].to_numpy().astype(int)
        cloud_rain = []
        
        for i in range(len(label)):
            if label[i] != int(1) and label[i] != int(2):
                cloud_rain.append(i)
        
        return  data.drop(cloud_rain, axis=0)


    def sun_cloud_dataset(self, data):
        label = data.iloc[:,-1].to_numpy().astype(int)
        sun_cloud = []
        
        for i in range(len(label)):
            if label[i] != int(0) and label[i] != int(1):
                sun_cloud.append(i)
        
        return data.drop(sun_cloud, axis=0)


    def sun_rain_dataset(self, data):
        label = data.iloc[:,-1].to_numpy().astype(int)
        sun_rain = []
        
        for i in range(len(label)):
            if label[i] != int(0) and label[i] != int(2) :
                sun_rain.append(i)
        
        return data.drop(sun_rain, axis=0)


if __name__ == '__main__':
    data = pd.read_csv('dataset/csv_dataset/mini_dataset.csv')
    
    proces_ex = data_preprocess(data)
    data = proces_ex.preprocess(data)

    #cloud = 1 , sunny = 0, rainy = 2

    data_cloud_rain = data.copy()
    data_cloud_rain = proces_ex.cloud_rain_dataset(data_cloud_rain)
    proces_ex.list2csv(data_cloud_rain, './train/cloudy_rainy/cloud_rain.csv')

    data_sun_rain = data.copy()
    sun_rain = proces_ex.sun_rain_dataset(data_sun_rain)
    proces_ex.list2csv(sun_rain, './train/sunny_rainy/sun_rain.csv')

    data_sun_cloud = data.copy()
    sun_cloud = proces_ex.sun_cloud_dataset(data_sun_cloud)
    proces_ex.list2csv(sun_cloud, './train/sunny_cloudy/sun_cloud.csv')