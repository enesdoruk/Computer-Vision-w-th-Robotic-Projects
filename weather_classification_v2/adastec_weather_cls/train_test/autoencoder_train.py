import os 
import sys
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
 
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.expanduser('~') + '/Desktop/weather_classification/adastec_weather_cls')

from process.feature_extractor import feature_extract
from process.autoencoder_model import Autoencoder


class AE_train:
    def __init__(self, path, in_shape, enc_shape, n_epochs, file_name):
        self.path = path 
        self. in_shape = in_shape
        self.enc_shape = enc_shape
        self.n_epochs = n_epochs
        self.file_name = file_name
    
    def preprocess(self):
        train_data = pd.read_csv(os.path.expanduser('~') + self.path)
        train_x = train_data.iloc[:,:-1]
        train_y = train_data.iloc[:,-1]
        
        train_x = MinMaxScaler().fit_transform(train_x)
        #train_x = np.array(train_x)        
                
        device = ('cuda' if torch.cuda.is_available() else 'cpu')

        tensor_x = torch.from_numpy(train_x).to(device)
    
        return tensor_x, train_y

    
    def train(self, data):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = Autoencoder(in_shape= self.in_shape, enc_shape= self.enc_shape).double().to(device)

        error = nn.MSELoss()

        optimizer = optim.Adam(model.parameters())
        
        model.train()
        for epoch in range(1, self.n_epochs + 1):
            optimizer.zero_grad()
            output = model(data)
            loss = error(output, data)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'epoch {epoch} \t Loss: {loss.item():.4g}')
        
        torch.save(model.state_dict(), 
                   os.path.expanduser('~') + '/Desktop//weather_classification/adastec_weather_cls/weights/autoencoder_feature_red.pth')

    def test(self, data):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
        encoder = Autoencoder(in_shape= self.in_shape, enc_shape= self.enc_shape).double().to(device)
        encoder.load_state_dict(torch.load(os.path.expanduser('~') + '/Desktop//weather_classification/adastec_weather_cls/weights/autoencoder_feature_red.pth'))
        encoder.eval()
        
        with torch.no_grad():
            loss = nn.MSELoss()
            
            encoded = encoder.encode(data)
            decoded = encoder.decode(encoded)
            
            mse = loss(decoded, data).item()
            
            enc = encoded.cpu().detach().numpy()
            dec = decoded.cpu().detach().numpy()
        
        print(f'Root mean squared error: {np.sqrt(mse):.4g}')

        return enc    
    
    def list2csv(self, total_list):
        df = pd.DataFrame(total_list)
        df.to_csv(os.path.expanduser('~') + '{}'.format(self.file_name), index=False, header=False)



if __name__ == '__main__':
    input_path =  '/Desktop/weather_classification/adastec_weather_cls/dataset/csv/dataset.csv'
    save_path =  '/Desktop/weather_classification/adastec_weather_cls/dataset/csv/reduct_weather_cls.csv'
    
    in_shape = 48000
    out_shape = 3000
    epochs = 5000
    
    ae_train = AE_train(input_path, in_shape, out_shape, epochs, save_path)
    
    pre_data, train_y = ae_train.preprocess()
    
    ae_train.train(pre_data)
    
    test = ae_train.test(pre_data)
    
    test_pd = pd.DataFrame(test)
    train_y_pd = pd.DataFrame(train_y)

    result = pd.concat([test_pd, train_y_pd], axis = 1, ignore_index=True)

    ae_train.list2csv(result)    
    