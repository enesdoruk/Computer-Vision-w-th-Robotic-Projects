import os 
import numpy as np
import pandas as pd



class performance_metrics:
    def __init__(self, y_true, y_predict):
        self.y_true = y_true
        self.y_predict = y_predict

    
    def mse(self, y_true, y_predict):
        return np.power((y_true - y_predict), 2) 

    
    def mae(self, y_true, y_predict):
        return np.abs(y_true - y_predict)

    
    def rmse(self, y_true, y_predict):
        mse = np.power((y_true - y_predict), 2)

        return np.sqrt(mse.mean())

   
    def r2cof(self, y_true, y_predict):
        se_line = sum(np.power((y_true - y_predict), 2))
        se_mean = sum(np.power((y_true - y_true.mean()), 2))

        return 1 - (se_line / se_mean)
    
    
    def find_TP(self, y_true, y_predict):
        return sum((y_true == 1) & (y_predict == 1))

    
    def find_FN(self, y_true, y_predict):
        return sum((y_true == 1) & (y_predict == -1))

    
    def find_FP(self, y_true, y_predict):
        return sum((y_true == -1) & (y_predict == 1))

    
    def find_TN(self, y_true, y_predict):
        return sum((y_true == -1) & (y_predict == -1))
    
    
    def accuracy(self, tp, tn, fp, fn):
        if tp + fp + fn + tn == 0:
            return 0
        else:
            return (tp + tn) / (tp + fp + fn + tn)

    
    def precision(self, tp, fp):
        if tp + fp == 0:
            return 0
        else:
            return tp / (tp + fp)
    
    
    def recall(self, tp, fn):
        if tp + fn == 0:
            return 0
        else:
            return tp / (tp + fn)
    

    def specifity(self, tn, fp):
        if tn + fp == 0:
            return 0 
        else:
            return tn / (tn + fp)
    

    def f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0 
        else:
            return 2 * ((precision * recall) / (precision + recall))

    
    def au_roc(self, tp, fn, fp, tn):
        if tp + fn == 0 or fp + tn == 0:
            return 0, 0
        else:
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            return tpr, fpr 
