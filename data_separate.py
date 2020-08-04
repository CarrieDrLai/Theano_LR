# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:14:53 2020

@author: CarrieLai
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

fn_dataset = ["X_train", "X_test", "y_train", "y_test"]


def SeperateData(dir_save_feature,fn_feature):
    
    if os.path.exists(dir_save_feature + "feature_pos.npy"):
        print(" =========  Feature_Pos File is already exist  =========")
        fea_pos = np.load(dir_save_feature + "feature_pos.npy")
    if os.path.exists(dir_save_feature + "feature_neg.npy"):
        print(" =========  Feature_Neg File is already exist  =========")
        fea_neg = np.load(dir_save_feature + "feature_neg.npy")    
    else:
        print(" >>>>>> Create Feature Pos and Feature Neg File !!!!!!  Start  !!!!!! ") 
    
        feature = []
        label = []
        for i in range(1,4):
            fn_feature = "feature"+str(i)
            feature.append(np.load(dir_save_feature + fn_feature + ".npy"))
            save_label_npy = ".\\sample\\label" + str(i) + ".npy"
            label.append(np.load(save_label_npy))
        feature = np.concatenate(feature)
    #    feature = np.reshape(feature,[np.shape(feature)[0],np.shape(feature)[1]])
        label = np.concatenate(label)
        
        ind_pos = np.array(np.where(label == 1))
        ind_pos = np.reshape(ind_pos, np.shape(ind_pos)[1])
        ind_neg = np.array(np.where(label == 0))
        ind_neg = np.reshape(ind_neg, np.shape(ind_neg)[1])
        fea_pos = feature[ind_pos]
        fea_neg = feature[ind_neg]
        np.save(dir_save_feature + "feature_pos.npy", fea_pos)
        np.save(dir_save_feature + "feature_neg.npy", fea_neg)
        np.save(dir_save_feature + "index_pos.npy", ind_pos)
        np.save(dir_save_feature + "index_neg.npy", ind_neg)
    
    if os.path.exists(dir_save_feature + fn_dataset[0] + ".npy"):
        print(" =========  Data Seperation is already done  =========")
    else:
        label_pos = np.uint8(np.ones(np.shape(fea_pos)[0]))
        label_neg = np.uint8(np.zeros(np.shape(fea_neg)[0]))
        X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(fea_pos, label_pos,test_size=0.2, random_state=0)
        X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(fea_neg, label_neg,test_size=0.2, random_state=0)
        X_train = np.array(list(X_train_pos) + list(X_train_neg))
        X_test = np.array(list(X_test_pos) + list(X_test_neg))
        y_train = np.array(list(y_train_pos) + list(y_train_neg))
        y_test = np.array(list(y_test_pos) + list(y_test_neg))    
        np.save(dir_save_feature + fn_dataset[0] + ".npy", X_train)
        np.save(dir_save_feature + fn_dataset[1] + ".npy", X_test)
        np.save(dir_save_feature + fn_dataset[2] + ".npy", y_train)
        np.save(dir_save_feature + fn_dataset[3] + ".npy", y_test)
        
        
        
def import_data_seperated(dir_save_feature, fn_dataset):
    
    X_train = np.load(dir_save_feature + fn_dataset[0] + ".npy")
    X_test = np.load(dir_save_feature + fn_dataset[1] + ".npy")
    y_train = np.load(dir_save_feature + fn_dataset[2] + ".npy")
    y_test = np.load(dir_save_feature + fn_dataset[3] + ".npy")    
    
    return X_train, X_test, y_train, y_test