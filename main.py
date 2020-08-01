# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:44:23 2020

@author: CarrieLai
"""

from get_dataset import GetHoopPos,GetDataset,create_annotation
from load_dataset import load_dataset
from extract_feature import extract_feature
from lr_msgd import MSGD_LogisticRegression
#from naive_classifier import Predict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#############################   Read Congig File   #############################

print(" ============== Read Config File ==============\n")
f = open('config.txt','r')
config = f.readlines()
f.close()

var_list=[]
for var in config:
    var = var.strip()
    var_name = var.split("=")[0].strip()
    var_value = var.split("=")[1].strip()
    if var_name == 'crop_size' or var_name == 'colume':
        var_value = eval(var_value)
    if var_name == 'block_size' or var_name == 'block_stride' or var_name == 'cell_size' or var_name == 'bin_num':
        var_value = eval(var_value)
    if var_name == 'training_set' or var_name == 'epoch' or var_name == 'learning_rate' or var_name == 'batch_size' or var_name == 'lamda' or var_name == 'improve_thresh':
        var_value = eval(var_value)    
    if var_name == 'frame' or var_name == 'line_search' or var_name == 'weight_decay' or var_name == 'early_stop':
        var_value = eval(var_value)    
    var_list.append([var_name, var_value])
    print( " " + var_name + " = " + str(var_value))
    
var_list = dict(var_list)
    
print(" ============== Read Config File Success ==============\n")

#############################        Parameter        ###################################


fn = var_list["fn"]
fn_data = fn + ".jpg"

crop_size = var_list["crop_size"]
colume = var_list["colume"]
fn_dataset = ["X_train", "X_test", "y_train", "y_test"]

#thresh = var_list["thresh"]
task_type = var_list["task_type"]
# 1. locate: get HoopPos
# 2. crop : get all patch, save data.jpg and rough separete(and then separate manually)  
# 3. annotation
# 4. feature 
# 5. classification


#############################  Step 1 : Prepare Dataset  ################################

if task_type == 'locate': 

    fn_video = var_list["fn_video"]
    dir_video = ".\\video\\" + fn_video
    
    ################  (a)Get Hoop Position
    hoop = GetHoopPos(fn_video,crop_size)
    hoop_pos = hoop.get_pos()        #hoop_pos = [(924,133)]

elif task_type == 'crop':
    
    fn_video = var_list["fn_video"]
    dir_video = ".\\" + fn_video
    path_data = ".\\sample\\" + fn_data
    save_data_npy = ".\\sample\\" + fn + ".npy"

    dir_pos = var_list["dir_pos"]
    dir_neg = var_list["dir_neg"]
    
    ################   (b)Get all data & save data as npy
    
    Data = GetDataset(fn_video, path_data, save_data_npy, dir_pos, dir_neg, crop_size,hoop_pos,colume)  
    patch_all = Data.get_data()
    Data.rough_separate()
    ################  (c)Manually seperate the data after rough seperation 
    
elif task_type == 'annotation':
    ################  (d)Make Annotation File
    fn_annotation = var_list["fn_annotation"]
    path_annotation = ".\\sample\\" + fn_annotation
    
    save_data_npy = ".\\sample\\" + fn + ".npy"
    save_label_npy = ".\\sample\\label" + fn[-1] + ".npy"

    dir_pos = var_list["dir_pos"]
    dir_neg = var_list["dir_neg"]

    create_annotation(dir_pos, dir_neg, path_annotation, save_label_npy, save_data_npy, crop_size, colume)

###########################    Step 2 : Train Model   ##############################

elif task_type == 'feature':

    path_data = ".\\sample\\" + fn_data
    fn_annotation = var_list["fn_annotation"]
    path_annotation = ".\\sample\\" + fn_annotation
    fn_feature = var_list["fn_feature"]
    dir_save_feature = var_list["dir_save_feature"]
    
    block_size = var_list["block_size"]
    block_stride = var_list["block_stride"]
    cell_size = var_list["cell_size"]
    bin_num = var_list["bin_num"]
    
    ###############  (a)load dataset

    Dataset = load_dataset(path_annotation,path_data,crop_size)
    data, label = Dataset.load_data() 
    #X_train,X_test, y_train, y_test = train_test_split(data,label,test_size=0.3, random_state=0)

    ###############  (b)Extract Feature  

    hog = extract_feature(dir_save_feature, fn_feature, data, block_size, block_stride, cell_size, bin_num)
    feature = hog.HoG_output_vector()
    
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

    X_train, X_test, y_train, y_test = train_test_split(feature, label,test_size=0.2, random_state=0)
    np.save(dir_save_feature + fn_dataset[0] + ".npy", X_train)
    np.save(dir_save_feature + fn_dataset[1] + ".npy", X_test)
    np.save(dir_save_feature + fn_dataset[2] + ".npy", y_train)
    np.save(dir_save_feature + fn_dataset[3] + ".npy", y_test)
    
#    hog_train = extract_feature(dir_save_feature, "X_train", X_train, block_size, block_stride, cell_size, bin_num)
#    train_feature = hog_train.HoG_output_vector()
#    print("\n >>>>>> Extract Train Feature !!!!!!  Suceess  !!!!!! \n") 
#    hog_test = extract_feature(dir_save_feature, "X_test", X_test[0:1], block_size, block_stride, cell_size, bin_num)
#    test_feature = hog_test.HoG_output_vector()
#    print("\n >>>>>> Extract Test Feature !!!!!!  Suceess  !!!!!! \n") 

elif task_type == 'classification':
    
    #fn_feature = var_list["fn_feature"]
    dir_save_feature = var_list["dir_save_feature"]
    X_train = np.load(dir_save_feature + fn_dataset[0] + ".npy")
    X_test = np.load(dir_save_feature + fn_dataset[1] + ".npy")
    y_train = np.load(dir_save_feature + fn_dataset[2] + ".npy")
    y_test = np.load(dir_save_feature + fn_dataset[3] + ".npy")    
    
    ##############  (c)Fit Model & Predict  
    
    training_set = var_list["training_set"]
    n_epochs = var_list["epoch"]
    lr = var_list["learning_rate"]
    batch_size = var_list["batch_size"]
    lamda = var_list["lamda"]
    improve_thresh = var_list["improve_thresh"]

    frame = var_list["frame"]
    line_search = var_list["line_search"]
    weight_decay = var_list["weight_decay"]
    early_stop = var_list["early_stop"]    
    
    idx = np.random.randint(0,np.shape(X_train)[0],training_set)
    X_train_part = X_train[idx]
    y_train_part = y_train[idx]
    
    
    model = MSGD_LogisticRegression(X_train_part, y_train_part, X_test, y_test, lr, batch_size, n_epochs, lamda, improve_thresh, early_stop,weight_decay, n_class=2)
    model.msgd_optimization()
    model.draw_ROC()
    model.save_model()


#    TPR = []
#    FPR = []
##    for t in range(10,37):
##        thresh = t/50
##        Pre = Predict(train_feature, test_feature, y_train, y_test, thresh)
##        train_predict, test_predict = Pre.predict()
#
############################    Step 3 : Analysis    ##############################
#    
#        #ROC曲线
#        true_pos = sum((y_test == 1) * test_predict)
#        true_neg = sum((y_test == 0) * (test_predict == 0))
#        false_pos = sum((y_test == 0) * (test_predict == 1))
#        false_neg = sum((y_test == 1) * (test_predict == 0))
#        TPR.append(true_pos/(false_neg + true_pos))
#        FPR.append(false_pos/(false_pos + true_neg))# False Positive Rate
#
#    plt.figure()
#    plt.plot(FPR,TPR,'r--',5)
#    plt.title('ROC Curve')
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('Ture Positive Rate')
#    plt.axis([0, 1, 0, 1])
#    plt.grid(True)
#    plt.show()