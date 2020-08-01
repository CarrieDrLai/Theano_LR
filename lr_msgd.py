# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 04:03:07 2020

@author: CarrieLai
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle


#def loadDataSet(p, file_n):
#
#    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
#            (test_set_x, test_set_y)]
#
#    return rval

class MSGD_LogisticRegression(object):
    
    def __init__(self, X_train, y_train, X_test, y_test, lr, batch_size, n_epochs, lamda, improve_thresh, early_stop_thresh, weight_decay,n_class=2):
        
        self.n_class = n_class
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lamda = lamda
        
        self.patience = 5000
        self.patience_increase = 2
        self.improve_thresh = improve_thresh
        self.early_stop_thresh = early_stop_thresh
        self.weight_decay = weight_decay
        self.done_looping = False
        
        self.fn_model = "train_set_" + str(self.n_epochs)
        self.TPR = []
        self.FPR = []
        
        self.X_train = X_train.reshape(np.shape(X_train)[0],np.shape(X_train)[1]*np.shape(X_train)[2])
        self.X_valid = self.X_train[int(np.shape(X_train)[0]*7/8):]
        self.y_valid = y_train[int(np.shape(X_train)[0]*7/8):]
        self.X_train = self.X_train[:int(np.shape(X_train)[0]*7/8)]
        self.y_train = y_train[:int(np.shape(X_train)[0]*7/8)]
        self.X_test = X_test.reshape(np.shape(X_test)[0],np.shape(X_test)[1]*np.shape(X_test)[2])
        self.y_test = y_test


        self.n_in = np.shape(X_train)[1]        
        self.W = np.zeros([self.n_in, self.n_class]) # mnist:( n_in=28 * 28, n_out=10) basket:(96*96, 2)
        self.b = np.zeros(self.n_class, ) # theta = (W,b)
        
    # def sigmoid(self, X):
    #     return 1.0 / (1.0 + np.exp(-X))

    # predict_function_ljf
    def compute_prob_y(self, x):
        # Prob_y = np.exp(np.dot(x, self.theta))
        # x : (n_sample,n_in) W: (n_in, n_out)
        Prob_y = np.exp(np.dot(x, self.W) + self.b)
        sum_k = np.sum(Prob_y, axis=1)
        Prob_y /= sum_k.reshape(sum_k.shape[0], 1)  # normalization
#        pred_y = []
#        for i in range(np.shape(Prob_y)[0]):
#            k = Prob_y[i].tolist()
#            max_index = k.index(max(k))
#            pred_y.append(max_index)
#        pred_y = np.c_[pred_y]  # transform list to column vector
        return Prob_y

    # loss_function_ljf
    def loss_function(self, x, y): # y = label, [1 2 ... n]

#        Prob_y = np.exp(np.dot(x, self.W) + self.b)
#        sum_k = np.sum(Prob_y, axis=1)
#        Prob_y /= sum_k.reshape(sum_k.shape[0], 1)  # normalization
        Prob_y = self.compute_prob_y(x)
        loss = 0
        for i in range(x.shape[0]):
            # loss -= np.log(Prob_y[i, y[i] - 1])
            # mean_cross_entropy
            loss -= y[i] * np.log(Prob_y[i,  y[i]]) + ( 1 - y[i]) * np.log(Prob_y[i, 1 - y[i]])

        return loss / x.shape[0]
    
    def error(self, flag):
        if flag == 1:
            x = self.X_train
            y = self.y_train
        else:
            x = self.X_valid
            y = self.y_valid
            
        y_pred = np.argmax(self.compute_prob_y(x),1)

        return np.mean(np.not_equal(y_pred, y))


    def compute_p_y_given_x(self, batch_ind, flag_dataset = 1):
        
        ind = batch_ind
        
        if flag_dataset ==1:
            x = self.X_train[ind * self.batch_size : (ind +1) * self.batch_size]
        elif flag_dataset==2:
            tt = int(self.X_valid.shape[0] / self.batch_size)
            x = self.X_valid[0 : tt * self.batch_size]
        else:
#            tt = int(self.X_test.shape[0] / self.batch_size)
#            x = self.X_test[0: tt * self.batch_size]
            x = self.X_test
            

#        self.exp_x_multipy_W_plus_b = np.exp(np.dot(x, self.W) + self.b)
#        sigma = np.sum(self.exp_x_multipy_W_plus_b, axis = 1)
#        self.p_y_given_x = self.exp_x_multipy_W_plus_b / sigma.reshape(sigma.shape[0], 1)
        self.p_y_given_x = self.compute_prob_y(x)
        return self.p_y_given_x
        
        
    def gradient_W_b(self, batch_ind):
        
        ind = batch_ind
        
        x = self.X_train[ind * self.batch_size: (ind + 1) * self.batch_size]
        y = self.y_train[ind * self.batch_size: (ind + 1) * self.batch_size]

        y_is_j = (y.reshape(y.shape[0], 1) == np.array(np.arange(self.n_class), dtype=int))*1
        coef = y_is_j - self.p_y_given_x
        
        if self.weight_decay == 0:
            self.delta_W = (-1.0 * np.dot(coef.transpose(), x) / y.shape[0]).transpose()
            self.delta_b = -1.0 * np.mean(coef, axis=0)
        else:
            self.delta_W = (-1.0 * np.dot(coef.transpose(), x) / y.shape[0]).transpose() + self.lamda * self.W
            self.delta_b = -1.0 * np.mean(coef, axis=0) + self.lamda * self.b
        
    def update_W_b(self):
        self.W -= self.lr * self.delta_W
        self.b -= self.lr * self.delta_b
        
    def predict(self,batch_ind):
        
        test_predict = self.compute_p_y_given_x(batch_ind, flag_dataset=3)
        test_predict = np.argmax(test_predict,1)
        #ROC曲线
        true_pos = sum((self.y_test == 1) * test_predict)
        true_neg = sum((self.y_test == 0) * (test_predict == 0))
        false_pos = sum((self.y_test == 0) * (test_predict == 1))
        false_neg = sum((self.y_test == 1) * (test_predict == 0))
        self.TPR.append(true_pos/(false_neg + true_pos))
        self.FPR.append(false_pos/(false_pos + true_neg))# False Positive Rate
        
    def draw_ROC(self):
        
        np.save("roc\\" + self.fn_model + ".npy", [self.TPR, self.FPR])
        plt.figure()
        plt.plot(self.FPR,self.TPR,'r--',5)
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Ture Positive Rate')
        plt.axis([0, 1, 0, 1])
        plt.grid(True)
        plt.show()
        
        
    def load_model(self):
        
        model = pickle("model\\" + self.fn_model + ".pkl")
        
        return model
    
    
    def save_model(self):
        
        with open("model\\" + self.fn_model + ".pkl", 'wb') as f:
            pickle.dump((self.W, self.b), f)
    
        
    def msgd_optimization(self):

        n_train_batches = self.X_train.shape[0] // self.batch_size
        valid_freq = min(n_train_batches, self.patience // 2)

        epoch = 0
        best_validation_loss = np.inf

        while (epoch < self.n_epochs) and (not self.done_looping):
            epoch = epoch + 1
            for batch_ind in np.arange(n_train_batches):

                prob = self.compute_p_y_given_x(batch_ind) #predict_y
                self.gradient_W_b(batch_ind) 
                self.update_W_b()               
                self.predict(batch_ind)
                    
                it = (epoch - 1) * n_train_batches + batch_ind
                if (it + 1) % valid_freq == 0:

                    loss_valid_curr = self.error(flag=2) 
                    train_loss = self.loss_function(self.X_train, self.y_train)
                    print(
                        'epoch %i, minibatch %i/%i, patience %d, train_loss %f, validation error %f %%' %
                        (
                            epoch,
                            batch_ind + 1,
                            n_train_batches,
                            self.patience,
                            train_loss,
                            loss_valid_curr * 100.
                        )
                    )
                    if loss_valid_curr < best_validation_loss:
                        if loss_valid_curr < best_validation_loss * self.improve_thresh:
                            self.patience = max(self.patience, it * self.patience_increase)
                        best_validation_loss = loss_valid_curr
                    if self.patience <= it:
                        done_looping = True
                        break
                    if best_validation_loss < self.early_stop_thresh:
                        break
                    
#        return self.W, self.b       
    
    




