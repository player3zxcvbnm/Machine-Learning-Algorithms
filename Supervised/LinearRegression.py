
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearReg:
    def __init__(self,LearnRate=0.01,Iterations=1000):
        self.LearnRate=LearnRate
        self.Iterations=Iterations
        self.w=None
        self.b=None
    
    def predict(self,x):
        return x.dot(self.w)+self.b #prediction function wx+b
    
    def update(self,x,y):
        m=len(x)
        if len(x.shape)==1:
            self.w=0
        else:
            self.w=np.zeros(x.shape[1])
        self.b=0

        for i in range(self.Iterations):
            errors=(x.dot(self.w)).T+self.b-y #find error from actual data
            gradients_w = (1/m) * x.T.dot(errors) 
            gradients_b = (1/m) * np.sum(errors)
            #adjust w and b based on the error
            self.w=self.w-self.LearnRate*(gradients_w.T)
            self.b=self.b-self.LearnRate*(gradients_b.T)
    
    def graph(self,x,y,feature_index):
        if len(x.shape)==1:
            x_feature=x
        else:
            x_feature=x[:,feature_index]
        plt.title('Linear Regression Model')
        plt.scatter( x_feature, y, c='r', label='True Value')
        est=self.predict(x)
        plt.plot( x_feature, est, c='b', label='Estimated Value')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        plt.show()
        

    def total_cost(self,x,y):
        m=len(y)
        y_est=self.predict(x)
        return (1/(2*m))*np.sum((y_est-y)**2)

    def r2_score(self, x, y):
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)  
        # how wrong your model is
        ss_tot = np.sum((y - np.mean(y)) ** 2)  
        # how wrong just using the mean would be
        return 1 - (ss_res / ss_tot)
        # ratio of your error vs baseline error

def train_test_split(x, y, test_size=0.2):
    m=len(x)
    indices = np.random.permutation(m)  # save it
    x = x[indices]  # apply shuffle to x
    y = y[indices]  # apply same shuffle to y
    split=int(0.8*m)
    x_train=x[:split]
    x_test=x[split:]
    y_train=y[:split]
    y_test=y[split:]
    return x_train, x_test, y_train, y_test
    
    

