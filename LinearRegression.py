
import numpy as np
from sklearn.datasets import fetch_california_housing
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
    
    
data = fetch_california_housing()
x = data.data[:, 0:1]  # just one feature first
y = data.target

model=LinearReg(0.03,1000)

model.update(x,y)

x_in = np.array([[5]]) 
y_out=model.predict(x_in)
print(f"Estimated output = {y_out[0]}")
model.graph(x,y,0)
print(f"Final cost: {model.total_cost(x, y):.4f}")
print(f"R2 Score: {model.r2_score(x, y):.4f}")
