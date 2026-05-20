
import numpy as np
import matplotlib.pyplot as plt

class LinearReg:
    def __init__(self,LearnRate=0.01,Iterations=1000):
        self.LearnRate=LearnRate
        self.Iterations=Iterations
        self.w=None
        self.b=None
    
    def predict(self,x):
        return x.dot(self.w)+self.b
    
    def update(self,x,y):
        m=len(x)
        if len(x.shape)==1:
            self.w=0
        else:
            self.w=np.zeros(x.shape[1])
        self.b=np.zeros(x.shape[0])

        for i in range(self.Iterations):
            errors=(x.dot(self.w)).T+self.b-y
            gradients_w = (1/m) * x.T.dot(errors)
            gradients_b = (1/m) * np.sum(errors)
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
    
    
x=np.array([3,4,7])
y=np.array([8,14,16])

model=LinearReg(0.03,1000)

model.update(x,y)

x_in=np.array([5])
y_out=model.predict(x_in)
print(f"Estimated output = {y_out[0]}")
model.graph(x,y,0)