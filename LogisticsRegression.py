import numpy as np
import matplotlib.pyplot as plt

class LogisticReg:
    def __init__(self, LearnRate=0.01, Iterations=1000):
        self.LearnRate = LearnRate
        self.Iterations = Iterations
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_func(self, x, y):
        cost = 0
        m = x.shape[0]
        z = np.dot(x, self.w) + self.b 
        f_wb = self.sigmoid(z)
        cost = - (1/m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
        return cost

    def update(self, x, y):
        m, n = x.shape  
        if self.w is None:
            self.w = np.zeros(n) 
        if self.b is None:
            self.b = 0  

        for _ in range(self.Iterations):
            z = np.dot(x, self.w) + self.b
            f_wb = self.sigmoid(z)
            errors = f_wb - y

            
            gradient_w = (1/m) * np.dot(x.T, errors)
            gradient_b = (1/m) * np.sum(errors)

            
            self.w -= self.LearnRate * gradient_w
            self.b -= self.LearnRate * gradient_b

    def output(self, x):
        z = np.dot(x, self.w) + self.b
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int) 

    def sigmoid_plot(self, x, y, feature_index):
        if len(x.shape) == 1:  
            x_feature = x
        else:
            x_feature = x[:, feature_index]  

        z = self.sigmoid(np.dot(x, self.w) + self.b)
        
        
        plt.title('Sigmoid Plot')
        plt.plot(x_feature, z, c='r', label='Sigmoid Curve')
        plt.scatter(x_feature, y, label='Data Points')
        plt.xlabel('Feature')
        plt.ylabel('Predicted Probability')
        plt.grid(True)
        plt.legend()
        plt.show()


x_tr=np.array([[0.2],[0.3],[0.5],[0.7],[1]])
y_tr=np.array([0,0,0,1,1])

model=LogisticReg(0.1,10000)
model.update(x_tr,y_tr)
x_test=np.array([0.9])
y_out=model.output(x_test)
print(f"Output is: {y_out}")
model.sigmoid_plot(x_tr,y_tr,0)
c=model.cost_func(x_tr,y_tr)
print(f"cost is {c}")