import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

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

    def accuracy(self, x, y):
        predictions = self.output(x)
        return np.mean(predictions == y)

def train_test_split(x,y,test_size):
    m=len(x)
    indices = np.random.permutation(m)  # save it
    x = x[indices]  # apply shuffle to x
    y = y[indices]
    split=int((1-test_size)*m)
    x_train=x[:split]
    x_test=x[split:]
    y_train=y[:split]
    y_test=y[split:]
    return x_train, x_test, y_train, y_test

data = load_breast_cancer()
x = data.data
y = data.target
x = (x - x.mean(axis=0)) / x.std(axis=0)
model=LogisticReg(0.1,10000)
# split here before model.update
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#update
model.update(x_train,y_train)
print(f"Output is: {model.output(x)}")
model.sigmoid_plot(x_train,y_train,0)
print(f"Train accuracy: {model.accuracy(x_train, y_train):.4f}")
print(f"Test accuracy: {model.accuracy(x_test, y_test):.4f}")
print(f"Train cost: {model.cost_func(x_train, y_train):.4f}")
print(f"Test cost: {model.cost_func(x_test, y_test):.4f}")