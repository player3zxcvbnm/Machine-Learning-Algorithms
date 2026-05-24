import numpy as np

class LinearReg:
    def __init__(self, LearnRate=0.01, Iterations=1000):
        self.LearnRate = LearnRate
        self.Iterations = Iterations
        self.w = None
        self.b = None

    def predict(self, x):
        return x.dot(self.w) + self.b

    def update(self, x, y):
        m = len(x)
        if len(x.shape) == 1:
            self.w = 0
        else:
            self.w = np.zeros(x.shape[1])
        self.b = 0

        for i in range(self.Iterations):
            errors = (x.dot(self.w)).T + self.b - y
            gradients_w = (1/m) * x.T.dot(errors)
            gradients_b = (1/m) * np.sum(errors)
            self.w = self.w - self.LearnRate * (gradients_w.T)
            self.b = self.b - self.LearnRate * (gradients_b.T)

    def total_cost(self, x, y):
        m = len(y)
        y_est = self.predict(x)
        return (1/(2*m)) * np.sum((y_est - y)**2)

    def r2_score(self, x, y):
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

def train_test_split(x, y, test_size=0.2):
    m = len(x)
    indices = np.random.permutation(m)
    x = x[indices]
    y = y[indices]
    split = int(0.8 * m)
    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]
    return x_train, x_test, y_train, y_test
