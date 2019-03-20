import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets



'''
f(x) = 1 / （1+exp(-x))
'''

def create_dataset():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    new_x = []
    new_y = []
    for i in range(y.shape[0]):
        if (y[i] < 2):
            new_x.append(x[i])
            new_y.append(y[i])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    return new_x, new_y

def sigmod(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, epochs=50, lr=1e-4, threshold=0.5):
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        
    def _extend_data(self, x):
        new_x = []
        for row in x:
            new_x.append([*row, 1.0])
        return np.array(new_x)
    def fit(self, x, y):
        '''
            x: the features set of the samples, n*m, one row represent one sample
            y: the label, {0, 1}
        '''
        
        x_ = self._extend_data(x)
        y = y[:, np.newaxis]
        self.w = np.zeros( (x_.shape[1], 1))
        for epoch in range(self.epochs):
            z = np.dot(x_, self.w)
            y_ = sigmod(z)
            delta = np.dot(x_.T, y-y_)
            self.w = self.w + self.lr * delta
            loss = -np.sum(y*z - np.log(1+np.exp(z)))
            print("Epoch: %d, Training Loss:%.4f, accurate: %f" % (epoch, loss, self.score(x, y)))

    def predict(self, x):
        x = self._extend_data(x)
        y = sigmod(np.dot(x, self.w))
        y = np.squeeze(y)
        return y

    def score(self, x, y):
        y_ = self.predict(x)
        y = np.squeeze(y)
        i = y_ >= self.threshold
        y_[i] = 1
        i = y_ < self.threshold
        y_[i] = 0
        return np.mean(np.abs(y == y_))

    def set_threshold(self, v):
        self.threshold = v

x, y = create_dataset()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(y_test)
lr = LogisticRegression(lr=1e-3, epochs=10)
lr.fit(x_train, y_train)
score = lr.score(x_test, y_test)
print(score)
