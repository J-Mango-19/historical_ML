from process_io import *
import numpy as np

class Adaline():
    def __init__(self, weights, lr):
        self.weights = weights
        self.lr = lr

    def forward(self, x):
        z = np.dot(self.weights, x)
        return z
    def step_fxn(self, z):
        if z <= 0:
            return -1
        elif z > 0:
            return 1

    def compute_MSE(self, points, targets, n):
        SSE = 0
        for i in range(n):
            x = points[i]
            y = targets[i]
            #yhat = np.dot(weights, x)
            yhat = self.forward(x)
            #print(f'yhat: {yhat}')
            SSE += (yhat - y)**2
        return (1/n) * SSE

    def update_weights(self, x, y):
        yhat = self.forward(x)
        grad_loss = 2 * (yhat - y) * x # from application of chain rule onto (yhat -y)^2 wrt weights
        #print(f'grad loss: {grad_loss}')
        self.weights = self.weights + (self.lr * -1 * grad_loss)

