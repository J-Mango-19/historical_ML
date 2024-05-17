from process_io import *
import numpy as np

class Adaline():
    def __init__(self, weights, lr):
        self.weights = weights
        self.lr = lr*0.1

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

def main():
    points, targets, weights, LR, max_epochs, graph = get_data()
    # add x_0 = 1 to input vectors so that it multiplies with w_0 to form the bias term
    points = np.insert(points, 0, 1, axis=1)
    weights = np.append(weights, 0)
    n = len(targets)
    epochs = 0

    Adaline1 = Adaline(weights, LR)
    loss_list = []

    epochs = 0
    while(epochs <= max_epochs):
        # compute gradient of the loss/error with respect to the weights:
        for i, datapoint in enumerate(points):
            x = points[i]
            y = targets[i]
            Adaline1.update_weights(x, y)

        # mean squared error
        MSE = Adaline1.compute_MSE(points, targets, n)  # n = the number of datapoints
        loss_list.append(MSE)

        # logging
        print(f'MSE: {MSE} after {epochs} epochs')
        epochs += 1

    print_output(epochs, Adaline1.weights, MSE, loss_list, graph, "ADALINE - Stochastic Gradient Descent")

if __name__ == '__main__':
    main()
