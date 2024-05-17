import numpy as np
from process_io import *

def step_fxn(z):
    if z <= 0:
        return -1
    elif z > 0:
        return 1

def compute_MSE(points, targets, weights, n):
    SSE = 0
    for i in range(n):
        x = points[i]
        y = targets[i]
        yhat = np.dot(weights, x)
        SSE += (yhat - y)**2
    return (1/n) * SSE

def compute_grad(points, targets, weights, n):
    grad = 0
    for i in range(n):
        x = points[i]
        y = targets[i]
        yhat = np.dot(weights, x)
        grad += 2*(yhat - y) * x # the gradient of the loss wrt yhat and the weights, according to the chain rule
    return (1/n) * grad

def main():
    loss_list = []
    points, targets, weights, LR, max_epochs, graph = get_data()
    points = np.insert(points, 0, 1, axis=1)
    weights = np.append(weights, 0)

    n = len(targets)
    epochs = 0
    while(epochs <= max_epochs):
        for i, weight in enumerate(weights):
            print(f'w_{i} = {weight}')
        # mean squared error
        MSE = compute_MSE(points, targets, weights, n)

        # logging
        print(f'MSE: {MSE} after {epochs} epochs')

        # compute gradient of the loss/error with respect to the weights:
        grad_loss = compute_grad(points, targets, weights, n)
        loss_list.append(MSE)

        # gradient descent is performed by adding the negative of a scaled gradient in the weight update:
        weights = weights + LR*(-1)*grad_loss
        epochs += 1

    print_output(epochs, weights, MSE, loss_list, graph, "Adaline - Gradient Descent")

if __name__ == '__main__':
    main()
