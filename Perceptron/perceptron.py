import numpy as np
from process_io import *

def step_fxn(z):
    if z <= 0:
        return 0
    if z > 0:
        return 1

def update_weights(x, y, y_hat, weights, convergence, LR):
    if y - y_hat != 0:
        convergence = False
        weights = weights + LR * (y - y_hat) * x
    return weights, convergence

def main():
    points, targets, weights, LR, max_epochs = get_data_perceptron()
    epochs = 0
    convergence = False
    while convergence == False:
        convergence = True
        epochs += 1
        for x, y in zip(points, targets):
            z = np.dot(weights, x)
            y_hat = step_fxn(z)
            weights, convergence = update_weights(x, y, y_hat, weights, convergence, LR)
        if epochs > max_epochs:
            print(f"Perceptron algorithm did not converge in {epochs} epochs. Data may not be linealy separable. Set max epochs using -e on command line")
            break

    print_output_perceptron(epochs, weights)

if __name__ == '__main__':
    main()

