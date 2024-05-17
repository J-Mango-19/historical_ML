from process_io import *
import numpy as np
from adaline_class_sgd import Adaline

def eval_model(model1, model2, points, targets):
    for i in range(len(points)):
        x = points[i]
        y = targets[i]
        z_1 = model1.forward(x)
        z_2 = model2.forward(x)
        a_1 = model1.step_fxn(z_1)
        a_2 = model2.step_fxn(z_2)

        z_layer2 = 0.5*a_1 + 0.5*a_2 + 0.5

        if z_layer2 <= 0:
            yhat = -1
        else:
            yhat = 1
        print(f'pattern: {x[1:]} target: {y} model output: {yhat}') # x excludes first element bc bias is not part of the original pattern 


def madaline_job_assigner(x, y, z1, z2, Adaline1, Adaline2):
    if y == 1:
        if z1**2 < z2**2:
            Adaline1.update_weights(x, y)
        else:
            Adaline2.update_weights(x,y)
    if y == -1:
        if z1 > 0:
            Adaline1.update_weights(x,y)
        if z2 > 0:
            Adaline2.update_weights(x, y)


def main():
    points, targets, weights1, weights2, LR, max_epochs, graph = get_data_madaline()
    n = len(targets)
    # add x_0 = 1 to input vectors so that it multiplies with w_0 to form the bias term
    '''
    points = np.insert(points, 0, 1, axis=1)
    weights1= np.insert(weights1, 0, 1)
    weights2= np.insert(weights2, 0, 1)
    weights1 = np.zeros(len(weights1))
    weights2 = np.zeros(len(weights2))
    '''
    Adaline1 = Adaline(weights1, LR)
    Adaline2 = Adaline(weights2, LR)
    epochs = 0
    convergence = 0
    while(convergence == 0 and epochs < max_epochs):
        convergence = 1
        # compute gradient of the loss/error with respect to the weights:
        for i in range(len(points)):
            x = points[i]
            y = targets[i]

            z_1 = Adaline1.forward(x)
            z_2 = Adaline2.forward(x)
            a_1 = Adaline1.step_fxn(z_1)
            a_2 = Adaline2.step_fxn(z_2)

            z_layer2 = 0.5*a_1 + 0.5*a_2 +0.5

            if z_layer2 <= 0:
                yhat = -1
            else:
                yhat = 1

            if yhat != y:
                convergence = 0
                madaline_job_assigner(x, y, z_1, z_2, Adaline1, Adaline2)

        epochs += 1
    eval_model(Adaline1, Adaline2, points, targets)
    print_output(epochs, [Adaline1.weights, Adaline2.weights], -1, [], graph, "MADALINE - Stochastic Gradient Descent")
    for pattern in points:
        a1 = Adaline1.step_fxn(Adaline1.forward(pattern))
        a2 = Adaline2.step_fxn(Adaline2.forward(pattern))
        print(f'{pattern[1:]} a1 = {a1}, a2 = {a2}')


if __name__ == '__main__':
    main()




