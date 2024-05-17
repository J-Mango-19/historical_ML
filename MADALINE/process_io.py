import sys
import matplotlib.pyplot as plt
import numpy as np
def get_data():
    graph = False
    LR = 0.01
    points = np.array([[3, 1, 1], [7, 3, 3], [1, 4, 6], [5, 7, 4], [3, 1, 2], [8, 2, 3], [2, 4, 7], [6, 6, 5], [5, 8, 3], [5, 5, 1], [3, 1, 4], [1, 3, 2], [5, 3, 7],
                    [1, 7, 4], [8, 4, 2], [2, 2, 9], [6, 4, 7], [10, 6, 5], [4, 8, 3], [5, 6, 1]])
    targets = [-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1]
    #targets = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    weights = np.zeros(points.shape[1])
    max_epochs = 1000
    sys.argv.pop(0)
    while sys.argv:
        curr_arg = sys.argv.pop(0)
        if curr_arg  == "-lr":
            LR = float(sys.argv.pop(0))
        elif curr_arg  == "-e":
            max_epochs = int(sys.argv.pop(0))
        elif curr_arg == "-g":
            graph = True
        elif curr_arg == "-h":
            usage(0)
        else:
            usage(1)
    return points, targets, weights, LR, max_epochs, graph

def get_data_madaline():
    graph = False
    LR = 0.1
    points = np.array([[-1.0,-1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    targets = np.array([-1, 1, 1, -1])
    weights1 = np.random.rand(points.shape[1])
    weights2 = np.random.rand(points.shape[1])
    # add x_0 = 1 to input vectors so that it multiplies with w_0 (an extra 1 added to weights vectors)to form the bias term
    points = np.insert(points, 0, 1, axis=1)
    weights1= np.insert(weights1, 0, 1)
    weights2= np.insert(weights2, 0, 1)
    max_epochs = 1000
    sys.argv.pop(0)
    while sys.argv:
        curr_arg = sys.argv.pop(0)
        if curr_arg  == "-lr":
            LR = float(sys.argv.pop(0))
        elif curr_arg  == "-e":
            max_epochs = int(sys.argv.pop(0))
        elif curr_arg == "-g":
            graph = True
        elif curr_arg == "-h":
            usage(0)
        else:
            usage(1)
    return points, targets, weights1, weights2, LR, max_epochs, graph

def min_max_normalize(data):
    num_rows = len(data)
    num_cols = len(data[0])

    # Transpose the data to handle columns more easily
    transposed_data = list(zip(*data))

    normalized_transposed_data = []

    for column in transposed_data:
        min_value = min(column)
        max_value = max(column)

        # Normalize the column
        if min_value == max_value:
            # Avoid division by zero when all values in the column are the same
            normalized_column = [0] * num_rows
        else:
            normalized_column = [(value - min_value) / (max_value - min_value) for value in column]

        normalized_transposed_data.append(normalized_column)

    # Transpose back to the original structure
    normalized_data = list(zip(*normalized_transposed_data))

    # Convert tuples back to lists
    normalized_data = [list(row) for row in normalized_data]

    return normalized_data

def get_data_perceptron():
    LR = 0.1
    points = np.array([[3, 1, 1], [7, 3, 3], [1, 4, 6], [5, 7, 4], [3, 1, 2], [8, 2, 3], [2, 4, 7], [6, 6, 5], [5, 8, 3], [5, 5, 1], [3, 1, 4], [1, 3, 2], [5, 3, 7],
                    [1, 7, 4], [8, 4, 2], [2, 2, 9], [6, 4, 7], [10, 6, 5], [4, 8, 3], [5, 6, 1]])
    # add x_0 to input vectors as bias
    points = np.insert(points, 0, 1, axis=1)
    targets = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    weights = np.zeros(points.shape[1])
    max_epochs = 1000
    sys.argv.pop(0)
    while sys.argv:
        curr_arg = sys.argv.pop(0)
        if curr_arg  == "-lr":
            LR = float(sys.argv.pop(0))
        elif curr_arg  == "-e":
            epochs = int(sys.argv.pop(0))
        elif curr_arg == "-h":
            usage(0)
        else:
            usage(1)
    return points, targets, weights, LR, max_epochs

def usage(exit_code):
    print("example usage: ")
    print("python perceptron2.py -e 500 -lr 0.2")
    print("-h to display help")
    print("-e to set the max number of training epochs")
    print("-lr to set a learning rate")
    print("-g to graph output")
    sys.exit(exit_code)

def print_output(epochs, weights, MSE, loss_list, graph, model_name):
    print(f'epochs: {epochs}')
    if MSE >= 0:
        print(f'MSE: {MSE}')
    for k, weight_set in enumerate(weights):
        if 'MADALINE' in model_name:
            print(f'Adaline Unit {k+1} has weights:')
            for i in range(len(weights[k])):
                print(f'w_{i} = {weights[k][i]}')
        else:
            print(f'w_{k} = {weights[k]}')
    if len(weights) == 4:
        print("calcplot3d decision boundary equation:")
        print(f'{weights[0]} + {weights[1]}x + {weights[2]}y + {weights[3]}z = 0')
    if graph is True:
        iterations = range(len(loss_list))
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, loss_list, marker='o', linestyle='-')
        plt.title(f'Loss (MSE) vs. parameter updates: {model_name}')
        plt.xlabel('Parameter Updates')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.show()

def print_output_perceptron(epochs, weights):
    print(f'epochs: {epochs}')
    for i in range(len(weights)):
        print(f'w_{i} = {weights[i]}')
    if len(weights) == 4:
        print("calcplot3d decision boundary equation:")
        print(f'{weights[0]} + {weights[1]}x + {weights[2]}y + {weights[3]}z = 0')

