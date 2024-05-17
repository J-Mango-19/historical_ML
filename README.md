# historical_ML
Python implementations of the Perceptron, ADALINE, and MADALINE

The programs in this repository implement the first linear classifiers, including the Perceptron (1957), ADALINE (1960), and MADALINE (1961). 
Each program comes with a toy dataset which can be modified in the process_io.py file. 
Set custom hyperparameters on the command line using
-h for help
-lr to adjust learning rate
-e to adjust max epochs
-g to graph the loss against the number of training iterations

example usage: python Adaline_GD_normalized.py -e 500 -lr 0.05

