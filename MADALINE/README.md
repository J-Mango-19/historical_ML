# Madaline

This folder contains a Python implementation of the MADALINE algorithm, an early machine learning model and successor of the ADALINE, as well as a supplementary file containing the ADALINE class. 

## Overview

MADALINE (Multiple ADAptive LINear Element) could be considered the first artificial neural network. Composed of multiple connected ADALINE units stored in two layers, MADALINE is capable of learning nonlinear decision boundaries and multiclass classification. What separates the modern MLP from MADALINE is its learning rule: MADALINE's learning rule (and hence its number of layers) was limited by the "credit assignment" problem. Without backpropagation, it became difficult to determine which ADALINE unit's weights to update in response to a misclassification. 
