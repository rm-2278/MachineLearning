# Multi-layered Perceptron
Implements multi-layered perceptron with numpy.
Uses mini-batch gradient descent for learning.

model.py:
Implements the multi-layered perceptron model.
It takes in in_dim, out_dim, function and deriv_function when initialized.
During training and validation, it takes in input and label, where each has size (batch_size, input_dim) and (batch_size, num_classes).

fashion_mnist.py:
Tests model under the fashion_mnist task, which classifies 28*28 images to 10 categories.
With 100 epochs,
valid_loss: 1.326 -> 0.360
valid_acc: 0.587 -> 0.875
evaluation_acc: 0.869

#TODO
1. Optimize learning rate using tensorboard or other tools.
2. Optimize model structure as well.
3. Use other datasets.
