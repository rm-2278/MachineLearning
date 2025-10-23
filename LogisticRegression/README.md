# Logistic Regression
model.py:
Implements the logistric regression model.
It takes in input_dim, num_classes when initialized.
During training and validation, it takes in input and label, where each has size (batch_size, input_dim) and (batch_size, num_classes).

mnist.py:
Uses the model for the mnist task, numbering 0-9.
With 100 epochs,
train_loss: 2.346 -> 0.617
valid_acc: 0.157 -> 0.857
evaluation_acc: 0.866

fashion_mnist.py:
Uses the model for the fashion_mnist task, with 10 categories.
With 100 epochs,
train_loss: 2.395 -> 0.712
valid_acc: 0.211 -> 0.769
evaluation_acc: 0.761