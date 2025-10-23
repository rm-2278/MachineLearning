import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import LogisticRegression

np.random.seed(42)

# Loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale the pixel value: 0 - 255 -> 0.0 - 1.0
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten each image: 28 * 28 -> 784
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Convert integer label to one-hot vector
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Split train dataset to train and validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

# Create the model
model = LogisticRegression(input_dim=784, num_classes=10)

# Training loop
for epoch in range(100):
    train_loss = model.train(x_train, y_train)
    valid_loss, pred = model.valid(x_valid, y_valid)
    
    if epoch % 10 == 9 or epoch == 0:
        print(f'[Epoch {epoch+1}] train_loss: {train_loss:.3f}, valid_acc: {accuracy_score(y_valid.argmax(axis=1), pred.argmax(axis=1)):.3f}')

# Evaluation
test_loss, pred = model.valid(x_test, y_test)
print(f'evaluation_loss: {test_loss:.3f} Accuracy: {accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)):.3f}')