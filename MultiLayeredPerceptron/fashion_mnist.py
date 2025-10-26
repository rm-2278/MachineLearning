import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from model import Model
from helper import relu, deriv_relu, softmax, deriv_softmax, create_batch, train, valid

# Load dataset (train: 60000 * 28*28, test: 10000 * 28*28)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessing (Normalization, one-hot encoding)
x_train, x_test = x_train / 255., x_test / 255.
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Split training data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

# Set random seed
random_state = 42
rng = np.random.RandomState(random_state)

# Hyperparameters
lr = 0.01
epochs = 100
batch_size = 128

# Initialize model
model = Model(hidden_dims=[784, 100, 100, 10], activation_functions=[relu, relu, softmax], deriv_functions=[deriv_relu, deriv_relu, deriv_softmax])

# Actual training loop
for epoch in range(epochs):
    # Shuffle randomly to recreate batches
    x_train, y_train = shuffle(x_train, y_train)
    x_train_batch, y_train_batch = create_batch(x_train, batch_size), create_batch(y_train, batch_size)
    
    for x, t in zip(x_train_batch, y_train_batch):
        loss = train(model, x, t)
    
    loss, y_pred = valid(model, x_val, y_val)
    accuracy = accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    if epoch == 0 or epoch % 10 == 9:
        print(f"EPOCH: {epoch+1} Valid Cost: {loss:.3f} Valid Accuracy: {accuracy:.3f}")
        
# Evaluation
test_loss, pred = valid(model, x_test, y_test)
print(f'evaluation_loss: {test_loss:.3f} Accuracy: {accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)):.3f}')