import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import wandb  # For tracking progress

from model import MLP, train, valid

# Setting project
wandb.init(
    project=""
)

# Seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Hyperparameters
batch_size = 32
n_epochs = 10
in_dim = 28 * 28
hid_dim = 128
out_dim = 10
lr = 0.001

# Loading dataset (28 * 28, 10 closs one-hot)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)


# Create the model
model = MLP(in_dim, hid_dim, out_dim)
optimizer = optim.Adam(model.parameters(), lr = lr)

# Training loop
train(model, dataloader_train, dataloader_valid, n_epochs)

# Evaluation
test_loss, pred = valid(mlp, dataloader_test)
print(f'evaluation_loss: {test_loss:.3f} Accuracy: {accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)):.3f}')