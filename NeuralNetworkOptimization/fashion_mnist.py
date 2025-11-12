import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import wandb  # For tracking progress

from model import MLP, train, evaluate

# Seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Hyperparameters
batch_size = 32
epochs = 10
in_dim = 28 * 28
hid_dim = 128
out_dim = 10
lr = 0.001

# Setting project
wandb.init(
    project="NNO_fashio_mnist",
    config = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "hidden_dim": hid_dim
    }
)


# Loading dataset (28 * 28, 10 closs one-hot)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Use y_train for validation split to ensure balanced split
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
dataset_train = TensorDataset(x_train_tensor, y_train_tensor)

x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
dataset_valid = TensorDataset(x_valid_tensor, y_valid_tensor)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
dataset_test = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoaders
dataloader_train = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True
)

dataloader_valid = DataLoader(
    dataset_valid,
    batch_size=batch_size,
    shuffle=False
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False
)

# Create the model
model = MLP(in_dim, hid_dim, out_dim)
optimizer = optim.Adam(model.parameters(), lr = lr)

# Training loop
print("Starting training...")
train(model, optimizer, dataloader_train, dataloader_valid, epochs)

print("Training completed.")
wandb.finish()

# Evaluation
print("Starting evaluation...")
test_loss, test_acc = evaluate(model, dataloader_test)

print(f'\nTest loss: {test_loss:.4f} Test accuracy: {test_acc:.4f}')