import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import wandb  # For tracking progress

from model import MLP, train, evaluate

def main():
    # Seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hyperparameters
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "hidden_dim": 128,
        "in_dim": 28 * 28,
        "out_dim": 10
    }

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setting project
    wandb.init(
        project="NNO_fashio_mnist",
        config = config
    )


    # Loading dataset (28 * 28, 10 class one-hot)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(-1, config['in_dim'])
    x_test = x_test.reshape(-1, config['in_dim'])

    # Use y_train for validation split to ensure balanced split
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
    )

    # Convert to PyTorch tensors
    # input tensor must be float, target tensor must be long
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset_train = TensorDataset(x_train_tensor, y_train_tensor)

    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
    dataset_valid = TensorDataset(x_valid_tensor, y_valid_tensor)

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    dataset_test = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        shuffle=True
    )

    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=config['batch_size'],
        shuffle=False
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Create the model
    model = MLP(config["in_dim"], config["hidden_dim"], config["out_dim"])
    # No need to reassign as nn.Module modifies in place
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Logging gradients and parameters
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Training loop
    print("Starting training...")
    train(model, 
          optimizer, 
          dataloader_train, 
          dataloader_valid, 
          config['epochs'], 
          criterion,
          device)

    print("Training completed.")

    # Evaluation
    print("Starting evaluation...")
    test_loss, test_acc = evaluate(
        model, 
        dataloader_test,
        criterion,
        device
    )

    print(f'\nTest loss: {test_loss:.4f} Test accuracy: {test_acc:.4f}')
    
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })
    wandb.finish()

# Code runs only if run as a script
if __name__ == "__main__":
    main()
