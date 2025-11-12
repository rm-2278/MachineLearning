import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Do not need softmax here as CrossEntropyLoss includes it
        return x
        
def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          dataloader_train: DataLoader,
          dataloader_valid: DataLoader,
          epochs: int,
          criterion: nn.Module,
          device: torch.device):
    
    # Send model to device
    model.to(device)
    
    for epoch in range(epochs):
        losses_train = []
        train_num = 0
        train_correct = 0
        
        model.train()
        for x, t in dataloader_train:
            # Send data to device
            x = x.to(device)
            t = t.to(device)
            
            pred = model(x)
            loss = criterion(pred, t) # Use logits and class indices
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred_labels = torch.argmax(pred, dim=1)
            train_correct += (pred_labels == t).sum().item()
            train_num += t.size(0)
            losses_train.append(loss.item())
        
        avg_train_loss = np.mean(losses_train)
        train_accuracy = train_correct / train_num
        
        # Reuse evaluate function for validation
        avg_valid_loss, valid_accuracy = evaluate(
            model, dataloader_valid, device
        )
            
        wandb.log({
            "epoch": epoch, 
            "train_loss": avg_train_loss, 
            "val_loss": avg_valid_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": valid_accuracy
            })
        
        print(f"""EPOCH: {epoch+1:02}/{epochs} | 
            Train Loss: {avg_train_loss:.4f} |
            Valid Loss: {avg_valid_loss:.4f} | 
            Train Accuracy: {train_accuracy:.4f} |
            Valid Accuracy: {valid_accuracy:.4f}""")
        
def evaluate(model: nn.Module,
             dataloader_test: DataLoader,
             criterion: nn.Module,
             device: torch.device):
    
    model.eval()
    losses = []
    num_total = 0
    num_correct = 0
    
    with torch.no_grad():
        for x, t in dataloader_test:
            x.to(device)
            t.to(device)
            
            pred = model(x)
            loss = criterion(pred, t)
            
            pred_labels = torch.argmax(pred, dim=1)
            num_correct += (pred_labels == t).sum().item()
            num_total += t.size(0)
            losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    accuracy = num_correct / num_total
    
    return avg_loss, accuracy
    