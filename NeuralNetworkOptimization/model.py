import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Do not need softmax here as CrossEntropyLoss includes it
        return x
        
def train(mlp, optimizer, dataloader_train, dataloader_valid, epochs):
    for epoch in range(epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_correct = 0
        valid_num = 0
        valid_correct = 0
        
        mlp.train()
        for x, t in dataloader_train:
            pred = mlp(x)
            loss = F.cross_entropy(pred, t) # Use logits and class indices
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred_labels = torch.argmax(pred, dim=1)
            train_correct += (pred_labels == t).sum().item()
            train_num += t.size(0)
            losses_train.append(loss.item())
            
        mlp.eval()
        with torch.no_grad(): # Disable gradient calculation for validation
            for x, t in dataloader_valid:
                pred = mlp(x)
                loss = F.cross_entropy(pred, t)
                
                pred_labels = torch.argmax(pred, dim=1)
                valid_correct += (pred_labels == t).sum().item()
                valid_num += t.size(0)
                losses_valid.append(loss.item())
        
        avg_train_loss = np.mean(losses_train)
        avg_valid_loss = np.mean(losses_valid)
        train_accuracy = train_correct / train_num
        valid_accuracy = valid_correct / valid_num
            
        wandb.log({
            "epoch": epoch, 
            "train_loss": avg_train_loss, 
            "val_loss": avg_valid_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": valid_accuracy
            })
        
        print(f"""EPOCH: {epoch+1:02}/{epochs} | 
            Train Loss: {avg_train_loss:.4f} |
            Train Accuracy: {train_accuracy:.4f} |
            Valid Loss: {avg_valid_loss:.4f} | 
            Valid Accuracy: {valid_accuracy:.4f}""")
        
def evaluate(model, dataloader_test):
    model.eval()
    test_losses = []
    test_num = 0
    test_correct = 0
    
    with torch.no_grad():
        for x, t in dataloader_test:
            pred = model(x)
            loss = F.cross_entropy(pred, t)
            
            pred_labels = torch.argmax(pred, dim=1)
            test_correct += (pred_labels == t).sum().item()
            test_num += t.size(0)
            test_losses.append(loss.item())
    
    avg_test_loss = np.mean(test_losses)
    test_accuracy = test_correct / test_num
    
    return avg_test_loss, test_accuracy
    