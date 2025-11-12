import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)
        return x
        
def train(mlp, optimizer, dataloader_train, dataloader_valid, n_epochs):
    losses_train = []
    losses_valid = []
    train_num = 0
    train_correct = 0
    valid_num = 0
    valid_correct = 0
    
    mlp.train()
    for x, t in dataloader_train:
        t_hot = torch.eye(t.size()[1])[t]
        pred = mlp(x)
        loss = F.cross_entropy(pred, t_hot)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses_train.append(loss.tolist())  
        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
        train_num += acc.size()[0]
        train_correct += acc.sum().item()
        
    mlp.eval()
    for x, t in dataloader_valid:
        t_hot = torch.eye(t.size()[1])[t]
        pred = mlp(x)
        loss = F.cross_entropy(pred, t_hot)
        
        losses_valid.append(loss.tolist())  
        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
        valid_num += acc.size()[0]
        valid_correct += acc.sum().item()
    
    
    
    print(f"""EPOCH: \n
          Train Loss: \n
          Train Accuracy: \n
          Valid Loss: \n
          Valid Accuracy: \n""")
        
def valid():
    