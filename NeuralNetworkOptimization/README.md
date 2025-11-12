# Neural Network with optimization
Implements neural network with Pytorch. Uses regularizers and optimizers to improve the performance.

To use it:
1. python -m venv venv
2. source venv/bin/activate (Linux/Bash/mac)
   .\venv\Scripts\activate (Windows Command Prompt)
3. pip install -r requirements.txt



Improvements from previous models:
- Uses schedular (Adam)
- Uses wandb for logging
- Uses nn.CrossEntropyLoss, which is more suitable than F.cross_entropy as former is for within models while latter is for one-time. Also, this internally does equivalent calculation as one-hot encoding so removes the code for manually transforming.
- Uses GPU if cuda exists
- Uses main guard
- Reuses evaluate function for validataion & testing


fashion_mnist.py:
Uses the model for the fashion_mnist task, with 10 categories.
With 10 epochs,
train_loss: 
valid_loss:
test_loss:
train_acc:
valid_acc:  
test_acc: 

